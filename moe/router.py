"""Router: gating, jitter, fp32 precision, top-k, and the load-balancing losses.

Shared by Switch (top-1), GShard (top-2) and DeepSeekMoE (top-K' among routed experts).

Selective precision (Switch Sec. 2.4, "Selective precision with large sparse models"):
everything inside :meth:`Router.forward` runs in float32 with autocast DISABLED: the input
cast, the jitter, the router matmul, the softmax, top-k and the aux losses. The router weight
is stored in fp32. Only the gate values are cast to the activation dtype later, right before
the combine multiply in the MoE layer (PLAN §7). Indices are int64.

Notation: N = number of tokens in the batch (B*T), E = number of routed experts, k = experts
chosen per token, d = d_model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from moe.layer_api import init_weight_, mark_no_weight_decay


@dataclass
class RouterOutput:
    """Everything the MoE layers need from the router (all fp32 / int64, shape-commented).

    probs:     fp32 [N, E]  softmax over ALL routed experts (Switch Eq. 1 p_i(x); DeepSeekMoE
                            Eq. 11 s_{i,t}). Never masked, so aux losses see the full distribution.
    topk_idx:  int64 [N, k] chosen experts, column 0 = highest probability (1st choice).
    topk_prob: fp32 [N, k]  probs at topk_idx (raw softmax values).
    gates:     fp32 [N, k]  gate used for the combine: = topk_prob, or topk_prob renormalised
                            over the k choices if ``renormalize`` is on.
    entropy:   fp32 []      mean over tokens of -sum_i p_i log p_i (nats), detached.
    """

    probs: torch.Tensor
    topk_idx: torch.Tensor
    topk_prob: torch.Tensor
    gates: torch.Tensor
    entropy: torch.Tensor


class Router(nn.Module):
    """Linear router ``logits = x @ W_r`` (no bias), W_r [d, E] stored in fp32.

    Args:
        d_model, n_experts: shapes.
        jitter_eps: Switch App. C "jitter noise": in training, the router INPUT is multiplied
            by U(1 - eps, 1 + eps) elementwise. 0 disables it.
        renormalize: if True, gates are renormalised to sum to 1 over the k chosen experts.
            Both papers use the raw softmax probability (default False).
        model_cfg: for :func:`init_weight_` (fan_in = d).
    """

    def __init__(self, d_model: int, n_experts: int, jitter_eps: float, renormalize: bool,
                 model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.d_model, self.n_experts = d_model, n_experts
        self.jitter_eps = float(jitter_eps)
        self.renormalize = bool(renormalize)
        self.weight = nn.Parameter(torch.empty(d_model, n_experts, dtype=torch.float32))  # W_r [d, E]
        init_weight_(self.weight, fan_in=d_model, model_cfg=model_cfg)
        mark_no_weight_decay(self.weight)  # PLAN §4: router excluded from weight decay

    def forward(self, x: torch.Tensor, k: int, n_disable_top: int = 0) -> RouterOutput:
        """x [N, d] (any dtype) -> RouterOutput.

        ``n_disable_top`` (E3, DeepSeekMoE Sec. 4.5 / Fig. 4): per token, the n highest-prob
        experts are masked out of the SELECTION; top-k is taken among the rest. The gate is the
        original softmax prob (no renormalisation over the survivors).

        The paper's wording is only "for each token, we mask a certain ratio of experts with
        the highest routing probability, and then select top-K experts from the remaining
        routed experts"; it says nothing about the gates. We keep Eq. 10 literally (g = s, the
        unmasked softmax). Checked on the smoke DeepSeek checkpoint (review gate, 2026-10-04):
        renormalising the survivors' gates by 1 / (1 - masked mass) gives the same E3 curve
        shape, so the choice does not drive the result there.
        """
        N, E = x.shape[0], self.n_experts
        if n_disable_top < 0 or n_disable_top + k > E:
            raise ValueError(f"need 0 <= n_disable_top ({n_disable_top}) and n_disable_top + k ({k}) <= E ({E})")
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()                                                   # [N, d] fp32
            if self.training and self.jitter_eps > 0:
                # Switch App. C: multiplicative jitter on the router input, training only.
                noise = torch.empty_like(xf).uniform_(1.0 - self.jitter_eps, 1.0 + self.jitter_eps)
                xf = xf * noise                                              # [N, d]
            logits = xf @ self.weight.float()                                # [N, E] h(x) = W_r x
            probs = torch.softmax(logits, dim=-1)                            # [N, E] Switch Eq. 1

            if n_disable_top > 0:
                # Mask the n_dis most probable experts for selection only.
                top_dis = probs.topk(n_disable_top, dim=-1).indices          # [N, n_dis]
                sel_scores = probs.scatter(-1, top_dis, float("-inf"))       # [N, E]
            else:
                sel_scores = probs
            topk_idx = sel_scores.topk(k, dim=-1).indices                    # [N, k], sorted desc
            topk_prob = probs.gather(-1, topk_idx)                           # [N, k] raw probs
            gates = topk_prob / topk_prob.sum(-1, keepdim=True) if self.renormalize else topk_prob

            entropy = torch.special.entr(probs).sum(-1).mean().detach()      # []
        return RouterOutput(probs, topk_idx, topk_prob, gates, entropy)


# ----------------------------------------------------------------------------------------
# Load-balancing losses (all fp32; called inside the fp32 region by construction because
# probs are fp32 and no autocast-eligible ops are used)
# ----------------------------------------------------------------------------------------

def switch_aux_loss(probs: torch.Tensor, top1_idx: torch.Tensor, alpha: float) -> torch.Tensor:
    """Switch Transformer load-balancing loss (Switch Eqs. 4-6).

        f_i = (1/T) * sum_x 1{argmax p(x) = i}        (Eq. 5, fraction of tokens dispatched)
        P_i = (1/T) * sum_x p_i(x)                    (Eq. 6, fraction of router prob mass)
        loss = alpha * N * sum_i f_i * P_i            (Eq. 4, N = number of experts)

    f_i uses the argmax BEFORE capacity dropping (PLAN §7) and is not differentiable; the
    gradient flows through P_i. For GShard we pass the 1st choice (top-1 f_i, as in GShard).
    Minimum value alpha (uniform routing).

    probs [T, E] fp32, top1_idx [T] int64 -> [] fp32.
    """
    T, E = probs.shape
    f = torch.bincount(top1_idx, minlength=E).float() / T   # [E]  Eq. 5
    P = probs.mean(dim=0)                                    # [E]  Eq. 6
    return alpha * E * torch.sum(f * P)                      # []   Eq. 4


def deepseek_expert_aux_loss(probs: torch.Tensor, topk_idx: torch.Tensor, alpha1: float) -> torch.Tensor:
    """DeepSeekMoE expert-level balance loss (Eqs. 12-14).

        f_i = N' / (K' T) * sum_t 1(token t selects expert i)    (Eq. 13)
        P_i = (1/T) * sum_t s_{i,t}                              (Eq. 14)
        L_ExpBal = alpha1 * sum_i f_i * P_i                      (Eq. 12)

    N' = number of routed experts, K' = routed experts chosen per token. Minimum alpha1.
    probs [T, N'] fp32 (softmax over routed experts), topk_idx [T, K'] -> [] fp32.
    """
    T, n_routed = probs.shape
    k = topk_idx.shape[1]
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_routed).float()   # [N']
    f = counts * n_routed / (k * T)                                             # [N'] Eq. 13
    P = probs.mean(dim=0)                                                       # [N'] Eq. 14
    return alpha1 * torch.sum(f * P)                                            # []   Eq. 12


def device_groups(n_routed: int, n_devices: int) -> list[torch.Tensor]:
    """Partition routed experts into D contiguous groups {E_1..E_D} (DeepSeekMoE Sec. 3.3).

    If n_routed is not divisible by D (31 routed experts), group sizes differ by at most 1
    (``torch.arange(n).tensor_split(D)``).
    """
    return list(torch.arange(n_routed).tensor_split(n_devices))


def deepseek_device_aux_loss(probs: torch.Tensor, topk_idx: torch.Tensor, n_devices: int,
                             alpha2: float) -> torch.Tensor:
    """DeepSeekMoE device-level balance loss (Eqs. 15-17). Off by default (alpha2 = 0).

        f'_i = (1/|E_i|) * sum_{j in E_i} f_j        (Eq. 16, f_j from Eq. 13)
        P'_i = sum_{j in E_i} P_j                    (Eq. 17, P_j from Eq. 14)
        L_DevBal = alpha2 * sum_{i=1..D} f'_i P'_i   (Eq. 15)
    """
    T, n_routed = probs.shape
    k = topk_idx.shape[1]
    counts = torch.bincount(topk_idx.reshape(-1), minlength=n_routed).float()
    f = counts * n_routed / (k * T)                     # [N'] Eq. 13
    P = probs.mean(dim=0)                               # [N'] Eq. 14
    loss = probs.new_zeros(())
    for group in device_groups(n_routed, n_devices):
        g = group.to(probs.device)
        f_dev = f[g].mean()                             # Eq. 16
        P_dev = P[g].sum()                              # Eq. 17
        loss = loss + f_dev * P_dev
    return alpha2 * loss                                # Eq. 15
