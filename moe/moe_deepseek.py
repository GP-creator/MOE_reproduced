"""DeepSeekMoE layer: shared-expert isolation + fine-grained routed experts, no dropping.

Paper: Dai et al., "DeepSeekMoE" (arXiv 2401.06066), Sec. 3.

Fine-grained expert segmentation (Sec. 3.1): each standard expert of width d_ff is split into
m smaller experts of width d_ff/m, and m times more are activated. Shared expert isolation
(Sec. 3.2): K_s experts are always on. With m = 4, 1 shared + 31 routed of width d_ff/4 and
K' = 3 routed per token: total = 32 * d_ff/4 = 8 d_ff, activated = (1 + 3) * d_ff/4 = d_ff.

For token t with input u_t (flattened to N = B*T rows):
    s_{i,t} = Softmax_i(u_t^T e_i)          over the N' routed experts       (Eq. 11)
    g_{i,t} = s_{i,t} if s_{i,t} in TopK({s_{j,t}}, K') else 0                 (Eq. 10)
    h_t = sum_{i=1..K_s} FFN_i(u_t)                     shared, no gate     (Eq. 9, 1st sum)
        + sum_{i=K_s+1..mN} g_{i,t} FFN_i(u_t)          routed              (Eq. 9, 2nd sum)
The residual "+ u_t" of Eq. 9 is added by model.py (layer_api contract item 2).

There is no capacity, so no token is ever dropped. Balance loss: expert-level Eqs. 12-14
(alpha1) plus optional device-level Eqs. 15-17 (alpha2, default 0).

Eval-time ablations (DeepSeekMoE Sec. 4.5, Figs. 4-5): ``disable_shared``,
``n_disable_top_routed`` (mask the top-n routed experts per token, pick top-K' of the rest),
``k_routed_override``.

Dispatch paths:
  * ``loop``   : Python loop over routed experts.
  * ``batched``: flatten the N*K' assignments, sort by expert (stable), pack into a padded
                 buffer [N', max_count, d] (ONE host sync to read max_count), two bmm, weighted
                 index_add back. Padding waste is reported by flops.moe_layer_cost.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from moe.experts import ExpertBank
from moe.layer_api import MoEAux, ffn_spec
from moe.moe_switch import compute_dtype
from moe.profiling import phase
from moe.router import Router, deepseek_device_aux_loss, deepseek_expert_aux_loss


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE layer. Config: ``moe.deepseek`` (m, n_shared, n_routed, k_routed, ...)."""

    ABLATION_ATTRS: tuple[str, ...] = ("n_disable_top_routed", "disable_shared", "k_routed_override")

    def __init__(self, d_model: int, d_ff: int, moe_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        sub = moe_cfg["deepseek"]
        # ffn_spec asserts (n_shared + k_routed) * d_ff/m == d_ff (MASTER §4.5).
        self.spec = ffn_spec("deepseek", {"model": {"d_ff": d_ff}, "moe": moe_cfg})
        self.d_model = d_model
        self.n_shared, self.n_routed = self.spec.n_shared, self.spec.n_routed
        self.top_k, self.width = self.spec.top_k, self.spec.width
        self.aux_alpha = float(sub["aux_alpha"])
        self.device_aux_alpha = float(sub.get("device_aux_alpha", 0.0))
        self.n_devices = int(sub.get("n_devices", 1))
        self.dispatch = str(moe_cfg.get("dispatch", "batched"))
        # Eval-time ablation switches (layer_api contract item 4).
        self.disable_shared = False
        self.n_disable_top_routed = 0
        self.k_routed_override: Optional[int] = None

        self.router = Router(d_model, self.n_routed, float(sub.get("jitter_eps", 0.0)),
                             bool(moe_cfg.get("renormalize_gates", False)), model_cfg)
        self.shared = ExpertBank(self.n_shared, d_model, self.width, model_cfg) if self.n_shared else None
        self.experts = ExpertBank(self.n_routed, d_model, self.width, model_cfg)   # routed, [N', d, w]

    @property
    def k_active(self) -> int:
        """Routed experts per token right now (K', or the eval override)."""
        return int(self.k_routed_override) if self.k_routed_override is not None else self.top_k

    def forward(self, x: torch.Tensor, return_routing: bool = False) -> tuple[torch.Tensor, MoEAux]:
        """x [B, T, d] -> (y [B, T, d], MoEAux)."""
        B, T, d = x.shape
        N, E, k = B * T, self.n_routed, self.k_active
        x_flat = x.reshape(N, d)                                              # [N, d]

        with phase("router"):                                                 # E5 annotation (no-op otherwise)
            r = self.router(x_flat, k, self.n_disable_top_routed)             # Eqs. 10-11, fp32
        if self.dispatch == "loop":
            y_routed, assign_norm = self._routed_loop(x_flat, r.topk_idx, r.gates, return_routing)
        elif self.dispatch == "batched":
            y_routed, assign_norm = self._routed_batched(x_flat, r.topk_idx, r.gates, return_routing)
        else:
            raise ValueError(f"unknown dispatch {self.dispatch!r}")

        y_shared = None
        if self.shared is not None and not self.disable_shared:
            # Eq. 9, first sum: shared experts on every token, output added with no gate.
            with phase("shared_expert"):
                for s in range(self.n_shared):
                    out_s = self.shared.expert_forward(s, x_flat).to(y_routed.dtype)   # [N, d]
                    y_shared = out_s if y_shared is None else y_shared + out_s
        with phase("combine"):
            y = y_routed if y_shared is None else y_routed + y_shared           # [N, d]

        with phase("router"):
            aux_expert = deepseek_expert_aux_loss(r.probs, r.topk_idx, self.aux_alpha)    # Eq. 12
            aux_loss = aux_expert
            extra = {"aux_expert": aux_expert.detach()}
            if self.device_aux_alpha > 0:
                aux_device = deepseek_device_aux_loss(r.probs, r.topk_idx, self.n_devices,
                                                      self.device_aux_alpha)              # Eq. 15
                aux_loss = aux_loss + aux_device
                extra["aux_device"] = aux_device.detach()
        counts = torch.bincount(r.topk_idx.reshape(-1), minlength=E).detach()             # [N']
        aux = MoEAux(
            aux_loss=aux_loss,
            n_experts=E,
            expert_counts=counts,
            kept_counts=counts,                      # no capacity, nothing dropped
            drop_fraction=torch.zeros((), device=x.device),
            router_entropy=r.entropy,
            top_k=k,
            n_tokens=N,
            extra=extra,
        )
        if return_routing:
            aux.topk_idx = r.topk_idx.detach()                                 # [N, k]
            aux.gates = r.gates.detach()                                       # [N, k]
            aux.kept = torch.ones_like(r.topk_idx, dtype=torch.bool)           # [N, k]
            aux.extra["router_probs"] = r.probs.detach()                       # [N, N'] fp32 (R1)
            aux.extra["routed_expert_out_norm"] = assign_norm                  # [N, k]  (R3)
            aux.extra["routed_out_norm"] = y_routed.detach().float().norm(dim=-1)            # [N] (R2)
            aux.extra["shared_out_norm"] = (y_shared.detach().float().norm(dim=-1) if y_shared is not None
                                            else torch.zeros(N, device=x.device))           # [N] (R2)
        return y.view(B, T, d), aux

    def _routed_loop(self, x_flat: torch.Tensor, topk_idx: torch.Tensor, gates: torch.Tensor,
                     want_norms: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Eq. 9 second sum, one routed expert at a time. Returns (y [N, d], norms [N, k] of
        each gated contribution if ``want_norms`` else None)."""
        N, d = x_flat.shape
        cdt = compute_dtype(x_flat)
        y = x_flat.new_zeros(N, d, dtype=cdt)                                 # [N, d]
        norms = x_flat.new_zeros(topk_idx.shape, dtype=torch.float32) if want_norms else None  # [N, k]
        for e in range(self.n_routed):
            with phase("dispatch"):
                tok, slot = (topk_idx == e).nonzero(as_tuple=True)            # [n_e], [n_e]
                if tok.numel() == 0:
                    continue
                x_e = x_flat[tok]                                             # [n_e, d] gather
            with phase("expert_gemm"):
                out = self.experts.expert_forward(e, x_e).to(cdt)             # [n_e, d] FFN_e(u_t)
            with phase("combine"):
                g = gates[tok, slot].to(cdt)[:, None]                         # [n_e, 1] g_{e,t}
                y = y.index_add(0, tok, g * out)                              # += g_{e,t} FFN_e(u_t)
            if want_norms:
                norms[tok, slot] = (g * out).detach().float().norm(dim=-1)
        return y, norms

    def _routed_batched(self, x_flat: torch.Tensor, topk_idx: torch.Tensor, gates: torch.Tensor,
                        want_norms: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sort assignments by expert, pack into [N', max_count, d], bmm, un-sort, sum the k slots.

        Bit-determinism (MASTER §5): no step accumulates two values into the same row through
        a parallel scatter. Every scatter/gather below uses UNIQUE indices (a permutation or
        unique buffer slots), and the k contributions of a token are summed with explicit adds
        in slot order, exactly like the loop path. In particular the input copy uses
        ``expand`` (whose backward is a plain sum over k) instead of ``x_flat[token]`` (whose
        backward would index_add k rows into one, in a thread-dependent order on CPU).
        """
        N, d = x_flat.shape
        E, k = self.n_routed, topk_idx.shape[1]
        cdt = compute_dtype(x_flat)
        with phase("dispatch"):
            expert = topk_idx.reshape(-1)                                     # [N*k] token-major: a = t*k + j
            gate = gates.reshape(-1)                                          # [N*k]
            x_rep = x_flat.to(cdt).unsqueeze(1).expand(N, k, d).reshape(N * k, d)  # [N*k, d] row a = token a // k
            order = torch.argsort(expert, stable=True)                        # [N*k] permutation, grouped by expert
            e_sorted = expert[order]                                          # [N*k]
            counts = torch.bincount(expert, minlength=E)                      # [E]
            starts = torch.cumsum(counts, 0) - counts                         # [E] segment starts
            pos = torch.arange(N * k, device=x_flat.device) - starts[e_sorted]  # [N*k] position in segment
            max_count = int(counts.max().item())                              # the one host sync
            dest = e_sorted * max_count + pos                                 # [N*k] unique buffer rows
            buf = x_rep.new_zeros(E * max_count, d).index_copy(0, dest, x_rep[order])      # [E*M, d]
        with phase("expert_gemm"):
            out = self.experts.batched_forward(buf.view(E, max_count, d)).to(cdt)         # [E, M, d]
        with phase("combine"):
            y_sorted = out.reshape(E * max_count, d)[dest] * gate[order].to(cdt)[:, None]  # [N*k, d] g * FFN
            # Undo the sort: assignment order[i] gets y_sorted[i] (unique targets, no accumulation).
            y_assign = y_sorted.new_zeros(N * k, d).index_copy(0, order, y_sorted).view(N, k, d)  # [N, k, d]
            y = y_assign[:, 0]                                                # [N, d]
            for j in range(1, k):                                             # fixed-order sum over slots
                y = y + y_assign[:, j]
        norms = y_assign.detach().float().norm(dim=-1) if want_norms else None  # [N, k]
        return y, norms
