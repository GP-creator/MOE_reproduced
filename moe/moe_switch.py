"""Switch Transformer MoE layer: top-1 routing, expert capacity, token dropping.

Paper: Fedus, Zoph, Shazeer, "Switch Transformers" (arXiv 2101.03961).

One layer, for tokens x (flattened to N = B*T rows):
  1. Router (moe/router.py, fp32): p(x) = softmax(W_r x)                       (Switch Eq. 1)
  2. Each token picks expert i = argmax_i p_i(x) (top-1, "switch" routing). Gate = p_i(x).
  3. Expert capacity C = floor(N / E * capacity_factor)                          (Switch Eq. 3)
     Tokens claim buffer slots in token order (the cumsum of Switch Fig. 15). A token whose
     position in its expert's buffer is >= C is DROPPED: its MoE output is exactly 0, so it
     only passes through the residual connection (Switch Sec. 2.2).
  4. y(x) = p_i(x) * E_i(x)                                                       (Switch Eq. 2)
  5. Aux loss = alpha * E * sum_i f_i P_i                                         (Switch Eqs. 4-6)

The capacity machinery here is written for general top-k because GShard (moe_gshard.py)
reuses it with k = 2. With k = 1 every formula reduces to the Switch paper's.

Two dispatch paths compute identical math from the same routing decisions and the same
stacked expert weights (moe/experts.py):
  * ``loop``    : a readable Python loop over experts (gather that expert's kept tokens,
                  run the FFN, scatter-add gate * output back).
  * ``batched`` : sort-and-segment into a capacity-padded buffer [E, C, d], two bmm, gather
                  back (PLAN §7). Equivalent to the dispatch/combine einsums of Switch Fig. 16
                  without materialising the [N, E, C] one-hot tensor; see
                  :func:`einsum_reference` for that literal formulation (used by the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from moe.experts import ExpertBank
from moe.layer_api import MoEAux, expert_capacity, ffn_spec
from moe.profiling import phase
from moe.router import Router, RouterOutput, switch_aux_loss


def compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype the expert matmuls will produce: the autocast dtype if autocast is on, else x's."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return x.dtype


@dataclass
class CapacityAssignment:
    """Result of assigning (token, choice) pairs to expert buffer slots under capacity C.

    Assignments are flattened SLOT-MAJOR: assignment a = j * N + t is token t's j-th choice.
    So all 1st choices (in token order) come before all 2nd choices. Taking buffer slots in
    this order gives GShard's priority "first choices fill capacity before second choices";
    with k = 1 it is plain token order, as in Switch Fig. 15.

    expert: int64 [k*N]  expert of assignment a
    pos:    int64 [k*N]  0-based position of assignment a in its expert's buffer
    keep:   bool  [k*N]  pos < C (False = dropped)
    token:  int64 [k*N]  token index t = a % N
    onehot: int64 [k*N, E] one-hot of ``expert`` (reused for counts)
    capacity: int        C
    """

    expert: torch.Tensor
    pos: torch.Tensor
    keep: torch.Tensor
    token: torch.Tensor
    onehot: torch.Tensor
    capacity: int


def assign_with_capacity(topk_idx: torch.Tensor, n_experts: int, capacity: int) -> CapacityAssignment:
    """topk_idx [N, k] -> CapacityAssignment (see that class for the priority order)."""
    N, k = topk_idx.shape
    expert = topk_idx.t().reshape(-1)                         # [k*N] slot-major
    onehot = F.one_hot(expert, n_experts)                     # [k*N, E]
    # Switch Fig. 15: position_in_expert = cumsum(one_hot) over the token axis, minus 1,
    # read off at each assignment's own expert.
    pos = ((onehot.cumsum(dim=0) - 1) * onehot).sum(dim=-1)   # [k*N]
    keep = pos < capacity                                     # [k*N]
    token = torch.arange(k * N, device=topk_idx.device) % N   # [k*N]
    return CapacityAssignment(expert, pos, keep, token, onehot, capacity)


def einsum_reference(x_flat: torch.Tensor, asg: CapacityAssignment, gate_flat: torch.Tensor,
                     experts: ExpertBank, k: int) -> torch.Tensor:
    """Literal Switch Fig. 16 formulation with [N, E, C] dispatch/combine tensors.

    Teaching/test reference only: memory is O(N * E * C), which is far too big at E5 sizes.

        dispatch[t, e, c] = 1 if token t sits in slot c of expert e        [N, E, C]
        combine[t, e, c]  = gate of that assignment (0 if dropped)          [N, E, C]
        expert_inputs     = einsum("tec,td->ecd", dispatch, x)              [E, C, d]
        expert_outputs    = FFN_e(expert_inputs[e])                         [E, C, d]
        y                 = einsum("tec,ecd->td", combine, expert_outputs)  [N, d]

    For top-k the k assignments of a token are summed into the same [N, E, C] tensors (a token
    never picks the same expert twice, so they never collide).
    """
    N, d = x_flat.shape
    E, C = experts.n_experts, asg.capacity
    if C == 0:
        return x_flat.new_zeros(N, d)
    keep_f = asg.keep.to(x_flat.dtype)                                        # [kN]
    oh_e = F.one_hot(asg.expert, E).to(x_flat.dtype)                          # [kN, E]
    oh_c = F.one_hot(asg.pos.clamp(max=C - 1), C).to(x_flat.dtype)            # [kN, C]
    per_assign = oh_e[:, :, None] * oh_c[:, None, :] * keep_f[:, None, None]  # [kN, E, C]
    dispatch = per_assign.view(k, N, E, C).sum(0)                             # [N, E, C]
    combine = (per_assign * gate_flat.to(x_flat.dtype)[:, None, None]).view(k, N, E, C).sum(0)  # [N, E, C]
    expert_inputs = torch.einsum("tec,td->ecd", dispatch, x_flat)             # [E, C, d]
    expert_outputs = experts.batched_forward(expert_inputs)                   # [E, C, d]
    return torch.einsum("tec,ecd->td", combine, expert_outputs)               # [N, d]


class CapacityMoE(nn.Module):
    """Top-k routed experts with capacity and dropping (base of SwitchMoE and GShardMoE).

    Contract: moe/layer_api.py. Subclasses set ``variant`` (the moe-config sub-key).
    Runtime attributes: ``dispatch`` ("loop" | "batched", from ``moe.dispatch``; tests flip it),
    and the ablation switches in ``ABLATION_ATTRS``.
    """

    variant: str = ""
    ABLATION_ATTRS: tuple[str, ...] = ("eval_capacity_factor", "n_disable_top_routed")

    def __init__(self, d_model: int, d_ff: int, moe_cfg: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        sub = moe_cfg[self.variant]
        self.spec = ffn_spec(self.variant, {"model": {"d_ff": d_ff}, "moe": moe_cfg})
        self.d_model = d_model
        self.n_routed, self.top_k, self.width = self.spec.n_routed, self.spec.top_k, self.spec.width
        self.capacity_factor = float(sub["capacity_factor"])
        ecf = sub.get("eval_capacity_factor")
        self.eval_capacity_factor: Optional[float] = None if ecf is None else float(ecf)
        self.aux_alpha = float(sub["aux_alpha"])
        self.dispatch = str(moe_cfg.get("dispatch", "batched"))
        self.n_disable_top_routed = 0
        self.router = Router(d_model, self.n_routed, float(sub.get("jitter_eps", 0.0)),
                             bool(moe_cfg.get("renormalize_gates", False)), model_cfg)
        self.experts = ExpertBank(self.n_routed, d_model, self.width, model_cfg)   # W_in [E,d,w], W_out [E,w,d]

    # -- capacity -------------------------------------------------------------------------
    def active_capacity_factor(self) -> float:
        """Training cf in train mode; eval_capacity_factor (if set) in eval mode."""
        if not self.training and self.eval_capacity_factor is not None:
            return self.eval_capacity_factor
        return self.capacity_factor

    def capacity(self, n_tokens: int) -> int:
        """C = min(N, floor(k * N / E * cf)) (Switch Eq. 3 for k = 1; layer_api.expert_capacity)."""
        return expert_capacity(n_tokens, self.n_routed, self.active_capacity_factor(), self.top_k)

    # -- aux loss -------------------------------------------------------------------------
    def aux_loss(self, r: RouterOutput) -> torch.Tensor:
        """Switch Eq. 4 with f_i from the 1st choice before dropping (top-1 f_i for GShard too)."""
        return switch_aux_loss(r.probs, r.topk_idx[:, 0], self.aux_alpha)

    # -- forward --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_routing: bool = False) -> tuple[torch.Tensor, MoEAux]:
        """x [B, T, d] -> (y [B, T, d], MoEAux)."""
        B, T, d = x.shape
        N, E, k = B * T, self.n_routed, self.top_k
        x_flat = x.reshape(N, d)                                         # [N, d]

        # phase(...) only annotates E5 profiling ranges (moe/profiling.py); no-op otherwise.
        with phase("router"):
            r = self.router(x_flat, k, self.n_disable_top_routed)        # fp32 routing
            aux_loss = self.aux_loss(r)
        with phase("dispatch"):
            asg = assign_with_capacity(r.topk_idx, E, self.capacity(N))
            gate_flat = r.gates.t().reshape(-1)                          # [kN] fp32, slot-major

        if self.dispatch == "loop":
            y_flat, assign_norm = self._dispatch_loop(x_flat, asg, gate_flat, return_routing)
        elif self.dispatch == "batched":
            y_flat, assign_norm = self._dispatch_batched(x_flat, asg, gate_flat, return_routing)
        else:
            raise ValueError(f"unknown dispatch {self.dispatch!r}")

        keep_f = asg.keep.float()                                        # [kN]
        aux = MoEAux(
            aux_loss=aux_loss,
            n_experts=E,
            expert_counts=asg.onehot.sum(0).detach(),                    # [E] pre-drop
            kept_counts=(asg.onehot * asg.keep[:, None]).sum(0).detach(),  # [E]
            drop_fraction=(1.0 - keep_f.mean()).detach(),                # [] dropped / all assignments
            router_entropy=r.entropy,
            top_k=k,
            n_tokens=N,
        )
        if return_routing:
            aux.topk_idx = r.topk_idx.detach()                           # [N, k]
            aux.gates = r.gates.detach()                                 # [N, k]
            aux.kept = asg.keep.view(k, N).t().contiguous()              # [N, k]
            aux.extra["router_probs"] = r.probs.detach()                 # [N, E] fp32 (R1)
            aux.extra["routed_expert_out_norm"] = assign_norm.view(k, N).t().contiguous()  # [N, k] (R3)
        return y_flat.view(B, T, d), aux

    def _dispatch_loop(self, x_flat: torch.Tensor, asg: CapacityAssignment, gate_flat: torch.Tensor,
                       want_norms: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Readable reference: one expert at a time. Returns (y [N, d], per-assignment norms
        [kN] of g * E(x) if ``want_norms`` else None)."""
        N, d = x_flat.shape
        cdt = compute_dtype(x_flat)
        y = x_flat.new_zeros(N, d, dtype=cdt)                                   # [N, d]
        norms = x_flat.new_zeros(asg.expert.shape[0], dtype=torch.float32) if want_norms else None  # [kN]
        for e in range(self.n_routed):
            # Kept assignments of expert e, in priority order. Their count is <= C.
            with phase("dispatch"):
                sel = ((asg.expert == e) & asg.keep).nonzero(as_tuple=True)[0]  # [n_e]
                if sel.numel() == 0:
                    continue
                tok = asg.token[sel]                                            # [n_e]
                x_e = x_flat[tok]                                               # [n_e, d] gather
            with phase("expert_gemm"):
                out = self.experts.expert_forward(e, x_e).to(cdt)               # [n_e, d]  E_e(x)
            with phase("combine"):
                g = gate_flat[sel].to(cdt)[:, None]                             # [n_e, 1]  gate cast last
                y = y.index_add(0, tok, g * out)                                # Switch Eq. 2: y += p_e(x) E_e(x)
            if want_norms:
                norms[sel] = (g * out).detach().float().norm(dim=-1)
        return y, norms

    def _dispatch_batched(self, x_flat: torch.Tensor, asg: CapacityAssignment, gate_flat: torch.Tensor,
                          want_norms: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sort-and-segment into a capacity-padded buffer [E, C, d] (PLAN §7). Returns
        (y [N, d], per-assignment norms [kN] if ``want_norms`` else None)."""
        N, d = x_flat.shape
        E, C, k = self.n_routed, asg.capacity, self.top_k
        cdt = compute_dtype(x_flat)
        EC = E * C
        # Slot of each assignment in the flattened buffer; dropped ones go to a dump row EC.
        with phase("dispatch"):
            dest = torch.where(asg.keep, asg.expert * C + asg.pos, torch.full_like(asg.pos, EC))  # [kN]
            x_rep = x_flat.to(cdt).repeat(k, 1)                                 # [kN, d] row a = token a % N
            # Kept destinations are unique, so this index_add is a pure scatter (the dump row
            # collects dropped tokens and is thrown away). Padding rows stay zero.
            buf = x_rep.new_zeros(EC + 1, d).index_add(0, dest, x_rep)          # [EC+1, d]
            buf = buf[:EC].view(E, C, d)                                        # [E, C, d]
        with phase("expert_gemm"):
            out = self.experts.batched_forward(buf).to(cdt)                     # [E, C, d]
        with phase("combine"):
            out = torch.cat([out.reshape(EC, d), out.new_zeros(1, d)], dim=0)   # [EC+1, d] dump row = 0
            g = (gate_flat * asg.keep).to(cdt)[:, None]                         # [kN, 1] 0 for dropped
            y_assign = (out[dest] * g).view(k, N, d)                            # [k, N, d]
            # Sum the k choices with explicit adds in a fixed order. (Not .sum(0): CUDA autocast
            # promotes torch.sum to fp32, and a fixed order keeps runs bit-reproducible.)
            y = y_assign[0]                                                     # [N, d]
            for j in range(1, k):
                y = y + y_assign[j]
        norms = y_assign.detach().float().norm(dim=-1).reshape(-1) if want_norms else None  # [kN]
        return y, norms


class SwitchMoE(CapacityMoE):
    """Switch Transformer layer (top-1). Config: ``moe.switch`` (CONFIG_SCHEMA)."""

    variant = "switch"
