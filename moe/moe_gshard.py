"""GShard-style top-2 MoE layer (Lepikhin et al., arXiv 2006.16668), as used for comparison in
the Switch paper. User-approved shape: 16 experts of width d_ff/2, top-2 (PLAN, resolutions),
so both total (8 x d_ff) and activated (d_ff) FFN params match the other variants.

It reuses the capacity machinery of :class:`moe.moe_switch.CapacityMoE` with k = 2:
  * Router: softmax over 16 experts in fp32; each token takes its 2 highest-prob experts.
  * Gates: raw softmax probs p_1, p_2 (no renormalisation unless ``moe.renormalize_gates``).
    (GShard Algorithm 1 normalises g1, g2 by g1 + g2 and dispatches the 2nd expert
    stochastically; we keep the raw-prob gate that both papers we reproduce use, and route the
    2nd choice deterministically. Documented simplification.)
  * Capacity C = floor(2 * N / E * cf) (GShard Alg. 1 uses 2N/E; layer_api.expert_capacity).
  * Priority: ALL first choices claim buffer slots (in token order) before ANY second choice
    (GShard Alg. 1 computes the 2nd-choice positions after adding the 1st-choice counts). A
    token whose 2nd choice is dropped still gets p_1 * E_1(x); if both are dropped, y = 0.
  * Aux loss: Switch Eq. 4 form with f_i from the 1st choice (GShard's "c_e / S" uses top-1).
  * Drop fraction = dropped (token, choice) assignments / (2 N).
"""

from __future__ import annotations

from moe.moe_switch import CapacityMoE


class GShardMoE(CapacityMoE):
    """Top-2 capacity MoE. Config: ``moe.gshard`` (n_experts, width_divisor, top_k, ...)."""

    variant = "gshard"
