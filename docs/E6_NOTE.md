# E6 note: expert placement and device imbalance (bridge to WSC-LLM)

**This is a simple analytical model, not a simulator.** It uses real routing decisions from
`route_trace.py` and simple arithmetic. It has no network topology, no overlap of compute
and communication, no queueing, and no kernel-efficiency model. Code: `scripts/placement_sim.py`.
Outputs: `results/<cfg>/e6_placement/<run_id>/` (format in `docs/DASHBOARD_SPEC.md` §1.8).

## What is modelled

| Item | Choice |
|---|---|
| Step | one trace batch of N tokens |
| Devices | D ∈ {2, 4, 8}, each holds a subset of every MoE layer's routed experts |
| Token location | data-parallel shards: token t lives on device `floor(t·D/N)` (contiguous, so sequences stay together) |
| Round-robin | expert e → device e mod D |
| Greedy | longest-processing-time-first on the per-expert load of a **calibration subset** (first half of the steps by default, `--calib-frac`), **evaluated on the other steps**. At most ceil(E/D) experts per device, so expert memory stays as even as round-robin. |
| Shared experts | replicated on every device, run on local tokens: balanced, no traffic |
| Dropped tokens | (Switch/GShard capacity) neither computed nor sent |
| Load | kept (token, expert) assignments per device; straggler = max / mean device load |
| All-to-all | dispatch + combine per layer, each moves every off-device assignment once: `a2a_fwd = 2 × offdevice × d_model × bytes_per_element`, and fwd+bwd = 2 × a2a_fwd. Top-k copies are not de-duplicated. |
| Step time | `estimate_step_time`: per layer, slowest device's compute (4·d·w FLOPs per assignment forward, ×3 with backward, plus the shared expert) at `device_tflops`, plus the slowest sender's bytes at `link_gbps`. Layers run back to back with a barrier at each all-to-all. step = Σ_L max_D compute + Σ_L max_D comm. The balanced part (Σ_L mean_D compute) and the imbalance penalty (the difference) are reported separately. Only MoE FFN layers are counted. |
| Defaults | `device_tflops` = measured GEMM roof of the newest E5 run of the same config (else 50), `link_gbps` = 64 (about PCIe 5.0 x16 one way; an assumption). Both can be changed on the dashboard. |
| "all" layer rows | straggler = Σ_L max / Σ_L mean (the effective slowdown when every layer has a barrier) |

## How this connects to WSC-LLM

WSC-LLM (ISCA '25) co-explores wafer-scale chip architecture and scheduling for LLM
serving. Its central question is how compute, memory capacity/bandwidth, and on-wafer
communication trade off when work is spread over many dies. This toy model shows the same
three forces on an MoE layer:

1. **Compute balance.** Routing is not uniform, so whichever device holds the popular
   experts sets the critical path. Greedy placement lowers the straggler factor, but only
   as far as the memory constraint (a fixed number of experts per device) allows. At
   D = E (one expert per device) placement cannot help at all, because the imbalance is
   inside the router.
2. **Communication.** With more devices, each device computes less, but a larger
   fraction (about 1 − 1/D) of assignments leave their home device. The all-to-all
   bytes grow, so link bandwidth decides whether a step is compute-bound or
   communication-bound. The dashboard's bandwidth slider shows where that crossover is.
3. **Memory.** Expert weights must fit on each device. That is why the greedy policy is
   capped, and why fine-grained experts (DeepSeekMoE) are easier to balance: many small
   experts can be packed more evenly than a few large ones.

A wafer-scale design changes the constants (very high die-to-die bandwidth, limited
memory per die) but not the structure of this trade-off. Real systems add topology,
contention, compute/communication overlap, and capacity-aware routing, none of which are
modelled here.
