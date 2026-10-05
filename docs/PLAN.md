# PLAN.md: Milestone 1 plan (moe-architect, 2026-10-04)

Status: **APPROVED by user 2026-10-04** ("continue"), with these resolutions of §9:
- **GShard = 16 experts × d_ff/2, top-2** (8× total and activated both matched; router 6,144/layer). E3 GShard counts: n_dis {0,1,2,3,4}/16 = exact r.
- **dense_x8**: optional; run only if the Turbo re-probe puts it under the 45-min budget.
- **LR check**: deferred; peak LRs as in §3/§4 for now. Re-offer to the user before Milestone 4.
- Turbo re-probe happens before Milestone 4 (main runs); it does not block Milestones 2–3.

Number labels used below:
- **[computed]**: exact output of the throwaway calculator (`scratchpad/params.py`, not in the repo).
- **[measured]**: timed by the throwaway probe (`scratchpad/probe.py`) on the RTX 5070 Ti Laptop GPU or the CPU.
- **[extrapolated]**: arithmetic on measured numbers. The method is stated.
- **[estimate]**: judgment, not measured.

> **Important probe caveat.** During the probe the GPU was **power-capped at 33 W** (nvidia-smi: "Current Power Limit 33 W", default 80 W, max 120 W; `SW Power Cap: Active`; P5). Under load the memory clock was pinned at **810 MHz** (max 14001 MHz). Measured device-to-device copy bandwidth was **25.8 GB/s** and bf16 4096³ GEMM ran at **19.7 TFLOP/s**. Milestone 0 measured ~37 TFLOP/s in a better power state. The laptop was on AC with the battery at 80% ("Not charging"). All throughput numbers below are therefore **pessimistic**. Before fixing the `main` budget, re-run the probe with G-Helper on Turbo/Performance and the dGPU out of Eco mode. Command: `.venv/bin/python <scratchpad>/probe.py main dense,switch,deepseek 32`. It takes about 1 min.

---

## 1. File tree

This follows MASTER_PROMPT §3. Additions are marked with `+`.

```
moe-repro/
├── README.md  NOTES.md  RESULTS.md  requirements.txt  Makefile
├── MASTER_PROMPT.md  PROGRESS.md
├── docs/
│   ├── PLAN.md                 + this file
│   └── DASHBOARD_SPEC.md       + architect writes before any dashboard code (Milestone 3)
├── configs/  smoke.yaml  small.yaml  main.yaml
├── moe/
│   ├── data.py  model.py  ffn.py  router.py  moe_switch.py  moe_deepseek.py  moe_gshard.py
│   ├── metrics.py  flops.py  utils.py
│   ├── layer_api.py            + tiny dataclass `MoEAux` (aux_loss, routing stats) shared by all FFN variants
│   └── schedule.py             + warmup + step-decay LR (kept separate so it is testable)
├── scripts/  train.py  run_main.py  sweep_capacity.py  ablate_experts.py  profile_layer.py
│             placement_sim.py  route_trace.py  make_report.py
├── dashboard/
│   ├── app.py                  entry point: sidebar run selector + page navigation
│   ├── theme.py                + variant color map, Plotly template, units/labels
│   ├── data.py                 + cached readers for results/ (jsonl/json/npz), empty-state helpers
│   ├── layout.py               + shared page header, "no data yet" box, hardware label
│   ├── results_io.py           + pure loaders (no streamlit), shared with make_report.py
│   ├── figures.py              + figure builders (no streamlit), shared with make_report.py
│   └── pages/                  + overview.py training.py routing.py capacity.py ablations.py
│                                 systems.py placement.py how_it_works.py
├── tests/  test_router.py test_switch.py test_deepseek.py test_gshard.py test_aux_loss.py
│           test_param_matching.py test_fp32_router.py test_model.py test_determinism.py test_schedule.py
└── results/   <config_name>/<experiment>/<run_id>/...   (see docs/DASHBOARD_SPEC.md §1; traces at results/<cfg>/traces/<run_id>/)
```

## 2. Backbone decisions

| Choice | Decision | Why |
|---|---|---|
| Norm | **RMSNorm** (weight only, eps 1e-6), pre-norm, plus a final norm | DeepSeekMoE's backbone uses it. It has fewer parameters and no bias, so the parameter accounting stays simple. |
| Positions | **Rotary (RoPE)**, base 10000 | Adds 0 parameters, so the param table is just embeddings + attention + FFN, and it is the DeepSeek-family choice. |
| Embeddings | **Tied** input/output (counted once) | Required by the spec. At V=8192 it saves 3.1M params, which is about 23% of the dense model. |
| Biases | **None** in attention, FFN/experts, or router | Exact 8× matching becomes a clean equality, and it matches modern MoE practice. |
| Activation | GELU (tanh-free `F.gelu`) | Required by the spec. |
| Attention | `F.scaled_dot_product_attention(is_causal=True)` | Required by the spec. |
| Expert storage | Each MoE layer stores stacked weights `W_in[E, d, w]` and `W_out[E, w, d]`. Loop and batched paths read the **same** tensors. | The loop-vs-batched equivalence test compares math, not two parameter copies. |
| Data packing | Stories joined with an EOS token, cut into contiguous `seq_len` windows | Standard practice, and every token is used. |

## 3. Configs

| | smoke | small | main |
|---|---|---|---|
| n_layers / d_model / n_heads | 2 / 64 / 2 | 4 / 256 / 4 | **6 / 384 / 6** (master default) |
| seq_len / d_ff | 64 / 256 | 256 / 1024 | 256 / 1536 |
| batch (seqs × len = tokens/step) | 16 × 64 = 1,024 | 32 × 256 = 8,192 | 32 × 256 = **8,192** |
| total steps / total tokens | 150 / 153,600 | 1,500 / 12.29M | **3,000 / 24.58M** |
| warmup steps | 15 | 100 | 200 |
| eval interval / eval batches | 50 / 4 | 150 / 10 | 150 / 20 (163,840 val tokens) |
| routing-log interval | 10 | 50 | 50 |
| checkpoint interval | end only | 500 | 500 (resume supported) |
| peak LR | 3e-3 | 1.5e-3 | 1e-3 |
| device / dtype | CPU / fp32 (8 threads) | CUDA / bf16 autocast, fp32 master weights, fp32 router | same as small |
| vocab | 8192 (same cached tokenizer for all configs) | | |

Why batch 32×256 instead of 64×256: under the current power cap, bs32 had higher measured tok/s (dense 18.0k vs 15.7k) **[measured]**, and it gives 2× more optimizer steps for the same tokens.

**What "smoke under ~3 min" means:** the **whole `make smoke` pipeline** (E1 with 5 variants, E2 with 4 capacity factors, E3/E4 eval-only, a tiny E5 run, E6, report) on CPU, *after* the one-time data download and tokenizer training are cached. It does not include `make test`. Each smoke training run is about 8–11 s of steps **[extrapolated: 150 steps × 50–69 ms/step measured on CPU, 8 threads]**, plus about 3 s of Python/torch startup. That is roughly 9 runs × ~14 s ≈ 2 min of training, plus about 1 min for everything else **[estimate]**. If it overshoots, cut smoke to 100 steps.

## 4. Peak LR and schedule

- **Peak LR 1e-3 for main.** For roughly 14M-active-param GPT-style models, AdamW with β=(0.9, 0.95) usually trains stably at 1e-3 to 3e-3 **[estimate]**. I picked the low end because (a) the MoE routers are the fragile part (Switch §2.4 ties instability to the router), (b) the token budget is short, so a warmup of 200 steps (6.7%) must be enough to settle the routers, and (c) 31 fine-grained experts each get only about 3/31 of the gradient signal. The same LR is used for every variant, with no per-variant tuning. **Option (needs your OK):** a one-time 3-point check {5e-4, 1e-3, 2e-3} on the `small` config for dense+switch only, with the chosen value applied to all variants and documented.
- **Schedule** (DeepSeekMoE warmup-and-step-decay, MASTER §4.8), with S = total steps:
  - `lr(t) = peak · (t+1)/warmup` for t < warmup
  - `lr(t) = peak` for warmup ≤ t < 0.8·S
  - `lr(t) = peak · 0.316` for 0.8·S ≤ t < 0.9·S
  - `lr(t) = peak · 0.316² ≈ 0.0999·peak` for t ≥ 0.9·S
  - Main: decay steps at 2400 and 2700.
- AdamW β=(0.9, 0.95), wd=0.1 (applied to matrices only; norms, embeddings, and router excluded), grad clip 1.0. Init: truncated normal ±2σ, σ = sqrt(0.1 / fan_in) (Switch §2.4), for every variant. For expert `W_in`, fan_in = d. For `W_out`, fan_in = w (expert width).

## 5. Param/FLOP matching (main config) [computed]

Formulas (no biases; FFN = `W_in[d,w] + W_out[w,d]`; FLOPs count 1 MAC = 2 FLOPs; GELU and gate multiplies are ignored):
- FFN params per expert = `2·d·w`. Total FFN/layer = `2·d·w·(n_shared + n_routed)`. Activated FFN/layer = `2·d·w·(n_shared + k)`.
- Router params/layer = `d·n_routed`.
- Total params = `V·d + L·(4d² + 2d + totFFN + router) + d`. Active params/token = the same expression with actFFN.
- FFN fwd FLOPs/token/layer = `2·actFFN`. Router fwd FLOPs/token/layer = `2·d·n_routed`.
- Whole-model fwd FLOPs/token (for reference) = `L·(8d² + 2·actFFN + 2·d·n_routed + 4·seq·d) + 2·V·d`. The attention-score term is the full non-causal count, so it is an upper bound.

Main: d=384, d_ff=1536, L=6, V=8192. Embedding = 3,145,728; attention/layer = 589,824; norms/layer = 768.

| variant | experts | expert width | activated/token | total FFN params/layer | activated FFN params/layer | router params/layer | total model params | active params/token | FFN fwd FLOPs/token/layer | FFN fwd FLOPs/token (6 layers) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 1 | 1536 | 1 | 1,179,648 | 1,179,648 | 0 | 13,767,552 | 13,767,552 | 2,359,296 | 14,155,776 |
| switch | 8 routed | 1536 | top-1 | 9,437,184 | 1,179,648 | 3,072 | 63,331,200 | 13,785,984 | 2,359,296 | 14,155,776 |
| deepseek | 1 shared + 31 routed | 384 | 1 + top-3 | 9,437,184 | 1,179,648 | 11,904 | 63,384,192 | 13,838,976 | 2,359,296 | 14,155,776 |
| gshard (approved: 16 × d_ff/2) | 16 routed | 768 | top-2 | 9,437,184 | 1,179,648 | 6,144 | 63,349,632 | 13,804,416 | 2,359,296 | 14,155,776 |
| dense_x8 | 1 | 12288 | 1 | 9,437,184 | 9,437,184 | 0 | 63,312,768 | 63,312,768 | 18,874,368 | 113,246,208 |

Notes:
- **DeepSeek subtlety:** 1 shared + 31 routed experts of width d_ff/4 = 32 × d_ff/4 = **8 × d_ff total**, the same as Switch's 8 × d_ff. Activated = (1 shared + 3 routed) × d_ff/4 = **d_ff**, the same as dense. The shared expert counts toward both totals.
- **GShard inconsistency in the master prompt:** the §4.3 row (8 experts × d_ff/2, top-2) matches *activated* compute but has only **4 × d_ff total** (4,718,592 vs 9,437,184). That contradicts the §4.3 rule that every MoE variant has 8× total. **Open question for you** (see §9). The option that keeps both totals matched is **16 experts × d_ff/2, top-2** → 9,437,184 total and 1,179,648 activated, router 6,144/layer.
- **Router params differ** between variants (0 / 3,072 / 6,144 / 11,904 per layer, at most 71,424 for the whole model, about 0.11% of total and 0.5% of active). The "matched" claim is stated precisely as: *expert-FFN total params and activated expert-FFN params/FLOPs are exactly equal; router params/FLOPs are reported separately and excluded from the matching.* `test_param_matching.py` asserts the exact expert-FFN equalities. The startup table and dashboard show the router column.
- Whole-model fwd FLOPs/token: dense 29,884,416; switch 29,921,280; deepseek 30,027,264; gshard (16×768) 29,958,144; dense_x8 128,974,848.

**Smoke config** (d=64, d_ff=256, L=2) **[computed]**: total / active params are dense 622,912 / 622,912; switch 1,082,688 / 623,936; deepseek 1,085,632 / 626,880; gshard(16×128) 1,083,712 / 624,960; dense_x8 1,081,664 / 1,081,664. Total FFN/layer is 262,144 (8×) for switch, deepseek, and dense_x8. Activated FFN/layer is 32,768 for all MoE variants. The embedding (524,288) dominates at this size, which is expected for a pipeline check.

**Small config** (d=256, d_ff=1024, L=4) **[computed]**: dense 5.25M; switch 19.93M total / 5.25M active; deepseek 19.96M / 5.28M.

## 6. Throughput probe and run-time estimates

Probe: a minimal pre-norm decoder (RMSNorm, RoPE, SDPA, tied embedding, AdamW fused, grad clip), bf16 autocast, fp32 router, random tokens. Each run used 10 warmup steps and then the median of 30 steps (20 at the large size). MoE layers use a **naive Python loop dispatch** (boolean mask per expert, index_add). The real batched path should be equal or faster, so the MoE numbers are conservative. **The GPU was in the 33 W power-capped state described at the top.**

**Measured, main config (6/384/1536, seq 256):**

| variant | bs64 (16,384 tok) step ms | bs64 tok/s | bs64 peak mem | bs32 (8,192 tok) step ms | bs32 tok/s | bs32 peak mem |
|---|---|---|---|---|---|---|
| dense | 1042.7 | 15,713 | 3,602 MB | 454.1 | 18,039 | 1,907 MB |
| switch (cf 1.25) | 1233.9 | 13,278 | 4,652 MB | n/a | n/a | n/a |
| deepseek | 1763.8 | 9,289 | 5,317 MB | 671.5 | 12,199 | 3,096 MB |
| gshard (8×768, probe only; shipped config is 16×768) | 1494.0 | 10,966 | 4,570 MB | n/a | n/a | n/a |
| dense_x8 | 2565.3 | 6,387 | 8,298 MB | n/a | n/a | n/a |

Also **measured**: small config at bs64 runs at 28.3k / 27.6k / 20.6k / 24.6k / 16.0k tok/s (dense / switch / deepseek / gshard / dense_x8), with peak memory ≤ 4.5 GB. Smoke on CPU with 8 threads takes 50.6 / 52.4 / 68.5 / 52.2 / 61.4 ms per step. 16 threads was slower (59.5 / 94.2 ms), so smoke pins 8 threads. A torch.profiler pass on dense showed the step dominated by elementwise kernels and copies (aten::mul + copy_ ≈ 42% of CUDA time) rather than GEMMs. That is consistent with the 810 MHz memory clock.

**Run-time estimates, main, 3000 steps × 8192 = 24.58M tokens, 33 W state [extrapolated].** Method: time = tokens / tok_s + eval cost. Eval cost assumes a forward pass costs about 1/3 of a training step **[estimate]** (20 evals × 20 batches). Switch, gshard, and dense_x8 bs32 throughput are scaled from their bs64 ratio to dense.

| experiment | runs | estimate (33 W state) |
|---|---|---|
| E1 required | dense 23.7 + switch 28.1 + deepseek 35.1 min | **~87 min** |
| E1 optional | gshard 34.0, dense_x8 **58.3 (over the 45-min budget at 33 W)** | +92 min |
| E1 second seed (main 3) | 3 runs | +87 min |
| E2 capacity sweep | cf ∈ {0.75, 1.0, 1.25, 2.0}. cf 1.25 reuses the E1 switch run (same seed and config), so 3 new runs of about 26–31 min each | **~85 min** |
| E3 / E4 (eval-only) | about 11 E3 points + about 6 E4 points, each over 50 val batches (410k tokens) | **~5 min [estimate]** |
| E5 profiling | 3 variants × 2 dispatch × fwd/fwd+bwd × 7 token counts (1K–64K) + expert-count sweep (4–64), median of 50 repeats | **~20–30 min [estimate]** |
| E6 | route_trace about 1 min per model, sim takes seconds | **~5 min [estimate]** |
| small (any variant) | 1500 × 8192 tokens | 7.4–13.1 min [extrapolated from bs64 small] |

At Turbo (80–120 W, full memory clock), I expect these to shrink substantially, perhaps 2–5× **[estimate, unmeasured]**. The dashboard and README will only show measured times.

**Headroom / optional larger main (proposal only, not applied).** VRAM is clearly not the constraint: main bs32 peaks at ≤ 3.1 GB (bs64 at ≤ 5.3 GB, dense_x8 8.3 GB) **[measured]**. A "main-L" config at **8 / 512 / 8 heads, d_ff 2048** is **[computed]** at 29.37M params for dense, 146.84M total / 29.40M active for switch, and 146.94M / 29.50M for deepseek. It measured 9.5k (dense) and 6.5k (deepseek) tok/s at bs32, with peak memory 2.8 / 5.3 GB (dense_x8 8.0 GB) **[measured, 33 W]**. That would be about 63 min for deepseek at 24.6M tokens in the current state, so I only recommend it if the Turbo re-probe shows a speedup of 1.5× or more. Otherwise keep main at the master default. A cheaper alternative with the same model is to double main's budget to 6000 steps (49M tokens) if Turbo gives 2× or more.

## 7. MoE layer design decisions

- **Common layer API** (`moe/layer_api.py`): every FFN variant's `forward(x[B,T,d]) -> (y[B,T,d], MoEAux)`. `MoEAux` holds `aux_loss` (scalar fp32 tensor, already multiplied by its α), `expert_counts[E]` (pre-drop, int), `kept_counts[E]`, `drop_fraction`, `router_entropy`, and optional `topk_idx[N,k]` / `gates[N,k]` / `kept[N,k]` when `return_routing=True`. Dense returns `MoEAux` with zero aux. `model.py` (builder) only sums `aux.aux_loss` and collects stats. It never looks inside routing.
- **Switch batched dispatch: sort-and-segment into a capacity-padded buffer**, not a materialized one-hot `[T, E, C]`. At E5's 64K tokens, `[T, E, C]` would be 65,536 × 8 × 10,240 ≈ 5.4G elements, which does not fit. Steps:
  1. `expert_idx[N]` = argmax.
  2. `pos_in_expert[N]` = cumsum over the one-hot `[N, E]` in token order (Switch Fig. 15).
  3. `keep = pos < C`.
  4. Scatter kept tokens into `buf[E, C, d]` (zeros elsewhere, so the padding overhead is real and measurable).
  5. `torch.bmm` with stacked `W_in[E, d, w]` → GELU → `W_out`.
  6. Gather back × gate.

  This is mathematically identical to the Fig. 16 einsums. `test_switch.py` builds the literal `[T, E, C]` dispatch/combine einsum on a tiny T and checks it gives the same output, so the paper's formulation is still in the repo as a teaching reference.
- **GShard:** top-2 uses the same machinery. Capacity is filled by **first choices first, then second choices** (GShard priority). The aux loss uses the top-1 `f_i` (GShard / Switch Eq. 4 form). Drop fraction = dropped (token, slot) assignments / all assignments.
- **DeepSeekMoE batched:** there is no capacity, so `[N, k]` assignments are flattened and sorted by expert id. `counts = bincount`. Tokens are packed into a padded `buf[E, max_count, d]` (one host sync for `max_count`), then bmm, then weighted `index_add` back. The shared expert is a plain dense FFN (width d_ff/4) applied to all tokens and added with no gate (DeepSeekMoE Eq. 9). E5 reports the padding waste. `torch._grouped_mm` is private, so it is not used by default.
- **fp32 router boundary:** inside `router.py`, under `torch.autocast(enabled=False)`: `x.float()` → jitter `× U(1−ε, 1+ε)` (train only, applied to the router *input*) → `W_r` (stored fp32) → logits → softmax → top-k → aux loss, all in fp32. Only the gate values are cast to the activation dtype, right before the combine multiply. Indices stay int64. Experts run under bf16 autocast. `test_fp32_router.py` checks the dtypes.
- **Dropped tokens (Switch, GShard):** a dropped token's MoE output is exactly zero, so it passes through the residual only. Priority is flattened (batch, seq) order, as in the paper's cumsum. `f_i` for the aux loss uses the argmax *before* dropping. Eval uses the training capacity factor unless overridden (`eval_capacity_factor`). Caveat for NOTES.md: dropping depends on other tokens in the batch, including later positions, so it is not strictly causal. The paper has the same property.
- **Aux loss across layers:** `total = CE + Σ_layers aux_l`. Each `aux_l` already includes its own coefficient (Switch: `α·N·Σ f_i P_i`, α=0.01; DeepSeek: `α1·Σ f_i P_i`, `f_i = N'/(K'T)·count_i`, α1=0.01). Summed, not averaged. Per-layer and summed aux are logged separately from CE. `f_i` and `P_i` are computed over all B×T tokens of the step at that layer.
- **E3 "disable top-r fraction" → discrete counts.** Per token, the n_dis highest-probability routed experts are masked out, then top-k is taken from the rest with the gate = the original softmax prob (no renormalization). E3 uses **no-drop eval** (capacity = N) for Switch and GShard, so the loss change reflects expert redundancy and not capacity effects. The r=0 point at the training cf is also shown for reference.

| variant | n_routed | n_dis used | fraction disabled | target r |
|---|---|---|---|---|
| DeepSeek | 31 | 0, 2, 4, 6, 8 | 0, 0.065, 0.129, 0.194, 0.258 | 0, 1/16, 2/16, 3/16, 4/16 (round(r·31)) |
| Switch | 8 | 0, 1, 2 | 0, 0.125, 0.25 | 0, 2/16, 4/16 (1/16 and 3/16 are not representable) |
| GShard (16, approved) | 16 | 0, 1, 2, 3, 4 | 0, 0.0625, 0.125, 0.1875, 0.25 | 0, 1/16, 2/16, 3/16, 4/16 (exact) |

  Plots use the actual fraction on the x-axis.
- **Routing logs during training:** the counts stay on the GPU and accumulate. They are copied to the CPU only every routing-log interval, to avoid a sync every step.
- **E6 routing traces:** `route_trace.py` loads a checkpoint, runs eval with `return_routing=True` over 50 fixed val batches, and saves `results/<cfg>/traces/<run_id>/trace.npz` (path per DASHBOARD_SPEC §1) with `topk_idx[n_batch, L, N, k]` (int16), `gates` (fp16), and `kept` (bool), plus metadata JSON. `placement_sim.py` treats each batch as one "step". The same function serves the dashboard's "How it works" page on a single typed sentence.

## 8. Delegation plan for Milestone 2

| # | Task | Owner | Files (exclusive) | Depends on |
|---|---|---|---|---|
| T1 | Data: TinyStories download (Shakespeare fallback), BPE-8192 training on a subset, uint16 `.bin` cache, fixed val split, batch iterator | builder | `moe/data.py` | none |
| T2 | Utils, configs, Makefile skeleton: seeding, device/dtype detection, env.json, run-id, checkpoint save/resume, jsonl logger; the 3 YAMLs from §3 | builder | `moe/utils.py`, `configs/*.yaml`, `Makefile` | none |
| T3 | Layer API + LR schedule + flops.py (formulas from §5) | architect | `moe/layer_api.py`, `moe/schedule.py`, `moe/flops.py` | none |
| T4 | Router + MoE layers (loop + batched): Switch, DeepSeek, GShard; metrics.py | architect | `moe/router.py`, `moe/moe_*.py`, `moe/metrics.py` | T3 |
| T5 | Backbone: dense `ffn.py`, `model.py` (RMSNorm, RoPE, attention, tied embedding, FFN factory by variant, sums `MoEAux.aux_loss`, startup param table via `flops.py`) | builder | `moe/ffn.py`, `moe/model.py` | T3 (API only) |
| T6 | Routing/dispatch tests: loop==batched (outputs + grads), einsum reference, capacity dropping, aux sanity, param matching, fp32 router, shared expert, E3 masking, schedule | architect | `tests/test_router.py`, `test_switch.py`, `test_deepseek.py`, `test_gshard.py`, `test_aux_loss.py`, `test_param_matching.py`, `test_fp32_router.py`, `test_schedule.py` | T3, T4 |
| T7 | Model shape tests for every variant + determinism (10 steps, same seed → same losses) | builder | `tests/test_model.py`, `tests/test_determinism.py` | T1 (tiny synthetic ok), T4, T5 |
| T8 | Review gate: how model.py calls the MoE layers and combines CE + aux | architect | read-only, then report | T5, T7 |
| T9 | `make test` green; report to user | orchestrator | none | all |

**Parallel waves (no shared files):**
- Wave 1: T1, T2, T3 at the same time.
- Wave 2: T4 (architect) and T5 (builder) at the same time. T5 codes against the T3 API.
- Wave 3: T6 and T7 at the same time.
- Then T8 and T9.

`docs/DASHBOARD_SPEC.md` (architect) can be written any time during Milestone 2, because it touches no code. `train.py` and the runners are Milestone 3.

## 9. Risks and open questions

1. **GPU power state (biggest issue).** The probe ran at a 33 W cap with an 810 MHz memory clock. Please switch G-Helper to Turbo/Performance, make sure the dGPU is not in Eco mode, then re-run the probe so the `main` budget can be fixed (3000 steps, or 6000 steps / main-L if there is headroom). E5 will log clock, temperature, and power around every benchmark anyway.
2. **GShard total params (decision needed).** Keep §4.3 as written (8 × d_ff/2, which is 4× total, not matched) or use **16 × d_ff/2 top-2** (8× total and activated both matched; recommended)? This affects the E3 counts (see table).
3. **dense_x8** takes about 58 min at 33 W, which is over the 45-min budget. Run it only if Turbo brings it under, or accept the overrun since it is optional.
4. **Will MoE beat dense?** At about 25M tokens on a 14M-active model, each Switch expert sees about 1/8 of the tokens, and DeepSeek's routed experts see about 3/31. The expected gain is small, perhaps a few hundredths of a nat **[estimate]**, and it could be within seed noise. That is why the 2nd seed matters. If MoE loses or ties, it gets reported honestly with likely reasons: short budget, small d_model, router warm-up cost.
5. **Python 3.14:** torch 2.11 and the pinned packages install and run. `torch.compile`/Dynamo support on 3.14 is uncertain, so it stays off by default. If a dependency fails on 3.14, I will report it rather than downgrade silently.
6. **WSL RAM ~15 GB:** tokenize in streamed chunks and cache only the needed train subset (≈ 2× the main budget ≈ 50–100M tokens ≈ 100–200 MB uint16) plus validation.
7. **Laptop thermals:** long runs may throttle mid-run, so per-step tok/s is logged to make it visible. Throughput comparisons should come from E5 microbenchmarks, not E1 wall-clock.
8. **LR not tuned per variant** (deliberate, for fairness). Should I run the optional 3-point LR check on `small` (about 30 min) before main?
