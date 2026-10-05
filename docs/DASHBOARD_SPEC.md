# DASHBOARD_SPEC.md: results contract + dashboard design (moe-architect, 2026-10-04)

Status: **design spec, no code yet.** Milestone 3 implements it.

- **architect** builds the shared layer (`app.py`, `theme.py`, `results_io.py`, `data.py`, `layout.py`, `figures.py` skeleton) and `pages/how_it_works.py`.
- **builder** implements the 7 standard pages and fills in their figure builders in `figures.py`.

**§1 is the single source of truth for every file under `results/`.** `train.py`, `run_main.py`, `sweep_capacity.py`, `ablate_experts.py`, `profile_layer.py`, `route_trace.py`, `placement_sim.py`, `make_report.py` and the dashboard all follow it. If a writer needs a new field, add it here first. Adding fields is fine. Renaming or removing a field needs the architect.

Contents:
1. Results file contract
2. Global layout
3. Theme
4. Pages (Overview, Training, Routing, Capacity, Ablations, Systems, Placement)
5. "How it works" page, plus requests to the MoE layer owner
6. Static report and shared figure builders
7. No-hard-coded-numbers rule
8. Builder checklist

---

## 1. Results file contract

### 1.1 Directory layout and smoke/small/main separation

**Decision:** the config name is the **first path component** under `results/`. Smoke, small and main outputs never mix, and the sidebar's config filter just picks a subdirectory. Every JSON file also stores `config_name`, so a file is self-describing even after it is moved.

```
results/
└── <config_name>/                      # smoke | small | main  (= cfg["name"])
    ├── e1_main/<run_id>/               # E1 training runs (run_main.py -> train.py)
    ├── e2_capacity/<run_id>/           # E2 training runs (sweep_capacity.py -> train.py)
    │   └── sweep_index.json            #   (directly under e2_capacity/, not in a run dir)
    ├── e3e4_ablation/<run_id>/         # one dir per ablate_experts.py invocation
    ├── e5_profile/<run_id>/            # one dir per profile_layer.py invocation
    ├── traces/<source_run_id>/         # route_trace.py output, one per trained checkpoint
    └── e6_placement/<run_id>/          # one dir per placement_sim.py invocation
```

- Writers create run dirs with the existing helper `moe.utils.make_run_dir(cfg, experiment=f"{cfg['name']}/<experiment>", run_id=...)`. No utils change is needed.
- **Amendment to PLAN §1/§7:** the trace path becomes `results/<config_name>/traces/<source_run_id>/` (PLAN said `results/traces/<run_id>/`). The orchestrator should mirror this in PLAN.md.
- `results/sample/` (kept by `make clean-results`) has the same layout, e.g. `results/sample/smoke/...`. The dashboard does **not** read it unless `MOE_RESULTS_DIR=results/sample` is set (see §2.6).
- Experiment directory names are constants in `dashboard/results_io.py`: `EXPERIMENTS = ("e1_main", "e2_capacity", "e3e4_ablation", "e5_profile", "traces", "e6_placement")`.

### 1.2 Run IDs and identity

- `run_id` = `moe.utils.make_run_id(variant_or_kind, tag)`, i.e. `YYYYmmdd-HHMMSS_<variant>[_<tag>]`:
  - E1: tag `s<seed>`, e.g. `20261004-153012_switch_s1337`
  - E2: tag `cf<cf:g>_s<seed>`, e.g. `..._switch_cf0.75_s1337`
  - E3/E4: `..._ablation`. E5: `..._profile`. E6: `..._placement`.
- **The dashboard never parses `run_id`.** Identity comes from `run.json` (below). `run_id` is only a display string and a sort key. The timestamp prefix makes lexical order = chronological order.
- A training run is **complete** iff `final.json` exists. Otherwise it is *running or crashed*. The dashboard shows such runs in the selector with a "(incomplete)" suffix and plots their partial logs. Headline numbers and tables skip them.

### 1.3 Common JSON conventions (all writers)

| Rule | Detail |
|---|---|
| Encoding | UTF-8. `.json` = one object. `.jsonl` = one object per line, `\n`-terminated, append-only. |
| Non-finite numbers | Write `null`, never `NaN`/`Infinity` (they are not valid JSON). "No-drop" capacity is written as `"capacity_factor": null` plus `"no_drop": true`. |
| Tensors | Convert with `moe.utils._to_jsonable` (0-dim → number, 1-dim → list). Counts are ints. |
| Units in names | Field suffixes carry the unit: `_ms`, `_s`, `_mb` (**MiB**, = bytes/1024², as `moe.utils.peak_memory_mb` computes), `_gbps` (1e9 B/s), `_tflops` (1e12 FLOP/s), `_bytes`, `flops` (count). |
| `schema_version` | `1` in `run.json`, `final.json`, `ablation_meta.json`, `profile_meta.json`, trace `meta.json`, `placement_params.json`. |
| Partial lines | The dashboard reader ignores a trailing line that fails to parse (file being written). |
| Resume | `train.py` appends to the same jsonl files after resuming. **The reader de-duplicates by `step` (and by `(step, layer)` for routing), keeping the last occurrence.** Writers do not need to truncate. |
| Paths inside files | Relative to `results/` (e.g. `"main/e1_main/20261004-153012_switch_s1337"`), so a results tree can be moved. |

### 1.4 E1/E2 training run directory (`train.py`)

```
<run_dir>/
  run.json            identity + status (written at start, rewritten at end)
  config.json         fully resolved config (CONFIG_SCHEMA keys, after overrides)
  env.json            moe.utils.get_env_info(...) output (+ fields below)
  train_log.jsonl     one record per log_interval steps
  eval_log.jsonl      one record per eval (incl. step 0)
  routing_log.jsonl   one record per (routing_log_interval, MoE layer); MoE variants only
  final.json          written once at the end; its presence = complete
  checkpoint.pt       final; ckpt_step<N>.pt periodic (moe.utils.save_checkpoint)
```

**`run.json`**

| field | type | meaning |
|---|---|---|
| `schema_version` | int | 1 |
| `run_id` | str | dir name |
| `experiment` | str | `e1_main` or `e2_capacity` |
| `config_name` | str | `cfg["name"]` |
| `variant` | str | `cfg["variant"]` |
| `seed` | int | `cfg["seed"]` |
| `group` | str | runs that differ only by seed share a group. E1: the variant (`"switch"`). E2: `"switch_cf0.75"`. **Seed aggregation groups by `group`.** |
| `capacity_factor` | float or null | training cf for switch/gshard, else null |
| `dispatch` | str | `cfg["moe"]["dispatch"]` |
| `overrides` | list[str] | the raw `--override` strings |
| `status` | str | `"running"`, then `"completed"` (`"failed"` if an exception is caught) |
| `started_at` / `finished_at` | ISO str / null | |
| `resumed_from_step` | list[int] | steps at which the run was resumed (empty if never) |

**`env.json`**: exactly `moe.utils.get_env_info(device, dtype)`. Keys used by the dashboard are `hardware_label` (must be `"RTX 5070 Ti Laptop GPU"` on this machine, `"CPU"` for smoke), `gpu_name`, `device_type`, `dtype`, `torch_version`, `cuda_version`, `total_vram_gb`, `gpu_state` (`sm_clock_mhz, mem_clock_mhz, temperature_c, power_draw_w, power_limit_w`), `timestamp`. Profile and placement runs write the same `env.json`.

**`train_log.jsonl`**: one record per logged step. `step` = number of optimizer updates completed (1-based).

| field | unit | meaning |
|---|---|---|
| `step` | int | 1..total_steps |
| `tokens_seen` | tokens | `step * batch_size * seq_len` |
| `lr` | | LR used for this step |
| `ce_loss` | nats | train CE of this step's batch |
| `aux_loss` | | Σ over MoE layers of aux (each already × α). 0.0 for dense. |
| `total_loss` | | `ce_loss + aux_loss` |
| `grad_norm` | | pre-clip global L2 norm (`clip_grad_norm_` return value) |
| `tokens_per_sec` | tok/s | `tokens_per_step / step_time` |
| `step_time_ms` | ms | wall time of fwd+bwd+opt step (data fetch included, eval excluded), after `cuda.synchronize()` |
| `peak_mem_mb` | MiB | `torch.cuda.max_memory_allocated` since run start (monotone). 0.0 on CPU. |
| `wall_time_s` | s | elapsed since run start, incl. eval |

**`eval_log.jsonl`**: one record at step 0 (before training), every `eval_interval`, and at the final step. Fixed `eval_batches` val batches. Model in eval mode, so eval capacity factor rules apply.

| field | unit | meaning |
|---|---|---|
| `step`, `tokens_seen` | | as above |
| `val_loss` | nats | mean CE over val tokens (CE only, no aux) |
| `val_ppl` | | `exp(val_loss)` |
| `val_aux_loss` | | Σ-layer aux averaged over val batches (logged only, never added) |
| `val_drop_fraction` | 0..1 | mean over MoE layers and val batches of `MoEAux.drop_fraction`. `null` for dense/dense_x8/deepseek. |
| `n_val_tokens` | tokens | |

**`routing_log.jsonl`**: written every `routing_log_interval` steps, **one record per MoE layer**. Counts are **accumulated over the window** (the steps since the previous record) on the GPU, as PLAN §7 says.

| field | type | meaning |
|---|---|---|
| `step` | int | last step of the window |
| `window_steps` | int | number of steps accumulated |
| `layer` | int | block index (0-based, `layer_api.is_moe_layer` positions only) |
| `n_experts` | int | routed experts E (`MoEAux.n_experts`) |
| `top_k` | int | k |
| `n_tokens` | int | tokens seen by the layer in the window (`window_steps * B * T`) |
| `expert_counts` | list[int] (E) | Σ over window of `MoEAux.expert_counts` (pre-drop demand, all k choices) |
| `kept_counts` | list[int] (E) | Σ over window of `MoEAux.kept_counts` |
| `drop_fraction` | 0..1 | `1 - sum(kept_counts)/sum(expert_counts)` over the window. 0.0 for deepseek. |
| `router_entropy` | nats | mean over window steps of `MoEAux.router_entropy` |
| `max_entropy` | nats | `ln(E)` (written so the dashboard need not recompute it) |
| `load_imbalance` | ratio | `max(expert_counts) / mean(expert_counts)` (demand, pre-drop) |
| `load_cv` | ratio | `std(expert_counts) / mean(expert_counts)`, population std (ddof=0) |
| `aux_loss` | | mean over window steps of this layer's `aux.aux_loss` |
| `extra` | dict[str, float] | mean over window of every **0-dim** tensor in `MoEAux.extra` (e.g. split aux terms). Non-0-dim entries are skipped (see §5.4). |

**`final.json`**

| field | unit | source |
|---|---|---|
| `schema_version`, `run_id`, `experiment`, `config_name`, `variant`, `seed`, `group`, `capacity_factor` | | same as run.json |
| `status` | | `"completed"` |
| `steps`, `tokens_seen` | | last step |
| `final_val_loss`, `final_val_ppl` | nats, | last eval_log record |
| `best_val_loss`, `best_val_step` | | min over eval_log |
| `final_train_ce` | nats | mean `ce_loss` over the last 5% of steps (≥ 1 step) |
| `total_params` | count | **measured**: `model.num_params()` |
| `total_params_analytic`, `active_params` | count | `flops.param_row(variant, cfg)["total_params"/"active_params"]` |
| `ffn_total_model`, `ffn_active_model`, `router_model` | count | `flops.param_row` |
| `model_fwd_flops_per_token`, `ffn_flops_per_token_model` | FLOPs | `flops.param_row` |
| `median_tokens_per_sec` | tok/s | median of train_log `tokens_per_sec`, skipping the first `max(10, 5% of steps)` steps (**the dashboard's throughput number**) |
| `mean_tokens_per_sec` | tok/s | `tokens_seen / Σ step_time` |
| `peak_mem_mb` | MiB | max over the run |
| `train_time_s`, `wall_time_s` | s | Σ step_time / total elapsed incl. eval |
| `val_drop_fraction` | 0..1 or null | final eval record |
| `train_drop_fraction_last` | 0..1 or null | mean `drop_fraction` over all layers in the routing records of the last 10% of steps |
| `checkpoint` | str | path relative to `results/` of `checkpoint.pt` |
| `hardware_label`, `device_type`, `dtype` | | copied from env.json |

**`e2_capacity/sweep_index.json`** (written by `sweep_capacity.py`; lets E2 reuse the E1 cf=1.25 run without copying it):
```json
{"schema_version": 1, "config_name": "main", "created_at": "...",
 "points": [{"capacity_factor": 0.75, "seed": 1337, "run_dir": "main/e2_capacity/2026..._switch_cf0.75_s1337"},
            {"capacity_factor": 1.25, "seed": 1337, "run_dir": "main/e1_main/2026..._switch_s1337", "reused_from": "e1_main"}]}
```
If the index is missing, the dashboard falls back to scanning `e2_capacity/*/run.json`.

### 1.5 E3/E4 ablation (`ablate_experts.py`, eval-only)

```
results/<cfg>/e3e4_ablation/<run_id>/
  ablation_meta.json   {schema_version, run_id, config_name, created_at, n_val_batches, n_val_tokens,
                        source_runs: [{run_dir, variant, seed}], note: "eval-time ablation; not retraining"}
  env.json
  ablation.jsonl       one row per evaluated point
```

**`ablation.jsonl` row**

| field | type | meaning |
|---|---|---|
| `study` | str | `"e3_disable_top"`, `"e4_shared"`, or `"e4_k_sweep"` |
| `source_run_dir` | str | relative to `results/` |
| `variant`, `seed` | | from the source run |
| `n_routed`, `top_k_trained` | int | from the model |
| `n_disable_top_routed` | int | count masked per token (0 = baseline) |
| `frac_disabled` | float | `n_disable_top_routed / n_routed` (**the E3 x-axis**) |
| `disable_shared` | bool | |
| `k_routed` | int | effective k at eval (`k_routed_override` or trained k) |
| `condition` | str | `e4_shared` only: `"baseline"`, `"no_shared_same_k"` (a), `"no_shared_plus_one"` (b: k+1) |
| `active_ffn_width` | int | `(n_shared_active + k_routed) * width` computed from `ffn_spec`, so the "compute kept equal" claim is visible |
| `capacity_factor` | float or null | eval cf. `null` with `no_drop: true` for E3 (PLAN §7) |
| `no_drop` | bool | |
| `is_reference` | bool | true for the extra r=0 point at the *training* cf (PLAN §7) |
| `val_loss`, `val_ppl` | nats, | |
| `baseline_val_loss` | nats | same source run, same study, `n_disable=0`, `disable_shared=False`, trained k, same cf/no_drop |
| `delta_val_loss` | nats | `val_loss - baseline_val_loss` |
| `val_drop_fraction` | 0..1 or null | |

### 1.6 E5 profiling (`profile_layer.py`)

```
results/<cfg>/e5_profile/<run_id>/
  profile_meta.json    {schema_version, run_id, config_name, created_at, d_model, base_d_ff, dtype,
                        warmup_iters, repeats, timer: "cuda_events"|"perf_counter",
                        flops_model: {hidden_traffic, include_autocast_weight_cast},
                        router_init: "random"|"checkpoint:<run_dir>", note}
  env.json
  bench.jsonl          layer-level microbenchmarks
  gemm.jsonl           isolated single-GEMM benchmarks (roofline GEMM points)
  breakdown.jsonl      time split per phase
  roofline.json        spec + measured ceilings + reference GPU table
  profiler_trace.json  torch.profiler Chrome trace (path also stored in profile_meta.trace_file)
```

**`bench.jsonl` row** (one per (sweep, variant, dispatch, pass, n_tokens, n_experts, cf)):

| field | type / unit | meaning |
|---|---|---|
| `sweep` | str | `"tokens"` (vary N), `"experts"` (vary E at fixed active width), or `"capacity"` (vary cf for switch/gshard memory) |
| `variant` | str | dense, switch, deepseek, gshard, dense_x8 |
| `dispatch` | str | `"loop"` or `"batched"` (dense: `"n/a"`) |
| `pass` | str | `"fwd"` or `"fwd_bwd"` (same spelling as `flops.expert_ffn_cost(pass_=...)`) |
| `n_tokens` | int | N |
| `n_experts`, `n_shared`, `top_k`, `expert_width` | int | layer shape actually built |
| `capacity_factor` | float or null | |
| `label` | str | display label, `"{variant}/{dispatch}/{pass}/N={n_tokens}/E={n_experts}[/cf=..]"` (roofline hover) |
| `median_ms`, `p10_ms`, `p90_ms`, `mean_ms` | ms | over `repeats` timed iterations |
| `flops` | FLOPs | executed, `flops.moe_layer_cost(...)["flops"]` (incl. padding) |
| `flops_useful` | FLOPs | `moe_layer_cost(...)["flops_useful"]` |
| `bytes` | B | `moe_layer_cost(...)["bytes"]` |
| `intensity` | FLOP/B | `flops / bytes` |
| `achieved_tflops` | TFLOP/s | `flops / (median_ms/1e3) / 1e12` |
| `achieved_tflops_useful` | TFLOP/s | same with `flops_useful` |
| `achieved_gbps` | GB/s | `bytes / (median_ms/1e3) / 1e9` |
| `padding_fraction` | 0..1 | from `moe_layer_cost` |
| `expert_counts` | list[int] | the counts fed to `moe_layer_cost` (measured from `MoEAux.expert_counts` in the benchmarked forward) |
| `peak_mem_mb` | MiB | `max_memory_allocated` during the timed loop, after `reset_peak_memory` |
| `mem_baseline_mb` | MiB | allocated before the loop (params + inputs) |
| `gpu_state_before`, `gpu_state_after` | dict or null | `moe.utils.query_gpu_state()` |

**`gemm.jsonl` row** (GEMM-level roofline points: one expert GEMM of the shape it sees in a real layer):

| field | meaning |
|---|---|
| `variant`, `n_tokens`, `n_experts` | the layer config this GEMM came from |
| `kernel` | `"fwd_W_in"` or `"fwd_W_out"` (names from `flops.expert_ffn_cost`) |
| `m`, `k`, `n` | GEMM shape (m = rows per expert) |
| `rows_per_expert` | m |
| `flops`, `bytes`, `intensity` | `flops.gemm_cost(m, k, n)` |
| `median_ms`, `p10_ms`, `p90_ms`, `achieved_tflops` | measured |
| `label` | `"{variant} {kernel} m={m} (N={n_tokens}, E={n_experts})"` |
| `gpu_state_before`, `gpu_state_after` | |

**`breakdown.jsonl` row**

| field | meaning |
|---|---|
| `variant`, `dispatch`, `pass`, `n_tokens`, `n_experts` | config |
| `phase` | one of `"router"`, `"dispatch"` (permute/scatter into buffers), `"expert_gemm"` (incl. GELU), `"shared_expert"`, `"combine"` (gather × gate, index_add), `"other"`. Fixed order = the stacking order. |
| `time_ms` | time attributed to the phase |
| `fraction` | `time_ms / Σ phases` for that config |
| `method` | `"torch.profiler"` (record_function ranges) or `"cuda_events"` |

**`roofline.json`**
```json
{"schema_version": 1, "device_label": "RTX 5070 Ti Laptop GPU", "dtype": "bf16",
 "ceilings": [
   {"kind": "spec", "peak_tflops": <float|null>, "peak_gbps": <float|null>,
    "source": "<document name>", "source_url": "<url>", "notes": "laptop SKU, dense bf16 tensor, no sparsity"},
   {"kind": "measured", "peak_tflops": <float>, "peak_gbps": <float>,
    "method": "bf16 GEMM 8192^3 median of N; D2D copy of X MiB median of N",
    "gemm_shape": [m, k, n], "copy_bytes": <int>, "repeats": <int>,
    "gpu_state_before": {...}, "gpu_state_after": {...}}],
 "reference_gpus": [{"name": "A100 SXM 80GB", "peak_tflops": ..., "peak_gbps": ..., "source": "...", "source_url": "..."}]}
```
- Spec peaks of `null` (laptop value not found with a citable source) are allowed. The dashboard then plots the measured ceiling only and says so.
- `reference_gpus` comes from a cited table in `profile_layer.py` (MASTER §7 E5). The dashboard shows it only in an expander labelled "spec-sheet values (not measured here)".

### 1.7 Routing traces (`route_trace.py`)

```
results/<cfg>/traces/<source_run_id>/
  trace.npz   topk_idx int16 [n_batch, L_moe, N, k]; gates float16 [n_batch, L_moe, N, k]; kept bool [n_batch, L_moe, N, k]
  meta.json   {schema_version, source_run_dir, source_run_id, variant, config_name, seed, n_batches,
               tokens_per_batch (N), moe_layers: [block indices], n_routed, top_k, n_shared, d_model,
               expert_width, capacity_factor, eval_capacity_factor, created_at}
```
`L_moe` indexes `meta.moe_layers`, not raw block ids.

### 1.8 E6 placement (`placement_sim.py`)

```
results/<cfg>/e6_placement/<run_id>/
  placement_params.json
  placement_summary.jsonl
  placement_arrays.npz
  env.json
```

**`placement_params.json`**: `{schema_version, run_id, config_name, created_at, D_values: [2,4,8], policies: ["round_robin","greedy"], bytes_per_element, include_backward_default: true, device_tflops_default, device_tflops_source: "measured roofline <e5 run_dir>" | "config", link_gbps_default, traces: [{trace_dir, variant, source_run_id, n_steps, flops_per_assignment, shared_flops_per_token}], note: "simple analytical model, not a simulator"}`

**`placement_summary.jsonl` row** (one per trace × D × policy × layer, plus `layer: "all"`):

| field | meaning |
|---|---|
| `trace_dir`, `variant`, `D`, `policy`, `layer` | key (`layer` is an int index into `moe_layers`, or `"all"` = Σ over layers) |
| `mean_tokens_per_device` | list[float] (D), mean over steps |
| `straggler_mean`, `straggler_p50`, `straggler_p90`, `straggler_max` | `max_device_load / mean_device_load` over steps |
| `offdevice_tokens_mean` | assignments sent off their home device per step |
| `a2a_bytes_fwd_mean`, `a2a_bytes_fwd_bwd_mean` | bytes per step (MASTER §7 E6 formula) |
| `step_ms_default`, `compute_ms_default`, `comm_ms_default` | `estimate_step_time` at the default device TFLOP/s and link GB/s |

**`placement_arrays.npz`**: keys `f"{source_run_id}|D{D}|{policy}|{name}"` with `name` in:
- `tokens_per_device`: int32 [S, L_moe, D], assignments processed per device per step
- `bytes_out_per_device`: float64 [S, L_moe, D], forward all-to-all bytes each device sends
- `placement`: int16 [L_moe, E], expert → device map

**Functions `placement_sim.py` must export** (the dashboard imports them; it never re-implements the model):
```python
def load_placement_arrays(run_dir: str | Path) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    """{(source_run_id, D, policy): {"tokens_per_device", "bytes_out_per_device", "placement"}}"""

def estimate_step_time(
    tokens_per_device: np.ndarray,      # [S, L, D] assignments per device
    bytes_out_per_device: np.ndarray,   # [S, L, D] forward all-to-all bytes sent per device
    *,
    flops_per_assignment: float,        # FLOPs of one routed (token, expert) assignment, fwd
    device_tflops: float,               # per-device compute, TFLOP/s
    link_gbps: float,                   # per-device link bandwidth, GB/s (1e9 B/s)
    include_backward: bool = True,      # FLOPs x3, bytes x2 (MASTER §7 E6 fwd+bwd)
    shared_flops_per_token: float = 0.0,
    tokens_per_step: int | None = None, # needed when shared_flops_per_token > 0
) -> dict[str, np.ndarray]:
    """Analytical estimate, per step [S]: compute_ms = Σ_L max_D(compute), compute_balanced_ms =
    Σ_L mean_D(compute), comm_ms = Σ_L max_D(bytes_out)/link, step_ms = compute_ms + comm_ms
    (MASTER §7 E6: max(compute) + communication). Also imbalance_ms = compute_ms - compute_balanced_ms."""
```
The exact physics (shared-expert placement, per-link vs aggregate bandwidth) is owned by the `placement_sim.py` author. The dashboard depends only on this signature and the four returned keys `compute_ms`, `compute_balanced_ms`, `imbalance_ms`, `comm_ms`, `step_ms`.

---

## 2. Global layout

### 2.1 Files (amends PLAN §1 dashboard tree)

```
dashboard/
├── app.py          entry: st.set_page_config, theme init, sidebar, st.navigation
├── theme.py        colours, dashes, symbols, Plotly templates, formatters  (NO streamlit import)
├── results_io.py   pure loaders/discovery for results/ (NO streamlit import; make_report uses it)
├── data.py         st.cache_data wrappers around results_io, keyed on file signature
├── layout.py       page_header(), empty_state(), sidebar(), Selection dataclass, run labels
├── figures.py      fig_* builder functions: data in -> go.Figure out (NO streamlit import)
└── pages/          overview.py training.py routing.py capacity.py ablations.py
                    systems.py placement.py how_it_works.py
```
Rule: a page file only does layout and widgets. Every chart is `st.plotly_chart(figures.fig_xxx(...), theme=None, use_container_width=True)`.

### 2.2 Navigation

`app.py`:
```python
pg = st.navigation({
  "Results": [st.Page("pages/overview.py", title="Overview", default=True),
              st.Page("pages/training.py", title="Training"),
              st.Page("pages/routing.py", title="Routing"),
              st.Page("pages/capacity.py", title="Capacity sweep (E2)"),
              st.Page("pages/ablations.py", title="Ablations (E3/E4)")],
  "Systems": [st.Page("pages/systems.py", title="Systems profile (E5)"),
              st.Page("pages/placement.py", title="Expert placement (E6)")],
  "Learn":   [st.Page("pages/how_it_works.py", title="How it works")]})
layout.sidebar()   # renders into st.sidebar, fills st.session_state
pg.run()
```
`st.navigation` disables the automatic `pages/` discovery, so keeping the folder name is safe. Page `url_path`s are the file stems.

### 2.3 Sidebar (`layout.sidebar()`, top to bottom)

| # | Widget | Key in `st.session_state` | Behaviour |
|---|---|---|---|
| 1 | Title "MoE repro" + hardware label | | Label = `env.json["hardware_label"]` of the newest run under the selected config (any experiment). Shown as `Hardware: RTX 5070 Ti Laptop GPU · bf16-autocast (router fp32)`. If no runs: "Hardware: no runs yet". Never hard-code the string. |
| 2 | Config radio `smoke / small / main` | `cfg_name` | Options = subdirectories of `results/` among those three that exist. Default = `main` if it exists, else `small`, else `smoke`. Disabled with a note if none exist. |
| 3 | Run multiselect "E1 runs" | `selected_runs` (list of run dir paths relative to results/) | Options = all run dirs in `<cfg>/e1_main`. Label `"{variant} · s{seed} · {run_id[:15]}"` + `" (incomplete)"`. **Default = for each (variant, seed), the latest complete run.** Used by Overview, Training, Routing and the How-it-works checkpoint list. |
| 4 | Toggle "Aggregate seeds" | `agg_seeds` (default True) | When on and a group has ≥ 2 selected seeds: line = mean, band = min-max, legend `"{variant} (mean of n seeds)"`. With n = 1 it has no effect. |
| 5 | Button "Reload results" | | `st.cache_data.clear()` then `st.rerun()`. |
| 6 | Caption | | `results dir: <abs path>` plus the count of runs found. |

`layout.get_selection() -> Selection(cfg_name, run_dirs, agg_seeds)` is the only way pages read sidebar state.

### 2.4 Shared header (`layout.page_header(title, question, experiment=None)`)

- `st.title(title)`, then a one-line `st.caption(question)` saying what the page answers.
- If `experiment` is given: a caption with the source path (`results/main/e5_profile/<run_id>`), its `created_at`, and the hardware label from that run's env.json.
- If the source run's `env.json["gpu_state"]` shows `power_limit_w` < 50% of the run-to-run max, or `sm_clock_mhz` after < 0.9 × before (E5), add `st.warning("GPU may have been power-capped/throttled during this run: ...")` with the numbers from the file.

### 2.5 Empty state (`layout.empty_state(experiment, cfg_name, what=None)`)

Exact rendering: `st.info(f"No data yet for {what or EXPERIMENT_TITLE[experiment]} ({cfg_name}). Run `{hint}` to produce it.")`, where `hint` comes from:

| experiment | cfg = main | cfg = small | cfg = smoke |
|---|---|---|---|
| e1_main | `make main` | `make small` | `make smoke` |
| e2_capacity | `make sweep` | `.venv/bin/python scripts/sweep_capacity.py --config configs/small.yaml` | `make smoke` |
| e3e4_ablation | `make ablate` | `... scripts/ablate_experts.py --config configs/small.yaml` | `make smoke` |
| e5_profile | `make profile` | `... scripts/profile_layer.py --config configs/small.yaml` | `make smoke` |
| traces | `make trace` | `... scripts/route_trace.py --config configs/small.yaml` | `make smoke` |
| e6_placement | `make placement` (needs traces: `make trace`) | `... scripts/placement_sim.py --config configs/small.yaml` | `make smoke` |

Granularity: the empty state replaces **one chart**, not the page, when only that chart's data is missing. Example: Routing for a dense-only selection shows "Routing logs exist only for MoE variants (switch, deepseek, gshard). Select an MoE run in the sidebar." Pages never raise. Any exception in a figure builder is caught by `layout.safe_chart(fn)`, which shows `st.error("Could not render <chart>: <exc>")` and continues.

### 2.6 Data loading and caching

- `results_io.RESULTS_ROOT = Path(os.environ.get("MOE_RESULTS_DIR", REPO_ROOT / "results"))`.
- Pure functions in `results_io` (no streamlit):
  - `list_configs()`, `list_runs(cfg_name, experiment) -> list[RunInfo]` (RunInfo = run.json fields + `run_dir`, `complete: bool`)
  - `read_json(path)`, `read_jsonl(path) -> pd.DataFrame` (skips bad trailing line, de-dups)
  - `load_run(run_dir) -> RunData` (config, env, final, train/eval/routing DataFrames, each possibly empty)
  - `load_ablation(run_dir)`, `load_profile(run_dir)`, `load_trace(dir)`, `load_placement(run_dir)`
  - `latest(runs, key=...)`
- `data.py` wraps each with `st.cache_data`. **Cache key = (path string, mtime_ns, size)** of every file read. Pattern:
  ```python
  def read_jsonl(path):            # public, cheap: os.stat each rerun
      st_ = os.stat(path); return _read_jsonl_cached(str(path), st_.st_mtime_ns, st_.st_size)
  @st.cache_data(show_spinner=False, max_entries=512)
  def _read_jsonl_cached(path, mtime_ns, size): return results_io.read_jsonl(path)
  ```
  Run discovery is keyed on the tuple of `(name, mtime_ns)` of the experiment directory's entries. A live training run therefore refreshes on each rerun with no TTL. Model objects (How it works) use `st.cache_resource` keyed on `(checkpoint path, mtime_ns)`.
- DataFrames returned from the cache are treated as read-only. Copy before mutating.

---

## 3. Theme (`dashboard/theme.py`)

### 3.1 Variant colours (fixed; colour follows the entity, never its rank)

Okabe-Ito colorblind-safe palette, in this fixed order. Checked with an OKLab + CVD simulation (Viénot/Brettel matrices): worst adjacent-pair ΔE is normal 15.6 (switch/dense_x8), protan 9.6, deutan 7.6 (deepseek/gshard), tritan 8.5. Deutan 7.6 is in the 6-8 floor band, so **a secondary encoding is mandatory**: every variant also has a fixed marker symbol, and all line charts use `mode="lines+markers"` with markers every ~10% of points.

| variant | colour | marker symbol | legend label |
|---|---|---|---|
| dense | `#0072B2` (blue) | `circle` | dense |
| switch | `#D55E00` (vermillion) | `square` | Switch (top-1) |
| deepseek | `#009E73` (bluish green) | `diamond` | DeepSeekMoE (1+31, top-3) — **label built from config**, e.g. `f"DeepSeekMoE ({n_shared}+{n_routed}, top-{k})"` |
| gshard | `#CC79A7` (reddish purple) | `triangle-up` | GShard (top-k) — built from config |
| dense_x8 | `#E69F00` (orange) | `x` | dense ×8 — built from config `width_mult` |

The same hexes are used in light and dark. Contrast on a dark surface (`#1a1a19`) is ≥ 3.36:1 for all five. On light, gshard (2.98:1) and dense_x8 (2.19:1) are below 3:1, so the legend is always shown and those series get direct end-labels on line charts. Text never uses series colour; labels use the template ink.

Other fixed maps:

| map | values |
|---|---|
| `DISPATCH_DASH` | `batched: "solid"`, `loop: "dash"`, `n/a: "solid"` |
| `PASS_FACET` | `fwd`, `fwd_bwd` are facet columns (never colour) |
| `POLICY_DASH` (E6) | `greedy: "solid"`, `round_robin: "dash"` |
| `CEILING_STYLE` (roofline) | spec: ink-muted, `dash="dot"`, width 1.5; measured: ink, `solid`, width 2 |
| `PHASE_COLOURS` (breakdown stack) | single-hue sequential ramp (blues), light→dark in phase order router, dispatch, expert_gemm, shared_expert, combine, other. Stacks are not variant-coloured; variant goes on the x axis. |
| `SEQUENTIAL` | `"Blues"` (load fraction, gate value) |
| `DIVERGING` | load ratio to uniform: blue `#0072B2` → grey `#BBBBBB` at 1.0 → vermillion `#D55E00`, on log2 scale |
| Expert-id colours (How it works only) | `theme.expert_colour(e, E)`: E evenly spaced samples of Plotly's cyclic `"Phase"` scale. **Identity aid only**: the expert id is always also printed in the cell. |

Unknown variant → `#777777`, `circle`, raw name (defensive; must not crash).

### 3.2 Plotly templates

- `theme.TEMPLATE_LIGHT` / `TEMPLATE_DARK`, registered as `pio.templates["moe_light"|"moe_dark"]`, based on `"plotly_white"`:
  - font `"Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"`, size 13; title 15; tick 12
  - `paper_bgcolor`/`plot_bgcolor` transparent (`rgba(0,0,0,0)`), so the chart sits on the Streamlit surface
  - ink: light `#1f1f1f` / dark `#e6e6e6`; muted ink light `#6b6b6b` / dark `#a0a0a0`; grid light `#e8e8e8` / dark `#333333`, width 1; no zero-line emphasis
  - `colorway` = the 5 variant colours in order (fallback only; builders always set colours explicitly)
  - lines width 2, markers size 8 with a 1 px surface-coloured outline, bars `marker_line_width=0`, `bargap=0.25`
  - legend horizontal on top (`y=1.02, x=0, xanchor="left"`), `hovermode="x unified"` for line charts, `"closest"` for scatter/heatmap
  - margins `l=60, r=20, t=50, b=50`
- Theme choice: `theme.current_template()` returns `"moe_dark"` if `st.context.theme.type == "dark"` else `"moe_light"`. This lives in `layout.py`, because theme.py must not import streamlit. Builders take `template: str` as a kwarg (default `"moe_light"`, which the static report uses).
- Always call `st.plotly_chart(fig, theme=None, ...)` so Streamlit does not override the template.

### 3.3 Units and number formatting (`theme.fmt_*`)

| quantity | axis title | axis format | table/tile format |
|---|---|---|---|
| tokens | "tokens seen" | SI `"~s"` (2M) | `12.3M` |
| loss | "val loss (nats)" / "train CE (nats)" | `.2f` | `3.142` (3 dp); deltas signed `+0.012` |
| perplexity | "val perplexity" | `.1f` | `23.14` (2 dp) |
| throughput | "tokens/s" | `"~s"` | `18.0k tok/s` |
| time | "latency (ms)" | auto, log option | 3 significant figures `0.412 ms` |
| memory | "peak memory (MiB)" | `,.0f` | `1,907 MiB` |
| FLOP/s | "achieved TFLOP/s" | `.3~g` | `12.4 TFLOP/s` |
| bandwidth | "GB/s" | `"~s"` | `245 GB/s` |
| bytes | "bytes / step" | `"~s"` + `B` suffix | `1.2 GB` (SI, 1e9) |
| intensity | "arithmetic intensity (FLOP/byte)" | log | `85.3` |
| params | n/a | n/a | exact integers with thousands separators `63,331,200` (tables); `63.3M` (tiles) |
| fractions | "drop fraction (%)" | `.1%` | `4.2%` |

Hover templates always show the run label, the x value with unit, and the y value with unit, using the formats above. Builders use `hovertemplate`, never Plotly's defaults.

---

## 4. Pages (builder implements)

Notation: **src** = file and fields. **Empty** = empty-state behaviour. Everything lives under `results/<cfg_name>/`. "Selected runs" = sidebar selection. Builder function names are given in `code` and live in `figures.py`.

### 4.1 Overview (`pages/overview.py`)

Header question: "What was built, on what hardware, and are the variants really compute-matched?"

| # | Element | Spec |
|---|---|---|
| O1 | Intro text | 2–3 fixed sentences (descriptive, no numbers): small-scale reproduction of Switch Transformer and DeepSeekMoE mechanisms, systems characterization, single GPU. |
| O2 | Environment card | `st.columns(4)` of `st.metric`: hardware label, dtype, torch/CUDA version, VRAM (GB). **src** env.json of the newest selected run. **Empty** "no runs" → e1 empty state. |
| O3 | Param/FLOP table `table_params(cfg, finals)` | Rows = `flops.param_table(cfg, variants)` where `cfg` = config.json of the selected runs (if their `model`/`moe`/`data` sections differ, show one table per distinct config with `st.warning`). If no runs: `cfg = moe.utils.load_config(f"configs/{cfg_name}.yaml")` with caption "analytic, from configs/<cfg_name>.yaml; no measured runs yet". Columns, in order: variant · experts · expert width · active/token · FFN total/layer · FFN active/layer · router/layer · total params (analytic) · **total params (measured)** = `final.json.total_params` (latest complete run of that variant, else "—") · active params · FFN fwd FLOPs/token · model fwd FLOPs/token · **matched** · **measured = analytic**. Check columns (text, not colour alone): `matched`: for MoE variants "yes" iff `ffn_total_model == 8 × dense.ffn_total_model` and `ffn_active_model == dense.ffn_active_model`; for dense_x8 "yes" iff total == 8× (and the note "active = 8× by design"); dense = "baseline". The 8 comes from `cfg["moe"]["dense_x8"]["width_mult"]`, not a literal. `measured = analytic`: "yes" / "NO" / "—". Caption under the table: "Matching is on expert-FFN params/FLOPs; router params are listed separately and excluded (PLAN §5)." Rendered with `st.dataframe(hide_index=True)` and `column_config.NumberColumn(format="%d")` (thousands separators via a pre-formatted string column). |
| O4 | Headline tiles `headline_numbers(finals)` | One `st.metric` row per complete variant group: final val loss (mean over seeds, ± half-range if ≥ 2), delta vs dense (signed, `delta_color="inverse"`), median tok/s, peak MiB. "Best MoE vs dense" tile = min over MoE variants of the val-loss delta, labelled with the variant. **src** final.json `final_val_loss`, `median_tokens_per_sec`, `peak_mem_mb`. **Empty**: per tile "—" with the caption "needs a complete dense run" when dense is missing. |
| O5 | Experiment status table | One row per experiment: experiment · #runs (complete/total) · newest run time · source dir. Data from `list_runs`. Lets the user see what exists. |

### 4.2 Training (`pages/training.py`)

Question: "Do MoE variants reach lower loss than dense for the same tokens, and at what throughput and memory cost?"

Page controls (one row above charts): `log y` toggle (default off), `x axis` radio `tokens | steps` (default tokens), `smoothing` slider for train curves (EMA weight 0.0–0.99, default 0.8; caption "smoothed (EMA); raw shown faint").

| # | Chart | Question | Type | x / y | Group / colour | Hover | src | Empty |
|---|---|---|---|---|---|---|---|---|
| T1 `fig_loss_curves(kind="val")` | Val loss | which variant generalizes best per token? | line+markers | `tokens_seen` (SI) / `val_loss` nats (log toggle) | colour = variant, one line per run, or mean+band if aggregated | run label, tokens, val loss (3 dp), val ppl | eval_log | e1 empty |
| T2 `fig_loss_curves(kind="train")` | Train CE | same, on train batches | line (EMA) + faint raw line (opacity 0.25, no hover) | `tokens_seen` / `ce_loss` | variant | run, tokens, raw and smoothed CE | train_log | e1 empty |
| T3 `fig_aux_curves` | Aux loss | is load balancing doing work / stable? | line | `tokens_seen` / `aux_loss` | variant (MoE only) | run, tokens, aux (4 sig figs) | train_log | "aux loss is 0 for dense variants" if no MoE selected |
| T4 `fig_lr` (expander "LR schedule") | LR actually applied | sanity check of warmup + step decay | line | step / `lr` | variant | step, lr | train_log | hidden |
| T5 `fig_throughput_bars` | Throughput | training speed per variant | bar (vertical) + error bar = min/max over seeds | variant / `median_tokens_per_sec` | variant colour | variant, n seeds, value, `mean_tokens_per_sec` | final.json | e1 empty; incomplete runs excluded with caption |
| T6 `fig_memory_bars` | Peak memory | memory cost of experts | bar | variant / `peak_mem_mb` MiB | variant | variant, MiB, VRAM total from env | final.json | as T5; on CPU runs show "memory not measured on CPU (smoke)" |
| T7 `fig_throughput_over_time` (expander "Throttling check") | Did tok/s drift during the run? | is the comparison confounded by throttling? | line (rolling median of 50 steps) | step / `tokens_per_sec` | variant | run, step, tok/s | train_log | hidden |

Caption under T5: "E1 wall-clock throughput on a laptop GPU; see the Systems page for controlled microbenchmarks." (MASTER PLAN §9.7).

Seed spread (MASTER §8): with `agg_seeds` on, T1/T2 show min-max bands, and a small table under T1 lists per-group `final_val_loss` per seed and the spread (max-min).

### 4.3 Routing (`pages/routing.py`)

Question: "How evenly are tokens spread over experts, and how does that evolve?"

Page controls: `run` selectbox (selected **MoE** runs only, default the first deepseek, else the first MoE run), `layer` selectbox (`sorted(routing_log.layer.unique())`, plus `"mean over layers"` for R3–R5).

| # | Chart | Type | x / y / z | Grouping | Hover | Interactions | src | Empty |
|---|---|---|---|---|---|---|---|---|
| R1 `fig_expert_heatmap` "Expert load over training" | heatmap | x = `step`, y = expert id 0..E-1, z = load (`expert_counts / sum`) | one run, one layer | step, expert, count, fraction, ratio to uniform | radio `z = fraction \| ratio to uniform`. Fraction uses SEQUENTIAL Blues 0..max. Ratio = `count / mean(count)` on the DIVERGING scale, log2, centred at 1, symmetric range from data. | routing_log `expert_counts`, `step`, `layer` | "Select an MoE run" message |
| R2 `fig_final_load_hist` "Final expert load per layer" | bar small multiples (one subplot per layer, shared y) | x = expert id, y = load fraction of the **last** routing record | variant colour | layer, expert, count, fraction, kept count | toggle `sort by load` (default off). Horizontal reference line at `1/E` labelled "uniform" (computed from `n_experts`). | routing_log last step per layer | same |
| R3 `fig_routing_metric(metric="load_imbalance")` | line | step / max÷mean load (≥ 1; reference line at 1) | **all selected MoE runs**, colour = variant | run, step, layer, value | layer selector (mean = mean over layers per step) | routing_log `load_imbalance` | same |
| R4 `fig_routing_metric(metric="router_entropy")` | line | step / entropy (nats); dashed muted line per variant at `max_entropy` labelled "ln E (uniform)" | colour = variant | run, step, entropy, `max_entropy`, normalized = H/ln E | layer selector; toggle `normalize by ln E` | routing_log `router_entropy`, `max_entropy` | same |
| R5 `fig_routing_metric(metric="drop_fraction")` | line | step / drop fraction (%) | switch and gshard runs only | run, step, layer, %, kept/total | layer selector | routing_log `drop_fraction` | "No token dropping in this selection (DeepSeekMoE has no capacity limit)" |
| R6 `fig_routing_metric(metric="load_cv")` (expander) | line | step / CV | colour = variant | | layer selector | routing_log `load_cv` | same |
| R7 `fig_routing_metric(metric="aux_loss")` (expander "Per-layer aux loss") | line | step / layer aux | one line per layer for the chosen run (colour = sequential Blues by layer index; this is the one place layers are coloured) | | | routing_log `aux_loss` | same |

Caption above R1: "Counts are accumulated over each logging window (`window_steps` steps), pre-drop demand."

### 4.4 Capacity sweep, E2 (`pages/capacity.py`)

Question: "How does the Switch capacity factor trade quality against dropped tokens, speed and memory?" (Switch Table 1 / Fig. 3 idea)

**Data:** points from `e2_capacity/sweep_index.json` (fallback: scan). One point per (cf, seed) → `final.json` of that run. With multiple seeds: mean marker + min/max error bars.

| # | Chart | Type | x / y | Hover | src | Empty |
|---|---|---|---|---|---|---|
| C1 `fig_cf_metric("final_val_loss")` | line+markers (switch colour) | cf (linear, ticks at the actual cf values) / val loss nats | cf, val loss, ppl, seeds, run_id | final.json `final_val_loss` | e2 empty |
| C2 `fig_cf_drop` | line+markers, two traces: **val** (solid, `val_drop_fraction`) and **train, last 10%** (dash, `train_drop_fraction_last`). The dash means *split* here, not dispatch; the legend says so. | cf / drop % | cf, both values | final.json | e2 empty |
| C3 `fig_cf_metric("median_tokens_per_sec")` | line+markers | cf / tok/s | cf, median and mean tok/s | final.json | e2 empty; caption: wall-clock, see E5 for controlled timing |
| C4 `fig_cf_metric("peak_mem_mb")` | line+markers | cf / MiB | | final.json | e2 empty; CPU: "not measured on CPU" |
| C5 `fig_cf_curves` (expander) | val loss vs tokens, one line per cf | colour = sequential Blues by cf rank (light = low cf), legend `cf=…` | tokens / val loss | eval_log | e2 empty |
| C6 table | cf · capacity per expert `C` (computed with `layer_api.expert_capacity(tokens_per_step, E, cf)` from config.json) · val loss · Δ vs cf=1.25 if present (else vs the largest cf) · val drop % · tok/s · MiB | | | | |

### 4.5 Ablations, E3/E4 (`pages/ablations.py`)

Question: "How redundant are the experts, and how much does the shared expert matter?" Banner (`st.info`, always shown): "Eval-time ablations on trained checkpoints; not the same as training from scratch (MASTER §7 E4)."

**Data:** latest `e3e4_ablation/<run_id>` (selectbox if several, default latest).

| # | Chart | Type | x / y | Grouping | Hover | Interactions | src | Empty |
|---|---|---|---|---|---|---|---|---|
| A1 `fig_disable_top` "Disable top-routed experts" (DeepSeekMoE Fig. 4 idea) | line+markers | `frac_disabled` (actual fraction, linear) / `val_loss` or `delta_val_loss` | colour = variant, one line per source run (or mean+band over seeds) | variant, n_disable / n_routed, frac, val loss, Δ, ppl | radio `y = absolute \| Δ vs r=0` (default Δ). The `is_reference` points are drawn as hollow markers at x=0 labelled "training cf (with drops)". | ablation.jsonl, `study=="e3_disable_top"` | e3e4 empty |
| A2 `fig_shared_ablation` "Shared expert" | grouped bar | condition (`baseline`, `no_shared_same_k`, `no_shared_plus_one`) / val loss; text labels on bars = Δ vs baseline (signed) | deepseek colour; seeds → error bars | condition, k_routed, active_ffn_width, val loss, Δ | — | `study=="e4_shared"` | "needs a DeepSeekMoE checkpoint" |
| A3 `fig_k_sweep` "k_routed at eval" (DeepSeekMoE Fig. 5 idea) | line+markers | `k_routed` (integer ticks) / val loss; vertical muted line at `top_k_trained` labelled "trained k" | deepseek colour | k, active width, val loss, Δ | — | `study=="e4_k_sweep"` | same |
| A4 table | all rows of the chosen study, sortable | | | | study selectbox | ablation.jsonl | |

Caption under A1 (text, not numbers): "Switch can only disable whole experts (1/8 steps); points sit at the representable fractions (PLAN §7 table)."

### 4.6 Systems, E5 (`pages/systems.py`)

Question: "Where does MoE layer time go on this GPU, and where do the kernels sit on the roofline?"

**Data:** latest `e5_profile/<run_id>` (selectbox if several). Global page controls in one row: `pass` radio (`fwd`, `fwd_bwd`, `both`→facets; default `fwd`), `variants` multiselect (default all present), `dispatch` multiselect (default both).

| # | Chart | Type | x / y | Grouping | Hover | Interactions | src | Empty |
|---|---|---|---|---|---|---|---|---|
| S1 `fig_latency_vs_tokens` | line+markers with p10–p90 error bars | `n_tokens` (log2 axis, ticks at measured N) / `median_ms` (log toggle, default log) | colour = variant, dash = dispatch, facet col = pass | label, N, median/p10/p90 ms, achieved TFLOP/s, padding %, GPU clock before→after | `n_experts` selectbox (for the tokens sweep; default = the config's E for each variant, i.e. rows where `n_experts == spec.n_routed`) | bench.jsonl `sweep=="tokens"` | e5 empty |
| S2 `fig_latency_vs_experts` | line+markers | `n_experts` (log2) / `median_ms` | colour = variant, dash = dispatch | same | `n_tokens` selectbox | `sweep=="experts"` | "no expert-count sweep in this profile run" |
| S3 `fig_speedup_table` | table | variant · N · E · pass · loop ms · batched ms · **speedup** (loop/batched, 2 dp) | | | | bench.jsonl pairs | |
| S4 `fig_breakdown` | stacked bar (vertical), 2 px gap | x = `"{variant}/{dispatch}"` categories / `time_ms` (toggle `ms \| % of total`) | stack = phase (PHASE_COLOURS, fixed order) | phase, ms, % | `n_tokens`, `n_experts` selectboxes | breakdown.jsonl | "no breakdown rows" |
| S5 `fig_achieved_tflops` | line+markers | `n_tokens` (log2) / `achieved_tflops` (+ toggle `useful FLOPs only` → `achieved_tflops_useful`) | colour = variant, dash = dispatch; horizontal ceiling lines: measured peak (ink solid) and spec peak (muted dotted), labelled with source | label, TFLOP/s, % of measured peak | | bench.jsonl + roofline.json | e5 empty |
| S6 `fig_roofline` | log-log scatter + ceiling polylines | x = `intensity` FLOP/B, y = `achieved_tflops` | points: colour = variant, symbol = variant symbol, **filled = batched, open (`-open` symbol suffix) = loop**; ceilings per §3.1 CEILING_STYLE: `y = min(peak_tflops, x·peak_gbps/1000)`, x from 0.1 to 10⁴; ridge point marked and annotated "ridge = P/B FLOP/B" for each ceiling | hover = `label`, intensity, TFLOP/s, % of measured roof at that intensity, bytes, FLOPs | radio `points = layer \| GEMM` (layer = bench.jsonl, GEMM = gemm.jsonl); pass selector; `n_experts` multiselect. Legend groups: variants, then "spec roof (source)" and "measured roof". If a spec ceiling is null: caption "spec-sheet peak unavailable; measured roof only". | bench.jsonl / gemm.jsonl, roofline.json | e5 empty |
| S7 `fig_memory` | grouped bars | x = variant / `peak_mem_mb` at the selected N (`fwd_bwd` preferred) | dispatch = pattern (`marker_pattern_shape="/"` for loop) | MiB, baseline MiB, delta | N selectbox | bench.jsonl | e5 empty |
| S8 `fig_capacity_memory` | line+markers | cf / `peak_mem_mb` and `padding_fraction` (two separate small charts side by side, never dual axis) | switch (and gshard) colour | | | `sweep=="capacity"` | "no capacity sweep rows" |
| S9 GPU-state table (expander "Benchmark conditions") | table: config · SM clock before/after · mem clock · temp · power, flag column "possible throttle" = after SM clock < 0.9 × before | | | | | bench.jsonl gpu_state_* | |
| S10 roofline sources (expander) | `st.json`-like table of `roofline.json.ceilings` (method/source/url) + `reference_gpus` labelled "spec-sheet values (not measured here)" | | | | | roofline.json | |
| S11 trace file | `st.caption(path)` + size + `st.download_button("Download profiler trace (open in chrome://tracing or Perfetto)")` | | | | | profile_meta.trace_file | "no trace saved" |

Caption under S6 (text only): "Arithmetic intensity uses compulsory traffic (each operand read once), so it is an upper bound; see `moe/flops.py`."

### 4.7 Placement, E6 (`pages/placement.py`)

Question: "If experts were spread across D devices, how much would load imbalance and link bandwidth slow a step?" Banner: "Simple analytical model, not a simulator (bridge to WSC-LLM)" + `placement_params.note`.

**Data:** latest `e6_placement/<run_id>`. Page controls row: `trace` selectbox (variant + source run; default deepseek), `D` radio from `D_values`, `layer` selectbox (`all` + indices).

| # | Chart | Type | x / y | Grouping | Hover | Interactions | src | Empty |
|---|---|---|---|---|---|---|---|---|
| P1 `fig_device_load` | grouped bars | device id / mean assignments per device (`mean_tokens_per_device`); horizontal muted line at the mean, labelled "perfect balance" | bar pattern = policy (solid greedy, `/` round-robin), colour = trace variant | device, tokens, % of mean | step slider (`0..S-1`, plus checkbox "mean over steps" default on) reads `tokens_per_device[s, layer, :]` from the npz | summary + npz | e6 empty (also says `make trace` is needed first) |
| P2 `fig_straggler_vs_D` | line+markers with p50–p90 error bars | D (categorical 2/4/8) / `straggler_mean` (≥ 1; ref line at 1) | colour = variant, dash = policy | variant, policy, D, mean/p50/p90/max | `layer` selector | summary | e6 empty |
| P3 `fig_a2a_bytes` | grouped bars | D / `a2a_bytes_fwd_bwd_mean` (toggle fwd only) | colour = variant, pattern = policy | bytes (SI), off-device tokens | | summary | e6 empty |
| P4 `fig_step_time_vs_bw` **live** | line (x log) + vertical marker at the slider value | x = link bandwidth GB/s (log, 50 points from slider min to max) / `step_ms` per step (mean over steps) | colour = variant, dash = policy, at the chosen D | bw, step ms, compute ms, imbalance ms, comm ms | `link_gbps` `st.select_slider` over a log grid (1, 2, 5, … 1000 GB/s; default = `link_gbps_default` snapped to the grid); `device_tflops` `st.number_input` (default `device_tflops_default`, caption with `device_tflops_source`); `include_backward` checkbox. Every interaction recomputes via `placement_sim.estimate_step_time(arrays["tokens_per_device"][:, Ls, :], arrays["bytes_out_per_device"][:, Ls, :], flops_per_assignment=..., device_tflops=..., link_gbps=bw, include_backward=..., shared_flops_per_token=..., tokens_per_step=...)` with `flops_per_assignment` and `shared_flops_per_token` from `placement_params.traces[i]`. | npz + params | e6 empty |
| P5 `fig_step_time_breakdown` **live** | stacked bars at the slider bandwidth | x = `"D{D} {policy}"` / ms; stack = `compute_balanced_ms`, `imbalance_ms`, `comm_ms` (sequential ramp, fixed order) | per selected trace | component ms, % | same widgets as P4 | npz + params | e6 empty |
| P6 note | `st.markdown` short fixed text (no numbers) relating imbalance (compute), all-to-all (communication) and memory to the trade-offs WSC-LLM studies | | | | | | |

Cache `load_placement_arrays` with `st.cache_data` keyed on npz mtime. `estimate_step_time` is cheap and is not cached.

---

## 5. "How it works" page (architect builds; `pages/how_it_works.py`)

Question: "What does the router actually do to my sentence?"

### 5.1 Flow and widgets

1. **Checkpoint selector**: options = complete runs in the current config's `e1_main` and `e2_capacity` that have `checkpoint.pt`. Label `"{variant} · s{seed} · {run_id[:15]}"`. Default = latest deepseek, else latest switch, else first. Device radio `cpu | cuda` (cuda only if available; default cpu, which is enough for one sentence).
2. **Input** `st.text_area("Your text", value="Once upon a time, there was a little girl named Lily.")`, plus a button "Run". Truncated to `cfg["model"]["seq_len"]` tokens, with a caption if truncated.
3. **Options**: Switch/GShard only, checkbox "no-drop eval (capacity = N)", default **off**, so the user sees real capacity behaviour at this tiny N. Caption: "With one short sentence, capacity = floor(k·N/E·cf) is small, so drops are expected; this is real layer behaviour for this batch size." Tick it → `set_eval_overrides(model, eval_capacity_factor=layer_api.NO_DROP)`, untick → `eval_capacity_factor=None`.
4. **Token × layer grid** (`figures.fig_token_layer_grid`).
5. **DeepSeekMoE contribution panel** (deepseek only).
6. **Next-token predictions**.
7. **Step-through diagram** for one (token, layer).

### 5.2 Calls (exact)

```python
cfg   = results_io.read_json(run_dir / "config.json")
model = MoETransformer(cfg); moe.utils.load_checkpoint(run_dir / "checkpoint.pt", model)  # model only
model.eval().to(device)                                   # st.cache_resource keyed (ckpt path, mtime_ns, device)
tok   = moe.data.load_tokenizer(cfg["data"], moe.utils.REPO_ROOT)   # cached resource
ids   = tok.encode(text).ids[: cfg["model"]["seq_len"]]
idx   = torch.tensor([ids], device=device)
with torch.no_grad():
    out = model(idx, return_routing=True)                 # ModelOutput
for i, aux in enumerate(out.layer_aux):                   # one MoEAux per block
    if aux.is_moe:  aux.topk_idx[N,k], aux.gates[N,k], aux.kept[N,k], aux.n_experts
                    aux.extra["router_probs"][N,E]                     (request R1)
                    aux.extra["shared_out_norm"][N], aux.extra["routed_out_norm"][N]   (request R2, deepseek)
                    aux.extra["routed_expert_out_norm"][N,k]           (request R3, optional)
probs = out.logits[0].float().softmax(-1)                 # [T, V]
```
N = T here (B = 1). Token strings come from `tok.id_to_token(i)` with the byte-level `Ġ` prefix shown as a visible space marker `·`.

### 5.3 Visual elements

| Element | Spec |
|---|---|
| Token × layer grid `fig_token_layer_grid(tokens, layer_aux, mode)` | Heatmap. x = token position (tick text = token string), y = block index (layer 0 at top). Cell **text** = top-1 expert id (always shown; `✕` suffix if top-1 was dropped). Cell colour by `mode` radio: **"expert id"** (default; `theme.expert_colour`, identity aid only) or **"gate value"** (SEQUENTIAL Blues, 0..1, of the top-1 gate). Non-MoE blocks (moe_every > 1) are grey with text "dense". Hover: token, position, layer, the top-k list `e12 (g=0.41, kept) · e3 (g=0.22, dropped)`; for deepseek also `shared ‖·‖`, `routed ‖·‖`, `routed/(shared+routed)`. Clicking a cell (`on_select="rerun"`, `selection_mode="points"`) sets the step-through (token, layer). Fallback: two selectboxes. |
| DeepSeek contribution `fig_shared_vs_routed(layer)` | Grouped bars per token: `shared_out_norm` vs `routed_out_norm` for the selected layer (layer selectbox). Colours: two neutral ink tones (not variant colours), legend "shared expert (always on, Eq. 9 first sum)" / "routed experts (gated, Eq. 9 second sum)". If the extra keys are missing: `st.info("Shared/routed norms need MoEAux.extra keys from the layer owner (DASHBOARD_SPEC §5.4)")`. |
| Next-token top-5 | Position selectbox (default = last token). Horizontal bar of the top-5 probabilities from `probs[t]` (single ink colour, values as text), token strings on y. |
| Step-through `render_token_path(layer, t)` | `st.segmented_control` / slider with steps 1–6. A Graphviz DOT diagram (`st.graphviz_chart`) of the path with the current node highlighted, plus a side panel showing for the current step: the paper equation (st.latex), the **code location**, and the live values for this token. |

**Steps** (the code location is resolved at runtime with `inspect.getsourcefile` / `inspect.getsourcelines` of the named object and shown as `moe/router.py:68 Router.forward`; if the object is missing, show `"(not found: update HOW_STEPS)"` instead of crashing):

| # | Step | Object | Equation shown | Live values |
|---|---|---|---|---|
| 1 | Pre-norm | `moe.model.RMSNorm.forward` (called from `moe.model.Block.forward`) | `u = RMSNorm(h)` | norm of h (needs nothing extra; computed from a forward hook on `block.norm2`, registered only on this page) |
| 2 | Router (fp32) | `moe.router.Router.forward` | Switch Eq. 1 `p_i(x) = e^{h_i}/Σ_j e^{h_j}`, `h = W_r·x`; DeepSeekMoE Eq. 11 `s_{i,t} = Softmax_i(u_t^T e_i)` | bar of all E router probs for the token (`extra["router_probs"]`), top-k highlighted |
| 3 | Top-k (+ capacity) | `moe.router.Router.forward` (top-k); switch/gshard capacity: the layer's dispatch function (`moe.moe_switch.SwitchMoE.forward`) | DeepSeekMoE Eq. 10 `g_{i,t} = s_{i,t} if s_{i,t} ∈ TopK(...) else 0`; Switch Eq. 3 capacity | chosen ids, gates, kept flags, capacity C = `layer_api.expert_capacity(N, E, cf, k)` |
| 4 | Experts | `moe.experts.ExpertBank.expert_forward` | `FFN_i(u) = W_out,i · GELU(W_in,i · u)` | per chosen expert: `routed_expert_out_norm` (R3) if present |
| 5 | Weighted sum (+ shared) | `moe.moe_deepseek.DeepSeekMoE.forward` / `moe.moe_switch.SwitchMoE.forward` | Switch Eq. 2 `y = Σ_i p_i(x) E_i(x)`; DeepSeekMoE Eq. 9 `h_t = Σ_shared FFN_i(u_t) + Σ_routed g_{i,t} FFN_i(u_t) + u_t` | shared / routed norms (R2); dropped token → "output = 0 (residual only)" |
| 6 | Residual | `moe.model.Block.forward` | `h ← h + y` | ‖y‖/‖h‖ from the norm2 hook input and the block output (forward hook on `block`) |

`HOW_STEPS` is a module-level list of `(title, "module:qualname", latex)` tuples in `how_it_works.py`. The exact qualnames get confirmed when the layer files land. Dense checkpoints show steps 1, 4 (single FFN: `moe.ffn.DenseFFN.forward`) and 6 only.

### 5.4 Requests to the MoE layer owner (`moe_switch.py`, `moe_gshard.py`, `moe_deepseek.py`)

These are only needed when `return_routing=True`, so training cost is unchanged:

| id | key in `MoEAux.extra` | shape / dtype | layers | meaning |
|---|---|---|---|---|
| R1 | `"router_probs"` | fp32 [N, E] | all MoE | full router softmax (`RouterOutput.probs`), detached, before any disable-top masking |
| R2 | `"shared_out_norm"`, `"routed_out_norm"` | fp32 [N] | deepseek | L2 norm over d of (a) Σ shared-expert outputs, (b) Σ_k gate·expert output, both before the residual add. 0 for the shared term if `disable_shared`. |
| R3 (optional) | `"routed_expert_out_norm"` | fp32 [N, k] | all MoE | L2 norm of `gate × expert_out` per assignment (0 where dropped) |

Contract change this implies: `MoEAux.extra` docstring says "0-dim tensors for logging". It should say "**0-dim tensors are logged to routing_log.extra; tensors with ndim ≥ 1 appear only when `return_routing=True` and are not logged**". `train.py` already filters 0-dim only (§1.4 `extra`). If the layer owner prefers, the How-it-works page can instead compute R2/R3 with forward hooks, but that couples the page to layer internals, so the extras are preferred.

---

## 6. Static report and shared figure builders

- **`dashboard/figures.py` is shared by Streamlit and `scripts/make_report.py`**, so the two show identical figures. Rules:
  - Pure functions `fig_*(..., template="moe_light") -> plotly.graph_objects.Figure`. Inputs are DataFrames/dicts from `results_io` loaders plus explicit option kwargs (the same values the page widgets produce).
  - No `streamlit` import in `figures.py`, `theme.py` or `results_io.py`. CI check: `python -c "import dashboard.figures"` must work with streamlit uninstalled (grep check acceptable).
  - Each builder raises `NoData(msg)` (defined in `results_io`) when its input is empty. Pages turn it into `empty_state`, and the report turns it into a "not available: <msg>" paragraph.
- `make_report.py` (builder) uses `results_io` to discover the latest runs for `--config`. It calls the builders with **default widget values** and writes:
  - `RESULTS.md`: tables and factual text only. Its numbers are computed by the same helpers as O3/O4/C6/S3 (`results_io.summarize_*` functions, shared with the pages).
  - `results/<cfg>/report.html`: one self-contained file. The first figure uses `fig.to_html(full_html=False, include_plotlyjs="inline")` and the rest use `include_plotlyjs=False`. Plotly JS is embedded once, so it works offline.
- "How it works" is interactive and is **not** in the static report (one static token grid for a fixed example sentence may be added later by the architect).
- Summary helpers to put in `results_io` (shared by pages and report): `summarize_e1(finals) -> DataFrame` (O4), `summarize_e2(points)`, `summarize_e3e4(rows)`, `summarize_e5_speedups(bench)`, `summarize_e6(summary)`, `param_table_with_checks(cfg, finals)` (O3).

---

## 7. No hard-coded numbers rule

1. Every displayed **result** number comes from a file in §1 or from `moe.flops` / `moe.layer_api` evaluated on a config (config.json of a run, or `configs/<name>.yaml` when explicitly labelled "analytic").
2. Derived values (deltas, ratios, means over seeds, ln E, 1/E, ridge point, speedups, capacity C) are computed on the fly from those numbers. They are never typed in.
3. Spec-sheet hardware peaks appear only via `roofline.json` with their `source`, labelled "spec sheet". The GPU name comes only from `env.json`.
4. **Paper reference values are left out of the dashboard.** If ever added, they must sit in a separate, clearly captioned element "paper (not measured here)" with the citation, and never on the same axes as measured data.
5. Allowed literals: colours, fonts, layout sizes, UI defaults (slider grids, default prompt text, EMA default), the `results/` layout names, and units. Builders must not contain any literal that describes a result (e.g. no "~37 TFLOP/s", no "8×" in a label; use `width_mult`/config values).
6. The architect's review (MASTER §1.5.2) greps `dashboard/` and `scripts/make_report.py` for numeric literals in strings as part of the check.

---

## 8. Builder checklist (per page)

1. Read data only through `dashboard.data` (cached) → `results_io`.
2. Put each chart in `figures.py` with the name given above, taking `template=`.
3. Use `theme.VARIANT_COLOUR/SYMBOL/label()`, `DISPATCH_DASH`, `POLICY_DASH` and `fmt_*`. No inline hexes in pages.
4. Wrap each chart in `layout.safe_chart`. Missing data → `layout.empty_state(experiment, cfg_name, what)`.
5. Smoke test: with `MOE_RESULTS_DIR` pointing at an empty dir, every page renders with empty states and no exceptions. After `make smoke`, every page renders with real data.
6. Do not modify `results_io`, `theme`, `layout` or `figures.py` helpers owned by the architect without asking. Add new `fig_*` functions freely.
