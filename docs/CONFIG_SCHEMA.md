# CONFIG_SCHEMA.md — single source of truth for config keys

Written by the orchestrator from `docs/PLAN.md` §3–§5 so that parallel agents use identical key names.
Changing a key requires updating this file and every consumer. YAML files live in `configs/`; the
resolved config (after `--override a.b=c`) is saved as `config.json` in each run dir.

Variants (string, passed as `--variant` to `scripts/train.py`, stored as top-level `variant`):
`dense`, `switch`, `deepseek`, `gshard`, `dense_x8`.

```yaml
name: main                 # smoke | small | main
seed: 1337
variant: dense             # filled in by train.py from --variant

data:
  dataset: tinystories     # tinystories | shakespeare (fallback used automatically if download fails)
  vocab_size: 8192
  cache_dir: data_cache    # relative to repo root; gitignored
  tokenizer_train_stories: 200000   # stories used to train BPE (subset)
  max_train_tokens: 60000000        # cap on cached train tokens (~2.4x main budget; no repeats in main)
  max_val_tokens: 2000000           # cap on cached val tokens (from TinyStories validation split)

model:
  n_layers: 6
  d_model: 384
  n_heads: 6
  seq_len: 256
  d_ff: 1536               # base dense FFN width (4 x d_model)
  moe_every: 1             # 1 = every FFN is the variant's FFN; 2 = every other layer (Switch paper), others dense
  init: switch             # switch: truncnormal(+-2 sigma), sigma = sqrt(init_scale / fan_in) | deepseek: std 0.006
  init_scale: 0.1          # Switch Sec. 2.4 reduced init scale s
  norm_eps: 1.0e-6
  rope_base: 10000

moe:
  dispatch: batched        # loop | batched  (identical math; loop is the readable reference)
  renormalize_gates: false # both papers use raw softmax prob as gate; flag kept for experiments
  switch:
    n_experts: 8
    capacity_factor: 1.25
    eval_capacity_factor: null   # null = same as capacity_factor
    aux_alpha: 0.01
    jitter_eps: 0.01             # Switch App. C, train only, multiplies router input
  gshard:
    n_experts: 16                # user-approved: 16 x d_ff/2, top-2 (8x total, 1x active)
    width_divisor: 2             # expert width = d_ff / width_divisor
    top_k: 2
    capacity_factor: 1.25
    eval_capacity_factor: null
    aux_alpha: 0.01
    jitter_eps: 0.0
  deepseek:
    m: 4                         # fine-grained segmentation factor: expert width = d_ff / m
    n_shared: 1
    n_routed: 31
    k_routed: 3
    aux_alpha: 0.01              # expert-level balance alpha1 (Eqs. 12-14)
    device_aux_alpha: 0.0        # device-level balance alpha2 (Eqs. 15-17), default off
    n_devices: 1                 # only used when device_aux_alpha > 0
    jitter_eps: 0.0
  dense_x8:
    width_mult: 8                # FFN width = width_mult * d_ff

train:
  batch_size: 32               # sequences per step; tokens/step = batch_size * seq_len
  total_steps: 3000
  warmup_steps: 200
  peak_lr: 1.0e-3
  lr_decay_points: [0.8, 0.9]  # fractions of total_steps
  lr_decay_factor: 0.316
  betas: [0.9, 0.95]
  weight_decay: 0.1            # matrices only (not norms, embedding, router)
  grad_clip: 1.0
  eval_interval: 150
  eval_batches: 20
  routing_log_interval: 50
  checkpoint_interval: 500     # 0 = only final checkpoint
  log_interval: 1
  compile: false

system:
  device: auto                 # auto | cuda | mps | cpu   (auto: cuda -> mps -> cpu)
  dtype: auto                  # auto | bf16 | fp32       (auto: bf16 autocast if cuda supports it, else fp32)
  cpu_threads: 8
  results_dir: results
```

Per-config values (from PLAN.md §3):

| key | smoke | small | main |
|---|---|---|---|
| model.n_layers / d_model / n_heads | 2 / 64 / 2 | 4 / 256 / 4 | 6 / 384 / 6 |
| model.seq_len / d_ff | 64 / 256 | 256 / 1024 | 256 / 1536 |
| train.batch_size | 16 | 32 | 32 |
| train.total_steps / warmup_steps | 150 / 15 | 1500 / 100 | 6000 / 200 (was 3000; user-approved 2026-10-05) |
| train.eval_interval / eval_batches | 50 / 4 | 150 / 10 | 150 / 20 |
| train.routing_log_interval | 10 | 50 | 50 |
| train.checkpoint_interval | 0 | 500 | 500 |
| train.peak_lr | 3.0e-3 | 1.5e-3 | 2.0e-3 (was 1e-3; LR check 2026-10-05, see PROGRESS) |
| system.device / dtype | cpu / fp32 | auto / auto | auto / auto |
| data.max_train_tokens | 2000000 | 30000000 | 60000000 |
| data.max_val_tokens | 200000 | 2000000 | 2000000 |

All three configs share one tokenizer (same `vocab_size`, same `tokenizer_train_stories`), so
smoke must not change tokenizer-affecting keys. Token caches may be keyed by max_*_tokens or simply
cache the largest and slice — data.py decides, but must document it.

## Notes added after T3 (architect)

- **Top-k capacity:** `C = min(N, floor(k * N / E * capacity_factor))` (GShard Alg. 1). For Switch (k=1) this is exactly Switch Eq. 3. Implemented in `moe.layer_api.expert_capacity`.
- **DeepSeek init std** is not a config key. It is hard-coded as `moe.layer_api.DEEPSEEK_INIT_STD = 0.006` (DeepSeekMoE paper), used when `model.init: deepseek`.
- **Ablation switches are runtime attributes, not config keys.** `disable_shared`, `n_disable_top_routed` / `disable_top_routed_ratio` and `k_routed_override` are set through `moe.layer_api.set_eval_overrides(model, ...)`. Only `eval_capacity_factor` also has a config default (`moe.switch` / `moe.gshard`). `layer_api.NO_DROP` (inf) means capacity = N.
