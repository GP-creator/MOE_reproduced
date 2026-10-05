# PROGRESS.md — shared handoff log

Every agent appends an entry when it finishes: date, agent, what was done, files touched, decisions, and notes for the next agent. Newest entries at the bottom.

## Environment facts (verified 2026-10-04)

- Repo root: `~/moe-repro` (WSL2 Ubuntu, kernel 6.18 WSL2)
- Python: **3.14.4** (system), venv at `.venv/` (created with `python3 -m venv --without-pip` + get-pip.py, because `python3.14-venv`/ensurepip is not installed and there is no passwordless sudo). Always use `.venv/bin/python`.
- GPU: **RTX 5070 Ti Laptop GPU**, 12 GB (12227 MiB reported by nvidia-smi), compute capability (12, 0), driver 616.92 (Windows side), 120 W power cap reported, max SM clock 3090 MHz.
- torch **2.11.0+cu128**, CUDA 12.8, arch list includes `sm_120`, bf16 supported.
- Rough bf16 4096³ GEMM sanity: ~37 TFLOP/s (single quick measurement, NOT a benchmark; E5 does it properly).
- WSL sees ~15 GB system RAM (WSL memory limit), not the full 32 GB. Keep dataloader/tokenization memory modest.
- Pinned packages: see `requirements.txt`. `kaleido` is installed but not required (static report embeds Plotly JS instead of rendering images).
- Hugging Face reachable (TinyStories API returned 200).

## Log

### 2026-10-04 — main session (orchestrator)
- Milestone 0 complete: nvidia-smi OK, venv created, torch cu128 installed, CUDA + sm_120 + bf16 matmul verified.
- Created `.claude/agents/moe-architect.md`, `.claude/agents/moe-builder.md`, `MASTER_PROMPT.md`, `PROGRESS.md`, `requirements.txt`, `.gitignore`.
- Note: the main session was launched from a different directory, so the project subagents in `.claude/agents/` are not auto-registered in this session. Delegation is done with the general-purpose agent, using a model override (opus/sonnet) and pasting the agent's system prompt. Future sessions launched from `~/moe-repro` will pick up the named agents directly.

### 2026-10-04 — moe-architect (Milestone 1: plan)
- Wrote `docs/PLAN.md` (file tree, backbone choices, smoke/small/main configs, LR + schedule, exact param/FLOP table, measured probe + run-time estimates, MoE design decisions, Milestone 2 delegation, risks). **Waiting for user approval; no project code written.**
- Files touched: `docs/PLAN.md` (new), `PROGRESS.md`. Throwaway scripts (`params.py`, `probe.py`, `runtime.py`, `prof.py`, `bw.py`, `sustain.py`) live in the session scratchpad, NOT the repo.
- Decisions: RMSNorm, RoPE, tied emb, **no biases anywhere** (attn/FFN/router); experts stored as stacked `W_in[E,d,w]`/`W_out[E,w,d]` shared by loop+batched paths; main = 6/384/6, seq 256, d_ff 1536, 32×256=8192 tok/step, 3000 steps (24.58M tok), warmup 200, peak LR 1e-3, step decay ×0.316 at 80%/90%; Switch batched = sort-and-segment into capacity-padded `[E,C,d]` + bmm (literal `[T,E,C]` einsum only as a tiny-T test reference); router fully fp32 incl. jitter + aux; aux summed over layers (each includes its α); E3 counts: DeepSeek n_dis {0,2,4,6,8}/31, Switch/GShard(8) {0,1,2}/8, no-drop eval.
- **Must know:** during the probe the GPU was power-capped at **33 W** (SW Power Cap active, P5, memory clock 810 MHz under load, D2D copy 25.8 GB/s, GEMM 19.7 TFLOP/s). All throughput numbers in PLAN.md are from that throttled state. Re-probe after the user sets Turbo before fixing the main budget.
- Open questions for user: GShard as written is only 4× total params (recommend 16×d_ff/2 top-2); dense_x8 ~58 min at 33 W; optional 3-point LR check on small.

### 2026-10-04 — main session (orchestrator)
- User approved PLAN.md. Resolutions recorded at top of `docs/PLAN.md`: GShard = 16 × d_ff/2 top-2; dense_x8 optional (only if under budget after Turbo re-probe); LR check deferred.
- Starting Milestone 2, Wave 1: T1 (builder, data.py), T2 (builder, utils/configs/Makefile), T3 (architect, layer_api/schedule/flops) in parallel.

### 2026-10-04 — moe-builder (T2: utils/configs/Makefile) [pasted by orchestrator]
- Done: `moe/utils.py`, `configs/{smoke,small,main}.yaml`, `Makefile`, `moe/__init__.py`, `scripts/__init__.py`.
- Acceptance passed: overrides + bad-key/bad-variant errors; cuda+bf16 (main) / cpu+None (smoke); env info with hardware_label "RTX 5070 Ti Laptop GPU"; checkpoint + RNG round trip; `make help` / `make -n smoke`.
- Notes: `utils` exports `REPO_ROOT`, `VARIANTS`. `find_latest_checkpoint` prefers `checkpoint.pt`, else highest `ckpt_step*.pt`; periodic saves prune older ones. `query_gpu_state` keys: sm_clock_mhz, mem_clock_mhz, temperature_c, power_draw_w, power_limit_w (power_limit is N/A → None under WSL). `clean-results` preserves `results/sample`. Scripts must add repo root to `sys.path`. `make smoke` passes `--config configs/smoke.yaml` to every script.

### 2026-10-04 — moe-builder (T1: data.py) [pasted by orchestrator]
- Done: `moe/data.py`. TinyStories streamed via HF datasets (no fallback needed; Shakespeare fallback implemented but untested). Shared byte-level BPE vocab 8192, EOS `<|endoftext|>`, trained on 200k stories.
- Cache `data_cache/` (gitignored): train.bin 60M tok, val.bin 2M tok, tokenizer.json, meta.json, 119 MB total. Full prep 63 s, peak RSS 1.8 GB; cached rerun ~1 s.
- Cache policy: holds largest cap requested; smaller configs slice via `TokenBatcher(max_tokens=...)`; larger cap triggers rebuild unless source exhausted. Dataset choice persists in meta.json (delete data_cache to retry).
- `TokenBatcher` is stateless: windows of seq_len+1, epoch e permutation = `default_rng([seed, e])`, batch `step` = slots [step*bs,(step+1)*bs) → deterministic in (seed, step), resume-safe. Eval batches = first N non-overlapping windows, seed-independent. Main train file = 233,463 windows of 257.
- Callers need repo root on sys.path (`PYTHONPATH=.`) to `import moe`.

### 2026-10-04 — moe-architect (T3: layer API, schedule, flops) [pasted by orchestrator]
- Added `moe/layer_api.py` (contract docstring, `MoEAux`, `FFNSpec`/`ffn_spec`, `is_moe_layer`, `expert_capacity`, `init_weight_`, `mark_no_weight_decay`/`build_param_groups`, `build_ffn` factory with lazy imports, `set_eval_overrides`), `moe/schedule.py` (`lr_at_step`, `decay_boundaries`, `set_lr`), `moe/flops.py` (`param_row`/`param_table`/`format_param_table`; E5 helpers `gemm_cost`, `expert_ffn_cost`, `moe_layer_cost`).
- Acceptance: flops tables match PLAN §5 exactly for main/smoke/small (gshard now 16×768: main 63,349,632 total / 13,804,416 active, router 6,144/layer, fwd FLOPs/token 29,958,144). LR schedule checked at boundaries.
- Decisions: top-k capacity `C = min(N, floor(k·N/E·cf))` (GShard Alg. 1; = Switch Eq. 3 for k=1). E3 no-drop eval via `eval_capacity_factor = NO_DROP` (inf). `moe_every` puts experts on the last block of each group. `ffn_spec` asserts activated width == d_ff. LR boundaries use `round(p·S)`. flops.py byte model counts compulsory traffic (each operand read once), so arithmetic intensity is an upper bound; options `include_autocast_weight_cast`, `hidden_traffic` ("unfused" default); backward = 3× forward FLOPs (total), per GEMM.
- For T4: MoE classes set `ABLATION_ATTRS`, expose `spec`/`n_routed`/`top_k`/`width`, use `ffn_spec`/`expert_capacity`/`init_weight_`. DeepSeek batched pads by up to 1 row for balanced loads (3N/31 not integer).
- For T5: `model.py` calls only `layer_api.build_ffn(variant, layer_idx, cfg)`; FFN returns (y, MoEAux), no norm/residual inside; must call `mark_no_weight_decay(tok_emb.weight)`; optimizer uses `build_param_groups(model, wd)`; startup table via `flops.format_param_table(flops.param_table(cfg))`.
- Orchestrator fixed the stale gshard numbers in PLAN.md and added T3 notes to CONFIG_SCHEMA.md.

### 2026-10-04 — main session (orchestrator)
- Wave 1 done (T1, T2, T3). Starting Wave 2: T4 (architect: router, Switch/GShard/DeepSeek layers, metrics) ‖ T5 (builder: ffn.py, model.py).

### 2026-10-04 — moe-builder (T5: ffn.py, model.py) [pasted by orchestrator]
- Done: `moe/ffn.py` (`DenseFFN`, W_in[d,w]/W_out[w,d] nn.Parameter, GELU, no biases), `moe/model.py` (`RMSNorm` fp32-normalized, `RotaryEmbedding` rotate-half w/ non-persistent buffers, `CausalSelfAttention` wq/wk/wv/wo no bias + SDPA causal, `Block`, `ModelOutput`, `MoETransformer`).
- Acceptance: `verify_param_count` matches flops.py for dense/dense_x8 on smoke+main (main 13,767,552 / 63,312,768); init CE ≈ ln V; no missing grads, tied emb gets grad; emb+norms in no-decay; CUDA bf16 logits bf16, ce fp32; main dense bs32×256 peak 1.89 GB; 30-step smoke on learnable synthetic pattern 9.02 → 2.66; switch build fails only on missing module.
- Notes: model imports only layer_api + flops. `loss = out.ce_loss + out.aux_loss` in train.py. `configure_optimizer` sets lr=peak_lr; train.py must call `schedule.set_lr` every step. RoPE buffers sized to seq_len (longer seqs unsupported). `generate(temperature=0)` = argmax. `moe_every=2` path untested until MoE modules exist.

### 2026-10-04 — moe-builder (T7: model + determinism tests) [pasted by orchestrator]
- Done: `tests/test_model.py` (all 5 variants: param count, shapes, CE≈lnV, aux dtype/sign, is_moe per layer, finite grads; return_routing topk shape; moe_every=2; loop vs batched agree under NO_DROP) and `tests/test_determinism.py` (dense/switch/deepseek: same seed identical, diff seed differs, resume == uninterrupted; batch_size 4).
- Result: 16 passed, 2 failed (~4 s). OPEN → escalated to moe-architect: DeepSeek batched dispatch is not bit-deterministic on multi-threaded CPU (losses differ ~1e-6 from step 3; deterministic with 1 thread or loop dispatch). Suspected cause: parallel `index_add` combine in `moe_deepseek._batched` with k=3 assignments per token row. GShard (k=2) not covered by determinism tests, may share the issue. Builder did not patch or pin threads.

### 2026-10-04 — moe-architect (DASHBOARD_SPEC) [pasted by orchestrator]
- Wrote `docs/DASHBOARD_SPEC.md`: results-file contract for all experiments (§1, single source of truth), sidebar/navigation/empty-state/caching, theme (Okabe-Ito + marker symbols, dash maps, Plotly light/dark templates, number formats), per-chart tables for all 8 pages, How-it-works design, static-report plan, no-hard-coded-numbers rule.
- Decisions: results under `results/<config_name>/<experiment>/<run_id>/` via `make_run_dir(cfg, f"{cfg['name']}/<experiment>", run_id)`; new `run.json` per run (group key for seed aggregation); run complete iff `final.json` exists; traces at `results/<cfg>/traces/<run_id>/`; dashboard tree adds `results_io.py` + `figures.py` (no streamlit, shared with make_report.py). `placement_sim.py` must export `load_placement_arrays` and `estimate_step_time(...)` (signature in spec).
- Requests to layer owner (forwarded by orchestrator): R1 `router_probs` [N,E] all MoE; R2 `shared_out_norm`/`routed_out_norm` [N] DeepSeek; R3 optional `routed_expert_out_norm` [N,k]; only when return_routing=True.
- Orchestrator updated PLAN §1/§7 paths.

### 2026-10-04 — moe-architect (T4 MoE layers + T6 tests) [pasted by orchestrator]
- New: `moe/router.py`, `moe/experts.py` (ExpertBank: stacked W_in[E,d,w]/W_out[E,w,d]; `expert_forward` loop path, `batched_forward` 2×bmm), `moe/moe_switch.py` (`CapacityMoE` shared top-k+capacity, `SwitchMoE` k=1, `assign_with_capacity` Fig.15 cumsum, `einsum_reference` Fig.16 test-only), `moe/moe_gshard.py` (k=2, choice-major so 1st choices fill capacity first; deterministic 2nd choice, raw-prob gates — documented), `moe/moe_deepseek.py` (shared ungated, softmax over routed only, no drop; batched uses only permutation/unique-index scatters + fixed-order k-sum → deterministic), `moe/metrics.py` (`RoutingAccumulator.update(out.layer_aux)` on-device, `.flush(step)` one sync, records exactly DASHBOARD_SPEC §1.4). Edited `layer_api.py` (MoEAux.top_k, .n_tokens; extra docstring; R1–R3 extras with return_routing=True).
- Tests: conftest + 8 files. Fixed builder's DeepSeek determinism escalation. Explicit fixed-order k-sum also avoids CUDA autocast promoting torch.sum to fp32.
- Must know: `return_routing=True` only for traces/How-it-works (~4 MB/layer at 8192 tok). Use `dispatch=batched` for training (loop syncs per expert). `layer.dispatch` is mutable. At init, cf 1.25 drop fraction is 0 (imbalance ~1.2); drops appear once routing skews. GPU memory clock flips 14 GHz ↔ 810 MHz (~7× timing change) — Turbo re-probe needed before Milestone 4.

### 2026-10-04 — main session (orchestrator) — MILESTONE 2 COMPLETE
- `make test`: 132 passed in 5.8 s (includes bf16 CUDA tests).
- Review gate (MASTER 1.5.3) will be done once, after train.py + runners exist (Milestone 3), before any main run.
- Milestone 3 parallel wave: B-A train.py + run_main + sweep_capacity; B-B ablate_experts + route_trace; A-A profile_layer (E5) + placement_sim (E6); A-B dashboard shared layer + How-it-works page.

### 2026-10-04 — moe-builder (B-B: ablate_experts.py + route_trace.py) [pasted by orchestrator]
- Done: `scripts/ablate_experts.py` (E3+E4 eval-only), `scripts/route_trace.py` (traces + importable `trace_text(model, tokenizer, text, *, max_tokens=None, autocast_dtype=None) -> dict`), shared helper `scripts/_eval_common.py` (run discovery, model load from run's config.json+checkpoint.pt, fixed val batches, results root). Follows DASHBOARD_SPEC §1.5/§1.7. Both accept `--runs`, `--variants`, `--results-dir`.
- E3: PLAN §7 counts with NO_DROP + `is_reference` row at training cf (Switch/GShard). E4: baseline, (a) no_shared_same_k, (b) no_shared_plus_one, k sweep {1,2,3} + k=4 labelled over-activation. Every row has `note: "eval-time change, not retrained"`.
- Traces: `results/<cfg>/traces/<source_run_id>/{trace.npz,meta.json}`, topk_idx int16 / gates f16 / kept bool, [n_batch, L_moe, N, k].
- Verified only on throwaway 60-step smoke checkpoints in scratchpad; CPU only. OPEN for review gate: on the fake DeepSeek ckpt, E3 loss jumps +0.25 at n_dis=2 then plateaus (likely undertraining; re-check on real checkpoints, and if it persists the architect should inspect masking).

### 2026-10-04 — moe-builder (Milestone 3 B-A: train.py, run_main.py, sweep_capacity.py) [pasted by orchestrator]
- Done: scripts/train.py (CLI + importable train(cfg, experiment, run_id, resume_dir, tag, max_steps, overrides, verbose)), scripts/run_main.py (E1), scripts/sweep_capacity.py (E2), scripts/_common.py (sys.path, find_runs, run_group, sanitize NaN->null, jsonl dedupe). Follows DASHBOARD_SPEC §1.4 (run/config/env/train_log/eval_log/routing_log/final.json, sweep_index.json).
- Acceptance: smoke switch run produces every spec file; resume (stop 60 → 150) continuous steps + bit-identical final val loss (5.039591550827026) on CPU; run_main smoke (5 variants) 50 s; sweep_capacity smoke 31 s (cf1.25 reused via sweep_index); GPU 50-step deepseek main: median 65k tok/s (75–80k once clocks ramp), ~125 ms/step, peak 5638 MiB; make test 132 passed.
- Notes: one host sync/step. --max-steps stops write ckpt but no final.json (incomplete/resumable). --tag extends run_id+group. skip/reuse require group+seed+identical config.json. final.json drop fractions null for dense/dense_x8/deepseek (spec); routing log keeps 0.0 for deepseek. Resume from non-window boundary loses partial routing window.
- FOR REVIEW GATE: smoke switch val_loss 5.04 vs dense 4.22 / deepseek 4.42 / gshard 4.66 (150 steps, lr 3e-3); switch val drop 0.148 at init, 0.52 at step 50, 0.083 at end; load imbalance ~1.9. Capacity sweep monotone (val drop 0.39/0.28/0.083/0.013 for cf 0.75/1/1.25/2; val loss 5.32/5.08/5.04/4.91).
- Orchestrator note: GPU throughput at main size (65–80k tok/s deepseek) is ~5–6× the 33 W probe estimate → main runs ≈ 6 min/variant, far under the 45-min budget. Revisit main config size with user before Milestone 4.

### 2026-10-04 — moe-architect (Milestone 3 A-B: dashboard shared layer + How it works) [pasted by orchestrator]
- New: dashboard/{__init__,app,theme,results_io,data,layout,figures,inference}.py, dashboard/pages/{overview,training,routing,capacity,ablations,systems,placement}.py (STUBS; training stub renders fig_loss_curves example), dashboard/pages/how_it_works.py (full, spec §5).
- How-it-works uses scripts.route_trace.trace_text + scripts._eval_common.load_run_model; dashboard/inference.py adds forward hooks for norms, capacity C, inspect-based code locations from live classes.
- Acceptance: streamlit serves (HTTP 200); AppTest of app.py + all 8 pages: 0 exceptions with empty MOE_RESULTS_DIR and with real results/smoke. theme/results_io/figures/inference import without streamlit.
- Deviations: data.py holds cache wrappers; width="stretch"; safe_chart(name, build, *, experiment, cfg_name, key); page_header(..., run_dir=None); results_root() re-reads MOE_RESULTS_DIR; How-it-works lists all complete e1/e2 checkpoints; theme.DIAGRAM.
- For builder: follow figures.py docstring + spec §8; summarize_* go in builder-owned section at bottom of results_io.py; never call st.plotly_chart directly.
- Note: sanity profile dirs under results/main and results/small make sidebar default to `main` (E1 empty) — orchestrator to clean before smoke check.

### 2026-10-04 — moe-architect (NOTES.md study guide) [pasted by orchestrator]
- Wrote `NOTES.md` (711 lines): 10 concept sections (each: plain explanation, paper equations with numbers, exact `file::symbol` refs, gotchas / code-vs-paper differences), code map, loop vs batched, roofline/AI (flops.gemm_cost), matched-compute setup, "What surprised me" PLACEHOLDER, 11 lab-meeting questions. No result numbers.
- No line numbers (files were being edited in parallel); refs are function names + exact quoted code lines. Verifier (scratchpad/verify_notes.py): 174 checks, 0 failures; only scripts/placement_sim.py pending (marked planned).
- TODO later: fill "What surprised me" from results only; update §10 once placement_sim.py exists and §12 once profile_layer.py is final; re-run verifier after renames.

### 2026-10-04 — main session (orchestrator) — PAUSED (usage limit reached)
Three agents were stopped mid-task. Their output may be partial; re-check each before trusting it:
1. **moe-builder: 7 dashboard pages + make_report.py**: stopped early (was reading data/flops API). Pages are probably still the architect's stubs; `scripts/make_report.py` probably doesn't exist. Re-run this task in full (brief: same as the launch prompt; rules in the dashboard shared-layer entry above).
2. **moe-architect: review gate (T8)**: stopped before reporting. It may have left `results/{smoke,small}/debug_review/` (delete them) and may have edited architect-owned moe/ files (run `make test`). Re-run the review: Part A loss-combining review, Part B smoke Switch diagnosis, Part C DeepSeek E3 masking vs the paper's Fig. 4 protocol. **Blocker before any main run.**
3. **moe-architect: E5 profile_layer.py + E6 placement_sim.py**: stopped while starting E6. `scripts/profile_layer.py` exists (possibly complete; sanity outputs in results/main/e5_profile, results/small/..., results/smoke/e5_profile). `scripts/placement_sim.py` is probably missing or partial. It may have added `moe/profiling.py` and phase hooks in moe_*.py; run `make test`.
Then: clean sanity/debug result dirs → `make smoke` from scratch → dashboard + report check → Milestone 3 report. Before Milestone 4: Turbo re-probe, revisit main size (GPU now ~65–80k tok/s → ~6 min/variant), LR-check decision.

### 2026-10-04 — moe-architect (Review gate MASTER 1.5.3 + Switch/E3 diagnoses) [pasted by orchestrator]
- REVIEW PASS, no blockers. Checked: loss = ce + Σaux train-only; CE fp32; autocast scope + fp32 router; eval mode/jitter/eval_cf; WD groups; clip/LR order; RoutingAccumulator; fixed eval batches; ckpt/resume incl. optimizer+RNG; E2 reuse; E3/E4 overrides + NO_DROP; trace dtypes; equal token budget/data order.
- Builder follow-ups: should-fix train.py:292 skip optimizer.step on non-finite grad; nits ablate_experts.py:53 (exp overflow), :62 (reset eval_cf to config value), :141 ("shared" column for switch/gshard), :181 (per-run val meta); train.py:190 (val max_tokens). Info: E3 uses 50 val batches on main vs E1's 20 → E3 baseline ≠ E1 final val loss; say so in dashboard.
- Switch smoke gap = expected small-scale behaviour, not a bug (exact 1-expert Switch == DenseFFN check passes). Near-uniform router + step-shared argmax during unigram plateau → per-step drops 0.3–0.5 (10-step routing windows average this to imbalance ~1.9). cf=8 recovers 0.42 of the 0.82 gap; rest = undertraining. No config change recommended (would be outcome-driven).
- small config GPU (1500 steps, 12.3M tok, 1 seed): val dense 1.956, switch 2.001, gshard 1.954, deepseek 1.886; switch val drop 0.8%. Throughput ~408k (dense, contended run—ignore its final.json mean) / 221k / 231k / 204k tok/s; MoE ≈ 1.8–2× dense step time; ~1–1.5 min/variant.
- E3 plateau: masking verified exactly (ranks m..m+k−1, gates = original probs). Plateau ≈ shared-only loss in undertrained smoke model; renormalized gates give same shape. Paper (Sec. 4.5) silent on gate handling; we keep Eq. 10 literally (documented in router.py docstring).
- Files: moe/router.py (docstring only). Debug dirs deleted. make test: 137 passed.

### 2026-10-04 — moe-builder (review-gate fixes) [pasted by orchestrator]
- train.py: skip optimizer.step on non-finite grad norm (zero_grad, log `skipped_nonfinite: true`), abort with status=failed after >10 consecutive skips; final.json `n_skipped_steps` (counted from train_log, resume-safe). Host sync moved before optimizer.step (no extra sync for the check); added `torch.cuda.synchronize()` after real steps on CUDA so step_time includes the optimizer — UNTESTED ON GPU, verify in the Milestone 4 re-probe (should cost little; one sync/step existed already).
- train.py val TokenBatcher uses max_val_tokens. ablate_experts.py: reset eval_cf to run config value, `-` for shared on switch/gshard, exp overflow guard, `per_run` val meta.
- Verified: make test 137; dense smoke n_skipped_steps 0; NaN-injection sim (1 bad step → skipped, weights finite; 11 consecutive → abort, status failed); ablate smoke prints `-`.

### 2026-10-04 — moe-builder (Milestone 3: dashboard pages + make_report) [pasted by orchestrator]
- Done: 7 pages (overview, training, routing, capacity, ablations, systems, placement), 26 fig_* builders appended to dashboard/figures.py, builder section of dashboard/results_io.py (param_table_with_checks, summarize_e1/e2/e3e4/e5_speedups/e6, headline_numbers, seed_spread_table, e2_runs, default_n_experts, placement_step_times, ...), scripts/make_report.py (RESULTS.md + results/<cfg>/report.html, Plotly JS inlined, same builders).
- Acceptance: AppTest 0 exceptions/0 st.error on all pages for empty dir, real results/smoke, and scratch copy with E6 data; make test 137; make_report smoke → RESULTS.md 9.7 kB + report.html 5.2 MB; empty dir → all "n/a: not run yet". Kaleido PNG skipped (needs Chrome).
- Decisions: S3 is a dataframe; summarize_e1/headline_numbers in results_io (re-exported); E2 delta reference = the E1-reused cf point; E5 expert filter defaults to config routed counts; E5 "skew" sweep ignored by pages; make_report uses newest E5 run with roofline.json.
- Flags (forwarded to E5 architect): theme.fmt_bytes prints "9.1e+02kB" for 3-digit values; smoke E5 deepseek batched fwd_bwd at N=1024 = 6.3× its fwd (loop 2.7×) → loop→batched speedup falls 5.0× → 1.15×; incomplete smoke e5 run 20261004-203841 (no roofline.json) listed as usable.

### 2026-10-04 — moe-architect (E5 profile_layer + E6 placement_sim) [pasted by orchestrator]
- New: moe/profiling.py (phase ranges + PhaseTimer, no-op by default), scripts/profile_layer.py, scripts/placement_sim.py, docs/E6_NOTE.md, tests/test_placement.py, tests/test_theme_format.py. Edits: phase annotations in moe_switch.py/moe_deepseek.py (no math change; Switch aux now computed inside router phase); dashboard/theme.py `_si` fixed-point fix. make test: 146 passed.
- E5: experts sweep holds activated compute fixed (Switch E×d_ff top-1; DeepSeek fine-grained total 8·d_ff, m=k=E/8, no shared, E∈{8..64}); skew via Zipf router-logit offset (default 0.3 → imbalance≈3); fwd = eval no_grad, fwd_bwd = train mode incl. aux backprop; FLOP model bf16 incl. autocast weight cast; GPU state per group; measured roof = best of 3 square GEMMs + 1 GiB copy, at start and end, after 5 s warmup; flags power_state_changed / low_power_state / gpu_busy_before_run. Laptop spec: 672 GB/s (NVIDIA laptops page); 62.0 TFLOP/s BF16 dense DERIVED (992 AI TOPS ÷ 16, ratio from Blackwell whitepaper), alt 52.3 at 2220 MHz. --quick auto for smoke (~16 s).
- E6: tokens data-parallel in contiguous blocks; greedy = LPT on calibration half (cap ceil(E/D)/device), scored on other half; shared experts replicated; a2a = dispatch+combine ×2 for backward; step = Σ_L max compute + Σ_L max comm. Only run on a scratchpad synthetic trace so far.
- Completion markers: E5 complete iff profile_meta.json exists; E6 complete iff placement_params.json exists. For fwd_bwd breakdown, "other" = whole backward (dashboard should caption). make_report's own fmt_bytes workaround can be dropped.
- OPEN: GPU sanity run never got a clean slot (overlapping jobs, one invisible to nvidia-smi). Run `make profile` alone on idle GPU, plugged in, Turbo. Smoke DeepSeek fwd_bwd anomaly = CPU noise + serial index_put backward on CPU; GPU ratio normal (2.0×).

### 2026-10-04 — main session (orchestrator) — MILESTONE 3 COMPLETE
- Fresh `make smoke` (results/smoke wiped first): exit 0, 153 s wall on CPU, max RSS ~1 GB. Produces E1 (5 variants), E2 (3 new + reused cf1.25), E3/E4, E5 quick, traces, E6, RESULTS.md, results/smoke/report.html (5.2 MB).
- AppTest of app.py + all 8 pages on fresh smoke results: 0 exceptions, 0 errors, 0 warnings; infos are only captions (CPU no-memory, eval-time label, analytical-model label). Live streamlit: HTTP 200, health ok. make test: 146 passed.
- Nit for later (builder): placement page info caption repeats "Simple analytical model, not a simulator" 3×.
- GPU NOTE: at 21:00 the GPU showed 93% util / 8.9 GB used / 87 W with no WSL process → a Windows-side app is using the dGPU. Must be closed before Milestone 4/6 timing.
- Before Milestone 4 (user decisions): close the Windows GPU app, plug in + Turbo + dGPU not Eco; re-probe (also verifies train.py per-step CUDA sync cost); choose main size (main ≈ 6 min/variant at measured speed); dense_x8 / 2nd seed; optional LR check.

### 2026-10-05 — main session (orchestrator) — pre-Milestone-4
- User said "continue" → recommended options applied: main = default model, total_steps 6000 (49.2M tokens, within the 60M cached train tokens, no repeats); E1 = all 5 variants + 2nd seed for dense/switch/deepseek; quick LR check.
- GPU idle (0%, P8). train.py probe at main size, 200 steps, bf16: dense 205k tok/s (40 ms/step, 2.0 GB), deepseek 97k (84 ms, 5.6 GB), dense_x8 77k (106 ms, 4.8 GB). Per-step CUDA sync cost acceptable. Est. at 6000 steps: dense ~4 min, deepseek ~8.5, dense_x8 ~11.
- LR check protocol (fixed BEFORE running): small config, dense + switch, peak_lr ∈ {5e-4, 1e-3, 2e-3}, seed 1337, results/small/lr_check/. Rule: pick the LR minimizing the mean val loss over the two variants; apply it to ALL variants on main. Small-config results are a proxy only.
- LR check result (results/small/lr_check/): mean val loss 5e-4 → 2.172, 1e-3 → 2.027, 2e-3 → 1.980 (dense 1.955 / switch 2.006 at 2e-3). Per the pre-set rule: peak_lr = 2e-3 for all variants on main. Caveat: winner is at the grid edge; grid NOT extended (would be tuning). Non-finite-grad guard in train.py protects against instability.
- configs/main.yaml changed by orchestrator: total_steps 3000 → 6000, peak_lr 1e-3 → 2e-3 (CONFIG_SCHEMA table updated). Decay steps at 4800/5400.

### 2026-10-05 — main session (orchestrator) — MILESTONE 4 (E1 main) RUNS COMPLETE
- 8 runs in results/main/e1_main/ (main config, 6000 steps, peak_lr 2e-3): seed 1337 all 5 variants, seed 1338 dense/switch/deepseek. Exit 0. Log: scratchpad e1_main.log.
- Paused here at user request (laptop shutdown). NEXT: E2 (`make sweep` / sweep_capacity.py main; reuses E1 cf1.25 seed 1337), E3/E4 (`make ablate`), E5 alone on idle GPU (`make profile`), traces + E6 (`make trace`, `make placement`), `make report`, then NOTES "What surprised me" + README + final summary (Milestones 5–8).

### 2026-10-05 — moe-architect (diagnosis: dense_x8 below deepseek on E1 main) [pasted by orchestrator]
- CPU-only, no code/config edits. NOT A BUG: params 63,312,768 = analytic; FFN 6×2×384×12288; plain DenseFFN, same param groups/LR/init path as dense.
- Ruled out: init (FFN output std at init dense 0.041–0.044 vs x8 0.040–0.042), instability (0 spikes, 0 skipped, clipping only in warmup), overfitting (train−val +0.06 for all runs), throttling (step time flat 113–115 ms).
- Most likely: peak LR 2e-3 too high for the 12288-wide FFN under standard parameterization. Evidence: x8 best of all at step 150, worse than dense steps 600–3600 (max +0.054 @1200), largest decay gain (0.232 vs 0.203–0.218); W_out relative drift 29× (dense 13×); FFN outputs/residual ~2.4× dense. Secondary: 0.78 tokens/total-param; x8 n=1.
- Recommendation: report as non-replication (draft text saved in docs/E1_DENSE_X8_NOTE.md). OPTIONAL post-hoc LR sensitivity after E2–E6 (needs user OK): dense_x8 @ {1e-3, 5e-4} + deepseek @ 1e-3, seed 1337, ~33 min GPU, all points reported, labelled post-hoc, headline unchanged. muP-style per-width LR NOT recommended now (changes every variant).

### 2026-10-05 — main session (orchestrator) — E2–E6 main runs complete (Milestones 5–7 data)
- Chain sweep → ablate → trace → placement → profile → report: all exit 0 (10:56–11:53). Log: scratchpad e2_e6.log.
- E2 val loss / val drop: cf0.75 1.8383 / 25.2%; cf1 1.7325 / 5.1%; cf1.25 1.7053 / 0.2% (E1 run); cf2 1.6982 / 0.0%. Monotone.
- PROBLEM: E2 cf=1 run took 2158 s (14k tok/s vs ~113–132k for the others) → GPU fell into a low-power state during that run. Its loss is valid; its THROUGHPUT is not. Plan: re-run cf=1 when power state is stable (~8 min), replace the run.
- E5 full mode, 181 s, GPU idle before run, but WARNING power state changed during run (mem clocks 14126/11126/9001/11126). Loop→batched fwd speedup falls with N: switch 2.84×@1K → 1.46×@8K → 0.89×@64K; gshard 3.83× → 1.77× → 0.99×; deepseek 7.16× → 2.90× → 1.03×. Real effect (launch/sync overhead amortizes at large N; batched pays capacity padding). Report headline currently quotes only the largest N (reads as "no speedup") → builder should report the curve / value at training N=8192 plus range.
- Plan: user confirms plugged in + Turbo → re-run E5 + E2 cf=1 (~12 min) → regenerate report. Optional (user decision): post-hoc dense_x8 LR sensitivity (~33 min).

### 2026-10-05 — main session (orchestrator) — redo A complete
- Superseded runs moved to results/main/_superseded/ (README explains): old E2 cf1 (low-power, 14k tok/s) and old E5 profile.
- E2 cf=1.0 re-run (20261005-115602): val 1.7325 (identical to superseded run), 133k tok/s, 407 s. sweep_index.json updated automatically.
- E5 re-run (20261005-120250_profile, 182 s): measured roof 57.3 TFLOP/s bf16 GEMM, 481 GB/s copy (start/end 53.8/57.3 TFLOP/s, 481/482 GB/s → ratio 1.07); spec 62.0 (derived) / 672 GB/s. 0 of 162 bench rows in low_power_state; max achieved 38.2 TFLOP/s < measured roof. `power_state_changed` flag still fires because the memory clock moves 9001/11126/14126 MHz between benchmark groups (it drops while briefly idle between groups); no row ran in the 810 MHz throttled state. Accepting this run, with the caveat stated in RESULTS/NOTES.

### 2026-10-05 — moe-builder (Makefile + README) [pasted by orchestrator]
- Makefile: `make main` = real E1 (`--include-optional --seeds 1337 --skip-existing`, then `--variants dense switch deepseek --seeds 1337 1338 --skip-existing`); new `make all` = main sweep ablate trace placement profile report (trace before placement); `make dashboard` adds `--server.headless true --browser.gatherUsageStats false` (Streamlit blocked on first-run email prompt). Verified make -n main/all + live curl (HTTP 200, health ok).
- README.md (new): install incl. venv --without-pip quirk, cu128 torch, verify; commands per experiment; resume; live-watch one-liner; measured run times (scratchpad runtimes.py from final.json + chain logs: E1 total 3600 s for 8 runs, whole chain ~87 min, smoke 153 s); dashboard + report.html; troubleshooting; repo map. No result numbers besides run times / peak memory.
- make test: 150 passed.

### 2026-10-05 — moe-architect (final dashboard review + NOTES final) [pasted by orchestrator]
- Reviewed all 8 pages vs DASHBOARD_SPEC on real results/main; AppTest main/empty/smoke 0 exceptions; 20 interactive states clean; live streamlit ok.
- BUG FIX figures.fig_breakdown: was averaging cuda_events and torch.profiler rows; now plots one method (default cuda_events), Systems page radio + caption, make_report names method.
- Fixes: theme.fmt_bytes(x, digits=2) public (make_report digits=3; NaN handling; new test); seed table 3 dp; roofline legends; Systems power_state_changed warning; placement slider grid includes 64 GB/s default; Ablations caption lists source checkpoint/seed (Switch/DeepSeek s1338, GShard s1337).
- NOTES.md: §10/§12 final, "What surprised me" filled (E1–E6, caveats, limitations), "Resume-ready facts" + do-not-claim list. Verifier (scratchpad verify_notes.py) recomputes ~90 numbers from results/: 401 checks.
- Follow-ups noted: R4 entropy reference line labelled only in hover; P2/P3 legends use raw variant ids (cosmetic).

### 2026-10-05 — main session (orchestrator)
- E6 re-run with measured device roof 57.33 TFLOP/s (old E6 run moved to _superseded with README note). make report + make test (150 passed). NOTES verifier: 1 failure (stale E6 path) → sent back to architect to update E6 refs/numbers.

### 2026-10-05 — main session (orchestrator) — MILESTONE 8 COMPLETE (project done)
- NOTES.md E6 refs/numbers updated to re-run E6 (20261005-122540_placement); verifier 421 checks, 0 failures.
- Final check: make test green; AppTest all 8 pages + app on real results (main default) 0 exceptions/errors; README, NOTES, RESULTS, results/{main,smoke}/report.html present.
- Remaining optional items (not done, user decision): post-hoc dense_x8 LR sensitivity (~33 min GPU); cosmetic dashboard nits (R4 entropy line legend, P2/P3 raw variant ids).
- 2026-10-05 orchestrator: pushed to https://github.com/GP-creator/MOE_reproduced (main). Author GP-creator with GitHub noreply email; excluded *.pt, data_cache, traces, profiler traces; PII/secret scan clean; PROGRESS home path scrubbed.
