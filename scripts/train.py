"""Train one model (one variant, one seed) and write the results files of DASHBOARD_SPEC 1.4.

CLI::

    python scripts/train.py --config configs/smoke.yaml --variant switch
    python scripts/train.py --config configs/main.yaml --variant deepseek --seed 1 --tag lr2e-3 \
        --override train.peak_lr=2e-3
    python scripts/train.py --resume results/main/e1_main/<run_id>      # continue a stopped run
    python scripts/train.py --config configs/main.yaml --variant switch --resume auto

In-process use (the runners do this to avoid ~3 s interpreter+torch start-up per run)::

    run_dir = train(cfg, experiment="e1_main")

Loop per step: set LR -> forward (autocast) -> loss = CE + aux -> backward -> clip -> optimizer
step -> update routing accumulator -> log. Step numbers in logs are 1-based (= optimizer updates
completed). Resuming restarts at checkpoint.step + 1; the batcher is a pure function of
(seed, step) so the data stream is identical to an uninterrupted run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import (REPO_ROOT, capacity_factor, dedupe_by_step, experiment_dir,  # noqa: E402
                             read_jsonl, rel_to_results, run_group, sanitize)

import torch  # noqa: E402

from moe import flops  # noqa: E402
from moe.data import TokenBatcher, prepare_data  # noqa: E402
from moe.metrics import RoutingAccumulator  # noqa: E402
from moe.model import MoETransformer  # noqa: E402
from moe.schedule import lr_at_step, set_lr  # noqa: E402
from moe.utils import (JsonlLogger, autocast_context, find_latest_checkpoint, get_autocast_dtype,  # noqa: E402
                       get_device, get_env_info, load_checkpoint, load_config, load_json,
                       make_run_dir, make_run_id, peak_memory_mb, reset_peak_memory,
                       save_checkpoint, save_json, seed_everything)

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model: MoETransformer, batches: list[tuple[torch.Tensor, torch.Tensor]],
             device: torch.device, amp_dtype: torch.dtype | None, has_capacity: bool) -> dict[str, Any]:
    """Validation pass: eval mode (eval capacity factor rules), autocast on, one host sync.

    ``val_loss`` is CE only. ``val_aux_loss`` is logged, never added. ``val_drop_fraction`` is the
    mean over MoE layers and batches of ``MoEAux.drop_fraction`` (None for variants without capacity).
    """
    was_training = model.training
    model.eval()
    ce = torch.zeros((), dtype=torch.float64, device=device)
    aux = torch.zeros((), dtype=torch.float64, device=device)
    drop = torch.zeros((), dtype=torch.float64, device=device)
    for x, y in batches:
        with autocast_context(device, amp_dtype):
            out = model(x, y)
        ce += out.ce_loss.double()
        aux += out.aux_loss.double()
        if has_capacity:
            drop += torch.stack([a.drop_fraction.float() for a in out.layer_aux if a.is_moe]).mean().double()
    ce_v, aux_v, drop_v = (torch.stack([ce, aux, drop]) / len(batches)).cpu().tolist()
    model.train(was_training)
    x0 = batches[0][0]
    return {"val_loss": ce_v, "val_ppl": math.exp(ce_v) if ce_v < 700 else None,
            "val_aux_loss": aux_v, "val_drop_fraction": drop_v if has_capacity else None,
            "n_val_tokens": len(batches) * x0.numel()}


# ----------------------------------------------------------------------------- resume helpers

def find_resume_dir(cfg: dict, experiment: str, tag: str | None = None) -> Path | None:
    """Latest incomplete run (no final.json) of the same group and seed with a checkpoint."""
    group, seed = run_group(cfg, experiment, tag), int(cfg["seed"])
    exp_dir = experiment_dir(cfg, experiment)
    if not exp_dir.is_dir():
        return None
    for d in sorted((p for p in exp_dir.iterdir() if p.is_dir()), reverse=True):
        run_json = d / "run.json"
        if (d / "final.json").exists() or not run_json.exists():
            continue
        meta = load_json(run_json)
        if meta.get("group") == group and meta.get("seed") == seed and find_latest_checkpoint(d):
            return d
    return None


#: abort the run (status=failed) after this many consecutive skipped (non-finite grad) steps
MAX_CONSECUTIVE_SKIPS = 10

# ----------------------------------------------------------------------------- final.json

def build_final(cfg: dict, run_meta: dict, run_dir: Path, env: dict, n_params: int) -> dict[str, Any]:
    """Assemble final.json (DASHBOARD_SPEC 1.4) from the on-disk logs (so resumed runs are right)."""
    tcfg = cfg["train"]
    total = int(tcfg["total_steps"])
    tokens_per_step = int(tcfg["batch_size"]) * int(cfg["model"]["seq_len"])
    train_rows = dedupe_by_step(read_jsonl(run_dir / "train_log.jsonl"))
    eval_rows = dedupe_by_step(read_jsonl(run_dir / "eval_log.jsonl"))
    routing = [r for r in read_jsonl(run_dir / "routing_log.jsonl")]
    last_eval = eval_rows[-1]
    best = min(eval_rows, key=lambda r: r["val_loss"] if r["val_loss"] is not None else float("inf"))

    n_last = max(1, int(round(0.05 * total)))
    last_ce = [r["ce_loss"] for r in train_rows if r["step"] > total - n_last and r["ce_loss"] is not None]
    skip = max(10, int(round(0.05 * total)))
    post_warm = [r["tokens_per_sec"] for r in train_rows if r["step"] > skip and r["tokens_per_sec"]]
    if not post_warm:  # very short runs: fall back to everything logged
        post_warm = [r["tokens_per_sec"] for r in train_rows if r["tokens_per_sec"]]
    train_time_s = sum(r["step_time_ms"] for r in train_rows) / 1000.0

    has_capacity = capacity_factor(cfg) is not None
    train_drop = None
    if has_capacity:
        recent = [r["drop_fraction"] for r in routing if r["step"] > 0.9 * total and r["drop_fraction"] is not None]
        train_drop = statistics.fmean(recent) if recent else None

    row = flops.param_row(cfg["variant"], cfg)
    final = {
        "schema_version": SCHEMA_VERSION,
        **{k: run_meta[k] for k in ("run_id", "experiment", "config_name", "variant", "seed", "group",
                                    "capacity_factor")},
        "status": "completed",
        "steps": train_rows[-1]["step"], "tokens_seen": train_rows[-1]["tokens_seen"],
        "final_val_loss": last_eval["val_loss"], "final_val_ppl": last_eval["val_ppl"],
        "best_val_loss": best["val_loss"], "best_val_step": best["step"],
        "final_train_ce": statistics.fmean(last_ce) if last_ce else None,
        "total_params": n_params,
        "total_params_analytic": row["total_params"], "active_params": row["active_params"],
        "ffn_total_model": row["ffn_total_model"], "ffn_active_model": row["ffn_active_model"],
        "router_model": row["router_model"],
        "model_fwd_flops_per_token": row["model_fwd_flops_per_token"],
        "ffn_flops_per_token_model": row["ffn_flops_per_token_model"],
        "median_tokens_per_sec": statistics.median(post_warm) if post_warm else None,
        "mean_tokens_per_sec": (len(train_rows) * tokens_per_step / train_time_s) if train_time_s > 0 else None,
        "peak_mem_mb": max((r["peak_mem_mb"] for r in train_rows), default=0.0),
        "train_time_s": train_time_s,
        "n_skipped_steps": sum(1 for r in train_rows if r.get("skipped_nonfinite")),
        "wall_time_s": train_rows[-1]["wall_time_s"],
        "val_drop_fraction": last_eval["val_drop_fraction"] if has_capacity else None,
        "train_drop_fraction_last": train_drop,
        "checkpoint": rel_to_results(cfg, run_dir / "checkpoint.pt"),
        "hardware_label": env["hardware_label"], "device_type": env["device_type"], "dtype": env["dtype"],
    }
    return sanitize(final)


# ----------------------------------------------------------------------------- train

def train(cfg: dict, experiment: str = "e1_main", run_id: str | None = None,
          resume_dir: Path | str | None = None, tag: str | None = None,
          max_steps: int | None = None, overrides: list[str] | None = None,
          verbose: bool = True) -> Path:
    """Train ``cfg["variant"]`` and return the run directory.

    resume_dir: continue that run (its config.json is authoritative; ``cfg`` is only used for the
        caller's convenience). max_steps: stop early after that many total steps (debug); no
        final.json is written then, but a checkpoint is, so the run can be resumed.
    tag: extra label appended to run_id and group. overrides: raw ``--override`` strings, recorded.
    """
    resuming = resume_dir is not None
    if resuming:
        run_dir = Path(resume_dir).resolve()
        cfg = load_json(run_dir / "config.json")
        run_meta = load_json(run_dir / "run.json")
        experiment = run_meta["experiment"]
    group = run_group(cfg, experiment, tag if not resuming else None)
    variant, seed = cfg["variant"], int(cfg["seed"])
    tcfg = cfg["train"]
    total = int(tcfg["total_steps"])
    last_step = min(total, max_steps) if max_steps else total

    seed_everything(seed)
    device = get_device(cfg)
    amp_dtype = get_autocast_dtype(cfg, device)
    if device.type == "cuda" and total > 1000:
        print("NOTE: long GPU run. Check: laptop plugged in, Windows power mode on Turbo/Performance, "
              "NVIDIA dGPU mode not Eco (clocks/throughput otherwise drop several-fold).")

    data = prepare_data(cfg["data"], REPO_ROOT)
    seq_len, bs = int(cfg["model"]["seq_len"]), int(tcfg["batch_size"])
    train_batcher = TokenBatcher(data["train_bin"], seq_len, bs, seed, max_tokens=int(cfg["data"]["max_train_tokens"]))
    val_batcher = TokenBatcher(data["val_bin"], seq_len, bs, seed, max_tokens=int(cfg["data"]["max_val_tokens"]))
    eval_batches = val_batcher.get_eval_batches(int(tcfg["eval_batches"]), device)

    model = MoETransformer(cfg).to(device)
    n_params = model.verify_param_count()
    if verbose:
        print(model.param_table_str())
    print(f"[{variant}] seed={seed} measured params={n_params:,} device={device} "
          f"dtype={'bf16-autocast' if amp_dtype else 'fp32'}")
    optimizer = model.configure_optimizer(tcfg)
    fwd = torch.compile(model) if tcfg.get("compile") else model
    env = get_env_info(device, amp_dtype)

    # ---- run directory + identity files
    start_step = 0
    if resuming:
        ckpt_path = find_latest_checkpoint(run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"no checkpoint to resume from in {run_dir}")
        ckpt = load_checkpoint(ckpt_path, model, optimizer, map_location=str(device))
        start_step = int(ckpt["step"])
        run_meta["status"] = "running"
        run_meta["resumed_from_step"] = list(run_meta.get("resumed_from_step", [])) + [start_step]
        env = load_json(run_dir / "env.json") if (run_dir / "env.json").exists() else env
        print(f"resuming {run_dir.name} from step {start_step}")
    else:
        default_tag = f"cf{capacity_factor(cfg):g}_s{seed}" if experiment == "e2_capacity" else f"s{seed}"
        run_id = run_id or make_run_id(variant, default_tag + (f"_{tag}" if tag else ""))
        run_dir = make_run_dir(cfg, f"{cfg['name']}/{experiment}", run_id)
        run_meta = {
            "schema_version": SCHEMA_VERSION, "run_id": run_dir.name, "experiment": experiment,
            "config_name": cfg["name"], "variant": variant, "seed": seed, "group": group,
            "capacity_factor": capacity_factor(cfg), "dispatch": cfg["moe"]["dispatch"],
            "overrides": list(overrides or []), "status": "running",
            "started_at": dt.datetime.now().isoformat(timespec="seconds"), "finished_at": None,
            "resumed_from_step": [],
        }
        save_json(run_dir / "config.json", cfg)
        save_json(run_dir / "env.json", sanitize(env))
    save_json(run_dir / "run.json", sanitize(run_meta))
    print(f"run dir: {run_dir}")

    try:
        _loop(cfg, run_dir, run_meta, model, fwd, optimizer, device, amp_dtype, train_batcher,
              eval_batches, env, start_step, last_step, total, n_params, verbose)
    except BaseException:
        run_meta["status"] = "failed"
        save_json(run_dir / "run.json", sanitize(run_meta))
        raise
    return run_dir


def _loop(cfg: dict, run_dir: Path, run_meta: dict, model: MoETransformer, fwd: torch.nn.Module,
          optimizer: torch.optim.Optimizer, device: torch.device, amp_dtype: torch.dtype | None,
          batcher: TokenBatcher, eval_batches: list, env: dict, start_step: int, last_step: int,
          total: int, n_params: int, verbose: bool) -> None:
    """The training loop proper (steps start_step+1 .. last_step) plus end-of-run bookkeeping."""
    tcfg = cfg["train"]
    seq_len, bs = int(cfg["model"]["seq_len"]), int(tcfg["batch_size"])
    tokens_per_step = bs * seq_len
    has_capacity = capacity_factor(cfg) is not None
    eval_every, route_every = int(tcfg["eval_interval"]), int(tcfg["routing_log_interval"])
    ckpt_every, log_every = int(tcfg["checkpoint_interval"]), int(tcfg["log_interval"])
    grad_clip = float(tcfg["grad_clip"])
    is_moe = cfg["variant"] in ("switch", "gshard", "deepseek")

    train_log = JsonlLogger(run_dir / "train_log.jsonl")
    eval_log = JsonlLogger(run_dir / "eval_log.jsonl")
    routing_log = JsonlLogger(run_dir / "routing_log.jsonl") if is_moe else None
    acc = RoutingAccumulator()

    prev = read_jsonl(run_dir / "train_log.jsonl")
    wall_offset = prev[-1]["wall_time_s"] if (start_step > 0 and prev) else 0.0
    reset_peak_memory(device)
    peak_prev = max((r["peak_mem_mb"] for r in prev), default=0.0) if start_step > 0 else 0.0
    t_start = time.perf_counter()
    warned_nan = False
    n_skipped = consec_skipped = 0

    def do_eval(step: int) -> None:
        rec = evaluate(model, eval_batches, device, amp_dtype, has_capacity)
        rec = {"step": step, "tokens_seen": step * tokens_per_step, **rec}
        eval_log.log(sanitize(rec))
        drop = rec["val_drop_fraction"]
        print(f"  eval step {step:>5}/{total}  val_loss {rec['val_loss']:.4f}  "
              f"ppl {rec['val_ppl'] if rec['val_ppl'] is None else round(rec['val_ppl'], 2)}"
              + (f"  drop {drop:.4f}" if drop is not None else ""))

    if start_step == 0:
        do_eval(0)

    model.train()
    for s0 in range(start_step, last_step):          # s0 = 0-based index of the update being made
        step = s0 + 1
        t0 = time.perf_counter()
        lr = lr_at_step(s0, tcfg)
        set_lr(optimizer, lr)
        x, y = batcher.get_train_batch(s0, device)
        with autocast_context(device, amp_dtype):
            out = fwd(x, y)
        loss = out.ce_loss + out.aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # One host sync per step (before the update, so a non-finite grad can skip it).
        ce, aux, gn = torch.stack([out.ce_loss.detach(), out.aux_loss.detach(),
                                   grad_norm.detach().float()]).cpu().tolist()
        skipped = not math.isfinite(gn)
        if skipped:
            optimizer.zero_grad(set_to_none=True)
            n_skipped += 1
            consec_skipped += 1
        else:
            optimizer.step()
            consec_skipped = 0
            if device.type == "cuda":                # keep step_time honest: include the async update
                torch.cuda.synchronize()
        step_time = time.perf_counter() - t0
        acc.update(out.layer_aux)

        if skipped and not warned_nan:
            warned_nan = True
            print(f"WARNING: non-finite grad norm at step {step} (ce={ce}, aux={aux}, grad_norm={gn}); "
                  "optimizer step skipped (further skips are logged with skipped_nonfinite=true)")
        if step % log_every == 0 or step == total or skipped:
            rec = {"step": step, "tokens_seen": step * tokens_per_step, "lr": lr, "ce_loss": ce,
                   "aux_loss": aux, "total_loss": ce + aux, "grad_norm": gn,
                   "tokens_per_sec": tokens_per_step / step_time, "step_time_ms": step_time * 1000.0,
                   "peak_mem_mb": max(peak_prev, peak_memory_mb(device)),
                   "wall_time_s": wall_offset + time.perf_counter() - t_start}
            if skipped:
                rec["skipped_nonfinite"] = True
            train_log.log(sanitize(rec))
        if consec_skipped > MAX_CONSECUTIVE_SKIPS:
            run_meta["failure_reason"] = (f"{consec_skipped} consecutive non-finite grad norms "
                                          f"(last at step {step}); aborting")
            raise RuntimeError(run_meta["failure_reason"])
        if step % max(1, total // 20) == 0 and verbose:
            print(f"  step {step:>5}/{total}  ce {ce:.4f}  aux {aux:.4f}  gnorm {gn:.2f}  lr {lr:.2e}  "
                  f"{tokens_per_step / step_time:,.0f} tok/s")
        if step % eval_every == 0 or step == total:
            do_eval(step)
        if routing_log is not None and (step % route_every == 0 or step == last_step):
            for rec in acc.flush(step):
                routing_log.log(sanitize(rec))
        if ckpt_every > 0 and step % ckpt_every == 0 and step < last_step:
            save_checkpoint(run_dir / f"ckpt_step{step:06d}.pt", model, optimizer, step, cfg)

    train_log.close(); eval_log.close()
    if routing_log is not None:
        routing_log.close()

    peak = max(peak_prev, peak_memory_mb(device))
    if last_step < total:                              # --max-steps debug stop: resumable, not complete
        save_checkpoint(run_dir / f"ckpt_step{last_step:06d}.pt", model, optimizer, last_step, cfg)
        rows = dedupe_by_step(read_jsonl(run_dir / "train_log.jsonl"))
        tail = [r for r in rows if r["step"] > start_step + 2] or rows   # skip first (warm-up) steps
        print(f"stopped at step {last_step}/{total} (--max-steps); no final.json. "
              f"median tok/s {statistics.median(r['tokens_per_sec'] for r in tail):,.0f}, "
              f"median step_time_ms {statistics.median(r['step_time_ms'] for r in tail):.1f}, "
              f"peak_mem_mb {peak:.0f}")
        return

    save_checkpoint(run_dir / "checkpoint.pt", model, optimizer, total, cfg)
    for old in run_dir.glob("ckpt_step*.pt"):          # periodic checkpoints are superseded
        old.unlink()
    final = build_final(cfg, run_meta, run_dir, env, n_params)
    save_json(run_dir / "final.json", final)
    run_meta["status"] = "completed"
    run_meta["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    save_json(run_dir / "run.json", sanitize(run_meta))
    print(f"done: val_loss {final['final_val_loss']:.4f}  ppl {final['final_val_ppl']:.2f}  "
          f"median {final['median_tokens_per_sec']:,.0f} tok/s  peak_mem {final['peak_mem_mb']:.0f} MiB")


# ----------------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config (required unless --resume <dir>)")
    ap.add_argument("--variant", choices=["dense", "switch", "deepseek", "gshard", "dense_x8"])
    ap.add_argument("--seed", type=int, help="overrides cfg seed")
    ap.add_argument("--experiment", default="e1_main", help="results sub-directory (default e1_main)")
    ap.add_argument("--tag", help="extra label appended to run_id and group")
    ap.add_argument("--override", action="append", default=[], metavar="a.b=v", help="repeatable")
    ap.add_argument("--resume", metavar="RUN_DIR|auto", help="resume a run dir, or 'auto' = latest incomplete "
                    "run of the same group/seed")
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (debug; no final.json)")
    args = ap.parse_args()

    if args.resume and args.resume != "auto":
        train(None, resume_dir=args.resume, max_steps=args.max_steps)  # type: ignore[arg-type]
        return
    if not args.config:
        ap.error("--config is required")
    cfg = load_config(args.config, args.override, args.variant)
    if args.seed is not None:
        cfg["seed"] = args.seed
    resume_dir = find_resume_dir(cfg, args.experiment, args.tag) if args.resume == "auto" else None
    if args.resume == "auto" and resume_dir is None:
        print("no incomplete run found; starting a new one")
    train(cfg, args.experiment, resume_dir=resume_dir, tag=args.tag, max_steps=args.max_steps,
          overrides=args.override)


if __name__ == "__main__":
    main()
