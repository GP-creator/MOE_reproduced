"""E3 (expert redundancy) + E4 (shared-expert ablation), eval-only.  MASTER §7, PLAN §7, DASHBOARD_SPEC §1.5.

Loads trained checkpoints and changes the layer behaviour at EVAL time only
(``layer_api.set_eval_overrides``). Nothing is retrained, so these numbers are not the same
as training a model without the shared expert / with a different k from scratch.

  E3: per token mask the n_dis highest-prob routed experts, top-k from the rest (gate = original
      softmax prob, no renorm). Switch/GShard evaluated with NO_DROP capacity, plus an r=0
      reference point at the training capacity factor.
  E4: DeepSeek only. baseline / (a) no shared, same k / (b) no shared, k+1 (compute matched) /
      k_routed sweep {1,2,3} (+ optional k=4 = over-activation, labelled).

Usage:
  python scripts/ablate_experts.py --config configs/main.yaml
  python scripts/ablate_experts.py --config configs/smoke.yaml --results-dir /path/to/results
  python scripts/ablate_experts.py --config configs/main.yaml --runs results/main/e1_main/<run_id> ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _eval_common as ec  # noqa: E402  (also puts the repo root on sys.path)

import torch  # noqa: E402

from moe import utils  # noqa: E402
from moe.layer_api import NO_DROP, set_eval_overrides  # noqa: E402

NOTE = "eval-time change, not retrained"

#: PLAN §7 E3 table (counts, not fractions): variant -> n_disable_top_routed values.
E3_COUNTS = {"deepseek": (0, 2, 4, 6, 8), "switch": (0, 1, 2), "gshard": (0, 1, 2, 3, 4)}


@torch.no_grad()
def evaluate(model, batches, device, ac_dtype) -> dict:
    """Mean CE over fixed batches (eval mode, autocast like training eval). Equal-size batches."""
    ce_sum, drop_sum, n_tok, drop_layers = 0.0, 0.0, 0, 0
    for x, y in batches:
        with utils.autocast_context(device, ac_dtype):
            out = model(x, y)
        ce_sum += float(out.ce_loss)
        n_tok += y.numel()
        drops = [float(a.drop_fraction) for a in out.layer_aux if a.is_moe and a.drop_fraction is not None]
        drop_sum += sum(drops)
        drop_layers += len(drops)
    val = ce_sum / len(batches)
    return {"val_loss": val, "val_ppl": math.exp(val) if val < 700 else None, "n_val_tokens": n_tok,
            "val_drop_fraction": (drop_sum / drop_layers) if drop_layers and model.variant != "deepseek" else None}


def reset_overrides(model, variant: str, rcfg: dict) -> None:
    kw = dict(n_disable_top_routed=0)
    if variant == "deepseek":
        kw.update(disable_shared=False, k_routed_override=None)
    else:
        kw.update(eval_capacity_factor=rcfg["moe"][variant]["eval_capacity_factor"])
    set_eval_overrides(model, **kw)


def run_e3_e4(run: dict, model, rcfg: dict, batches, device, ac_dtype, root: Path, with_k4: bool) -> list[dict]:
    variant = run["variant"]
    layer = ec.moe_layers(model)[0]
    spec = layer.spec
    train_cf = getattr(layer, "capacity_factor", None)          # None for deepseek (no capacity)
    base = {"source_run_dir": ec.rel_to(root, run["run_dir"]), "variant": variant, "seed": run["seed"],
            "n_routed": spec.n_routed, "n_shared": spec.n_shared, "top_k_trained": spec.top_k, "note": NOTE}

    def point(study, n_dis=0, disable_shared=False, k=None, cf="nodrop", **fields) -> dict:
        """Evaluate one configuration and return its row (baseline filled in by the caller)."""
        reset_overrides(model, variant, rcfg)
        kw = dict(n_disable_top_routed=n_dis)
        if variant == "deepseek":
            kw.update(disable_shared=disable_shared, k_routed_override=k)
        elif cf == "nodrop":
            kw.update(eval_capacity_factor=NO_DROP)
        set_eval_overrides(model, **kw)
        res = evaluate(model, batches, device, ac_dtype)
        reset_overrides(model, variant, rcfg)
        k_eff = k if k is not None else spec.top_k
        n_shared_active = 0 if disable_shared else spec.n_shared
        no_drop = variant == "deepseek" or cf == "nodrop"       # deepseek has no capacity at all
        return {"study": study, **base,
                "n_disable_top_routed": n_dis, "frac_disabled": n_dis / spec.n_routed,
                "disable_shared": disable_shared, "k_routed": k_eff, "condition": None,
                "active_ffn_width": (n_shared_active + k_eff) * spec.width,
                "capacity_factor": None if no_drop else train_cf, "no_drop": no_drop,
                "is_reference": False, **res, **fields}

    rows: list[dict] = []
    # ---- E3
    e3 = [point("e3_disable_top", n) for n in E3_COUNTS[variant]]
    for r in e3:
        r["baseline_val_loss"] = e3[0]["val_loss"]
    rows += e3
    if variant != "deepseek":                                    # r=0 at the training capacity factor
        ref = point("e3_disable_top", 0, cf="train", is_reference=True)
        ref["baseline_val_loss"] = ref["val_loss"]               # baseline of its own cf
        rows.append(ref)

    # ---- E4 (DeepSeek only)
    if variant == "deepseek":
        k0 = spec.top_k
        e4 = [point("e4_shared", condition="baseline", label="baseline (shared on, trained k)"),
              point("e4_shared", disable_shared=True, condition="no_shared_same_k",
                    label="(a) shared off, k unchanged"),
              point("e4_shared", disable_shared=True, k=k0 + 1, condition="no_shared_plus_one",
                    label="(b) shared off, k+1 (compute matched)")]
        for k in (1, 2, 3, 4):
            if k > spec.n_routed or (k > k0 and not with_k4):
                continue
            lab = f"k={k}, shared on" + (" (trained k)" if k == k0 else "")
            if k > k0:
                lab += " (over-activation: k above trained k)"
            e4.append(point("e4_k_sweep", k=k, label=lab, over_activation=k > k0))
        for r in e4:
            r["baseline_val_loss"] = e4[0]["val_loss"]
        rows += e4
    for r in rows:
        r["delta_val_loss"] = r["val_loss"] - r["baseline_val_loss"]
    return rows


def print_tables(all_rows: list[dict]) -> None:
    for study, title in (("e3_disable_top", "E3: disable top-n routed experts per token"),
                         ("e4_shared", "E4: shared-expert ablation (DeepSeek)"),
                         ("e4_k_sweep", "E4: k_routed sweep at eval (DeepSeek, shared on)")):
        rows = [r for r in all_rows if r["study"] == study]
        if not rows:
            continue
        print(f"\n== {title}  [{NOTE}] ==")
        out = []
        for r in rows:
            out.append({"variant": r["variant"], "n_dis": r["n_disable_top_routed"],
                        "frac": f"{r['frac_disabled']:.4f}", "k": r["k_routed"],
                        "shared": "-" if not r["n_shared"] else ("off" if r["disable_shared"] else "on"), "width": r["active_ffn_width"],
                        "cf": "no-drop" if r["no_drop"] else f"{r['capacity_factor']:g}",
                        "val_loss": f"{r['val_loss']:.4f}", "delta": f"{r['delta_val_loss']:+.4f}",
                        "drop": "-" if r["val_drop_fraction"] is None else f"{r['val_drop_fraction']:.3f}",
                        "what": r.get("label") or ("train-cf reference" if r["is_reference"] else "")})
        print(utils.format_table(out, list(out[0])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--override", action="append", default=[],
                    help="a.b.c=value (only device/results_dir/eval_batches matter; models use their run's config.json)")
    ap.add_argument("--results-dir", default=None, help="results root (default: config system.results_dir)")
    ap.add_argument("--runs", nargs="+", default=None, help="explicit run dirs (default: latest complete per MoE variant)")
    ap.add_argument("--variants", nargs="+", default=list(ec.MOE_VARIANTS))
    ap.add_argument("--eval-batches", type=int, default=None,
                    help="val batches per point (default: 50 for the main config, else train.eval_batches)")
    ap.add_argument("--no-k4", action="store_true", help="skip the k=4 over-activation point of the E4 sweep")
    args = ap.parse_args()

    cfg0 = utils.load_config(args.config, args.override)
    root = ec.results_root(cfg0, args.results_dir)
    device = utils.get_device(cfg0)
    ac_dtype = utils.get_autocast_dtype(cfg0, device)
    n_batches = args.eval_batches or (50 if cfg0["name"] == "main" else int(cfg0["train"]["eval_batches"]))

    runs = ec.find_runs(root, cfg0["name"], tuple(args.variants), args.runs)
    if not runs:
        sys.exit(f"no complete runs found under {root / cfg0['name'] / 'e1_main'} (run E1 first, or pass --runs)")
    run_id = utils.make_run_id("ablation")
    cfg_out = dict(cfg0, system=dict(cfg0["system"], results_dir=str(root)))
    out_dir = utils.make_run_dir(cfg_out, f"{cfg0['name']}/e3e4_ablation", run_id)

    all_rows: list[dict] = []
    per_run: dict[str, dict] = {}
    for run in runs:
        print(f"[ablate] {run['variant']}  {run['run_id']}  ({n_batches} batches)")
        model, rcfg = ec.load_run_model(run, device)
        batches = ec.val_batches(rcfg, n_batches, device)
        new_rows = run_e3_e4(run, model, rcfg, batches, device, ac_dtype, root, with_k4=not args.no_k4)
        all_rows += new_rows
        n_used, tok_used = len(batches), new_rows[-1]["n_val_tokens"]
        per_run[run["run_id"]] = {"n_val_batches": n_used, "n_val_tokens": tok_used}
        del model

    with utils.JsonlLogger(out_dir / "ablation.jsonl") as log:
        for r in all_rows:
            log.log(r)
    utils.save_json(out_dir / "env.json", utils.get_env_info(device, ac_dtype))
    utils.save_json(out_dir / "ablation_meta.json", {
        "schema_version": 1, "run_id": run_id, "config_name": cfg0["name"],
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_val_batches": n_used, "n_val_tokens": tok_used, "per_run": per_run,
        "source_runs": [{"run_dir": ec.rel_to(root, r["run_dir"]), "variant": r["variant"], "seed": r["seed"]} for r in runs],
        "note": "eval-time ablation; not retraining"})
    print_tables(all_rows)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
