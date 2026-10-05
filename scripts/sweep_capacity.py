"""E2: Switch capacity-factor sweep (training cf in {0.75, 1.0, 1.25, 2.0} by default).

    python scripts/sweep_capacity.py --config configs/main.yaml
    python scripts/sweep_capacity.py --config configs/main.yaml --cfs 1.0 1.5 --seeds 1 2

The point at the config's own switch capacity factor reuses a completed E1 switch run (same
resolved config and seed) instead of retraining; ``e2_capacity/sweep_index.json`` records which
run directory backs every point (DASHBOARD_SPEC 1.4). Eval capacity factor follows
``moe.switch.eval_capacity_factor`` (null = same as the training cf of each point).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import (experiment_dir, find_runs, fmt, rel_to_results, run_group,  # noqa: E402
                             sanitize, seeded)
from scripts.train import train  # noqa: E402

from moe.utils import format_table, load_config, load_json, save_json  # noqa: E402

DEFAULT_CFS = [0.75, 1.0, 1.25, 2.0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cfs", nargs="+", type=float, default=DEFAULT_CFS, help="capacity factors to sweep")
    ap.add_argument("--seeds", nargs="+", type=int, help="default: [cfg.seed]")
    ap.add_argument("--skip-existing", action="store_true", help="also reuse completed E2 runs (same cf/config/seed)")
    ap.add_argument("--override", action="append", default=[], metavar="a.b=v", help="repeatable")
    ap.add_argument("--max-steps", type=int, help="debug: stop each run early (no final.json)")
    args = ap.parse_args()

    base = load_config(args.config, args.override, "switch")
    if base["moe"]["switch"].get("eval_capacity_factor") is not None:
        print("NOTE: moe.switch.eval_capacity_factor is fixed; eval will NOT follow each point's training cf.")
    seeds = args.seeds or [int(base["seed"])]
    t_all = time.perf_counter()

    points: list[dict] = []
    for seed in seeds:
        for cf in args.cfs:
            cfg = seeded(base, seed)
            cfg["moe"]["switch"]["capacity_factor"] = cf
            reused = find_runs(cfg, "e1_main", "switch", seed, completed=True)
            if reused:                                    # E1 switch run with identical config + seed
                run_dir, reused_from = reused[-1], "e1_main"
                print(f"cf={cf:g} seed={seed}: reusing E1 run {run_dir.name}")
            else:
                existing = (find_runs(cfg, "e2_capacity", run_group(cfg, "e2_capacity"), seed, completed=True)
                            if args.skip_existing else [])
                if existing:
                    run_dir, reused_from = existing[-1], None
                    print(f"cf={cf:g} seed={seed}: reusing E2 run {run_dir.name}")
                else:
                    print(f"\n=== E2: switch cf={cf:g} seed={seed} ===")
                    t0 = time.perf_counter()
                    run_dir = train(cfg, "e2_capacity", max_steps=args.max_steps,
                                    overrides=args.override, verbose=False)
                    print(f"=== cf={cf:g} seed={seed} took {time.perf_counter() - t0:.1f} s ===")
                    reused_from = None
            if (run_dir / "final.json").exists():
                point = {"capacity_factor": cf, "seed": seed, "run_dir": rel_to_results(cfg, run_dir)}
                if reused_from:
                    point["reused_from"] = reused_from
                points.append(point)

    # sweep_index.json: merge with an existing index (other seeds / cfs), this invocation wins ties.
    index_path = experiment_dir(base, "e2_capacity") / "sweep_index.json"
    merged = {(p["capacity_factor"], p["seed"]): p
              for p in (load_json(index_path)["points"] if index_path.exists() else [])}
    merged.update({(p["capacity_factor"], p["seed"]): p for p in points})
    save_json(index_path, sanitize({
        "schema_version": 1, "config_name": base["name"],
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "points": [merged[k] for k in sorted(merged)]}))

    rows = []
    for p in points:
        f = load_json(index_path.parent.parent.parent / p["run_dir"] / "final.json")
        rows.append({"cf": f"{p['capacity_factor']:g}", "seed": p["seed"], "val_loss": fmt(f["final_val_loss"]),
                     "val_drop": fmt(f["val_drop_fraction"]), "train_drop": fmt(f["train_drop_fraction_last"]),
                     "tok/s": fmt(f["median_tokens_per_sec"], ",.0f"), "peak_mem_MiB": fmt(f["peak_mem_mb"], ".0f"),
                     "from": p.get("reused_from", "e2_capacity")})
    print(f"\nE2 capacity sweep ({base['name']}), total wall time {time.perf_counter() - t_all:.1f} s")
    print(format_table(rows, ["cf", "seed", "val_loss", "val_drop", "train_drop", "tok/s", "peak_mem_MiB", "from"]))
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
