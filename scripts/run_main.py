"""E1: train the model variants with identical settings and print a summary table.

    python scripts/run_main.py --config configs/main.yaml                       # dense switch deepseek
    python scripts/run_main.py --config configs/main.yaml --include-optional    # + gshard dense_x8
    python scripts/run_main.py --config configs/main.yaml --variants switch --seeds 1 2 3 --skip-existing

With ``--config configs/smoke.yaml`` and no ``--variants``, all 5 variants run so the smoke
pipeline exercises every code path. Runs go in-process through ``scripts.train.train``.
Output: ``results/<config>/e1_main/<run_id>/`` (DASHBOARD_SPEC 1.4).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import find_runs, fmt, rel_to_results, run_group, seeded  # noqa: E402
from scripts.train import train  # noqa: E402

from moe.utils import VARIANTS, format_table, load_config, load_json  # noqa: E402

MAIN_VARIANTS = ["dense", "switch", "deepseek"]
OPTIONAL_VARIANTS = ["gshard", "dense_x8"]


def summary_rows(run_dirs: list[Path]) -> list[dict]:
    """Table rows read back from each run's final.json."""
    rows = []
    for d in run_dirs:
        f = load_json(d / "final.json")
        rows.append({"variant": f["variant"], "seed": f["seed"], "val_loss": fmt(f["final_val_loss"]),
                     "ppl": fmt(f["final_val_ppl"], ".2f"), "tok/s": fmt(f["median_tokens_per_sec"], ",.0f"),
                     "peak_mem_MiB": fmt(f["peak_mem_mb"], ".0f"), "run": d.name})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), help="default: dense switch deepseek "
                    "(all 5 for the smoke config)")
    ap.add_argument("--seeds", nargs="+", type=int, help="default: [cfg.seed]")
    ap.add_argument("--include-optional", action="store_true", help="also run gshard and dense_x8")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (variant, seed) if a completed run with the same group and config exists")
    ap.add_argument("--override", action="append", default=[], metavar="a.b=v", help="repeatable")
    ap.add_argument("--tag", help="extra label appended to run_id and group")
    ap.add_argument("--max-steps", type=int, help="debug: stop each run early (no final.json)")
    args = ap.parse_args()

    base = load_config(args.config, args.override)
    variants = args.variants or (list(VARIANTS) if base["name"] == "smoke" else list(MAIN_VARIANTS))
    if args.include_optional:
        variants += [v for v in OPTIONAL_VARIANTS if v not in variants]
    seeds = args.seeds or [int(base["seed"])]

    t_all = time.perf_counter()
    done: list[Path] = []
    for seed in seeds:
        for variant in variants:
            cfg = seeded(base, seed, variant)
            if args.skip_existing:
                existing = find_runs(cfg, "e1_main", run_group(cfg, "e1_main", args.tag), seed, completed=True)
                if existing:
                    print(f"skip {variant} seed={seed}: completed run {existing[-1].name}")
                    done.append(existing[-1])
                    continue
            print(f"\n=== E1: {variant} seed={seed} ===")
            t0 = time.perf_counter()
            run_dir = train(cfg, "e1_main", tag=args.tag, max_steps=args.max_steps,
                            overrides=args.override, verbose=False)
            print(f"=== {variant} seed={seed} took {time.perf_counter() - t0:.1f} s ===")
            if (run_dir / "final.json").exists():
                done.append(run_dir)

    print(f"\nE1 summary ({base['name']}), total wall time {time.perf_counter() - t_all:.1f} s")
    print(format_table(summary_rows(done), ["variant", "seed", "val_loss", "ppl", "tok/s", "peak_mem_MiB", "run"]))


if __name__ == "__main__":
    main()
