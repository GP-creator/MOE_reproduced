"""Shared helpers for the eval-only scripts (ablate_experts.py, route_trace.py).

Run discovery follows DASHBOARD_SPEC §1: ``results/<cfg>/e1_main/<run_id>/`` with ``run.json``,
``final.json`` (= complete), ``checkpoint.pt`` and ``config.json``. Identity comes from
``run.json``, never from parsing the run_id; the run_id is only a sort key (timestamp prefix).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from moe import data as moe_data  # noqa: E402
from moe import utils  # noqa: E402
from moe.layer_api import MOE_VARIANTS  # noqa: E402
from moe.model import MoETransformer  # noqa: E402


def results_root(cfg: dict, results_dir: str | None) -> Path:
    """Absolute results root: --results-dir if given, else <repo>/<system.results_dir>."""
    base = Path(results_dir) if results_dir else Path(cfg["system"]["results_dir"])
    return (base if base.is_absolute() else REPO_ROOT / base).resolve()


def rel_to(root: Path, path: Path) -> str:
    """Path relative to the results root (spec §1.3: paths inside files are relative)."""
    try:
        return str(Path(path).resolve().relative_to(root))
    except ValueError:
        return str(path)


def _describe(run_dir: Path) -> dict | None:
    """Run record from run.json, or None if the run is incomplete / has no checkpoint."""
    if not (run_dir / "final.json").exists() or not (run_dir / "checkpoint.pt").exists():
        return None
    meta = utils.load_json(run_dir / "run.json")
    return {"run_dir": run_dir, "run_id": run_dir.name, "variant": meta["variant"],
            "seed": int(meta["seed"]), "config_name": meta.get("config_name")}


def find_runs(root: Path, cfg_name: str, variants: tuple[str, ...] = MOE_VARIANTS,
              run_dirs: list[str] | None = None) -> list[dict]:
    """Complete runs to evaluate, one per variant.

    ``run_dirs`` given: exactly those directories (any variant in ``variants``).
    Otherwise: the latest complete run (by run_id timestamp) per variant in
    ``<root>/<cfg_name>/e1_main``. Variants with no complete run are skipped with a note.
    """
    if run_dirs:
        runs = []
        for d in run_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = REPO_ROOT / p if (REPO_ROOT / p).exists() else root / p
            rec = _describe(p)
            if rec is None:
                raise FileNotFoundError(f"{d}: needs final.json + checkpoint.pt + run.json (complete run)")
            if rec["variant"] in variants:
                runs.append(rec)
        return runs
    base = root / cfg_name / "e1_main"
    best: dict[str, dict] = {}
    for d in sorted(base.glob("*")) if base.exists() else []:
        rec = _describe(d) if d.is_dir() else None
        if rec and rec["variant"] in variants:
            best[rec["variant"]] = rec          # sorted ascending, so the last one wins
    for v in variants:
        if v not in best:
            print(f"[note] no complete {v} run under {base}; skipping")
    return [best[v] for v in variants if v in best]


def load_run_model(run: dict, device: torch.device) -> tuple[MoETransformer, dict]:
    """Build the model from the run's own config.json, load checkpoint weights, eval mode."""
    cfg = utils.load_json(run["run_dir"] / "config.json")
    model = MoETransformer(cfg)
    utils.load_checkpoint(run["run_dir"] / "checkpoint.pt", model)
    return model.to(device).eval(), cfg


def val_batches(cfg: dict, n_batches: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """First ``n_batches`` fixed validation batches (seed-independent, identical for every model)."""
    info = moe_data.prepare_data(cfg["data"], REPO_ROOT)
    batcher = moe_data.TokenBatcher(info["val_bin"], int(cfg["model"]["seq_len"]),
                                    int(cfg["train"]["batch_size"]), int(cfg["seed"]),
                                    max_tokens=int(cfg["data"]["max_val_tokens"]))
    n = min(n_batches, batcher.n_windows // batcher.batch_size)
    if n < n_batches:
        print(f"[warn] only {n} val batches available (asked {n_batches})")
    return batcher.get_eval_batches(n, device)


def moe_layers(model: MoETransformer) -> list:
    """The MoE FFN modules (those declaring ABLATION_ATTRS), in block order."""
    return [m for m in model.modules() if getattr(type(m), "ABLATION_ATTRS", ())]
