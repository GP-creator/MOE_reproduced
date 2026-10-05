"""Shared helpers for the experiment scripts (train / run_main / sweep_capacity).

Importing this module puts the repo root on ``sys.path`` so ``import moe`` works no matter how
the script is launched. Everything here is plain file/JSON bookkeeping; no model code.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe.utils import _to_jsonable  # noqa: E402  (needs the sys.path line above)


def results_root(cfg: dict) -> Path:
    """Absolute path of the ``results/`` directory for this config."""
    return REPO_ROOT / cfg["system"]["results_dir"]


def experiment_dir(cfg: dict, experiment: str) -> Path:
    """``results/<config_name>/<experiment>`` (DASHBOARD_SPEC 1.1)."""
    return results_root(cfg) / cfg["name"] / experiment


def rel_to_results(cfg: dict, path: Path) -> str:
    """Path relative to ``results/`` (what goes inside JSON files; DASHBOARD_SPEC 1.3)."""
    return str(Path(path).resolve().relative_to(results_root(cfg).resolve()))


def capacity_factor(cfg: dict) -> float | None:
    """Training capacity factor for switch/gshard, else None (dense, dense_x8, deepseek)."""
    variant = cfg["variant"]
    if variant in ("switch", "gshard"):
        return float(cfg["moe"][variant]["capacity_factor"])
    return None


def run_group(cfg: dict, experiment: str, tag: str | None = None) -> str:
    """Group key: runs that differ only by seed share it (DASHBOARD_SPEC 1.4 run.json).

    E1 -> ``switch``; E2 -> ``switch_cf0.75``. A user ``--tag`` is appended so tagged
    experiments never aggregate with the untagged ones.
    """
    group = cfg["variant"]
    if experiment == "e2_capacity":
        group = f"{group}_cf{capacity_factor(cfg):g}"
    return group + (f"_{tag}" if tag else "")


def sanitize(obj: Any) -> Any:
    """Recursively convert tensors/numpy to plain Python and NaN/Inf to None (valid JSON)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return sanitize(_to_jsonable(obj))


def read_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file, ignoring a trailing partial line; later duplicates of a ``step`` win."""
    rows: list[dict] = []
    if not Path(path).exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedupe_by_step(rows: list[dict]) -> list[dict]:
    """Keep the last record per ``step`` (resume appends), sorted by step."""
    by_step: dict[int, dict] = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]


def same_config(cfg_a: dict, cfg_b: dict) -> bool:
    """True if two resolved configs are identical after a JSON round trip."""
    def norm(c: dict) -> Any:
        return json.loads(json.dumps(c, default=_to_jsonable, sort_keys=True))
    return norm(cfg_a) == norm(cfg_b)


def find_runs(cfg: dict, experiment: str, group: str, seed: int, completed: bool | None = None,
              match_config: bool = True) -> list[Path]:
    """Run dirs under results/<cfg>/<experiment>/ with this group and seed, oldest first.

    completed=True keeps runs with final.json, False keeps those without, None keeps both.
    With match_config the stored config.json must equal ``cfg`` (so an override changes identity).
    """
    exp_dir = experiment_dir(cfg, experiment)
    found: list[Path] = []
    if not exp_dir.is_dir():
        return found
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        run_json, config_json = d / "run.json", d / "config.json"
        if not run_json.exists() or not config_json.exists():
            continue
        try:
            meta = json.loads(run_json.read_text())
            stored = json.loads(config_json.read_text())
        except json.JSONDecodeError:
            continue
        if meta.get("group") != group or meta.get("seed") != seed:
            continue
        if completed is not None and (d / "final.json").exists() != completed:
            continue
        if match_config and not same_config(stored, cfg):
            continue
        found.append(d)
    return found


def seeded(cfg: dict, seed: int, variant: str | None = None) -> dict:
    """Deep copy of cfg with a new seed (and optionally variant)."""
    out = copy.deepcopy(cfg)
    out["seed"] = int(seed)
    if variant is not None:
        out["variant"] = variant
    return out


def fmt(v: Any, spec: str = ".4f") -> str:
    """Format a number for a summary table; None -> '-'."""
    return "-" if v is None else format(v, spec)
