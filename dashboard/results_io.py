"""Pure loaders and discovery for ``results/`` (DASHBOARD_SPEC §1, §2.6).

NO streamlit import here: ``scripts/make_report.py`` uses this module directly, and
``dashboard/data.py`` wraps these functions with ``st.cache_data``.

Conventions implemented here (spec §1.2-§1.3):
  * Layout ``<root>/<config_name>/<experiment>/<run_id>/`` (config name = first path part).
  * The results root is ``$MOE_RESULTS_DIR`` if set, else ``<repo>/results`` (read on every
    call through :func:`results_root`, so tests can switch it in-process).
  * A training run is complete iff ``final.json`` exists. Identity comes from ``run.json``
    (never from parsing the run_id); ``run_id`` is only a display string and sort key
    (timestamp prefix -> lexical order = chronological order).
  * ``.jsonl`` readers skip lines that fail to parse (e.g. a trailing line being written) and
    de-duplicate by ``step`` (by ``(step, layer)`` when a ``layer`` column exists), keeping
    the LAST occurrence, because ``train.py`` appends again after a resume.
  * Paths stored inside files and the ``run_dir`` strings returned here are relative to the
    results root (e.g. ``"smoke/e1_main/20261004-161717_dense_s1337"``).
  * Seed aggregation groups runs by ``run.json["group"]``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

CONFIG_NAMES: tuple[str, ...] = ("smoke", "small", "main")
EXPERIMENTS: tuple[str, ...] = ("e1_main", "e2_capacity", "e3e4_ablation", "e5_profile", "traces", "e6_placement")
TRAINING_EXPERIMENTS: tuple[str, ...] = ("e1_main", "e2_capacity")

EXPERIMENT_TITLE: dict[str, str] = {
    "e1_main": "E1 training runs",
    "e2_capacity": "the E2 capacity sweep",
    "e3e4_ablation": "the E3/E4 ablations",
    "e5_profile": "the E5 systems profile",
    "traces": "routing traces",
    "e6_placement": "the E6 placement model",
}

#: Metadata file that identifies a run dir, per experiment (spec §1.4-§1.8).
META_FILE: dict[str, str] = {
    "e1_main": "run.json",
    "e2_capacity": "run.json",
    "e3e4_ablation": "ablation_meta.json",
    "e5_profile": "profile_meta.json",
    "traces": "meta.json",
    "e6_placement": "placement_params.json",
}

_SMALL = ".venv/bin/python scripts/{script} --config configs/small.yaml"
#: Spec §2.5 empty-state hint table: experiment -> {config_name: command}.
MAKE_HINT: dict[str, dict[str, str]] = {
    "e1_main": {"main": "make main", "small": "make small", "smoke": "make smoke"},
    "e2_capacity": {"main": "make sweep", "small": _SMALL.format(script="sweep_capacity.py"), "smoke": "make smoke"},
    "e3e4_ablation": {"main": "make ablate", "small": _SMALL.format(script="ablate_experts.py"), "smoke": "make smoke"},
    "e5_profile": {"main": "make profile", "small": _SMALL.format(script="profile_layer.py"), "smoke": "make smoke"},
    "traces": {"main": "make trace", "small": _SMALL.format(script="route_trace.py"), "smoke": "make smoke"},
    "e6_placement": {"main": "make placement (needs traces: make trace)",
                     "small": _SMALL.format(script="placement_sim.py"), "smoke": "make smoke"},
}


class NoData(Exception):
    """Raised by figure builders / loaders when their input is empty (spec §6).

    Pages turn it into ``layout.empty_state``; the static report into "not available: <msg>".
    """


def make_hint(experiment: str, cfg_name: Optional[str]) -> str:
    """Command that produces ``experiment`` for ``cfg_name`` (spec §2.5)."""
    row = MAKE_HINT.get(experiment, {})
    return row.get(cfg_name or "", row.get("smoke", "make smoke"))


# ----------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------

def results_root() -> Path:
    """``$MOE_RESULTS_DIR`` (absolute, or relative to the repo) else ``<repo>/results``."""
    env = os.environ.get("MOE_RESULTS_DIR")
    if env:
        p = Path(env)
        return (p if p.is_absolute() else REPO_ROOT / p).resolve()
    return REPO_ROOT / "results"


#: Import-time value (spec §2.6 name). Code should prefer :func:`results_root`.
RESULTS_ROOT: Path = results_root()


def resolve(run_dir: str | Path) -> Path:
    """Absolute path of a run dir given relative to the results root (or already absolute)."""
    p = Path(run_dir)
    return p if p.is_absolute() else results_root() / p


def rel(path: str | Path) -> str:
    """Path relative to the results root (falls back to the absolute path string)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(results_root().resolve()))
    except ValueError:
        return str(p)


# ----------------------------------------------------------------------------------------
# File readers
# ----------------------------------------------------------------------------------------

def read_json(path: str | Path) -> Optional[dict]:
    """One JSON object, or None if the file is missing or not valid JSON (being written)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def read_jsonl(path: str | Path, dedup: Any = "auto") -> pd.DataFrame:
    """Records of a ``.jsonl`` file as a DataFrame (empty DataFrame if missing).

    Lines that fail to parse are skipped (spec §1.3: a trailing partial line while a writer
    appends). ``dedup``: ``"auto"`` = by ``(step, layer)`` if both columns exist, else by
    ``step`` if it exists, else no de-dup; or an explicit tuple of columns; or None. The LAST
    occurrence wins (resume appends to the same file); result is sorted by those keys.
    """
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except (FileNotFoundError, NotADirectoryError):
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if dedup == "auto":
        keys: tuple[str, ...] = tuple(c for c in ("step", "layer") if c in df.columns) if "step" in df.columns else ()
    else:
        keys = tuple(dedup or ())
    if keys:
        df = df.drop_duplicates(subset=list(keys), keep="last").sort_values(list(keys), kind="stable")
        df = df.reset_index(drop=True)
    return df


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """All arrays of an ``.npz`` file, loaded eagerly (empty dict if missing)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (FileNotFoundError, NotADirectoryError):
        return {}


def file_signature(path: str | Path) -> Optional[tuple[str, int, int]]:
    """``(path, mtime_ns, size)`` cache key for one file (None if missing)."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def dir_signature(path: str | Path, pattern: str = "*") -> tuple:
    """Cache key for a directory: ``(name, mtime_ns, size)`` of its matching files."""
    p = Path(path)
    if not p.is_dir():
        return ()
    out = []
    for f in sorted(p.glob(pattern)):
        try:
            st = f.stat()
        except OSError:
            continue
        out.append((f.name, st.st_mtime_ns, st.st_size))
    return tuple(out)


def experiment_signature(cfg_name: str, experiment: str) -> tuple:
    """Cache key for run discovery in one experiment dir (spec §2.6).

    Entry names + mtimes, plus the meta / final.json stats of each run, so a run turning
    complete (final.json appears) or rewriting run.json invalidates the cache.
    """
    base = results_root() / cfg_name / experiment
    if not base.is_dir():
        return (str(base),)
    sig: list = [str(base)]
    meta = META_FILE.get(experiment, "run.json")
    for d in sorted(base.iterdir()):
        try:
            sig.append((d.name, d.stat().st_mtime_ns,
                        file_signature(d / meta), file_signature(d / "final.json"),
                        file_signature(d / "checkpoint.pt")))
        except OSError:
            continue
    return tuple(sig)


# ----------------------------------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunInfo:
    """One run directory. Identity fields come from the experiment's meta file (run.json etc.).

    ``run_dir`` is relative to the results root; ``complete`` = final.json exists (training
    runs) or the meta file exists (other experiments, which write it once at the start but
    are only listed when their main data file exists, see :func:`list_runs`).
    """

    run_dir: str
    run_id: str
    experiment: str
    config_name: str
    variant: Optional[str] = None
    seed: Optional[int] = None
    group: Optional[str] = None
    capacity_factor: Optional[float] = None
    status: Optional[str] = None
    started_at: Optional[str] = None
    complete: bool = False
    has_checkpoint: bool = False
    meta: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    @property
    def path(self) -> Path:
        return resolve(self.run_dir)

    @property
    def label(self) -> str:
        """Sidebar label ``"{variant} · s{seed} · {run_id[:15]}"`` (+ " (incomplete)")."""
        parts = [self.variant or self.experiment]
        if self.seed is not None:
            parts.append(f"s{self.seed}")
        parts.append(self.run_id[:15])
        s = " · ".join(parts)
        return s if self.complete else s + " (incomplete)"


#: Main data file whose presence marks a non-training run as usable.
_DATA_FILE: dict[str, str] = {
    "e3e4_ablation": "ablation.jsonl",
    "e5_profile": "bench.jsonl",
    "traces": "trace.npz",
    "e6_placement": "placement_summary.jsonl",
}


def list_configs() -> list[str]:
    """Config names (spec order smoke, small, main) that exist as dirs under the root."""
    root = results_root()
    return [c for c in CONFIG_NAMES if (root / c).is_dir()]


def _run_info(d: Path, cfg_name: str, experiment: str) -> Optional[RunInfo]:
    meta = read_json(d / META_FILE.get(experiment, "run.json")) or {}
    if experiment in TRAINING_EXPERIMENTS:
        if not meta:
            # run.json not written yet (or mid-write): fall back to config.json for identity.
            cfg = read_json(d / "config.json")
            if cfg is None:
                return None
            meta = {"variant": cfg.get("variant"), "seed": cfg.get("seed"), "group": cfg.get("variant")}
        complete = (d / "final.json").exists()
    else:
        if not meta and not (d / _DATA_FILE.get(experiment, "")).exists():
            return None
        complete = (d / _DATA_FILE.get(experiment, "")).exists()
    seed = meta.get("seed")
    return RunInfo(
        run_dir=f"{cfg_name}/{experiment}/{d.name}",
        run_id=d.name,
        experiment=experiment,
        config_name=meta.get("config_name", cfg_name),
        variant=meta.get("variant"),
        seed=int(seed) if seed is not None else None,
        group=meta.get("group", meta.get("variant")),
        capacity_factor=meta.get("capacity_factor"),
        status=meta.get("status"),
        started_at=meta.get("started_at", meta.get("created_at")),
        complete=complete,
        has_checkpoint=(d / "checkpoint.pt").exists(),
        meta=meta,
    )


def list_runs(cfg_name: str, experiment: str) -> list[RunInfo]:
    """All run dirs of ``experiment`` under ``cfg_name``, oldest first (run_id order)."""
    base = results_root() / cfg_name / experiment
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            info = _run_info(d, cfg_name, experiment)
            if info is not None:
                out.append(info)
    return out


def latest(runs: Iterable[RunInfo], key: Optional[Callable[[RunInfo], Any]] = None,
           where: Optional[Callable[[RunInfo], bool]] = None) -> Optional[RunInfo]:
    """Newest run (max run_id, or max ``key``) among those passing ``where``; None if none."""
    pool = [r for r in runs if where is None or where(r)]
    if not pool:
        return None
    return max(pool, key=key or (lambda r: r.run_id))


def default_selection(runs: Sequence[RunInfo]) -> list[str]:
    """Spec §2.3 default: for each (variant, seed), the latest complete run (run_dir strings)."""
    best: dict[tuple, RunInfo] = {}
    for r in runs:
        if r.complete:
            k = (r.variant, r.seed)
            if k not in best or r.run_id > best[k].run_id:
                best[k] = r
    return [r.run_dir for r in sorted(best.values(), key=lambda r: r.run_id)]


def newest_env(cfg_name: str) -> Optional[dict]:
    """env.json of the newest run (by run_id) in any experiment of ``cfg_name``."""
    newest: Optional[tuple[str, Path]] = None
    for exp in EXPERIMENTS:
        base = results_root() / cfg_name / exp
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if (d / "env.json").exists() and (newest is None or d.name > newest[0]):
                newest = (d.name, d)
    return read_json(newest[1] / "env.json") if newest else None


# ----------------------------------------------------------------------------------------
# Loaders (each returns plain data; missing files -> None / empty DataFrame)
# ----------------------------------------------------------------------------------------

@dataclass
class RunData:
    """Everything in one E1/E2 training run dir (spec §1.4). DataFrames may be empty."""

    info: RunInfo
    config: Optional[dict]
    env: Optional[dict]
    final: Optional[dict]
    train: pd.DataFrame
    eval: pd.DataFrame
    routing: pd.DataFrame

    @property
    def variant(self) -> Optional[str]:
        return self.info.variant


def run_info_for(run_dir: str | Path) -> Optional[RunInfo]:
    """RunInfo for one run dir (relative to the root or absolute)."""
    p = resolve(run_dir)
    parts = Path(rel(p)).parts
    if len(parts) < 3:
        return None
    return _run_info(p, parts[-3], parts[-2])


def load_run(run_dir: str | Path) -> Optional[RunData]:
    """Load a training run dir. None if the dir has no identity (no run.json/config.json)."""
    p = resolve(run_dir)
    info = run_info_for(p)
    if info is None:
        return None
    return RunData(
        info=info,
        config=read_json(p / "config.json"),
        env=read_json(p / "env.json"),
        final=read_json(p / "final.json"),
        train=read_jsonl(p / "train_log.jsonl"),
        eval=read_jsonl(p / "eval_log.jsonl"),
        routing=read_jsonl(p / "routing_log.jsonl"),
    )


def load_sweep_index(cfg_name: str) -> Optional[dict]:
    """``e2_capacity/sweep_index.json`` (spec §1.4), or None (pages fall back to list_runs)."""
    return read_json(results_root() / cfg_name / "e2_capacity" / "sweep_index.json")


def load_ablation(run_dir: str | Path) -> dict[str, Any]:
    """E3/E4 dir (spec §1.5): ``{"meta", "env", "rows"}``."""
    p = resolve(run_dir)
    return {"meta": read_json(p / "ablation_meta.json"), "env": read_json(p / "env.json"),
            "rows": read_jsonl(p / "ablation.jsonl", dedup=None)}


def load_profile(run_dir: str | Path) -> dict[str, Any]:
    """E5 dir (spec §1.6): ``{"meta", "env", "bench", "gemm", "breakdown", "roofline", "trace_file"}``."""
    p = resolve(run_dir)
    meta = read_json(p / "profile_meta.json")
    trace = (meta or {}).get("trace_file") or "profiler_trace.json"
    trace_path = Path(trace) if Path(trace).is_absolute() else p / Path(trace).name
    return {"meta": meta, "env": read_json(p / "env.json"),
            "bench": read_jsonl(p / "bench.jsonl", dedup=None),
            "gemm": read_jsonl(p / "gemm.jsonl", dedup=None),
            "breakdown": read_jsonl(p / "breakdown.jsonl", dedup=None),
            "roofline": read_json(p / "roofline.json"),
            "trace_file": trace_path if trace_path.exists() else None}


def load_trace(trace_dir: str | Path) -> dict[str, Any]:
    """Routing trace dir (spec §1.7): ``{"meta", "arrays"}`` (arrays: topk_idx, gates, kept)."""
    p = resolve(trace_dir)
    return {"meta": read_json(p / "meta.json"), "arrays": read_npz(p / "trace.npz")}


def load_placement(run_dir: str | Path) -> dict[str, Any]:
    """E6 dir (spec §1.8): ``{"params", "env", "summary", "arrays_path"}``.

    The npz is NOT parsed here: pages call ``scripts.placement_sim.load_placement_arrays``
    (spec §1.8, the dashboard never re-implements the model) on ``arrays_path``.
    """
    p = resolve(run_dir)
    npz = p / "placement_arrays.npz"
    return {"params": read_json(p / "placement_params.json"), "env": read_json(p / "env.json"),
            "summary": read_jsonl(p / "placement_summary.jsonl", dedup=None),
            "arrays_path": npz if npz.exists() else None}


# ----------------------------------------------------------------------------------------
# Summary helpers shared by pages and make_report.py (spec §6).
# BUILDER-OWNED SECTION: add summarize_e1 / summarize_e2 / summarize_e3e4 /
# summarize_e5_speedups / summarize_e6 / param_table_with_checks here (pure pandas, no
# streamlit). Everything above this line is architect-owned; ask before changing it.
# ----------------------------------------------------------------------------------------


MOE_VARIANTS: tuple[str, ...] = ("switch", "deepseek", "gshard")


def _final_of(r: "RunData") -> Optional[dict]:
    return r.final if r is not None and r.final else None


def complete_runs(runs: Sequence["RunData"]) -> list["RunData"]:
    """Runs with a final.json (spec §1.2: headline numbers and tables skip incomplete runs)."""
    return [r for r in runs if _final_of(r) is not None]


def latest_final_by_variant(runs: Sequence["RunData"]) -> dict[str, dict]:
    """``{variant: final.json}`` of the newest complete run (by run_id) of each variant."""
    best: dict[str, "RunData"] = {}
    for r in complete_runs(runs):
        v = r.variant or "?"
        if v not in best or r.info.run_id > best[v].info.run_id:
            best[v] = r
    return {v: r.final for v, r in best.items()}


def distinct_configs(runs: Sequence["RunData"]) -> list[tuple[dict, list["RunData"]]]:
    """Group runs by their (model, moe, data) config sections (spec O3: one table per config)."""
    groups: list[tuple[str, dict, list]] = []
    for r in runs:
        if not r.config:
            continue
        key = json.dumps({k: r.config.get(k) for k in ("model", "moe", "data")}, sort_keys=True)
        for g in groups:
            if g[0] == key:
                g[2].append(r)
                break
        else:
            groups.append((key, r.config, [r]))
    return [(c, rs) for _, c, rs in groups]


def default_n_experts(cfg: dict) -> dict[str, int]:
    """Routed-expert count per MoE variant from a config (``ffn_spec``), for E5 default filters."""
    from moe import layer_api
    out: dict[str, int] = {}
    for v in MOE_VARIANTS:
        try:
            out[v] = layer_api.ffn_spec(v, cfg).n_routed
        except Exception:  # noqa: BLE001  (config lacks that variant's section)
            pass
    return out


def param_table_with_checks(cfg: dict, finals: dict[str, dict]) -> pd.DataFrame:
    """O3: ``moe.flops.param_table(cfg)`` + measured params + the two check columns.

    ``finals`` = ``{variant: final.json}``. ``matched`` follows spec O3 (expert-FFN totals and
    active widths against dense; the multiple comes from ``cfg["moe"]["dense_x8"]["width_mult"]``).
    ``measured = analytic`` compares ``final.json.total_params`` with the flops.py total.
    """
    from moe import flops
    rows = flops.param_table(cfg)
    mult = int(cfg["moe"]["dense_x8"]["width_mult"])
    dense = next((r for r in rows if r["variant"] == "dense"), None)
    out = []
    for r in rows:
        v = r["variant"]
        fin = finals.get(v)
        measured = fin.get("total_params") if fin else None
        if v == "dense":
            matched = "baseline"
        elif dense is None:
            matched = DASH_
        elif v == "dense_x8":
            ok = r["ffn_total_model"] == mult * dense["ffn_total_model"]
            matched = "yes (active = %d× by design)" % mult if ok else "NO"
        else:
            ok = (r["ffn_total_model"] == mult * dense["ffn_total_model"]
                  and r["ffn_active_model"] == dense["ffn_active_model"])
            matched = "yes" if ok else "NO"
        out.append({**r, "total_params_measured": measured, "matched": matched,
                    "measured_eq_analytic": DASH_ if measured is None
                    else ("yes" if int(measured) == int(r["total_params"]) else "NO")})
    return pd.DataFrame(out)


DASH_ = "—"


def summarize_e1(runs: Sequence["RunData"]) -> pd.DataFrame:
    """O4 / RESULTS: one row per seed group (complete runs only), sorted in variant order.

    Columns: group, variant, n_seeds, final_val_loss (mean), val_loss_min/max/half_range,
    final_val_ppl, delta_vs_dense (mean - dense mean, None without a dense group),
    median_tokens_per_sec, peak_mem_mb, final_train_ce, seeds (list).
    """
    from dashboard import theme
    groups: dict[str, list[dict]] = {}
    for r in complete_runs(runs):
        groups.setdefault(r.info.group or r.variant or r.info.run_id, []).append(r.final)
    rows = []
    for g, fins in groups.items():
        vl = [f["final_val_loss"] for f in fins if f.get("final_val_loss") is not None]
        if not vl:
            continue
        mean = lambda k: (float(np.mean([f[k] for f in fins if f.get(k) is not None]))  # noqa: E731
                          if any(f.get(k) is not None for f in fins) else None)
        rows.append({"group": g, "variant": fins[0].get("variant"), "n_seeds": len(fins),
                     "final_val_loss": float(np.mean(vl)), "val_loss_min": min(vl), "val_loss_max": max(vl),
                     "half_range": (max(vl) - min(vl)) / 2, "final_val_ppl": mean("final_val_ppl"),
                     "median_tokens_per_sec": mean("median_tokens_per_sec"),
                     "peak_mem_mb": mean("peak_mem_mb"), "final_train_ce": mean("final_train_ce"),
                     "seeds": [f.get("seed") for f in fins],
                     "run_ids": [f.get("run_id") for f in fins]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dense = df[df["variant"] == "dense"]
    base = float(dense["final_val_loss"].iloc[0]) if len(dense) else None
    df["delta_vs_dense"] = [None if base is None else v - base for v in df["final_val_loss"]]
    df["_k"] = [theme_key(v) for v in df["variant"]]
    return df.sort_values(["_k", "group"]).drop(columns="_k").reset_index(drop=True)


def theme_key(variant: Optional[str]) -> tuple[int, str]:
    from dashboard import theme
    return theme.variant_sort_key(variant)


def seed_spread_table(runs: Sequence["RunData"]) -> pd.DataFrame:
    """Per group: final_val_loss of each seed and spread (max - min). Groups with 1 seed included."""
    rows = []
    groups: dict[str, list["RunData"]] = {}
    for r in complete_runs(runs):
        groups.setdefault(r.info.group or r.variant or "?", []).append(r)
    for g, rs in groups.items():
        vals = {f"s{r.info.seed}": r.final["final_val_loss"] for r in rs}
        v = list(vals.values())
        rows.append({"group": g, "n_seeds": len(rs), **vals, "spread (max-min)": (max(v) - min(v)) if len(v) > 1 else None})
    return pd.DataFrame(rows)


def headline_numbers(summary: pd.DataFrame) -> dict[str, Any]:
    """O4 'best MoE vs dense' tile from :func:`summarize_e1`: variant, delta, or None."""
    if summary.empty or "delta_vs_dense" not in summary:
        return {"best_moe_variant": None, "best_moe_delta": None}
    moe = summary[summary["variant"].isin(MOE_VARIANTS) & summary["delta_vs_dense"].notna()]
    if moe.empty:
        return {"best_moe_variant": None, "best_moe_delta": None}
    i = moe["delta_vs_dense"].idxmin()
    return {"best_moe_variant": moe.loc[i, "variant"], "best_moe_delta": float(moe.loc[i, "delta_vs_dense"])}


def e2_runs(cfg_name: str, load: Callable[[str], Optional["RunData"]] = None) -> list["RunData"]:
    """E2 points as RunData (complete only): ``sweep_index.json`` first, else scan ``e2_capacity``.

    The cf=1.25 point is typically the E1 run reused through the index (spec §1.4). Each
    returned RunData's ``info.capacity_factor`` is the sweep cf (index value wins).
    """
    load = load or load_run
    idx = load_sweep_index(cfg_name)
    out: list["RunData"] = []
    if idx and idx.get("points"):
        for p in idx["points"]:
            rd = load(p["run_dir"])
            if rd is not None and rd.final:
                rd.info = replace_cf(rd.info, p.get("capacity_factor"))
                out.append(rd)
    else:
        for info in list_runs(cfg_name, "e2_capacity"):
            rd = load(info.run_dir)
            if rd is not None and rd.final:
                out.append(rd)
    return out


def replace_cf(info: "RunInfo", cf: Optional[float]) -> "RunInfo":
    import dataclasses
    return dataclasses.replace(info, capacity_factor=cf)


E2_METRICS = ("final_val_loss", "final_val_ppl", "val_drop_fraction", "train_drop_fraction_last",
              "median_tokens_per_sec", "mean_tokens_per_sec", "peak_mem_mb")


def summarize_e2(runs: Sequence["RunData"]) -> pd.DataFrame:
    """One row per capacity factor (mean over seeds, plus ``<m>_min`` / ``<m>_max``).

    Also ``capacity_per_expert`` = ``layer_api.expert_capacity(tokens_per_step, E, cf, k)`` from
    the run's config.json, and ``delta_loss_vs_ref`` against the E1 training cf (the point reused from e1_main) if present,
    else the largest cf.
    """
    by_cf: dict[float, list["RunData"]] = {}
    for r in runs:
        if r.final and r.info.capacity_factor is not None:
            by_cf.setdefault(float(r.info.capacity_factor), []).append(r)
    if not by_cf:
        return pd.DataFrame()
    from moe import layer_api
    rows = []
    for cf in sorted(by_cf):
        rs = by_cf[cf]
        row: dict[str, Any] = {"capacity_factor": cf, "n_seeds": len(rs),
                               "run_dirs": [r.info.run_dir for r in rs], "seeds": [r.info.seed for r in rs]}
        for m in E2_METRICS:
            vals = [r.final[m] for r in rs if r.final.get(m) is not None]
            row[m] = float(np.mean(vals)) if vals else None
            row[m + "_min"] = float(min(vals)) if vals else None
            row[m + "_max"] = float(max(vals)) if vals else None
        cfg = rs[0].config or {}
        try:
            n_tok = int(cfg["train"]["batch_size"]) * int(cfg["model"]["seq_len"])
            row["capacity_per_expert"] = layer_api.expert_capacity(n_tok, int(cfg["moe"]["switch"]["n_experts"]), cf, 1)
        except (KeyError, TypeError):
            row["capacity_per_expert"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    # Reference = the E1 training cf: the cf of the run reused from e1_main (sweep_index
    # "reused_from"); each E2 run's own config.json carries its own cf, so it cannot say.
    ref = next((float(r.info.capacity_factor) for r in runs if r.info.experiment == "e1_main"
                and r.info.capacity_factor is not None), None)
    ref_row = df[df["capacity_factor"] == ref] if ref is not None else df.iloc[0:0]
    if ref_row.empty:
        ref_row = df[df["capacity_factor"] == df["capacity_factor"].max()]
    ref_loss = float(ref_row["final_val_loss"].iloc[0])
    df["ref_capacity_factor"] = float(ref_row["capacity_factor"].iloc[0])
    df["delta_loss_vs_ref"] = df["final_val_loss"] - ref_loss
    return df


def summarize_e3e4(rows: pd.DataFrame) -> pd.DataFrame:
    """Compact E3/E4 table: study, variant, seed, condition/k/fraction, val loss, delta, drop."""
    if rows is None or rows.empty:
        return pd.DataFrame()
    cols = [c for c in ("study", "variant", "seed", "condition", "n_disable_top_routed", "frac_disabled",
                        "k_routed", "active_ffn_width", "capacity_factor", "is_reference", "val_loss",
                        "val_ppl", "delta_val_loss", "val_drop_fraction") if c in rows]
    return rows[cols].reset_index(drop=True)


def summarize_e5_speedups(bench: pd.DataFrame) -> pd.DataFrame:
    """Loop vs batched pairs: variant, sweep, N, E, pass, loop_ms, batched_ms, speedup = loop/batched."""
    if bench is None or bench.empty:
        return pd.DataFrame()
    keys = ["sweep", "variant", "pass", "n_tokens", "n_experts", "capacity_factor", "routing", "skew_strength"]
    keys = [k for k in keys if k in bench]
    b = bench[bench["dispatch"].isin(["loop", "batched"])].copy()
    b["_cf"] = b["capacity_factor"].fillna(-1) if "capacity_factor" in b else -1
    k2 = [("_cf" if k == "capacity_factor" else k) for k in keys]
    loop = b[b["dispatch"] == "loop"].set_index(k2)["median_ms"].rename("loop_ms")
    bat = b[b["dispatch"] == "batched"].set_index(k2)["median_ms"].rename("batched_ms")
    j = pd.concat([loop, bat], axis=1, join="inner").reset_index()
    if j.empty:
        return pd.DataFrame()
    j = j.rename(columns={"_cf": "capacity_factor"})
    j["speedup"] = j["loop_ms"] / j["batched_ms"]
    return j.sort_values([c for c in ("sweep", "variant", "pass", "n_experts", "n_tokens") if c in j]).reset_index(drop=True)


def summarize_e6(summary: pd.DataFrame) -> pd.DataFrame:
    """Rows with ``layer == "all"`` of placement_summary: straggler / bytes / step time per (trace, D, policy)."""
    if summary is None or summary.empty:
        return pd.DataFrame()
    s = summary[summary["layer"].astype(str) == "all"]
    cols = [c for c in ("variant", "D", "policy", "straggler_mean", "straggler_p50", "straggler_p90",
                        "straggler_max", "offdevice_tokens_mean", "a2a_bytes_fwd_mean",
                        "a2a_bytes_fwd_bwd_mean", "step_ms_default", "compute_ms_default",
                        "comm_ms_default") if c in s]
    return s[cols].sort_values([c for c in ("variant", "D", "policy") if c in cols]).reset_index(drop=True)


def placement_step_times(arrays: dict, params: dict, trace_idx: int, source_run_id: str, D: int,
                         policies: Sequence[str], layers: Optional[Sequence[int]], bw_grid: Sequence[float],
                         *, device_tflops: float, include_backward: bool) -> pd.DataFrame:
    """Live E6 curve data: ``placement_sim.estimate_step_time`` per (policy, bandwidth), mean over steps.

    Imported lazily; raises :class:`NoData` if ``scripts.placement_sim`` is unavailable.
    Columns: policy, bw, step_ms, compute_ms, compute_balanced_ms, imbalance_ms, comm_ms.
    """
    try:
        from scripts.placement_sim import estimate_step_time
    except ImportError as exc:
        raise NoData(f"scripts/placement_sim.py is not available ({exc}); run `make placement` once it exists")
    tr = params["traces"][trace_idx]
    rows = []
    for pol in policies:
        a = arrays.get((source_run_id, int(D), pol))
        if a is None:
            continue
        tpd, bout = a["tokens_per_device"], a["bytes_out_per_device"]
        if layers is not None:
            tpd, bout = tpd[:, list(layers), :], bout[:, list(layers), :]
        n_step_tokens = tr.get("tokens_per_step")
        for bw in bw_grid:
            est = estimate_step_time(tpd, bout, flops_per_assignment=tr["flops_per_assignment"],
                                     device_tflops=device_tflops, link_gbps=float(bw),
                                     include_backward=include_backward,
                                     shared_flops_per_token=tr.get("shared_flops_per_token", 0.0),
                                     tokens_per_step=n_step_tokens)
            rows.append({"policy": pol, "bw": float(bw), **{k: float(np.mean(est[k])) for k in
                         ("step_ms", "compute_ms", "compute_balanced_ms", "imbalance_ms", "comm_ms")}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# E5 loop -> batched speedup views (token sweep): headline at the training N, range over N
# ----------------------------------------------------------------------------------------

def train_tokens_per_step(cfg: dict) -> Optional[int]:
    """Training tokens per step = ``train.batch_size * model.seq_len`` from a config (None if missing)."""
    try:
        return int(cfg["train"]["batch_size"]) * int(cfg["model"]["seq_len"])
    except (KeyError, TypeError, ValueError):
        return None


def speedup_token_sweep(sp: pd.DataFrame, default_E: Optional[dict[str, int]] = None) -> pd.DataFrame:
    """Token-sweep rows of ``summarize_e5_speedups`` at each variant's config expert count."""
    if sp is None or sp.empty:
        return pd.DataFrame()
    t = sp[sp["sweep"] == "tokens"]
    if default_E:
        t = t[[(v not in default_E) or (e == default_E[v]) for v, e in zip(t["variant"], t["n_experts"])]]
    return t.sort_values(["variant", "pass", "n_tokens"]).reset_index(drop=True)


def speedup_headline(sp: pd.DataFrame, train_n: Optional[int], default_E: Optional[dict[str, int]] = None) -> pd.DataFrame:
    """Per (variant, pass): speedup at the training N and the range over the token sweep with the N at each end.

    Columns: variant, pass, train_n, speedup_train (NaN if train_n is not in the sweep), min_speedup, min_n,
    max_speedup, max_n. Variant order switch, gshard, deepseek.
    """
    t = speedup_token_sweep(sp, default_E)
    rows = []
    for v in ("switch", "gshard", "deepseek"):
        for ps in ("fwd", "fwd_bwd"):
            g = t[(t["variant"] == v) & (t["pass"] == ps)] if not t.empty else t
            if g.empty:
                continue
            lo, hi = g.loc[g["speedup"].idxmin()], g.loc[g["speedup"].idxmax()]
            at = g[g["n_tokens"] == train_n]
            rows.append({"variant": v, "pass": ps, "train_n": train_n,
                         "speedup_train": float(at["speedup"].iloc[0]) if len(at) else float("nan"),
                         "min_speedup": float(lo["speedup"]), "min_n": int(lo["n_tokens"]),
                         "max_speedup": float(hi["speedup"]), "max_n": int(hi["n_tokens"])})
    return pd.DataFrame(rows)


def speedup_headline_text(h: pd.DataFrame) -> list[str]:
    """One string per row of ``speedup_headline``, e.g. 'switch fwd: 1.46x at N=8,192 (training size); range 0.89x (N=65,536) - 2.84x (N=1,024)'."""
    out = []
    for _, r in h.iterrows():
        if np.isfinite(r["speedup_train"]):
            at = f"{r['speedup_train']:.2f}x at N={int(r['train_n']):,} (training size)"
        elif r["train_n"]:
            at = f"training size N={int(r['train_n']):,} not in the sweep"
        else:
            at = "training size unknown"
        out.append(f"{r['variant']} {r['pass']}: {at}; range {r['min_speedup']:.2f}x (N={r['min_n']:,}) "
                   f"to {r['max_speedup']:.2f}x (N={r['max_n']:,})")
    return out


def speedup_vs_n_table(sp: pd.DataFrame, default_E: Optional[dict[str, int]] = None) -> pd.DataFrame:
    """Wide table: one row per (variant, pass), one column per N of the token sweep, values = speedup (float)."""
    t = speedup_token_sweep(sp, default_E)
    if t.empty:
        return pd.DataFrame()
    w = t.pivot_table(index=["variant", "pass"], columns="n_tokens", values="speedup").reset_index()
    w["_o"] = w["variant"].map(theme_key)
    return w.sort_values(["_o", "pass"]).drop(columns="_o").reset_index(drop=True)


def ablation_eval_note(meta: Optional[dict], e1_eval_batches: Optional[int]) -> str:
    """Caveat: E3/E4 use a different number of val batches than E1 (``n_val_batches`` in ablation_meta.json)."""
    n = (meta or {}).get("n_val_batches")
    a = str(int(n)) if n is not None else "an unrecorded number of"
    b = str(e1_eval_batches) if e1_eval_batches is not None else "a different number of"
    return (f"E3/E4 evaluate on {a} validation batches, E1 on `train.eval_batches` = {b}; "
            "the E3 r=0 baseline therefore need not equal the E1 final val loss.")
