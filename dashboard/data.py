"""Streamlit cache wrappers around ``results_io`` (DASHBOARD_SPEC §2.6).

Pages read results ONLY through this module. Every cached function is keyed on the file
signature ``(path, mtime_ns, size)`` of what it reads (computed fresh on every rerun with a
cheap ``os.stat``), so a live training run refreshes on each rerun with no TTL, and nothing is
re-parsed when the files did not change. The results root is part of every key, so switching
``MOE_RESULTS_DIR`` never serves stale data.

Returned DataFrames / dicts are shared cache objects: treat them as read-only (copy first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from dashboard import results_io as rio

# ----------------------------------------------------------------------------------------
# Single files
# ----------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=512)
def _read_json_cached(path: str, mtime_ns: int, size: int) -> Optional[dict]:
    return rio.read_json(path)


@st.cache_data(show_spinner=False, max_entries=512)
def _read_jsonl_cached(path: str, mtime_ns: int, size: int, dedup: Any):
    return rio.read_jsonl(path, dedup=dedup)


def read_json(path: str | Path) -> Optional[dict]:
    sig = rio.file_signature(path)
    return None if sig is None else _read_json_cached(*sig)


def read_jsonl(path: str | Path, dedup: Any = "auto"):
    sig = rio.file_signature(path)
    if sig is None:
        return rio.read_jsonl(path)          # empty DataFrame
    return _read_jsonl_cached(*sig, dedup if dedup is None or isinstance(dedup, str) else tuple(dedup))


# ----------------------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------------------


def list_configs() -> list[str]:
    return rio.list_configs()                # a few stat calls; not worth caching


@st.cache_data(show_spinner=False, max_entries=128)
def _list_runs_cached(cfg_name: str, experiment: str, sig: tuple) -> list[rio.RunInfo]:
    return rio.list_runs(cfg_name, experiment)


def list_runs(cfg_name: Optional[str], experiment: str) -> list[rio.RunInfo]:
    """Runs of one experiment, oldest first. Empty list if ``cfg_name`` is None."""
    if not cfg_name:
        return []
    return _list_runs_cached(cfg_name, experiment, rio.experiment_signature(cfg_name, experiment))


def newest_env(cfg_name: Optional[str]) -> Optional[dict]:
    if not cfg_name:
        return None
    sig = tuple(rio.experiment_signature(cfg_name, e) for e in rio.EXPERIMENTS)
    return _newest_env_cached(cfg_name, sig)


@st.cache_data(show_spinner=False, max_entries=16)
def _newest_env_cached(cfg_name: str, sig: tuple) -> Optional[dict]:
    return rio.newest_env(cfg_name)


# ----------------------------------------------------------------------------------------
# Whole run directories (keyed on the signature of every file in the dir)
# ----------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=256)
def _load_run_cached(abs_dir: str, sig: tuple) -> Optional[rio.RunData]:
    return rio.load_run(abs_dir)


def load_run(run_dir: str | Path) -> Optional[rio.RunData]:
    p = rio.resolve(run_dir)
    return _load_run_cached(str(p), rio.dir_signature(p, "*.json*"))


def load_runs(run_dirs: list[str]) -> list[rio.RunData]:
    """Load several runs, silently skipping dirs that vanished or have no identity."""
    return [r for r in (load_run(d) for d in run_dirs) if r is not None]


@st.cache_data(show_spinner=False, max_entries=64)
def _load_dir_cached(kind: str, abs_dir: str, sig: tuple) -> dict:
    return {"ablation": rio.load_ablation, "profile": rio.load_profile,
            "trace": rio.load_trace, "placement": rio.load_placement}[kind](abs_dir)


def load_ablation(run_dir: str | Path) -> dict:
    p = rio.resolve(run_dir)
    return _load_dir_cached("ablation", str(p), rio.dir_signature(p))


def load_profile(run_dir: str | Path) -> dict:
    p = rio.resolve(run_dir)
    return _load_dir_cached("profile", str(p), rio.dir_signature(p))


def load_trace(trace_dir: str | Path) -> dict:
    p = rio.resolve(trace_dir)
    return _load_dir_cached("trace", str(p), rio.dir_signature(p))


def load_placement(run_dir: str | Path) -> dict:
    p = rio.resolve(run_dir)
    return _load_dir_cached("placement", str(p), rio.dir_signature(p))


def load_sweep_index(cfg_name: Optional[str]) -> Optional[dict]:
    if not cfg_name:
        return None
    return read_json(rio.results_root() / cfg_name / "e2_capacity" / "sweep_index.json")


# ----------------------------------------------------------------------------------------
# Models (How it works): st.cache_resource keyed on (checkpoint path, mtime_ns, device)
# ----------------------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading checkpoint…", max_entries=4)
def _load_model_cached(abs_dir: str, ckpt_mtime_ns: int, device: str):
    from dashboard import inference
    return inference.load_model(abs_dir, device)


def load_model(run_dir: str | Path, device: str = "cpu"):
    """(model, cfg) for a run dir with checkpoint.pt; shared across sessions (read-only use)."""
    p = rio.resolve(run_dir)
    sig = rio.file_signature(p / "checkpoint.pt")
    if sig is None:
        raise FileNotFoundError(f"{p / 'checkpoint.pt'} not found")
    return _load_model_cached(str(p), sig[1], device)


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_tokenizer_cached(path: str, mtime_ns: int, cfg_json: str):
    import json
    from dashboard import inference
    return inference.load_tokenizer(json.loads(cfg_json))


def load_tokenizer(cfg: dict):
    """Shared BPE tokenizer (``moe.data.load_tokenizer``), cached on the tokenizer file mtime."""
    import json
    from dashboard import inference
    path = inference.tokenizer_path(cfg)
    sig = rio.file_signature(path)
    if sig is None:
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    return _load_tokenizer_cached(str(path), sig[1], json.dumps({"data": cfg["data"]}, sort_keys=True))
