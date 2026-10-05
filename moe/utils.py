"""Shared utilities: config loading, seeding, device/dtype selection, run dirs, logging, checkpoints."""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
VARIANTS: tuple[str, ...] = ("dense", "switch", "deepseek", "gshard", "dense_x8")


# ----------------------------------------------------------------------------- config

def _set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg[a][b][c] = value for 'a.b.c'; the full key path must already exist."""
    keys = dotted.split(".")
    node: Any = cfg
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            where = ".".join(keys[:i]) or "<root>"
            avail = sorted(node.keys()) if isinstance(node, dict) else []
            raise KeyError(f"Unknown config key '{dotted}': '{key}' not found under '{where}' "
                           f"(available: {avail})")
        if i == len(keys) - 1:
            node[key] = value
        else:
            node = node[key]


def load_config(path: str, overrides: list[str] | None = None, variant: str | None = None) -> dict:
    """Load a YAML config, apply 'a.b.c=value' overrides, and optionally set the variant.

    Override values are parsed with yaml.safe_load (ints, floats, bools, null, lists).
    Unknown key paths and unknown variants raise.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' must look like 'section.key=value'")
        key, raw = ov.split("=", 1)
        _set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    if variant is not None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'; expected one of {list(VARIANTS)}")
        cfg["variant"] = variant
    return cfg


# ----------------------------------------------------------------------------- seeding / device

def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg: dict) -> torch.device:
    """Resolve system.device ('auto' = cuda -> mps -> cpu). Sets CPU threads when on cpu."""
    want = cfg["system"]["device"]
    if want == "auto":
        if torch.cuda.is_available():
            want = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            want = "mps"
        else:
            want = "cpu"
    device = torch.device(want)
    if device.type == "cpu":
        torch.set_num_threads(int(cfg["system"]["cpu_threads"]))
    return device


def get_autocast_dtype(cfg: dict, device: torch.device) -> torch.dtype | None:
    """None = fp32 (no autocast). 'auto' = bf16 on CUDA if supported. Explicit bf16 must be supported."""
    want = cfg["system"]["dtype"]
    bf16_ok = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if want == "fp32":
        return None
    if want == "bf16":
        if not bf16_ok:
            raise RuntimeError(f"dtype=bf16 requested but bf16 autocast is unsupported on device '{device}'")
        return torch.bfloat16
    if want == "auto":
        return torch.bfloat16 if bf16_ok else None
    raise ValueError(f"Unknown system.dtype '{want}' (expected auto | bf16 | fp32)")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    """Autocast context for `dtype`, or a no-op context when dtype is None (fp32)."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


# ----------------------------------------------------------------------------- environment info

def query_gpu_state() -> dict | None:
    """Query SM/mem clocks (MHz), temperature (C), power draw/limit (W) via nvidia-smi; None on failure."""
    cmd = ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True).stdout
        first = out.strip().splitlines()[0]
        vals = [v.strip() for v in first.split(",")]
        names = ["sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_draw_w", "power_limit_w"]
        if len(vals) != len(names):
            return None
        state: dict[str, float | None] = {}
        for name, v in zip(names, vals):
            try:
                state[name] = float(v)
            except ValueError:  # e.g. "[N/A]"
                state[name] = None
        return state
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def get_env_info(device: torch.device, dtype: torch.dtype | None) -> dict:
    """Collect hardware/software info for env.json."""
    info: dict[str, Any] = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dtype": "bf16-autocast (router fp32)" if dtype is not None else "fp32",
        "cpu_threads": torch.get_num_threads(),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        name = torch.cuda.get_device_name(idx)
        info["gpu_name"] = name
        info["hardware_label"] = "RTX 5070 Ti Laptop GPU" if "5070 Ti Laptop" in name else name
        info["compute_capability"] = f"{props.major}.{props.minor}"
        info["total_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_state"] = query_gpu_state()
    else:
        info["gpu_name"] = None
        info["hardware_label"] = "CPU"
        info["compute_capability"] = None
        info["total_vram_gb"] = None
        info["gpu_state"] = None
    return info


# ----------------------------------------------------------------------------- run directories

def make_run_id(variant: str, tag: str | None = None) -> str:
    """'YYYYmmdd-HHMMSS_<variant>[_<tag>]'."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{variant}" + (f"_{tag}" if tag else "")


def make_run_dir(cfg: dict, experiment: str, run_id: str) -> Path:
    """Create and return <repo>/<results_dir>/<experiment>/<run_id>/."""
    run_dir = REPO_ROOT / cfg["system"]["results_dir"] / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ----------------------------------------------------------------------------- json

def _to_jsonable(obj: Any) -> Any:
    """json.dump `default` hook: tensors, numpy values, paths -> plain Python."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(path: str | Path, obj: Any) -> None:
    """Write obj as pretty JSON (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class JsonlLogger:
    """Append-only JSON-lines logger; flushes every write; usable as a context manager."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def log(self, record: dict) -> None:
        """Write one record as one line."""
        self._f.write(json.dumps(record, default=_to_jsonable) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ----------------------------------------------------------------------------- checkpoints

def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
                    step: int, cfg: dict, extra: dict | None = None) -> None:
    """Atomically save model/optimizer/step/cfg/RNG states.

    Periodic checkpoints (named ckpt_step{N:06d}.pt) prune older periodic ones in the same
    directory, so only the latest periodic checkpoint plus the final checkpoint.pt remain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "cfg": cfg,
        "extra": extra or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if path.name.startswith("ckpt_step"):
        for old in path.parent.glob("ckpt_step*.pt"):
            if old != path:
                old.unlink()


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None, map_location: str = "cpu") -> dict:
    """Restore model (+ optimizer, + RNG states) and return the payload (step, cfg, extra, ...)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if rng.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])
    return payload


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Return the checkpoint to resume from: final checkpoint.pt if present, else latest ckpt_step*.pt."""
    run_dir = Path(run_dir)
    final = run_dir / "checkpoint.pt"
    if final.exists():
        return final
    periodic = sorted(run_dir.glob("ckpt_step*.pt"))
    return periodic[-1] if periodic else None


# ----------------------------------------------------------------------------- misc

def count_parameters(module: torch.nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in module.parameters())


def peak_memory_mb(device: torch.device) -> float:
    """Peak allocated CUDA memory in MiB (0.0 on non-CUDA devices)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def reset_peak_memory(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (no-op elsewhere)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def format_table(rows: list[dict], columns: list[str]) -> str:
    """Plain-text aligned table with a header and separator line."""
    cells = [[str(r.get(c, "")) for c in columns] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(columns)]
    def fmt(vals: list[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths)).rstrip()
    lines = [fmt(columns), "  ".join("-" * w for w in widths)] + [fmt(r) for r in cells]
    return "\n".join(lines)
