"""Single-sentence inference for the "How it works" page (DASHBOARD_SPEC §5). No streamlit.

Routing itself comes from ``scripts.route_trace.trace_text`` (builder-owned; the same code
path that writes the E6 traces), so the page never re-implements the call
``model(idx, return_routing=True)``. Model loading reuses ``scripts._eval_common.load_run_model``.

This module only adds what the step-through needs on top of ``trace_text``:
  * forward hooks (registered for the one call, then removed) on each block's ``norm2`` and
    ``ffn`` and on the block itself, giving ‖h‖ (pre-norm input), ‖u‖ (norm2 output),
    ‖y‖ (FFN output) and ‖h_out‖ (block output) per token  (spec §5.3 steps 1 and 6);
  * the capacity C / active cf per capacity layer (spec step 3);
  * code locations resolved at RUNTIME from the live model's classes with ``inspect``
    (:func:`code_location`), so file:line and qualnames cannot go stale.
"""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from dashboard.results_io import REPO_ROOT

# ----------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------


def _route_trace():
    """Import the builder's trace module lazily (it imports torch + the model)."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts import route_trace   # noqa: WPS433  (puts scripts/ on sys.path itself)
    return route_trace


def load_model(run_dir: str | Path, device: str = "cpu"):
    """(model in eval mode on ``device``, cfg) from ``<run_dir>/config.json`` + ``checkpoint.pt``.

    Uses ``scripts._eval_common.load_run_model`` (which calls ``moe.utils.load_checkpoint``;
    note that also restores the RNG state saved in the checkpoint, harmless here because
    inference in eval mode draws no random numbers).
    """
    _route_trace()
    import _eval_common as ec   # on sys.path once route_trace is imported
    model, cfg = ec.load_run_model({"run_dir": Path(run_dir)}, torch.device(device))
    return model, cfg


def tokenizer_path(cfg: dict) -> Path:
    """Shared tokenizer file (``moe.data.load_tokenizer`` reads the same path)."""
    return REPO_ROOT / cfg["data"]["cache_dir"] / "tokenizer.json"


def load_tokenizer(cfg: dict):
    from moe.data import load_tokenizer as _load
    return _load(cfg["data"], REPO_ROOT)


# ----------------------------------------------------------------------------------------
# One sentence
# ----------------------------------------------------------------------------------------


@dataclass
class SentenceTrace:
    """``trace_text`` output plus per-block hook norms and capacity info.

    ``layers[i]`` is the trace_text dict for block i, extended with:
      ``h_norm``, ``u_norm``, ``y_norm``, ``hout_norm``: float32 [T] (hook norms);
      ``capacity`` (int or None), ``capacity_factor`` (float or None, inf = no-drop),
      ``ffn_class`` (live class name), ``dispatch`` (str or None).
    """

    variant: str
    tokens: list[str]
    token_ids: np.ndarray
    truncated: bool
    layers: list[dict]
    top5_ids: np.ndarray
    top5_probs: np.ndarray
    top5_tokens: list[list[str]]
    n_tokens: int
    n_tokens_in_text: int
    cfg: dict = field(repr=False, default_factory=dict)


def _norm_rows(t: torch.Tensor) -> np.ndarray:
    """[1, T, d] -> per-token L2 norm float32 [T]."""
    return t.detach().float().reshape(-1, t.shape[-1]).norm(dim=-1).cpu().numpy()


def run_sentence(model, tokenizer, text: str, cfg: dict, *, no_drop: bool = False) -> SentenceTrace:
    """Route ``text`` through ``model`` (batch of one) and collect everything the page shows.

    ``no_drop`` (Switch/GShard only): eval capacity = N via
    ``layer_api.set_eval_overrides(eval_capacity_factor=NO_DROP)``; otherwise the override is
    reset to None so the layer's configured eval cf applies (spec §5.1 item 3). The model is a
    cached object shared by sessions, so the override is set explicitly on EVERY call.
    """
    from moe import layer_api

    rt = _route_trace()
    if model.variant in ("switch", "gshard"):
        layer_api.set_eval_overrides(model, eval_capacity_factor=layer_api.NO_DROP if no_drop else None)

    seq_len = int(cfg["model"]["seq_len"])
    captured: dict[int, dict[str, np.ndarray]] = {i: {} for i in range(len(model.blocks))}
    handles = []
    for i, block in enumerate(model.blocks):
        def norm2_hook(mod, inp, out, i=i):
            captured[i]["h_norm"] = _norm_rows(inp[0])        # h after attention, before norm2
            captured[i]["u_norm"] = _norm_rows(out)           # u = RMSNorm(h)
        def ffn_hook(mod, inp, out, i=i):
            captured[i]["y_norm"] = _norm_rows(out[0])        # y = FFN(u), before the residual add
        def block_hook(mod, inp, out, i=i):
            captured[i]["hout_norm"] = _norm_rows(out[0])     # h + y
        handles += [block.norm2.register_forward_hook(norm2_hook),
                    block.ffn.register_forward_hook(ffn_hook),
                    block.register_forward_hook(block_hook)]
    try:
        tr = rt.trace_text(model, tokenizer, text, max_tokens=seq_len)
    finally:
        for h in handles:
            h.remove()

    n = len(tr["tokens"])
    layers = []
    for i, rec in enumerate(tr["layers"]):
        rec = dict(rec)
        rec.update(captured[i])
        ffn = model.blocks[i].ffn
        rec["ffn_class"] = type(ffn).__name__
        rec["dispatch"] = getattr(ffn, "dispatch", None) if rec["is_moe"] else None
        rec["capacity"] = rec["capacity_factor"] = None
        if rec["is_moe"] and hasattr(ffn, "capacity"):
            was = ffn.training
            ffn.eval()   # trace_text ran in eval mode, so report the eval capacity
            rec["capacity_factor"] = float(ffn.active_capacity_factor())
            rec["capacity"] = int(ffn.capacity(n))
            ffn.train(was)
        layers.append(rec)
    return SentenceTrace(
        variant=tr["variant"], tokens=list(tr["tokens"]), token_ids=tr["token_ids"],
        truncated=bool(tr["truncated"]), layers=layers, top5_ids=tr["top5_ids"],
        top5_probs=tr["top5_probs"], top5_tokens=tr["top5_tokens"], n_tokens=n,
        n_tokens_in_text=len(tokenizer.encode(text).ids), cfg=cfg)


# ----------------------------------------------------------------------------------------
# Code locations, resolved at runtime (spec §5.3 "Steps")
# ----------------------------------------------------------------------------------------

NOT_FOUND = "(not found: update HOW_STEPS)"


def resolve_object(spec: str) -> Any:
    """``"moe.router:Router.forward"`` -> the object, or None if it does not exist."""
    try:
        mod_name, qual = spec.split(":", 1)
        obj: Any = importlib.import_module(mod_name)
        for part in qual.split("."):
            obj = getattr(obj, part)
        return obj
    except (ValueError, ImportError, AttributeError):
        return None


def code_location(obj_or_spec: Any) -> dict[str, Any]:
    """Where an object really lives: ``{"file", "line", "qualname", "text"}``.

    ``text`` is e.g. ``"moe/router.py:68 Router.forward"``. The qualname is the object's own
    ``__qualname__``, so an inherited method shows its defining class (``SwitchMoE.forward``
    resolves to ``CapacityMoE.forward``). Missing object -> ``text = NOT_FOUND``.
    """
    obj = resolve_object(obj_or_spec) if isinstance(obj_or_spec, str) else obj_or_spec
    if obj is None:
        return {"file": None, "line": None, "qualname": None, "text": NOT_FOUND}
    try:
        fn = inspect.unwrap(obj)
        src = Path(inspect.getsourcefile(fn) or "")
        _, line = inspect.getsourcelines(fn)
        try:
            shown = str(src.resolve().relative_to(REPO_ROOT))
        except ValueError:
            shown = str(src)
        qual = getattr(fn, "__qualname__", repr(fn))
        return {"file": shown, "line": line, "qualname": qual, "text": f"{shown}:{line} {qual}"}
    except (TypeError, OSError):
        return {"file": None, "line": None, "qualname": None, "text": NOT_FOUND}


def method_of(instance: Any, name: str) -> Any:
    """``type(instance).<name>`` or None: resolves against the LIVE class of a module."""
    return getattr(type(instance), name, None) if instance is not None else None


def entropy_nats(p: np.ndarray) -> float:
    """Entropy -Σ p ln p (nats) of one probability row."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def fmt_cf(cf: Optional[float]) -> str:
    if cf is None:
        return "—"
    return "no-drop (C = N)" if math.isinf(cf) else f"{cf:g}"
