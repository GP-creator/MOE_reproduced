"""Routing statistics from :class:`moe.layer_api.MoEAux` (training logs, eval, dashboard).

Definitions (per MoE layer, over the tokens of a step or of a log interval):
  * counts[i]      : assignments routed to expert i before dropping (MoEAux.expert_counts)
  * kept_counts[i] : assignments actually processed (after capacity)
  * drop_fraction  : dropped assignments / all assignments
  * router_entropy : mean over tokens of the router softmax entropy (nats); max = ln E
  * load_imbalance : max_i counts[i] / mean_i counts[i]   (1.0 = perfectly balanced)
  * load_cv        : std_i counts[i] / mean_i counts[i]   (coefficient of variation,
                     population std; 0.0 = perfectly balanced)

``train.py`` usage, with ONE host sync per log interval (PLAN §7 "routing logs")::

    acc = RoutingAccumulator()
    for step ...:
        out = model(idx, targets)
        acc.update(out.layer_aux)                 # GPU-only adds, no sync (return_routing=False is fine)
        if (step + 1) % routing_log_interval == 0:
            for rec in acc.flush(step):           # one .cpu() for all layers
                routing_logger.log(rec)           # -> routing_log.jsonl
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch

from moe.layer_api import MoEAux


def load_imbalance(counts: Sequence[float]) -> float:
    """max / mean of per-expert counts (1.0 = balanced). NaN if there are no assignments."""
    total = float(sum(counts))
    if total == 0:
        return float("nan")
    return max(counts) / (total / len(counts))


def load_cv(counts: Sequence[float]) -> float:
    """Coefficient of variation std / mean of per-expert counts (population std)."""
    n = len(counts)
    mean = float(sum(counts)) / n
    if mean == 0:
        return float("nan")
    var = sum((c - mean) ** 2 for c in counts) / n
    return math.sqrt(var) / mean


def layer_record(step: int, layer: int, expert_counts: Sequence[float], kept_counts: Sequence[float],
                 router_entropy: float, aux_loss: float, *, window_steps: int = 1, top_k: int = 0,
                 n_tokens: int = 0, extra: dict[str, float] | None = None) -> dict[str, Any]:
    """One ``routing_log.jsonl`` record (plain Python numbers) for one MoE layer.

    Field set and meaning are exactly DASHBOARD_SPEC §1.4 ``routing_log.jsonl``.
    """
    total = float(sum(expert_counts))
    E = len(expert_counts)
    return {
        "step": int(step),
        "window_steps": int(window_steps),
        "layer": int(layer),
        "n_experts": E,
        "top_k": int(top_k),
        "n_tokens": int(n_tokens),
        "expert_counts": [int(round(c)) for c in expert_counts],
        "kept_counts": [int(round(c)) for c in kept_counts],
        "drop_fraction": (1.0 - float(sum(kept_counts)) / total) if total else 0.0,
        "router_entropy": float(router_entropy),
        "max_entropy": math.log(E),
        "load_imbalance": load_imbalance(expert_counts),
        "load_cv": load_cv(expert_counts),
        "aux_loss": float(aux_loss),
        "extra": dict(extra or {}),
    }


def summarize_aux(layer_aux: Sequence[MoEAux], step: int = 0) -> list[dict[str, Any]]:
    """Records for a single forward (eval scripts, tests). One host sync."""
    acc = RoutingAccumulator()
    acc.update(layer_aux)
    return acc.flush(step)


class RoutingAccumulator:
    """Accumulates per-layer routing stats on the device; :meth:`flush` syncs once.

    Over a window of steps: counts and n_tokens are summed (so drop_fraction = dropped / all
    assignments over the whole window); router_entropy, aux_loss and every 0-dim entry of
    ``MoEAux.extra`` are averaged over the steps. Entries of ``extra`` with ndim >= 1 (the
    ``return_routing`` tensors) are ignored. Dense layers (``aux.is_moe == False``) are
    skipped; ``layer`` in the records is the block index.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Per layer: list of fp64 device tensors [counts(E), kept(E), entropy, aux, extra...]
        self._sums: dict[int, torch.Tensor] = {}
        self._meta: dict[int, dict[str, Any]] = {}   # E, top_k, extra keys, n_tokens (host ints)
        self.window_steps = 0

    @torch.no_grad()
    def update(self, layer_aux: Iterable[MoEAux]) -> None:
        """Add one step's MoEAux list (one entry per block). Device-side only, no sync."""
        for i, a in enumerate(layer_aux):
            if not a.is_moe:
                continue
            keys = sorted(k for k, v in a.extra.items() if v.ndim == 0)
            vec = torch.cat([
                a.expert_counts.detach().to(torch.float64),
                a.kept_counts.detach().to(torch.float64),
                a.router_entropy.detach().to(torch.float64).view(1),
                a.aux_loss.detach().to(torch.float64).view(1),
            ] + [a.extra[k].detach().to(torch.float64).view(1) for k in keys])
            meta = self._meta.get(i)
            if meta is None or meta["extra_keys"] != keys or meta["E"] != a.n_experts:
                if meta is not None:
                    raise ValueError(f"layer {i}: routing stats changed shape within a window")
                self._meta[i] = {"E": a.n_experts, "top_k": a.top_k, "extra_keys": keys, "n_tokens": 0}
                self._sums[i] = vec.clone()
            else:
                self._sums[i] += vec
            self._meta[i]["n_tokens"] += int(a.n_tokens)
            self._meta[i]["top_k"] = a.top_k
        self.window_steps += 1

    def flush(self, step: int) -> list[dict[str, Any]]:
        """Return one record per MoE layer (DASHBOARD_SPEC §1.4) and reset.

        Exactly one device->host copy for all layers.
        """
        if not self._sums or self.window_steps == 0:
            self.reset()
            return []
        layers = sorted(self._sums)
        packed = torch.cat([self._sums[i] for i in layers])
        flat = packed.cpu().tolist()                                  # the single sync
        recs, off, W = [], 0, self.window_steps
        for i in layers:
            m = self._meta[i]
            E, keys = m["E"], m["extra_keys"]
            row = flat[off:off + 2 * E + 2 + len(keys)]
            off += len(row)
            recs.append(layer_record(
                step, i, row[:E], row[E:2 * E], row[2 * E] / W, row[2 * E + 1] / W,
                window_steps=W, top_k=m["top_k"], n_tokens=m["n_tokens"],
                extra={k: row[2 * E + 2 + j] / W for j, k in enumerate(keys)}))
        self.reset()
        return recs
