"""Phase annotations for E5 profiling (router / dispatch / expert_gemm / shared_expert / combine).

The MoE layers wrap each phase of their forward in ``with phase("router"): ...``. This costs
nothing in normal training: if no :class:`PhaseTimer` is active and the torch profiler is not
running, :func:`phase` just yields.

Two consumers (scripts/profile_layer.py):
  * ``torch.profiler``: when the profiler is on, each phase becomes a
    ``record_function("moe::<phase>")`` range, so the Chrome trace shows which kernels
    belong to which phase, and profile_layer.py sums kernel device time per range.
  * :class:`PhaseTimer` (manual synchronized timers): when active, every phase records a
    pair of CUDA events (or perf_counter stamps on CPU). After ``torch.cuda.synchronize()``
    the elapsed times are summed per phase name. A phase that occurs several times in one
    forward (e.g. once per expert in the loop path) is summed.

Phase names are fixed (DASHBOARD_SPEC §1.6 breakdown.jsonl): ``router`` (gating softmax,
top-k, aux loss), ``dispatch`` (capacity assignment, permutation, scatter into buffers),
``expert_gemm`` (the two expert matmuls incl. GELU), ``shared_expert`` (DeepSeekMoE shared
experts), ``combine`` (gather x gate, scatter/sum back to token order). Anything outside these
ranges (MoEAux statistics, reshapes) is reported as ``other`` by the caller.

Only FORWARD phases are annotated. Backward kernels run later inside autograd and are not
inside these ranges.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Iterator, Optional

import torch

PHASES: tuple[str, ...] = ("router", "dispatch", "expert_gemm", "shared_expert", "combine", "other")

_ACTIVE_TIMER: Optional["PhaseTimer"] = None


@contextlib.contextmanager
def phase(name: str) -> Iterator[None]:
    """Annotate one phase of an MoE forward. No-op unless profiling or a PhaseTimer is active."""
    timer = _ACTIVE_TIMER
    if timer is None and not torch.autograd._profiler_enabled():
        yield
        return
    with torch.profiler.record_function(f"moe::{name}"):
        if timer is None:
            yield
        else:
            token = timer._start(name)
            try:
                yield
            finally:
                timer._stop(token)


class PhaseTimer:
    """Manual phase timer. Usage::

        with PhaseTimer(device) as t:
            layer(x)
        t.totals_ms()   # {"router": 0.12, "dispatch": 0.30, ...} (synchronizes)

    On CUDA each phase is bracketed by two CUDA events on the current stream; the elapsed
    time is GPU-stream time between the markers (it includes any idle gap while the CPU
    launches the phase's kernels). On CPU, perf_counter is used (CPU ops are synchronous).
    """

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self._records: list[tuple[str, object, object]] = []
        self._open: dict[int, tuple[str, object]] = {}
        self._next = 0
        self._prev: Optional[PhaseTimer] = None

    def __enter__(self) -> "PhaseTimer":
        global _ACTIVE_TIMER
        self._prev, _ACTIVE_TIMER = _ACTIVE_TIMER, self
        return self

    def __exit__(self, *exc: object) -> None:
        global _ACTIVE_TIMER
        _ACTIVE_TIMER = self._prev

    def _stamp(self) -> object:
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    def _start(self, name: str) -> int:
        token = self._next
        self._next += 1
        self._open[token] = (name, self._stamp())
        return token

    def _stop(self, token: int) -> None:
        name, start = self._open.pop(token)
        self._records.append((name, start, self._stamp()))

    def totals_ms(self) -> dict[str, float]:
        """Sum of elapsed ms per phase name (synchronizes CUDA first)."""
        if self.cuda:
            torch.cuda.synchronize()
        out: dict[str, float] = defaultdict(float)
        for name, a, b in self._records:
            if self.cuda:
                out[name] += a.elapsed_time(b)  # type: ignore[union-attr]
            else:
                out[name] += (b - a) * 1e3      # type: ignore[operator]
        return dict(out)
