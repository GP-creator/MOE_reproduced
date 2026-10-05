"""Learning-rate schedule: linear warmup, then multi-step decay (PLAN §4, MASTER §4.8).

DeepSeekMoE (Sec. 4.1, training settings) uses a "warmup-and-step-decay strategy": the LR
rises linearly during the first warmup steps, then is multiplied by 0.316 at 80% and again
at 90% of the training steps. We use the same shape, scaled to our step counts::

    lr(t) = peak * (t + 1) / warmup          for t < warmup
    lr(t) = peak                             for warmup <= t < round(0.8 * S)
    lr(t) = peak * 0.316                     for round(0.8 * S) <= t < round(0.9 * S)
    lr(t) = peak * 0.316**2  (~0.0999 peak)  for t >= round(0.9 * S)

``t`` is the 0-based optimizer step about to be taken, S = ``train.total_steps``.
Main (S=3000, warmup=200): decays at steps 2400 and 2700. The boundaries are rounded to
integers so float error in ``0.9 * S`` cannot move them by one step.
"""

from __future__ import annotations

from typing import Any, Mapping


def decay_boundaries(cfg_train: Mapping[str, Any]) -> list[int]:
    """Integer steps at which the LR is multiplied by ``lr_decay_factor``."""
    total = int(cfg_train["total_steps"])
    return [int(round(float(p) * total)) for p in cfg_train.get("lr_decay_points", [0.8, 0.9])]


def lr_at_step(step: int, cfg_train: Mapping[str, Any]) -> float:
    """LR for 0-based ``step`` given the ``train`` config section (CONFIG_SCHEMA keys
    ``peak_lr``, ``warmup_steps``, ``total_steps``, ``lr_decay_points``, ``lr_decay_factor``).
    """
    peak = float(cfg_train["peak_lr"])
    warmup = int(cfg_train["warmup_steps"])
    factor = float(cfg_train.get("lr_decay_factor", 0.316))
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step < warmup:
        # Linear warmup; (t+1) so step 0 already has a non-zero LR and step warmup-1 hits peak.
        return peak * (step + 1) / warmup
    n_decays = sum(1 for b in decay_boundaries(cfg_train) if step >= b)
    return peak * factor ** n_decays


def set_lr(optimizer, lr: float) -> None:
    """Write ``lr`` into every param group (call once per step before ``optimizer.step()``)."""
    for group in optimizer.param_groups:
        group["lr"] = lr
