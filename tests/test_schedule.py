"""LR schedule: linear warmup then x0.316 at 80% and 90% (DeepSeekMoE; PLAN §4)."""

import math

import pytest

from moe.schedule import decay_boundaries, lr_at_step

MAIN = {"total_steps": 3000, "warmup_steps": 200, "peak_lr": 1e-3,
        "lr_decay_points": [0.8, 0.9], "lr_decay_factor": 0.316}


@pytest.mark.parametrize("step,expected", [
    (0, 1e-3 / 200), (99, 1e-3 * 100 / 200), (199, 1e-3), (200, 1e-3), (2399, 1e-3),
    (2400, 3.16e-4), (2699, 3.16e-4), (2700, 1e-3 * 0.316 ** 2), (2999, 1e-3 * 0.316 ** 2),
])
def test_main_schedule(step, expected):
    assert math.isclose(lr_at_step(step, MAIN), expected, rel_tol=1e-12)


@pytest.mark.parametrize("total,warm,b", [(150, 15, [120, 135]), (1500, 100, [1200, 1350]), (3000, 200, [2400, 2700])])
def test_boundaries(total, warm, b):
    cfg = dict(MAIN, total_steps=total, warmup_steps=warm)
    assert decay_boundaries(cfg) == b
    assert lr_at_step(b[0] - 1, cfg) == cfg["peak_lr"]


def test_monotone_and_positive():
    lrs = [lr_at_step(s, MAIN) for s in range(3000)]
    assert all(lr > 0 for lr in lrs)
    assert all(a <= b for a, b in zip(lrs[:200], lrs[1:200]))         # warmup increasing
    assert all(a >= b for a, b in zip(lrs[200:], lrs[201:]))          # never increases after


def test_negative_step_raises():
    with pytest.raises(ValueError):
        lr_at_step(-1, MAIN)
