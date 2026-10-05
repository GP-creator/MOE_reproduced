"""E6 placement model (scripts/placement_sim.py) and E5 phase hooks (moe/profiling.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.placement_sim import (estimate_step_time, greedy_placement, home_devices, round_robin_placement,
                                   simulate, step_layer_traffic, straggler)


def test_round_robin_and_greedy_shapes():
    assert round_robin_placement(8, 4).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    load = np.array([10.0, 1, 1, 1, 1, 1, 1, 4])
    pl = greedy_placement(load, 2)
    # cap = 4 experts per device; heaviest expert alone balances against the rest as far as the cap allows
    assert np.bincount(pl, minlength=2).tolist() == [4, 4]
    dev_load = np.bincount(pl, weights=load, minlength=2)
    rr_load = np.bincount(round_robin_placement(8, 2), weights=load, minlength=2)
    assert dev_load.max() <= rr_load.max()


def test_balanced_routing_gives_slowdown_one():
    # 8 experts on 4 devices, every expert gets exactly 2 tokens -> straggler 1.
    N, E, D = 16, 8, 4
    topk = (np.arange(N) % E)[:, None]
    kept = np.ones_like(topk, dtype=bool)
    tok, _, _, _ = step_layer_traffic(topk, kept, round_robin_placement(E, D), D, home_devices(N, D))
    assert tok.tolist() == [4, 4, 4, 4]
    assert straggler(tok[None, None, :])[0, 0] == pytest.approx(1.0)


def test_comm_bytes_hand_built():
    # N=4 tokens, D=2: homes = [0, 0, 1, 1]. Experts 0 -> dev 0, 1 -> dev 1.
    # Tokens choose experts [1, 0, 0, 1]: token 0 (home 0) -> dev 1 (off), token 2 (home 1) -> dev 0 (off).
    topk = np.array([[1], [0], [0], [1]])
    kept = np.ones_like(topk, dtype=bool)
    placement = np.array([0, 1], dtype=np.int16)
    tok, dout, cout, off = step_layer_traffic(topk, kept, placement, 2, home_devices(4, 2))
    assert off == 2 and tok.tolist() == [2, 2]
    assert dout.tolist() == [1, 1] and cout.tolist() == [1, 1]
    d, b = 16, 2
    sim = simulate(topk[None, None], kept[None, None], placement[None], 2, d, b)
    # each device sends 1 dispatch + 1 combine row of d elements
    assert sim["bytes_out_per_device"][0, 0].tolist() == [2 * d * b, 2 * d * b]
    # dropped assignments are neither computed nor sent
    kept2 = np.array([[False], [True], [True], [True]])
    tok2, _, _, off2 = step_layer_traffic(topk, kept2, placement, 2, home_devices(4, 2))
    assert off2 == 1 and tok2.tolist() == [2, 1]


def test_estimate_step_time_hand_computed():
    tpd = np.array([[[3.0, 1.0]]])            # [S=1, L=1, D=2]
    bout = np.array([[[2e9, 1e9]]])
    r = estimate_step_time(tpd, bout, flops_per_assignment=1e12, device_tflops=1.0, link_gbps=1.0,
                           include_backward=False)
    assert r["compute_ms"][0] == pytest.approx(3000.0)
    assert r["compute_balanced_ms"][0] == pytest.approx(2000.0)
    assert r["imbalance_ms"][0] == pytest.approx(1000.0)
    assert r["comm_ms"][0] == pytest.approx(2000.0)
    assert r["step_ms"][0] == pytest.approx(5000.0)
    rb = estimate_step_time(tpd, bout, flops_per_assignment=1e12, device_tflops=1.0, link_gbps=1.0,
                            include_backward=True, shared_flops_per_token=1e12, tokens_per_step=2)
    assert rb["compute_ms"][0] == pytest.approx(3 * (3000.0 + 1000.0))   # shared: 2 tokens / 2 devices
    assert rb["comm_ms"][0] == pytest.approx(4000.0)


def test_phase_hooks_noop_and_timer():
    from moe.profiling import PhaseTimer, phase
    from tests.conftest import build_layer, make_cfg

    layer = build_layer("deepseek", make_cfg())
    x = torch.randn(2, 8, 16)
    y0, _ = layer(x)                           # no timer, no profiler: plain run
    with PhaseTimer(torch.device("cpu")) as t:
        y1, _ = layer(x)
    totals = t.totals_ms()
    assert torch.equal(y0, y1)
    assert {"router", "dispatch", "expert_gemm", "shared_expert", "combine"} <= set(totals)
    with phase("router"):                      # outside a timer: still a no-op context
        pass
