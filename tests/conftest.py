"""Shared fixtures for the routing/dispatch tests (architect, T6).

Everything here is tiny so the suite runs on CPU in seconds. ``make_cfg`` returns a full
config dict with the CONFIG_SCHEMA key names; override any nested key with ``a__b=value``
style keywords, e.g. ``make_cfg(moe__switch__capacity_factor=1.0)``.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE_CFG = {
    "name": "test",
    "seed": 0,
    "variant": "dense",
    "data": {"vocab_size": 64},
    "model": {"n_layers": 2, "d_model": 16, "n_heads": 2, "seq_len": 8, "d_ff": 32,
              "moe_every": 1, "init": "switch", "init_scale": 0.1, "norm_eps": 1e-6, "rope_base": 10000},
    "moe": {
        "dispatch": "batched",
        "renormalize_gates": False,
        "switch": {"n_experts": 8, "capacity_factor": 1.25, "eval_capacity_factor": None,
                   "aux_alpha": 0.01, "jitter_eps": 0.01},
        "gshard": {"n_experts": 16, "width_divisor": 2, "top_k": 2, "capacity_factor": 1.25,
                   "eval_capacity_factor": None, "aux_alpha": 0.01, "jitter_eps": 0.0},
        "deepseek": {"m": 4, "n_shared": 1, "n_routed": 31, "k_routed": 3, "aux_alpha": 0.01,
                     "device_aux_alpha": 0.0, "n_devices": 1, "jitter_eps": 0.0},
        "dense_x8": {"width_mult": 8},
    },
    "train": {"batch_size": 2, "total_steps": 3000, "warmup_steps": 200, "peak_lr": 1e-3,
              "lr_decay_points": [0.8, 0.9], "lr_decay_factor": 0.316},
    "system": {"device": "cpu", "dtype": "fp32"},
}


def make_cfg(**overrides):
    cfg = copy.deepcopy(BASE_CFG)
    for key, val in overrides.items():
        node = cfg
        parts = key.split("__")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return cfg


def build_layer(variant: str, cfg=None, seed: int = 0):
    """Build the variant's MoE layer (layer index 0, moe_every=1) with a fixed seed."""
    from moe.moe_deepseek import DeepSeekMoE
    from moe.moe_gshard import GShardMoE
    from moe.moe_switch import SwitchMoE
    cfg = cfg or make_cfg()
    torch.manual_seed(seed)
    cls = {"switch": SwitchMoE, "gshard": GShardMoE, "deepseek": DeepSeekMoE}[variant]
    return cls(cfg["model"]["d_model"], cfg["model"]["d_ff"], cfg["moe"], cfg["model"])


def no_drop_cfg(**overrides):
    """Capacity large enough that nothing drops (cf = E gives C = N), and no jitter."""
    kw = dict(moe__switch__capacity_factor=8.0, moe__switch__jitter_eps=0.0,
              moe__gshard__capacity_factor=16.0)
    kw.update(overrides)
    return make_cfg(**kw)


CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


@pytest.fixture
def cfg():
    return make_cfg()


def run_dispatch(layer, x, dispatch, seed=123, autocast_dtype=None):
    """Forward+backward with the given dispatch path; returns (y, aux, grads dict).

    The loss is a fixed random projection of y plus the aux loss, so gradients reach the
    experts, the router (through the gates and through aux) and the input.
    """
    layer.dispatch = dispatch
    layer.zero_grad(set_to_none=True)
    xg = x.detach().clone().requires_grad_(True)
    torch.manual_seed(seed)  # identical jitter (if any) on both paths
    if autocast_dtype is not None:
        with torch.autocast(x.device.type, dtype=autocast_dtype):
            y, aux = layer(xg, return_routing=True)
    else:
        y, aux = layer(xg, return_routing=True)
    g = torch.Generator(device="cpu").manual_seed(7)
    proj = torch.randn(y.shape, generator=g).to(y.device)
    loss = (y.float() * proj).sum() + aux.aux_loss
    loss.backward()
    grads = {"x": xg.grad.detach().clone()}
    for name, p in layer.named_parameters():
        grads[name] = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
    return y.detach(), aux, grads


def assert_dispatch_equal(layer, x, atol=1e-5, rtol=1e-4, autocast_dtype=None):
    y_l, aux_l, g_l = run_dispatch(layer, x, "loop", autocast_dtype=autocast_dtype)
    y_b, aux_b, g_b = run_dispatch(layer, x, "batched", autocast_dtype=autocast_dtype)
    assert torch.equal(aux_l.topk_idx, aux_b.topk_idx)
    assert torch.equal(aux_l.kept, aux_b.kept)
    torch.testing.assert_close(y_l.float(), y_b.float(), atol=atol, rtol=rtol)
    assert g_l.keys() == g_b.keys()
    for name in g_l:
        torch.testing.assert_close(g_l[name].float(), g_b[name].float(), atol=atol, rtol=rtol,
                                   msg=lambda m, n=name: f"grad {n}: {m}")
    return aux_l
