"""Switch layer: loop == batched (outputs + grads), Fig. 16 einsum reference, capacity and
dropping, eval capacity override (MASTER §5 items 2, 3)."""

import math

import pytest
import torch

from conftest import assert_dispatch_equal, build_layer, make_cfg, no_drop_cfg, requires_cuda
from moe.layer_api import NO_DROP, expert_capacity
from moe.moe_switch import assign_with_capacity, einsum_reference


def _x(B=2, T=8, d=16, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, generator=g).to(device)


def _force_expert0(layer):
    """Router weights so that every token with positive features picks expert 0."""
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, 0] = 1.0


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_loop_equals_batched_no_drop(mode):
    layer = build_layer("switch", no_drop_cfg(moe__switch__jitter_eps=0.01))
    layer.train(mode == "train")
    aux = assert_dispatch_equal(layer, _x())
    assert aux.drop_fraction.item() == 0.0


def test_loop_equals_batched_with_drops():
    layer = build_layer("switch", make_cfg(moe__switch__capacity_factor=0.5)).eval()
    aux = assert_dispatch_equal(layer, _x(B=4))
    assert aux.drop_fraction.item() > 0


@requires_cuda
def test_loop_equals_batched_bf16_cuda():
    layer = build_layer("switch", no_drop_cfg()).cuda().eval()
    assert_dispatch_equal(layer, _x(B=4, T=32, device="cuda"), atol=3e-2, rtol=3e-2,
                          autocast_dtype=torch.bfloat16)


@pytest.mark.parametrize("cf", [0.5, 1.0, 1.25, 8.0])
def test_einsum_reference_matches_batched(cf):
    """The literal [T, E, C] dispatch/combine einsums of Switch Fig. 16 == sort-and-segment."""
    layer = build_layer("switch", make_cfg(moe__switch__capacity_factor=cf)).eval()
    x = _x(B=3)
    y, _ = layer(x)
    x_flat = x.reshape(-1, 16)
    r = layer.router(x_flat, 1)
    asg = assign_with_capacity(r.topk_idx, 8, layer.capacity(x_flat.shape[0]))
    y_ref = einsum_reference(x_flat, asg, r.gates.t().reshape(-1), layer.experts, 1)
    torch.testing.assert_close(y.reshape(-1, 16), y_ref, atol=1e-5, rtol=1e-4)


def test_capacity_formula():
    # Switch Eq. 3: floor(N / E * cf)
    assert expert_capacity(16, 8, 1.25) == 2
    assert expert_capacity(8192, 8, 1.25) == 1280
    assert expert_capacity(8192, 8, 0.75) == 768
    assert expert_capacity(100, 8, 100.0) == 100        # clamped to N
    assert expert_capacity(100, 8, NO_DROP) == 100


@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_capacity_dropping_all_to_one_expert(dispatch):
    """All tokens prefer expert 0: exactly C are processed (the first C in token order), the
    rest output exactly zero, and drop_fraction = (N - C) / N."""
    layer = build_layer("switch").eval()
    layer.dispatch = dispatch
    _force_expert0(layer)
    x = _x().abs() + 0.1                                 # positive features -> expert 0
    N = 16
    C = expert_capacity(N, 8, 1.25)                      # = 2
    y, aux = layer(x, return_routing=True)
    y = y.reshape(N, 16)
    assert (aux.topk_idx[:, 0] == 0).all()
    assert aux.expert_counts.tolist() == [N] + [0] * 7   # pre-drop
    assert aux.kept_counts.tolist() == [C] + [0] * 7
    assert aux.kept[:, 0].tolist() == [True] * C + [False] * (N - C)
    assert math.isclose(aux.drop_fraction.item(), (N - C) / N)
    assert (y[:C].abs().sum(-1) > 0).all()
    assert torch.equal(y[C:], torch.zeros_like(y[C:]))   # dropped: residual only
    # The kept tokens get exactly gate * E_0(x) (Switch Eq. 2).
    x_flat = x.reshape(N, 16)
    expected = aux.gates[:C, 0:1] * layer.experts.expert_forward(0, x_flat[:C])
    torch.testing.assert_close(y[:C], expected)


def test_aux_uses_pre_drop_argmax():
    """f_i counts dropped tokens too (PLAN §7): with all tokens on expert 0, f_0 = 1."""
    layer = build_layer("switch").eval()
    _force_expert0(layer)
    x = _x().abs() + 0.1
    _, aux = layer(x)
    probs = torch.softmax(x.reshape(-1, 16) @ layer.router.weight, -1)
    expected = 0.01 * 8 * 1.0 * probs[:, 0].mean()       # alpha * E * f_0 * P_0
    assert math.isclose(aux.aux_loss.item(), expected.item(), rel_tol=1e-5)


def test_eval_capacity_factor_only_in_eval():
    layer = build_layer("switch", make_cfg(moe__switch__capacity_factor=0.5))
    _force_expert0(layer)
    x = _x().abs() + 0.1
    layer.eval_capacity_factor = NO_DROP
    layer.train()
    torch.manual_seed(0)
    _, aux_train = layer(x)
    assert aux_train.kept_counts[0].item() == expert_capacity(16, 8, 0.5) == 1
    layer.eval()
    _, aux_eval = layer(x)
    assert aux_eval.drop_fraction.item() == 0.0 and aux_eval.kept_counts[0].item() == 16
    layer.eval_capacity_factor = None                    # back to the training cf
    _, aux_eval2 = layer(x)
    assert aux_eval2.kept_counts[0].item() == 1


def test_return_routing_shapes_and_dtype_contract():
    layer = build_layer("switch").eval()
    y, aux = layer(_x(B=2, T=5))
    assert y.shape == (2, 5, 16) and aux.topk_idx is None
    y, aux = layer(_x(B=2, T=5), return_routing=True)
    assert aux.topk_idx.shape == (10, 1) and aux.gates.shape == (10, 1) and aux.kept.shape == (10, 1)
    assert aux.gates.dtype == torch.float32 and aux.kept.dtype == torch.bool
    assert aux.aux_loss.dtype == torch.float32 and aux.aux_loss.requires_grad
    assert aux.n_experts == 8 and aux.expert_counts.sum().item() == 10
