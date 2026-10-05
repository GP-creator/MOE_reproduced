"""GShard (16 x d_ff/2, top-2): loop == batched, einsum reference, capacity priority
(first choices before second choices), drop fraction over assignments."""

import math

import pytest
import torch

from conftest import assert_dispatch_equal, build_layer, make_cfg, no_drop_cfg, requires_cuda
from moe.layer_api import expert_capacity
from moe.moe_switch import assign_with_capacity, einsum_reference


def _x(B=2, T=8, d=16, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, generator=g).to(device)


def test_shape_and_params():
    layer = build_layer("gshard")
    assert (layer.n_routed, layer.top_k, layer.width) == (16, 2, 16)        # 16 x d_ff/2
    n_ffn = layer.experts.W_in.numel() + layer.experts.W_out.numel()
    assert n_ffn == 8 * 2 * 16 * 32                                         # 8 x dense FFN


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_loop_equals_batched_no_drop(mode):
    layer = build_layer("gshard", no_drop_cfg())
    layer.train(mode == "train")
    aux = assert_dispatch_equal(layer, _x())
    assert aux.drop_fraction.item() == 0.0
    assert aux.expert_counts.sum().item() == 2 * 16


def test_loop_equals_batched_with_drops():
    layer = build_layer("gshard", make_cfg(moe__gshard__capacity_factor=0.5)).eval()
    aux = assert_dispatch_equal(layer, _x(B=4))
    assert aux.drop_fraction.item() > 0


@requires_cuda
def test_loop_equals_batched_bf16_cuda():
    layer = build_layer("gshard", no_drop_cfg()).cuda().eval()
    assert_dispatch_equal(layer, _x(B=4, T=32, device="cuda"), atol=3e-2, rtol=3e-2,
                          autocast_dtype=torch.bfloat16)


@pytest.mark.parametrize("cf", [0.5, 1.25])
def test_einsum_reference_matches_batched(cf):
    layer = build_layer("gshard", make_cfg(moe__gshard__capacity_factor=cf)).eval()
    x = _x(B=3)
    y, _ = layer(x)
    x_flat = x.reshape(-1, 16)
    r = layer.router(x_flat, 2)
    asg = assign_with_capacity(r.topk_idx, 16, layer.capacity(x_flat.shape[0]))
    y_ref = einsum_reference(x_flat, asg, r.gates.t().reshape(-1), layer.experts, 2)
    torch.testing.assert_close(y.reshape(-1, 16), y_ref, atol=1e-5, rtol=1e-4)


def test_capacity_is_top_k_aware():
    # C = floor(k * N / E * cf), GShard Alg. 1 (2N/E for top-2).
    assert expert_capacity(8192, 16, 1.25, top_k=2) == 1280
    layer = build_layer("gshard").eval()
    assert layer.capacity(8192) == 1280


@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_first_choices_fill_capacity_before_second_choices(dispatch):
    """Tokens 0..7 prefer (A, B), tokens 8..15 prefer (B, A). With C = 8 the first choices fill
    A and B exactly, so EVERY second choice is dropped. (Token-order priority would instead
    let early tokens' 2nd choices steal slots from later tokens' 1st choices.)"""
    cfg = make_cfg(moe__gshard__capacity_factor=4.0)    # C = floor(2*16/16*4) = 8
    layer = build_layer("gshard", cfg).eval()
    layer.dispatch = dispatch
    A, B = 3, 11
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0, A], layer.router.weight[1, A] = 10.0, 5.0
        layer.router.weight[0, B], layer.router.weight[1, B] = 5.0, 10.0
    x = torch.zeros(1, 16, 16)
    x[0, :8, 0] = 1.0                                     # type 1: A then B
    x[0, 8:, 1] = 1.0                                     # type 2: B then A
    N = 16
    assert layer.capacity(N) == 8
    y, aux = layer(x, return_routing=True)
    assert aux.topk_idx[:8].tolist() == [[A, B]] * 8 and aux.topk_idx[8:].tolist() == [[B, A]] * 8
    assert aux.kept[:, 0].all() and not aux.kept[:, 1].any()
    assert math.isclose(aux.drop_fraction.item(), 0.5)    # 16 of 32 assignments dropped
    assert aux.kept_counts[A].item() == 8 and aux.kept_counts[B].item() == 8
    assert aux.expert_counts[A].item() == 16              # pre-drop counts include 2nd choices
    # Output = p_1 * E_1(x) only.
    x_flat = x.reshape(N, 16)
    exp = torch.cat([aux.gates[:8, :1] * layer.experts.expert_forward(A, x_flat[:8]),
                     aux.gates[8:, :1] * layer.experts.expert_forward(B, x_flat[8:])])
    torch.testing.assert_close(y.reshape(N, 16), exp)


def test_aux_uses_first_choice():
    layer = build_layer("gshard").eval()
    x = _x()
    _, aux = layer(x, return_routing=True)
    probs = torch.softmax(x.reshape(-1, 16) @ layer.router.weight, -1)
    f = torch.bincount(aux.topk_idx[:, 0], minlength=16).float() / 16
    expected = 0.01 * 16 * (f * probs.mean(0)).sum()
    assert math.isclose(aux.aux_loss.item(), expected.item(), rel_tol=1e-5)


def test_disable_top_routed_counts():
    """E3 with 16 experts: n_dis in {0..4} are exact r = n/16."""
    layer = build_layer("gshard", no_drop_cfg()).eval()
    x = _x()
    for n in range(5):
        layer.n_disable_top_routed = n
        _, aux = layer(x, return_routing=True)
        r = layer.router(x.reshape(-1, 16), 2)
        order = r.probs.argsort(-1, descending=True)
        assert torch.equal(aux.topk_idx, order[:, n:n + 2])
