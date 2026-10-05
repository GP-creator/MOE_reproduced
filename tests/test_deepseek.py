"""DeepSeekMoE: loop == batched, shared-expert path (MASTER §5 item 6), no dropping,
k_routed_override, E3 masking, activated-width assertion."""

import pytest
import torch

from conftest import assert_dispatch_equal, build_layer, make_cfg, requires_cuda
from moe.layer_api import ffn_spec


def _x(B=2, T=8, d=16, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, d, generator=g).to(device)


def test_shape_and_params():
    layer = build_layer("deepseek")
    assert (layer.n_shared, layer.n_routed, layer.top_k, layer.width) == (1, 31, 3, 8)   # d_ff/4
    n_ffn = sum(p.numel() for n, p in layer.named_parameters() if not n.startswith("router"))
    assert n_ffn == 8 * 2 * 16 * 32                       # (1 + 31) * d_ff/4 = 8 x dense FFN
    assert layer.router.weight.shape == (16, 31)          # router over routed experts only


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_loop_equals_batched(mode):
    layer = build_layer("deepseek", make_cfg(moe__deepseek__jitter_eps=0.01))
    layer.train(mode == "train")
    aux = assert_dispatch_equal(layer, _x(B=3))
    assert aux.drop_fraction.item() == 0.0 and aux.kept.all()


def test_loop_equals_batched_with_device_loss_and_override():
    layer = build_layer("deepseek", make_cfg(moe__deepseek__device_aux_alpha=0.05,
                                             moe__deepseek__n_devices=4))
    layer.k_routed_override = 5
    layer.n_disable_top_routed = 2
    assert_dispatch_equal(layer, _x())


@requires_cuda
def test_loop_equals_batched_bf16_cuda():
    layer = build_layer("deepseek").cuda().eval()
    assert_dispatch_equal(layer, _x(B=4, T=32, device="cuda"), atol=3e-2, rtol=3e-2,
                          autocast_dtype=torch.bfloat16)


@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_disable_shared_changes_output(dispatch):
    layer = build_layer("deepseek").eval()
    layer.dispatch = dispatch
    x = _x()
    y_on, _ = layer(x)
    layer.disable_shared = True
    y_off, _ = layer(x)
    assert not torch.allclose(y_on, y_off)
    shared = layer.shared.expert_forward(0, x.reshape(-1, 16)).reshape_as(y_on)
    torch.testing.assert_close(y_on - y_off, shared, atol=1e-6, rtol=1e-5)   # Eq. 9: no gate


@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_zero_routed_gates_gives_shared_output(dispatch, monkeypatch):
    layer = build_layer("deepseek").eval()
    layer.dispatch = dispatch
    orig = layer.router.forward

    def zero_gates(*a, **kw):
        out = orig(*a, **kw)
        out.gates = torch.zeros_like(out.gates)
        return out

    monkeypatch.setattr(layer.router, "forward", zero_gates)
    x = _x()
    y, _ = layer(x)
    shared = layer.shared.expert_forward(0, x.reshape(-1, 16)).reshape_as(y)
    torch.testing.assert_close(y, shared)


def test_gates_are_raw_softmax_over_routed():
    layer = build_layer("deepseek").eval()
    x = _x()
    _, aux = layer(x, return_routing=True)
    s = torch.softmax(x.reshape(-1, 16) @ layer.router.weight, -1)     # Eq. 11
    torch.testing.assert_close(aux.gates, s.topk(3, -1).values)         # Eq. 10


def test_no_dropping_even_when_unbalanced():
    layer = build_layer("deepseek").eval()
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, 0] = 1.0
        layer.router.weight[:, 1] = 0.9
        layer.router.weight[:, 2] = 0.8
    _, aux = layer(_x().abs() + 0.1, return_routing=True)
    assert aux.expert_counts[:3].tolist() == [16, 16, 16]
    assert torch.equal(aux.kept_counts, aux.expert_counts) and aux.drop_fraction.item() == 0.0


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_k_routed_override(k):
    layer = build_layer("deepseek").eval()
    layer.k_routed_override = k
    y, aux = layer(_x(), return_routing=True)
    assert aux.topk_idx.shape == (16, k) and aux.expert_counts.sum().item() == 16 * k


def test_disable_top_routed():
    layer = build_layer("deepseek").eval()
    x = _x()
    probs = torch.softmax(x.reshape(-1, 16) @ layer.router.weight, -1)
    order = probs.argsort(-1, descending=True)
    for n in (0, 2, 4, 6, 8):                             # PLAN §7 E3 counts for 31 experts
        layer.n_disable_top_routed = n
        _, aux = layer(x, return_routing=True)
        assert torch.equal(aux.topk_idx, order[:, n:n + 3])
        torch.testing.assert_close(aux.gates, probs.gather(-1, order[:, n:n + 3]))


def test_activated_width_assertion():
    bad = make_cfg(moe__deepseek__k_routed=4)             # (1 + 4) * d_ff/4 != d_ff
    with pytest.raises(AssertionError):
        ffn_spec("deepseek", bad)


@pytest.mark.parametrize("variant", ["deepseek", "gshard", "switch"])
@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_bitwise_deterministic_multithreaded_cpu(variant, dispatch):
    """Same weights + same input -> bit-identical outputs and grads with 4 CPU threads
    (regression test for the DeepSeek index_add combine, MASTER §5 determinism)."""
    from conftest import run_dispatch
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        layer = build_layer(variant, make_cfg(model__d_model=64, model__d_ff=256)).train()
        x = _x(B=8, T=64, d=64)
        y1, _, g1 = run_dispatch(layer, x, dispatch)
        y2, _, g2 = run_dispatch(layer, x, dispatch)
        assert torch.equal(y1, y2)
        for name in g1:
            assert torch.equal(g1[name], g2[name]), name
    finally:
        torch.set_num_threads(prev)


@pytest.mark.parametrize("variant", ["switch", "gshard", "deepseek"])
@pytest.mark.parametrize("dispatch", ["loop", "batched"])
def test_return_routing_extras(variant, dispatch):
    """DASHBOARD_SPEC R1-R3: router_probs [N, E] (rows sum to 1), routed_expert_out_norm [N, k];
    DeepSeek also shared_out_norm / routed_out_norm [N]. Absent without return_routing."""
    cfg = make_cfg(moe__switch__capacity_factor=0.5)        # some drops for switch
    layer = build_layer(variant, cfg).eval()
    layer.dispatch = dispatch
    x = _x()
    _, plain = layer(x)
    assert all(v.ndim == 0 for v in plain.extra.values())
    y, aux = layer(x, return_routing=True)
    N, E, k = 16, layer.n_routed, aux.top_k
    p = aux.extra["router_probs"]
    assert p.shape == (N, E) and p.dtype == torch.float32
    torch.testing.assert_close(p.sum(-1), torch.ones(N))
    nrm = aux.extra["routed_expert_out_norm"]
    assert nrm.shape == (N, k) and nrm.dtype == torch.float32
    assert (nrm[~aux.kept] == 0).all() and (nrm[aux.kept] > 0).all()
    if variant == "deepseek":
        sh, ro = aux.extra["shared_out_norm"], aux.extra["routed_out_norm"]
        assert sh.shape == (N,) and ro.shape == (N,)
        shared = layer.shared.expert_forward(0, x.reshape(N, 16))
        torch.testing.assert_close(sh, shared.norm(dim=-1))
        torch.testing.assert_close(ro, (y.reshape(N, 16) - shared).norm(dim=-1), atol=1e-6, rtol=1e-5)
    else:
        assert "shared_out_norm" not in aux.extra
