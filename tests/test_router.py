"""Router: top-k, gates, renormalisation, jitter (train only), E3 top-r masking, entropy,
plus the eval-override plumbing in layer_api (MASTER §5 item 1)."""

import math

import pytest
import torch

from conftest import build_layer, make_cfg
from moe.layer_api import set_eval_overrides
from moe.router import Router

MODEL_CFG = make_cfg()["model"]


def _router(E=8, jitter=0.0, renorm=False, seed=0):
    torch.manual_seed(seed)
    return Router(16, E, jitter, renorm, MODEL_CFG)


def test_topk_and_raw_softmax_gates():
    r = _router().eval()
    x = torch.randn(32, 16)
    out = r(x, k=2)
    probs = torch.softmax(x @ r.weight, -1)
    assert torch.allclose(out.probs, probs, atol=1e-6)
    ref = probs.topk(2, -1)
    assert torch.equal(out.topk_idx, ref.indices)
    assert torch.allclose(out.gates, ref.values)            # raw softmax prob, no renorm
    assert torch.allclose(out.topk_prob, ref.values)
    assert (out.gates[:, 0] >= out.gates[:, 1]).all()        # column 0 = 1st choice
    assert out.topk_idx.dtype == torch.int64


def test_renormalize_gates():
    r = _router(renorm=True).eval()
    out = r(torch.randn(32, 16), k=3)
    assert torch.allclose(out.gates.sum(-1), torch.ones(32), atol=1e-6)
    assert torch.allclose(out.gates, out.topk_prob / out.topk_prob.sum(-1, keepdim=True))


def test_jitter_train_only_and_bounded():
    r = _router(jitter=0.01)
    x = torch.randn(64, 16)
    r.eval()
    a, b = r(x, 1), r(x, 1)
    assert torch.equal(a.probs, b.probs)                     # eval: deterministic, no noise
    assert torch.allclose(a.probs, torch.softmax(x @ r.weight, -1), atol=1e-6)
    r.train()
    torch.manual_seed(1)
    c = r(x, 1)
    assert not torch.allclose(c.probs, a.probs)              # train: noise applied
    # Reproduce the noise: router INPUT times U(1-eps, 1+eps) (Switch App. C).
    torch.manual_seed(1)
    noise = torch.empty_like(x).uniform_(0.99, 1.01)
    assert torch.allclose(c.probs, torch.softmax((x * noise) @ r.weight, -1), atol=1e-6)


def test_entropy():
    r = _router().eval()
    out = r(torch.randn(10, 16), 1)
    p = out.probs
    assert math.isclose(out.entropy.item(), (-(p * p.log()).sum(-1)).mean().item(), rel_tol=1e-5)
    assert out.entropy.item() <= math.log(8) + 1e-6


@pytest.mark.parametrize("n_dis,k", [(1, 1), (2, 3), (4, 2)])
def test_disable_top_routed_masking(n_dis, k):
    """E3 (DeepSeekMoE Sec. 4.5): mask the n_dis most probable experts, then top-k of the rest,
    gate = original softmax prob (no renormalisation)."""
    r = _router(E=8).eval()
    x = torch.randn(50, 16)
    out = r(x, k, n_disable_top=n_dis)
    order = out.probs.argsort(-1, descending=True)           # [N, E]
    assert torch.equal(out.topk_idx, order[:, n_dis:n_dis + k])
    assert torch.allclose(out.gates, out.probs.gather(-1, out.topk_idx))
    full = r(x, k)
    assert torch.equal(out.probs, full.probs)                # probs themselves are not masked


def test_disable_top_routed_bounds():
    r = _router(E=8).eval()
    with pytest.raises(ValueError):
        r(torch.randn(4, 16), 3, n_disable_top=6)


def test_set_eval_overrides():
    sw = build_layer("switch")
    model = torch.nn.Sequential(sw, build_layer("switch"))
    assert set_eval_overrides(model, eval_capacity_factor=2.0, n_disable_top_routed=1) == 2
    assert sw.eval_capacity_factor == 2.0 and sw.n_disable_top_routed == 1
    with pytest.raises(ValueError):
        set_eval_overrides(model, disable_shared=True)       # Switch has no shared expert
    ds = build_layer("deepseek")
    set_eval_overrides(ds, disable_top_routed_ratio=2 / 16, disable_shared=True, k_routed_override=2)
    assert ds.n_disable_top_routed == 4 and ds.disable_shared and ds.k_routed_override == 2  # round(31/8)
    set_eval_overrides(ds, n_disable_top_routed=0, disable_shared=False, k_routed_override=None)
    assert ds.n_disable_top_routed == 0 and not ds.disable_shared and ds.k_routed_override is None
