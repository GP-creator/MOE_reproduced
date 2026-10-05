"""Full-model tests (MASTER section 5 item 1 at model level), smoke config on CPU."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from moe import layer_api
from moe.model import MoETransformer
from moe.utils import load_config, seed_everything

VARIANTS = list(layer_api.ALL_VARIANTS)
B = 2


def smoke_cfg(variant, **moe_over):
    cfg = load_config("configs/smoke.yaml", variant=variant)
    cfg["system"]["device"] = "cpu"
    cfg["system"]["dtype"] = "fp32"
    cfg["moe"].update(moe_over)
    return cfg


def batch(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    T, V = cfg["model"]["seq_len"], cfg["data"]["vocab_size"]
    return (torch.randint(0, V, (B, T), generator=g), torch.randint(0, V, (B, T), generator=g))


def build(cfg, seed=0):
    seed_everything(seed)
    return MoETransformer(cfg)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_backward(variant):
    cfg = smoke_cfg(variant)
    model = build(cfg)
    model.verify_param_count()
    x, y = batch(cfg)
    T, V = cfg["model"]["seq_len"], cfg["data"]["vocab_size"]
    out = model(x, y)
    assert out.logits.shape == (B, T, V)
    assert torch.isfinite(out.ce_loss)
    assert abs(out.ce_loss.item() - math.log(V)) < 0.5
    assert out.aux_loss.ndim == 0 and out.aux_loss.dtype == torch.float32
    if variant in layer_api.MOE_VARIANTS:
        assert out.aux_loss.item() > 0
    else:
        assert out.aux_loss.item() == 0.0

    n_layers, every = cfg["model"]["n_layers"], cfg["model"]["moe_every"]
    assert len(out.layer_aux) == n_layers
    for i, a in enumerate(out.layer_aux):
        expect = variant in layer_api.MOE_VARIANTS and layer_api.is_moe_layer(i, every)
        assert a.is_moe == expect

    (out.ce_loss + out.aux_loss).backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            # Only an expert weight that received no token may legitimately have no grad.
            assert ".ffn." in name and variant in layer_api.MOE_VARIANTS, f"{name} has no grad"
        else:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    # Backbone and router must always get gradient.
    assert model.tok_emb.weight.grad is not None
    assert model.blocks[0].attn.wq.weight.grad is not None


@pytest.mark.parametrize("variant", list(layer_api.MOE_VARIANTS))
def test_return_routing_topk_idx(variant):
    cfg = smoke_cfg(variant)
    model = build(cfg)
    x, y = batch(cfg)
    out = model(x, y, return_routing=True)
    k = layer_api.ffn_spec(variant, cfg).top_k
    for a in out.layer_aux:
        assert a.is_moe
        assert a.topk_idx is not None
        assert tuple(a.topk_idx.shape) == (B * cfg["model"]["seq_len"], k)
        assert a.topk_idx.dtype == torch.int64
        assert 0 <= int(a.topk_idx.min()) and int(a.topk_idx.max()) < a.n_experts
    out = model(x, y)  # default: routing not materialised
    assert all(a.topk_idx is None for a in out.layer_aux)


def test_moe_every_2_switch():
    cfg = smoke_cfg("switch")
    cfg["model"]["n_layers"] = 4
    cfg["model"]["moe_every"] = 2
    model = build(cfg)
    model.verify_param_count()
    x, y = batch(cfg)
    out = model(x, y)
    flags = [a.is_moe for a in out.layer_aux]
    assert flags == [layer_api.is_moe_layer(i, 2) for i in range(4)] == [False, True, False, True]
    assert out.aux_loss.item() > 0


@pytest.mark.parametrize("variant", list(layer_api.MOE_VARIANTS))
def test_loop_and_batched_dispatch_agree(variant):
    x = y = None
    models = {}
    for d in ("loop", "batched"):
        cfg = smoke_cfg(variant, dispatch=d)
        # generous capacity so nothing is dropped in train mode either
        cfg["moe"]["switch"]["capacity_factor"] = 8.0
        cfg["moe"]["gshard"]["capacity_factor"] = 16.0
        models[d] = build(cfg)
        x, y = x if x is not None else batch(cfg)[0], y if y is not None else batch(cfg)[1]
    models["batched"].load_state_dict(models["loop"].state_dict())
    for m in models.values():
        m.eval()
        layer_api.set_eval_overrides(m, eval_capacity_factor=layer_api.NO_DROP) \
            if variant != "deepseek" else None
    with torch.no_grad():
        o_l = models["loop"](x, y, return_routing=True)
        o_b = models["batched"](x, y, return_routing=True)
    for a in o_l.layer_aux + o_b.layer_aux:
        assert a.drop_fraction.item() == 0.0
    torch.testing.assert_close(o_l.logits, o_b.logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(o_l.ce_loss, o_b.ce_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(o_l.aux_loss, o_b.aux_loss, atol=1e-7, rtol=1e-5)
    for a, b in zip(o_l.layer_aux, o_b.layer_aux):
        assert torch.equal(a.topk_idx, b.topk_idx)
