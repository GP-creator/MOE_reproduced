"""Param/FLOP matching (MASTER §4.3, §5 item 5; PLAN §5): real modules == flops.py, and the
matching rule (total expert FFN = 8 x dense FFN, activated = 1 x dense FFN) holds exactly."""

import copy

import pytest
import torch

from conftest import make_cfg
from moe.flops import param_row, param_table
from moe.layer_api import build_ffn, build_param_groups, is_moe_layer

MAIN = make_cfg(model__n_layers=6, model__d_model=384, model__n_heads=6, model__seq_len=256,
                model__d_ff=1536, data__vocab_size=8192)
SMOKE = make_cfg(model__n_layers=2, model__d_model=64, model__n_heads=2, model__seq_len=64,
                 model__d_ff=256, data__vocab_size=8192)
VARIANTS = ["dense", "switch", "deepseek", "gshard", "dense_x8"]

try:  # DenseFFN is the builder's (T5); skip dense-backed checks if it is not there yet.
    import moe.ffn  # noqa: F401
    HAVE_FFN = True
except Exception:  # pragma: no cover
    HAVE_FFN = False
needs_ffn = pytest.mark.skipif(not HAVE_FFN, reason="moe.ffn not available yet")


def _split_params(module):
    router = sum(p.numel() for n, p in module.named_parameters() if n.startswith("router"))
    return sum(p.numel() for p in module.parameters()) - router, router


def _active_ffn(module):
    """Activated FFN params per token from the module's own shapes."""
    if hasattr(module, "spec"):
        s = module.spec
        return 2 * module.d_model * s.width * (s.n_shared + s.top_k)
    return sum(p.numel() for p in module.parameters())


@pytest.mark.parametrize("variant", VARIANTS)
def test_main_layer_params_match_flops_table(variant):
    if variant in ("dense", "dense_x8") and not HAVE_FFN:
        pytest.skip("moe.ffn not available yet")
    torch.manual_seed(0)
    layer = build_ffn(variant, 0, MAIN)
    ffn, router = _split_params(layer)
    row = param_row(variant, MAIN)
    assert ffn == row["ffn_total_per_layer"]
    assert router == row["router_per_layer"]
    assert _active_ffn(layer) == row["ffn_active_per_layer"]


def test_matching_rule_exact_main():
    rows = {r["variant"]: r for r in param_table(MAIN)}
    dense_ffn = rows["dense"]["ffn_total_per_layer"]
    assert dense_ffn == 2 * 384 * 1536
    for v in ("switch", "deepseek", "gshard", "dense_x8"):
        assert rows[v]["ffn_total_per_layer"] == 8 * dense_ffn == 9_437_184
    for v in ("switch", "deepseek", "gshard"):
        assert rows[v]["ffn_active_per_layer"] == dense_ffn
        assert rows[v]["ffn_flops_per_token_model"] == rows["dense"]["ffn_flops_per_token_model"]
    assert {v: rows[v]["router_per_layer"] for v in rows} == {
        "dense": 0, "switch": 3072, "deepseek": 11904, "gshard": 6144, "dense_x8": 0}
    # PLAN §5 headline numbers.
    assert rows["switch"]["total_params"] == 63_331_200
    assert rows["deepseek"]["total_params"] == 63_384_192
    assert rows["gshard"]["total_params"] == 63_349_632
    assert rows["dense"]["total_params"] == 13_767_552


@needs_ffn
@pytest.mark.parametrize("moe_every", [1, 2])
def test_build_ffn_honors_moe_every(moe_every):
    from moe.ffn import DenseFFN
    from moe.moe_switch import SwitchMoE
    cfg = copy.deepcopy(SMOKE)
    cfg["model"]["n_layers"] = 4
    cfg["model"]["moe_every"] = moe_every
    for i in range(4):
        layer = build_ffn("switch", i, cfg)
        assert isinstance(layer, SwitchMoE if is_moe_layer(i, moe_every) else DenseFFN)
    total = sum(sum(p.numel() for p in build_ffn("switch", i, cfg).parameters()) for i in range(4))
    row = param_row("switch", cfg)
    assert total == row["ffn_total_model"] + row["router_model"]


@needs_ffn
@pytest.mark.parametrize("variant", VARIANTS)
def test_full_model_param_count_smoke(variant):
    try:
        from moe.model import MoETransformer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"moe.model not importable yet: {e}")
    cfg = copy.deepcopy(SMOKE)
    cfg["variant"] = variant
    model = MoETransformer(cfg)
    assert sum(p.numel() for p in model.parameters()) == param_row(variant, cfg)["total_params"]


def test_router_excluded_from_weight_decay():
    torch.manual_seed(0)
    layer = build_ffn("deepseek", 0, SMOKE)
    groups = build_param_groups(layer, 0.1)
    no_decay = {id(p) for p in groups[1]["params"]}
    assert id(layer.router.weight) in no_decay
    assert id(layer.experts.W_in) not in no_decay and id(layer.shared.W_out) not in no_decay
