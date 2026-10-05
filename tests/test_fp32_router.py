"""Selective precision (Switch Sec. 2.4): the router runs in fp32 even under bf16 autocast;
experts run in the autocast dtype; gates are cast only at the combine (MASTER §5 item 7)."""

import pytest
import torch

from conftest import CUDA, build_layer, no_drop_cfg

DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("variant", ["switch", "gshard", "deepseek"])
def test_router_fp32_under_autocast(variant, device):
    layer = build_layer(variant, no_drop_cfg()).to(device).train()
    seen = {}

    def hook(mod, inp, out):
        seen["out"] = out

    layer.router.register_forward_hook(hook)
    x = torch.randn(2, 8, 16, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        y, aux = layer(x, return_routing=True)
    r = seen["out"]
    assert layer.router.weight.dtype == torch.float32
    assert r.probs.dtype == torch.float32 and r.gates.dtype == torch.float32
    assert r.topk_idx.dtype == torch.int64
    assert aux.aux_loss.dtype == torch.float32
    assert y.dtype == torch.bfloat16                      # experts ran under autocast
    (y.float().sum() + aux.aux_loss).backward()
    assert layer.router.weight.grad.dtype == torch.float32
    assert layer.experts.W_in.grad.dtype == torch.float32  # fp32 master weights


@pytest.mark.parametrize("device", DEVICES)
def test_router_matches_fp32_reference_under_autocast(device):
    """Probabilities under autocast equal a pure-fp32 computation (no bf16 rounding)."""
    layer = build_layer("switch", no_drop_cfg()).to(device).eval()
    x = torch.randn(64, 16, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        r = layer.router(x.to(torch.bfloat16).float(), 1)
    ref = torch.softmax(x.to(torch.bfloat16).float() @ layer.router.weight, -1)
    torch.testing.assert_close(r.probs, ref, atol=1e-6, rtol=1e-5)
