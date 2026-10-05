"""Load-balancing losses (Switch Eqs. 4-6, DeepSeekMoE Eqs. 12-17) and routing metrics
(MASTER §5 item 4)."""

import math

import pytest
import torch

from conftest import build_layer, make_cfg
from moe.layer_api import MoEAux
from moe.metrics import RoutingAccumulator, load_cv, load_imbalance, summarize_aux
from moe.router import (deepseek_device_aux_loss, deepseek_expert_aux_loss, device_groups,
                        switch_aux_loss)


def test_switch_aux_uniform_is_alpha():
    """Perfect balance (f_i = P_i = 1/E) gives alpha * E * E * (1/E^2) = alpha."""
    E, T = 8, 64
    probs = torch.full((T, E), 1.0 / E)
    top1 = torch.arange(T) % E
    assert math.isclose(switch_aux_loss(probs, top1, 0.01).item(), 0.01, rel_tol=1e-6)


def test_switch_aux_collapse_is_alpha_E():
    """All tokens and all mass on one expert: f_0 = P_0 = 1 -> alpha * E."""
    E, T = 8, 32
    probs = torch.zeros(T, E)
    probs[:, 0] = 1.0
    assert math.isclose(switch_aux_loss(probs, torch.zeros(T, dtype=torch.long), 0.01).item(), 0.08, rel_tol=1e-6)


def test_switch_aux_manual():
    g = torch.Generator().manual_seed(0)
    probs = torch.softmax(torch.randn(20, 4, generator=g), -1)
    top1 = probs.argmax(-1)
    f = torch.tensor([(top1 == i).float().mean().item() for i in range(4)])
    P = probs.mean(0)
    assert math.isclose(switch_aux_loss(probs, top1, 0.02).item(), (0.02 * 4 * (f * P).sum()).item(), rel_tol=1e-6)


def test_switch_aux_gradient_flows_through_P_only():
    logits = torch.randn(16, 8, requires_grad=True)
    probs = torch.softmax(logits, -1)
    loss = switch_aux_loss(probs, probs.argmax(-1), 0.01)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_deepseek_expert_aux_manual_and_uniform():
    g = torch.Generator().manual_seed(1)
    T, Np, K = 40, 31, 3
    probs = torch.softmax(torch.randn(T, Np, generator=g), -1)
    idx = probs.topk(K, -1).indices
    counts = torch.bincount(idx.reshape(-1), minlength=Np).float()
    f = Np / (K * T) * counts                               # Eq. 13
    P = probs.mean(0)                                       # Eq. 14
    assert math.isclose(deepseek_expert_aux_loss(probs, idx, 0.01).item(), (0.01 * (f * P).sum()).item(), rel_tol=1e-6)
    # Balanced: every expert chosen K*T/N' times, uniform probs -> f_i = 1, P_i = 1/N' -> alpha1.
    T2 = Np * 4
    idx_bal = torch.stack([(torch.arange(T2) + j) % Np for j in range(K)], 1)
    uni = torch.full((T2, Np), 1.0 / Np)
    assert math.isclose(deepseek_expert_aux_loss(uni, idx_bal, 0.01).item(), 0.01, rel_tol=1e-6)


def test_deepseek_device_aux_manual():
    g = torch.Generator().manual_seed(2)
    T, Np, K, D = 30, 31, 3, 4
    probs = torch.softmax(torch.randn(T, Np, generator=g), -1)
    idx = probs.topk(K, -1).indices
    f = Np / (K * T) * torch.bincount(idx.reshape(-1), minlength=Np).float()
    P = probs.mean(0)
    groups = device_groups(Np, D)
    assert [len(gr) for gr in groups] == [8, 8, 8, 7] and torch.equal(torch.cat(groups), torch.arange(Np))
    expected = 0.05 * sum(f[gr].mean() * P[gr].sum() for gr in groups)     # Eqs. 15-17
    assert math.isclose(deepseek_device_aux_loss(probs, idx, D, 0.05).item(), expected.item(), rel_tol=1e-6)


def test_device_loss_off_by_default_and_added_when_on():
    x = torch.randn(2, 8, 16)
    off = build_layer("deepseek").eval()
    _, a_off = off(x, return_routing=True)
    probs = torch.softmax(x.reshape(-1, 16) @ off.router.weight, -1)
    assert math.isclose(a_off.aux_loss.item(), deepseek_expert_aux_loss(probs, a_off.topk_idx, 0.01).item(), rel_tol=1e-6)
    on = build_layer("deepseek", make_cfg(moe__deepseek__device_aux_alpha=0.05, moe__deepseek__n_devices=4)).eval()
    _, a_on = on(x, return_routing=True)                    # same seed -> same weights
    extra = deepseek_device_aux_loss(probs, a_on.topk_idx, 4, 0.05)
    assert math.isclose(a_on.aux_loss.item(), a_off.aux_loss.item() + extra.item(), rel_tol=1e-6)


@pytest.mark.parametrize("variant", ["switch", "gshard", "deepseek"])
def test_layer_aux_reaches_router(variant):
    layer = build_layer(variant).train()
    _, aux = layer(torch.randn(2, 8, 16))
    aux.aux_loss.backward()
    assert layer.router.weight.grad.abs().sum() > 0
    assert aux.aux_loss.dtype == torch.float32


def test_dense_aux_is_zero():
    a = MoEAux.empty("cpu")
    assert a.aux_loss.item() == 0.0 and not a.is_moe


# ---- routing metrics --------------------------------------------------------------------

def test_imbalance_and_cv():
    assert load_imbalance([4, 4, 4, 4]) == 1.0 and load_cv([4, 4, 4, 4]) == 0.0
    assert load_imbalance([16, 0, 0, 0]) == 4.0
    assert math.isclose(load_cv([16, 0, 0, 0]), math.sqrt(3))


def test_accumulator_sums_over_steps_and_skips_dense():
    sw = build_layer("switch", make_cfg(moe__switch__capacity_factor=0.5)).eval()
    xs = [torch.randn(2, 8, 16) for _ in range(3)]
    acc = RoutingAccumulator()
    singles = []
    for x in xs:
        _, a = sw(x)
        acc.update([MoEAux.empty(), a])                     # block 0 dense, block 1 MoE
        singles.append(a)
    recs = acc.flush(step=49)
    assert len(recs) == 1
    rec = recs[0]
    assert list(rec) == ["step", "window_steps", "layer", "n_experts", "top_k", "n_tokens",
                         "expert_counts", "kept_counts", "drop_fraction", "router_entropy",
                         "max_entropy", "load_imbalance", "load_cv", "aux_loss", "extra"]  # DASHBOARD_SPEC §1.4
    assert rec["layer"] == 1 and rec["step"] == 49 and rec["window_steps"] == 3
    assert rec["top_k"] == 1 and rec["n_tokens"] == 3 * 16 and rec["n_experts"] == 8
    assert math.isclose(rec["max_entropy"], math.log(8))
    counts = sum(a.expert_counts for a in singles).tolist()
    kept = sum(a.kept_counts for a in singles).tolist()
    assert rec["expert_counts"] == counts and rec["kept_counts"] == kept
    assert math.isclose(rec["drop_fraction"], 1 - sum(kept) / sum(counts))
    assert math.isclose(rec["router_entropy"], sum(a.router_entropy.item() for a in singles) / 3, rel_tol=1e-6)
    assert math.isclose(rec["load_imbalance"], load_imbalance(counts))
    assert rec["extra"] == {}
    assert acc.flush(step=50) == []                         # reset after flush
    assert summarize_aux([singles[0]])[0]["expert_counts"] == singles[0].expert_counts.tolist()


def test_accumulator_extra_scalars_only():
    ds = build_layer("deepseek", make_cfg(moe__deepseek__device_aux_alpha=0.05, moe__deepseek__n_devices=4)).eval()
    _, a = ds(torch.randn(2, 8, 16), return_routing=True)   # has [N, E] extras too
    rec = summarize_aux([a], step=1)[0]
    assert set(rec["extra"]) == {"aux_expert", "aux_device"}
    assert math.isclose(rec["extra"]["aux_expert"] + rec["extra"]["aux_device"], rec["aux_loss"], rel_tol=1e-6)
    assert rec["top_k"] == 3 and rec["drop_fraction"] == 0.0
