"""Determinism and resume tests (MASTER section 5 item 8): smoke config, CPU, synthetic data."""

from __future__ import annotations

import pytest
import torch

from moe.data import make_synthetic_batcher
from moe.model import MoETransformer
from moe.schedule import lr_at_step, set_lr
from moe.utils import load_checkpoint, load_config, save_checkpoint, seed_everything

N_STEPS = 10


def smoke_cfg(variant):
    cfg = load_config("configs/smoke.yaml", variant=variant)
    cfg["system"]["device"] = "cpu"
    cfg["system"]["dtype"] = "fp32"
    cfg["train"]["batch_size"] = 4
    return cfg


def setup(cfg, seed):
    seed_everything(seed)
    model = MoETransformer(cfg)
    opt = model.configure_optimizer(cfg["train"])
    return model, opt


def batcher_for(cfg):
    return make_synthetic_batcher(cfg["data"]["vocab_size"], cfg["model"]["seq_len"],
                                  cfg["train"]["batch_size"], seed=cfg["seed"], n_tokens=20_000)


def train_steps(model, opt, cfg, batcher, start, stop):
    losses = []
    model.train()
    for step in range(start, stop):
        x, y = batcher.get_train_batch(step, "cpu")
        set_lr(opt, lr_at_step(step, cfg["train"]))
        out = model(x, y)
        loss = out.ce_loss + out.aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        losses.append(loss.item())
    return losses


def run(cfg, seed):
    model, opt = setup(cfg, seed)
    return train_steps(model, opt, cfg, batcher_for(cfg), 0, N_STEPS)


@pytest.mark.parametrize("variant", ["dense", "switch", "deepseek"])
def test_same_seed_identical_different_seed_differs(variant):
    cfg = smoke_cfg(variant)
    a, b = run(cfg, 1337), run(cfg, 1337)
    assert a == b, "same seed must give bit-identical losses on CPU"
    c = run(cfg, 1338)
    assert a != c


@pytest.mark.parametrize("variant", ["dense", "switch", "deepseek"])
def test_resume_matches_uninterrupted(variant, tmp_path):
    cfg = smoke_cfg(variant)
    batcher = batcher_for(cfg)
    full = run(cfg, 1337)

    model, opt = setup(cfg, 1337)
    first = train_steps(model, opt, cfg, batcher, 0, 5)
    ckpt = tmp_path / "ckpt_step000005.pt"
    save_checkpoint(ckpt, model, opt, step=5, cfg=cfg)
    cont = train_steps(model, opt, cfg, batcher, 5, N_STEPS)
    assert first + cont == full

    model2, opt2 = setup(cfg, 999)  # different init: everything must come from the checkpoint
    payload = load_checkpoint(ckpt, model2, opt2)
    assert payload["step"] == 5
    resumed = train_steps(model2, opt2, cfg, batcher_for(cfg), 5, N_STEPS)
    assert resumed == full[5:]
