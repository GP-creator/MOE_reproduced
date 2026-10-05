"""Routing traces for E6 / the dashboard.  DASHBOARD_SPEC §1.7 and §5.

Batch mode (CLI): for each MoE variant's latest complete E1 run (or ``--runs``), run eval with
``return_routing=True`` over N fixed val batches and save

    results/<cfg>/traces/<source_run_id>/trace.npz   topk_idx int16, gates float16, kept bool,
                                                     each [n_batch, L_moe, N, k]
    results/<cfg>/traces/<source_run_id>/meta.json

Importable: ``trace_text(model, tokenizer, text)`` runs ONE sentence and returns per-token,
per-layer routing (used by the dashboard's "How it works" page).

Usage:
  python scripts/route_trace.py --config configs/main.yaml            # 50 batches
  python scripts/route_trace.py --config configs/smoke.yaml --results-dir <dir>   # 4 batches
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _eval_common as ec  # noqa: E402  (also puts the repo root on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moe import utils  # noqa: E402


def _np(t: torch.Tensor | None):
    return None if t is None else t.detach().float().cpu().numpy() if t.is_floating_point() else t.detach().cpu().numpy()


def _token_str(tokenizer, i: int) -> str:
    """Token text for display: byte-level BPE space marker 'Ġ' shown as '·'."""
    s = tokenizer.id_to_token(int(i))
    return (s if s is not None else "<unk>").replace("Ġ", "·")


@torch.no_grad()
def trace_text(model, tokenizer, text: str, *, max_tokens: int | None = None,
               autocast_dtype: torch.dtype | None = None) -> dict:
    """Route a single piece of text (batch of 1) and return everything the page needs.

    Returns a dict (all arrays are numpy, T = number of tokens after truncation to
    ``max_tokens`` or ``model.seq_len``)::

        token_ids   int64 [T]            tokens      list[str] (Ġ shown as '·')
        truncated   bool                 variant     str
        layers      list (one per block) of dicts:
            layer int, is_moe bool, and for MoE layers
            n_experts int, top_k int, topk_idx int64 [T,k], gates float32 [T,k], kept bool [T,k],
            router_probs float32 [T,E] or None, routed_expert_out_norm float32 [T,k] or None,
            shared_out_norm float32 [T] or None, routed_out_norm float32 [T] (DeepSeek only)
        top5_ids int64 [T,5], top5_probs float32 [T,5], top5_tokens list[list[str]]
            (next-token distribution after each position; softmax of fp32 logits)

    The model is run in eval mode (previous mode restored). Capacity overrides are the
    caller's business (``layer_api.set_eval_overrides``).
    """
    device = next(model.parameters()).device
    limit = int(max_tokens or model.seq_len)
    ids_all = tokenizer.encode(text).ids
    ids = ids_all[:limit]
    if not ids:
        raise ValueError("text produced no tokens")
    was_training = model.training
    model.eval()
    try:
        with utils.autocast_context(device, autocast_dtype):
            out = model(torch.tensor([ids], device=device), return_routing=True)
    finally:
        model.train(was_training)

    layers = []
    for i, aux in enumerate(out.layer_aux):
        rec: dict = {"layer": i, "is_moe": bool(aux.is_moe)}
        if aux.is_moe:
            ex = aux.extra
            rec.update(
                n_experts=int(aux.n_experts), top_k=int(aux.top_k),
                topk_idx=_np(aux.topk_idx), gates=_np(aux.gates).astype(np.float32), kept=_np(aux.kept),
                router_probs=_np(ex.get("router_probs")),
                routed_expert_out_norm=_np(ex.get("routed_expert_out_norm")),
                shared_out_norm=_np(ex.get("shared_out_norm")),
                routed_out_norm=_np(ex.get("routed_out_norm")))
        layers.append(rec)

    probs = out.logits[0].float().softmax(-1)                    # [T, V]
    p5, i5 = probs.topk(5, dim=-1)
    i5n = i5.cpu().numpy()
    return {"variant": model.variant, "token_ids": np.asarray(ids, dtype=np.int64),
            "tokens": [_token_str(tokenizer, t) for t in ids], "truncated": len(ids_all) > limit,
            "layers": layers, "top5_ids": i5n, "top5_probs": p5.cpu().numpy(),
            "top5_tokens": [[_token_str(tokenizer, t) for t in row] for row in i5n]}


@torch.no_grad()
def trace_batches(model, batches, device, ac_dtype) -> dict[str, np.ndarray]:
    """Eval over ``batches`` with return_routing=True; arrays [n_batch, L_moe, N, k]."""
    topk, gates, kept = [], [], []
    for x, _ in batches:
        with utils.autocast_context(device, ac_dtype):
            out = model(x, return_routing=True)
        moe = [a for a in out.layer_aux if a.is_moe]
        topk.append(torch.stack([a.topk_idx for a in moe]).to(torch.int16).cpu())
        gates.append(torch.stack([a.gates for a in moe]).to(torch.float16).cpu())
        kept.append(torch.stack([a.kept for a in moe]).cpu())
    return {"topk_idx": torch.stack(topk).numpy(), "gates": torch.stack(gates).numpy(),
            "kept": torch.stack(kept).numpy().astype(bool)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--runs", nargs="+", default=None, help="explicit run dirs (default: latest complete per MoE variant)")
    ap.add_argument("--variants", nargs="+", default=list(ec.MOE_VARIANTS))
    ap.add_argument("--n-batches", type=int, default=None, help="default: 50 for main, 4 otherwise")
    args = ap.parse_args()

    cfg0 = utils.load_config(args.config, args.override)
    root = ec.results_root(cfg0, args.results_dir)
    device = utils.get_device(cfg0)
    ac_dtype = utils.get_autocast_dtype(cfg0, device)
    n_batches = args.n_batches or (50 if cfg0["name"] == "main" else 4)
    runs = ec.find_runs(root, cfg0["name"], tuple(args.variants), args.runs)
    if not runs:
        sys.exit(f"no complete runs found under {root / cfg0['name'] / 'e1_main'}")

    for run in runs:
        model, rcfg = ec.load_run_model(run, device)
        layers = ec.moe_layers(model)
        spec = layers[0].spec
        batches = ec.val_batches(rcfg, n_batches, device)
        tr = trace_batches(model, batches, device, ac_dtype)
        moe_idx = [i for i, b in enumerate(model.blocks) if hasattr(b.ffn, "ABLATION_ATTRS")]
        out_dir = root / cfg0["name"] / "traces" / run["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "trace.npz", **tr)
        utils.save_json(out_dir / "meta.json", {
            "schema_version": 1, "source_run_dir": ec.rel_to(root, run["run_dir"]), "source_run_id": run["run_id"],
            "variant": run["variant"], "config_name": cfg0["name"], "seed": run["seed"],
            "n_batches": len(batches), "tokens_per_batch": int(batches[0][0].numel()),
            "moe_layers": moe_idx, "n_routed": spec.n_routed, "top_k": spec.top_k, "n_shared": spec.n_shared,
            "d_model": int(rcfg["model"]["d_model"]), "expert_width": spec.width,
            "capacity_factor": getattr(layers[0], "capacity_factor", None),
            "eval_capacity_factor": getattr(layers[0], "eval_capacity_factor", None),
            "created_at": dt.datetime.now().isoformat(timespec="seconds")})
        dropped = 1.0 - tr["kept"].mean()
        print(f"[trace] {run['variant']:9s} {run['run_id']}: topk_idx {tr['topk_idx'].dtype}{list(tr['topk_idx'].shape)} "
              f"gates {tr['gates'].dtype} kept {tr['kept'].dtype}  dropped={dropped:.4f}  -> {out_dir}")
        del model


if __name__ == "__main__":
    main()
