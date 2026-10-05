#!/usr/bin/env python
"""E6: expert placement / device-imbalance model (MASTER §7 E6; contract DASHBOARD_SPEC §1.8).

**A simple analytical model, not a simulator.** It takes real routing decisions recorded
by ``route_trace.py`` (``results/<cfg>/traces/<run_id>/trace.npz``) and asks: if the routed
experts of each MoE layer were spread over D devices (expert parallelism), how unevenly
would the work land, how many bytes would the all-to-all move, and what step time would a
simple max(compute) + communication model predict? There is no network topology, no
overlap of compute and communication, no queueing and no kernel efficiency model.

Model (all choices are deliberate and documented in docs/E6_NOTE.md):

* **Steps.** Each trace batch (``tokens_per_batch`` = N tokens) is one step.
* **Calibration split.** The greedy policy needs expert loads to balance on. It uses the
  aggregate per-expert load of the first ``--calib-frac`` of the steps (the calibration
  subset) and is EVALUATED on the remaining steps, so it cannot overfit the steps it is
  scored on. Round-robin needs no calibration but is scored on the same evaluation steps.
  If a trace has a single step, it is used for both and ``calibration_overlap`` is set.
* **Placement policies** (per MoE layer, routed experts only):
    - ``round_robin``: expert e -> device e mod D.
    - ``greedy``: longest-processing-time-first. Experts sorted by calibration load
      (descending) go to the device with the smallest accumulated load among devices that
      still have room; room = ceil(E / D) experts per device, so expert MEMORY stays as
      balanced as round-robin (only which experts share a device changes).
* **Shared experts** (DeepSeekMoE) are replicated on every device and run on that device's
  own tokens: perfectly balanced compute, no communication.
* **Where tokens live.** Tokens are sharded uniformly by data parallelism: token t of an
  N-token batch lives on device ``home(t) = floor(t * D / N)`` (contiguous blocks, i.e.
  whole sequences stay together). An assignment (t, expert e) is *off-device* when
  ``placement[e] != home(t)``. Dropped assignments (Switch/GShard capacity) are neither
  computed nor sent. Each top-k assignment is sent separately (no de-duplication of a
  token going to two experts on the same remote device).
* **Load.** ``tokens_per_device[s, l, D]`` = kept assignments processed by each device.
  ``straggler = max_device_load / mean_device_load`` (1 = perfect balance).
* **All-to-all bytes** (MASTER §7 E6: tokens sent off-device x d_model x bytes_per_element):
  expert parallelism needs TWO all-to-alls per layer in the forward pass, dispatch (token to
  its expert's device) and combine (expert output back home), each moving the off-device
  assignments once, so ``a2a_bytes_fwd = 2 x offdevice x d_model x bytes``. The backward
  pass repeats both in reverse: ``a2a_bytes_fwd_bwd = 2 x a2a_bytes_fwd``.
  ``bytes_out_per_device`` = bytes each device SENDS in the forward (its own off-device
  tokens for dispatch + the outputs it computed for other devices' tokens for combine).
* **Step time** (:func:`estimate_step_time`): per layer, every device computes its routed
  assignments (``flops_per_assignment`` = 4 d w forward, x3 with backward) plus the shared
  expert on its own tokens at ``device_tflops``; the layer waits for the slowest device
  (barrier at the all-to-all) and for the slowest sender at ``link_gbps`` per device. Layers
  run back to back. ``step = sum_L max_D compute + sum_L max_D comm``; the balanced part
  ``sum_L mean_D compute`` and the imbalance penalty (the difference) are reported
  separately. Only the MoE FFN layers are modelled (no attention, no optimizer).

Usage:
  python scripts/placement_sim.py --config configs/main.yaml
  python scripts/placement_sim.py --config configs/smoke.yaml --traces path/to/trace_dir [...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

POLICIES = ("round_robin", "greedy")
NOTE = (
    "Simple analytical model, not a simulator. Experts of each MoE layer are placed on D devices; tokens are "
    "data-parallel sharded (contiguous blocks). Load imbalance sets the compute critical path (max over devices "
    "per layer), the dispatch+combine all-to-all sets communication (off-device assignments x d_model x bytes, "
    "x2 for backward), and expert memory is kept even (ceil(E/D) experts per device). This is a toy version of "
    "the compute / memory / communication trade-off that WSC-LLM explores when it co-designs wafer-scale "
    "hardware and placement: balancing compute (greedy placement) is bounded by memory per device, and "
    "spreading experts over more devices shrinks per-device compute but grows all-to-all traffic, so link "
    "bandwidth decides which resource limits the step. No topology, overlap, queueing or kernel efficiency.")


# ========================================================================================
# Placement policies
# ========================================================================================

def round_robin_placement(n_experts: int, n_devices: int) -> np.ndarray:
    """expert e -> device e mod D. Returns int16 [E]."""
    return (np.arange(n_experts) % n_devices).astype(np.int16)


def greedy_placement(load: np.ndarray, n_devices: int, max_per_device: Optional[int] = None) -> np.ndarray:
    """Longest-processing-time-first placement with a per-device expert-count cap.

    load [E] = calibration load per expert. Experts in descending load order go to the
    least-loaded device that still has fewer than ``max_per_device`` experts (default
    ceil(E/D), which keeps expert memory as even as round-robin). Ties: lowest device id.
    Returns int16 [E].
    """
    E = int(load.shape[0])
    cap = int(math.ceil(E / n_devices)) if max_per_device is None else int(max_per_device)
    if cap * n_devices < E:
        raise ValueError(f"cap {cap} x {n_devices} devices < {E} experts")
    placement = np.full(E, -1, dtype=np.int16)
    dev_load = np.zeros(n_devices, dtype=np.float64)
    dev_count = np.zeros(n_devices, dtype=np.int64)
    order = np.argsort(-load, kind="stable")              # heaviest first; stable -> deterministic
    for e in order:
        open_devs = np.flatnonzero(dev_count < cap)
        d = open_devs[np.argmin(dev_load[open_devs])]     # argmin returns the lowest id on ties
        placement[e] = d
        dev_load[d] += load[e]
        dev_count[d] += 1
    return placement


# ========================================================================================
# Per-step device loads and all-to-all traffic
# ========================================================================================

def home_devices(n_tokens: int, n_devices: int) -> np.ndarray:
    """Data-parallel home of each token: home(t) = floor(t * D / N). int64 [N]."""
    return (np.arange(n_tokens) * n_devices) // n_tokens


def step_layer_traffic(topk_idx: np.ndarray, kept: np.ndarray, placement: np.ndarray,
                       n_devices: int, home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One (step, layer).

    topk_idx [N, k] expert ids, kept [N, k] bool, placement [E] -> device, home [N].
    Returns (tokens_per_device [D], dispatch_out [D], combine_out [D], offdevice):
      tokens_per_device[d]: kept assignments computed on device d
      dispatch_out[d]:     off-device assignments whose token lives on d (sent by d)
      combine_out[d]:      off-device assignments computed on d (outputs sent back by d)
      offdevice:           total off-device assignments
    """
    dev = placement[topk_idx.astype(np.int64)]                       # [N, k] device of each assignment
    home_b = np.broadcast_to(home[:, None], dev.shape)              # [N, k]
    tokens = np.bincount(dev[kept], minlength=n_devices)
    off = kept & (dev != home_b)
    dispatch_out = np.bincount(home_b[off], minlength=n_devices)
    combine_out = np.bincount(dev[off], minlength=n_devices)
    return tokens, dispatch_out, combine_out, int(off.sum())


def simulate(topk_idx: np.ndarray, kept: np.ndarray, placements: np.ndarray, n_devices: int,
             d_model: int, bytes_per_element: int) -> dict[str, np.ndarray]:
    """All steps and layers. topk_idx/kept [S, L, N, k], placements [L, E].

    Returns tokens_per_device int32 [S, L, D], bytes_out_per_device float64 [S, L, D]
    (forward all-to-all bytes sent per device = (dispatch_out + combine_out) * d * bytes),
    offdevice int64 [S, L].
    """
    S, L, N, _ = topk_idx.shape
    home = home_devices(N, n_devices)
    tpd = np.zeros((S, L, n_devices), dtype=np.int32)
    bout = np.zeros((S, L, n_devices), dtype=np.float64)
    off = np.zeros((S, L), dtype=np.int64)
    for s in range(S):
        for l in range(L):
            t, dout, cout, o = step_layer_traffic(topk_idx[s, l], kept[s, l], placements[l], n_devices, home)
            tpd[s, l] = t
            bout[s, l] = (dout + cout) * d_model * bytes_per_element
            off[s, l] = o
    return {"tokens_per_device": tpd, "bytes_out_per_device": bout, "offdevice": off}


# ========================================================================================
# Step-time model (exported; the dashboard calls this, DASHBOARD_SPEC §1.8)
# ========================================================================================

def estimate_step_time(
    tokens_per_device: np.ndarray,      # [S, L, D] assignments per device
    bytes_out_per_device: np.ndarray,   # [S, L, D] forward all-to-all bytes sent per device
    *,
    flops_per_assignment: float,        # FLOPs of one routed (token, expert) assignment, fwd
    device_tflops: float,               # per-device compute, TFLOP/s
    link_gbps: float,                   # per-device link bandwidth, GB/s (1e9 B/s)
    include_backward: bool = True,      # FLOPs x3, bytes x2 (MASTER §7 E6 fwd+bwd)
    shared_flops_per_token: float = 0.0,
    tokens_per_step: int | None = None, # needed when shared_flops_per_token > 0
) -> dict[str, np.ndarray]:
    """Analytical estimate, per step [S]: compute_ms = Σ_L max_D(compute), compute_balanced_ms =
    Σ_L mean_D(compute), comm_ms = Σ_L max_D(bytes_out)/link, step_ms = compute_ms + comm_ms
    (MASTER §7 E6: max(compute) + communication). Also imbalance_ms = compute_ms - compute_balanced_ms.

    Physics (simple, documented): each device computes its routed assignments plus the
    (replicated) shared expert on its own tokens_per_step / D tokens, at ``device_tflops``
    with perfect efficiency. Every layer ends at a barrier (the all-to-all), so a layer takes
    its slowest device's compute plus its slowest sender's bytes / ``link_gbps`` (per-device
    injection bandwidth, no contention, no overlap). Backward = 2x forward FLOPs (3x total)
    and the same all-to-alls again (2x bytes total).
    """
    tpd = np.asarray(tokens_per_device, dtype=np.float64)
    bout = np.asarray(bytes_out_per_device, dtype=np.float64)
    if tpd.ndim != 3 or bout.shape != tpd.shape:
        raise ValueError(f"expected matching [S, L, D] arrays, got {tpd.shape} and {bout.shape}")
    if device_tflops <= 0 or link_gbps <= 0:
        raise ValueError("device_tflops and link_gbps must be > 0")
    D = tpd.shape[2]
    flop_mult = 3.0 if include_backward else 1.0
    byte_mult = 2.0 if include_backward else 1.0
    flops_dev = tpd * float(flops_per_assignment) * flop_mult                   # [S, L, D]
    if shared_flops_per_token:
        if tokens_per_step is None:
            raise ValueError("tokens_per_step is required when shared_flops_per_token > 0")
        flops_dev = flops_dev + float(shared_flops_per_token) * flop_mult * tokens_per_step / D
    compute_dev_ms = flops_dev / (device_tflops * 1e12) * 1e3                   # [S, L, D]
    comm_dev_ms = bout * byte_mult / (link_gbps * 1e9) * 1e3                    # [S, L, D]
    compute_ms = compute_dev_ms.max(axis=2).sum(axis=1)                         # [S]
    compute_balanced_ms = compute_dev_ms.mean(axis=2).sum(axis=1)               # [S]
    comm_ms = comm_dev_ms.max(axis=2).sum(axis=1)                               # [S]
    return {"compute_ms": compute_ms, "compute_balanced_ms": compute_balanced_ms,
            "imbalance_ms": compute_ms - compute_balanced_ms, "comm_ms": comm_ms,
            "step_ms": compute_ms + comm_ms}


def load_placement_arrays(run_dir: str | Path) -> dict[tuple[str, int, str], dict[str, np.ndarray]]:
    """{(source_run_id, D, policy): {"tokens_per_device", "bytes_out_per_device", "placement"}}"""
    out: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    with np.load(Path(run_dir) / "placement_arrays.npz") as z:
        for key in z.files:
            rid, dpart, policy, name = key.split("|")
            out.setdefault((rid, int(dpart[1:]), policy), {})[name] = z[key]
    return out


# ========================================================================================
# Driver
# ========================================================================================

def find_traces(cfg: dict, explicit: Optional[list[str]]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    root = REPO_ROOT / cfg["system"]["results_dir"] / cfg["name"] / "traces"
    return sorted(Path(p).parent for p in glob.glob(str(root / "*" / "trace.npz")))


def latest_measured_tflops(cfg: dict) -> tuple[Optional[float], Optional[str]]:
    """Measured GEMM roof from the newest e5_profile run of this config, if any."""
    results = REPO_ROOT / cfg["system"]["results_dir"]
    for p in sorted(glob.glob(str(results / cfg["name"] / "e5_profile" / "*" / "roofline.json")), reverse=True):
        try:
            roof = json.loads(Path(p).read_text())
            for c in roof.get("ceilings", []):
                if c.get("kind") == "measured" and c.get("peak_tflops"):
                    return float(c["peak_tflops"]), str(Path(p).parent.relative_to(results))
        except (OSError, ValueError):
            continue
    return None, None


def straggler(tpd: np.ndarray) -> np.ndarray:
    """max/mean device load per (step, layer); 1.0 where a layer had no load. [S, L]"""
    mean = tpd.mean(axis=-1)
    mx = tpd.max(axis=-1)
    return np.where(mean > 0, mx / np.where(mean > 0, mean, 1), 1.0)


def main() -> None:
    from moe.utils import (JsonlLogger, get_env_info, load_config, make_run_dir, make_run_id,  # noqa: E402
                           save_json)
    import torch  # noqa: E402  (only for env info)

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--traces", nargs="*", help="trace dirs (default: results/<cfg>/traces/*)")
    ap.add_argument("--D", default="2,4,8", help="device counts")
    ap.add_argument("--calib-frac", type=float, default=0.5, help="fraction of steps used to calibrate greedy")
    ap.add_argument("--bytes-per-element", type=int, default=2, help="activation bytes on the wire (bf16 = 2)")
    ap.add_argument("--device-tflops", type=float, default=None,
                    help="per-device TFLOP/s (default: measured roof of the newest E5 run, else 50)")
    ap.add_argument("--link-gbps", type=float, default=64.0,
                    help="per-device link GB/s (default 64 ~ PCIe 5.0 x16 one direction; an assumption)")
    ap.add_argument("--out-dir", help="write here instead of results/<cfg>/e6_placement/<run_id>")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    D_values = [int(d) for d in args.D.split(",")]
    traces = find_traces(cfg, args.traces)
    if not traces:
        print(f"E6: no traces under results/{cfg['name']}/traces/ (run `make trace` / route_trace.py first). Nothing to do.")
        return

    measured, measured_src = latest_measured_tflops(cfg)
    if args.device_tflops is not None:
        device_tflops, tflops_src = args.device_tflops, "config"
    elif measured is not None:
        device_tflops, tflops_src = measured, f"measured roofline {measured_src}"
    else:
        device_tflops, tflops_src = 50.0, "config"
    link_gbps = args.link_gbps

    run_id = make_run_id("placement")
    run_dir = Path(args.out_dir) if args.out_dir else make_run_dir(cfg, f"{cfg['name']}/e6_placement", run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "env.json", get_env_info(torch.device("cpu"), None))

    arrays: dict[str, np.ndarray] = {}
    trace_meta = []
    summary = JsonlLogger(run_dir / "placement_summary.jsonl")
    table = []
    for tdir in traces:
        meta = json.loads((tdir / "meta.json").read_text())
        with np.load(tdir / "trace.npz") as z:
            topk = z["topk_idx"].astype(np.int64)                  # [S, L, N, k]
            kept = z["kept"].astype(bool)
        S_all, L, N, k = topk.shape
        E, d, w = int(meta["n_routed"]), int(meta["d_model"]), int(meta["expert_width"])
        n_shared = int(meta.get("n_shared", 0) or 0)
        rid, variant = str(meta["source_run_id"]), str(meta["variant"])
        n_cal = max(1, int(round(args.calib_frac * S_all))) if S_all > 1 else 1
        overlap = S_all <= 1 or n_cal >= S_all
        if overlap:
            n_cal = S_all
        cal = slice(0, n_cal)
        ev = slice(0, S_all) if overlap else slice(n_cal, S_all)
        flops_per_assignment = 4.0 * d * w                         # two GEMMs, 2 FLOPs per MAC
        shared_flops = 4.0 * d * w * n_shared
        try:
            tdir_rel = str(tdir.resolve().relative_to((REPO_ROOT / cfg["system"]["results_dir"]).resolve()))
        except ValueError:
            tdir_rel = str(tdir)
        trace_meta.append({"trace_dir": tdir_rel, "variant": variant, "source_run_id": rid,
                           "n_steps": ev.stop - ev.start, "n_calib_steps": n_cal, "calibration_overlap": overlap,
                           "n_layers": L, "tokens_per_step": N, "top_k": k, "n_routed": E, "n_shared": n_shared,
                           "d_model": d, "expert_width": w,
                           "flops_per_assignment": flops_per_assignment, "shared_flops_per_token": shared_flops})
        # Calibration load per (layer, expert): kept assignments over the calibration steps.
        cal_load = np.zeros((L, E), dtype=np.float64)
        for l in range(L):
            idx = topk[cal, l][kept[cal, l]]
            cal_load[l] = np.bincount(idx, minlength=E)
        for D in D_values:
            if D > E:
                print(f"  skip D={D} > E={E} for {rid}")
                continue
            for policy in POLICIES:
                if policy == "round_robin":
                    placements = np.stack([round_robin_placement(E, D) for _ in range(L)])
                else:
                    placements = np.stack([greedy_placement(cal_load[l], D) for l in range(L)])
                sim = simulate(topk[ev], kept[ev], placements, D, d, args.bytes_per_element)
                tpd, bout, off = sim["tokens_per_device"], sim["bytes_out_per_device"], sim["offdevice"]
                key = f"{rid}|D{D}|{policy}"
                arrays[f"{key}|tokens_per_device"] = tpd
                arrays[f"{key}|bytes_out_per_device"] = bout
                arrays[f"{key}|placement"] = placements.astype(np.int16)
                est = estimate_step_time(tpd, bout, flops_per_assignment=flops_per_assignment,
                                         device_tflops=device_tflops, link_gbps=link_gbps, include_backward=True,
                                         shared_flops_per_token=shared_flops, tokens_per_step=N)
                strag = straggler(tpd)                                        # [S, L]
                kept_sl = kept[ev].sum(axis=(2, 3))                           # [S, L] kept assignments
                for layer in list(range(L)) + ["all"]:
                    if layer == "all":
                        # Effective slowdown with a barrier per layer: sum_L max / sum_L mean.
                        s_vals = tpd.max(axis=2).sum(axis=1) / np.maximum(tpd.mean(axis=2).sum(axis=1), 1e-12)
                        mtpd = tpd.sum(axis=1).mean(axis=0)
                        offd = off.sum(axis=1)
                        e_l = est
                    else:
                        s_vals = strag[:, layer]
                        mtpd = tpd[:, layer].mean(axis=0)
                        offd = off[:, layer]
                        e_l = estimate_step_time(tpd[:, layer:layer + 1], bout[:, layer:layer + 1],
                                                 flops_per_assignment=flops_per_assignment,
                                                 device_tflops=device_tflops, link_gbps=link_gbps,
                                                 include_backward=True, shared_flops_per_token=shared_flops,
                                                 tokens_per_step=N)
                    a2a_fwd = 2.0 * offd * d * args.bytes_per_element
                    row = {"trace_dir": tdir_rel, "variant": variant, "source_run_id": rid, "D": D,
                           "policy": policy, "layer": layer,
                           "mean_tokens_per_device": [float(v) for v in mtpd],
                           "straggler_mean": float(s_vals.mean()), "straggler_p50": float(np.percentile(s_vals, 50)),
                           "straggler_p90": float(np.percentile(s_vals, 90)), "straggler_max": float(s_vals.max()),
                           "offdevice_tokens_mean": float(offd.mean()),
                           "offdevice_fraction_mean": float(offd.sum() / max(1, kept_sl.sum() if layer == "all"
                                                                              else kept_sl[:, layer].sum())),
                           "a2a_bytes_fwd_mean": float(a2a_fwd.mean()), "a2a_bytes_fwd_bwd_mean": float(2 * a2a_fwd.mean()),
                           "step_ms_default": float(e_l["step_ms"].mean()),
                           "compute_ms_default": float(e_l["compute_ms"].mean()),
                           "compute_balanced_ms_default": float(e_l["compute_balanced_ms"].mean()),
                           "imbalance_ms_default": float(e_l["imbalance_ms"].mean()),
                           "comm_ms_default": float(e_l["comm_ms"].mean())}
                    summary.log(row)
                    if layer == "all":
                        table.append(row)
    summary.close()
    np.savez_compressed(run_dir / "placement_arrays.npz", **arrays)
    save_json(run_dir / "placement_params.json", {
        "schema_version": 1, "run_id": run_dir.name, "config_name": cfg["name"],
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "D_values": D_values, "policies": list(POLICIES), "bytes_per_element": args.bytes_per_element,
        "include_backward_default": True, "device_tflops_default": device_tflops,
        "device_tflops_source": tflops_src, "link_gbps_default": link_gbps,
        "link_gbps_source": "assumption (--link-gbps; default 64 GB/s ~ PCIe 5.0 x16 per direction)",
        "calib_frac": args.calib_frac, "token_home": "floor(t * D / N) (data-parallel contiguous shards)",
        "greedy": "LPT on calibration-step load, cap ceil(E/D) experts per device, evaluated on held-out steps",
        "a2a_model": "dispatch + combine per layer: a2a_fwd = 2 * offdevice * d_model * bytes; fwd_bwd = 2 * a2a_fwd",
        "traces": trace_meta, "note": "simple analytical model, not a simulator. " + NOTE})

    # Console table: slowdown vs D (layer = "all").
    print(f"E6 placement -> {run_dir}")
    print(f"  device {device_tflops:.1f} TFLOP/s ({tflops_src}), link {link_gbps:g} GB/s, "
          f"bytes/elem {args.bytes_per_element}")
    hdr = f"  {'variant':<9} {'D':>2} {'policy':<11} {'straggler':>9} {'p90':>6} {'offdev/step':>11} " \
          f"{'a2a fwd+bwd':>12} {'step ms':>8} {'bal ms':>7} {'imb ms':>7} {'comm ms':>8}"
    print(hdr)
    for r in table:
        print(f"  {r['variant']:<9} {r['D']:>2} {r['policy']:<11} {r['straggler_mean']:>9.3f} {r['straggler_p90']:>6.3f} "
              f"{r['offdevice_tokens_mean']:>11.0f} {r['a2a_bytes_fwd_bwd_mean'] / 1e6:>10.2f}MB "
              f"{r['step_ms_default']:>8.3f} {r['compute_balanced_ms_default']:>7.3f} "
              f"{r['imbalance_ms_default']:>7.3f} {r['comm_ms_default']:>8.3f}")


if __name__ == "__main__":
    main()
