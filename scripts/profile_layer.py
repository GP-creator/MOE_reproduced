#!/usr/bin/env python
"""E5: single-layer MoE systems profiling (MASTER §7 E5, §2.3; output contract DASHBOARD_SPEC §1.6).

What it measures, for ONE FFN layer (no attention, no norm, no residual) on N tokens:

1. ``bench.jsonl``: latency of forward (``fwd``) and forward+backward (``fwd_bwd``) for
   dense / switch / gshard / deepseek (dense_x8 on request), ``loop`` and ``batched``
   dispatch, over several sweeps:
     * ``tokens``  : N = 1K ... 64K (powers of 2) at the config's layer shapes.
     * ``experts`` : number of routed experts E at fixed N. **Policy: activated compute per
       token is held fixed (= one dense FFN of width d_ff) in both families.**
         - Switch-style (Switch Fig. 4 "scaling the number of experts"): E experts of width
           d_ff, top-1, E in {4, 8, 16, 32, 64}. Total params grow with E.
         - DeepSeek-style fine-grained segmentation (DeepSeekMoE Sec. 3.1): total routed
           params held at 8 x d_ff, E in {8, 16, 32, 64} experts of width 8*d_ff/E with
           top-k = E/8 (m = E/8), NO shared expert (isolates the segmentation effect).
           E = 4 is impossible under this policy (k would be 1/2).
     * ``capacity``: Switch and GShard (batched) at capacity_factor in {0.75, 1, 1.25, 2}:
       latency, peak memory, and padding fraction (Switch Table 1 / Fig. 3 trade-off).
     * ``skew``    : the MoE variants with SKEWED routing (see below) at one N.
2. ``gemm.jsonl``: one isolated expert GEMM at the exact shape each layer config runs
   (m = rows per expert in the batched path: capacity C for Switch/GShard, max_count for
   DeepSeek, N for dense), so fine-grained experts with few rows can be placed on the
   roofline next to one big dense GEMM.
3. ``breakdown.jsonl``: time per phase (router / dispatch / expert_gemm / shared_expert /
   combine / other). Phases are the ``moe.profiling.phase`` ranges inside the layers.
     * method ``cuda_events`` (``perf_counter`` on CPU): a ``PhaseTimer`` records a CUDA
       event pair around every phase; per-phase medians over the repeats. ``other`` = the
       median total minus the summed phases (MoEAux statistics, reshapes, launch gaps). For
       ``fwd_bwd`` only the forward is split; **the whole backward is in ``other``**.
     * method ``torch.profiler`` (forward only): kernel device time of every kernel launched
       inside each ``record_function("moe::<phase>")`` range; ``other`` = the remaining
       kernel time inside the benchmark's ``bench::<label>`` range.
   Kernel mapping (what to expect in the Chrome trace): router = fp32 ``mm`` + softmax +
   topk + bincount/mean (aux loss); dispatch = one_hot/cumsum (capacity), argsort/bincount
   (DeepSeek), ``nonzero`` + index gather (loop), zero-fill + ``index_add``/``index_copy``
   scatter (batched); expert_gemm = ``bmm``/``mm`` (cutlass/cublas GEMM kernels) + GELU
   elementwise + the autocast fp32->bf16 weight casts; combine = gather, gate multiply,
   ``index_add`` (loop) / ``index_copy`` + adds (batched); shared_expert = the DeepSeek shared
   expert's two ``mm`` + GELU.
4. ``roofline.json``: spec-sheet ceilings (cited) + measured ceilings (large bf16 GEMM and a
   large device-to-device copy) + a cited reference-GPU table.
5. ``profiler_trace.json``: Chrome trace (open in chrome://tracing or https://ui.perfetto.dev).

Timing method (MASTER §2.3): every config is warmed up, then timed for ``repeats``
iterations. On CUDA each iteration is bracketed by CUDA events (one ``synchronize`` at the
end of the group); on CPU by ``time.perf_counter`` (CPU ops are synchronous). We report the
median and p10/p90. ``moe.utils.query_gpu_state()`` (nvidia-smi: SM/mem clock, temperature,
power) is recorded right after the warmup of a benchmark group and after its last timed
iteration ("group" = one sweep x variant x dispatch x pass over all its N; nvidia-smi takes
~0.7 s under WSL so it is not called per row). Under bf16 autocast the fp32 master weights
are re-cast every forward, so the FLOP/byte model uses ``include_autocast_weight_cast=True``.

Inputs: x ~ N(0, 1) [1, N, d_model] fp32 (as the residual stream reaches the FFN), freshly
initialised layer (``router_init="random"``). Random inputs with a random router give
near-uniform routing, which is the BEST case for the batched paths (little padding). The
``skew`` sweep adds a constant vector mu to every token, solved (least squares) so that the
router logits get a Zipf-shaped offset ``skew_strength * (1/(e+1) - mean)`` per expert; the
resulting imbalance (max/mean expert load) is measured and written to each row.

Modes: ``--quick`` (automatic for the smoke config or when CUDA is unavailable) runs a tiny
subset in seconds on CPU so ``make smoke`` can include E5. Without it the full sweep runs
(Milestone 6, ~minutes on the RTX 5070 Ti Laptop GPU).

Usage:
  python scripts/profile_layer.py --config configs/smoke.yaml               # quick (CPU)
  python scripts/profile_layer.py --config configs/main.yaml                # full (GPU)
  python scripts/profile_layer.py --config configs/main.yaml --tokens 8192 --variants switch,deepseek \
      --sweeps tokens,gemm,breakdown,roofline                               # sanity subset
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moe import flops  # noqa: E402
from moe.layer_api import build_ffn, ffn_spec  # noqa: E402
from moe.profiling import PHASES, PhaseTimer, phase  # noqa: E402
from moe.utils import (JsonlLogger, autocast_context, get_autocast_dtype, get_device,  # noqa: E402
                       get_env_info, load_config, make_run_dir, make_run_id, query_gpu_state,
                       save_json, seed_everything)

# ========================================================================================
# Spec-sheet numbers (cited). Never edit these without updating the source fields.
# ========================================================================================

_BLACKWELL_WP = ("NVIDIA RTX Blackwell GPU Architecture whitepaper v1.1",
                 "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf")

#: RTX 5070 Ti Laptop GPU (NOT the desktop RTX 5070 Ti, which is a different chip: GB203,
#: 8960 CUDA cores, 896 GB/s). NVIDIA's laptop spec page lists: 992 AI TOPS, 5888 CUDA
#: cores, boost clock 1447-2220 MHz, 12 GB GDDR7, 672 GB/s. It does NOT list a dense BF16
#: tensor figure, so peak_tflops is DERIVED (see notes) and labelled as such.
SPEC_5070TI_LAPTOP: dict[str, Any] = {
    "kind": "spec",
    "peak_tflops": 62.0,
    "peak_gbps": 672.0,
    "source": "NVIDIA GeForce RTX 50 Series Laptops spec page (AI TOPS 992, memory bandwidth 672 GB/s); "
              "BF16 ratio from " + _BLACKWELL_WP[0] + " Appendix A-C tables",
    "source_url": "https://www.nvidia.com/en-us/geforce/laptops/50-series/",
    "source_url_ratio": _BLACKWELL_WP[1],
    "notes": (
        "Laptop SKU. peak_gbps = 672 GB/s is NVIDIA's published figure (192-bit GDDR7). "
        "peak_tflops is DERIVED, not printed by NVIDIA: dense BF16 tensor with FP32 accumulate "
        "(what PyTorch bf16 matmul uses) = AI TOPS / 16, because NVIDIA's 'AI TOPS' is FP4 "
        "with 2:4 sparsity, and in every GeForce Blackwell table of the whitepaper "
        "(RTX 5090 3352 -> 209.5, RTX 5080 1801 -> 112.6, RTX 5070 Ti 1406 -> 87.9, "
        "RTX 5070 987.8 -> 61.7) BF16-FP32-acc dense = FP4-sparse / 16. 992 / 16 = 62.0 TFLOP/s. "
        "Cross-check from clocks: 184 tensor cores x 128 dense BF16 FLOP/clk/TC (= 61.7e12 / "
        "(192 TC x 2512 MHz) for the desktop RTX 5070) x 2220 MHz (top of NVIDIA's laptop boost "
        "range) = 52.3 TFLOP/s; NVIDIA's 992 AI TOPS implies ~2.63 GHz, above that range, so the "
        "62.0 figure is the optimistic one. Laptop TGP and clocks are OEM-configured (this machine "
        "reported a 120 W cap) and GPU Boost can exceed the listed boost clock, so treat the spec "
        "roof as approximate; the measured roof is the one to compare against. No sparsity."),
    "alternatives": {"bf16_dense_fp32acc_at_2220mhz_tflops": 52.3,
                     "bf16_dense_fp16acc_from_ai_tops_tflops": 124.0},
}

#: MASTER §7 E5 reference table. Spec-sheet values, NOT measured here. Dense (no sparsity)
#: tensor throughput at the precision noted; memory bandwidth as published.
REFERENCE_GPUS: list[dict[str, Any]] = [
    {"name": "Tesla T4", "peak_tflops": 65.0, "peak_gbps": 320.0, "precision": "FP16 mixed (no BF16 support)",
     "source": "NVIDIA T4 product page ('65 FP16 TFLOPS', '320+ GB/s')",
     "source_url": "https://www.nvidia.com/en-us/data-center/tesla-t4/"},
    {"name": "Tesla V100 SXM2", "peak_tflops": 125.0, "peak_gbps": 900.0, "precision": "FP16 tensor (no BF16 support)",
     "source": "NVIDIA V100 datasheet (Tensor Performance 125 TFLOPS, 900 GB/s)",
     "source_url": "https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf"},
    {"name": "A100 SXM 80GB", "peak_tflops": 312.0, "peak_gbps": 2039.0, "precision": "BF16 tensor dense (624 with sparsity)",
     "source": "NVIDIA A100 product page spec table",
     "source_url": "https://www.nvidia.com/en-us/data-center/a100/"},
    {"name": "H100 SXM", "peak_tflops": 989.5, "peak_gbps": 3350.0,
     "precision": "BF16 tensor dense (page lists 1979 with sparsity; dense = 1979 / 2)",
     "source": "NVIDIA H100 product page spec table",
     "source_url": "https://www.nvidia.com/en-us/data-center/h100/"},
    {"name": "GeForce RTX 3090", "peak_tflops": 71.2, "peak_gbps": 936.0, "precision": "BF16 tensor dense, FP32 accumulate",
     "source": _BLACKWELL_WP[0] + ", Table 3", "source_url": _BLACKWELL_WP[1]},
    {"name": "GeForce RTX 4090", "peak_tflops": 165.2, "peak_gbps": 1008.0, "precision": "BF16 tensor dense, FP32 accumulate",
     "source": _BLACKWELL_WP[0] + ", Table 3", "source_url": _BLACKWELL_WP[1]},
    {"name": "GeForce RTX 5090", "peak_tflops": 209.5, "peak_gbps": 1792.0, "precision": "BF16 tensor dense, FP32 accumulate",
     "source": _BLACKWELL_WP[0] + ", Table 3", "source_url": _BLACKWELL_WP[1]},
    {"name": "GeForce RTX 5070 Ti (desktop, GB203; not this laptop)", "peak_tflops": 87.9, "peak_gbps": 896.0,
     "precision": "BF16 tensor dense, FP32 accumulate",
     "source": _BLACKWELL_WP[0] + ", Table 5", "source_url": _BLACKWELL_WP[1]},
]

# ========================================================================================
# Run plan (quick vs full)
# ========================================================================================

ALL_SWEEPS = ("tokens", "experts", "capacity", "skew", "gemm", "breakdown", "trace", "roofline")


def build_plan(args: argparse.Namespace, cfg: dict, device: torch.device) -> dict[str, Any]:
    """Resolve which configs to run. CLI flags override the mode defaults."""
    quick = args.quick or (not args.full and (cfg["name"] == "smoke" or device.type != "cuda"))
    tok_per_step = int(cfg["train"]["batch_size"]) * int(cfg["model"]["seq_len"])
    if quick:
        plan = dict(mode="quick", tokens=[256, 1024], expert_tokens=512, skew_tokens=[512], cap_tokens=512,
                    switch_experts=[4, 8, 16], deepseek_experts=[8, 16], warmup=3, repeats=10,
                    breakdown_repeats=5, trace_iters=2, gemm_size=1024 if device.type == "cpu" else 2048,
                    copy_mib=64, roof_repeats=5)
    else:
        plan = dict(mode="full", tokens=[2 ** p for p in range(10, 17)], expert_tokens=tok_per_step,
                    skew_tokens=[tok_per_step], cap_tokens=tok_per_step,
                    switch_experts=[4, 8, 16, 32, 64], deepseek_experts=[8, 16, 32, 64], warmup=10, repeats=50,
                    breakdown_repeats=20, trace_iters=2, gemm_size=8192, copy_mib=1024, roof_repeats=30)
    plan["variants"] = args.variants.split(",") if args.variants else ["dense", "switch", "gshard", "deepseek"]
    plan["dispatch"] = args.dispatch.split(",") if args.dispatch else ["loop", "batched"]
    plan["passes"] = args.passes.split(",") if args.passes else ["fwd", "fwd_bwd"]
    plan["sweeps"] = args.sweeps.split(",") if args.sweeps else list(ALL_SWEEPS)
    bad = set(plan["sweeps"]) - set(ALL_SWEEPS)
    if bad:
        raise SystemExit(f"unknown sweeps {sorted(bad)}; choose from {ALL_SWEEPS}")
    if args.tokens:
        plan["tokens"] = [int(t) for t in args.tokens.split(",")]
        plan["skew_tokens"] = [plan["tokens"][len(plan["tokens"]) // 2]]
    if args.expert_tokens:
        plan["expert_tokens"] = plan["cap_tokens"] = int(args.expert_tokens)
    if args.experts:
        es = [int(e) for e in args.experts.split(",")]
        plan["switch_experts"] = es
        plan["deepseek_experts"] = [e for e in es if e % 8 == 0]
    if args.repeats:
        plan["repeats"] = int(args.repeats)
        plan["breakdown_repeats"] = min(plan["breakdown_repeats"], int(args.repeats))
    if args.warmup is not None:
        plan["warmup"] = int(args.warmup)
    plan["cfs"] = [float(c) for c in args.cfs.split(",")] if args.cfs else [0.75, 1.0, 1.25, 2.0]
    plan["skew_strength"] = float(args.skew_strength)
    plan["trace_tokens"] = tok_per_step if not quick else plan["tokens"][-1]
    if args.tokens:
        plan["trace_tokens"] = plan["tokens"][-1] if quick else plan["skew_tokens"][0]
    return plan


# ========================================================================================
# Layer construction
# ========================================================================================

def layer_cfg(base: dict, variant: str, *, n_experts: Optional[int] = None, cf: Optional[float] = None) -> dict:
    """Config for one benchmarked layer. ``n_experts`` applies the experts-sweep policy
    (module docstring): Switch keeps width d_ff and top-1; DeepSeek becomes fine-grained with
    m = E/8, top-k = m, no shared expert (total routed params = 8 x d_ff, active = d_ff)."""
    cfg = copy.deepcopy(base)
    cfg["variant"] = variant
    cfg["model"]["moe_every"] = 1          # layer 0 must be the variant's FFN
    if n_experts is not None:
        if variant == "switch":
            cfg["moe"]["switch"]["n_experts"] = int(n_experts)
        elif variant == "deepseek":
            if n_experts % 8:
                raise ValueError(f"deepseek experts sweep needs E % 8 == 0 (got {n_experts})")
            m = n_experts // 8
            cfg["moe"]["deepseek"].update(m=m, n_shared=0, n_routed=int(n_experts), k_routed=m)
        else:
            raise ValueError(f"experts sweep not defined for {variant}")
    if cf is not None:
        cfg["moe"][variant]["capacity_factor"] = float(cf)
        cfg["moe"][variant]["eval_capacity_factor"] = None
    return cfg


def router_weight(layer: torch.nn.Module) -> Optional[torch.Tensor]:
    r = getattr(layer, "router", None)
    return None if r is None else r.weight


def skew_offset(layer: torch.nn.Module, strength: float) -> Optional[torch.Tensor]:
    """Constant input offset mu [d] with W_r^T mu = strength * (z - mean z), z_e = 1/(e+1)
    (min-norm least squares), so every token's router logits get a Zipf-shaped bias."""
    W = router_weight(layer)
    if W is None or strength == 0:
        return None
    with torch.no_grad():
        E = W.shape[1]
        z = 1.0 / torch.arange(1, E + 1, dtype=torch.float32, device=W.device)
        target = strength * (z - z.mean())                                  # [E]
        mu = torch.linalg.pinv(W.detach().float().t()) @ target             # [d]
    return mu


# ========================================================================================
# Timing primitives
# ========================================================================================

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_iters(fn: Callable[[], Any], device: torch.device, repeats: int) -> list[float]:
    """Per-iteration wall time in ms. CUDA: event pair per iteration, one sync at the end.
    CPU: perf_counter (ops are synchronous). Caller does the warmup."""
    _sync(device)
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for i in range(repeats):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) for s, e in zip(starts, ends)]
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def summarize(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)),
            "p90_ms": float(np.percentile(a, 90)), "mean_ms": float(a.mean())}


def mib(n_bytes: float) -> float:
    return n_bytes / 1024 ** 2


class LayerRunner:
    """Wraps one built layer + input so fwd / fwd_bwd closures share setup."""

    def __init__(self, cfg: dict, variant: str, n_tokens: int, device: torch.device,
                 amp_dtype: Optional[torch.dtype], skew_strength: float = 0.0, seed: int = 0) -> None:
        seed_everything(seed)
        self.cfg, self.variant, self.N, self.device, self.amp = cfg, variant, n_tokens, device, amp_dtype
        self.spec = ffn_spec(variant, cfg)
        self.layer = build_ffn(variant, 0, cfg).to(device)
        d = int(cfg["model"]["d_model"])
        self.x = torch.randn(1, n_tokens, d, device=device)                # fp32 residual stream
        self.mu = skew_offset(self.layer, skew_strength)
        if self.mu is not None:
            self.x = self.x + self.mu
        self.x.requires_grad_(False)
        self.dy: Optional[torch.Tensor] = None

    def set_dispatch(self, dispatch: str) -> None:
        if hasattr(self.layer, "dispatch"):
            self.layer.dispatch = dispatch

    def _forward(self):
        with autocast_context(self.device, self.amp):
            if self.spec.is_moe:
                return self.layer(self.x)
            # DenseFFN has no internal phase ranges: the whole FFN is "expert_gemm".
            with phase("expert_gemm"):
                return self.layer(self.x)

    def fwd(self) -> None:
        with torch.no_grad():
            self._forward()

    def fwd_bwd(self) -> None:
        for p in self.layer.parameters():
            p.grad = None                                  # optimizer.zero_grad(set_to_none=True)
        self.x.grad = None
        y, aux = self._forward()
        if self.dy is None or self.dy.dtype != y.dtype:
            self.dy = torch.randn_like(y)
        tensors, grads = [y], [self.dy]
        if aux.aux_loss.requires_grad:                     # backprop the balance loss too
            tensors.append(aux.aux_loss)
            grads.append(torch.ones_like(aux.aux_loss))
        torch.autograd.backward(tensors, grads)

    def prepare(self, pass_: str) -> Callable[[], None]:
        """Set train/eval mode and input grad flag for this pass; return the closure."""
        if pass_ == "fwd":
            self.layer.eval()                              # inference forward (no jitter)
            self.x.requires_grad_(False)
            return self.fwd
        self.layer.train()                                 # training step (Switch jitter on)
        self.x.requires_grad_(True)                        # a real layer also computes dx
        return self.fwd_bwd

    def routing_stats(self) -> dict[str, Any]:
        """One extra forward (no grad) to read the per-expert counts the timed run uses."""
        if not self.spec.is_moe:
            return {"expert_counts": [], "kept_counts": [], "load_imbalance": None, "drop_fraction": None}
        with torch.no_grad():
            _, aux = self._forward()
        counts = aux.expert_counts.cpu().tolist()
        mean = sum(counts) / len(counts)
        return {"expert_counts": [int(c) for c in counts],
                "kept_counts": [int(c) for c in aux.kept_counts.cpu().tolist()],
                "load_imbalance": (max(counts) / mean) if mean else None,
                "drop_fraction": float(aux.drop_fraction)}

    def free(self) -> None:
        del self.layer, self.x, self.dy
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# ========================================================================================
# Benchmarks
# ========================================================================================

class Profiler:
    """Holds the run state (device, plan, writers) and implements every sweep."""

    def __init__(self, cfg: dict, plan: dict, device: torch.device, amp: Optional[torch.dtype], run_dir: Path) -> None:
        self.cfg, self.plan, self.device, self.amp, self.run_dir = cfg, plan, device, amp, run_dir
        self.dtype_bytes = 2 if amp is not None else 4
        self.include_cast = amp is not None                # autocast re-casts fp32 weights each forward
        self.bench = JsonlLogger(run_dir / "bench.jsonl")
        self.gemm = JsonlLogger(run_dir / "gemm.jsonl")
        self.breakdown = JsonlLogger(run_dir / "breakdown.jsonl")
        self.rows: list[dict] = []
        self.gemm_rows: list[dict] = []
        self.gemm_shapes: dict[tuple, dict] = {}   # (variant, N, E, kernel) -> shape info
        self._gemm_cache: dict[tuple[int, int, int], dict] = {}

    # -- helpers -------------------------------------------------------------------------
    def gpu_state(self) -> Optional[dict]:
        return query_gpu_state() if self.device.type == "cuda" else None

    def _cost(self, variant: str, cfg: dict, N: int, counts: list[int], dispatch: str, pass_: str,
              cf: Optional[float]) -> dict:
        return flops.moe_layer_cost(
            variant, cfg, N, counts or None, dispatch="batched" if dispatch == "n/a" else dispatch,
            capacity_factor=cf, dtype_bytes=self.dtype_bytes, hidden_traffic="unfused", pass_=pass_,
            include_autocast_weight_cast=self.include_cast)

    # -- one benchmark group: fixed (sweep, variant-shape, dispatch, pass), vary a list ---
    def run_group(self, sweep: str, items: list[dict], dispatch: str, pass_: str) -> None:
        """``items``: dicts with keys variant, cfg, N, E_label, cf, skew. Runs them in order,
        recording the GPU state after the first warmup and after the last timed loop."""
        p = self.plan
        group_rows: list[dict] = []
        state_before = None
        for i, it in enumerate(items):
            variant, cfg, N = it["variant"], it["cfg"], it["N"]
            disp = dispatch if ffn_spec(variant, cfg).is_moe else "n/a"
            try:
                run = LayerRunner(cfg, variant, N, self.device, self.amp, it.get("skew", 0.0), seed=int(self.cfg["seed"]))
                run.set_dispatch(disp if disp != "n/a" else "batched")
                fn = run.prepare(pass_)
                for _ in range(p["warmup"]):
                    fn()
                _sync(self.device)
                if i == 0:
                    state_before = self.gpu_state()
                stats = run.routing_stats()
                # Peak memory of the timed loop, starting from params + input (grads freed).
                for prm in run.layer.parameters():
                    prm.grad = None
                run.x.grad = None
                _sync(self.device)
                base = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else None
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                ms = time_iters(fn, self.device, p["repeats"])
                peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM: {sweep} {variant}/{disp}/{pass_} N={N}; skipped")
                torch.cuda.empty_cache()
                continue
            spec = run.spec
            cf = it.get("cf")
            if cf is None and variant in ("switch", "gshard"):
                cf = float(cfg["moe"][variant]["capacity_factor"])
            cost = self._cost(variant, cfg, N, stats["expert_counts"], disp, pass_, cf)
            summ = summarize(ms)
            sec = summ["median_ms"] / 1e3
            label = f"{variant}/{disp}/{pass_}/N={N}/E={spec.n_routed}"
            if sweep == "capacity":
                label += f"/cf={cf:g}"
            if it.get("skew"):
                label += "/skewed"
            row = {
                "sweep": sweep, "variant": variant, "dispatch": disp, "pass": pass_, "n_tokens": N,
                "n_experts": spec.n_routed, "n_shared": spec.n_shared, "top_k": spec.top_k,
                "expert_width": spec.width, "capacity_factor": cf, "label": label,
                **summ, "repeats": len(ms),
                "flops": cost["flops"], "flops_useful": cost["flops_useful"], "bytes": cost["bytes"],
                "intensity": cost["flops"] / cost["bytes"],
                "achieved_tflops": cost["flops"] / sec / 1e12,
                "achieved_tflops_useful": cost["flops_useful"] / sec / 1e12,
                "achieved_gbps": cost["bytes"] / sec / 1e9,
                "padding_fraction": cost["padding_fraction"],
                "capacity": cost.get("capacity"),
                "executed_rows": cost.get("executed_rows"),
                "expert_counts": stats["expert_counts"],
                "load_imbalance": stats["load_imbalance"], "drop_fraction": stats["drop_fraction"],
                "routing": "skewed" if it.get("skew") else "uniform",
                "skew_strength": it.get("skew") or 0.0,
                "peak_mem_mb": mib(peak) if peak is not None else None,
                "mem_baseline_mb": mib(base) if base is not None else None,
            }
            group_rows.append(row)
            # Remember the per-expert GEMM shape this config runs (batched path rows).
            if sweep in ("tokens", "experts") and (disp in ("batched", "n/a")):
                ex = cost.get("executed_rows")
                m_rows = N if not spec.is_moe else (int(ex[0]) if ex else 0)
                d = int(cfg["model"]["d_model"])
                self.gemm_shapes[(variant, N, spec.n_routed)] = {"m": m_rows, "d": d, "w": spec.width}
            run.free()
        state_after = self.gpu_state()
        low = any((st or {}).get("mem_clock_mhz") is not None and st["mem_clock_mhz"] < 2000
                  for st in (state_before, state_after))
        for row in group_rows:
            row["gpu_state_before"], row["gpu_state_after"] = state_before, state_after
            # nvidia-smi memory clock < 2 GHz = the laptop's low-power memory state (810 MHz).
            row["low_power_state"] = bool(low)
            self.bench.log(row)
            self.rows.append(row)
            print(f"  {row['label']:<52} median {row['median_ms']:9.3f} ms  "
                  f"[{row['p10_ms']:.3f}, {row['p90_ms']:.3f}]  {row['achieved_tflops']:7.2f} TFLOP/s"
                  + (f"  pad {row['padding_fraction']*100:4.1f}%" if row['padding_fraction'] else "")
                  + (f"  imb {row['load_imbalance']:.2f}" if row['load_imbalance'] else "")
                  + (f"  peak {row['peak_mem_mb']:.0f} MiB" if row['peak_mem_mb'] is not None else ""))

    def _dispatches(self, variant: str) -> list[str]:
        return ["n/a"] if variant in ("dense", "dense_x8") else self.plan["dispatch"]

    def sweep_tokens(self) -> None:
        for variant in self.plan["variants"]:
            cfg = layer_cfg(self.cfg, variant)
            for disp in self._dispatches(variant):
                for pass_ in self.plan["passes"]:
                    items = [{"variant": variant, "cfg": cfg, "N": N} for N in self.plan["tokens"]]
                    self.run_group("tokens", items, disp, pass_)

    def sweep_experts(self) -> None:
        N = self.plan["expert_tokens"]
        for variant, es in (("switch", self.plan["switch_experts"]), ("deepseek", self.plan["deepseek_experts"])):
            if variant not in self.plan["variants"] or not es:
                continue
            for disp in self.plan["dispatch"]:
                for pass_ in self.plan["passes"]:
                    items = [{"variant": variant, "cfg": layer_cfg(self.cfg, variant, n_experts=E), "N": N} for E in es]
                    self.run_group("experts", items, disp, pass_)

    def sweep_capacity(self) -> None:
        N = self.plan["cap_tokens"]
        for variant in ("switch", "gshard"):
            if variant not in self.plan["variants"]:
                continue
            for pass_ in self.plan["passes"]:
                items = [{"variant": variant, "cfg": layer_cfg(self.cfg, variant, cf=cf), "N": N, "cf": cf}
                         for cf in self.plan["cfs"]]
                self.run_group("capacity", items, "batched", pass_)

    def sweep_skew(self) -> None:
        s = self.plan["skew_strength"]
        for variant in self.plan["variants"]:
            if variant in ("dense", "dense_x8"):
                continue
            cfg = layer_cfg(self.cfg, variant)
            for disp in self.plan["dispatch"]:
                for pass_ in self.plan["passes"]:
                    items = [{"variant": variant, "cfg": cfg, "N": N, "skew": s} for N in self.plan["skew_tokens"]]
                    self.run_group("skew", items, disp, pass_)

    # -- isolated GEMMs ------------------------------------------------------------------
    def time_gemm(self, m: int, k: int, n: int) -> dict:
        key = (m, k, n)
        if key in self._gemm_cache:
            return self._gemm_cache[key]
        dt = torch.bfloat16 if self.amp is not None else torch.float32
        a = torch.randn(m, k, device=self.device, dtype=dt)
        b = torch.randn(k, n, device=self.device, dtype=dt)
        c = torch.empty(m, n, device=self.device, dtype=dt)
        fn = lambda: torch.matmul(a, b, out=c)  # noqa: E731
        for _ in range(self.plan["warmup"]):
            fn()
        res = summarize(time_iters(fn, self.device, self.plan["repeats"]))
        self._gemm_cache[key] = res
        return res

    def gemm_points(self) -> None:
        if not self.gemm_shapes:
            print("  (no batched/dense bench rows from the tokens/experts sweeps; skipping GEMM points)")
            return
        before = self.gpu_state()
        rows = []
        for (variant, N, E), sh in sorted(self.gemm_shapes.items()):
            m, d, w = sh["m"], sh["d"], sh["w"]
            if m <= 0:
                continue
            for kernel, (k, n) in (("fwd_W_in", (d, w)), ("fwd_W_out", (w, d))):
                res = self.time_gemm(m, k, n)
                c = flops.gemm_cost(m, k, n, self.dtype_bytes)
                rows.append({"variant": variant, "n_tokens": N, "n_experts": E, "kernel": kernel,
                             "m": m, "k": k, "n": n, "rows_per_expert": m,
                             "rows_source": "batched path executed rows (capacity C / max_count / N)",
                             "flops": c["flops"], "bytes": c["bytes"], "intensity": c["intensity"],
                             **res, "achieved_tflops": c["flops"] / (res["median_ms"] / 1e3) / 1e12,
                             "label": f"{variant} {kernel} m={m} (N={N}, E={E})"})
        after = self.gpu_state()
        for r in rows:
            r["gpu_state_before"], r["gpu_state_after"] = before, after
            self.gemm.log(r)
            self.gemm_rows.append(r)
        print(f"  {len(rows)} GEMM points ({len(self._gemm_cache)} unique shapes)")

    # -- phase breakdown (manual timers) ------------------------------------------------
    def breakdown_timers(self) -> None:
        method = "cuda_events" if self.device.type == "cuda" else "perf_counter"
        R = self.plan["breakdown_repeats"]
        for variant in self.plan["variants"]:
            cfg = layer_cfg(self.cfg, variant)
            for disp in self._dispatches(variant):
                for pass_ in self.plan["passes"]:
                    for N in self.plan["tokens"]:
                        try:
                            run = LayerRunner(cfg, variant, N, self.device, self.amp, seed=int(self.cfg["seed"]))
                            run.set_dispatch(disp if disp != "n/a" else "batched")
                            fn = run.prepare(pass_)
                            for _ in range(self.plan["warmup"]):
                                fn()
                            per_phase: dict[str, list[float]] = {ph: [] for ph in PHASES}
                            totals = []
                            for _ in range(R):
                                with PhaseTimer(self.device) as t:
                                    t0 = t._stamp()
                                    fn()
                                    t1 = t._stamp()
                                ph = t.totals_ms()       # synchronizes
                                total = t0.elapsed_time(t1) if self.device.type == "cuda" else (t1 - t0) * 1e3
                                totals.append(total)
                                for name in PHASES[:-1]:
                                    per_phase[name].append(ph.get(name, 0.0))
                                per_phase["other"].append(max(0.0, total - sum(ph.values())))
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            continue
                        med = {name: statistics.median(v) for name, v in per_phase.items()}
                        s = sum(med.values()) or 1.0
                        for name in PHASES:
                            self.breakdown.log({
                                "variant": variant, "dispatch": disp, "pass": pass_, "n_tokens": N,
                                "n_experts": run.spec.n_routed, "phase": name, "time_ms": med[name],
                                "fraction": med[name] / s, "method": method, "repeats": R,
                                "total_median_ms": statistics.median(totals),
                                "other_includes": "backward pass + unannotated ops" if pass_ == "fwd_bwd"
                                else "unannotated ops (MoEAux stats, reshapes, launch gaps)"})
                        run.free()
        print("  breakdown (manual timers) done")

    # -- torch.profiler: Chrome trace + profiler-based breakdown -------------------------
    def trace_and_profiler_breakdown(self) -> Optional[str]:
        from torch.profiler import ProfilerActivity, profile, record_function

        N = self.plan["trace_tokens"]
        iters = self.plan["trace_iters"]
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.device.type == "cuda" else [])
        runs = []
        for variant in self.plan["variants"]:
            cfg = layer_cfg(self.cfg, variant)
            for disp in self._dispatches(variant):
                run = LayerRunner(cfg, variant, N, self.device, self.amp, seed=int(self.cfg["seed"]))
                run.set_dispatch(disp if disp != "n/a" else "batched")
                runs.append((variant, disp, run))
        # Warm up everything outside the profiler.
        for _, _, run in runs:
            for pass_ in ("fwd", "fwd_bwd"):
                fn = run.prepare(pass_)
                for _ in range(max(2, self.plan["warmup"] // 2)):
                    fn()
        _sync(self.device)
        with profile(activities=acts, record_shapes=False) as prof:
            for variant, disp, run in runs:
                for pass_ in ("fwd", "fwd_bwd"):
                    fn = run.prepare(pass_)
                    for _ in range(iters):
                        with record_function(f"bench::{variant}/{disp}/{pass_}/N={N}"):
                            fn()
                        _sync(self.device)
        trace_path = self.run_dir / "profiler_trace.json"
        prof.export_chrome_trace(str(trace_path))

        # Attribute kernel time to phases (forward iterations only; see module docstring).
        use_dev = self.device.type == "cuda"

        def t_us(ev) -> float:
            if use_dev:
                return float(getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0.0))
            return float(ev.cpu_time_total)

        phase_t: dict[str, dict[str, float]] = {}
        bench_t: dict[str, float] = {}
        for ev in prof.events():
            if ev.name.startswith("bench::") and ev.name.endswith(f"/fwd/N={N}"):
                bench_t[ev.name] = bench_t.get(ev.name, 0.0) + t_us(ev)
            elif ev.name.startswith("moe::"):
                par = ev.cpu_parent
                while par is not None and not par.name.startswith("bench::"):
                    par = par.cpu_parent
                if par is None or not par.name.endswith(f"/fwd/N={N}"):
                    continue
                d = phase_t.setdefault(par.name, {})
                name = ev.name[len("moe::"):]
                d[name] = d.get(name, 0.0) + t_us(ev)
        for variant, disp, run in runs:
            key = f"bench::{variant}/{disp}/fwd/N={N}"
            if key not in bench_t:
                continue
            ph = {k: v / iters / 1e3 for k, v in phase_t.get(key, {}).items()}   # ms per iteration
            total = bench_t[key] / iters / 1e3
            ph["other"] = max(0.0, total - sum(ph.values()))
            s = sum(ph.values()) or 1.0
            for name in PHASES:
                self.breakdown.log({
                    "variant": variant, "dispatch": disp, "pass": "fwd", "n_tokens": N,
                    "n_experts": run.spec.n_routed, "phase": name, "time_ms": ph.get(name, 0.0),
                    "fraction": ph.get(name, 0.0) / s, "method": "torch.profiler", "repeats": iters,
                    "time_basis": "kernel device time" if use_dev else "CPU op time",
                    "other_includes": "kernels in the bench range outside moe:: phase ranges"})
            run.free()
        print(f"  Chrome trace: {trace_path.relative_to(REPO_ROOT)} ({trace_path.stat().st_size / 1e6:.1f} MB)")
        return str(trace_path)

    # -- roofline ceilings ---------------------------------------------------------------
    def wake_gpu(self, seconds: float) -> Optional[dict]:
        """Run back-to-back 4096^3 GEMMs for ``seconds`` so a laptop GPU leaves its low-power
        state (this machine flips between a 14 GHz and an 810 MHz memory clock) before any
        measurement. Returns the GPU state at the end."""
        if self.device.type != "cuda" or seconds <= 0:
            return None
        dt = torch.bfloat16 if self.amp is not None else torch.float32
        a = torch.randn(4096, 4096, device=self.device, dtype=dt)
        t0 = time.time()
        while time.time() - t0 < seconds:
            for _ in range(20):
                a @ a
            torch.cuda.synchronize(self.device)
        del a
        return self.gpu_state()

    def measure_ceilings(self) -> dict:
        """Measured roof: square GEMMs (bf16 on CUDA) and a large device-to-device copy.

        GEMM: several square sizes (gemm_size/2, 3*gemm_size/4, gemm_size); the roof is the
        best median. One size is not enough on this laptop: a long 8192^3 GEMM pulls the
        power limit and drops the SM clock (~1.9 GHz), while shorter GEMMs run at ~2.4 GHz
        boost, so a single big GEMM under-states what short expert GEMMs can reach.
        Copy bandwidth counts read + write (2 x bytes), the STREAM 'copy' convention."""
        p = self.plan
        n_max = p["gemm_size"]
        sizes = sorted({max(256, n_max // 2), max(256, (3 * n_max) // 4), n_max})
        copy_bytes = p["copy_mib"] * 1024 ** 2
        dt = torch.bfloat16 if self.amp is not None else torch.float32
        before = self.gpu_state()
        gemm_all = []
        for n in sizes:
            a = torch.randn(n, n, device=self.device, dtype=dt)
            b = torch.randn(n, n, device=self.device, dtype=dt)
            c = torch.empty(n, n, device=self.device, dtype=dt)
            g = lambda: torch.matmul(a, b, out=c)  # noqa: E731
            for _ in range(max(3, p["warmup"])):
                g()
            g_ms = summarize(time_iters(g, self.device, p["roof_repeats"]))
            gemm_all.append({"shape": [n, n, n], **g_ms, "tflops": 2 * n ** 3 / (g_ms["median_ms"] / 1e3) / 1e12})
            del a, b, c
        best = max(gemm_all, key=lambda r: r["tflops"])
        src = torch.empty(copy_bytes, dtype=torch.uint8, device=self.device).random_(0, 255)
        dst = torch.empty_like(src)
        cp = lambda: dst.copy_(src)  # noqa: E731
        for _ in range(max(3, p["warmup"])):
            cp()
        c_ms = summarize(time_iters(cp, self.device, p["roof_repeats"]))
        del src, dst
        after = self.gpu_state()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gbps = 2 * copy_bytes / (c_ms["median_ms"] / 1e3) / 1e9
        return {"kind": "measured", "peak_tflops": best["tflops"], "peak_gbps": gbps,
                "method": (f"{'bf16' if dt == torch.bfloat16 else 'fp32'} square GEMM, best median over sizes "
                           f"{sizes} ({p['roof_repeats']} repeats each); D2D copy of {p['copy_mib']} MiB "
                           f"(read+write counted) median of {p['roof_repeats']}"),
                "gemm_shape": best["shape"], "gemm_all": gemm_all, "copy_bytes": copy_bytes,
                "repeats": p["roof_repeats"], "gemm_ms": {k: best[k] for k in ("median_ms", "p10_ms", "p90_ms", "mean_ms")},
                "copy_ms": c_ms, "device": self.device.type,
                "gpu_state_before": before, "gpu_state_after": after}

    def close(self) -> None:
        self.bench.close()
        self.gemm.close()
        self.breakdown.close()


def other_gpu_processes() -> Optional[list[str]]:
    """PIDs of other processes holding a CUDA context (nvidia-smi), to flag contention.
    None if nvidia-smi is unavailable. Under WSL process names show as [Not Found]."""
    import os
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    me = str(os.getpid())
    return [ln.strip() for ln in out.splitlines() if ln.strip() and ln.strip() != me]


def gpu_busy() -> Optional[dict]:
    """GPU utilization (%) and used memory (MiB) right now, before this process allocates."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        u, m = [float(v) for v in out.split(",")]
        return {"utilization_pct": u, "memory_used_mib": m}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def spec_ceiling(device: torch.device, args: argparse.Namespace) -> dict:
    """The cited laptop spec (CUDA on the RTX 5070 Ti Laptop GPU), a user override, or null."""
    if args.peak_tflops is not None or args.peak_gbps is not None:
        return {"kind": "spec", "peak_tflops": args.peak_tflops, "peak_gbps": args.peak_gbps,
                "source": "user-entered via --peak-tflops / --peak-gbps", "source_url": None,
                "notes": "custom peaks entered on the command line; not verified",
                "cited_default": SPEC_5070TI_LAPTOP}
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    if "5070 Ti Laptop" in name:
        return dict(SPEC_5070TI_LAPTOP)
    return {"kind": "spec", "peak_tflops": None, "peak_gbps": None,
            "source": None, "source_url": None,
            "notes": (f"this run executed on {name or device.type.upper()}, not the RTX 5070 Ti Laptop GPU, so "
                      "no spec-sheet ceiling applies (pass --peak-tflops/--peak-gbps to enter one). The cited "
                      "laptop values are kept in 'laptop_reference' for information only."),
            "laptop_reference": SPEC_5070TI_LAPTOP}


# ========================================================================================
# main
# ========================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="tiny CPU-feasible subset (auto for smoke / no CUDA)")
    mode.add_argument("--full", action="store_true", help="force the full sweep even on the smoke config")
    ap.add_argument("--variants", help="comma list (default dense,switch,gshard,deepseek; dense_x8 allowed)")
    ap.add_argument("--dispatch", help="comma list of loop,batched")
    ap.add_argument("--passes", help="comma list of fwd,fwd_bwd")
    ap.add_argument("--tokens", help="comma list of token counts for the tokens sweep / breakdown")
    ap.add_argument("--experts", help="comma list of expert counts for the experts sweep")
    ap.add_argument("--expert-tokens", help="N for the experts and capacity sweeps")
    ap.add_argument("--cfs", help="capacity factors for the capacity sweep (default 0.75,1,1.25,2)")
    ap.add_argument("--sweeps", help=f"comma list from {','.join(ALL_SWEEPS)} (default all)")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--wake-seconds", type=float, default=None,
                    help="sustained GEMM load before measuring (default 5 s full / 1 s quick on CUDA)")
    ap.add_argument("--skew-strength", type=float, default=0.3,
                    help="Zipf logit offset scale for the skewed-routing sweep (nats)")
    ap.add_argument("--peak-tflops", type=float, help="custom spec peak TFLOP/s (replaces the cited value)")
    ap.add_argument("--peak-gbps", type=float, help="custom spec peak GB/s (replaces the cited value)")
    ap.add_argument("--tag", help="extra run-id tag")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = get_device(cfg)
    amp = get_autocast_dtype(cfg, device)
    plan = build_plan(args, cfg, device)
    run_id = make_run_id("profile", args.tag)
    run_dir = make_run_dir(cfg, f"{cfg['name']}/e5_profile", run_id)
    env = get_env_info(device, amp)
    save_json(run_dir / "env.json", env)
    print(f"E5 profile [{plan['mode']}] on {env['hardware_label']} ({device}, "
          f"{'bf16 autocast' if amp else 'fp32'}) -> {run_dir.relative_to(REPO_ROOT)}")
    if device.type == "cuda":
        st = query_gpu_state()
        print(f"  GPU state at start: {st}")
        if st and st.get("mem_clock_mhz") is not None and st["mem_clock_mhz"] < 2000:
            print("  WARNING: memory clock is in the low-power state (<2 GHz). Results will be throttled; "
                  "plug in, set G-Helper Turbo, disable Eco mode.")

    others_start = other_gpu_processes() if device.type == "cuda" else None
    if others_start:
        print(f"  WARNING: other GPU processes running (pids {others_start}); timings may be contended")
    busy_start = gpu_busy() if device.type == "cuda" else None
    if busy_start and (busy_start.get("utilization_pct") or 0) > 20:
        # Under WSL, processes from other sandboxes/namespaces may be invisible to
        # --query-compute-apps, so idle utilization is the more reliable contention check.
        print(f"  WARNING: GPU already {busy_start['utilization_pct']:.0f}% busy before this run "
              f"({busy_start.get('memory_used_mib')} MiB used by others); timings will be contended")
    t_start = time.time()
    prof = Profiler(cfg, plan, device, amp, run_dir)
    sweeps = plan["sweeps"]
    measured_start = measured_end = None
    if "roofline" in sweeps:
        print("[roofline] measuring ceilings (start)")
        wake = args.wake_seconds if args.wake_seconds is not None else (5.0 if plan["mode"] == "full" else 1.0)
        st = prof.wake_gpu(wake)
        if st is not None:
            print(f"  GPU state after {wake:g} s wake-up load: {st}")
        measured_start = prof.measure_ceilings()
        print(f"  measured: {measured_start['peak_tflops']:.2f} TFLOP/s GEMM, {measured_start['peak_gbps']:.1f} GB/s copy")
    if "tokens" in sweeps:
        print("[tokens sweep]")
        prof.sweep_tokens()
    if "experts" in sweeps:
        print("[experts sweep] (activated compute fixed; see docstring)")
        prof.sweep_experts()
    if "capacity" in sweeps:
        print("[capacity sweep]")
        prof.sweep_capacity()
    if "skew" in sweeps:
        print(f"[skewed routing] strength {plan['skew_strength']}")
        prof.sweep_skew()
    if "gemm" in sweeps:
        print("[GEMM points]")
        prof.gemm_points()
    if "breakdown" in sweeps:
        print("[breakdown: manual timers]")
        prof.breakdown_timers()
    trace_file = None
    if "trace" in sweeps:
        print("[torch.profiler trace]")
        trace_file = prof.trace_and_profiler_breakdown()
    if "roofline" in sweeps:
        print("[roofline] re-measuring ceilings (end)")
        measured_end = prof.measure_ceilings()
        print(f"  measured: {measured_end['peak_tflops']:.2f} TFLOP/s GEMM, {measured_end['peak_gbps']:.1f} GB/s copy")
    prof.close()
    others_end = other_gpu_processes() if device.type == "cuda" else None
    if others_end:
        print(f"  WARNING: other GPU processes running at the end (pids {others_end}); timings may be contended")

    # roofline.json: the measured ceiling is the START measurement; the end one is kept to
    # show drift (throttling). The plotted roof uses the max of the two so no honest point
    # can sit above it because of a clock change between start and end.
    if measured_start is not None:
        measured = dict(measured_start)
        measured["peak_tflops"] = max(measured_start["peak_tflops"], measured_end["peak_tflops"])
        measured["peak_gbps"] = max(measured_start["peak_gbps"], measured_end["peak_gbps"])
        measured["start"] = {k: measured_start[k] for k in ("peak_tflops", "peak_gbps", "gpu_state_before", "gpu_state_after")}
        measured["end"] = {k: measured_end[k] for k in ("peak_tflops", "peak_gbps", "gpu_state_before", "gpu_state_after")}
        measured["gpu_state_after"] = measured_end["gpu_state_after"]
        measured["note"] = "peak = max(start, end) measurement; see start/end for drift"
        def _mem(stt):
            return (stt or {}).get("mem_clock_mhz")
        mems = [_mem(measured_start["gpu_state_before"]), _mem(measured_start["gpu_state_after"]),
                _mem(measured_end["gpu_state_before"]), _mem(measured_end["gpu_state_after"])]
        drift = max(measured_start["peak_tflops"], measured_end["peak_tflops"]) / \
            max(1e-9, min(measured_start["peak_tflops"], measured_end["peak_tflops"]))
        measured["power_state_changed"] = bool(len({m for m in mems if m is not None}) > 1 or drift > 1.25)
        if measured["power_state_changed"]:
            print(f"  WARNING: GPU power state changed during the run (mem clocks {mems}, start/end GEMM "
                  f"ratio {drift:.2f}). Timings from different states are not comparable; rerun plugged in / Turbo.")
        roof = {"schema_version": 1, "config_name": cfg["name"], "device_label": env["hardware_label"],
                "dtype": "bf16" if amp else "fp32",
                "ceilings": [spec_ceiling(device, args), measured],
                "reference_gpus": REFERENCE_GPUS,
                "reference_note": "spec-sheet values (not measured here)"}
        save_json(run_dir / "roofline.json", roof)

        # Physical-plausibility check (MASTER 1.5.3): nothing may beat the measured roof.
        peak = measured["peak_tflops"]
        worst = [r for r in prof.rows + prof.gemm_rows if r["achieved_tflops"] > peak]
        if worst:
            print(f"  WARNING: {len(worst)} points exceed the measured GEMM peak ({peak:.2f} TFLOP/s): "
                  + ", ".join(r["label"] for r in worst[:5]))
        else:
            mx = max((r["achieved_tflops"] for r in prof.rows + prof.gemm_rows), default=0.0)
            print(f"  check OK: max achieved {mx:.2f} TFLOP/s <= measured peak {peak:.2f} TFLOP/s")

    save_json(run_dir / "profile_meta.json", {
        "schema_version": 1, "run_id": run_id, "config_name": cfg["name"],
        "created_at": env["timestamp"], "d_model": int(cfg["model"]["d_model"]),
        "base_d_ff": int(cfg["model"]["d_ff"]), "dtype": "bf16-autocast (router fp32)" if amp else "fp32",
        "device_label": env["hardware_label"], "mode": plan["mode"],
        "warmup_iters": plan["warmup"], "repeats": plan["repeats"],
        "breakdown_repeats": plan["breakdown_repeats"],
        "timer": "cuda_events" if device.type == "cuda" else "perf_counter",
        "flops_model": {"hidden_traffic": "unfused", "include_autocast_weight_cast": amp is not None,
                        "dtype_bytes": 2 if amp else 4, "note": "compulsory traffic -> intensity is an upper bound"},
        "router_init": "random",
        "plan": {k: v for k, v in plan.items()},
        "experts_sweep_policy": ("activated compute per token fixed at d_ff. switch: E experts of width d_ff, "
                                 "top-1 (total params grow with E, Switch Fig. 4). deepseek: fine-grained, total "
                                 "routed params 8*d_ff, width 8*d_ff/E, top-k=E/8, no shared expert."),
        "skew_method": ("x += mu, mu = pinv(W_r^T) @ (s * (z - mean z)), z_e = 1/(e+1): Zipf-shaped router-logit "
                        "offset of strength s nats; realised imbalance is in each row's load_imbalance"),
        "gpu_state_policy": "queried after the first warmup and after the last timed loop of each group",
        "trace_file": str(Path(trace_file).relative_to(REPO_ROOT / cfg["system"]["results_dir"])) if trace_file else None,
        "wall_time_s": time.time() - t_start,
        "gpu_busy_before_run": busy_start,
        "other_gpu_processes": {"start": others_start, "end": others_end,
                                "note": "other CUDA processes seen by nvidia-smi (contention check); "
                                        "a job that started and ended mid-run is not detected"},
        "note": ("Single FFN layer only (no attention/norm/residual). fwd = inference forward under no_grad in "
                 "eval mode; fwd_bwd = train-mode forward + backward of y (random dy) and the aux loss, grads "
                 "set to None each iteration. Random input -> near-uniform routing unless sweep == 'skew'."),
    })
    print(f"done in {time.time() - t_start:.1f} s -> {run_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
