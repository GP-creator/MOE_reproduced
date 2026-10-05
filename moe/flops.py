"""Analytic parameter, FLOP and byte counts (PLAN §5; feeds the startup table, the dashboard,
``tests/test_param_matching.py`` and E5's roofline).

Counting conventions (PLAN §5):
  * No biases anywhere. An FFN / expert of width w is ``W_in[d, w] + W_out[w, d]`` = 2*d*w params.
  * 1 multiply-accumulate = 2 FLOPs. GELU, softmax, norms, gate multiplies and residual adds
    are NOT counted in FLOPs.
  * Router params/layer = d * n_routed (fp32, no bias). The router runs on every token, so
    it counts as active.
  * Attention/layer = 4*d^2 (Wq, Wk, Wv, Wo). Norms/layer = 2*d (two RMSNorm weights).
    Final norm = d. Tied embedding counted once = V*d. RoPE has no params.
  * Total params = V*d + sum_l (4d^2 + 2d + totFFN_l + router_l) + d.
    Active params/token = same with actFFN_l (shared + top-k experts) instead of totFFN_l.
  * Whole-model fwd FLOPs/token = sum_l (8d^2 + 2*actFFN_l + 2*d*n_routed_l + 4*seq*d) + 2*V*d.
    ``4*seq*d`` is QK^T plus AV with the full (non-causal) seq length, so it is an upper
    bound; ``2*V*d`` is the tied output head.
  * ``model.moe_every`` is honored with :func:`moe.layer_api.is_moe_layer`; non-MoE blocks
    are counted as a plain dense FFN of width d_ff with no router.

Byte counts for E5 (:func:`gemm_cost`, :func:`expert_ffn_cost`, :func:`moe_layer_cost`) are a
*compulsory-traffic* model: every GEMM operand is read from DRAM exactly once and the output
is written exactly once (perfect on-chip reuse). That is the minimum traffic, so the
arithmetic intensity is an upper bound (the standard roofline convention). Assumptions are
repeated in each function's docstring.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from moe.layer_api import ALL_VARIANTS, FFNSpec, expert_capacity, ffn_spec, is_moe_layer

# ========================================================================================
# Part 1: parameter / FLOP table (PLAN §5)
# ========================================================================================

_EXPERT_LABEL = {
    "dense": lambda s: "1",
    "dense_x8": lambda s: "1",
    "switch": lambda s: f"{s.n_routed} routed",
    "gshard": lambda s: f"{s.n_routed} routed",
    "deepseek": lambda s: f"{s.n_shared} shared + {s.n_routed} routed",
}
_ACTIVE_LABEL = {
    "dense": lambda s: "1",
    "dense_x8": lambda s: "1",
    "switch": lambda s: f"top-{s.top_k}",
    "gshard": lambda s: f"top-{s.top_k}",
    "deepseek": lambda s: f"{s.n_shared} + top-{s.top_k}",
}


def ffn_total_params(spec: FFNSpec, d: int) -> int:
    """Total FFN params of one layer: 2*d*w*(n_shared + n_routed)."""
    return 2 * d * spec.width * (spec.n_shared + spec.n_routed)


def ffn_active_params(spec: FFNSpec, d: int) -> int:
    """Activated FFN params per token of one layer: 2*d*w*(n_shared + k)."""
    return 2 * d * spec.width * (spec.n_shared + spec.top_k)


def router_params(spec: FFNSpec, d: int) -> int:
    """Router params of one layer: d*n_routed (0 for dense)."""
    return d * spec.n_routed


def layer_specs(variant: str, cfg: Mapping[str, Any]) -> list[FFNSpec]:
    """FFNSpec for each of the n_layers blocks, honoring ``model.moe_every``."""
    m = cfg["model"]
    every = int(m.get("moe_every", 1))
    return [ffn_spec(variant, cfg, moe_layer=is_moe_layer(i, every)) for i in range(int(m["n_layers"]))]


def param_row(variant: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """One row of the param/FLOP table for ``variant`` under config ``cfg``.

    Keys ("per_layer" = one block that uses the variant's FFN, i.e. a MoE position):
      variant, experts, expert_width, activated, n_layers, n_moe_layers,
      embedding_params, attn_params_per_layer, norm_params_per_layer,
      ffn_total_per_layer, ffn_active_per_layer, router_per_layer,
      ffn_total_model, ffn_active_model, router_model,
      total_params, active_params,
      ffn_flops_per_token_per_layer, ffn_flops_per_token_model,
      router_flops_per_token_model, attn_flops_per_token_model, head_flops_per_token,
      model_fwd_flops_per_token
    """
    m = cfg["model"]
    d, L, V, seq = int(m["d_model"]), int(m["n_layers"]), int(cfg["data"]["vocab_size"]), int(m["seq_len"])
    specs = layer_specs(variant, cfg)
    main = ffn_spec(variant, cfg)  # the spec at MoE positions (what the table shows)
    every = int(m.get("moe_every", 1))
    n_moe = sum(is_moe_layer(i, every) for i in range(L))

    emb = V * d
    attn = 4 * d * d
    norms = 2 * d
    tot_ffn = [ffn_total_params(s, d) for s in specs]
    act_ffn = [ffn_active_params(s, d) for s in specs]
    rtr = [router_params(s, d) for s in specs]

    total = emb + sum(attn + norms + t + r for t, r in zip(tot_ffn, rtr)) + d
    active = emb + sum(attn + norms + a + r for a, r in zip(act_ffn, rtr)) + d

    ffn_flops = [2 * a for a in act_ffn]
    router_flops = [2 * r for r in rtr]  # 2*d*n_routed per token
    attn_flops = L * (8 * d * d + 4 * seq * d)
    head = 2 * V * d
    return {
        "variant": variant,
        "experts": _EXPERT_LABEL[variant](main),
        "expert_width": main.width,
        "activated": _ACTIVE_LABEL[variant](main),
        "n_layers": L,
        "n_moe_layers": n_moe,
        "embedding_params": emb,
        "attn_params_per_layer": attn,
        "norm_params_per_layer": norms,
        "ffn_total_per_layer": ffn_total_params(main, d),
        "ffn_active_per_layer": ffn_active_params(main, d),
        "router_per_layer": router_params(main, d),
        "ffn_total_model": sum(tot_ffn),
        "ffn_active_model": sum(act_ffn),
        "router_model": sum(rtr),
        "total_params": total,
        "active_params": active,
        "ffn_flops_per_token_per_layer": 2 * ffn_active_params(main, d),
        "ffn_flops_per_token_model": sum(ffn_flops),
        "router_flops_per_token_model": sum(router_flops),
        "attn_flops_per_token_model": attn_flops,
        "head_flops_per_token": head,
        "model_fwd_flops_per_token": attn_flops + sum(ffn_flops) + sum(router_flops) + head,
    }


def param_table(cfg: Mapping[str, Any], variants: Iterable[str] = ALL_VARIANTS) -> list[dict[str, Any]]:
    """The param/FLOP table for all (or the given) variants under ``cfg``."""
    return [param_row(v, cfg) for v in variants]


_TABLE_COLS = [
    ("variant", "variant"),
    ("experts", "experts"),
    ("expert_width", "width"),
    ("activated", "active/tok"),
    ("ffn_total_per_layer", "FFN total/layer"),
    ("ffn_active_per_layer", "FFN active/layer"),
    ("router_per_layer", "router/layer"),
    ("total_params", "total params"),
    ("active_params", "active params"),
    ("ffn_flops_per_token_model", "FFN fwd FLOPs/tok"),
    ("model_fwd_flops_per_token", "model fwd FLOPs/tok"),
]


def format_param_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Plain-text table for the startup printout (exact integers with thousands separators)."""
    def fmt(v: Any) -> str:
        return f"{v:,}" if isinstance(v, int) else str(v)

    cells = [[h for _, h in _TABLE_COLS]] + [[fmt(r[k]) for k, _ in _TABLE_COLS] for r in rows]
    widths = [max(len(row[j]) for row in cells) for j in range(len(_TABLE_COLS))]
    lines = []
    for i, row in enumerate(cells):
        lines.append("  ".join(c.rjust(w) if j >= 4 else c.ljust(w) for j, (c, w) in enumerate(zip(row, widths))))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    if rows and rows[0]["n_moe_layers"] != rows[0]["n_layers"]:
        lines.append(f"(moe_every: {rows[0]['n_moe_layers']}/{rows[0]['n_layers']} blocks use the variant FFN; "
                     "per-layer columns describe those blocks)")
    return "\n".join(lines)


# ========================================================================================
# Part 2: FLOPs and bytes moved for E5 (roofline)
# ========================================================================================

def _cost(flops: float, bytes_: float, **extra: Any) -> dict[str, Any]:
    out = {"flops": flops, "bytes": bytes_, "intensity": (flops / bytes_) if bytes_ else float("inf")}
    out.update(extra)
    return out


def gemm_cost(m: int, k: int, n: int, dtype_bytes: int = 2) -> dict[str, Any]:
    """C[m, n] = A[m, k] @ B[k, n].

    FLOPs = 2*m*k*n. Bytes = (m*k + k*n + m*n) * dtype_bytes: A and B read once, C written
    once (compulsory traffic, perfect reuse). Intensity = FLOPs / bytes (FLOP/byte).
    """
    return _cost(2 * m * k * n, (m * k + k * n + m * n) * dtype_bytes, m=m, k=k, n=n)


def expert_ffn_cost(
    n_tokens: int,
    d: int,
    w: int,
    dtype_bytes: int = 2,
    *,
    hidden_traffic: str = "unfused",
    pass_: str = "fwd",
    include_autocast_weight_cast: bool = False,
    master_weight_bytes: int = 4,
) -> dict[str, Any]:
    """FLOPs and bytes for ONE expert's GEMM pair on ``n_tokens`` rows.

    Computation: ``h = x[n, d] @ W_in[d, w]``, ``a = GELU(h)``, ``y = a @ W_out[w, d]``.

    Assumptions:
      * Each weight matrix is read once per forward per expert (2*d*w elements), at
        ``dtype_bytes`` (the bf16 copy autocast feeds to the GEMM). The fp32->bf16 autocast
        cast kernel is NOT counted unless ``include_autocast_weight_cast=True``, which adds
        ``2*d*w*(master_weight_bytes + dtype_bytes)`` (read fp32 master, write bf16 copy).
        Under autocast this cast really happens on every forward, so turn it on when
        comparing to measured eager-mode time.
      * Activations: x read once (n*d), y written once (n*d).
      * The hidden activation (n*w) depends on ``hidden_traffic``:
          ``"fused"``   : never leaves the chip (ideal fused kernel) -> 0 bytes.
          ``"gemm"``    : GEMM1 writes h, GEMM2 reads it (GELU fused into one of them) -> 2*n*w.
          ``"unfused"`` : eager PyTorch: GEMM1 writes h, GELU reads h and writes a, GEMM2
                          reads a -> 4*n*w. This is what our eager code does (default).
      * GELU FLOPs are not counted (PLAN §5 convention).
      * ``pass_="fwd_bwd"``: adds the backward. For each GEMM Y = X W there are two backward
        GEMMs with the same FLOPs (dX = dY W^T, dW = X^T dY), each costed with
        :func:`gemm_cost`, so FLOPs are 3x forward. The GELU backward reads h and the
        incoming grad and writes the outgoing grad (3*n*w elements; 0 for "fused").
        Optimizer traffic is not included.

    Returns a dict with ``flops``, ``bytes``, ``intensity`` and a per-kernel breakdown
    ``kernels`` (list of dicts), so each GEMM can also be placed on the roofline separately.
    """
    if hidden_traffic not in ("fused", "gemm", "unfused"):
        raise ValueError(f"hidden_traffic must be fused|gemm|unfused, got {hidden_traffic!r}")
    if pass_ not in ("fwd", "fwd_bwd"):
        raise ValueError(f"pass_ must be fwd|fwd_bwd, got {pass_!r}")
    b = dtype_bytes
    kernels: list[dict[str, Any]] = []

    # Forward GEMMs, with their compulsory traffic.
    g1 = gemm_cost(n_tokens, d, w, b)
    g2 = gemm_cost(n_tokens, w, d, b)
    g1["name"], g2["name"] = "fwd_W_in", "fwd_W_out"
    kernels += [g1, g2]
    # gemm_cost already counts h written by GEMM1 and a read by GEMM2 (2*n*w) = "gemm".
    if hidden_traffic == "unfused":
        kernels.append(_cost(0, 2 * n_tokens * w * b, name="fwd_gelu"))
    elif hidden_traffic == "fused":
        # Remove the h/a round trip from the two GEMMs' bytes.
        g1["bytes"] -= n_tokens * w * b
        g2["bytes"] -= n_tokens * w * b
        g1["intensity"] = g1["flops"] / g1["bytes"]
        g2["intensity"] = g2["flops"] / g2["bytes"]

    if pass_ == "fwd_bwd":
        # GEMM2 backward: da = dy @ W_out^T  [n,d]x[d,w];  dW_out = a^T @ dy  [w,n]x[n,d]
        kernels.append(dict(gemm_cost(n_tokens, d, w, b), name="bwd_dA"))
        kernels.append(dict(gemm_cost(w, n_tokens, d, b), name="bwd_dW_out"))
        if hidden_traffic != "fused":
            kernels.append(_cost(0, 3 * n_tokens * w * b, name="bwd_gelu"))
        # GEMM1 backward: dx = dh @ W_in^T  [n,w]x[w,d];  dW_in = x^T @ dh  [d,n]x[n,w]
        kernels.append(dict(gemm_cost(n_tokens, w, d, b), name="bwd_dX"))
        kernels.append(dict(gemm_cost(d, n_tokens, w, b), name="bwd_dW_in"))

    if include_autocast_weight_cast:
        kernels.append(_cost(0, 2 * d * w * (master_weight_bytes + b), name="autocast_weight_cast"))

    flops = sum(k["flops"] for k in kernels)
    bytes_ = sum(k["bytes"] for k in kernels)
    return _cost(flops, bytes_, n_tokens=n_tokens, d=d, w=w, dtype_bytes=b,
                 hidden_traffic=hidden_traffic, pass_=pass_, kernels=kernels)


def moe_layer_cost(
    variant: str,
    cfg: Mapping[str, Any],
    n_tokens: int,
    expert_counts: Optional[Sequence[int]] = None,
    *,
    dispatch: str = "batched",
    capacity_factor: Optional[float] = None,
    dtype_bytes: int = 2,
    router_bytes: int = 4,
    hidden_traffic: str = "unfused",
    pass_: str = "fwd",
    include_autocast_weight_cast: bool = False,
    disable_shared: bool = False,
) -> dict[str, Any]:
    """FLOPs and bytes for one whole FFN layer (any variant) on ``n_tokens`` tokens, given the
    actual per-expert assignment counts.

    Args:
      expert_counts: assignments per ROUTED expert BEFORE dropping (``MoEAux.expert_counts``),
        length n_routed. Ignored for dense/dense_x8. If None, a perfectly balanced split of
        ``top_k * n_tokens`` assignments is assumed.
      dispatch: ``"batched"`` or ``"loop"`` (PLAN §7):
        * Switch/GShard batched: every expert runs a capacity-padded buffer of C rows
          (C from :func:`moe.layer_api.expert_capacity`, cf = ``capacity_factor`` or the
          config's training cf), so padding rows cost real FLOPs and bytes.
        * DeepSeek batched: every routed expert runs ``max(counts)`` rows (padded buffer).
        * loop: each expert runs exactly its kept rows; an expert with 0 rows is skipped and
          its weights are not read.
      dtype_bytes: activation/weight element size inside the experts (2 for bf16, 4 for fp32).
      router_bytes: router element size (4: the router is fp32, PLAN §7).

    Assumptions (in addition to :func:`expert_ffn_cost`):
      * Kept rows per expert = min(count, C) (true for any priority order, Switch/GShard).
      * Router: one fp32 GEMM [N, d] x [d, E] (FLOPs 2*N*d*E); softmax/top-k not counted.
      * Dispatch/combine data movement (``dispatch_bytes``), counted only for routed experts:
        batched: read kept rows of x (kept*d) + write the whole buffer (executed rows * d,
        which includes the zero-fill of padding) + read kept output rows (kept*d) + write
        y (N*d). loop: gather read+write (2*kept*d) + index_add read+write (2*kept*d).
        Index and gate tensors are ignored. For ``pass_="fwd_bwd"`` dispatch bytes are
        doubled (the backward is the transpose gather/scatter).
      * ``flops_useful`` counts only kept rows; ``flops`` (executed) includes padding rows.
        ``padding_fraction`` = 1 - useful/executed FFN FLOPs.
    """
    m = cfg["model"]
    d = int(m["d_model"])
    spec = ffn_spec(variant, cfg)
    b = dtype_bytes
    kw = dict(hidden_traffic=hidden_traffic, pass_=pass_, include_autocast_weight_cast=include_autocast_weight_cast)

    if not spec.is_moe:
        c = expert_ffn_cost(n_tokens, d, spec.width, b, **kw)
        return _cost(c["flops"], c["bytes"], flops_useful=c["flops"], padding_fraction=0.0,
                     variant=variant, n_tokens=n_tokens, experts=[c], router=None, shared=None,
                     dispatch_bytes=0, capacity=None)

    E, k, w = spec.n_routed, spec.top_k, spec.width
    if expert_counts is None:
        base, rem = divmod(k * n_tokens, E)
        expert_counts = [base + (1 if i < rem else 0) for i in range(E)]
    counts = [int(c) for c in expert_counts]
    if len(counts) != E:
        raise ValueError(f"{variant}: expected {E} expert counts, got {len(counts)}")

    capacity = None
    if variant in ("switch", "gshard"):
        sub = cfg["moe"][variant]
        cf = float(sub["capacity_factor"]) if capacity_factor is None else float(capacity_factor)
        capacity = expert_capacity(n_tokens, E, cf, top_k=k)
        kept = [min(c, capacity) for c in counts]
    else:
        kept = counts
    if dispatch == "batched":
        executed = [capacity] * E if capacity is not None else [max(kept)] * E
    elif dispatch == "loop":
        executed = kept
    else:
        raise ValueError(f"dispatch must be batched|loop, got {dispatch!r}")

    experts = []
    for rows in executed:
        experts.append(expert_ffn_cost(rows, d, w, b, **kw) if rows > 0
                       else _cost(0, 0, n_tokens=0, d=d, w=w, kernels=[]))
    useful = sum(expert_ffn_cost(r, d, w, b, **kw)["flops"] for r in kept if r > 0)

    bwd_mult = 3 if pass_ == "fwd_bwd" else 1
    router = _cost(2 * n_tokens * d * E * bwd_mult, (n_tokens * d + d * E + n_tokens * E) * router_bytes * bwd_mult)

    shared = None
    if spec.n_shared and not disable_shared:
        shared = expert_ffn_cost(n_tokens, d, w * spec.n_shared, b, **kw)

    n_kept, n_exec = sum(kept), sum(executed)
    if dispatch == "batched":
        disp = (n_kept * d + n_exec * d + n_kept * d + n_tokens * d) * b
    else:
        disp = 4 * n_kept * d * b
    disp *= 2 if pass_ == "fwd_bwd" else 1

    ffn_exec = sum(e["flops"] for e in experts)
    flops = ffn_exec + router["flops"] + (shared["flops"] if shared else 0)
    flops_useful = useful + router["flops"] + (shared["flops"] if shared else 0)
    bytes_ = sum(e["bytes"] for e in experts) + router["bytes"] + (shared["bytes"] if shared else 0) + disp
    return _cost(
        flops, bytes_,
        flops_useful=flops_useful,
        padding_fraction=(1.0 - useful / ffn_exec) if ffn_exec else 0.0,
        variant=variant, n_tokens=n_tokens, capacity=capacity,
        expert_counts=counts, kept_rows=kept, executed_rows=executed,
        experts=experts, router=router, shared=shared, dispatch_bytes=disp,
    )
