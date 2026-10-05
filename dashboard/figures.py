"""Figure builders shared by the Streamlit pages and ``scripts/make_report.py`` (DASHBOARD_SPEC §6).

CONVENTIONS (every ``fig_*`` in this file follows them; the builder adds the page figures):

1. Signature ``fig_xxx(<plain data>, *, <options>, template: str = "moe_light") -> go.Figure``.
   Inputs are DataFrames / dicts / dataclasses from ``dashboard.results_io`` (e.g. ``RunData``)
   plus explicit option kwargs holding exactly what the page widgets produce. The static
   report calls the same builder with the default option values.
2. NO streamlit import here (also none in ``theme.py`` / ``results_io.py``); this file must
   import with streamlit uninstalled.
3. Empty input -> ``raise NoData("<what is missing>")``. Never return an empty figure. Pages
   wrap every call in ``layout.safe_chart`` which turns NoData into an empty state.
4. Colours / symbols / dashes only from ``theme`` (``theme.colour(v)``, ``theme.symbol(v)``,
   ``theme.line_marker``, ``DISPATCH_DASH``, ``POLICY_DASH``, ``SEQUENTIAL``...). No hex
   literals here except via theme. Legend names via ``theme.variant_label(variant, cfg)``.
5. Axes via ``theme.AXIS[...]``; every trace sets an explicit ``hovertemplate`` (run label,
   x with unit, y with unit, formats from ``theme.HOVER_FMT``), ending in ``<extra></extra>``.
6. No result numbers typed in (spec §7): reference lines (1/E, ln E, trained k, ...) are
   computed from the data / config passed in.
7. Finish with :func:`_finish` (applies the template, hovermode and optional title).

Builder TODO (names from spec §4; add below, do not rename): table_params, headline_numbers,
fig_aux_curves, fig_lr, fig_throughput_bars, fig_memory_bars, fig_throughput_over_time,
fig_expert_heatmap, fig_final_load_hist, fig_routing_metric, fig_cf_metric, fig_cf_drop,
fig_cf_curves, fig_disable_top, fig_shared_ablation, fig_k_sweep, fig_latency_vs_tokens,
fig_latency_vs_experts, fig_speedup_table, fig_breakdown, fig_achieved_tflops, fig_roofline,
fig_memory, fig_capacity_memory, fig_device_load, fig_straggler_vs_D, fig_a2a_bytes,
fig_step_time_vs_bw, fig_step_time_breakdown.
"""

from __future__ import annotations

import html
import math
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard import theme
from dashboard.results_io import NoData, RunData

# ----------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------


def _finish(fig: go.Figure, template: str, *, title: Optional[str] = None,
            hovermode: str = "x unified", height: Optional[int] = None) -> go.Figure:
    """Apply the house template and common layout (spec §3.2)."""
    fig.update_layout(template=template, hovermode=hovermode)
    if title:
        fig.update_layout(title=dict(text=title))
    if height:
        fig.update_layout(height=height)
    return fig


def ema(values: Sequence[float], weight: float) -> np.ndarray:
    """Exponential moving average with bias correction (TensorBoard-style smoothing).

    s_t = w * s_{t-1} + (1 - w) * x_t, divided by (1 - w^t) so early points are not pulled
    toward 0. ``weight = 0`` returns the raw values.
    """
    x = np.asarray(values, dtype=np.float64)
    if weight <= 0 or x.size == 0:
        return x
    out = np.empty_like(x)
    s = 0.0
    for t, v in enumerate(x, start=1):
        s = weight * s + (1.0 - weight) * v
        out[t - 1] = s / (1.0 - weight ** t)
    return out


def run_legend(run: RunData) -> str:
    """Per-run legend/hover name: config-built variant label + seed."""
    base = theme.variant_label(run.variant, run.config)
    return f"{base} · s{run.info.seed}" if run.info.seed is not None else base


def aggregate_runs(frames: Sequence[pd.DataFrame], x: str, y: str) -> pd.DataFrame:
    """Mean / min / max of ``y`` over runs at each ``x`` present in ALL runs (seed bands).

    Runs of one group log at the same steps, so an inner join on ``x`` is exact; points that
    only some seeds reached (e.g. a still-running seed) are dropped rather than biased.
    """
    cols = [f.set_index(x)[y].rename(i) for i, f in enumerate(frames) if not f.empty and y in f]
    if not cols:
        return pd.DataFrame(columns=[x, "mean", "min", "max", "n"])
    wide = pd.concat(cols, axis=1, join="inner").dropna()
    return pd.DataFrame({x: wide.index.values, "mean": wide.mean(axis=1).values,
                         "min": wide.min(axis=1).values, "max": wide.max(axis=1).values,
                         "n": wide.shape[1]})


# ----------------------------------------------------------------------------------------
# Example page builder: T1/T2 loss curves (spec §4.2). Pattern for all line charts.
# ----------------------------------------------------------------------------------------

_LOSS_SRC = {
    "val": ("eval", "val_loss", "val_loss"),     # (RunData attribute, column, AXIS key)
    "train": ("train", "ce_loss", "train_ce"),
}


def fig_loss_curves(runs: Sequence[RunData], *, kind: Literal["val", "train"] = "val",
                    x: Literal["tokens", "steps"] = "tokens", log_y: bool = False,
                    agg_seeds: bool = True, smoothing: float = 0.8,
                    template: str = "moe_light") -> go.Figure:
    """Val loss (T1) or train CE (T2) vs tokens/steps, one line per run or mean+band per group.

    * colour = variant (theme), marker symbol = variant, markers on ~10% of points;
    * ``agg_seeds`` and a group with ≥ 2 runs: mean line + min-max band, legend
      ``"<label> (mean of n seeds)"``; otherwise one line per run;
    * ``kind="train"``: EMA-smoothed line (``smoothing``) over a faint raw line (no hover).
    """
    attr, col, axis_key = _LOSS_SRC[kind]
    xcol = "tokens_seen" if x == "tokens" else "step"
    usable = [r for r in runs if not getattr(r, attr).empty and col in getattr(r, attr)]
    if not usable:
        raise NoData(f"no {'eval_log' if kind == 'val' else 'train_log'} records in the selected runs")

    fig = go.Figure()
    x_fmt = theme.HOVER_FMT["tokens"] if x == "tokens" else "d"
    x_name = "tokens" if x == "tokens" else "step"

    # Group by run.json "group" (spec §1.4: seed aggregation groups by group).
    groups: dict[str, list[RunData]] = {}
    for r in sorted(usable, key=lambda r: (theme.variant_sort_key(r.variant), r.info.run_id)):
        groups.setdefault(r.info.group or r.info.run_id, []).append(r)

    for gname, members in groups.items():
        v = members[0].variant
        c = theme.colour(v)
        if agg_seeds and len(members) >= 2:
            frames = [getattr(m, attr)[[xcol, col]].copy() for m in members]
            if kind == "train":
                for f in frames:
                    f[col] = ema(f[col].values, smoothing)
            agg = aggregate_runs(frames, xcol, col)
            if agg.empty:
                continue
            name = f"{theme.variant_label(v, members[0].config)} (mean of {int(agg['n'].iloc[0])} seeds)"
            # min-max band: upper edge, then lower edge filled to it
            fig.add_trace(go.Scatter(x=agg[xcol], y=agg["max"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip", legendgroup=gname))
            fig.add_trace(go.Scatter(x=agg[xcol], y=agg["min"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=theme.rgba(c, 0.18),
                                     showlegend=False, hoverinfo="skip", legendgroup=gname))
            fig.add_trace(go.Scatter(
                x=agg[xcol], y=agg["mean"], mode="lines+markers", name=name, legendgroup=gname,
                line=dict(color=c), marker=theme.line_marker(v, len(agg)),
                customdata=np.stack([agg["min"], agg["max"]], axis=-1),
                hovertemplate=(f"{html.escape(name)}<br>{x_name} %{{x:{x_fmt}}}<br>"
                               f"mean %{{y:.3f}} nats (min %{{customdata[0]:.3f}}, max %{{customdata[1]:.3f}})"
                               "<extra></extra>")))
            continue
        for m in members:
            df = getattr(m, attr)
            name = run_legend(m)
            if kind == "train":
                smooth = ema(df[col].values, smoothing)
                fig.add_trace(go.Scatter(x=df[xcol], y=df[col], mode="lines", line=dict(color=c, width=1),
                                         opacity=0.25, showlegend=False, hoverinfo="skip",
                                         legendgroup=m.info.run_dir))
                fig.add_trace(go.Scatter(
                    x=df[xcol], y=smooth, mode="lines+markers", name=name, legendgroup=m.info.run_dir,
                    line=dict(color=c), marker=theme.line_marker(v, len(df)),
                    customdata=df[col].values,
                    hovertemplate=(f"{html.escape(name)}<br>{x_name} %{{x:{x_fmt}}}<br>"
                                   "CE %{y:.3f} nats smoothed (raw %{customdata:.3f})<extra></extra>")))
            else:
                ppl = df["val_ppl"].values if "val_ppl" in df else np.exp(df[col].values)
                fig.add_trace(go.Scatter(
                    x=df[xcol], y=df[col], mode="lines+markers", name=name, legendgroup=m.info.run_dir,
                    line=dict(color=c), marker=theme.line_marker(v, len(df)), customdata=ppl,
                    hovertemplate=(f"{html.escape(name)}<br>{x_name} %{{x:{x_fmt}}}<br>"
                                   "val loss %{y:.3f} nats · ppl %{customdata:.2f}<extra></extra>")))

    if not fig.data:
        raise NoData("no overlapping steps to aggregate")
    fig.update_xaxes(**(theme.AXIS["tokens"] if x == "tokens" else theme.AXIS["step"]))
    fig.update_yaxes(**theme.AXIS[axis_key])
    if log_y:
        fig.update_yaxes(type="log", tickformat=None)
    return _finish(fig, template)


# ----------------------------------------------------------------------------------------
# How it works (spec §5.3). Input: dashboard.inference.SentenceTrace (plain numpy inside).
# ----------------------------------------------------------------------------------------


def _token_ticks(tokens: Sequence[str]) -> dict[str, Any]:
    # Numeric x positions with tick text, because repeated tokens would merge on a category axis.
    return dict(tickmode="array", tickvals=list(range(len(tokens))), ticktext=list(tokens),
                tickangle=-60, title="token position")


def _topk_hover(rec: dict, t: int) -> str:
    """'e12 (g=0.410, kept) · e3 (g=0.220, dropped)' for token t of one MoE layer."""
    parts = []
    for j in range(rec["topk_idx"].shape[1]):
        kept = bool(rec["kept"][t, j])
        parts.append(f"e{int(rec['topk_idx'][t, j])} (g={float(rec['gates'][t, j]):.3f}, "
                     f"{'kept' if kept else 'dropped'})")
    return " · ".join(parts)


def fig_token_layer_grid(trace: Any, *, mode: Literal["expert id", "gate value"] = "expert id",
                         template: str = "moe_light") -> go.Figure:
    """Token × layer grid (spec §5.3). x = token position, y = block (layer 0 on top).

    Cell text = top-1 expert id (``✕`` suffix if the top-1 assignment was dropped), "dense" for
    non-MoE blocks. Colour: expert id (identity aid, cyclic Phase samples) or the top-1 gate
    (sequential Blues, 0..1). Hover: token, position, layer, top-k list with gates and kept
    flags; DeepSeek adds ‖shared‖, ‖routed‖ and routed/(shared+routed).
    """
    tokens, layers = trace.tokens, trace.layers
    T, L = len(tokens), len(layers)
    if T == 0 or L == 0:
        raise NoData("no tokens to show")
    moe_recs = [r for r in layers if r.get("is_moe")]
    E = max((int(r["n_experts"]) for r in moe_recs), default=0)

    z = np.full((L, T), np.nan)
    dense_z = np.full((L, T), np.nan)
    text = np.empty((L, T), dtype=object)
    hover = np.empty((L, T), dtype=object)
    for li, rec in enumerate(layers):
        for t in range(T):
            head = f"token {html.escape(tokens[t])!s} (pos {t})<br>layer {li}"
            if not rec.get("is_moe"):
                dense_z[li, t] = 0.0
                text[li, t] = "dense"
                hover[li, t] = f"{head}<br>dense FFN (no routing)"
                continue
            e1 = int(rec["topk_idx"][t, 0])
            g1 = float(rec["gates"][t, 0])
            z[li, t] = e1 if mode == "expert id" else g1
            text[li, t] = f"{e1}" + ("" if bool(rec["kept"][t, 0]) else "✕")
            h = f"{head}<br>top-{rec['topk_idx'].shape[1]}: {_topk_hover(rec, t)}"
            if rec.get("shared_out_norm") is not None and rec.get("routed_out_norm") is not None:
                s, r = float(rec["shared_out_norm"][t]), float(rec["routed_out_norm"][t])
                frac = r / (s + r) if (s + r) > 0 else float("nan")
                h += f"<br>‖shared‖ {s:.3f} · ‖routed‖ {r:.3f} · routed/(shared+routed) {frac:.2f}"
            hover[li, t] = h

    fig = go.Figure()
    if not np.all(np.isnan(dense_z)):
        fig.add_trace(go.Heatmap(z=dense_z, x=list(range(T)), y=list(range(L)), text=text,
                                 texttemplate="%{text}", customdata=hover,
                                 colorscale=[[0, theme.DENSE_CELL], [1, theme.DENSE_CELL]],
                                 showscale=False, xgap=1, ygap=1,
                                 hovertemplate="%{customdata}<extra></extra>"))
    if moe_recs:
        if mode == "expert id":
            colour_kw = dict(colorscale=theme.expert_colourscale(E), zmin=-0.5, zmax=E - 0.5, showscale=False)
        else:
            colour_kw = dict(colorscale=theme.SEQUENTIAL, zmin=0.0, zmax=1.0,
                             colorbar=dict(title=dict(text="top-1 gate"), thickness=12))
        fig.add_trace(go.Heatmap(z=z, x=list(range(T)), y=list(range(L)), text=text,
                                 texttemplate="%{text}", customdata=hover, xgap=1, ygap=1,
                                 hovertemplate="%{customdata}<extra></extra>", **colour_kw))
    fig.update_xaxes(**_token_ticks(tokens), showgrid=False)
    fig.update_yaxes(title="layer (block)", autorange="reversed", tickmode="array",
                     tickvals=list(range(L)), showgrid=False)
    return _finish(fig, template, hovermode="closest", height=170 + 46 * L)


def fig_router_probs(rec: dict, t: int, *, template: str = "moe_light") -> go.Figure:
    """Step 2: all E router probabilities of token t (``extra["router_probs"]``), top-k highlighted."""
    probs = rec.get("router_probs")
    if probs is None:
        raise NoData("router_probs missing (MoEAux.extra R1)")
    p = np.asarray(probs[t], dtype=np.float64)
    chosen = set(int(e) for e in rec["topk_idx"][t])
    hi, lo = theme.neutral_pair(template)
    colours = [hi if e in chosen else lo for e in range(p.size)]
    fig = go.Figure(go.Bar(
        x=list(range(p.size)), y=p, marker_color=colours, marker_opacity=[1.0 if e in chosen else 0.45 for e in range(p.size)],
        hovertemplate="expert %{x}<br>p = %{y:.4f}<extra></extra>"))
    uniform = 1.0 / p.size   # reference computed from E, not typed
    fig.add_hline(y=uniform, line=dict(dash="dot", width=1, color=theme.ink(template, muted=True)),
                  annotation_text="uniform 1/E", annotation_position="top right",
                  annotation_font_color=theme.ink(template, muted=True))
    fig.update_xaxes(title="routed expert id", tickmode="linear" if p.size <= 32 else "auto", dtick=1 if p.size <= 16 else None)
    fig.update_yaxes(title="router probability", tickformat=".2f", rangemode="tozero")
    return _finish(fig, template, hovermode="closest", height=260)


def fig_top5(trace: Any, t: int, *, template: str = "moe_light") -> go.Figure:
    """Next-token top-5 after position t: horizontal bars, single ink colour, values as text."""
    if trace.top5_probs is None or len(trace.top5_probs) == 0:
        raise NoData("no predictions")
    probs = np.asarray(trace.top5_probs[t], dtype=np.float64)
    toks = [f"{s!s}" for s in trace.top5_tokens[t]]
    # y as rank labels so equal token strings never merge; token text shown on the axis
    ylab = [f"{i + 1}. {s}" for i, s in enumerate(toks)]
    fig = go.Figure(go.Bar(x=probs, y=ylab, orientation="h", marker_color=theme.ink(template),
                           text=[f"{p:.1%}" for p in probs], textposition="outside",
                           hovertemplate="%{y}<br>p = %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="probability of next token", tickformat=".0%",
                     range=[0, min(1.0, float(probs.max()) * 1.25 + 1e-6)])
    return _finish(fig, template, hovermode="closest", height=250)


def fig_shared_vs_routed(trace: Any, layer: int, *, template: str = "moe_light") -> go.Figure:
    """DeepSeek: per-token ‖Σ shared outputs‖ vs ‖Σ g·routed outputs‖ for one block (spec §5.3)."""
    rec = trace.layers[layer]
    s, r = rec.get("shared_out_norm"), rec.get("routed_out_norm")
    if s is None or r is None:
        raise NoData("Shared/routed norms need MoEAux.extra keys from the layer owner (DASHBOARD_SPEC §5.4)")
    x = list(range(len(trace.tokens)))
    c_shared, c_routed = theme.neutral_pair(template)
    fig = go.Figure([
        go.Bar(x=x, y=s, name="shared expert (always on, Eq. 9 first sum)", marker_color=c_shared,
               hovertemplate="pos %{x}<br>‖shared‖ %{y:.3f}<extra></extra>"),
        go.Bar(x=x, y=r, name="routed experts (gated, Eq. 9 second sum)", marker_color=c_routed,
               hovertemplate="pos %{x}<br>‖routed‖ %{y:.3f}<extra></extra>"),
    ])
    fig.update_layout(barmode="group")
    fig.update_xaxes(**_token_ticks(trace.tokens))
    fig.update_yaxes(title="L2 norm of contribution")
    return _finish(fig, template, hovermode="x unified", height=320)


# ========================================================================================
# Page builders (DASHBOARD_SPEC §4). Appended by the builder; everything above is the
# architect's shared layer.
# ========================================================================================

from plotly.subplots import make_subplots  # noqa: E402

from dashboard.results_io import MOE_VARIANTS, param_table_with_checks as table_params, headline_numbers  # noqa: E402,F401

_PASS_TITLE = {"fwd": "forward", "fwd_bwd": "forward + backward"}


def _by_group(runs: Sequence[RunData]) -> dict[str, list[RunData]]:
    groups: dict[str, list[RunData]] = {}
    for r in sorted(runs, key=lambda r: (theme.variant_sort_key(r.variant), r.info.run_id)):
        groups.setdefault(r.info.group or r.info.run_id, []).append(r)
    return groups


def _run_lines(runs: Sequence[RunData], attr: str, col: str, *, x: str, y_axis: dict, y_name: str,
               y_fmt: str, smoothing: float = 0.0, agg_seeds: bool = False, window: int = 0,
               template: str = "moe_light") -> go.Figure:
    """Generic per-run line chart over a log DataFrame column (aux, lr, rolling tok/s)."""
    xcol = "tokens_seen" if x == "tokens" else "step"
    x_fmt = theme.HOVER_FMT["tokens"] if x == "tokens" else "d"
    x_name = "tokens" if x == "tokens" else "step"
    fig = go.Figure()
    for gname, members in _by_group(runs).items():
        v = members[0].variant
        c = theme.colour(v)
        frames = []
        for m in members:
            df = getattr(m, attr)
            if df.empty or col not in df:
                continue
            f = df[[xcol, col]].copy()
            if window:
                f[col] = f[col].rolling(window, min_periods=1).median()
            elif smoothing > 0:
                f[col] = ema(f[col].values, smoothing)
            frames.append((m, f))
        if not frames:
            continue
        if agg_seeds and len(frames) >= 2:
            agg = aggregate_runs([f for _, f in frames], xcol, col)
            if agg.empty:
                continue
            name = f"{theme.variant_label(v, members[0].config)} (mean of {int(agg['n'].iloc[0])} seeds)"
            fig.add_trace(go.Scatter(x=agg[xcol], y=agg["max"], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip", legendgroup=gname))
            fig.add_trace(go.Scatter(x=agg[xcol], y=agg["min"], mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=theme.rgba(c, 0.18), showlegend=False,
                                     hoverinfo="skip", legendgroup=gname))
            fig.add_trace(go.Scatter(x=agg[xcol], y=agg["mean"], mode="lines+markers", name=name,
                                     legendgroup=gname, line=dict(color=c),
                                     marker=theme.line_marker(v, len(agg)),
                                     hovertemplate=f"{html.escape(name)}<br>{x_name} %{{x:{x_fmt}}}<br>"
                                                   f"{y_name} %{{y:{y_fmt}}}<extra></extra>"))
            continue
        for m, f in frames:
            name = run_legend(m)
            fig.add_trace(go.Scatter(x=f[xcol], y=f[col], mode="lines+markers", name=name,
                                     legendgroup=m.info.run_dir, line=dict(color=c),
                                     marker=theme.line_marker(v, len(f)),
                                     hovertemplate=f"{html.escape(name)}<br>{x_name} %{{x:{x_fmt}}}<br>"
                                                   f"{y_name} %{{y:{y_fmt}}}<extra></extra>"))
    if not fig.data:
        raise NoData(f"no {col} records in the selected runs")
    fig.update_xaxes(**(theme.AXIS["tokens"] if x == "tokens" else theme.AXIS["step"]))
    fig.update_yaxes(**y_axis)
    return _finish(fig, template)


def fig_aux_curves(runs: Sequence[RunData], *, x: str = "tokens", agg_seeds: bool = True,
                   smoothing: float = 0.0, template: str = "moe_light") -> go.Figure:
    """T3: train aux loss (Σ layers, already × α) vs tokens, MoE variants only."""
    moe_runs = [r for r in runs if r.variant in MOE_VARIANTS]
    if not moe_runs:
        raise NoData("aux loss is 0 for dense variants; select an MoE run")
    return _run_lines(moe_runs, "train", "aux_loss", x=x, y_axis=dict(title="aux loss (Σ layers, × α)", tickformat=".3g"),
                      y_name="aux", y_fmt=".4g", smoothing=smoothing, agg_seeds=agg_seeds, template=template)


def fig_lr(runs: Sequence[RunData], *, template: str = "moe_light") -> go.Figure:
    """T4: learning rate actually applied vs step."""
    return _run_lines(runs, "train", "lr", x="steps", y_axis=dict(title="learning rate", tickformat=".2e"),
                      y_name="lr", y_fmt=".3e", template=template)


def fig_throughput_over_time(runs: Sequence[RunData], *, window: int = 50, x: str = "steps",
                             template: str = "moe_light") -> go.Figure:
    """T7: rolling median (``window`` steps) of tokens/s, to spot throttling drift."""
    return _run_lines(runs, "train", "tokens_per_sec", x=x, y_axis=dict(**theme.AXIS["tps"]),
                      y_name="tok/s (rolling median)", y_fmt="~s", window=window, template=template)


def _final_bars(runs: Sequence[RunData], key: str, y_axis: dict, y_name: str, y_fmt: str,
                extra_key: Optional[str], template: str, env_vram: bool = False) -> go.Figure:
    by_v: dict[str, list[RunData]] = {}
    for r in runs:
        if r.final and r.final.get(key) is not None:
            by_v.setdefault(r.variant or "?", []).append(r)
    if not by_v:
        raise NoData("no complete runs selected")
    fig = go.Figure()
    for v in sorted(by_v, key=theme.variant_sort_key):
        rs = by_v[v]
        vals = [r.final[key] for r in rs]
        mean = float(np.mean(vals))
        name = theme.variant_label(v, rs[0].config)
        extra = float(np.mean([r.final[extra_key] for r in rs if r.final.get(extra_key) is not None])) \
            if extra_key and any(r.final.get(extra_key) is not None for r in rs) else float("nan")
        vram = (rs[0].env or {}).get("total_vram_gb") if env_vram else None
        fig.add_trace(go.Bar(
            x=[name], y=[mean], name=name, marker_color=theme.colour(v),
            error_y=dict(type="data", symmetric=False, array=[max(vals) - mean], arrayminus=[mean - min(vals)],
                         visible=len(vals) > 1),
            customdata=[[len(vals), extra, vram if vram is not None else float("nan")]],
            hovertemplate=(f"{html.escape(name)}<br>n seeds %{{customdata[0]}}<br>{y_name} %{{y:{y_fmt}}}"
                           + (f"<br>mean %{{customdata[1]:{y_fmt}}}" if extra_key else "")
                           + ("<br>VRAM total %{customdata[2]:.1f} GB" if vram is not None else "")
                           + "<extra></extra>")))
    fig.update_yaxes(**y_axis, rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _finish(fig, template, hovermode="closest")


def fig_throughput_bars(runs: Sequence[RunData], *, template: str = "moe_light") -> go.Figure:
    """T5: median training tokens/s per variant (error bar = min/max over seeds)."""
    return _final_bars(runs, "median_tokens_per_sec", dict(**theme.AXIS["tps"]), "median tok/s", "~s",
                       "mean_tokens_per_sec", template)


def fig_memory_bars(runs: Sequence[RunData], *, template: str = "moe_light") -> go.Figure:
    """T6: peak memory per variant (MiB)."""
    done = [r for r in runs if r.final and (r.final.get("peak_mem_mb") or 0) > 0]
    if not done:
        raise NoData("memory not measured on CPU (smoke), or no complete runs selected")
    return _final_bars(done, "peak_mem_mb", dict(**theme.AXIS["mib"]), "peak memory MiB", ",.0f", None,
                       template, env_vram=True)


# ---- Routing ------------------------------------------------------------------------------


def _layer_frame(routing: pd.DataFrame, layer: Any) -> pd.DataFrame:
    if routing is None or routing.empty:
        raise NoData("Routing logs exist only for MoE variants (switch, deepseek, gshard). Select an MoE run in the sidebar.")
    if layer in (None, "mean over layers"):
        return routing
    return routing[routing["layer"] == layer]


def fig_expert_heatmap(routing: pd.DataFrame, *, layer: Any = "mean over layers", z: str = "fraction",
                       run_name: str = "", template: str = "moe_light") -> go.Figure:
    """R1: expert × step load heatmap. ``z`` = "fraction" (Blues) or "ratio to uniform" (log2, diverging)."""
    df = _layer_frame(routing, layer)
    if df.empty:
        raise NoData("no routing records for that layer")
    steps = sorted(df["step"].unique())
    E = int(df["n_experts"].iloc[0])
    counts = np.zeros((E, len(steps)))
    for j, s in enumerate(steps):
        for cl in df[df["step"] == s]["expert_counts"]:
            counts[:, j] += np.asarray(cl, dtype=float)
    frac = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1)
    ratio = frac * E                     # count / mean(count)
    hover_cd = np.stack([counts, frac, ratio], axis=-1)
    ht = (f"{html.escape(run_name)}<br>step %{{x}}<br>expert %{{y}}<br>count %{{customdata[0]:,.0f}}<br>"
          "fraction %{customdata[1]:.2%}<br>ratio to uniform %{customdata[2]:.2f}<extra></extra>")
    if z == "ratio to uniform":
        with np.errstate(divide="ignore"):
            zz = np.log2(np.maximum(ratio, 1e-12))
        lim = max(float(np.nanmax(np.abs(zz[np.isfinite(zz)]))) if np.isfinite(zz).any() else 1.0, 1e-3)
        lim = min(lim, 4.0 if lim > 4.0 else lim)
        ticks = list(range(-int(math.ceil(lim)), int(math.ceil(lim)) + 1))
        fig = go.Figure(go.Heatmap(z=np.clip(zz, -lim, lim), x=steps, y=list(range(E)), customdata=hover_cd,
                                   colorscale=theme.DIVERGING, zmin=-lim, zmax=lim, zmid=0,
                                   colorbar=dict(title=dict(text="load ÷ uniform"), tickvals=ticks,
                                                 ticktext=[f"{2.0 ** t:g}×" for t in ticks]),
                                   hovertemplate=ht))
    else:
        fig = go.Figure(go.Heatmap(z=frac, x=steps, y=list(range(E)), customdata=hover_cd,
                                   colorscale=theme.SEQUENTIAL, zmin=0,
                                   colorbar=dict(title=dict(text="load fraction"), tickformat=".0%"),
                                   hovertemplate=ht))
    fig.update_xaxes(**theme.AXIS["step"])
    fig.update_yaxes(title="expert id", autorange="reversed", dtick=1 if E <= 16 else None)
    return _finish(fig, template, hovermode="closest", height=max(300, min(640, 14 * E + 120)))


def fig_final_load_hist(routing: pd.DataFrame, *, variant: Optional[str] = None, sort: bool = False,
                        run_name: str = "", template: str = "moe_light") -> go.Figure:
    """R2: final routing-window expert load per layer (small multiples, shared y)."""
    df = _layer_frame(routing, None)
    last = df.sort_values("step").groupby("layer").tail(1).sort_values("layer")
    L = len(last)
    cols = min(L, 4)
    rows = int(math.ceil(L / cols))
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=True, subplot_titles=[f"layer {int(l)}" for l in last["layer"]])
    c = theme.colour(variant)
    for i, (_, rec) in enumerate(last.iterrows()):
        r, k = divmod(i, cols)
        counts = np.asarray(rec["expert_counts"], dtype=float)
        kept = np.asarray(rec["kept_counts"], dtype=float)
        E = counts.size
        order = np.argsort(-counts, kind="stable") if sort else np.arange(E)
        frac = counts / max(counts.sum(), 1)
        fig.add_trace(go.Bar(x=list(range(E)), y=frac[order], marker_color=c, showlegend=False,
                             customdata=np.stack([order, counts[order], kept[order]], axis=-1),
                             hovertemplate=(f"{html.escape(run_name)}<br>layer {int(rec['layer'])}"
                                            "<br>expert %{customdata[0]}<br>load %{y:.2%}<br>count %{customdata[1]:,.0f}"
                                            "<br>kept %{customdata[2]:,.0f}<extra></extra>")), row=r + 1, col=k + 1)
        fig.add_trace(go.Scatter(x=[-0.5, E - 0.5], y=[1 / E, 1 / E], mode="lines", name="uniform (1/E)",
                                 line=dict(color=theme.ink(template, muted=True), dash="dot", width=1.5),
                                 showlegend=(i == 0), hovertemplate="uniform 1/E = %{y:.2%}<extra></extra>"),
                      row=r + 1, col=k + 1)
        fig.update_xaxes(title="expert id" if r == rows - 1 else None, row=r + 1, col=k + 1,
                         tickmode="array" if E <= 16 else "auto", tickvals=list(range(E)) if E <= 16 else None,
                         ticktext=[str(int(o)) for o in order] if E <= 16 else None)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_yaxes(title="load fraction", col=1)
    return _finish(fig, template, hovermode="closest", height=230 * rows + 90)


_ROUTING_METRICS = {
    "load_imbalance": ("max ÷ mean expert load", ".2f", "load imbalance"),
    "router_entropy": ("router entropy (nats)", ".3f", "entropy"),
    "drop_fraction": ("drop fraction (%)", ".1%", "drop"),
    "load_cv": ("load CV (std ÷ mean)", ".2f", "CV"),
    "aux_loss": ("aux loss per layer", ".3g", "aux"),
}


def fig_routing_metric(runs: Sequence[RunData], *, metric: str = "load_imbalance", layer: Any = "mean over layers",
                       normalize: bool = False, template: str = "moe_light") -> go.Figure:
    """R3-R7: a routing_log metric vs step. One line per MoE run (colour = variant).

    ``metric="aux_loss"`` instead draws one line per layer of the FIRST run (Blues by layer).
    ``drop_fraction`` keeps only runs that can drop (not DeepSeekMoE).
    """
    title, fmt, short = _ROUTING_METRICS[metric]
    moe = [r for r in runs if not r.routing.empty and metric in r.routing]
    if not moe:
        raise NoData("Routing logs exist only for MoE variants (switch, deepseek, gshard). Select an MoE run in the sidebar.")
    fig = go.Figure()
    if metric == "aux_loss":
        r = moe[0]
        layers = sorted(r.routing["layer"].unique())
        cols = pcolors_blues(len(layers))
        for l, c in zip(layers, cols):
            d = r.routing[r.routing["layer"] == l]
            fig.add_trace(go.Scatter(x=d["step"], y=d[metric], mode="lines+markers", name=f"layer {int(l)}",
                                     line=dict(color=c), marker=dict(symbol="circle", color=c, maxdisplayed=11),
                                     hovertemplate=f"{html.escape(run_legend(r))}<br>layer {int(l)}<br>step %{{x}}<br>aux %{{y:.4g}}<extra></extra>"))
        fig.update_xaxes(**theme.AXIS["step"])
        fig.update_yaxes(title=title, tickformat=".2g")
        return _finish(fig, template)
    if metric == "drop_fraction":
        moe = [r for r in moe if r.variant in ("switch", "gshard")]
        if not moe:
            raise NoData("No token dropping in this selection (DeepSeekMoE has no capacity limit)")
    ymax_ref = []
    for r in moe:
        d = _layer_frame(r.routing, layer)
        if d.empty:
            continue
        cols = [metric] + (["max_entropy"] if metric == "router_entropy" else [])
        g = d.groupby("step")[cols].mean().reset_index()
        yv = g[metric] / g["max_entropy"] if (metric == "router_entropy" and normalize) else g[metric]
        name = run_legend(r)
        cd = (g["max_entropy"] if metric == "router_entropy" else g[metric] * 0)
        fig.add_trace(go.Scatter(
            x=g["step"], y=yv, mode="lines+markers", name=name, legendgroup=r.info.run_dir,
            line=dict(color=theme.colour(r.variant)), marker=theme.line_marker(r.variant, len(g)), customdata=cd,
            hovertemplate=(f"{html.escape(name)}<br>step %{{x}}<br>{short} %{{y:{'.3f' if normalize else fmt}}}"
                           + ("<br>ln E %{customdata:.3f}" if metric == "router_entropy" else "") + "<extra></extra>")))
        if metric == "router_entropy" and not normalize:
            fig.add_trace(go.Scatter(x=g["step"], y=g["max_entropy"], mode="lines", legendgroup=r.info.run_dir,
                                     showlegend=False, line=dict(color=theme.ink(template, muted=True), dash="dash", width=1),
                                     hovertemplate=f"{html.escape(name)}: ln E (uniform) %{{y:.3f}}<extra></extra>"))
    if not fig.data:
        raise NoData("no routing records for that layer")
    if metric == "load_imbalance":
        fig.add_hline(y=1, line=dict(dash="dot", width=1, color=theme.ink(template, muted=True)),
                      annotation_text="perfect balance", annotation_position="bottom right")
    if metric == "router_entropy" and normalize:
        fig.add_hline(y=1, line=dict(dash="dot", width=1, color=theme.ink(template, muted=True)),
                      annotation_text="ln E (uniform)", annotation_position="bottom right")
    fig.update_xaxes(**theme.AXIS["step"])
    fig.update_yaxes(title=("router entropy ÷ ln E" if (metric == "router_entropy" and normalize) else title),
                     tickformat=(".2f" if normalize else fmt))
    return _finish(fig, template)


def pcolors_blues(n: int) -> list[str]:
    import plotly.colors as pc
    if n <= 1:
        return pc.sample_colorscale(theme.SEQUENTIAL, [0.7])
    return pc.sample_colorscale(theme.SEQUENTIAL, [0.3 + 0.7 * i / (n - 1) for i in range(n)])


# ---- Capacity sweep (E2) ------------------------------------------------------------------


def _cf_ticks(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.update_xaxes(title="capacity factor (training)", type="linear", tickmode="array",
                     tickvals=list(df["capacity_factor"]), ticktext=[f"{c:g}" for c in df["capacity_factor"]])


def _cf_trace(df: pd.DataFrame, col: str, name: str, *, dash: str = "solid", scale: float = 1.0,
              fmt: str = ".3f") -> go.Scatter:
    y = df[col].astype(float) * scale
    lo, hi = (df[col + "_min"].astype(float) * scale), (df[col + "_max"].astype(float) * scale)
    multi = bool((df["n_seeds"] > 1).any())
    return go.Scatter(
        x=df["capacity_factor"], y=y, mode="lines+markers", name=name,
        line=dict(color=theme.colour("switch"), dash=dash), marker=theme.line_marker("switch", len(df)),
        error_y=dict(type="data", symmetric=False, array=(hi - y).values, arrayminus=(y - lo).values, visible=multi),
        customdata=np.stack([df["n_seeds"], lo, hi], axis=-1),
        hovertemplate=f"{name}<br>cf %{{x:g}}<br>%{{y:{fmt}}} (seeds %{{customdata[0]}}, min %{{customdata[1]:{fmt}}}, max %{{customdata[2]:{fmt}}})<extra></extra>")


def fig_cf_metric(e2: pd.DataFrame, metric: str = "final_val_loss", *, template: str = "moe_light") -> go.Figure:
    """C1/C3/C4: one final.json metric vs capacity factor (``e2`` = ``summarize_e2``)."""
    spec = {"final_val_loss": (theme.AXIS["val_loss"], ".3f", "val loss (nats)"),
            "median_tokens_per_sec": (theme.AXIS["tps"], "~s", "median tok/s"),
            "peak_mem_mb": (theme.AXIS["mib"], ",.0f", "peak MiB"),
            "final_val_ppl": (theme.AXIS["ppl"], ".2f", "val ppl")}[metric]
    if e2 is None or e2.empty or e2[metric].isna().all():
        raise NoData("no E2 capacity-sweep runs" if (e2 is None or e2.empty) else f"{metric} not measured in these runs (CPU?)")
    if metric == "peak_mem_mb" and (e2[metric].fillna(0) <= 0).all():
        raise NoData("memory not measured on CPU (smoke)")
    fig = go.Figure(_cf_trace(e2, metric, spec[2], fmt=spec[1]))
    _cf_ticks(fig, e2)
    fig.update_yaxes(**spec[0])
    fig.update_layout(showlegend=False)
    return _finish(fig, template, hovermode="closest")


def fig_cf_drop(e2: pd.DataFrame, *, template: str = "moe_light") -> go.Figure:
    """C2: drop fraction vs capacity factor. Solid = val, dash = train (last 10%); dash means *split* here."""
    if e2 is None or e2.empty:
        raise NoData("no E2 capacity-sweep runs")
    fig = go.Figure()
    for col, name, dash in (("val_drop_fraction", "val (solid)", "solid"),
                            ("train_drop_fraction_last", "train, last 10% of steps (dash)", "dash")):
        if not e2[col].isna().all():
            fig.add_trace(_cf_trace(e2, col, name, dash=dash, fmt=".1%"))
    if not fig.data:
        raise NoData("no drop fractions recorded")
    _cf_ticks(fig, e2)
    fig.update_yaxes(**theme.AXIS["drop"], rangemode="tozero")
    return _finish(fig, template, hovermode="x unified")


def fig_cf_curves(runs: Sequence[RunData], *, x: str = "tokens", template: str = "moe_light") -> go.Figure:
    """C5: val loss vs tokens, one line per capacity factor (Blues by cf rank, light = low cf)."""
    usable = [r for r in runs if not r.eval.empty and r.info.capacity_factor is not None]
    if not usable:
        raise NoData("no E2 eval logs")
    usable.sort(key=lambda r: (r.info.capacity_factor, r.info.run_id))
    cfs = sorted({r.info.capacity_factor for r in usable})
    colour = dict(zip(cfs, pcolors_blues(len(cfs))))
    xcol = "tokens_seen" if x == "tokens" else "step"
    fig = go.Figure()
    for r in usable:
        cf = r.info.capacity_factor
        df = r.eval
        name = f"cf={cf:g}" + (f" · s{r.info.seed}" if sum(1 for q in usable if q.info.capacity_factor == cf) > 1 else "")
        fig.add_trace(go.Scatter(x=df[xcol], y=df["val_loss"], mode="lines+markers", name=name,
                                 line=dict(color=colour[cf]), marker=dict(symbol="circle", color=colour[cf], maxdisplayed=11),
                                 hovertemplate=f"{name}<br>{x} %{{x:~s}}<br>val loss %{{y:.3f}} nats<extra></extra>"))
    fig.update_xaxes(**(theme.AXIS["tokens"] if x == "tokens" else theme.AXIS["step"]))
    fig.update_yaxes(**theme.AXIS["val_loss"])
    return _finish(fig, template)


# ---- Ablations (E3/E4) --------------------------------------------------------------------


def _study(rows: pd.DataFrame, study: str, msg: str) -> pd.DataFrame:
    if rows is None or rows.empty or "study" not in rows:
        raise NoData("no ablation rows")
    d = rows[rows["study"] == study]
    if d.empty:
        raise NoData(msg)
    return d


def fig_disable_top(rows: pd.DataFrame, *, y: str = "delta", agg_seeds: bool = True,
                    cfgs: Optional[dict[str, dict]] = None, template: str = "moe_light") -> go.Figure:
    """A1: val loss (or Δ vs r=0 no-drop) vs the ACTUAL fraction of disabled routed experts.

    One line per source run, or mean + min-max band per variant over seeds. ``is_reference``
    rows (Switch/GShard r=0 at the training cf, with drops) are hollow markers at x=0.
    Δ is recomputed here against the same source run's r=0 non-reference point.
    """
    d = _study(rows, "e3_disable_top", "needs a trained checkpoint (E3)").copy()
    cfgs = cfgs or {}
    fig = go.Figure()
    base = {}
    for src, g in d[~d["is_reference"].astype(bool)].groupby("source_run_dir"):
        b = g[g["n_disable_top_routed"] == 0]
        if not b.empty:
            base[src] = float(b["val_loss"].iloc[0])
    d["yv"] = [r.val_loss - base.get(r.source_run_dir, np.nan) if y == "delta" else r.val_loss for r in d.itertuples()]
    ytitle = "Δ val loss vs r=0 (nats)" if y == "delta" else theme.AXIS["val_loss"]["title"]
    for variant in sorted(d["variant"].unique(), key=theme.variant_sort_key):
        dv = d[d["variant"] == variant]
        label = theme.variant_label(variant, cfgs.get(variant))
        main = dv[~dv["is_reference"].astype(bool)]
        srcs = sorted(main["source_run_dir"].unique())
        c = theme.colour(variant)
        if agg_seeds and len(srcs) >= 2:
            a = main.groupby("frac_disabled")["yv"].agg(["mean", "min", "max", "count"]).reset_index()
            fig.add_trace(go.Scatter(x=a["frac_disabled"], y=a["max"], mode="lines", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip", legendgroup=variant))
            fig.add_trace(go.Scatter(x=a["frac_disabled"], y=a["min"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=theme.rgba(c, 0.18), showlegend=False, hoverinfo="skip", legendgroup=variant))
            fig.add_trace(go.Scatter(x=a["frac_disabled"], y=a["mean"], mode="lines+markers", name=f"{label} (mean of {len(srcs)} seeds)",
                                     legendgroup=variant, line=dict(color=c), marker=theme.line_marker(variant, len(a)),
                                     hovertemplate=f"{html.escape(label)}<br>fraction disabled %{{x:.1%}}<br>%{{y:.3f}} nats<extra></extra>"))
        else:
            for src in srcs:
                m = main[main["source_run_dir"] == src].sort_values("frac_disabled")
                seed = m["seed"].iloc[0]
                name = f"{label} · s{seed}" if len(srcs) > 1 else label
                fig.add_trace(go.Scatter(
                    x=m["frac_disabled"], y=m["yv"], mode="lines+markers", name=name, legendgroup=variant,
                    line=dict(color=c), marker=theme.line_marker(variant, len(m)),
                    customdata=np.stack([m["n_disable_top_routed"], m["n_routed"], m["val_loss"], m["val_ppl"]], axis=-1),
                    hovertemplate=(f"{html.escape(name)}<br>disabled %{{customdata[0]}} / %{{customdata[1]}} "
                                   "(%{x:.1%})<br>" + ("Δ %{y:+.3f}" if y == "delta" else "val loss %{y:.3f}")
                                   + " nats<br>val loss %{customdata[2]:.3f} · ppl %{customdata[3]:.2f}<extra></extra>")))
        ref = dv[dv["is_reference"].astype(bool)]
        for r in ref.itertuples():
            fig.add_trace(go.Scatter(
                x=[0.0], y=[r.yv], mode="markers", name=f"{label}: training cf (with drops)", legendgroup=variant,
                marker=dict(symbol=theme.symbol(variant) + "-open", color=c, size=11, line=dict(width=2, color=c)),
                hovertemplate=(f"{html.escape(label)}: r=0 at training cf (with drops)<br>%{{y:.3f}} nats<extra></extra>")))
    fig.update_xaxes(title="fraction of routed experts disabled (n_disabled / n_routed)", tickformat=".0%")
    fig.update_yaxes(title=ytitle, tickformat=".2f")
    return _finish(fig, template)


_COND_ORDER = ("baseline", "no_shared_same_k", "no_shared_plus_one")
_COND_LABEL = {"baseline": "baseline", "no_shared_same_k": "no shared, same k (a)",
               "no_shared_plus_one": "no shared, k + 1 (b)"}


def fig_shared_ablation(rows: pd.DataFrame, *, cfg: Optional[dict] = None, template: str = "moe_light") -> go.Figure:
    """A2: shared-expert ablation bars (val loss; text = Δ vs baseline). Eval-time change, not retrained."""
    d = _study(rows, "e4_shared", "needs a DeepSeekMoE checkpoint")
    xs, ys, lo, hi, txt, cds = [], [], [], [], [], []
    base = d[d["condition"] == "baseline"]["val_loss"]
    b = float(base.mean()) if len(base) else float("nan")
    for cond in _COND_ORDER:
        g = d[d["condition"] == cond]
        if g.empty:
            continue
        m = float(g["val_loss"].mean())
        xs.append(_COND_LABEL[cond]); ys.append(m)
        lo.append(m - float(g["val_loss"].min())); hi.append(float(g["val_loss"].max()) - m)
        txt.append(f"{m - b:+.3f}" if cond != "baseline" else "baseline")
        cds.append([int(g["k_routed"].iloc[0]), int(g["active_ffn_width"].iloc[0]), m - b])
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=theme.colour("deepseek"), text=txt, textposition="outside",
        error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo, visible=bool((d.groupby("condition").size() > 1).any())),
        customdata=cds, hovertemplate=("%{x}<br>k_routed %{customdata[0]} · active FFN width %{customdata[1]}"
                                       "<br>val loss %{y:.3f} nats<br>Δ vs baseline %{customdata[2]:+.3f}<extra></extra>")))
    fig.update_yaxes(**theme.AXIS["val_loss"], rangemode="tozero")
    fig.update_xaxes(title="condition (eval-time change, not retrained)")
    return _finish(fig, template, hovermode="closest")


def fig_k_sweep(rows: pd.DataFrame, *, template: str = "moe_light") -> go.Figure:
    """A3: val loss vs k_routed at eval (vertical line at the trained k). Eval-time change, not retrained."""
    d = _study(rows, "e4_k_sweep", "needs a DeepSeekMoE checkpoint")
    a = d.groupby("k_routed").agg(val_loss=("val_loss", "mean"), lo=("val_loss", "min"), hi=("val_loss", "max"),
                                  width=("active_ffn_width", "first"), delta=("delta_val_loss", "mean")).reset_index()
    fig = go.Figure(go.Scatter(
        x=a["k_routed"], y=a["val_loss"], mode="lines+markers", name="DeepSeekMoE",
        line=dict(color=theme.colour("deepseek")), marker=theme.line_marker("deepseek", len(a)),
        error_y=dict(type="data", symmetric=False, array=(a["hi"] - a["val_loss"]).values,
                     arrayminus=(a["val_loss"] - a["lo"]).values, visible=bool((d.groupby("k_routed").size() > 1).any())),
        customdata=np.stack([a["width"], a["delta"]], axis=-1),
        hovertemplate="k_routed %{x}<br>active FFN width %{customdata[0]}<br>val loss %{y:.3f} nats<br>Δ vs trained k %{customdata[1]:+.3f}<extra></extra>"))
    trained = int(d["top_k_trained"].iloc[0])
    fig.add_vline(x=trained, line=dict(dash="dot", width=1, color=theme.ink(template, muted=True)),
                  annotation_text="trained k", annotation_position="top")
    fig.update_xaxes(title="k_routed at eval (eval-time change, not retrained)", tickmode="array", tickvals=list(a["k_routed"]))
    fig.update_yaxes(**theme.AXIS["val_loss"])
    fig.update_layout(showlegend=False)
    return _finish(fig, template, hovermode="closest")


# ---- Systems (E5) -------------------------------------------------------------------------


def _bench_filter(bench: pd.DataFrame, sweep: str, passes: Sequence[str], variants: Optional[Sequence[str]],
                  dispatches: Optional[Sequence[str]], n_experts: Optional[dict[str, int]] = None) -> pd.DataFrame:
    if bench is None or bench.empty:
        raise NoData("")
    d = bench[(bench["sweep"] == sweep) & (bench["pass"].isin(list(passes)))]
    if variants:
        d = d[d["variant"].isin(list(variants))]
    if dispatches:
        d = d[d["dispatch"].isin(list(dispatches) + ["n/a"])]
    if n_experts:
        keep = [(v == "dense") or (v not in n_experts) or (e == n_experts[v]) for v, e in zip(d["variant"], d["n_experts"])]
        d = d[keep]
    return d


def _gpu_clock(r: Any) -> str:
    b, a = getattr(r, "gpu_state_before", None), getattr(r, "gpu_state_after", None)
    if isinstance(b, dict) and isinstance(a, dict):
        return f"{b.get('sm_clock_mhz', '?')} → {a.get('sm_clock_mhz', '?')} MHz"
    return "n/a"


def _facet_fig(passes: Sequence[str]) -> go.Figure:
    return make_subplots(rows=1, cols=len(passes), shared_yaxes=True,
                         subplot_titles=[_PASS_TITLE.get(p, p) for p in passes], horizontal_spacing=0.06)


def _line_by_variant(fig: go.Figure, d: pd.DataFrame, passes: Sequence[str], xcol: str, ycol: str, *,
                     err: bool, ylabel: str, yfmt: str, xlabel: str, cfgs: Optional[dict] = None) -> None:
    seen: set[str] = set()
    for ci, p in enumerate(passes, start=1):
        dp = d[d["pass"] == p]
        for (variant, disp), g in dp.groupby(["variant", "dispatch"]):
            g = g.sort_values(xcol)
            name = theme.variant_label(variant, cfgs) + ("" if disp == "n/a" else f" · {disp}")
            cd = np.stack([g["n_tokens"], g["median_ms"], g["p10_ms"], g["p90_ms"], g["achieved_tflops"],
                           g["padding_fraction"].fillna(0.0), [_gpu_clock(r) for r in g.itertuples()], g["label"]], axis=-1)
            kw = dict(error_y=dict(type="data", symmetric=False, array=(g["p90_ms"] - g["median_ms"]).values,
                                   arrayminus=(g["median_ms"] - g["p10_ms"]).values, thickness=1)) if err else {}
            fig.add_trace(go.Scatter(
                x=g[xcol], y=g[ycol], mode="lines+markers", name=name, legendgroup=name, showlegend=name not in seen,
                line=dict(color=theme.colour(variant), dash=theme.DISPATCH_DASH.get(disp, "solid")),
                marker=theme.line_marker(variant, len(g)), customdata=cd, **kw,
                hovertemplate=("%{customdata[7]}<br>N %{customdata[0]:,} tokens<br>median %{customdata[1]:.3g} ms "
                               "(p10 %{customdata[2]:.3g}, p90 %{customdata[3]:.3g})<br>%{customdata[4]:.3g} TFLOP/s · "
                               "padding %{customdata[5]:.1%}<br>GPU SM clock %{customdata[6]}<extra></extra>")), row=1, col=ci)
            seen.add(name)


def fig_latency_vs_tokens(bench: pd.DataFrame, *, passes: Sequence[str] = ("fwd",), variants: Optional[Sequence[str]] = None,
                          dispatches: Optional[Sequence[str]] = None, n_experts: Optional[dict[str, int]] = None,
                          log_y: bool = True, template: str = "moe_light") -> go.Figure:
    """S1: median latency vs token count (p10-p90 error bars), colour = variant, dash = dispatch, facet = pass."""
    d = _bench_filter(bench, "tokens", passes, variants, dispatches, n_experts)
    if d.empty:
        raise NoData("no token-sweep rows for this selection")
    fig = _facet_fig(passes)
    _line_by_variant(fig, d, passes, "n_tokens", "median_ms", err=True, ylabel="latency (ms)", yfmt=".3g", xlabel="tokens N")
    ns = sorted(d["n_tokens"].unique())
    fig.update_xaxes(type="log", tickmode="array", tickvals=ns, ticktext=[f"{n:,}" for n in ns], title="tokens per layer call (N)")
    fig.update_yaxes(title="latency (ms)", col=1)
    if log_y:
        fig.update_yaxes(type="log")
    return _finish(fig, template, hovermode="closest")


def fig_latency_vs_experts(bench: pd.DataFrame, *, passes: Sequence[str] = ("fwd",), variants: Optional[Sequence[str]] = None,
                           dispatches: Optional[Sequence[str]] = None, n_tokens: Optional[int] = None,
                           log_y: bool = True, template: str = "moe_light") -> go.Figure:
    """S2: median latency vs number of experts (fixed active width), at one token count."""
    try:
        d = _bench_filter(bench, "experts", passes, variants, dispatches)
    except NoData:
        raise NoData("no expert-count sweep in this profile run")
    if n_tokens is not None:
        d = d[d["n_tokens"] == n_tokens]
    if d.empty:
        raise NoData("no expert-count sweep in this profile run")
    fig = _facet_fig(passes)
    _line_by_variant(fig, d, passes, "n_experts", "median_ms", err=True, ylabel="latency (ms)", yfmt=".3g", xlabel="experts")
    es = sorted(d["n_experts"].unique())
    fig.update_xaxes(type="log", tickmode="array", tickvals=es, ticktext=[str(e) for e in es], title="routed experts E")
    fig.update_yaxes(title="latency (ms)", col=1)
    if log_y:
        fig.update_yaxes(type="log")
    return _finish(fig, template, hovermode="closest")


def fig_breakdown(breakdown: pd.DataFrame, *, n_tokens: Optional[int] = None, n_experts: Optional[dict[str, int]] = None,
                  pass_: str = "fwd", percent: bool = False, method: Optional[str] = None,
                  template: str = "moe_light") -> go.Figure:
    """S4: stacked phase times per ``variant/dispatch`` (fixed phase order and colours).

    breakdown.jsonl holds one row set per timing ``method`` ("cuda_events" = PhaseTimer wall
    time per phase incl. launch gaps; "torch.profiler" = summed kernel device time). They
    measure different things, so exactly ONE method is plotted, never an average of both.
    Default: "cuda_events" if present (comparable to bench.jsonl medians), else the first.
    """
    if breakdown is None or breakdown.empty:
        raise NoData("no breakdown rows")
    d = breakdown[breakdown["pass"] == pass_]
    if "method" in d and d["method"].notna().any():
        methods = sorted(d["method"].dropna().unique())
        m = method if method in methods else ("cuda_events" if "cuda_events" in methods else methods[0])
        d = d[d["method"] == m]
    if n_tokens is not None:
        d = d[d["n_tokens"] == n_tokens]
    if n_experts:
        d = d[[(v == "dense") or (v not in n_experts) or (e == n_experts[v]) for v, e in zip(d["variant"], d["n_experts"])]]
    if d.empty:
        raise NoData("no breakdown rows for this selection")
    d = d.assign(cat=d["variant"] + "/" + d["dispatch"])
    cats = sorted(d["cat"].unique(), key=lambda c: (theme.variant_sort_key(c.split("/")[0]), c))
    surface = theme.SURFACE["dark" if template.endswith("dark") else "light"]
    fig = go.Figure()
    for ph in theme.PHASE_ORDER:
        g = d[d["phase"] == ph].groupby("cat")[["time_ms", "fraction"]].mean().reindex(cats).fillna(0.0)
        yv = g["fraction"] if percent else g["time_ms"]
        fig.add_trace(go.Bar(x=cats, y=yv, name=ph, marker=dict(color=theme.PHASE_COLOURS[ph], line=dict(width=2, color=surface)),
                             customdata=np.stack([g["time_ms"], g["fraction"]], axis=-1),
                             hovertemplate=f"{ph}<br>%{{x}}<br>%{{customdata[0]:.3g}} ms · %{{customdata[1]:.1%}}<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="share of layer time" if percent else "time (ms)", tickformat=".0%" if percent else None)
    fig.update_xaxes(title="variant / dispatch")
    return _finish(fig, template, hovermode="closest")


def _ceilings(roofline: Optional[dict]) -> dict[str, Optional[dict]]:
    out: dict[str, Optional[dict]] = {"spec": None, "measured": None}
    for c in (roofline or {}).get("ceilings", []):
        if c.get("kind") in out:
            out[c["kind"]] = c
    return out


def fig_achieved_tflops(bench: pd.DataFrame, roofline: Optional[dict], *, passes: Sequence[str] = ("fwd",),
                        variants: Optional[Sequence[str]] = None, dispatches: Optional[Sequence[str]] = None,
                        n_experts: Optional[dict[str, int]] = None, useful: bool = False,
                        template: str = "moe_light") -> go.Figure:
    """S5: achieved TFLOP/s vs N with measured (solid) and spec (dotted) ceilings."""
    d = _bench_filter(bench, "tokens", passes, variants, dispatches, n_experts)
    if d.empty:
        raise NoData("no token-sweep rows for this selection")
    ycol = "achieved_tflops_useful" if useful else "achieved_tflops"
    d = d.assign(achieved_tflops=d[ycol])
    fig = _facet_fig(passes)
    _line_by_variant(fig, d, passes, "n_tokens", ycol, err=False, ylabel="", yfmt=".3g", xlabel="")
    ceil = _ceilings(roofline)
    for kind, nm in (("measured", "measured peak"), ("spec", "spec-sheet peak")):
        c = ceil[kind]
        if c and c.get("peak_tflops"):
            st_ = theme.CEILING_STYLE[kind]
            src = c.get("source") or c.get("method") or ""
            fig.add_hline(y=c["peak_tflops"], row="all", col="all",
                          line=dict(color=theme.ink(template, muted=st_["ink"] == "muted"), dash=st_["dash"], width=st_["width"]),
                          annotation_text=f"{nm}: {c['peak_tflops']:.3g} TFLOP/s", annotation_position="top left",
                          annotation_font_size=11)
    ns = sorted(d["n_tokens"].unique())
    fig.update_xaxes(type="log", tickmode="array", tickvals=ns, ticktext=[f"{n:,}" for n in ns], title="tokens per layer call (N)")
    fig.update_yaxes(title="achieved TFLOP/s (useful FLOPs)" if useful else "achieved TFLOP/s", col=1)
    return _finish(fig, template, hovermode="closest")


def fig_roofline(bench: pd.DataFrame, roofline: Optional[dict], *, gemm: Optional[pd.DataFrame] = None,
                 points: str = "layer", pass_: str = "fwd", n_experts: Optional[Sequence[int]] = None,
                 template: str = "moe_light") -> go.Figure:
    """S6: log-log roofline. Filled markers = batched, open = loop; ceilings per CEILING_STYLE."""
    ceil = _ceilings(roofline)
    meas = ceil["measured"]
    if points == "gemm":
        if gemm is None or gemm.empty:
            raise NoData("no GEMM rows in this profile run")
        d = gemm.assign(dispatch="n/a")
        d["pass"] = pass_
    else:
        if bench is None or bench.empty:
            raise NoData("")
        d = bench[(bench["pass"] == pass_) & (bench["sweep"].isin(["tokens", "experts"]))]
    if n_experts:
        d = d[d["n_experts"].isin(list(n_experts)) | (d["variant"] == "dense")]
    d = d[d["intensity"].notna() & d["achieved_tflops"].notna()]
    if d.empty:
        raise NoData("no roofline points for this selection")
    fig = go.Figure()
    for (variant, disp), g in d.groupby(["variant", "dispatch"]):
        sym = theme.symbol(variant) + ("-open" if disp == "loop" else "")
        name = theme.variant_label(variant) + ("" if disp == "n/a" else f" · {disp}")
        pk = meas["peak_tflops"] if meas else None
        pg = meas["peak_gbps"] if meas else None
        pct = [100 * a / min(pk, x * pg / 1000) if (pk and pg) else float("nan") for a, x in zip(g["achieved_tflops"], g["intensity"])]
        fig.add_trace(go.Scatter(
            x=g["intensity"], y=g["achieved_tflops"], mode="markers", name=name,
            marker=dict(symbol=sym, color=theme.colour(variant), size=10, line=dict(width=1.5, color=theme.colour(variant))),
            customdata=np.stack([g["label"], pct, g["bytes"] if "bytes" in g else g["flops"] * 0, g["flops"]], axis=-1),
            hovertemplate=("%{customdata[0]}<br>intensity %{x:.3g} FLOP/B<br>%{y:.3g} TFLOP/s<br>"
                           "%{customdata[1]:.0f}% of measured roof at this intensity<br>bytes %{customdata[2]:.3g} · FLOPs %{customdata[3]:.3g}<extra></extra>")))
    xs = np.logspace(-1, 4, 200)
    for kind, nm in (("spec", "spec-sheet roof"), ("measured", "measured roof")):
        c = ceil[kind]
        if not c or not c.get("peak_tflops") or not c.get("peak_gbps"):
            continue
        P, B = c["peak_tflops"], c["peak_gbps"]
        st_ = theme.CEILING_STYLE[kind]
        col = theme.ink(template, muted=st_["ink"] == "muted")
        src = c.get("source") or c.get("method") or ""
        # Legend: the two numbers that define the roof (from roofline.json); full source in hover / S10.
        src_short = src if len(src) <= 90 else src[:90].rsplit(" ", 1)[0] + " …"
        fig.add_trace(go.Scatter(x=xs, y=np.minimum(P, xs * B / 1000), mode="lines",
                                 name=f"{nm} ({P:.3g} TFLOP/s, {B:.3g} GB/s)",
                                 line=dict(color=col, dash=st_["dash"], width=st_["width"]),
                                 hovertemplate=(f"{nm}<br>%{{x:.3g}} FLOP/B → %{{y:.3g}} TFLOP/s"
                                                + (f"<br>source: {src_short}" if src_short else "") + "<extra></extra>")))
        ridge = P * 1000 / B
        fig.add_trace(go.Scatter(x=[ridge], y=[P], mode="markers+text", showlegend=False, text=[f"ridge = {ridge:.3g} FLOP/B"],
                                 textposition="bottom right", textfont=dict(size=11, color=col),
                                 marker=dict(symbol="diamond-open", size=9, color=col),
                                 hovertemplate=f"{nm} ridge point<br>%{{x:.3g}} FLOP/B<extra></extra>"))
    fig.update_xaxes(**theme.AXIS["intensity"])
    fig.update_yaxes(title="achieved TFLOP/s", type="log")
    return _finish(fig, template, hovermode="closest", height=520)


def fig_memory(bench: pd.DataFrame, *, n_tokens: Optional[int] = None, template: str = "moe_light") -> go.Figure:
    """S7: peak memory per variant at one N (fwd_bwd preferred), loop = hatched."""
    if bench is None or bench.empty:
        raise NoData("")
    d = bench[bench["sweep"] == "tokens"]
    if d["peak_mem_mb"].isna().all():
        raise NoData("peak memory not measured in this profile run (CPU)")
    p = "fwd_bwd" if (d["pass"] == "fwd_bwd").any() and d[d["pass"] == "fwd_bwd"]["peak_mem_mb"].notna().any() else "fwd"
    d = d[(d["pass"] == p) & d["peak_mem_mb"].notna()]
    if n_tokens is None:
        n_tokens = int(d["n_tokens"].max())
    d = d[d["n_tokens"] == n_tokens]
    if d.empty:
        raise NoData("no memory rows at this N")
    fig = go.Figure()
    for disp in ("batched", "loop", "n/a"):
        g = d[d["dispatch"] == disp].sort_values("variant", key=lambda s: s.map(lambda v: theme.variant_sort_key(v)))
        if g.empty:
            continue
        fig.add_trace(go.Bar(x=list(g["variant"]), y=g["peak_mem_mb"], name=disp if disp != "n/a" else "dense (no dispatch)",
                             marker=dict(color=[theme.colour(v) for v in g["variant"]], pattern_shape="/" if disp == "loop" else ""),
                             customdata=np.stack([g["mem_baseline_mb"].fillna(0), (g["peak_mem_mb"] - g["mem_baseline_mb"].fillna(0))], axis=-1),
                             hovertemplate=f"%{{x}} · {disp} · {p}<br>peak %{{y:,.0f}} MiB<br>baseline %{{customdata[0]:,.0f}} MiB · delta %{{customdata[1]:,.0f}} MiB<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(**theme.AXIS["mib"])
    fig.update_xaxes(title=f"variant (N = {n_tokens:,} tokens, {_PASS_TITLE[p]})")
    return _finish(fig, template, hovermode="closest")


def fig_capacity_memory(bench: pd.DataFrame, *, passes: str = "fwd_bwd", template: str = "moe_light") -> go.Figure:
    """S8: peak memory and padding fraction vs capacity factor (two charts side by side, no dual axis)."""
    if bench is None or bench.empty or "sweep" not in bench:
        raise NoData("no capacity sweep rows")
    d = bench[bench["sweep"] == "capacity"]
    if d.empty:
        raise NoData("no capacity sweep rows")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["peak memory", "padding fraction (wasted expert rows)"], horizontal_spacing=0.12)
    mem_any = False
    pass_ = passes if (d["pass"] == passes).any() else d["pass"].iloc[0]
    for variant, g in d[d["pass"] == pass_].groupby("variant"):
        g = g.sort_values("capacity_factor")
        name = variant
        fig.add_trace(go.Scatter(x=g["capacity_factor"], y=g["padding_fraction"], mode="lines+markers", name=name, legendgroup=name,
                                 line=dict(color=theme.colour(variant)), marker=theme.line_marker(variant, len(g)),
                                 hovertemplate=f"{name}<br>cf %{{x:g}}<br>padding %{{y:.1%}}<extra></extra>"), row=1, col=2)
        if g["peak_mem_mb"].notna().any():
            mem_any = True
            fig.add_trace(go.Scatter(x=g["capacity_factor"], y=g["peak_mem_mb"], mode="lines+markers", name=name, legendgroup=name,
                                     showlegend=False, line=dict(color=theme.colour(variant)), marker=theme.line_marker(variant, len(g)),
                                     hovertemplate=f"{name}<br>cf %{{x:g}}<br>%{{y:,.0f}} MiB<extra></extra>"), row=1, col=1)
    if not mem_any:
        fig.add_annotation(text="memory not measured (CPU run)", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False, row=1, col=1)
    fig.update_xaxes(title="capacity factor")
    fig.update_yaxes(title="peak memory (MiB)", tickformat=",.0f", col=1)
    fig.update_yaxes(title="padding fraction", tickformat=".0%", col=2)
    return _finish(fig, template, hovermode="closest")


# ---- Placement (E6) -----------------------------------------------------------------------


def _need(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        raise NoData("")
    return df


def fig_device_load(loads: pd.DataFrame, *, template: str = "moe_light") -> go.Figure:
    """P1: assignments per device. ``loads`` columns: variant, policy, device, tokens. Greedy solid, round-robin hatched."""
    loads = _need(loads)
    fig = go.Figure()
    for (variant, pol), g in loads.groupby(["variant", "policy"]):
        mean = float(g["tokens"].mean())
        fig.add_trace(go.Bar(x=g["device"], y=g["tokens"], name=f"{variant} · {pol}",
                             marker=dict(color=theme.colour(variant), pattern_shape="/" if pol == "round_robin" else ""),
                             customdata=100 * g["tokens"] / max(mean, 1e-12),
                             hovertemplate=f"{variant} · {pol}<br>device %{{x}}<br>%{{y:,.0f}} assignments<br>%{{customdata:.0f}}% of mean<extra></extra>"))
    mean_all = float(loads["tokens"].mean())
    fig.add_hline(y=mean_all, line=dict(dash="dot", width=1.5, color=theme.ink(template, muted=True)),
                  annotation_text="perfect balance (mean)", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="device id", type="category")
    fig.update_yaxes(title="routed assignments per device", tickformat="~s", rangemode="tozero")
    return _finish(fig, template, hovermode="closest")


def fig_straggler_vs_D(summary: pd.DataFrame, *, template: str = "moe_light") -> go.Figure:
    """P2: straggler slowdown (max/mean device load) vs D with p50-p90 whiskers; colour = variant, dash = policy."""
    s = _need(summary)
    fig = go.Figure()
    for (variant, pol), g in s.groupby(["variant", "policy"]):
        g = g.sort_values("D")
        fig.add_trace(go.Scatter(
            x=g["D"].astype(str), y=g["straggler_mean"], mode="lines+markers", name=f"{variant} · {pol}",
            line=dict(color=theme.colour(variant), dash=theme.POLICY_DASH.get(pol, "solid")),
            marker=dict(symbol=theme.symbol(variant), color=theme.colour(variant)),
            error_y=dict(type="data", symmetric=False, array=(g["straggler_p90"] - g["straggler_mean"]).clip(lower=0).values,
                         arrayminus=(g["straggler_mean"] - g["straggler_p50"]).clip(lower=0).values, thickness=1),
            customdata=np.stack([g["straggler_p50"], g["straggler_p90"], g["straggler_max"]], axis=-1),
            hovertemplate=f"{variant} · {pol}<br>D %{{x}}<br>mean %{{y:.3f}}× (p50 %{{customdata[0]:.3f}}, p90 %{{customdata[1]:.3f}}, max %{{customdata[2]:.3f}})<extra></extra>"))
    fig.add_hline(y=1, line=dict(dash="dot", width=1, color=theme.ink(template, muted=True)), annotation_text="perfect balance")
    fig.update_xaxes(title="devices D", type="category")
    fig.update_yaxes(title="straggler slowdown (max ÷ mean load)", tickformat=".2f")
    return _finish(fig, template, hovermode="closest")


def fig_a2a_bytes(summary: pd.DataFrame, *, fwd_only: bool = False, template: str = "moe_light") -> go.Figure:
    """P3: all-to-all bytes per step vs D; colour = variant, hatched = round-robin."""
    s = _need(summary)
    col = "a2a_bytes_fwd_mean" if fwd_only else "a2a_bytes_fwd_bwd_mean"
    fig = go.Figure()
    for (variant, pol), g in s.groupby(["variant", "policy"]):
        g = g.sort_values("D")
        fig.add_trace(go.Bar(x=g["D"].astype(str), y=g[col], name=f"{variant} · {pol}",
                             marker=dict(color=theme.colour(variant), pattern_shape="/" if pol == "round_robin" else ""),
                             customdata=g["offdevice_tokens_mean"],
                             hovertemplate=f"{variant} · {pol}<br>D %{{x}}<br>%{{y:.3s}}B per step<br>off-device assignments %{{customdata:,.0f}}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="devices D", type="category")
    fig.update_yaxes(**theme.AXIS["bytes"], rangemode="tozero")
    return _finish(fig, template, hovermode="closest")


def fig_step_time_vs_bw(curves: pd.DataFrame, *, marker_bw: Optional[float] = None, variant: Optional[str] = None,
                        template: str = "moe_light") -> go.Figure:
    """P4: estimated step time vs link bandwidth (``placement_step_times`` output), vertical marker at the slider."""
    c = _need(curves)
    fig = go.Figure()
    for pol, g in c.groupby("policy"):
        g = g.sort_values("bw")
        fig.add_trace(go.Scatter(
            x=g["bw"], y=g["step_ms"], mode="lines+markers", name=pol,
            line=dict(color=theme.colour(variant), dash=theme.POLICY_DASH.get(pol, "solid")),
            marker=theme.line_marker(variant, len(g)),
            customdata=np.stack([g["compute_ms"], g["imbalance_ms"], g["comm_ms"]], axis=-1),
            hovertemplate=f"{pol}<br>link %{{x:.3g}} GB/s<br>step %{{y:.3g}} ms<br>compute %{{customdata[0]:.3g}} ms · imbalance %{{customdata[1]:.3g}} ms · comm %{{customdata[2]:.3g}} ms<extra></extra>"))
    if marker_bw is not None:
        fig.add_vline(x=marker_bw, line=dict(dash="dot", width=1.5, color=theme.ink(template, muted=True)),
                      annotation_text="selected bandwidth", annotation_position="top")
    fig.update_xaxes(title="link bandwidth per device (GB/s)", type="log")
    fig.update_yaxes(title="estimated step time (ms)", tickformat=".3g", rangemode="tozero")
    return _finish(fig, template, hovermode="x unified")


def fig_step_time_breakdown(parts: pd.DataFrame, *, template: str = "moe_light") -> go.Figure:
    """P5: stacked step-time components at the chosen bandwidth. ``parts`` columns: label, compute_balanced_ms, imbalance_ms, comm_ms."""
    p = _need(parts)
    import plotly.colors as pc
    cols = pc.sample_colorscale(theme.SEQUENTIAL, [0.35, 0.65, 0.95])
    surface = theme.SURFACE["dark" if template.endswith("dark") else "light"]
    fig = go.Figure()
    for key, nm, col in zip(("compute_balanced_ms", "imbalance_ms", "comm_ms"),
                            ("balanced compute", "imbalance penalty", "communication"), cols):
        total = p[["compute_balanced_ms", "imbalance_ms", "comm_ms"]].sum(axis=1)
        fig.add_trace(go.Bar(x=p["label"], y=p[key], name=nm, marker=dict(color=col, line=dict(width=2, color=surface)),
                             customdata=100 * p[key] / total.replace(0, np.nan),
                             hovertemplate=f"{nm}<br>%{{x}}<br>%{{y:.3g}} ms · %{{customdata:.1f}}% of step<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="estimated step time (ms)", tickformat=".3g")
    fig.update_xaxes(title="devices and placement policy")
    return _finish(fig, template, hovermode="closest")


def fig_speedup_vs_tokens(sp: pd.DataFrame, *, n_experts: Optional[dict[str, int]] = None, train_n: Optional[int] = None,
                          template: str = "moe_light") -> go.Figure:
    """Loop/batched speedup vs token count N per variant (solid = fwd, dashed = fwd_bwd); 1x = parity."""
    from dashboard.results_io import speedup_token_sweep
    t = speedup_token_sweep(sp, n_experts)
    if t.empty:
        raise NoData("no loop/batched pairs in the token sweep")
    fig = go.Figure()
    for (variant, ps), g in t.groupby(["variant", "pass"]):
        g = g.sort_values("n_tokens")
        fig.add_trace(go.Scatter(
            x=g["n_tokens"], y=g["speedup"], mode="lines+markers", name=f"{theme.variant_label(variant)} · {ps}",
            line=dict(color=theme.colour(variant), dash="dash" if ps == "fwd_bwd" else "solid"),
            marker=theme.line_marker(variant, len(g)),
            hovertemplate=f"{variant} {ps}<br>N %{{x:,}}<br>%{{y:.2f}}x loop/batched<extra></extra>"))
    ns = sorted(t["n_tokens"].unique())
    fig.add_hline(y=1.0, line=dict(color="#888", width=1, dash="dot"))
    if train_n is not None and train_n in ns:
        fig.add_vline(x=train_n, line=dict(color="#888", width=1, dash="dot"), annotation_text="training N", annotation_position="top")
    fig.update_xaxes(type="log", tickmode="array", tickvals=ns, ticktext=[f"{n:,}" for n in ns], title="tokens per layer call (N)")
    fig.update_yaxes(title="speedup (loop ms / batched ms)", ticksuffix="x")
    return _finish(fig, template, hovermode="closest")
