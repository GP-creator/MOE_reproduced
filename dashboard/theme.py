"""Colours, markers, dash maps, Plotly templates and number formatters (DASHBOARD_SPEC §3).

NO streamlit import here: ``scripts/make_report.py`` uses this module too. The Streamlit-only
theme switch (light/dark) lives in ``dashboard/layout.py::current_template``.

Rules (spec §3.1):
  * Colour follows the entity (variant), never its rank, and is the same in light and dark.
  * Colour is never the only encoding: every variant also has a fixed marker symbol, and line
    charts use ``mode="lines+markers"`` with ``marker.maxdisplayed`` so markers appear on
    about 10% of the points (see :func:`line_marker`).
  * Text never uses a series colour; labels use the template ink.
  * Legend labels that contain numbers (expert counts, top-k, width multiplier) are built from
    the run's config (:func:`variant_label`), never typed in (spec §7).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import plotly.colors as pcolors
import plotly.graph_objects as go
import plotly.io as pio

# ----------------------------------------------------------------------------------------
# Variant identity (spec §3.1 table). Okabe-Ito palette, fixed order.
# ----------------------------------------------------------------------------------------

VARIANT_ORDER: tuple[str, ...] = ("dense", "switch", "deepseek", "gshard", "dense_x8")

VARIANT_COLOUR: dict[str, str] = {
    "dense": "#0072B2",     # blue
    "switch": "#D55E00",    # vermillion
    "deepseek": "#009E73",  # bluish green
    "gshard": "#CC79A7",    # reddish purple
    "dense_x8": "#E69F00",  # orange
}
VARIANT_SYMBOL: dict[str, str] = {
    "dense": "circle",
    "switch": "square",
    "deepseek": "diamond",
    "gshard": "triangle-up",
    "dense_x8": "x",
}
UNKNOWN_COLOUR = "#777777"
UNKNOWN_SYMBOL = "circle"

#: Variants whose light-mode contrast is < 3:1 (spec §3.1); line charts give them end labels.
LOW_CONTRAST_LIGHT: frozenset[str] = frozenset({"gshard", "dense_x8"})


def colour(variant: Optional[str]) -> str:
    """Fixed colour of a variant; grey for unknown names (defensive, never raises)."""
    return VARIANT_COLOUR.get(variant or "", UNKNOWN_COLOUR)


def symbol(variant: Optional[str]) -> str:
    """Fixed marker symbol of a variant (the mandatory secondary encoding)."""
    return VARIANT_SYMBOL.get(variant or "", UNKNOWN_SYMBOL)


def variant_label(variant: Optional[str], cfg: Optional[Mapping[str, Any]] = None) -> str:
    """Legend label. Numbers come from ``cfg`` (a run's config.json), never literals (spec §7).

    Without a config the label omits the numbers instead of guessing them.
    """
    v = variant or "?"
    moe = (cfg or {}).get("moe") or {}
    try:
        if v == "dense":
            return "dense"
        if v == "switch":
            return "Switch (top-1)"          # top-1 is the definition of Switch, not a result
        if v == "deepseek":
            s = moe["deepseek"]
            return f"DeepSeekMoE ({s['n_shared']}+{s['n_routed']}, top-{s['k_routed']})"
        if v == "gshard":
            g = moe["gshard"]
            return f"GShard ({g['n_experts']} experts, top-{g['top_k']})"
        if v == "dense_x8":
            return f"dense ×{moe['dense_x8']['width_mult']}"
    except (KeyError, TypeError):
        pass
    return {"deepseek": "DeepSeekMoE", "gshard": "GShard", "dense_x8": "dense (wide)"}.get(v, v)


def variant_sort_key(variant: Optional[str]) -> tuple[int, str]:
    """Sort variants in the fixed spec order, unknown ones last (alphabetically)."""
    v = variant or ""
    return (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER), v)


# ----------------------------------------------------------------------------------------
# Other fixed maps (spec §3.1)
# ----------------------------------------------------------------------------------------

DISPATCH_DASH: dict[str, str] = {"batched": "solid", "loop": "dash", "n/a": "solid"}
POLICY_DASH: dict[str, str] = {"greedy": "solid", "round_robin": "dash"}
PASS_FACETS: tuple[str, ...] = ("fwd", "fwd_bwd")   # facet columns, never colour

PHASE_ORDER: tuple[str, ...] = ("router", "dispatch", "expert_gemm", "shared_expert", "combine", "other")
#: Single-hue sequential ramp (Blues), light -> dark in PHASE_ORDER.
PHASE_COLOURS: dict[str, str] = dict(zip(
    PHASE_ORDER, pcolors.sample_colorscale("Blues", [0.25 + 0.75 * i / (len(PHASE_ORDER) - 1)
                                                     for i in range(len(PHASE_ORDER))])))

SEQUENTIAL = "Blues"
#: Load ratio to uniform, used on a log2 axis centred at ratio 1 (spec §3.1).
DIVERGING: list[list[Any]] = [[0.0, "#0072B2"], [0.5, "#BBBBBB"], [1.0, "#D55E00"]]

#: Grey for non-MoE blocks in the How-it-works grid.
DENSE_CELL = "#BDBDBD"


def expert_colour(e: int, n_experts: int) -> str:
    """Colour of routed expert ``e`` out of ``n_experts`` (How-it-works only).

    E evenly spaced samples of Plotly's cyclic "Phase" scale. Identity aid only: the expert id
    is always printed in the cell too, because adjacent ids are hard to tell apart for E=31.
    """
    if n_experts <= 0:
        return UNKNOWN_COLOUR
    return pcolors.sample_colorscale("Phase", [e / n_experts])[0]


def expert_colourscale(n_experts: int) -> list[list[Any]]:
    """Discrete colourscale for a heatmap whose z is the expert id with zmin=-0.5, zmax=E-0.5."""
    scale: list[list[Any]] = []
    for e in range(n_experts):
        c = expert_colour(e, n_experts)
        scale += [[e / n_experts, c], [(e + 1) / n_experts, c]]
    return scale


# ----------------------------------------------------------------------------------------
# Ink and Plotly templates (spec §3.2)
# ----------------------------------------------------------------------------------------

FONT_FAMILY = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
INK = {"light": "#1f1f1f", "dark": "#e6e6e6"}
INK_MUTED = {"light": "#6b6b6b", "dark": "#a0a0a0"}
GRID = {"light": "#e8e8e8", "dark": "#333333"}
SURFACE = {"light": "#ffffff", "dark": "#1a1a19"}   # for marker outlines only (paper is transparent)
TRANSPARENT = "rgba(0,0,0,0)"

#: Spec §3.1 CEILING_STYLE for roofline ceilings; "colour" is resolved per template by ink().
CEILING_STYLE: dict[str, dict[str, Any]] = {
    "spec": {"ink": "muted", "dash": "dot", "width": 1.5},
    "measured": {"ink": "ink", "dash": "solid", "width": 2},
}


def _mode(template: str) -> str:
    return "dark" if template.endswith("dark") else "light"


def ink(template: str = "moe_light", muted: bool = False) -> str:
    """Text/annotation ink for a template (labels never use series colours)."""
    return (INK_MUTED if muted else INK)[_mode(template)]


def neutral_pair(template: str = "moe_light") -> tuple[str, str]:
    """Two neutral ink tones (How-it-works shared vs routed bars; not variant colours)."""
    return (ink(template), "#8c8c8c")


def _make_template(mode: str) -> go.layout.Template:
    t = go.layout.Template(pio.templates["plotly_white"])
    ink_, grid, surface = INK[mode], GRID[mode], SURFACE[mode]
    axis = dict(gridcolor=grid, gridwidth=1, zeroline=False, linecolor=grid,
                tickfont=dict(size=12, color=ink_), title=dict(font=dict(color=ink_)),
                tickcolor=grid)
    t.layout.update(
        font=dict(family=FONT_FAMILY, size=13, color=ink_),
        title=dict(font=dict(size=15, color=ink_), x=0, xanchor="left"),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        colorway=[VARIANT_COLOUR[v] for v in VARIANT_ORDER],
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left", yanchor="bottom",
                    font=dict(color=ink_), bgcolor=TRANSPARENT),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT_FAMILY)),
        margin=dict(l=60, r=20, t=50, b=50),
        bargap=0.25,
        coloraxis=dict(colorbar=dict(tickfont=dict(color=ink_), outlinewidth=0)),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2),
                                 marker=dict(size=8, line=dict(width=1, color=surface)))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    t.data.heatmap = [go.Heatmap(colorbar=dict(outlinewidth=0, tickfont=dict(color=ink_)))]
    return t


TEMPLATE_LIGHT = _make_template("light")
TEMPLATE_DARK = _make_template("dark")
pio.templates["moe_light"] = TEMPLATE_LIGHT
pio.templates["moe_dark"] = TEMPLATE_DARK


def line_marker(variant: Optional[str], n_points: int, **extra: Any) -> dict[str, Any]:
    """Marker dict for a line+markers trace: variant symbol, markers on ~10% of points."""
    return dict(symbol=symbol(variant), color=colour(variant),
                maxdisplayed=max(2, min(n_points, 11)), **extra)


def rgba(hex_colour: str, alpha: float) -> str:
    """'#RRGGBB' -> 'rgba(r,g,b,a)' (for seed min-max bands)."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# ----------------------------------------------------------------------------------------
# Axis titles and d3 formats (spec §3.3); use these instead of typing strings in builders
# ----------------------------------------------------------------------------------------

AXIS = {
    "tokens": dict(title="tokens seen", tickformat="~s"),
    "step": dict(title="step", tickformat="~s"),
    "val_loss": dict(title="val loss (nats)", tickformat=".2f"),
    "train_ce": dict(title="train CE (nats)", tickformat=".2f"),
    "ppl": dict(title="val perplexity", tickformat=".1f"),
    "tps": dict(title="tokens/s", tickformat="~s"),
    "ms": dict(title="latency (ms)"),
    "mib": dict(title="peak memory (MiB)", tickformat=",.0f"),
    "tflops": dict(title="achieved TFLOP/s", tickformat=".3~g"),
    "gbps": dict(title="GB/s", tickformat="~s"),
    "bytes": dict(title="bytes / step", tickformat="~s", ticksuffix="B"),
    "intensity": dict(title="arithmetic intensity (FLOP/byte)", type="log"),
    "drop": dict(title="drop fraction (%)", tickformat=".1%"),
}

#: d3 formats for hovertemplates, matching the table formats below.
HOVER_FMT = {"tokens": "~s", "loss": ".3f", "ppl": ".2f", "tps": "~s", "ms": ".3~g",
             "mib": ",.0f", "tflops": ".3~g", "pct": ".1%", "gate": ".3f"}


# ----------------------------------------------------------------------------------------
# Formatters for tables / tiles (spec §3.3). All return "—" for None / NaN.
# ----------------------------------------------------------------------------------------

DASH = "—"


def _bad(x: Any) -> bool:
    """None, NaN or inf (also numpy scalars such as float32 NaN and pandas NA)."""
    if x is None:
        return True
    try:
        return not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def _si(x: float, digits: int = 3) -> str:
    """SI suffix, about ``digits`` significant figures, always fixed-point (never 'e+02'):
    12345678 -> '12.3M', 910000 (digits=2) -> '910k', 999999 -> '1M', 0.5 -> '0.5'.
    The value is rounded to ``digits`` significant figures BEFORE the suffix is chosen, so
    rounding up to 1000 moves to the next suffix. A mantissa with more integer digits than
    ``digits`` is shown as an integer (no false precision, no exponent). Trailing zeros after
    the decimal point are dropped (like '%g')."""
    if x == 0 or not math.isfinite(x):
        return f"{x:g}"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    ax = round(ax, digits - 1 - math.floor(math.log10(ax)))          # sig-fig rounding
    div, suf = 1.0, ""
    for d_, s_ in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if ax >= d_:
            div, suf = d_, s_
            break
    m = ax / div
    decimals = max(0, digits - 1 - math.floor(math.log10(m))) if m > 0 else 0
    txt = f"{m:.{decimals}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return f"{sign}{txt}{suf}"


def fmt_tokens(x: Any) -> str:
    return DASH if _bad(x) else _si(float(x))


def fmt_loss(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.3f}"


def fmt_delta(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):+.3f}"


def fmt_ppl(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.2f}"


def fmt_tps(x: Any) -> str:
    return DASH if _bad(x) else f"{_si(float(x))} tok/s"


def fmt_ms(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.3g} ms"


def fmt_mib(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):,.0f} MiB"


def fmt_tflops(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.3g} TFLOP/s"


def fmt_gbps(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.3g} GB/s"


def fmt_bytes(x: Any, digits: int = 2) -> str:
    """SI bytes (1e9 = GB, spec §3.3). ``digits`` significant figures: 2 for compact tiles and
    axes (912000 -> '910kB'), 3 for report tables (912000 -> '912kB')."""
    return DASH if _bad(x) else f"{_si(float(x), digits)}B"


def fmt_params(x: Any) -> str:
    """Exact integer with thousands separators (tables)."""
    return DASH if _bad(x) else f"{int(x):,}"


def fmt_params_short(x: Any) -> str:
    """Compact form for tiles: 63331200 -> '63.3M'."""
    return DASH if _bad(x) else _si(float(x))


def fmt_pct(x: Any) -> str:
    """Fraction 0..1 -> '4.2%'."""
    return DASH if _bad(x) else f"{100 * float(x):.1f}%"


def fmt_intensity(x: Any) -> str:
    return DASH if _bad(x) else f"{float(x):.3g}"


# ----------------------------------------------------------------------------------------
# Step-through diagram (How it works, Graphviz DOT). Neutral tones; the current step is the
# only filled node, so the highlight does not rely on hue alone (also bold + thicker border).
# ----------------------------------------------------------------------------------------

DIAGRAM = {
    "node_fill": "#ffffff", "node_border": "#8c8c8c", "node_font": "#1f1f1f",
    "active_fill": "#0072B2", "active_border": "#00456b", "active_font": "#ffffff",
    "edge": "#8c8c8c", "edge_skip": "#bdbdbd",
}
