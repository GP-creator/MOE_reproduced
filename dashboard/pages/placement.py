"""Placement page, E6 (DASHBOARD_SPEC §4.7): device load, straggler, all-to-all bytes, live step-time model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Expert placement (E6)", "If experts were spread across D devices, how much would load imbalance and link bandwidth slow a step?", experiment="e6_placement")
tpl = layout.current_template()
runs = data.list_runs(sel.cfg_name, "e6_placement")
if not runs:
    layout.empty_state("e6_placement", sel.cfg_name)
    st.stop()
chosen = rio.latest(runs)
pl = data.load_placement(chosen.run_dir)
params, summary = pl["params"], pl["summary"]
if not params or summary.empty:
    layout.empty_state("e6_placement", sel.cfg_name)
    st.stop()
_note = str(params.get("note", ""))
_note = " ".join(_note.replace("Simple analytical model, not a simulator", "").replace("simple analytical model, not a simulator", "").split()).lstrip(" .,;:")
st.info("Simple analytical model, not a simulator (bridge to WSC-LLM)." + (" " + _note if _note else ""))

# arrays through the architect's loader (lazy import; the page must survive its absence)
arrays: dict = {}
arrays_err = None
try:
    from scripts.placement_sim import load_placement_arrays
    if pl["arrays_path"] is not None:
        arrays = load_placement_arrays(Path(pl["arrays_path"]).parent)
except ImportError as exc:
    arrays_err = f"scripts/placement_sim.py is not importable ({exc})"
except Exception as exc:  # noqa: BLE001
    arrays_err = f"could not load placement_arrays.npz: {type(exc).__name__}: {exc}"
if arrays_err:
    st.warning(arrays_err + ". Charts that need per-step arrays (P1 per-step view, P4, P5) are unavailable.")

traces = params.get("traces", [])
if not traces:
    layout.empty_state("e6_placement", sel.cfg_name)
    st.stop()
tr_idx_default = next((i for i, t in enumerate(traces) if t.get("variant") == "deepseek"), 0)
c1, c2, c3 = st.columns(3)
ti = c1.selectbox("trace", list(range(len(traces))), index=tr_idx_default,
                  format_func=lambda i: f"{traces[i].get('variant')} · {traces[i].get('source_run_id')}")
trace = traces[ti]
D_values = list(params.get("D_values", sorted(summary["D"].unique())))
D = c2.radio("D", D_values, horizontal=True, index=len(D_values) - 1)
layer_opts = ["all"] + sorted({int(l) for l in summary["layer"].astype(str) if l != "all"})
layer = c3.selectbox("layer", layer_opts)
policies = list(params.get("policies", sorted(summary["policy"].unique())))
sum_layer = summary[summary["layer"].astype(str) == str(layer)]

# P1 device load
st.subheader("Load per device")
s_one = sum_layer[(sum_layer["D"] == D) & (sum_layer["trace_dir"] == trace["trace_dir"])]
loads = []
mean_over_steps = True
step = 0
Ls = None if layer == "all" else [int(layer)]
have_arr = all((trace["source_run_id"], int(D), p) in arrays for p in policies)
if have_arr:
    S = next(iter(arrays[(trace["source_run_id"], int(D), policies[0])].values())).shape[0]
    mean_over_steps = st.checkbox("mean over steps", value=True)
    if not mean_over_steps and S > 1:
        step = st.slider("step", 0, int(S) - 1, 0)
for pol in policies:
    if have_arr:
        tpd = arrays[(trace["source_run_id"], int(D), pol)]["tokens_per_device"]
        tpd = tpd[:, Ls, :] if Ls is not None else tpd
        v = tpd.sum(axis=1).mean(axis=0) if mean_over_steps else tpd.sum(axis=1)[step]
        if Ls is not None and mean_over_steps:
            v = tpd[:, 0, :].mean(axis=0)
        loads += [{"variant": trace["variant"], "policy": pol, "device": d, "tokens": float(x)} for d, x in enumerate(v)]
    else:
        row = s_one[s_one["policy"] == pol]
        if not row.empty:
            loads += [{"variant": trace["variant"], "policy": pol, "device": d, "tokens": float(x)}
                      for d, x in enumerate(row["mean_tokens_per_device"].iloc[0])]
layout.safe_chart("device load (P1)", lambda: figures.fig_device_load(pd.DataFrame(loads), template=tpl),
                  experiment="e6_placement", cfg_name=sel.cfg_name, key="p1")
if layer == "all" and have_arr:
    st.caption("Layer 'all' sums the assignments of every MoE layer per device.")

st.subheader("Straggler slowdown and all-to-all traffic")
a, b = st.columns(2)
with a:
    layout.safe_chart("straggler vs D (P2)", lambda: figures.fig_straggler_vs_D(sum_layer, template=tpl),
                      experiment="e6_placement", cfg_name=sel.cfg_name, key="p2")
with b:
    fwd_only = st.toggle("forward only", value=False)
    layout.safe_chart("all-to-all bytes (P3)", lambda: figures.fig_a2a_bytes(sum_layer, fwd_only=fwd_only, template=tpl),
                      experiment="e6_placement", cfg_name=sel.cfg_name, key="p3")

st.subheader("Step-time estimate (live)")
if not arrays:
    st.info(arrays_err or "placement_arrays.npz not found; run `make placement`.")
else:
    grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    dflt = float(params.get("link_gbps_default") or 100)
    if grid[0] <= dflt <= grid[-1] and dflt not in grid:   # keep the run's default exactly (RESULTS uses it)
        grid = sorted(grid + [int(dflt) if dflt.is_integer() else dflt])
    snap = min(grid, key=lambda g: abs(np.log(g) - np.log(dflt)))
    w1, w2, w3 = st.columns(3)
    bw = w1.select_slider("link bandwidth (GB/s per device)", options=grid, value=snap)
    tflops = w2.number_input("device TFLOP/s", min_value=0.001, value=float(params.get("device_tflops_default") or 1.0),
                             help=f"source: {params.get('device_tflops_source', 'unknown')}")
    incl_bwd = w3.checkbox("include backward", value=bool(params.get("include_backward_default", True)))
    st.caption(f"device TFLOP/s source: {params.get('device_tflops_source', 'unknown')}")
    bw_grid = np.logspace(np.log10(grid[0]), np.log10(grid[-1]), 50)
    try:
        curves = rio.placement_step_times(arrays, params, ti, trace["source_run_id"], int(D), policies, Ls, bw_grid,
                                          device_tflops=float(tflops), include_backward=incl_bwd)
        at_bw = rio.placement_step_times(arrays, params, ti, trace["source_run_id"], int(D), policies, Ls, [bw],
                                         device_tflops=float(tflops), include_backward=incl_bwd)
        at_bw = at_bw.assign(label=[f"D{D} {p}" for p in at_bw["policy"]])
    except rio.NoData as exc:
        st.info(str(exc))
        curves = at_bw = None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not compute the step-time model: {type(exc).__name__}: {exc}")
        curves = at_bw = None
    if curves is not None:
        layout.safe_chart("step time vs bandwidth (P4)", lambda: figures.fig_step_time_vs_bw(
            curves, marker_bw=float(bw), variant=trace["variant"], template=tpl), experiment="e6_placement", cfg_name=sel.cfg_name, key="p4")
        layout.safe_chart("step time breakdown (P5)", lambda: figures.fig_step_time_breakdown(at_bw, template=tpl),
                          experiment="e6_placement", cfg_name=sel.cfg_name, key="p5")

st.markdown(
    "**How this relates to WSC-LLM.** Expert parallelism trades three things: *compute* (a layer waits for its most "
    "loaded device, so routing imbalance turns directly into idle time), *communication* (every off-device token pays an "
    "all-to-all, limited by link bandwidth) and *memory* (each device holds only its share of the experts). The model "
    "above isolates the first two on real routing traces; it ignores overlap, topology and kernel efficiency.")
