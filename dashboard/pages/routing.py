"""Routing page (DASHBOARD_SPEC §4.3): expert load heatmap, final load, imbalance/entropy/drop over time."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Routing", "How evenly are tokens spread over experts, and how does that evolve?", experiment="e1_main")
runs = data.list_runs(sel.cfg_name, "e1_main")
if not runs:
    layout.empty_state("e1_main", sel.cfg_name)
    st.stop()

tpl = layout.current_template()
sel_runs = data.load_runs(list(sel.run_dirs))
moe_runs = [r for r in sel_runs if not r.routing.empty]
if not moe_runs:
    st.info("Routing logs exist only for MoE variants (switch, deepseek, gshard). Select an MoE run in the sidebar.")
    st.stop()

default_i = next((i for i, r in enumerate(moe_runs) if r.variant == "deepseek"), 0)
c1, c2 = st.columns(2)
run = c1.selectbox("run", moe_runs, index=default_i, format_func=lambda r: r.info.label)
layers = sorted(int(l) for l in run.routing["layer"].unique())
all_layers = sorted({int(l) for r in moe_runs for l in r.routing["layer"].unique()})
layer_choice = c2.selectbox("layer", ["mean over layers"] + all_layers)
layer_for_run = layer_choice
r1_layer = layer_choice if (layer_choice == "mean over layers" or layer_choice in layers) else "mean over layers"
if r1_layer != layer_choice:
    st.caption(f"Selected run has no layer {layer_choice}; showing mean over layers in the heatmap.")

st.subheader("Expert load over training")
zmode = st.radio("z", ["fraction", "ratio to uniform"], horizontal=True)
st.caption("Counts are accumulated over each logging window (`window_steps` steps), pre-drop demand.")
layout.safe_chart("expert load heatmap (R1)", lambda: figures.fig_expert_heatmap(
    run.routing, layer=r1_layer, z=zmode, run_name=layout.run_label(run.info), template=tpl),
    experiment="e1_main", cfg_name=sel.cfg_name, key="r1")

st.subheader("Final expert load per layer")
sort_load = st.toggle("sort by load", value=False)
layout.safe_chart("final expert load (R2)", lambda: figures.fig_final_load_hist(
    run.routing, variant=run.variant, sort=sort_load, run_name=layout.run_label(run.info), template=tpl),
    experiment="e1_main", cfg_name=sel.cfg_name, key="r2")

st.subheader("Load imbalance, entropy and drops over training (all selected MoE runs)")
_ws = sorted({int(w) for r in moe_runs if "window_steps" in r.routing for w in r.routing["window_steps"].dropna().unique()})
st.caption(f"Values are aggregated over each routing-log window (`window_steps` = {', '.join(map(str, _ws)) or '?'} steps); "
           "per-step imbalance can be higher than the windowed value (applies to the imbalance and drop-fraction charts).")
layout.safe_chart("load imbalance (R3)", lambda: figures.fig_routing_metric(
    moe_runs, metric="load_imbalance", layer=layer_choice, template=tpl), experiment="e1_main", cfg_name=sel.cfg_name, key="r3")
norm = st.toggle("normalize entropy by ln E", value=False)
layout.safe_chart("router entropy (R4)", lambda: figures.fig_routing_metric(
    moe_runs, metric="router_entropy", layer=layer_choice, normalize=norm, template=tpl), experiment="e1_main", cfg_name=sel.cfg_name, key="r4")
layout.safe_chart("drop fraction (R5)", lambda: figures.fig_routing_metric(
    moe_runs, metric="drop_fraction", layer=layer_choice, template=tpl), experiment="e1_main", cfg_name=sel.cfg_name, key="r5")

with st.expander("Load coefficient of variation"):
    layout.safe_chart("load CV (R6)", lambda: figures.fig_routing_metric(
        moe_runs, metric="load_cv", layer=layer_choice, template=tpl), experiment="e1_main", cfg_name=sel.cfg_name, key="r6")
with st.expander("Per-layer aux loss"):
    layout.safe_chart("per-layer aux loss (R7)", lambda: figures.fig_routing_metric(
        [run], metric="aux_loss", template=tpl), experiment="e1_main", cfg_name=sel.cfg_name, key="r7")
