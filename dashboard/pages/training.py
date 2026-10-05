"""Training page (DASHBOARD_SPEC §4.2): loss curves, aux loss, throughput and memory, seed spread."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Training", "Do MoE variants reach lower loss than dense for the same tokens, and at what throughput and memory cost?", experiment="e1_main")
runs = data.list_runs(sel.cfg_name, "e1_main")
if not runs:
    layout.empty_state("e1_main", sel.cfg_name)
    st.stop()

sel_runs = data.load_runs(list(sel.run_dirs))
tpl = layout.current_template()
cfg_name = sel.cfg_name
if not sel_runs:
    st.info("No runs selected. Choose E1 runs in the sidebar.")
    st.stop()

c1, c2, c3 = st.columns(3)
log_y = c1.toggle("log y", value=False)
x_axis = c2.radio("x axis", ["tokens", "steps"], horizontal=True)
smoothing = c3.slider("smoothing (EMA weight)", 0.0, 0.99, 0.8, 0.01, help="Applies to train curves; raw is shown faint.")

layout.safe_chart("val loss (T1)", lambda: figures.fig_loss_curves(
    sel_runs, kind="val", x=x_axis, log_y=log_y, agg_seeds=sel.agg_seeds, template=tpl),
    experiment="e1_main", cfg_name=cfg_name, key="t1")
spread = rio.seed_spread_table(sel_runs)
if sel.agg_seeds and not spread.empty and (spread["n_seeds"] > 1).any():
    st.caption("Seed spread of final val loss (nats)")
    _num = [c for c in spread.columns if c not in ("group", "n_seeds")]
    st.dataframe(spread.assign(**{c: spread[c].map(theme.fmt_loss) for c in _num}), hide_index=True, width="stretch")

layout.safe_chart("train CE (T2)", lambda: figures.fig_loss_curves(
    sel_runs, kind="train", x=x_axis, log_y=log_y, agg_seeds=sel.agg_seeds, smoothing=smoothing, template=tpl),
    experiment="e1_main", cfg_name=cfg_name, key="t2")
st.caption("Smoothed (EMA); raw shown faint.")

st.subheader("Auxiliary (load-balancing) loss")
layout.safe_chart("aux loss (T3)", lambda: figures.fig_aux_curves(
    sel_runs, x=x_axis, agg_seeds=sel.agg_seeds, smoothing=smoothing, template=tpl),
    experiment="e1_main", cfg_name=cfg_name, key="t3")

st.subheader("Throughput and memory")
done = [r for r in sel_runs if r.final]
if len(done) < len(sel_runs):
    st.caption(f"{len(sel_runs) - len(done)} incomplete run(s) excluded from the bars.")
a, b = st.columns(2)
with a:
    layout.safe_chart("throughput (T5)", lambda: figures.fig_throughput_bars(done, template=tpl),
                      experiment="e1_main", cfg_name=cfg_name, key="t5")
with b:
    layout.safe_chart("peak memory (T6)", lambda: figures.fig_memory_bars(done, template=tpl),
                      experiment="e1_main", cfg_name=cfg_name, key="t6")
st.caption("E1 wall-clock throughput on a laptop GPU; see the Systems page for controlled microbenchmarks.")

with st.expander("LR schedule"):
    layout.safe_chart("LR (T4)", lambda: figures.fig_lr(sel_runs, template=tpl),
                      experiment="e1_main", cfg_name=cfg_name, key="t4")
with st.expander("Throttling check"):
    layout.safe_chart("throughput over time (T7)", lambda: figures.fig_throughput_over_time(sel_runs, template=tpl),
                      experiment="e1_main", cfg_name=cfg_name, key="t7")
