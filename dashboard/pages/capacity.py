"""Capacity sweep page, E2 (DASHBOARD_SPEC §4.4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Capacity sweep (E2)", "How does the Switch capacity factor trade quality against dropped tokens, speed and memory?", experiment="e2_capacity")
tpl = layout.current_template()
e2_runs = rio.e2_runs(sel.cfg_name, data.load_run) if sel.cfg_name else []
e2 = rio.summarize_e2(e2_runs)
if e2.empty:
    layout.empty_state("e2_capacity", sel.cfg_name)
    st.stop()

a, b = st.columns(2)
with a:
    layout.safe_chart("val loss vs capacity factor (C1)", lambda: figures.fig_cf_metric(e2, "final_val_loss", template=tpl),
                      experiment="e2_capacity", cfg_name=sel.cfg_name, key="c1")
with b:
    layout.safe_chart("drop fraction vs capacity factor (C2)", lambda: figures.fig_cf_drop(e2, template=tpl),
                      experiment="e2_capacity", cfg_name=sel.cfg_name, key="c2")
a, b = st.columns(2)
with a:
    layout.safe_chart("throughput vs capacity factor (C3)", lambda: figures.fig_cf_metric(e2, "median_tokens_per_sec", template=tpl),
                      experiment="e2_capacity", cfg_name=sel.cfg_name, key="c3")
    st.caption("Wall-clock training throughput; see the Systems page (E5) for controlled timing.")
with b:
    layout.safe_chart("peak memory vs capacity factor (C4)", lambda: figures.fig_cf_metric(e2, "peak_mem_mb", template=tpl),
                      experiment="e2_capacity", cfg_name=sel.cfg_name, key="c4")

with st.expander("Val loss curves per capacity factor"):
    layout.safe_chart("val loss curves (C5)", lambda: figures.fig_cf_curves(e2_runs, template=tpl),
                      experiment="e2_capacity", cfg_name=sel.cfg_name, key="c5")

st.subheader("Table")
ref = float(e2["ref_capacity_factor"].iloc[0])
t = pd.DataFrame({
    "capacity factor": e2["capacity_factor"].map(lambda c: f"{c:g}"),
    "capacity per expert C": e2["capacity_per_expert"].map(lambda c: "—" if pd.isna(c) else f"{int(c):,}"),
    "seeds": e2["n_seeds"],
    "val loss": e2["final_val_loss"].map(theme.fmt_loss),
    f"Δ vs cf={ref:g}": e2["delta_loss_vs_ref"].map(theme.fmt_delta),
    "val drop": e2["val_drop_fraction"].map(theme.fmt_pct),
    "tok/s": e2["median_tokens_per_sec"].map(theme.fmt_tps),
    "peak memory": e2["peak_mem_mb"].map(lambda m: theme.fmt_mib(m) if (m or 0) > 0 else "— (CPU)"),
})
st.dataframe(t, hide_index=True, width="stretch")
