"""Ablations page, E3/E4 (DASHBOARD_SPEC §4.5). All ablations are eval-time changes on trained checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Ablations (E3/E4)", "How redundant are the experts, and how much does the shared expert matter?", experiment="e3e4_ablation")
st.info("Eval-time ablations on trained checkpoints; not the same as training from scratch (MASTER §7 E4).")
tpl = layout.current_template()
abl_runs = data.list_runs(sel.cfg_name, "e3e4_ablation")
if not abl_runs:
    layout.empty_state("e3e4_ablation", sel.cfg_name)
    st.stop()
chosen = st.selectbox("ablation run", list(reversed(abl_runs)), format_func=lambda r: r.run_id) if len(abl_runs) > 1 else abl_runs[0]
ab = data.load_ablation(chosen.run_dir)
rows = ab["rows"]
if rows is None or rows.empty:
    layout.empty_state("e3e4_ablation", sel.cfg_name)
    st.stop()

_e1_eval = None
try:
    from moe.utils import load_config, REPO_ROOT
    _cp = REPO_ROOT / "configs" / f"{sel.cfg_name}.yaml"
    if _cp.exists():
        _e1_eval = int(load_config(str(_cp))["train"]["eval_batches"])
except Exception:  # noqa: BLE001
    _e1_eval = None
st.caption(rio.ablation_eval_note(ab["meta"], _e1_eval))
_src = rows[["variant", "seed", "source_run_dir"]].drop_duplicates()
st.caption("Checkpoints evaluated (one per variant unless listed twice): "
           + "; ".join(f"{v} s{sd} (`{d}`)" for v, sd, d in _src.itertuples(index=False)))

# legend labels from the source runs' configs
cfgs = {}
for v, g in rows.groupby("variant"):
    src = data.load_run(g["source_run_dir"].iloc[0])
    cfgs[v] = src.config if src else None

st.subheader("Disable top-routed experts (E3)")
ymode = st.radio("y", ["Δ vs r=0", "absolute"], horizontal=True)
layout.safe_chart("disable top experts (A1)", lambda: figures.fig_disable_top(
    rows, y="delta" if ymode.startswith("Δ") else "abs", agg_seeds=sel.agg_seeds, cfgs=cfgs, template=tpl),
    experiment="e3e4_ablation", cfg_name=sel.cfg_name, key="a1")
st.caption("Switch can only disable whole experts (1/E steps); points sit at the representable fractions (PLAN §7 table). "
           "Hollow markers are the r=0 point at the training capacity factor (with drops).")

st.subheader("Shared expert and k (E4, eval-time change, not retrained)")
a, b = st.columns(2)
with a:
    layout.safe_chart("shared expert (A2)", lambda: figures.fig_shared_ablation(rows, template=tpl),
                      experiment="e3e4_ablation", cfg_name=sel.cfg_name, key="a2")
with b:
    layout.safe_chart("k_routed at eval (A3)", lambda: figures.fig_k_sweep(rows, template=tpl),
                      experiment="e3e4_ablation", cfg_name=sel.cfg_name, key="a3")

st.subheader("All rows")
studies = sorted(rows["study"].unique())
study = st.selectbox("study", studies)
st.dataframe(rio.summarize_e3e4(rows[rows["study"] == study]), hide_index=True, width="stretch")
