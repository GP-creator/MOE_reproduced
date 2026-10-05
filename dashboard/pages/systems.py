"""Systems page, E5 (DASHBOARD_SPEC §4.6): latency, loop vs batched, breakdown, TFLOP/s, roofline, memory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Systems profile (E5)", "Where does MoE layer time go on this GPU, and where do the kernels sit on the roofline?")
tpl = layout.current_template()
profs = data.list_runs(sel.cfg_name, "e5_profile")
if not profs:
    layout.empty_state("e5_profile", sel.cfg_name)
    st.stop()
chosen = st.selectbox("profile run", list(reversed(profs)), format_func=lambda r: r.run_id) if len(profs) > 1 else profs[0]
prof = data.load_profile(chosen.run_dir)
bench, gemm, bd, roof, meta = prof["bench"], prof["gemm"], prof["breakdown"], prof["roofline"], prof["meta"] or {}
env = prof["env"]
st.caption(f"Source: `results/{chosen.run_dir}` · {layout.hardware_caption(env)} · device {meta.get('device_label', '—')} · dtype {meta.get('dtype', '—')} · timer {meta.get('timer', '—')}")
for n in layout.throttle_notes(env):
    st.warning("GPU may have been power-capped/throttled during this run: " + n)
if bench.empty:
    layout.empty_state("e5_profile", sel.cfg_name)
    st.stop()
if roof is None:
    st.warning("roofline.json is missing (profile run still in progress or incomplete); roofline ceilings are not shown.")
if meta.get("note"):
    st.caption(meta["note"])

# Power-state flags written by profile_layer.py (roofline.json measured ceiling + bench rows).
_meas = next((c for c in (roof or {}).get("ceilings", []) if c.get("kind") == "measured"), None)
if _meas and _meas.get("power_state_changed"):
    _st, _en = _meas.get("start") or {}, _meas.get("end") or {}
    _mclk = sorted({int(g["mem_clock_mhz"]) for col in ("gpu_state_before", "gpu_state_after") if col in bench
                    for g in bench[col] if isinstance(g, dict) and g.get("mem_clock_mhz")})
    _nlow = int(bench["low_power_state"].fillna(False).astype(bool).sum()) if "low_power_state" in bench else None
    st.warning(
        "profile_layer.py flagged `power_state_changed` for this run: the GPU clocks moved between benchmark groups"
        + (f" (memory clock values seen: {', '.join(map(str, _mclk))} MHz)" if _mclk else "") + ". "
        + (f"Measured roof at start / end: {_st.get('peak_tflops', float('nan')):.3g} / {_en.get('peak_tflops', float('nan')):.3g} TFLOP/s, "
           f"{_st.get('peak_gbps', float('nan')):.3g} / {_en.get('peak_gbps', float('nan')):.3g} GB/s. " if _st and _en else "")
        + (f"{_nlow} of {len(bench)} benchmark rows ran in a low-power state." if _nlow is not None else ""))

# default expert counts per variant from the config of this results set
default_E: dict = {}
train_n = None
try:
    from moe.utils import load_config, REPO_ROOT
    cfg_path = REPO_ROOT / "configs" / f"{sel.cfg_name}.yaml"
    if cfg_path.exists():
        _cfg = load_config(str(cfg_path))
        default_E = rio.default_n_experts(_cfg)
        train_n = rio.train_tokens_per_step(_cfg)
except Exception:  # noqa: BLE001
    default_E = {}

variants_all = sorted(bench["variant"].unique(), key=theme.variant_sort_key)
c1, c2, c3 = st.columns(3)
pass_choice = c1.radio("pass", ["fwd", "fwd_bwd", "both"], horizontal=True)
passes = ("fwd", "fwd_bwd") if pass_choice == "both" else (pass_choice,)
variants = c2.multiselect("variants", variants_all, default=variants_all)
disp_all = [d for d in ("batched", "loop") if (bench["dispatch"] == d).any()]
dispatches = c3.multiselect("dispatch", disp_all, default=disp_all)

tok = bench[bench["sweep"] == "tokens"]
e_opts = sorted(int(e) for e in tok[tok["variant"] != "dense"]["n_experts"].unique())
e_pick = st.selectbox("experts for the token sweep", ["config default"] + e_opts)
n_exp = (default_E or None) if e_pick == "config default" else {v: int(e_pick) for v in variants_all if v != "dense"}
ek = dict(passes=passes, variants=variants, dispatches=dispatches)

st.subheader("Latency")
log_y = st.toggle("log y", value=True)
layout.safe_chart("latency vs tokens (S1)", lambda: figures.fig_latency_vs_tokens(bench, n_experts=n_exp, log_y=log_y, template=tpl, **ek),
                  experiment="e5_profile", cfg_name=sel.cfg_name, key="s1")
n_opts = sorted(int(n) for n in bench[bench["sweep"] == "experts"]["n_tokens"].unique())
n_pick = st.selectbox("tokens for the expert-count sweep", n_opts) if n_opts else None
layout.safe_chart("latency vs experts (S2)", lambda: figures.fig_latency_vs_experts(bench, n_tokens=n_pick, log_y=log_y, template=tpl, **ek),
                  experiment="e5_profile", cfg_name=sel.cfg_name, key="s2")

st.subheader("Loop vs batched dispatch")
sp = rio.summarize_e5_speedups(bench)
if sp.empty:
    st.info("No loop/batched pairs in this profile run.")
else:
    layout.safe_chart("speedup vs tokens", lambda: figures.fig_speedup_vs_tokens(sp, n_experts=default_E or None, train_n=train_n, template=tpl),
                      experiment="e5_profile", cfg_name=sel.cfg_name, key="s3")
    st.caption("Speedup = loop median ms / batched median ms at the config's expert count (solid = fwd, dashed = fwd_bwd; "
               "dotted line = parity). It falls as N grows" + (f"; training uses N = {train_n:,} tokens per step." if train_n else "."))
    show = sp.assign(loop_ms=sp["loop_ms"].map(lambda x: f"{x:.3g}"), batched_ms=sp["batched_ms"].map(lambda x: f"{x:.3g}"),
                     speedup=sp["speedup"].map(lambda x: f"{x:.2f}×"))
    st.dataframe(show[[c for c in ("variant", "sweep", "n_tokens", "n_experts", "pass", "loop_ms", "batched_ms", "speedup") if c in show]],
                 hide_index=True, width="stretch")

st.subheader("Time breakdown")
b1, b2, b3, b4 = st.columns(4)
bd_n = sorted(int(n) for n in bd["n_tokens"].unique()) if not bd.empty else []
bd_pick = b1.selectbox("tokens (breakdown)", bd_n, index=len(bd_n) - 1 if bd_n else 0) if bd_n else None
bd_pass = b2.radio("pass (breakdown)", ["fwd", "fwd_bwd"], horizontal=True)
bd_unit = b3.radio("unit", ["ms", "% of total"], horizontal=True)
bd_methods = sorted(bd["method"].dropna().unique()) if (not bd.empty and "method" in bd) else []
bd_method = (b4.radio("timing method", bd_methods, horizontal=True,
                      index=bd_methods.index("cuda_events") if "cuda_events" in bd_methods else 0)
             if len(bd_methods) > 1 else (bd_methods[0] if bd_methods else None))
layout.safe_chart("time breakdown (S4)", lambda: figures.fig_breakdown(
    bd, n_tokens=bd_pick, n_experts=default_E or None, pass_=bd_pass, percent=bd_unit != "ms", method=bd_method,
    template=tpl), experiment="e5_profile", cfg_name=sel.cfg_name, key="s4")
if bd_method:
    st.caption(f"Phase timing method: {bd_method}. cuda_events = wall time of each phase (includes launch gaps and "
               "host syncs; comparable to the latency charts); torch.profiler = summed GPU kernel time inside each "
               "phase range. The two are never averaged.")
_inc = {}
if not bd.empty and "other_includes" in bd:
    _inc = bd[["pass", "other_includes"]].dropna().drop_duplicates("pass").set_index("pass")["other_includes"].to_dict()
if bd_pass == "fwd_bwd":
    st.caption("For pass = fwd_bwd, 'other' contains the whole backward pass (the manual phase timers wrap forward phases only)."
               + (f" Recorded in breakdown.jsonl: other = {_inc['fwd_bwd']}." if _inc.get("fwd_bwd") else ""))
else:
    st.caption("'other' = " + (_inc.get("fwd") or "unannotated ops") + ".")

st.subheader("Achieved TFLOP/s and roofline")
useful = st.toggle("useful FLOPs only (exclude padding)", value=False)
layout.safe_chart("achieved TFLOP/s (S5)", lambda: figures.fig_achieved_tflops(bench, roof, n_experts=n_exp, useful=useful, template=tpl, **ek),
                  experiment="e5_profile", cfg_name=sel.cfg_name, key="s5")
r1, r2 = st.columns(2)
pts = r1.radio("points", ["layer", "GEMM"], horizontal=True)
r_pass = r2.radio("pass (roofline)", ["fwd", "fwd_bwd"], horizontal=True, disabled=pts == "GEMM")
src_df = gemm if pts == "GEMM" else bench
e_all = sorted(int(e) for e in src_df["n_experts"].unique()) if not src_df.empty else []
e_sel = st.multiselect("n_experts (roofline)", e_all, default=e_all)
layout.safe_chart("roofline (S6)", lambda: figures.fig_roofline(
    bench, roof, gemm=gemm, points="gemm" if pts == "GEMM" else "layer", pass_=r_pass, n_experts=e_sel or None, template=tpl),
    experiment="e5_profile", cfg_name=sel.cfg_name, key="s6")
spec_c = next((c for c in (roof or {}).get("ceilings", []) if c.get("kind") == "spec"), None)
if roof is not None and not (spec_c and spec_c.get("peak_tflops")):
    st.caption("spec-sheet peak unavailable; measured roof only")
st.caption("Arithmetic intensity uses compulsory traffic (each operand read once), so it is an upper bound; see `moe/flops.py`.")
if roof is not None:
    meas = next((c for c in roof.get("ceilings", []) if c.get("kind") == "measured"), None)
    if meas and meas.get("peak_tflops") and (bench["achieved_tflops"] > 1.05 * meas["peak_tflops"]).any():
        st.warning("Some achieved TFLOP/s exceed the measured peak ceiling by more than 5%: check timing and the FLOP model.")

st.subheader("Memory")
m_opts = sorted(int(n) for n in tok["n_tokens"].unique())
m_pick = st.selectbox("tokens (memory)", m_opts, index=len(m_opts) - 1) if m_opts else None
layout.safe_chart("memory (S7)", lambda: figures.fig_memory(bench, n_tokens=m_pick, template=tpl),
                  experiment="e5_profile", cfg_name=sel.cfg_name, key="s7")
layout.safe_chart("capacity memory (S8)", lambda: figures.fig_capacity_memory(bench, template=tpl),
                  experiment="e5_profile", cfg_name=sel.cfg_name, key="s8")

with st.expander("Benchmark conditions"):
    if "gpu_state_before" in bench and bench["gpu_state_before"].apply(lambda x: isinstance(x, dict)).any():
        rows = []
        for r in bench.itertuples():
            b, a = r.gpu_state_before, r.gpu_state_after
            if isinstance(b, dict) and isinstance(a, dict):
                rows.append({"config": r.label, "SM clock before": b.get("sm_clock_mhz"), "SM clock after": a.get("sm_clock_mhz"),
                             "mem clock": a.get("mem_clock_mhz"), "temp C": a.get("temperature_c"), "power W": a.get("power_draw_w"),
                             "possible throttle": bool(b.get("sm_clock_mhz") and a.get("sm_clock_mhz") and a["sm_clock_mhz"] < 0.9 * b["sm_clock_mhz"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No GPU state recorded (CPU run).")
with st.expander("Roofline sources"):
    if roof:
        st.dataframe(pd.DataFrame([{k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in c.items()} for c in roof.get("ceilings", [])]),
                     hide_index=True, width="stretch")
        if roof.get("reference_gpus"):
            st.caption("spec-sheet values (not measured here)")
            st.dataframe(pd.DataFrame(roof["reference_gpus"]), hide_index=True, width="stretch")
    else:
        st.info("No roofline.json.")
tf = prof["trace_file"]
if tf is not None:
    st.caption(f"Profiler trace: `{tf}` ({Path(tf).stat().st_size / 1e6:.2f} MB)")
    st.download_button("Download profiler trace (open in chrome://tracing or Perfetto)", data=Path(tf).read_bytes(),
                       file_name=Path(tf).name, mime="application/json")
else:
    st.caption("no trace saved")
