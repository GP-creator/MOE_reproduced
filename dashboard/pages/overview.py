"""Overview page (DASHBOARD_SPEC §4.1): what was built, hardware, param/FLOP matching, headline tiles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, figures, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

sel = layout.get_selection()
layout.page_header("Overview", "What was built, on what hardware, and are the variants really compute-matched?", experiment="e1_main")

# O1 intro (fixed descriptive text, no numbers)
st.markdown(
    "A small-scale reproduction of the **Switch Transformer** (top-1 routing, capacity factor, load-balancing loss) "
    "and **DeepSeekMoE** (fine-grained experts plus an always-on shared expert) mechanisms, with the dense baselines "
    "matched in activated compute. Besides model quality it characterises the systems side on a single GPU: kernel "
    "timings, roofline position, and an analytical expert-placement model. Everything shown is read from `results/`.")

runs = data.list_runs(sel.cfg_name, "e1_main")
sel_runs = data.load_runs(list(sel.run_dirs))

# O2 environment card
st.subheader("Environment")
newest = rio.latest([r.info for r in sel_runs]) if sel_runs else None
env = next((r.env for r in sel_runs if newest and r.info.run_id == newest.run_id), None)
if not env:
    layout.empty_state("e1_main", sel.cfg_name)
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hardware", env.get("hardware_label") or "unknown")
    c2.metric("Dtype", env.get("dtype") or "—")
    c3.metric("torch / CUDA", f"{env.get('torch_version', '—')} / {env.get('cuda_version') or '—'}")
    vram = env.get("total_vram_gb")
    c4.metric("VRAM", f"{vram:.1f} GB" if isinstance(vram, (int, float)) else "— (CPU)")

# O3 param / FLOP table
st.subheader("Parameters and FLOPs per variant")


def _param_df(cfg: dict, finals: dict) -> pd.DataFrame:
    t = figures.table_params(cfg, finals)
    f = theme.fmt_params
    return pd.DataFrame({
        "variant": t["variant"], "experts": t["experts"], "expert width": t["expert_width"],
        "active/token": t["activated"], "FFN total/layer": t["ffn_total_per_layer"].map(f),
        "FFN active/layer": t["ffn_active_per_layer"].map(f), "router/layer": t["router_per_layer"].map(f),
        "total params (analytic)": t["total_params"].map(f),
        "total params (measured)": t["total_params_measured"].map(f),
        "active params": t["active_params"].map(f),
        "FFN fwd FLOPs/token": t["ffn_flops_per_token_model"].map(f),
        "model fwd FLOPs/token": t["model_fwd_flops_per_token"].map(f),
        "matched": t["matched"], "measured = analytic": t["measured_eq_analytic"]})


def _show_params() -> None:
    cfgs = rio.distinct_configs(sel_runs)
    if not cfgs:
        from moe.utils import load_config, REPO_ROOT
        path = REPO_ROOT / "configs" / f"{sel.cfg_name}.yaml"
        if not sel.cfg_name or not path.exists():
            st.info("No runs selected and no config file to compute the analytic table from.")
            return
        st.dataframe(_param_df(load_config(str(path)), {}), hide_index=True, width="stretch")
        st.caption(f"analytic, from configs/{sel.cfg_name}.yaml; no measured runs yet")
    else:
        if len(cfgs) > 1:
            st.warning("The selected runs use different model/moe/data configs; one table per config.")
        for cfg, rs in cfgs:
            st.dataframe(_param_df(cfg, rio.latest_final_by_variant(rs)), hide_index=True, width="stretch")
    st.caption("Matching is on expert-FFN params/FLOPs; router params are listed separately and excluded (PLAN §5).")


try:
    _show_params()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not render the parameter table: {type(exc).__name__}: {exc}")

# O4 headline tiles
st.subheader("Headline results")
summary = rio.summarize_e1(sel_runs)
if summary.empty:
    st.info("Needs complete E1 runs; none selected." + f" Run `{rio.make_hint('e1_main', sel.cfg_name)}`.")
else:
    if not (summary["variant"] == "dense").any():
        st.caption("needs a complete dense run for the delta tiles")
    best = figures.headline_numbers(summary)
    if best["best_moe_variant"]:
        st.metric(f"Best MoE vs dense ({best['best_moe_variant']})", theme.fmt_delta(best["best_moe_delta"]),
                  delta=theme.fmt_delta(best["best_moe_delta"]), delta_color="inverse",
                  help="val-loss difference in nats, MoE minus dense; negative = MoE better")
    for _, row in summary.iterrows():
        cols = st.columns(4)
        label = theme.variant_label(row["variant"], next((r.config for r in sel_runs if r.variant == row["variant"]), None))
        loss = theme.fmt_loss(row["final_val_loss"]) + (f" ± {row['half_range']:.3f}" if row["n_seeds"] > 1 else "")
        cols[0].metric(f"{label}: val loss", loss + (f" (n={row['n_seeds']})" if row["n_seeds"] > 1 else ""))
        cols[1].metric("Δ vs dense", theme.fmt_delta(row["delta_vs_dense"]),
                       delta=None if row["delta_vs_dense"] is None or row["variant"] == "dense" else theme.fmt_delta(row["delta_vs_dense"]),
                       delta_color="inverse")
        cols[2].metric("median tok/s", theme.fmt_tps(row["median_tokens_per_sec"]))
        cols[3].metric("peak memory", theme.fmt_mib(row["peak_mem_mb"]) if (row["peak_mem_mb"] or 0) > 0 else "— (not measured on CPU)")

# O5 experiment status
st.subheader("Experiment status")
rows = []
for exp in rio.EXPERIMENTS:
    rs = data.list_runs(sel.cfg_name, exp)
    newest_run = rio.latest(rs)
    rows.append({"experiment": rio.EXPERIMENT_TITLE[exp], "runs (complete/total)": f"{sum(r.complete for r in rs)}/{len(rs)}",
                 "newest run": newest_run.run_id[:15] if newest_run else "—",
                 "source dir": f"results/{sel.cfg_name}/{exp}" if rs else "—"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
