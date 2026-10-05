#!/usr/bin/env python
"""Build RESULTS.md and a self-contained static HTML report from ``results/<config>/``.

    .venv/bin/python scripts/make_report.py --config configs/smoke.yaml

* ``RESULTS.md`` (repo root): tables and short factual text only. Every table cites the run
  directory/file it was computed from; every number is computed here from results files
  (DASHBOARD_SPEC §7). Missing sources print ``n/a: not run yet``.
* ``results/<config>/report.html``: one file, Plotly JS inlined once (works offline), figures
  from the SAME builders as the dashboard (``dashboard/figures.py``) with default widget values.

Uses ``dashboard.results_io`` / ``dashboard.figures`` only (no streamlit).
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NA = "n/a: not run yet"


def md_table(df: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table from an all-string-able DataFrame."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row.tolist()) + " |")
    return "\n".join(lines)


class Report:
    """Collects the markdown and html bodies in parallel."""

    def __init__(self) -> None:
        self.md: list[str] = []
        self.html: list[str] = []
        self._first_fig = True

    def h(self, level: int, text: str) -> None:
        self.md.append(f"\n{'#' * level} {text}\n")
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def p(self, text: str, md_only: bool = False) -> None:
        self.md.append(text + "\n")
        if not md_only:
            self.html.append(f"<p>{html.escape(text)}</p>")

    def src(self, text: str) -> None:
        self.md.append(f"*Source: {text}*\n")
        self.html.append(f"<p class='src'>Source: {html.escape(text)}</p>")

    def table(self, df: pd.DataFrame, source: str) -> None:
        self.md.append(md_table(df) + "\n")
        self.html.append(df.to_html(index=False, border=0, escape=True, classes="t"))
        self.src(source)

    def banner(self, text: str) -> None:
        self.md.append(f"> **{text}**\n")
        self.html.append(f"<div class='banner'>{html.escape(text)}</div>")

    def figure(self, name: str, build: Callable[[], Any]) -> None:
        from dashboard.results_io import NoData
        try:
            fig = build()
        except NoData as exc:
            self.html.append(f"<p class='na'>{html.escape(name)}: not available: {html.escape(str(exc) or 'no data')}</p>")
            return
        except Exception as exc:  # noqa: BLE001
            self.html.append(f"<p class='na'>{html.escape(name)}: could not render ({type(exc).__name__}: {html.escape(str(exc))})</p>")
            print(f"[report] figure {name!r} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        fig.update_layout(title=dict(text=name))
        self.html.append(fig.to_html(full_html=False, include_plotlyjs="inline" if self._first_fig else False))
        self._first_fig = False


CSS = """body{font-family:Inter,system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0 auto;max-width:1100px;padding:16px 24px;
background:#fff;color:#1f1f1f;line-height:1.45}h1{margin-top:8px}h2{border-bottom:1px solid #e8e8e8;padding-bottom:4px;margin-top:36px}
table.t{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}table.t th,table.t td{padding:4px 10px;border-bottom:1px solid #e8e8e8;text-align:right;white-space:nowrap}
table.t th{text-align:right;background:#f6f6f6}.src{color:#6b6b6b;font-size:12px}.na{color:#6b6b6b;font-style:italic}
.banner{background:#fff3cd;border:1px solid #e0c36a;padding:10px 14px;font-weight:600;margin:12px 0}"""


def fmt_loss(x: Any) -> str:
    from dashboard import theme
    return theme.fmt_loss(x)


def fmt_bytes(x: Any) -> str:
    """SI bytes with 3 significant figures for report tables (912000 -> '912kB')."""
    from dashboard import theme
    return theme.fmt_bytes(x, digits=3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--results-dir", default=None, help="results root (default: $MOE_RESULTS_DIR or ./results)")
    ap.add_argument("--out-md", default=str(REPO / "RESULTS.md"))
    ap.add_argument("--out-html", default=None, help="default: <results>/<config>/report.html")
    args = ap.parse_args()
    if args.results_dir:
        os.environ["MOE_RESULTS_DIR"] = args.results_dir

    from dashboard import figures as F
    from dashboard import results_io as rio
    from dashboard import theme
    from moe.utils import load_config

    cfg_path = Path(args.config) if Path(args.config).is_absolute() else REPO / args.config
    base_cfg = load_config(str(cfg_path))
    name = base_cfg["name"]
    is_smoke = name == "smoke"
    root = rio.results_root()
    R = Report()

    # ---------------------------------------------------------------- load everything
    e1_infos = rio.list_runs(name, "e1_main")
    e1_runs = [r for r in (rio.load_run(d) for d in rio.default_selection(e1_infos)) if r]
    e1_done = rio.complete_runs(e1_runs)
    e2_runs = rio.e2_runs(name)
    e2 = rio.summarize_e2(e2_runs)
    abl_info = rio.latest(rio.list_runs(name, "e3e4_ablation"))
    abl = rio.load_ablation(abl_info.run_dir) if abl_info else None
    # newest profile run that finished (has roofline.json); fall back to the newest with data
    prof_infos = rio.list_runs(name, "e5_profile")
    prof_info = rio.latest(prof_infos, where=lambda r: (r.path / "roofline.json").exists()) or rio.latest(prof_infos)
    prof = rio.load_profile(prof_info.run_dir) if prof_info else None
    pl_info = rio.latest(rio.list_runs(name, "e6_placement"))
    pl = rio.load_placement(pl_info.run_dir) if pl_info else None
    default_E = rio.default_n_experts(base_cfg)
    train_n = rio.train_tokens_per_step(base_cfg)

    env = next((r.env for r in sorted(e1_runs, key=lambda r: r.info.run_id, reverse=True) if r.env), None) \
        or (prof or {}).get("env") or rio.newest_env(name)
    tpl = "moe_light"

    # ---------------------------------------------------------------- header
    R.h(1, "RESULTS")
    if is_smoke:
        R.banner("SMOKE (pipeline check, not meaningful results)")
    R.p(f"Config: `{name}` (`{Path(args.config).name}`). Generated by `scripts/make_report.py` from `{(root / name).relative_to(REPO) if (root / name).is_relative_to(REPO) else root / name}`. "
        "Tables and factual descriptions only; every number is computed from the cited results files.", md_only=True)
    R.html.append(f"<p>Config <code>{html.escape(name)}</code>. Every number is computed from the results files cited under each table.</p>")
    if env:
        R.h(2, "Hardware and precision")
        hw = pd.DataFrame([{"hardware": env.get("hardware_label"), "device": env.get("device_type"), "dtype": env.get("dtype"),
                            "torch": env.get("torch_version"), "CUDA": env.get("cuda_version"),
                            "VRAM (GB)": env.get("total_vram_gb") if env.get("total_vram_gb") is not None else "—"}])
        newest_e1 = max(e1_runs, key=lambda r: r.info.run_id).info.run_dir if e1_runs else "newest run"
        R.table(hw, f"results/{newest_e1}/env.json (profile runs write their own env.json, see E5)")
        pe = (prof or {}).get("env")
        if pe and pe.get("hardware_label") != env.get("hardware_label"):
            R.p(f"Note: the E5 profile run was executed on `{pe.get('hardware_label')}` ({pe.get('dtype')}), "
                f"different from the training runs (`{env.get('hardware_label')}`, {env.get('dtype')}).")
    else:
        R.p("No runs found: hardware is unknown. " + NA)

    # ---------------------------------------------------------------- headline numbers
    R.h(2, "Headline numbers")
    head: list[dict] = []
    s1 = rio.summarize_e1(e1_done)
    b = rio.headline_numbers(s1)
    if b["best_moe_variant"]:
        dense = s1[s1["variant"] == "dense"].iloc[0]
        best = s1[s1["variant"] == b["best_moe_variant"]].iloc[0]
        head.append({"quantity": "best MoE vs dense, final val loss (MoE minus dense, nats)",
                     "value": f"{b['best_moe_delta']:+.3f} ({b['best_moe_variant']} {best['final_val_loss']:.3f} vs dense {dense['final_val_loss']:.3f}; "
                              f"n seeds {int(best['n_seeds'])}/{int(dense['n_seeds'])})",
                     "source": "E1 final.json"})
    else:
        head.append({"quantity": "best MoE vs dense, final val loss", "value": NA if not e1_done else "n/a: needs a complete dense run and an MoE run",
                     "source": "E1 final.json"})
    sp = rio.summarize_e5_speedups(prof["bench"]) if prof else pd.DataFrame()
    if not sp.empty:
        lines = rio.speedup_headline_text(rio.speedup_headline(sp, train_n, default_E))
        head.append({"quantity": "loop to batched dispatch speedup (loop ms / batched ms) at the training tokens per step, "
                                 "and its range over the token sweep",
                     "value": "; ".join(lines) or "n/a: no token-sweep pairs", "source": "E5 bench.jsonl"})
    else:
        head.append({"quantity": "loop to batched dispatch speedup", "value": NA, "source": "E5 bench.jsonl"})
    if not e2.empty:
        parts = [f"cf={r.capacity_factor:g}: val {theme.fmt_pct(r.val_drop_fraction)}, train(last 10%) {theme.fmt_pct(r.train_drop_fraction_last)}"
                 for r in e2.itertuples()]
        head.append({"quantity": "Switch token drop rate at each capacity factor", "value": "; ".join(parts), "source": "E2 final.json"})
    else:
        head.append({"quantity": "Switch token drop rate at each capacity factor", "value": NA, "source": "E2 final.json"})
    e6 = rio.summarize_e6(pl["summary"]) if pl else pd.DataFrame()
    if not e6.empty and (e6["D"] == 8).any():
        g = e6[e6["D"] == 8]
        head.append({"quantity": "straggler slowdown at D=8 (max device load / mean, all layers)",
                     "value": "; ".join(f"{r.variant} {r.policy}: {r.straggler_mean:.3f}x" for r in g.itertuples()),
                     "source": "E6 placement_summary.jsonl"})
    else:
        head.append({"quantity": "straggler slowdown at D=8", "value": NA if e6.empty else "n/a: D=8 not in this E6 run",
                     "source": "E6 placement_summary.jsonl"})
    srcs = {"E1 final.json": ", ".join(f"results/{r.info.run_dir}/final.json" for r in e1_done) or "none",
            "E2 final.json": ", ".join(f"results/{d}/final.json" for d in sorted({d for rd in e2.get("run_dirs", []) for d in rd})) or "none",
            "E5 bench.jsonl": f"results/{prof_info.run_dir}/bench.jsonl" if prof_info else "none",
            "E6 placement_summary.jsonl": f"results/{pl_info.run_dir}/placement_summary.jsonl" if pl_info else "none"}
    hdf = pd.DataFrame(head)
    hdf["source"] = hdf["source"].map(lambda s: srcs.get(s, s))
    R.table(hdf, "computed in scripts/make_report.py from the files in the last column")
    if is_smoke:
        R.p("These are SMOKE numbers: tiny model, few steps, CPU. They check that the pipeline runs end to end and say nothing about the method.", md_only=True)

    # ---------------------------------------------------------------- E1
    R.h(2, "E1: variants at matched activated compute")
    if not e1_runs:
        R.p(NA)
    else:
        cfgs = rio.distinct_configs(e1_runs)
        for cfg, rs in cfgs:
            pt = F.table_params(cfg, rio.latest_final_by_variant(rs))
            f = theme.fmt_params
            df = pd.DataFrame({"variant": pt["variant"], "experts": pt["experts"], "expert width": pt["expert_width"],
                               "active/token": pt["activated"], "FFN total/layer": pt["ffn_total_per_layer"].map(f),
                               "FFN active/layer": pt["ffn_active_per_layer"].map(f), "router/layer": pt["router_per_layer"].map(f),
                               "total params (analytic)": pt["total_params"].map(f), "total params (measured)": pt["total_params_measured"].map(f),
                               "active params": pt["active_params"].map(f), "model fwd FLOPs/token": pt["model_fwd_flops_per_token"].map(f),
                               "matched": pt["matched"], "measured = analytic": pt["measured_eq_analytic"]})
            R.table(df, "moe.flops.param_table(config.json of " + ", ".join(f"results/{r.info.run_dir}" for r in rs[:1]) + ") + final.json total_params")
        R.p("Matching is on expert-FFN params/FLOPs; router params are listed separately and excluded.")
        if s1.empty:
            R.p("No complete E1 runs. " + NA)
        else:
            df = pd.DataFrame({
                "variant": [theme.variant_label(v, next((r.config for r in e1_done if r.variant == v), None)) for v in s1["variant"]],
                "seeds": s1["n_seeds"], "final val loss": s1["final_val_loss"].map(fmt_loss),
                "± half-range": [f"{h:.3f}" if n > 1 else "—" for h, n in zip(s1["half_range"], s1["n_seeds"])],
                "val ppl": s1["final_val_ppl"].map(theme.fmt_ppl), "delta vs dense": s1["delta_vs_dense"].map(theme.fmt_delta),
                "train CE (last 5%)": s1["final_train_ce"].map(fmt_loss),
                "median tok/s": s1["median_tokens_per_sec"].map(theme.fmt_tps),
                "peak memory": [theme.fmt_mib(m) if (m or 0) > 0 else "— (not measured on CPU)" for m in s1["peak_mem_mb"]]})
            R.table(df, "results/" + ", ".join(f"{r.info.run_dir}/final.json" for r in e1_done))
        sp_t = rio.seed_spread_table(e1_done)
        if not sp_t.empty and (sp_t["n_seeds"] > 1).any():
            _num = [c for c in sp_t.columns if c not in ("group", "n_seeds")]
            R.table(sp_t.assign(**{c: sp_t[c].map(fmt_loss) for c in _num}), "E1 final.json per seed")
        for nm, fn in (("Val loss vs tokens", lambda: F.fig_loss_curves(e1_runs, kind="val", template=tpl)),
                       ("Train CE vs tokens (EMA)", lambda: F.fig_loss_curves(e1_runs, kind="train", template=tpl)),
                       ("Aux loss", lambda: F.fig_aux_curves(e1_runs, template=tpl)),
                       ("Training throughput (median tok/s)", lambda: F.fig_throughput_bars(e1_done, template=tpl)),
                       ("Peak memory", lambda: F.fig_memory_bars(e1_done, template=tpl))):
            R.figure(nm, fn)

    # ---------------------------------------------------------------- routing
    R.h(2, "Routing statistics (E1 runs)")
    moe_runs = [r for r in e1_runs if not r.routing.empty]
    if not moe_runs:
        R.p(NA)
    else:
        rows = []
        for r in moe_runs:
            last = r.routing.sort_values("step").groupby("layer").tail(1)
            rows.append({"run": r.info.label, "last logged step": int(last["step"].max()),
                         "load imbalance (max/mean), mean over layers": f"{last['load_imbalance'].mean():.2f}",
                         "router entropy / ln E": f"{(last['router_entropy'] / last['max_entropy']).mean():.3f}",
                         "drop fraction (window)": theme.fmt_pct(last["drop_fraction"].mean()) if r.variant in ("switch", "gshard") else "— (no drops)"})
        R.table(pd.DataFrame(rows), "results/<run>/routing_log.jsonl, last record per layer: " + ", ".join(f"{r.info.run_dir}" for r in moe_runs))
        ds = next((r for r in moe_runs if r.variant == "deepseek"), moe_runs[0])
        R.figure("Expert load over training", lambda: F.fig_expert_heatmap(ds.routing, run_name=ds.info.label, template=tpl))
        R.figure("Final expert load per layer", lambda: F.fig_final_load_hist(ds.routing, variant=ds.variant, run_name=ds.info.label, template=tpl))
        for m in ("load_imbalance", "router_entropy", "drop_fraction"):
            R.figure(m.replace("_", " "), lambda m=m: F.fig_routing_metric(moe_runs, metric=m, template=tpl))

    # ---------------------------------------------------------------- E2
    R.h(2, "E2: Switch capacity factor sweep")
    if e2.empty:
        R.p(NA)
    else:
        ref = float(e2["ref_capacity_factor"].iloc[0])
        df = pd.DataFrame({
            "capacity factor": e2["capacity_factor"].map(lambda c: f"{c:g}"),
            "capacity per expert C": e2["capacity_per_expert"].map(lambda c: "—" if pd.isna(c) else f"{int(c):,}"),
            "seeds": e2["n_seeds"], "val loss": e2["final_val_loss"].map(fmt_loss),
            f"delta vs cf={ref:g}": e2["delta_loss_vs_ref"].map(theme.fmt_delta),
            "val drop": e2["val_drop_fraction"].map(theme.fmt_pct),
            "train drop (last 10%)": e2["train_drop_fraction_last"].map(theme.fmt_pct),
            "median tok/s": e2["median_tokens_per_sec"].map(theme.fmt_tps),
            "peak memory": [theme.fmt_mib(m) if (m or 0) > 0 else "— (not measured on CPU)" for m in e2["peak_mem_mb"]]})
        R.table(df, "results/" + name + "/e2_capacity/sweep_index.json -> " + ", ".join(sorted({d for rd in e2["run_dirs"] for d in rd})))
        for nm, fn in (("Val loss vs capacity factor", lambda: F.fig_cf_metric(e2, "final_val_loss", template=tpl)),
                       ("Drop fraction vs capacity factor", lambda: F.fig_cf_drop(e2, template=tpl)),
                       ("Throughput vs capacity factor", lambda: F.fig_cf_metric(e2, "median_tokens_per_sec", template=tpl)),
                       ("Peak memory vs capacity factor", lambda: F.fig_cf_metric(e2, "peak_mem_mb", template=tpl))):
            R.figure(nm, fn)

    # ---------------------------------------------------------------- E3/E4
    R.h(2, "E3/E4: expert ablations (eval-time changes on trained checkpoints, not retrained)")
    if abl is None or abl["rows"].empty:
        R.p(NA)
    else:
        rows = abl["rows"]
        a_src = f"results/{abl_info.run_dir}/ablation.jsonl"
        try:
            e1_eval = int(base_cfg["train"]["eval_batches"])
        except (KeyError, TypeError, ValueError):
            e1_eval = None
        R.p(rio.ablation_eval_note(abl["meta"], e1_eval))
        e3 = rows[rows["study"] == "e3_disable_top"].copy()
        if not e3.empty:
            e3["reference"] = e3["is_reference"].map(lambda x: "at training cf (with drops)" if x else "no-drop")
            df = pd.DataFrame({"variant": e3["variant"], "seed": e3["seed"], "disabled / routed": [f"{int(a)} / {int(b)}" for a, b in zip(e3["n_disable_top_routed"], e3["n_routed"])],
                               "fraction disabled": e3["frac_disabled"].map(lambda x: f"{x:.1%}"), "eval": e3["reference"],
                               "val loss": e3["val_loss"].map(fmt_loss), "delta vs r=0": e3["delta_val_loss"].map(theme.fmt_delta)})
            R.p("E3: disable the top-r routed experts per token (r as a count; the fraction column is the actual fraction).")
            R.table(df, a_src + " (study e3_disable_top)")
        e4 = rows[rows["study"].isin(["e4_shared", "e4_k_sweep"])]
        if not e4.empty:
            df = pd.DataFrame({"study": e4["study"], "condition": e4["condition"].fillna("—"), "k_routed": e4["k_routed"],
                               "active FFN width": e4["active_ffn_width"], "val loss": e4["val_loss"].map(fmt_loss),
                               "delta vs baseline": e4["delta_val_loss"].map(theme.fmt_delta)})
            R.p("E4: shared expert disabled (a: same k, b: k+1) and k_routed swept at eval.")
            R.table(df, a_src + " (studies e4_shared, e4_k_sweep)")
        cfgs_v = {}
        for v, g in rows.groupby("variant"):
            src_r = rio.load_run(g["source_run_dir"].iloc[0])
            cfgs_v[v] = src_r.config if src_r else None
        for nm, fn in (("Disable top routed experts", lambda: F.fig_disable_top(rows, cfgs=cfgs_v, template=tpl)),
                       ("Shared-expert ablation", lambda: F.fig_shared_ablation(rows, template=tpl)),
                       ("k_routed at eval", lambda: F.fig_k_sweep(rows, template=tpl))):
            R.figure(nm, fn)

    # ---------------------------------------------------------------- E5
    R.h(2, "E5: single-layer systems profile")
    if prof is None or prof["bench"].empty:
        R.p(NA)
    else:
        pm = prof["meta"] or {}
        p_src = f"results/{prof_info.run_dir}"
        R.p(f"Profile run `{prof_info.run_id}`: device `{pm.get('device_label', '—')}`, dtype `{pm.get('dtype', '—')}`, timer `{pm.get('timer', '—')}`, "
            f"{pm.get('repeats', '—')} timed repeats after {pm.get('warmup_iters', '—')} warmup iterations (median reported).")
        if pm.get("note"):
            R.p(pm["note"])
        if not sp.empty:
            wide = rio.speedup_vs_n_table(sp, default_E)
            if wide.empty:
                R.p("No loop/batched pairs in the token sweep.")
            else:
                wide = wide.rename(columns={c: f"N={int(c):,}" for c in wide.columns if c not in ("variant", "pass")})
                for c in wide.columns[2:]:
                    wide[c] = wide[c].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}x")
                R.p(f"Loop to batched speedup (loop median ms / batched median ms) vs tokens N at the config expert count"
                    + (f"; training uses N = {train_n:,} tokens per step." if train_n else "."))
                R.table(wide, p_src + "/bench.jsonl (sweep=tokens)")
        roof = prof["roofline"]
        if roof:
            rows = []
            for c in roof.get("ceilings", []):
                rows.append({"ceiling": c.get("kind"), "peak TFLOP/s": theme.fmt_intensity(c.get("peak_tflops")) if c.get("peak_tflops") else "unavailable",
                             "peak GB/s": theme.fmt_intensity(c.get("peak_gbps")) if c.get("peak_gbps") else "unavailable",
                             "method / source": c.get("method") or c.get("source") or "—"})
            R.table(pd.DataFrame(rows), p_src + "/roofline.json")
            meas = next((c for c in roof.get("ceilings", []) if c.get("kind") == "measured"), None)
            b = prof["bench"]
            if meas and meas.get("peak_tflops"):
                mx = float(b["achieved_tflops"].max())
                frac = mx / meas["peak_tflops"]
                R.p(f"Check: highest achieved layer throughput in bench.jsonl is {mx:.3g} TFLOP/s = {frac:.0%} of the measured peak "
                    f"({meas['peak_tflops']:.3g} TFLOP/s)" + (". **EXCEEDS the measured peak: investigate.**" if frac > 1.05 else "."))
        else:
            R.p("roofline.json missing for the newest profile run.")
        ek = dict(n_experts=default_E or None, template=tpl)
        inc = prof["breakdown"]["other_includes"].dropna().unique().tolist() if "other_includes" in prof["breakdown"] else []
        R.p("Time breakdown: for pass fwd_bwd, the 'other' category contains the whole backward pass (the manual phase timers wrap "
            "forward phases only)." + (f" breakdown.jsonl other_includes: {'; '.join(inc)}." if inc else ""))
        for nm, fn in (("Latency vs tokens", lambda: F.fig_latency_vs_tokens(prof["bench"], passes=("fwd", "fwd_bwd"), **ek)),
                       ("Latency vs expert count", lambda: F.fig_latency_vs_experts(prof["bench"], passes=("fwd", "fwd_bwd"), template=tpl)),
                       ("Time breakdown (fwd, cuda_events phase timers, largest N)", lambda: F.fig_breakdown(prof["breakdown"], n_tokens=int(prof["breakdown"]["n_tokens"].max()), n_experts=default_E or None, method="cuda_events", template=tpl)),
                       ("Achieved TFLOP/s", lambda: F.fig_achieved_tflops(prof["bench"], roof, passes=("fwd", "fwd_bwd"), **ek)),
                       ("Roofline (layers)", lambda: F.fig_roofline(prof["bench"], roof, template=tpl)),
                       ("Roofline (single GEMMs)", lambda: F.fig_roofline(prof["bench"], roof, gemm=prof["gemm"], points="gemm", template=tpl)),
                       ("Peak memory", lambda: F.fig_memory(prof["bench"], template=tpl)),
                       ("Capacity factor: memory and padding", lambda: F.fig_capacity_memory(prof["bench"], template=tpl))):
            R.figure(nm, fn)

    # ---------------------------------------------------------------- E6
    R.h(2, "E6: expert placement (simple analytical model, not a simulator)")
    if pl is None or e6.empty:
        R.p(NA)
    else:
        params = pl["params"] or {}
        note = " ".join(str(params.get("note", "")).replace("simple analytical model, not a simulator.", "")
                        .replace("Simple analytical model, not a simulator.", "").split())
        if note:
            R.p(note)
        df = pd.DataFrame({"variant": e6["variant"], "D": e6["D"], "policy": e6["policy"],
                           "straggler mean": e6["straggler_mean"].map(lambda x: f"{x:.3f}x"), "p90": e6["straggler_p90"].map(lambda x: f"{x:.3f}x"),
                           "all-to-all fwd+bwd / step": e6["a2a_bytes_fwd_bwd_mean"].map(fmt_bytes),
                           "est. step ms (default params)": e6["step_ms_default"].map(lambda x: f"{x:.3g}")})
        R.table(df, f"results/{pl_info.run_dir}/placement_summary.jsonl (layer = all); default device TFLOP/s {params.get('device_tflops_default')} "
                    f"({params.get('device_tflops_source')}), link {params.get('link_gbps_default')} GB/s")
        s_all = pl["summary"][pl["summary"]["layer"].astype(str) == "all"]
        tr = next((t for t in params.get("traces", []) if t.get("variant") == "deepseek"), (params.get("traces") or [None])[0])
        Dmax = int(max(params.get("D_values", [1])))
        loads = []
        if tr:
            for r in s_all[(s_all["D"] == Dmax) & (s_all["trace_dir"] == tr["trace_dir"])].itertuples():
                loads += [{"variant": tr["variant"], "policy": r.policy, "device": d, "tokens": float(x)} for d, x in enumerate(r.mean_tokens_per_device)]
        R.figure(f"Load per device (D={Dmax}, all layers, mean over steps)", lambda: F.fig_device_load(pd.DataFrame(loads), template=tpl))
        R.figure("Straggler slowdown vs D", lambda: F.fig_straggler_vs_D(s_all, template=tpl))
        R.figure("All-to-all bytes per step", lambda: F.fig_a2a_bytes(s_all, template=tpl))
        try:
            from scripts.placement_sim import load_placement_arrays
            arrays = load_placement_arrays(Path(pl["arrays_path"]).parent) if pl["arrays_path"] else {}
            ti = params["traces"].index(tr)
            grid = np.logspace(0, 3, 50)
            curves = rio.placement_step_times(arrays, params, ti, tr["source_run_id"], Dmax, params["policies"], None, grid,
                                              device_tflops=float(params["device_tflops_default"]),
                                              include_backward=bool(params.get("include_backward_default", True)))
            R.figure(f"Estimated step time vs link bandwidth (D={Dmax}, {tr['variant']})",
                     lambda: F.fig_step_time_vs_bw(curves, marker_bw=float(params["link_gbps_default"]), variant=tr["variant"], template=tpl))
        except Exception as exc:  # noqa: BLE001
            R.html.append(f"<p class='na'>Step-time curve not available: {html.escape(str(exc))}</p>")

    # ---------------------------------------------------------------- excluded runs
    sup_readme = root / name / "_superseded" / "README.md"
    if sup_readme.exists():
        R.h(2, "Excluded runs")
        text = sup_readme.read_text(encoding="utf-8").strip()
        R.md.append(text + "\n")
        R.html.append(f"<pre>{html.escape(text)}</pre>")

    # ---------------------------------------------------------------- write
    md_path = Path(args.out_md)
    md_path.write_text("\n".join(R.md).strip() + "\n", encoding="utf-8")
    out_html = Path(args.out_html) if args.out_html else root / name / "report.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(R.html)
    out_html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>MoE repro report ({html.escape(name)})</title>"
                        f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    print(f"wrote {md_path} ({md_path.stat().st_size:,} B) and {out_html} ({out_html.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
