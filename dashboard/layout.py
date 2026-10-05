"""Shared Streamlit layout: sidebar, Selection, page header, empty states, safe charts
(DASHBOARD_SPEC §2.3-§2.5). Pages use only these helpers for chrome and state.

Session-state keys owned by the sidebar (spec §2.3): ``cfg_name``, ``selected_runs``,
``agg_seeds``. Pages must read them through :func:`get_selection`, never directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard import data
from dashboard import results_io as rio
from dashboard import theme

# ----------------------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------------------


def current_template() -> str:
    """``"moe_dark"`` when Streamlit runs in dark mode, else ``"moe_light"`` (spec §3.2)."""
    try:
        t = st.context.theme.type
    except Exception:  # noqa: BLE001  (bare mode / AppTest / older runtimes)
        t = None
    return "moe_dark" if t == "dark" else "moe_light"


# ----------------------------------------------------------------------------------------
# Selection (sidebar state)
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """What the sidebar selected. ``cfg_name`` is None when no results exist at all."""

    cfg_name: Optional[str]
    run_dirs: tuple[str, ...] = field(default_factory=tuple)
    agg_seeds: bool = True


def get_selection() -> Selection:
    """The only way pages read sidebar state (spec §2.3)."""
    ss = st.session_state
    if "cfg_name" in ss:
        cfg_name = ss.get("cfg_name")
        run_dirs = ss.get("selected_runs")
    else:
        # Page run without the sidebar (standalone script / test): use the spec defaults.
        configs = data.list_configs()
        cfg_name = next((c for c in ("main", "small", "smoke") if c in configs), None)
        run_dirs = None
    if run_dirs is None:
        run_dirs = rio.default_selection(data.list_runs(cfg_name, "e1_main"))
    return Selection(cfg_name=cfg_name, run_dirs=tuple(run_dirs),
                     agg_seeds=bool(ss.get("agg_seeds", True)))


def run_label(info: rio.RunInfo) -> str:
    """``"{variant} · s{seed} · {run_id[:15]}"`` (+ " (incomplete)")."""
    return info.label


def hardware_caption(env: Optional[dict]) -> str:
    """``Hardware: <hardware_label> · <dtype>`` from env.json (never hard-coded)."""
    if not env:
        return "Hardware: no runs yet"
    hw = env.get("hardware_label") or env.get("gpu_name") or env.get("device_type") or "unknown"
    dt = env.get("dtype")
    return f"Hardware: {hw}" + (f" · {dt}" if dt else "")


def sidebar() -> Selection:
    """Render the sidebar (spec §2.3) into ``st.sidebar`` and fill ``st.session_state``."""
    ss = st.session_state
    configs = data.list_configs()
    with st.sidebar:
        st.markdown("### MoE repro")
        hw_slot = st.empty()

        # 2. config radio
        if configs:
            default = next((c for c in ("main", "small", "smoke") if c in configs), configs[0])
            if ss.get("cfg_name") not in configs:
                ss["cfg_name"] = default
            st.radio("Config", configs, key="cfg_name", horizontal=True)
        else:
            ss["cfg_name"] = None
            st.radio("Config", list(rio.CONFIG_NAMES), index=None, disabled=True, key="_cfg_disabled")
            st.caption("No results yet: run `make smoke` to produce some.")
        cfg_name = ss.get("cfg_name")

        # 3. E1 run multiselect
        runs = data.list_runs(cfg_name, "e1_main")
        by_dir = {r.run_dir: r for r in runs}
        options = [r.run_dir for r in runs]
        # Re-default when the config changes or the remembered runs vanished.
        if ss.get("_runs_cfg") != cfg_name or not set(ss.get("selected_runs") or ()) <= set(options):
            ss["selected_runs"] = rio.default_selection(runs)
            ss["_runs_cfg"] = cfg_name
        st.multiselect("E1 runs", options, key="selected_runs",
                       format_func=lambda d: by_dir[d].label if d in by_dir else d,
                       disabled=not options, placeholder="no E1 runs" if not options else "choose runs")

        # 4. seed aggregation
        st.toggle("Aggregate seeds", value=True, key="agg_seeds",
                  help="Groups with ≥ 2 selected seeds: line = mean, band = min-max.")

        # 5. reload
        if st.button("Reload results", width="stretch"):
            st.cache_data.clear()
            st.rerun()

        # 6. caption
        n_total = sum(len(data.list_runs(cfg_name, e)) for e in rio.EXPERIMENTS) if cfg_name else 0
        st.caption(f"results dir: `{rio.results_root()}`  \n{n_total} run(s) found"
                   + (f" under `{cfg_name}`" if cfg_name else ""))

        # 1. hardware label (filled last; it depends on the chosen config)
        hw_slot.caption(hardware_caption(data.newest_env(cfg_name)))
    return get_selection()


# ----------------------------------------------------------------------------------------
# Header, empty state, safe chart
# ----------------------------------------------------------------------------------------


def throttle_notes(env: Optional[dict], peer_envs: list[dict] | None = None) -> list[str]:
    """Spec §2.4 power-cap check from env.json gpu_state (numbers taken from the files).

    Flags ``power_limit_w`` below 50% of the max over ``peer_envs`` (run-to-run). The E5
    SM-clock before/after check uses bench rows and lives on the Systems page.
    """
    notes: list[str] = []
    gs = (env or {}).get("gpu_state") or {}
    pl = gs.get("power_limit_w")
    peers = [((e or {}).get("gpu_state") or {}).get("power_limit_w") for e in (peer_envs or [])]
    peers = [p for p in peers + [pl] if isinstance(p, (int, float))]
    if isinstance(pl, (int, float)) and peers and pl < 0.5 * max(peers):
        notes.append(f"power limit {pl:g} W vs {max(peers):g} W max across runs")
    return notes


def page_header(title: str, question: str, experiment: Optional[str] = None,
                run_dir: Optional[str] = None) -> None:
    """Title + one-line question; with ``experiment``/``run_dir`` also the source caption.

    ``run_dir`` given: describe that run. Only ``experiment`` given: describe the newest run
    of that experiment under the selected config (if any).
    """
    st.title(title)
    st.caption(question)
    if experiment is None and run_dir is None:
        return
    info: Optional[rio.RunInfo] = None
    if run_dir is not None:
        info = rio.run_info_for(run_dir)
    elif experiment is not None:
        info = rio.latest(data.list_runs(get_selection().cfg_name, experiment))
    if info is None:
        return
    env = data.read_json(info.path / "env.json")
    when = info.started_at or (env or {}).get("timestamp") or "—"
    st.caption(f"Source: `results/{info.run_dir}` · created {when} · {hardware_caption(env)}")
    peers = [data.read_json(r.path / "env.json") or {} for r in data.list_runs(info.config_name, info.experiment)]
    notes = throttle_notes(env, peers)
    if notes:
        st.warning("GPU may have been power-capped/throttled during this run: " + "; ".join(notes))


def empty_state(experiment: str, cfg_name: Optional[str], what: Optional[str] = None) -> None:
    """Spec §2.5 exact rendering: "No data yet for <what> (<cfg>). Run `<hint>` to produce it."."""
    label = what or rio.EXPERIMENT_TITLE.get(experiment, experiment)
    hint = rio.make_hint(experiment, cfg_name)
    st.info(f"No data yet for {label} ({cfg_name or 'no config'}). Run `{hint}` to produce it.")


def plot(fig: go.Figure, key: Optional[str] = None, **kwargs: Any) -> Any:
    """``st.plotly_chart`` with the house settings (template kept: ``theme=None``)."""
    return st.plotly_chart(fig, theme=None, width="stretch", key=key, **kwargs)


def safe_chart(name: str, build: Callable[[], go.Figure], *, experiment: Optional[str] = None,
               cfg_name: Optional[str] = None, key: Optional[str] = None, **plot_kwargs: Any) -> Any:
    """Build and show one chart; never raise (spec §2.5).

    ``NoData`` -> empty state for that chart only (with the make hint if ``experiment`` is
    given, else the NoData message). Any other exception -> ``st.error`` and continue.
    Returns the ``st.plotly_chart`` return value (selection state) or None.
    """
    try:
        fig = build()
    except rio.NoData as exc:
        if experiment is not None and not str(exc):
            empty_state(experiment, cfg_name, what=name)
        else:
            st.info(str(exc) or f"No data for {name}.")
        return None
    except Exception as exc:  # noqa: BLE001  (pages never raise)
        st.error(f"Could not render {name}: {type(exc).__name__}: {exc}")
        return None
    return plot(fig, key=key, **plot_kwargs)
