"""How it works (DASHBOARD_SPEC §5, architect-owned): what does the router do to my sentence?

Flow: checkpoint selector → text → token × layer grid → (DeepSeek) shared vs routed panel →
next-token top-5 → step-through of one token's path through one block.

Routing comes from ``scripts.route_trace.trace_text`` via ``dashboard.inference.run_sentence``;
every number shown is computed from that forward pass (nothing typed in, spec §7). Code
locations are resolved at runtime from the live model's classes (``inference.code_location``),
falling back to the static ``HOW_STEPS`` strings, then to "(not found: update HOW_STEPS)".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data, figures, inference, layout, theme  # noqa: E402
from dashboard import results_io as rio  # noqa: E402

DEFAULT_TEXT = "Once upon a time, there was a little girl named Lily."

# ----------------------------------------------------------------------------------------
# Steps (spec §5.3). (title, "module:qualname" static fallback, LaTeX per variant family)
# The live code location is resolved from the loaded model in _step_locations(); these
# strings are only used if a live object cannot be found.
# ----------------------------------------------------------------------------------------

_EQ_ROUTER_SWITCH = (r"p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^{E} e^{h(x)_j}},\qquad h(x) = W_r\, x"
                     r"\qquad\text{(Switch Eq. 1)}")
_EQ_ROUTER_DEEPSEEK = (r"s_{i,t} = \operatorname{Softmax}_i\!\left(\mathbf{u}_t^{l\,\top}\mathbf{e}_i^l\right)"
                       r"\qquad\text{(DeepSeekMoE Eq. 11)}")
_EQ_TOPK_SWITCH = (r"\mathcal{T} = \operatorname{argmax}_i\, p_i(x),\qquad "
                   r"C = \left\lfloor \frac{\text{tokens per batch}}{\text{experts}} \times \text{capacity factor}\right\rfloor"
                   r"\qquad\text{(Switch Eq. 3)}")
_EQ_TOPK_GSHARD = (r"\mathcal{T} = \operatorname{TopK}_i\big(p_i(x), k\big),\qquad "
                   r"C = \min\!\left(N, \left\lfloor \frac{k\,N}{E} \times \text{cf} \right\rfloor\right)"
                   r"\qquad\text{(GShard Alg. 1)}")
_EQ_TOPK_DEEPSEEK = (r"g_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} \in \operatorname{TopK}\big(\{s_{j,t}\}, K\big) \\ "
                     r"0, & \text{otherwise}\end{cases}\qquad\text{(DeepSeekMoE Eq. 10)}")
_EQ_EXPERT = r"\mathrm{FFN}_i(\mathbf{u}) = W_{\text{out},i}\;\mathrm{GELU}\!\left(W_{\text{in},i}\,\mathbf{u}\right)"
_EQ_DENSE = r"\mathrm{FFN}(\mathbf{u}) = W_{\text{out}}\;\mathrm{GELU}\!\left(W_{\text{in}}\,\mathbf{u}\right)"
_EQ_SUM_SWITCH = r"y = \sum_{i \in \mathcal{T}} p_i(x)\, E_i(x)\qquad\text{(Switch Eq. 2)}"
_EQ_SUM_DEEPSEEK = (r"\mathbf{h}_t^l = \sum_{i=1}^{K_s}\mathrm{FFN}_i(\mathbf{u}_t^l) + "
                    r"\sum_{i=K_s+1}^{mN} g_{i,t}\,\mathrm{FFN}_i(\mathbf{u}_t^l) + \mathbf{u}_t^l"
                    r"\qquad\text{(DeepSeekMoE Eq. 9)}")
_EQ_NORM = (r"\mathbf{u} = \operatorname{RMSNorm}(\mathbf{h}) = "
            r"\frac{\mathbf{h}}{\sqrt{\tfrac{1}{d}\sum_j h_j^2 + \epsilon}} \odot \mathbf{w}")
_EQ_RESID = r"\mathbf{h} \leftarrow \mathbf{h} + \mathbf{y}"

#: (title, static "module:qualname", {variant family: latex}); family = switch|gshard|deepseek|dense
HOW_STEPS: list[tuple[str, str, dict[str, str]]] = [
    ("Pre-norm", "moe.model:RMSNorm.forward", {"*": _EQ_NORM}),
    ("Router (fp32)", "moe.router:Router.forward",
     {"switch": _EQ_ROUTER_SWITCH, "gshard": _EQ_ROUTER_SWITCH, "deepseek": _EQ_ROUTER_DEEPSEEK}),
    ("Top-k (+ capacity)", "moe.router:Router.forward",
     {"switch": _EQ_TOPK_SWITCH, "gshard": _EQ_TOPK_GSHARD, "deepseek": _EQ_TOPK_DEEPSEEK}),
    ("Experts", "moe.experts:ExpertBank.expert_forward", {"*": _EQ_EXPERT, "dense": _EQ_DENSE}),
    ("Weighted sum (+ shared)", "moe.moe_switch:SwitchMoE.forward",
     {"switch": _EQ_SUM_SWITCH, "gshard": _EQ_SUM_SWITCH, "deepseek": _EQ_SUM_DEEPSEEK}),
    ("Residual", "moe.model:Block.forward", {"*": _EQ_RESID}),
]
_DENSE_STEPS = (1, 4, 6)          # spec §5.3: dense blocks show steps 1, 4 (single FFN) and 6
_STATIC_DENSE_FFN = "moe.ffn:DenseFFN.forward"


def _family(variant: str, rec: dict) -> str:
    if not rec.get("is_moe"):
        return "dense"
    return variant if variant in ("switch", "gshard", "deepseek") else "switch"


def _eq(step: int, family: str) -> str:
    eqs = HOW_STEPS[step - 1][2]
    return eqs.get(family, eqs.get("*", ""))


def _step_locations(step: int, block: Any, family: str) -> list[tuple[str, dict]]:
    """[(role, code_location)] for one step, resolved from the LIVE modules of ``block``."""
    ffn = block.ffn
    m = inference.method_of
    if step == 1:
        objs = [("RMSNorm", m(block.norm2, "forward")), ("called from", m(block, "forward"))]
    elif step == 2:
        objs = [("router", m(getattr(ffn, "router", None), "forward"))]
    elif step == 3:
        objs = [("top-k", m(getattr(ffn, "router", None), "forward"))]
        if hasattr(ffn, "capacity"):
            objs += [("capacity C", m(ffn, "capacity")),
                     ("slot assignment", inference.resolve_object("moe.moe_switch:assign_with_capacity"))]
    elif step == 4:
        if family == "dense":
            objs = [("dense FFN", m(ffn, "forward"))]
        else:
            batched = getattr(ffn, "dispatch", "batched") == "batched"
            disp = [n for n in (("_dispatch_batched", "_routed_batched") if batched else
                                ("_dispatch_loop", "_routed_loop")) if hasattr(type(ffn), n)]
            objs = [(f"dispatch ({ffn.dispatch})", m(ffn, disp[0]) if disp else None),
                    ("experts", m(getattr(ffn, "experts", None), "batched_forward" if batched else "expert_forward"))]
            if getattr(ffn, "shared", None) is not None:
                objs.append(("shared expert", m(ffn.shared, "expert_forward")))
    elif step == 5:
        objs = [("layer forward", m(ffn, "forward"))]
    else:
        objs = [("residual add", m(block, "forward"))]
    out = []
    for role, obj in objs:
        loc = inference.code_location(obj)
        if loc["file"] is None:   # live object missing: fall back to the static spec string
            static = _STATIC_DENSE_FFN if (step == 4 and family == "dense") else HOW_STEPS[step - 1][1]
            loc = inference.code_location(static)
        out.append((role, loc))
    return out


# ----------------------------------------------------------------------------------------
# Cached sentence trace (keyed on checkpoint mtime, device, text, no-drop)
# ----------------------------------------------------------------------------------------


@st.cache_data(show_spinner="Routing your sentence…", max_entries=32)
def _trace(run_dir: str, ckpt_mtime_ns: int, device: str, text: str, no_drop: bool) -> inference.SentenceTrace:
    model, cfg = data.load_model(run_dir, device)
    tok = data.load_tokenizer(cfg)
    return inference.run_sentence(model, tok, text, cfg, no_drop=no_drop)


def _tok_label(trace: inference.SentenceTrace, t: int) -> str:
    return f"{t}: {trace.tokens[t]}"


# ----------------------------------------------------------------------------------------
# Step-through diagram
# ----------------------------------------------------------------------------------------


def _dot(steps: list[int], current: int, family: str, rec: dict, t: int) -> str:
    """Graphviz DOT of the token's path; the current step is the filled node."""
    D = theme.DIAGRAM
    labels = {1: "RMSNorm\\nu = norm(h)", 2: "Router (fp32)\\nsoftmax(W_r u)", 3: "Top-k",
              4: "Experts", 5: "Weighted sum", 6: "+ residual"}
    if rec.get("is_moe"):
        ids = [int(e) for e in rec["topk_idx"][t]]
        kept = [bool(k) for k in rec["kept"][t]]
        labels[3] = "Top-" + str(len(ids)) + "\\n" + ", ".join(
            f"e{e}" + ("" if k else " ✕") for e, k in zip(ids, kept))
        labels[4] = "Experts\\n" + ", ".join(f"FFN_{e}" for e in ids)
        labels[5] = "Σ g·FFN" + (" + shared" if family == "deepseek" else "")
    else:
        labels[4] = "Dense FFN"

    def node(i: int) -> str:
        on = i == current
        return (f'  s{i} [label="{i}. {labels[i]}", style="filled,rounded{",bold" if on else ""}", '
                f'fillcolor="{D["active_fill"] if on else D["node_fill"]}", '
                f'color="{D["active_border"] if on else D["node_border"]}", '
                f'fontcolor="{D["active_font"] if on else D["node_font"]}", penwidth={2.5 if on else 1}];')

    lines = ["digraph G {", "  rankdir=LR; bgcolor=transparent;",
             f'  node [shape=box, fontname="Helvetica", fontsize=11]; edge [color="{D["edge"]}"];',
             f'  h [label="h (token {t})", shape=ellipse, color="{D["node_border"]}", fontcolor="{D["node_font"]}"];',
             f'  out [label="h + y", shape=ellipse, color="{D["node_border"]}", fontcolor="{D["node_font"]}"];']
    lines += [node(i) for i in steps]
    chain = ["h"] + [f"s{i}" for i in steps] + ["out"]
    lines += [f"  {a} -> {b};" for a, b in zip(chain, chain[1:])]
    lines.append(f'  h -> s{steps[-1]} [style=dashed, color="{D["edge_skip"]}", label="skip", '
                 f'fontcolor="{D["node_border"]}", fontsize=9];')
    if family == "deepseek" and 5 in steps:
        lines.append(f'  sh [label="shared expert\\n(always on)", style=rounded, color="{D["node_border"]}", '
                     f'fontcolor="{D["node_font"]}"]; s1 -> sh; sh -> s5;')
    lines.append("}")
    return "\n".join(lines)


def _live_values(step: int, family: str, rec: dict, t: int, trace: inference.SentenceTrace, tpl: str) -> None:
    """Right-hand panel: the actual numbers for token t at this step."""
    f3 = "{:.3f}".format
    if step == 1:
        c1, c2 = st.columns(2)
        c1.metric("‖h‖ (input to norm2)", f3(rec["h_norm"][t]))
        c2.metric("‖u‖ = ‖RMSNorm(h)‖", f3(rec["u_norm"][t]))
        st.caption("Measured with a forward hook on `block.norm2`, registered only for this call.")
    elif step == 2:
        layout.safe_chart("router probabilities", lambda: figures.fig_router_probs(rec, t, template=tpl),
                          key="hiw_router_probs")
        p = rec.get("router_probs")
        if p is not None:
            E = int(rec["n_experts"])
            st.caption(f"Router entropy for this token: {inference.entropy_nats(p[t]):.3f} nats "
                       f"(uniform over E={E}: ln E = {math.log(E):.3f}). Chosen experts are dark bars.")
    elif step == 3:
        k = rec["topk_idx"].shape[1]
        st.dataframe(pd.DataFrame({
            "rank": list(range(1, k + 1)),
            "expert": [int(e) for e in rec["topk_idx"][t]],
            "gate g": [f3(g) for g in rec["gates"][t]],
            "kept": ["yes" if bool(x) else "dropped (over capacity)" for x in rec["kept"][t]],
        }), hide_index=True, width="stretch")
        if rec.get("capacity") is not None:
            st.caption(f"Capacity C = {rec['capacity']} slots per expert for N = {trace.n_tokens} tokens, "
                       f"E = {rec['n_experts']}, k = {k}, eval cf = {inference.fmt_cf(rec['capacity_factor'])} "
                       "(`layer_api.expert_capacity`). Slots are filled in token order, first choices first.")
        else:
            st.caption("DeepSeekMoE has no capacity limit: nothing is dropped.")
    elif step == 4:
        if family == "dense":
            st.metric("‖FFN(u)‖", f3(rec["y_norm"][t]))
            return
        rn = rec.get("routed_expert_out_norm")
        ids, gates = rec["topk_idx"][t], rec["gates"][t]
        rows = {"expert": [int(e) for e in ids], "gate g": [f3(g) for g in gates]}
        if rn is not None:
            rows["‖g·FFN_i(u)‖"] = [f3(v) for v in rn[t]]
            rows["‖FFN_i(u)‖ = ‖g·FFN‖ / g"] = [f3(v / g) if g > 0 and v > 0 else "— (dropped)"
                                               for v, g in zip(rn[t], gates)]
        else:
            st.info("Per-expert norms need MoEAux.extra['routed_expert_out_norm'] (spec R3).")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    elif step == 5:
        all_dropped = not bool(np.any(rec["kept"][t]))
        if all_dropped:
            st.warning("Every choice of this token was dropped: y = 0, the token passes on the residual only.")
        cols = st.columns(3 if family == "deepseek" else 1)
        cols[0].metric("‖y‖ (layer output)", f3(rec["y_norm"][t]))
        if family == "deepseek" and rec.get("shared_out_norm") is not None:
            s, r = float(rec["shared_out_norm"][t]), float(rec["routed_out_norm"][t])
            cols[1].metric("‖shared sum‖", f3(s))
            cols[2].metric("‖routed sum‖", f3(r))
            st.caption("In this code the residual term of Eq. 9 is added by `Block.forward` (step 6) "
                       "to the pre-norm stream h, not to u.")
    else:
        h, y, ho = float(rec["h_norm"][t]), float(rec["y_norm"][t]), float(rec["hout_norm"][t])
        c1, c2, c3 = st.columns(3)
        c1.metric("‖h‖ before", f3(h))
        c2.metric("‖y‖ / ‖h‖", f3(y / h) if h > 0 else "—")
        c3.metric("‖h + y‖ after", f3(ho))


def render_token_path(trace: inference.SentenceTrace, model_blocks: Any, layer: int, t: int, tpl: str) -> None:
    """Spec §5.3 step-through for (token t, block ``layer``)."""
    rec = trace.layers[layer]
    family = _family(trace.variant, rec)
    steps = list(range(1, 7)) if rec.get("is_moe") else list(_DENSE_STEPS)
    titles = {i: f"{i}. {HOW_STEPS[i - 1][0]}" for i in steps}
    if family == "dense":
        titles[4] = "4. Dense FFN"
    if st.session_state.get("hiw_step") not in steps:
        st.session_state["hiw_step"] = steps[0]
    step = st.segmented_control("Step", steps, format_func=lambda i: titles[i], key="hiw_step",
                                selection_mode="single")
    if step is None:
        step = steps[0]
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.graphviz_chart(_dot(steps, step, family, rec, t), width="stretch")
    with right:
        st.markdown(f"**{titles[step]}** — token `{trace.tokens[t]}` (position {t}), block {layer}")
        eq = _eq(step, family)
        if eq:
            st.latex(eq)
        for role, loc in _step_locations(step, model_blocks[layer], family):
            st.markdown(f"<small>{role}:</small> `{loc['text']}`", unsafe_allow_html=True)
        _live_values(step, family, rec, t, trace, tpl)


# ----------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------


def _checkpoint_options(cfg_name: Optional[str]) -> list[rio.RunInfo]:
    runs = [r for e in rio.TRAINING_EXPERIMENTS for r in data.list_runs(cfg_name, e)
            if r.complete and r.has_checkpoint]
    return sorted(runs, key=lambda r: (theme.variant_sort_key(r.variant), r.run_id))


def _default_index(runs: list[rio.RunInfo]) -> int:
    for v in ("deepseek", "switch"):     # spec §5.1: latest deepseek, else latest switch, else first
        r = rio.latest(runs, where=lambda r, v=v: r.variant == v)
        if r is not None:
            return runs.index(r)
    return 0


def main() -> None:
    sel = layout.get_selection()
    tpl = layout.current_template()
    layout.page_header("How it works", "What does the router actually do to my sentence?")

    runs = _checkpoint_options(sel.cfg_name)
    if not runs:
        layout.empty_state("e1_main", sel.cfg_name, what="trained checkpoints")
        return

    # 1. checkpoint + device
    import torch
    by_dir = {r.run_dir: r for r in runs}
    c1, c2 = st.columns([4, 1])
    run_dir = c1.selectbox("Checkpoint", [r.run_dir for r in runs], index=_default_index(runs),
                           format_func=lambda d: by_dir[d].label + f" · {by_dir[d].experiment}",
                           key="hiw_ckpt")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    device = c2.radio("Device", devices, horizontal=True, key="hiw_device")
    info = by_dir[run_dir]

    # 2. text (committed with the Run button; the default sentence runs on first load)
    text_in = st.text_area("Your text", value=DEFAULT_TEXT, key="hiw_text_area", height=80)
    if st.button("Run", type="primary") or "hiw_text" not in st.session_state:
        st.session_state["hiw_text"] = text_in
    text = st.session_state["hiw_text"]
    if not text.strip():
        st.info("Type some text and press Run.")
        return

    # 3. no-drop option (capacity variants only)
    no_drop = False
    if info.variant in ("switch", "gshard"):
        no_drop = st.checkbox("no-drop eval (capacity = N)", value=False, key="hiw_nodrop")
        st.caption("With one short sentence, capacity = floor(k·N/E·cf) is small, so drops are expected; "
                   "this is real layer behaviour for this batch size.")

    ckpt_sig = rio.file_signature(info.path / "checkpoint.pt")
    try:
        trace = _trace(str(info.path), ckpt_sig[1] if ckpt_sig else 0, device, text, no_drop)
        model, cfg = data.load_model(str(info.path), device)
    except Exception as exc:  # noqa: BLE001  (pages never raise)
        st.error(f"Could not run this checkpoint: {type(exc).__name__}: {exc}")
        return
    if trace.truncated:
        st.caption(f"Truncated to the model's seq_len = {cfg['model']['seq_len']} tokens "
                   f"(your text has {trace.n_tokens_in_text}).")
    st.caption(f"{trace.n_tokens} tokens · variant `{trace.variant}` · "
               f"{theme.variant_label(trace.variant, cfg)} · source `results/{info.run_dir}`")

    # 4. token × layer grid
    st.subheader("Which expert did each token go to?")
    mode = st.radio("Colour cells by", ["expert id", "gate value"], horizontal=True, key="hiw_mode")
    st.caption("Cell text = top-1 routed expert id (✕ = that assignment was dropped over capacity). "
               "Hover for the full top-k with gates. Click a cell to step through it below.")
    event = layout.safe_chart("token × layer grid",
                              lambda: figures.fig_token_layer_grid(trace, mode=mode, template=tpl),
                              key="hiw_grid", on_select="rerun", selection_mode="points")
    T, L = trace.n_tokens, len(trace.layers)
    try:
        pts = event.selection.points if event is not None else []
    except AttributeError:
        pts = []
    if pts:
        click = (int(round(pts[0]["x"])), int(round(pts[0]["y"])))
        if click != st.session_state.get("_hiw_last_click") and 0 <= click[0] < T and 0 <= click[1] < L:
            st.session_state["_hiw_last_click"] = click
            st.session_state["hiw_t"], st.session_state["hiw_layer"] = click

    # 5. DeepSeek shared vs routed
    moe_layers = [i for i, r in enumerate(trace.layers) if r.get("is_moe")]
    if trace.variant == "deepseek" and moe_layers:
        st.subheader("Shared vs routed contribution (DeepSeekMoE)")
        lay = st.selectbox("Layer", moe_layers, key="hiw_contrib_layer")
        layout.safe_chart("shared vs routed", lambda: figures.fig_shared_vs_routed(trace, lay, template=tpl),
                          key="hiw_contrib")

    # 6. next-token predictions
    st.subheader("Next-token predictions")
    if st.session_state.get("hiw_pred_pos") not in range(T):
        st.session_state["hiw_pred_pos"] = T - 1
    pos = st.selectbox("After position", list(range(T)), key="hiw_pred_pos",
                       format_func=lambda t: _tok_label(trace, t))
    layout.safe_chart("top-5 predictions", lambda: figures.fig_top5(trace, pos, template=tpl), key="hiw_top5")
    if pos + 1 < T:
        st.caption(f"Actual next token in your text: `{trace.tokens[pos + 1]}`.")

    # 7. step-through
    st.subheader("Step through one token")
    if st.session_state.get("hiw_t") not in range(T):
        st.session_state["hiw_t"] = 0
    if st.session_state.get("hiw_layer") not in range(L):
        st.session_state["hiw_layer"] = moe_layers[0] if moe_layers else 0
    c1, c2 = st.columns(2)
    t = c1.selectbox("Token", list(range(T)), key="hiw_t", format_func=lambda t: _tok_label(trace, t))
    layer = c2.selectbox("Layer (block)", list(range(L)), key="hiw_layer",
                         format_func=lambda i: f"{i}" + ("" if trace.layers[i].get("is_moe") else " (dense)"))
    render_token_path(trace, model.blocks, layer, t, tpl)


main()
