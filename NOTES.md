# NOTES.md: study guide (paper idea → equation → code)

This is a reading companion for the code in `moe/`. Each concept gets the same four parts:

1. **Plain language**: what the idea is and why it exists.
2. **Paper equation**: written out with every symbol defined. Equation numbers refer to
   - **Switch**: Fedus, Zoph, Shazeer, *Switch Transformers*, arXiv **2101.03961**
   - **DeepSeekMoE**: Dai et al., *DeepSeekMoE*, arXiv **2401.06066**
3. **Where it lives**: `file::Class.method` or `file::function`, checked against the code.
   Line numbers are deliberately omitted: several agents were editing `moe/` in parallel
   when this was written (E5 profiling annotations, review-gate comments), so line numbers
   would go stale immediately. Instead, each reference names the function and, where it
   helps, quotes the exact line of code so you can find it with search.
4. **Gotchas / design choices**: what this repo decided, why, and **where it differs from the paper**.

Sections 1–14 contain no experimental results. Every measured number belongs in
`RESULTS.md`, which `make_report.py` generates from `results/`; the only numbers in those
sections are config dimensions (d = 384, d_ff = 1536, ...) and quantities derived from them by
algebra. The two sections at the end ("What surprised me / what didn't replicate" and
"Resume-ready facts") quote results, and every number there cites the `RESULTS.md` section
or the `results/main/...` file it comes from.

**Notation used throughout.** N = tokens in the batch at one layer (B·T; the code flattens
`[B, T, d]` to `[N, d]`). E = number of **routed** experts. k = experts chosen per token.
d = `d_model`. w = expert width. C = expert capacity. Note that Switch uses **T** for "tokens
in the batch" and **N** for "number of experts", which is the opposite of the code. Each
section says which convention it uses.

---

## Map of the code (one MoE layer, forward)

```
model.py::Block.forward              h = h + attn(norm1(h));  y, aux = ffn(norm2(h));  h = h + y
  └─ layer_api.py::build_ffn         picks DenseFFN / SwitchMoE / GShardMoE / DeepSeekMoE
       ├─ router.py::Router.forward          fp32: jitter → logits → softmax → top-k → gates
       ├─ moe_switch.py::assign_with_capacity   (Switch/GShard) cumsum position, keep = pos < C
       ├─ dispatch: _dispatch_loop | _dispatch_batched      (DeepSeek: _routed_loop | _routed_batched)
       │    └─ experts.py::ExpertBank.expert_forward | .batched_forward
       ├─ router.py::switch_aux_loss | deepseek_expert_aux_loss | deepseek_device_aux_loss
       └─ returns (y, layer_api.py::MoEAux)   → metrics.py::RoutingAccumulator for logs
model.py::MoETransformer.forward     aux_loss = sum over layers; train.py does loss = ce + aux
```

The norm and the residual add are in `model.py::Block.forward`, **not** inside the
MoE layers. So "a dropped token passes through the residual" means the layer returns
`y = 0` for that row, and `h + 0 = h`.

---

## 1. Top-k routing

**Plain language.** A router is a tiny linear classifier that looks at each token and scores
every expert. The token is sent only to its k highest-scoring experts, and each expert's
output is weighted by its score. Think of a hospital triage desk: each patient sees one
specialist (Switch, k = 1), two (GShard, k = 2), or a GP plus three specialists (DeepSeekMoE,
1 shared + top-3). Compute per token stays fixed while total parameters grow with E.

**Paper equations.**

Switch Eq. (1), router probabilities:
$$p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^{N} e^{h(x)_j}}, \qquad h(x) = W_r \cdot x$$
- x: the token representation (input to the FFN position)
- W_r: router weights; h(x): the logits, one per expert
- N (Switch notation): number of experts; p_i(x): probability of expert i

Switch Eq. (2), output of the layer over the selected set 𝒯 (top-k indices):
$$y = \sum_{i \in \mathcal{T}} p_i(x)\, E_i(x)$$
- E_i(x): output of expert i. Switch uses |𝒯| = 1.

DeepSeekMoE Eqs. (3)–(5) are the same idea in DeepSeek notation (u_t = token t's input,
e_i = expert i's "centroid", s_{i,t} = softmax score, g_{i,t} = gate):
$$h_t = \sum_{i=1}^{N} g_{i,t}\,\mathrm{FFN}_i(u_t) + u_t,\quad
g_{i,t} = \begin{cases} s_{i,t} & s_{i,t} \in \mathrm{Topk}(\{s_{j,t}\}, K)\\ 0 & \text{otherwise}\end{cases},\quad
s_{i,t} = \mathrm{Softmax}_i(u_t^\top e_i)$$

**Where it lives.**
- `moe/router.py::Router`. In `moe/router.py::Router.forward`: `logits = xf @ self.weight.float()`,
  `probs = torch.softmax(logits, dim=-1)` (Switch Eq. 1), `topk_idx = sel_scores.topk(k, dim=-1).indices`,
  then `gates = ... if self.renormalize else topk_prob`.
- `moe/router.py::RouterOutput`: the fields `probs`, `topk_idx`, `topk_prob`, `gates`, `entropy`.
- The weighted sum (Switch Eq. 2) happens in the dispatch paths:
  `moe/moe_switch.py::CapacityMoE._dispatch_loop` (the line `y = y.index_add(0, tok, g * out)`)
  and `moe/moe_switch.py::CapacityMoE._dispatch_batched`; for DeepSeek, `moe/moe_deepseek.py::DeepSeekMoE._routed_loop` / `_routed_batched`.
- k per variant comes from `moe/layer_api.py::ffn_spec`.

**Gotchas / design choices.**
- **Gate = raw softmax probability, not renormalized** over the k chosen experts. That is
  what both papers' equations say (Switch Eq. 2, DeepSeekMoE Eq. 10). Renormalization is a
  config flag `moe.renormalize_gates` (default `false`), handled in the `gates = ...` line of `Router.forward`. With
  k = 1 the raw gate is < 1, so the expert output is always scaled down. That is how the
  router gets a gradient: the gate is the only differentiable path from the loss to W_r
  besides the aux loss.
- `probs` is never masked: the aux losses see the full distribution over all E experts.
- **No bias** in the router (`W_r` is `[d, E]`), matching "no biases anywhere" (PLAN §2).
- **Jitter noise** (Switch App. C): in training only, the router *input* is multiplied
  elementwise by U(1−ε, 1+ε), ε = 0.01 for Switch (config `moe.switch.jitter_eps`; 0 for GShard
  and DeepSeek by default). Code: the `if self.training and self.jitter_eps > 0:` block in `Router.forward`. It perturbs the input, not the logits.
- **E3 ablation hook** (`n_disable_top` in `Router.forward`, the `sel_scores = probs.scatter(-1, top_dis, float("-inf"))`
  branch): mask the n highest-probability experts *for selection only*, then take top-k of
  the rest, and keep the original softmax probability as gate. This implements DeepSeekMoE
  Sec. 4.5 / Fig. 4. The paper's text says nothing about the gates of the surviving
  experts; the repo keeps Eq. 10 literally (g = unmasked s) and says so in the docstring. It is set at eval time
  through `moe/layer_api.py::set_eval_overrides`, which raises if no layer supports
  the requested switch, so an ablation can never silently do nothing.
- **GShard differs from the GShard paper** (`moe/moe_gshard.py`, module docstring): GShard's
  Algorithm 1 normalizes g1, g2 by g1 + g2 and dispatches the 2nd expert *stochastically*.
  This repo keeps the raw-prob gate and routes the 2nd choice deterministically. This is a
  documented simplification. GShard here is the user-approved 16 experts × d_ff/2, top-2.
- `router_entropy` (`torch.special.entr(probs).sum(-1).mean()` in `Router.forward`) is in **nats**; its maximum is ln E. `metrics.py` logs both.

---

## 2. Capacity factor

**Plain language.** On real hardware every expert gets a fixed-size buffer, because static
shapes are needed for compilation (TPU/XLA in the Switch paper) and for predictable memory.
Capacity is "how many tokens each expert's buffer holds". The capacity factor (cf) is
slack: cf = 1.0 means "exactly the fair share", cf = 1.25 means 25% headroom for imbalance.
Bigger cf means fewer dropped tokens, but more padding (wasted FLOPs and memory) and more
communication.

**Paper equation.** Switch Eq. (3):
$$\text{expert capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}$$

**What the code computes** (generalized to top-k):
$$C = \min\!\Big(N,\ \Big\lfloor \frac{k \cdot N}{E} \cdot cf \Big\rfloor\Big)$$
- N: tokens in the batch at this layer; E: routed experts; k: choices per token; cf: capacity factor.

**Where it lives.**
- `moe/layer_api.py::expert_capacity`.
- Called by `moe/moe_switch.py::CapacityMoE.capacity`, which picks the training or eval cf through `moe/moe_switch.py::CapacityMoE.active_capacity_factor`.
- Also used analytically by `moe/flops.py::moe_layer_cost` to cost the padded buffer.
- Tests: `tests/test_switch.py::test_capacity_formula`, `tests/test_gshard.py::test_capacity_is_top_k_aware`.

**Gotchas / design choices.**
- **Floor and clamp.** The paper writes Eq. 3 without rounding. The code uses `floor` (the
  Mesh-TF reference casts to int) and clamps to N, because one expert can receive at most one
  assignment per token.
- **The k factor for top-k.** With k = 2 there are 2N assignments, so the fair share per
  expert is 2N/E. Without the k factor, GShard would drop about half its assignments even
  with perfect balance. GShard's Algorithm 1 uses 2N/E. For k = 1 the formula reduces exactly
  to Switch Eq. 3.
- **Train vs eval cf.** `eval_capacity_factor` (config, default `null` = same as training)
  applies only in eval mode. `layer_api.NO_DROP` (= ∞) gives C = N. E3 uses it so that loss
  changes reflect expert redundancy, not capacity effects (PLAN §7).
- **DeepSeekMoE has no capacity at all** (Section 6).
- The capacity factor is the main knob of **E2** (`scripts/sweep_capacity.py`: cf ∈ {0.75, 1.0, 1.25, 2.0}).

---

## 3. Token dropping

**Plain language.** If more tokens pick an expert than its buffer holds, the overflow tokens
are skipped: the expert does not process them, the MoE layer outputs 0 for them, and they
continue to the next layer only through the residual connection. Who gets a seat is first
come, first served, in flattened (batch, position) order.

**Paper mechanism.** Switch Sec. 2.2 ("overflow tokens ... passed directly to the next layer
through the residual connection"), and the position-in-expert cumsum of **Switch Fig. 15**
(the Mesh-TF router code):
$$\text{pos}(a) = \Big(\sum_{a' \le a} \mathbb{1}[\text{expert}(a') = \text{expert}(a)]\Big) - 1, \qquad \text{keep}(a) = \text{pos}(a) < C$$
- a: one (token, choice) assignment, in priority order; expert(a): the expert it chose.

**Where it lives.**
- `moe/moe_switch.py::assign_with_capacity`: cumsum over the one-hot
  (`pos = ((onehot.cumsum(dim=0) - 1) * onehot).sum(dim=-1)`), then `keep = pos < capacity`.
- `moe/moe_switch.py::CapacityAssignment` documents the priority order.
- Dropped rows produce 0 in both paths: the loop path selects only kept assignments
  (`(asg.expert == e) & asg.keep`); the batched path sends them to a dump row
  (`dest = torch.where(asg.keep, ..., EC)`) and multiplies by a zero gate (`gate_flat * asg.keep`).
- Stats: `drop_fraction`, `expert_counts` (pre-drop) and `kept_counts` in `MoEAux`, built in `moe/moe_switch.py::CapacityMoE.forward`.
- Tests: `tests/test_switch.py::test_capacity_dropping_all_to_one_expert` (everyone picks one
  expert, so exactly C tokens are processed and the rest output 0),
  `tests/test_gshard.py::test_first_choices_fill_capacity_before_second_choices`.

**Gotchas / design choices.**
- **Choice-major (slot-major) priority for top-k.** Assignments are flattened as
  `a = j·N + t` (choice j, token t; `expert = topk_idx.t().reshape(-1)` in `assign_with_capacity`). So **all first choices
  claim buffer slots before any second choice**, which is GShard's priority rule. The
  alternative, token-major (t·k + j), would let token 0's 2nd choice take a slot before
  token 900's 1st choice. With k = 1 both orders are plain token order, as in Fig. 15.
  If a GShard token's 2nd choice is dropped it still gets p_1·E_1(x); if both are dropped, y = 0.
- **Drop fraction** = dropped (token, choice) assignments / all assignments (k·N), not dropped tokens / N.
- **Not causal.** Whether token t is dropped depends on every token ahead of it in the
  flattened batch, including tokens from other sequences. Later positions and later
  sequences in the batch are dropped more often. The paper has the same property. It means
  eval loss depends on batch composition whenever drops happen.
- **The dump row** (`CapacityMoE._dispatch_batched`): the buffer has `E·C + 1` rows. Every dropped
  assignment's destination is the extra row `E·C`, so the scatter needs no boolean
  compaction, which would create a data-dependent shape and a host sync. The dump row is
  discarded before the expert GEMMs; on the way back the output gets a zero row appended at
  the same index, so dropped assignments gather 0. The gate is also zeroed for them.
- **Aux loss uses pre-drop routing.** See Section 4.

---

## 4. Load-balancing loss (Switch)

**Plain language.** Without pressure, the router tends to collapse: a few experts get good
early, receive more tokens, get better still, and the rest starve. The aux loss penalizes
the case where the same experts get both many tokens and high probability. The token count
is not differentiable (it comes from an argmax), so the gradient flows through the mean
probability, which pushes probability mass away from overloaded experts.

**Paper equations.** Switch Eqs. (4)–(6), in Switch notation (N = number of experts,
𝔅 = batch with T tokens):
$$\text{loss} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i \qquad (4)$$
$$f_i = \frac{1}{T} \sum_{x \in \mathfrak{B}} \mathbb{1}\{\operatorname{argmax} p(x) = i\} \qquad (5)$$
$$P_i = \frac{1}{T} \sum_{x \in \mathfrak{B}} p_i(x) \qquad (6)$$
- f_i: fraction of tokens dispatched to expert i (not differentiable)
- P_i: fraction of router probability mass on expert i (differentiable)
- α: coefficient, 10⁻² (config `moe.switch.aux_alpha`)
- The factor N keeps the loss constant as the number of experts changes: under perfectly
  uniform routing f_i = P_i = 1/N, so loss = α·N·N·(1/N²) = α. The paper says the loss is
  "minimized under a uniform distribution".

**Where it lives.**
- `moe/router.py::switch_aux_loss`: `f = torch.bincount(top1_idx, minlength=E).float() / T` (Eq. 5),
  `P = probs.mean(dim=0)` (Eq. 6), `return alpha * E * torch.sum(f * P)` (Eq. 4). In code, E plays the role of the paper's N.
- Called from `moe/moe_switch.py::CapacityMoE.aux_loss` with `r.topk_idx[:, 0]`.
- Summed over layers in `moe/model.py::MoETransformer.forward` (the `aux_total = torch.stack(...).sum()` line).
- Tests: `tests/test_aux_loss.py::test_switch_aux_uniform_is_alpha`,
  `test_switch_aux_collapse_is_alpha_E` (full collapse gives α·E),
  `test_switch_aux_gradient_flows_through_P_only`, and `tests/test_switch.py::test_aux_uses_pre_drop_argmax`.

**Gotchas / design choices.**
- **f_i is computed before dropping.** It uses the router's argmax, not what the experts
  actually processed. If f_i were computed after dropping, an overloaded expert would look
  capped at C/N and the loss would underestimate the imbalance it is supposed to fix.
- **GShard uses top-1 f_i** (first choice only), as in GShard's `c_e/S`. The DeepSeek-style
  alternative would count all k choices.
- **Summed, not averaged, over layers**: `total = CE + Σ_layers aux_l`, and each `aux_l`
  already includes its α (PLAN §7). Logs keep CE and aux separate so loss curves compare like with like.
- Computed in **fp32** (Section 5), over all N = B·T tokens of the step at that layer.
- The aux loss is computed in eval mode too (so it can be logged); `train.py` adds it to CE only in training.

---

## 5. Selective precision (fp32 router)

**Plain language.** bf16 has only 8 bits of mantissa. Softmax involves exponentials, and
small logit errors become large probability errors. A wrong argmax sends a token to the
wrong expert, which is a discrete error that does not average out. Switch found that
training in bf16 everywhere was unstable, and fp32 everywhere was slow. The fix is to run
just the router in fp32 (it is tiny) and keep the experts in bf16.

**Paper.** Switch Sec. 2.4, "Selective precision with large sparse models" (and Table 2).
The router input is cast to fp32, the router computation runs in fp32, and the dispatch and
combine tensors are cast back to bf16. There is no equation; it is a numerics rule.

**Where it lives.**
- `moe/router.py::Router.forward`: the whole body runs under
  `with torch.autocast(device_type=x.device.type, enabled=False):`, starting with `xf = x.float()`.
  The weight is stored as fp32 (`moe/router.py::Router.__init__`: `torch.empty(d_model, n_experts, dtype=torch.float32)`).
- The gate is cast to the activation dtype only at the combine multiply:
  `moe/moe_switch.py::CapacityMoE._dispatch_loop` (`g = gate_flat[sel].to(cdt)[:, None]`),
  `moe/moe_switch.py::CapacityMoE._dispatch_batched` (`g = (gate_flat * asg.keep).to(cdt)[:, None]`),
  and the same pattern in `moe/moe_deepseek.py::DeepSeekMoE._routed_loop` / `_routed_batched`.
- `moe/moe_switch.py::compute_dtype` finds the dtype the expert GEMMs will produce under autocast.
- Tests: `tests/test_fp32_router.py::test_router_fp32_under_autocast`,
  `tests/test_fp32_router.py::test_router_matches_fp32_reference_under_autocast`.

**Gotchas / design choices.**
- **Where the fp32 boundary is.** Everything from the input cast through jitter, logits,
  softmax, top-k, entropy and **the aux losses** is fp32. That is slightly more than the
  paper states: the paper does not mention where jitter or the aux loss run. Indices are
  int64. Only gate values cross back to bf16.
- Simply calling `.float()` would not be enough: under autocast, `matmul` re-casts its
  inputs to bf16. That is why autocast is explicitly *disabled* in the router region.
- In the sort-and-segment implementation there is no float "dispatch tensor" to cast (the
  dispatch is integer indices), so "cast the combine tensor" becomes "cast the gate".
- The router is also excluded from weight decay (`moe/layer_api.py::mark_no_weight_decay`,
  called in `Router.__init__`; test `tests/test_param_matching.py::test_router_excluded_from_weight_decay`).
  That is a repo choice (PLAN §4), not a Switch-paper rule.

---

## 6. Init scale

**Plain language.** Large sparse models were more fragile at the start of training. Switch
found that shrinking the standard Transformer init by 10× (smaller initial weights) made
training more stable, with lower variance across runs. It is one line of code but it
affects every weight matrix.

**Paper.** Switch Sec. 2.4, "Smaller parameter initialization for stability" (Table 3):
truncated normal with mean 0 and
$$\sigma = \sqrt{s / n}$$
- s: scale hyperparameter, reduced from the default 1.0 to **0.1**
- n: number of input units of the weight tensor (fan-in)
- values more than 2σ from the mean are redrawn (truncation at ±2σ)

DeepSeekMoE Sec. 4.1 instead initializes all parameters with std 0.006 (plain normal).

**Where it lives.**
- `moe/layer_api.py::init_weight_`; `moe/layer_api.py::DEEPSEEK_INIT_STD` for `init: deepseek`.
- Config: `model.init: switch`, `model.init_scale: 0.1` (`configs/*.yaml`).
- Applied to the expert stacks (`moe/experts.py::ExpertBank.__init__`), the router
  (`moe/router.py::Router.__init__`), attention and the embedding (`moe/model.py`).

**Gotchas / design choices.**
- **fan_in for stacked experts is d (or w), not E·d.** `W_in[E, d, w]` is E separate `[d, w]`
  matrices, so each gets σ = √(s/d); `W_out[E, w, d]` uses fan_in = w (`moe/experts.py::ExpertBank.__init__`: `fan_in=d_model` for `W_in`, `fan_in=width` for `W_out`).
  Using the tensor's flattened fan-in would make the init depend on the number of experts.
- The **realized std is about 0.88σ** because of the truncation. σ is the std of the normal
  *before* truncation, as in TF/Mesh-TF `truncated_normal` and `torch.nn.init.trunc_normal_`.
- Every variant (including dense) uses the same init, so the comparison is fair (MASTER §4.7).
- The `deepseek` init applies no truncation, because the paper does not mention any.

---

## 7. Fine-grained expert segmentation (DeepSeekMoE)

**Plain language.** Split each big expert into m smaller ones (width d_ff/m) and activate m
times as many, so compute per token stays the same. Why bother? The number of possible
expert *combinations* grows enormously. With 8 experts choosing 1 there are 8 options; with
32 small experts choosing 4 there are C(32, 4) = 35,960 (in this repo 1 of the 32 is shared,
so the routed combinations are C(31, 3) = 4,495, still far more than 8). That lets experts specialize more
narrowly and combine flexibly, instead of each big expert having to cover many unrelated
kinds of knowledge.

**Paper equations.** DeepSeekMoE Eqs. (6)–(8) (fine-grained only, before shared isolation):
$$h_t = \sum_{i=1}^{mN} g_{i,t}\,\mathrm{FFN}_i(u_t) + u_t \qquad (6)$$
$$g_{i,t} = \begin{cases} s_{i,t} & s_{i,t} \in \mathrm{Topk}(\{s_{j,t} \mid 1 \le j \le mN\},\ mK)\\ 0 & \text{otherwise}\end{cases} \qquad (7)$$
$$s_{i,t} = \mathrm{Softmax}_i(u_t^\top e_i) \qquad (8)$$
- N, K: experts and active experts of the coarse MoE; m: segmentation factor
- each fine-grained FFN_i has width d_ff/m (the paper reduces the FFN intermediate hidden dimension to 1/m)

This repo: m = 4, so 32 experts of width d_ff/4 (1 of which is shared, Section 8) and 4 active per token.

**Where it lives.**
- `moe/layer_api.py::ffn_spec`: width = `d_ff // m` for deepseek, and the assert that
  activated width `(spec.n_shared + spec.top_k) * spec.width == d_ff`.
- `moe/moe_deepseek.py::DeepSeekMoE`: routed bank `self.experts = ExpertBank(self.n_routed, d_model, self.width, model_cfg)` in `__init__`.
- `moe/experts.py::ExpertBank`: the stacked `W_in[E, d, w]`, `W_out[E, w, d]` used by all MoE variants.
- Test: `tests/test_deepseek.py::test_activated_width_assertion`.

**Gotchas / design choices.**
- **No token dropping in DeepSeekMoE.** MASTER §4.6 specifies no dropping, because the
  paper's small-scale experiments keep all experts on one device (so there is no
  communication buffer to cap). This is a deliberate difference from Switch. The DeepSeek layer has no capacity: `kept_counts = counts`,
  `drop_fraction=torch.zeros((), ...)` (in `DeepSeekMoE.forward`). Test: `tests/test_deepseek.py::test_no_dropping_even_when_unbalanced`.
- **31 routed experts is not a power of 2**, and 3N/31 is never an integer, so even
  perfectly balanced routing makes the batched buffer pad by up to 1 row per expert.
- **Fine-grained experts are worse for the hardware** (Section 13): each GEMM is narrower
  (w = 384 instead of 1536) and sees fewer tokens (about 3N/31 instead of N/8), so its
  arithmetic intensity is lower. That is a central systems trade-off this repo measures in E5.

---

## 8. Shared experts (shared expert isolation)

**Plain language.** Some knowledge is needed by almost every token (common syntax, frequent
words). If every routed expert has to relearn it, parameters are wasted on redundant copies.
DeepSeekMoE isolates K_s experts that **every** token always uses, with no gate, so the routed
experts are free to specialize. Analogy: a general practitioner everyone sees, plus
specialists by referral.

**Paper equations.** DeepSeekMoE Eqs. (9)–(11):
$$h_t = \sum_{i=1}^{K_s} \mathrm{FFN}_i(u_t) + \sum_{i=K_s+1}^{mN} g_{i,t}\,\mathrm{FFN}_i(u_t) + u_t \qquad (9)$$
$$g_{i,t} = \begin{cases} s_{i,t} & s_{i,t} \in \mathrm{Topk}(\{s_{j,t} \mid K_s+1 \le j \le mN\},\ mK - K_s)\\ 0 & \text{otherwise}\end{cases} \qquad (10)$$
$$s_{i,t} = \mathrm{Softmax}_i(u_t^\top e_i) \qquad (11)$$
- K_s: number of shared experts (here 1); the routed experts are indices K_s+1..mN (here 31 of them)
- mK − K_s: routed experts chosen per token (here 4 − 1 = 3), so total active stays mK

**Where it lives.**
- `moe/moe_deepseek.py::DeepSeekMoE.forward`. Shared sum with no gate: the
  `for s in range(self.n_shared)` loop (Eq. 9, first sum).
  Routed sum: `moe/moe_deepseek.py::DeepSeekMoE._routed_loop` or `moe/moe_deepseek.py::DeepSeekMoE._routed_batched` (Eq. 9, second sum).
- Shared bank: `self.shared = ExpertBank(self.n_shared, ...)` in `DeepSeekMoE.__init__`.
- The residual `+ u_t` is added by `moe/model.py::Block.forward`, not here.
- Ablations (eval-time, E4): `disable_shared`, `k_routed_override` (`moe/moe_deepseek.py::DeepSeekMoE.k_active`), set via `moe/layer_api.py::set_eval_overrides`; driven by `scripts/ablate_experts.py`.
- Tests: `tests/test_deepseek.py::test_disable_shared_changes_output`, `test_zero_routed_gates_gives_shared_output`,
  `test_gates_are_raw_softmax_over_routed`, `test_k_routed_override`.

**Gotchas / design choices.**
- **The softmax is over the 31 routed experts only.** The shared expert has no router
  column. In Eq. 11 the paper does not spell out the index range of the softmax for the
  shared-expert case; normalizing over routed experts only is this repo's reading (and it is
  what `router_probs` and the aux loss use). If you compare gate magnitudes with another
  implementation, check this first.
- **The shared expert counts toward both totals**: total = 32 × d_ff/4 = 8·d_ff (matches
  Switch), activated = (1 + 3) × d_ff/4 = d_ff (matches dense).
- The shared expert is stored as an `ExpertBank` with 1 expert and applied with
  `expert_forward`, so it is the same GELU FFN math as everything else.
- **E4 is an eval-time ablation, not retraining.** Turning off the shared expert in a model
  trained with it measures *dependence*, not what a model trained without it would do. The
  paper's Fig. 5 compares a k sweep; the repo labels every such row "eval-time change, not retrained".

---

## 9. Expert-level and device-level balance losses (DeepSeekMoE)

**Plain language.** Same goal as Switch's loss (Section 4), with two differences.
(1) The *expert-level* loss counts all K' choices per token and normalizes f_i so that a
perfectly balanced expert has f_i = 1. (2) The *device-level* loss groups experts by the
device they would live on and balances **devices**, not experts. With expert parallelism
that is what matters for step time, because the slowest device sets the pace. DeepSeekMoE uses
a small expert-level α1 (only prevents collapse) and a larger device-level α2 at scale.

**Paper equations.** Expert-level, DeepSeekMoE Eqs. (12)–(14):
$$\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i \qquad (12)$$
$$f_i = \frac{N'}{K' T} \sum_{t=1}^{T} \mathbb{1}(\text{token } t \text{ selects expert } i) \qquad (13)$$
$$P_i = \frac{1}{T} \sum_{t=1}^{T} s_{i,t} \qquad (14)$$
- N' = mN − K_s: number of routed experts (31); K' = mK − K_s: routed experts per token (3)
- T: number of tokens; s_{i,t}: router softmax (Eq. 11); α1: expert-level factor (0.01 here)
- With uniform routing, f_i = 1 and P_i = 1/N', so the loss equals α1. The test checks this.

Device-level, DeepSeekMoE Eqs. (15)–(17), with routed experts partitioned into D groups {ℰ_1, …, ℰ_D}:
$$\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} f'_i P'_i \qquad (15)$$
$$f'_i = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j \qquad (16)$$
$$P'_i = \sum_{j \in \mathcal{E}_i} P_j \qquad (17)$$
- f'_i is the *mean* f over the experts on device i; P'_i is the *sum* of P over them.

**Where it lives.**
- `moe/router.py::deepseek_expert_aux_loss`: counts over all K' choices (`torch.bincount(topk_idx.reshape(-1), ...)`),
  `f = counts * n_routed / (k * T)` (Eq. 13), `P = probs.mean(dim=0)` (Eq. 14), `alpha1 * torch.sum(f * P)` (Eq. 12).
- `moe/router.py::deepseek_device_aux_loss` and `moe/router.py::device_groups`.
- Called from `moe/moe_deepseek.py::DeepSeekMoE.forward`. The two parts are logged separately as `extra["aux_expert"]` / `extra["aux_device"]`.
- Config: `moe.deepseek.aux_alpha` (α1 = 0.01), `device_aux_alpha` (α2, **default 0 = off**), `n_devices`.
- Tests: `tests/test_aux_loss.py::test_deepseek_expert_aux_manual_and_uniform`, `test_deepseek_device_aux_manual`, `test_device_loss_off_by_default_and_added_when_on`.

**Gotchas / design choices.**
- **Switch's f_i vs DeepSeek's f_i.** Switch's f_i sums to 1 and needs the explicit ×N in
  Eq. 4. DeepSeek folds the scaling into f_i (sum of f_i = N'). Both give "α at perfect balance".
- **31 experts don't split evenly into D devices.** `device_groups` uses
  `torch.arange(n).tensor_split(D)`: contiguous groups whose sizes differ by at most 1.
  Eq. 16 uses the group mean, so unequal group sizes are handled correctly.
- **The device loss is off by default.** This repo trains on one GPU, so there are no
  devices to balance. It is implemented and tested for E6 discussion. Note that E6
  (`scripts/placement_sim.py`) measures device imbalance from routing traces; it does not
  retrain with α2 > 0.

---

## 10. Expert parallelism and all-to-all

**Plain language.** At scale, the experts don't fit on one chip, so each device holds a
subset of experts (expert parallelism, EP). Each device routes its own tokens; then every
device must send each token to whichever device holds its chosen expert. That is an
**all-to-all**: every device sends a different slice to every other device. After the
experts run, a second all-to-all brings outputs back (combine). The backward pass repeats
both in reverse. Two things now set the step time: (1) the **slowest device** (load
imbalance becomes tail latency), and (2) **interconnect bandwidth** for the all-to-alls.
Capacity factor also sets the all-to-all message size, because padded buffers are what is sent.

**Paper.** Switch Sec. 5 ("Designing models with data, model, and expert-parallelism",
Fig. 9) describes how experts and tokens are sharded across cores. There is no single
equation. DeepSeekMoE Sec. 3.3 motivates the device-level loss (Section 9) for the same reason.

**The analytical model this repo uses** (MASTER §7, E6), per step and for D devices:
$$\text{straggler slowdown} = \frac{\max_d \text{load}_d}{\operatorname{mean}_d \text{load}_d}$$
$$\text{all-to-all bytes} = (\text{tokens sent off-device}) \times d_{\text{model}} \times \text{bytes per element}\quad(\text{forward} + \text{backward})$$
$$\text{step time} = \sum_{\text{layers}} \max_d(\text{compute}_d) + \sum_{\text{layers}} \max_d\Big(\frac{\text{bytes sent}_d}{\text{link bandwidth}}\Big)$$
- load_d: kept (token, expert) assignments that land on device d's experts; "off-device" means the
  token's home device (data-parallel shard, `home(t) = floor(t·D/N)`) differs from its expert's device.
- All-to-all bytes count **two** all-to-alls per layer in the forward pass (dispatch, then
  combine), each moving every off-device assignment once: `a2a_fwd = 2 × offdevice × d × bytes`;
  forward + backward doubles it again. Top-k copies of one token are not de-duplicated.
- Compute per assignment is 4·d·w FLOPs forward (two GEMMs), ×3 with backward; DeepSeek's shared
  expert is replicated on every device and runs on local tokens (balanced, no traffic).

**Where it lives.**
- **There is no real multi-GPU expert parallelism in this repo** (single GPU, MASTER §0). EP is studied analytically.
- `moe/router.py::device_groups` and `deepseek_device_aux_loss`: the device partition used by the loss.
- `scripts/route_trace.py::trace_text` and `scripts/route_trace.py::trace_batches`: dump real
  routing decisions (`topk_idx`, `gates`, `kept`) from trained checkpoints. Those traces feed E6.
- `scripts/placement_sim.py` (E6, final; design notes in `docs/E6_NOTE.md`):
  - `scripts/placement_sim.py::round_robin_placement`: expert e → device e mod D.
  - `scripts/placement_sim.py::greedy_placement`: longest-processing-time-first on the
    per-expert load of the **calibration** half of the trace steps, capped at ceil(E/D)
    experts per device (so expert memory per device is the same as round-robin), and scored
    on the **other** half (`--calib-frac`, default 0.5) so it cannot overfit the steps it is judged on.
  - `scripts/placement_sim.py::home_devices`: data-parallel token homes, contiguous blocks.
  - `scripts/placement_sim.py::step_layer_traffic` and `scripts/placement_sim.py::simulate`:
    per (step, layer) assignments per device, bytes each device sends (dispatch + combine), off-device count.
  - `scripts/placement_sim.py::straggler`: max ÷ mean device load.
  - `scripts/placement_sim.py::estimate_step_time`: the step-time formula above, returning
    `compute_ms`, `compute_balanced_ms`, `imbalance_ms`, `comm_ms`, `step_ms` per step. The
    dashboard's bandwidth slider calls this function live on the saved arrays.
  - `scripts/placement_sim.py::load_placement_arrays`: reads `placement_arrays.npz` for the dashboard and report.
  - `scripts/placement_sim.py::latest_measured_tflops`: default device TFLOP/s = measured GEMM
    roof of the newest complete E5 run of the same config, else the config value. (In the main
    results E6 ran before the final E5 run existed, so it used the config value; see "What surprised me".)
  - Outputs (`results/main/e6_placement/<run_id>/`): `placement_summary.jsonl`,
    `placement_arrays.npz`, `placement_params.json` (contract: `docs/DASHBOARD_SPEC.md` §1.8).
- `moe/metrics.py::load_imbalance` (max/mean over experts) is the same statistic at expert level.

**Gotchas / design choices.**
- E6 is a **simple analytical model, not a simulator**: no link contention, no topology, no
  overlap of compute and communication, no queueing, perfect kernel efficiency. Absolute step
  times are therefore only illustrative; the comparisons (policy vs policy, D vs D, compute vs
  communication as bandwidth changes) are the point.
- When E = D (Switch with 8 experts on 8 devices) every placement puts one expert per device,
  so greedy and round-robin are identical by construction: placement cannot fix imbalance
  that is inside the router.
- Expert imbalance ≠ device imbalance: grouping experts can average out imbalance (or
  concentrate it, if correlated experts share a device). This is why placement matters.
- Bridge to WSC-LLM: on a wafer-scale chip the "devices" are dies on a 2D mesh, so the
  all-to-all cost depends on hop distance and bisection bandwidth, not just one link bandwidth.

---

## 11. Loop vs batched dispatch

Both paths compute **identical math** from the same routing decisions and the **same
stacked weights** (`moe/experts.py::ExpertBank`). Tests check outputs and gradients agree:
`tests/test_switch.py::test_loop_equals_batched_no_drop` / `test_loop_equals_batched_with_drops`,
`tests/test_gshard.py::test_loop_equals_batched_with_drops`, `tests/test_deepseek.py::test_loop_equals_batched`,
and the model-level `tests/test_model.py::test_loop_and_batched_dispatch_agree`.

**Loop (readable reference).** `moe/moe_switch.py::CapacityMoE._dispatch_loop`,
`moe/moe_deepseek.py::DeepSeekMoE._routed_loop`. For each expert: find its tokens with
`nonzero`, gather them, call `ExpertBank.expert_forward` (two small GEMMs + GELU), and `index_add`
the gated output back.

**Batched.** `moe/moe_switch.py::CapacityMoE._dispatch_batched`: compute each
assignment's buffer slot `expert·C + pos`, scatter into `buf[E, C, d]`, run
`ExpertBank.batched_forward` (two `torch.bmm`), gather back × gate.
`moe/moe_deepseek.py::DeepSeekMoE._routed_batched`: stable `argsort` by expert,
`bincount` → segment starts, pack into `buf[E, max_count, d]`, bmm, unsort with `index_copy`, sum the k slots.

**Why batched is faster:**
1. **Fewer kernel launches.** The loop issues roughly 5–6 kernels per expert (compare,
   nonzero, gather, 2 GEMMs, GELU, scale, index_add). For DeepSeek's 31 experts that is
   well over a hundred small launches per layer, and launch overhead (microseconds each)
   dominates when each GEMM is small. Batched issues a fixed number of kernels independent of E.
2. **No per-expert host sync.** `nonzero` returns a data-dependent shape, so the CPU must wait
   for the GPU to finish before it can launch the next kernel. That happens once per expert
   per layer and serializes CPU and GPU. The Switch batched path has **zero** host syncs
   (C is computed from N on the host). The DeepSeek batched path has **one**
   (`max_count = int(counts.max().item())` in `_routed_batched`, to size the buffer).
3. **bmm.** One batched GEMM over E experts keeps the GPU's SMs busy. A single small expert
   GEMM may not fill the GPU, and the loop runs them one after another.

**The price of batched: padding.** Switch pads every expert to C rows; DeepSeek pads to
`max_count` rows, so a skewed load (one hot expert) makes *every* expert do the hot expert's
work. Padding rows are zeros and produce zero output (GELU(0)·W = 0), but they cost real
FLOPs, bytes and memory. `moe/flops.py::moe_layer_cost` reports `padding_fraction`; E5 measures it.
Whether batched beats loop is a measurement, not a given; E5 (`scripts/profile_layer.py`) measures it
(results: "What surprised me" below and `RESULTS.md` › "E5: single-layer systems profile").

**How E5 sees the phases.** The layers wrap each forward phase in `with phase("router")`,
`phase("dispatch")`, `phase("expert_gemm")`, `phase("shared_expert")`, `phase("combine")`
(`moe/profiling.py::phase`). In normal training this is a no-op. Under `torch.profiler` each
phase becomes a `record_function` range, and `moe/profiling.py::PhaseTimer` records CUDA event
pairs per phase. In the loop path a phase occurs once per expert, and the timer sums them.
Only forward phases are annotated; backward time is reported as a whole.

**Sort-and-segment vs the paper's [T, E, C] einsum (memory-motivated).** Switch Fig. 16
builds one-hot `dispatch` and `combine` tensors of shape `[tokens, experts, capacity]` and uses
two einsums. That tensor has N·E·C ≈ N²·cf elements (because C ≈ N·cf/E): it grows
**quadratically** in N. PLAN §7 works it out for E5's largest case (64K tokens, 8 experts,
cf = 1.25): about 5.4 billion elements, which does not fit in 12 GB. Sort-and-segment needs
only O(k·N) indices plus the `[E, C, d]` buffer. The literal formulation is kept as a
teaching and test reference: **`moe/moe_switch.py::einsum_reference`**, checked by
`tests/test_switch.py::test_einsum_reference_matches_batched` and `tests/test_gshard.py::test_einsum_reference_matches_batched`.

**Determinism and dtype details.**
- **Fixed-order k-sum** (the `for j in range(1, k): y = y + y_assign[...]` loops at the end of
  `CapacityMoE._dispatch_batched` and `DeepSeekMoE._routed_batched`): the k
  contributions of a token are added with explicit `y = y + y_assign[j]` in slot order, not
  `.sum(0)` and not a scatter-add. Two reasons: (a) a parallel `index_add` with repeated
  targets accumulates in a thread-dependent order, which made DeepSeek non-bit-deterministic
  on multithreaded CPU (escalated in PROGRESS.md, fixed); (b) under CUDA autocast `torch.sum`
  is on the fp32 list, so `.sum(0)` would silently promote the output to fp32.
- In DeepSeek batched every scatter uses **unique** indices (a permutation or unique buffer
  slots), and the input copy uses `expand` rather than fancy indexing so that its backward is
  a plain sum. Test: `tests/test_deepseek.py::test_bitwise_deterministic_multithreaded_cpu`.
- The loop path is the *reference*; use `moe.dispatch: batched` for training (PROGRESS: loop syncs per expert).

---

## 12. Arithmetic intensity and the roofline for expert GEMMs

**Plain language.** A kernel is **memory-bound** if the GPU spends its time waiting on DRAM,
and **compute-bound** if it spends it doing math. Arithmetic intensity (AI) = FLOPs per byte
moved. The roofline says attainable FLOP/s = min(peak FLOP/s, AI × memory bandwidth). The
**ridge point** AI* = peak FLOP/s ÷ bandwidth separates the two regimes.

**The formula in this repo.** `moe/flops.py::gemm_cost`, for `C[m, n] = A[m, k] @ B[k, n]`:
$$\text{FLOPs} = 2mkn, \qquad \text{bytes} = (mk + kn + mn)\cdot b, \qquad
\text{AI} = \frac{2mkn}{(mk + kn + mn)\, b}$$
- b: bytes per element (2 for bf16). This is **compulsory traffic**: each operand read once,
  output written once (perfect on-chip reuse). Real traffic is higher, so this AI is an **upper bound**.

For an expert's first GEMM, m = n_e (tokens routed to that expert), k = d, n = w:
$$\text{AI} = \frac{2\, n_e\, d\, w}{(n_e d + d w + n_e w)\, b}$$
Two limits (pure algebra, not measurements):
- **Few tokens** (n_e ≪ d, w): the weight term dw dominates the bytes, so
  AI ≈ 2 n_e / b, i.e. **AI ≈ n_e FLOP/byte in bf16**. Intensity grows linearly with the
  number of tokens the expert gets. The weights must be read whether the expert processes 1 token or 1000.
- **Many tokens** (n_e ≫ d, w): AI → 2dw / ((d + w) b), a ceiling set by the expert's shape.
  A narrower expert (smaller w) has a lower ceiling.

**Why fine-grained experts become memory-bound.** DeepSeek's routed experts are narrower
(w = d_ff/4) **and** each sees fewer tokens (≈ k·N/E = 3N/31, vs N/8 for Switch and N for
the dense FFN). Both effects lower AI. At small N (small batches, or inference decode) every
expert GEMM can fall below the ridge point. Meanwhile, the whole layer still reads all
8·d_ff·d·2 expert weights (in the batched path) while doing only the FLOPs of one d_ff FFN.
That makes the MoE layer much more weight-bandwidth-hungry per FLOP than the dense FFN.

**Where it lives.**
- `moe/flops.py::gemm_cost`, `moe/flops.py::expert_ffn_cost` (per-kernel breakdown,
  options `hidden_traffic` = "unfused" (default; what eager PyTorch does: h written, GELU
  reads/writes, GEMM2 reads) / "gemm" / "fused", `pass_="fwd_bwd"` (backward = 2 GEMMs per
  forward GEMM, so 3× forward FLOPs), and `include_autocast_weight_cast` (the fp32→bf16
  weight copy autocast does on every forward)).
- `moe/flops.py::moe_layer_cost`: a whole layer given the actual per-expert counts;
  reports executed vs useful FLOPs (`flops`, `flops_useful`, `padding_fraction`), router cost in fp32, and dispatch bytes.
- `scripts/profile_layer.py` (E5, final). One FFN layer only (no attention/norm/residual):
  - `scripts/profile_layer.py::LayerRunner`: builds one layer for a (variant, N, E, cf) and
    exposes `fwd` (eval mode, `no_grad`) and `fwd_bwd` (train mode, backward of y with a random
    upstream gradient plus the aux loss); `set_dispatch` switches loop/batched on the same weights.
  - `scripts/profile_layer.py::time_iters`: CUDA-event timing after warmup; the median of the
    repeats is reported (`summarize` adds p10/p90/mean).
  - `scripts/profile_layer.py::Profiler.sweep_tokens` / `sweep_experts` / `sweep_capacity` /
    `sweep_skew` → `bench.jsonl`; `Profiler._cost` attaches the analytic FLOPs and bytes from
    `moe/flops.py::moe_layer_cost` (bf16, autocast weight cast included), so achieved
    TFLOP/s = analytic FLOPs ÷ measured time and intensity = analytic FLOPs ÷ analytic bytes.
    The expert-count sweep keeps activated compute fixed (Switch: E experts of width d_ff,
    top-1; DeepSeek: total routed width 8·d_ff split into E experts, top-E/8, no shared expert).
    The skew sweep adds a Zipf-shaped offset to the router logits (`skew_offset`).
  - `scripts/profile_layer.py::Profiler.gemm_points` + `time_gemm` → `gemm.jsonl`: one isolated
    expert GEMM at the per-expert row count the batched path actually executes (C for
    Switch/GShard, `max_count` for DeepSeek), so each expert GEMM can be placed on the roofline by itself.
  - `scripts/profile_layer.py::Profiler.breakdown_timers` (`moe/profiling.py::PhaseTimer`,
    CUDA-event wall time per phase) and `Profiler.trace_and_profiler_breakdown`
    (`torch.profiler` kernel time inside each `moe/profiling.py::phase` range) → `breakdown.jsonl`,
    one row set per method. Only forward phases are annotated, so in `fwd_bwd` the "other"
    phase contains the whole backward pass.
  - `scripts/profile_layer.py::Profiler.measure_ceilings` → the **measured** roof in
    `roofline.json`: best median bf16 square GEMM over three sizes and a 1 GiB device-to-device
    copy (read + write counted), at the start and at the end of the run (the higher is kept;
    both are stored, plus a `power_state_changed` flag). `scripts/profile_layer.py::spec_ceiling`
    gives the **spec-sheet** roof: 672 GB/s is NVIDIA's published bandwidth; the bf16 peak is
    *derived* (992 "AI TOPS" ÷ 16, the FP4-sparse → BF16-dense ratio in NVIDIA's Blackwell
    whitepaper tables), not printed by NVIDIA, and `roofline.json` records the source and a
    lower clock-based alternative.
  - `scripts/profile_layer.py::Profiler.gpu_state` / `gpu_busy` / `other_gpu_processes`:
    clocks, temperature, power before and after each benchmark group, and a contention check.

**Gotchas.** GELU, softmax, norms and gate multiplies are not counted in FLOPs (PLAN §5
convention). Each expert GEMM can be placed on the roofline separately via the `kernels` list.

---

## 13. How the matched-compute comparison is set up

**The rule** (MASTER §4.3). Every MoE variant has **total expert-FFN params = 8 × the dense
FFN** and **activated expert-FFN params (and FLOPs) per token = 1 × the dense FFN**:

| variant | experts | width | active per token | total FFN width | active FFN width |
|---|---|---|---|---|---|
| dense | 1 | d_ff | 1 | d_ff | d_ff |
| switch | 8 routed | d_ff | top-1 | 8·d_ff | d_ff |
| gshard | 16 routed | d_ff/2 | top-2 | 8·d_ff | d_ff |
| deepseek | 1 shared + 31 routed | d_ff/4 | 1 + top-3 | 8·d_ff | d_ff |
| dense_x8 (upper bound) | 1 | 8·d_ff | 1 | 8·d_ff | 8·d_ff |

Per layer: FFN params of width w = 2·d·w (no biases), total = 2·d·w·(n_shared + n_routed),
active = 2·d·w·(n_shared + k), FFN forward FLOPs/token = 2 × active.

**Where it lives.**
- `moe/layer_api.py::ffn_spec` is the single source of truth for widths and counts, used by
  both the modules and `flops.py`, so they cannot disagree. Its assert enforces active width == d_ff.
- `moe/flops.py::param_row`, `param_table`, `format_param_table`:
  the table printed at startup (`moe/model.py::MoETransformer.param_table_str`) and shown in the dashboard.
- `moe/model.py::MoETransformer.verify_param_count`: real module count == analytic count.
- Tests: `tests/test_param_matching.py::test_matching_rule_exact_main` and `test_main_layer_params_match_flops_table`.
- The full numeric table for the main config is in `docs/PLAN.md` §5 (analytic, computed from config dimensions).

**Why router params are excluded from the matching.** The router has d·E params per layer,
and E differs by variant (0 / 8 / 16 / 31), so router params cannot be equal across variants
while the expert structure differs. They are tiny (PLAN §5: about 0.1% of total), and they
are routing overhead, not FFN capacity. So the claim is stated precisely: *expert-FFN total
params and activated expert-FFN params/FLOPs are exactly equal; router params and FLOPs are
reported separately (the `router/layer` column) and excluded from the matching.* They **are**
included in `total_params`, `active_params` and `model_fwd_flops_per_token`.

**What "matched compute" does not cover.** It is matched *analytic FFN FLOPs*. It does not
include dispatch/combine data movement, capacity padding, the router, or kernel-launch
overhead, so wall-clock time is not matched. That gap is exactly what E5 measures. The tokens
per step and the number of steps are the same for every variant (same token budget).

---

## 14. Smaller details worth knowing when reading the code

- **The `MoEAux` contract** (`moe/layer_api.py::MoEAux`): every FFN returns
  `(y, MoEAux)`. `expert_counts` is pre-drop and summed over all k choices; `kept_counts` is
  post-capacity; extra per-token tensors appear only with `return_routing=True` (they are big).
- **Routing logs with one sync**: `moe/metrics.py::RoutingAccumulator` keeps sums on the GPU
  (`update`) and copies to the CPU once per log interval (`flush`). `load_imbalance` = max/mean,
  `load_cv` = std/mean (population std).
- **`moe_every`** (`moe/layer_api.py::is_moe_layer`): 2 puts MoE on every other block, like the
  Switch paper. The default (1) replaces every FFN, like DeepSeekMoE.
- **E3 expert counts** (PLAN §7 table): "disable top-r fraction" becomes integer counts per
  variant (e.g. Switch with 8 experts cannot represent r = 1/16). Plots use the actual fraction.

---

## What surprised me / what didn't replicate

Main config only: 6 layers, d = 384, 6000 steps × 8,192 tokens = 49.2M TinyStories tokens,
peak LR 2e-3 for every variant, one RTX 5070 Ti Laptop GPU, bf16 autocast with an fp32 router.
Every number below is copied from the cited `RESULTS.md` section (generated by
`scripts/make_report.py` from `results/main/`) or computed from the cited file. Seeds: dense,
Switch and DeepSeekMoE have 2 seeds (1337, 1338); GShard and dense ×8 have 1.

**E1: quality at matched activated compute** (`RESULTS.md` › "E1: variants at matched activated compute";
`results/main/e1_main/*/final.json`).
- **DeepSeekMoE is the clear winner.** Final val loss 1.658 vs dense 1.708: **−0.050 nats**
  (mean of 2 seeds each). The seed spread (max − min) is 0.004 for both dense and DeepSeekMoE,
  so the gap is more than 10× the seed-to-seed variation we measured. With only 2 seeds this is
  a strong hint, not a confidence interval.
- **Switch did not beat dense.** Switch 1.704 vs dense 1.708 (−0.004), which is the size of
  the dense seed spread (0.004). At this scale Switch is **indistinguishable from dense**. The
  Switch paper's headline (Switch beats a FLOP-matched dense model) did **not** replicate here.
  Plausible reasons, none tested: tiny model and short training (0.78 training tokens per total
  parameter for the 63M-parameter models), only 8 experts, and the main LR was chosen on a
  proxy (`PROGRESS.md`, 2026-10-05 LR check), not tuned per variant.
- **GShard (16 experts, top-2) sits in between:** 1.678, −0.030 vs dense, **1 seed**. The
  ordering DeepSeekMoE < GShard < dense matches DeepSeekMoE's claim that fine-grained + shared
  experts beat GShard-style top-2 at the same compute, but the GShard point has no seed spread.
- **The "dense upper bound" was not an upper bound.** dense ×8 (all 63M parameters active,
  4.3× the model FLOPs/token) reached 1.678 (1 seed): tied with GShard and **worse than
  DeepSeekMoE**. DeepSeekMoE reports the opposite ordering. The diagnosis in
  `docs/E1_DENSE_X8_NOTE.md` found no implementation bug; the likely cause is that one peak LR
  (2e-3) is too high for a 12,288-wide FFN under standard parameterization (its W_out moved 29×
  its init norm vs 13× for dense), and every 63M model is badly undertrained at 49.2M tokens.
  Weak evidence about the paper's claim either way.
- **MoE is not free on one GPU.** Median training throughput (same `RESULTS.md` table):
  dense 201k tok/s, Switch 126k (63% of dense), DeepSeekMoE 110k (55%), GShard 109k (54%),
  dense ×8 71.9k. Peak memory: dense 2,005 MiB, Switch 3,087 (1.5×), GShard 3,135 (1.6×),
  DeepSeekMoE 5,809 MiB (**2.9×**). Matched FLOPs (model forward FLOPs/token within 0.5%:
  29.9M vs 30.0M, `RESULTS.md` param table) does not mean matched wall-clock: the experts'
  total weights, optimizer state and dispatch buffers cost memory, and dispatch costs time (E5).
- Routing stayed healthy: final windowed load imbalance (max ÷ mean) 1.07–1.08 for Switch,
  1.16–1.17 for DeepSeekMoE, 1.45 for GShard (`RESULTS.md` › "Routing statistics (E1 runs)").

**E2: capacity factor** (`RESULTS.md` › "E2: Switch capacity factor sweep"; 1 seed per point).
- A clean, **monotone** trade-off. cf 0.75 → 1 → 1.25 → 2: val loss 1.838 → 1.732 → 1.705 →
  1.698, val drop rate 25.2% → 5.1% → 0.2% → 0.0%. The returns diminish quickly: going from
  1.25 to 2 buys only −0.007 nats.
- The cost side is real but small at this size: median throughput 132k / 133k / 125k / 113k
  tok/s and peak memory 2,926 → 3,330 MiB (+14%) from cf 0.75 to cf 2. This matches the
  direction of Switch Table 1 / Sec. 2.2 (higher capacity = better quality, slower step).
- At cf = 1.25 (the default) Switch drops 0.2% of validation tokens: dropping is not what
  holds Switch back in E1.

**E3/E4: expert ablations** (`RESULTS.md` › "E3/E4: expert ablations";
`results/main/e3e4_ablation/20261005-114839_ablation/ablation.jsonl`). These are **eval-time
changes to trained checkpoints, not retraining**, on one checkpoint per variant (Switch s1338,
DeepSeekMoE s1338, GShard s1337), on 50 validation batches (so the r = 0 baselines, 1.813 /
1.768 / 1.789, differ from the E1 numbers, which use 20 batches).
- **Direction matches DeepSeekMoE Fig. 4, magnitude does not.** Disabling each token's top
  routed experts (the token falls back to its next-ranked experts; gates stay the original
  softmax probabilities, Eq. 10 literally) hurts DeepSeekMoE the most at every comparable
  fraction: +3.04 nats at 6.5% disabled (2 of 31) vs GShard +2.33 at 6.2% (1 of 16); +3.19 at
  12.9% vs GShard +2.74 and Switch +1.72 at 12.5%; +3.26 at 25.8% vs GShard +3.12 and Switch
  +2.10 at 25%. The paper reads a steeper curve as "less redundant, more specialized experts".
  But here every curve is a **cliff followed by a plateau**: the first disabled expert(s) do
  almost all the damage and every model ends up at 3.9–5.0 nats. So this mostly says that
  replacing a token's chosen experts with the wrong ones is catastrophic for all three
  variants, and only weakly orders them.
- **Which experts, not how many.** In the same file, DeepSeekMoE evaluated with only its top-1
  routed expert (k = 1 instead of 3, `e4_k_sweep`) loses just +0.029 nats, while keeping 3
  experts but shifting them to ranks 3–5 (`e3_disable_top`, 2 disabled) loses +3.04. The k
  sweep is flat above the trained k: k = 2: +0.004, k = 4: −0.001. The routed computation is
  very concentrated in the top choice.
- **The shared expert matters and an extra routed expert does not stand in for it.** Removing
  it costs +0.340 nats (`no_shared_same_k`); removing it and activating one more routed expert
  to restore the active FFN width still costs +0.328 (`no_shared_plus_one`). This is the same
  qualitative result as the shared-expert experiment in DeepSeekMoE Sec. 4.5. It measures how a
  model trained *with* a shared expert reacts to losing it, not what a model trained without
  one would reach.

**E5: one MoE layer on the laptop GPU** (`RESULTS.md` › "E5: single-layer systems profile";
`results/main/e5_profile/20261005-120250_profile/{bench,gemm,breakdown}.jsonl`, `roofline.json`).
- **Batched dispatch wins at training batch size, but the win shrinks as N grows.**
  Loop ÷ batched latency at the training N = 8,192 tokens: Switch 1.46× fwd / 1.90× fwd+bwd,
  GShard 1.74× / 2.62×, DeepSeekMoE 2.91× / 4.11×. At N = 1,024: 2.86× / 5.91× fwd for Switch /
  DeepSeekMoE; at N = 65,536: 0.89× (loop faster) / 1.10×. Fine-grained DeepSeekMoE (31
  experts) gains most because the loop pays per-expert launches and host syncs 31 times; at
  large N those fixed costs amortize while batched pays capacity padding (Switch/GShard
  batched rows at N = 8,192 have a padding fraction of 0.20, DeepSeekMoE 0.12, in `bench.jsonl`).
- **Measured vs spec roof.** Measured: 57.3 TFLOP/s bf16 GEMM and 481.5 GB/s copy
  (`RESULTS.md` rounds the latter to 482). Spec: 672 GB/s published; 62.0 TFLOP/s *derived*
  (see Hardware caveats). Measured ridge point 57.3 ÷ 0.4815 ≈ 119 FLOP/byte (spec ridge ≈ 92).
  No layer exceeded the measured roof; the best layer was the **dense** FFN at 38.2 TFLOP/s
  (67% of the measured peak).
- **MoE layers reach a small fraction of the roof.** Forward, batched, N = 8,192 (`bench.jsonl`):
  dense 38.0 TFLOP/s, Switch 9.3, DeepSeekMoE 7.8, GShard 6.1. Relative to the measured roof
  at each point's own arithmetic intensity, that is about 66% for dense but 15–24% for the
  three MoE layers.
- **The expected "fine-grained experts are memory-bound" result only half shows up.**
  *Layer level:* in the forward token sweep (config E), the MoE layer points have
  compulsory-traffic intensity 23–127 FLOP/B (DeepSeekMoE 23–84), almost all left of the
  119 FLOP/B ridge, so by position they are in the memory-bound region. But DeepSeekMoE batched
  forward moves only 87–117 GB/s against a 481.5 GB/s measured copy roof, so DRAM bandwidth is
  **not** what limits it either.
  *GEMM level* (`gemm.jsonl`, one expert's GEMM at the rows the batched path executes):
  DeepSeekMoE's 384 × 384 expert GEMMs have intensity 76–105 FLOP/B (below the ridge) only at
  N ≤ 2,048 (≤ 231 rows per expert). At the training N = 8,192 (900 rows) the intensity is 158,
  right of the ridge, yet the GEMM runs at 15.5 TFLOP/s = 27% of the roof; the dense FFN's GEMM
  at the same N runs at 48.5 TFLOP/s. The honest reading: small expert GEMMs under-fill the
  GPU and the layer is dominated by **data movement around the GEMMs and launch overhead**,
  not by DRAM bandwidth. The phase timers agree: at N = 8,192 the `dispatch` phase is 46% of
  the batched Switch layer's forward time, 61% for GShard and 29% for DeepSeekMoE
  (`breakdown.jsonl`, method `cuda_events`). Intensity is an upper bound (compulsory traffic),
  so the true points sit further left; the conclusion does not change.

**E6: placement model** (`RESULTS.md` › "E6: expert placement"; analytical, not a simulator;
`results/main/e6_placement/20261005-122540_placement/`).
- **Straggler slowdown grows with D.** All-layer max ÷ mean device load at D = 2 / 4 / 8,
  greedy: DeepSeekMoE 1.028 / 1.068 / 1.149, GShard 1.022 / 1.061 / 1.129, Switch 1.026 / 1.073 / 1.203.
- **Greedy beats round-robin, modestly.** At D = 8: GShard 1.251 → 1.129, DeepSeekMoE 1.187 →
  1.149; the estimated step at default parameters falls 1.46 → 1.34 ms (GShard) and 1.70 → 1.67 ms
  (DeepSeekMoE). For Switch at D = 8 the two policies are identical (1.203 both): 8 experts on 8
  devices leaves nothing to place, so the imbalance is in the router itself.
- **Communication vs compute.** All-to-all fwd+bwd bytes per step at D = 8: DeepSeekMoE 396 MB,
  GShard 251 MB, Switch 131 MB, i.e. roughly proportional to k (3 / 2 / 1). With the default
  parameters (57.3 TFLOP/s per device, 64 GB/s link) compute still exceeds communication for every
  case, but for DeepSeekMoE at D = 8 they are nearly equal: compute 0.84 ms vs all-to-all 0.83 ms (greedy;
  `scripts/placement_sim.py::estimate_step_time` on `placement_arrays.npz`). By the same function,
  communication overtakes compute below about 63 GB/s for DeepSeekMoE, 41 GB/s for GShard and
  20 GB/s for Switch at D = 8 (greedy). Fine-grained top-k routing buys model quality with link bandwidth.
- Default device TFLOP/s = 57.3, the measured GEMM roof of the accepted E5 run
  (`placement_params.json` › `device_tflops_source`). An earlier E6 run that used the config value
  (50) is kept in `results/main/_superseded/`; straggler and byte numbers are identical, only the
  absolute ms and the crossover bandwidths differ.

**Hardware caveats** (laptop GPU; `results/main/_superseded/README.md`, `PROGRESS.md` 2026-10-05).
- Laptop power states change timings by large factors. The first E2 cf = 1.0 run ran in a
  low-power state (14k tok/s vs ~113–133k for the other E2 runs); its loss was identical
  (1.7325) but it was superseded and re-run (133k tok/s). The first E5 run was superseded for
  a memory-clock change. Both are kept in `results/main/_superseded/` and ignored by the report.
- The accepted E5 run still has `power_state_changed: true` in `roofline.json`: the memory clock
  moves between 9,001 / 11,126 / 14,126 MHz while the GPU idles between benchmark groups. The
  measured roof at start / end was 53.8 / 57.3 TFLOP/s and 481 / 482 GB/s, and 0 of 162 bench rows
  were flagged `low_power_state`. The Systems page shows this as a warning.
- The spec bf16 peak (62.0 TFLOP/s) is **derived** from NVIDIA's 992 "AI TOPS" (FP4 with
  sparsity) ÷ 16, not a printed number; a clock-based estimate gives 52.3. Our measured 57.3
  exceeds 52.3, so the 62.0 figure is plausible, but treat the spec roof as approximate
  (`roofline.json` › ceilings › spec › notes).
- E1/E2 throughput is wall-clock on a laptop (thermal and power limits, WSL2). E5 is the
  controlled measurement; E1 tok/s ratios are indicative only.

**Limitations.**
- Small scale: 6 layers, d = 384, 8–31 routed experts, 13.8M active / 63M total parameters,
  49.2M training tokens (one TinyStories pass of ~60M cached tokens, no repeats). The papers
  train 100–1000× larger models on far more data; MoE gains are known to grow with scale.
- 1–2 seeds. Differences of a few thousandths of a nat are within noise; GShard and dense ×8
  have no seed spread at all.
- One peak LR for all variants, chosen on the small config by a pre-set rule at the edge of its
  grid (not extended); dense ×8 likely suffers from it.
- E3/E4 are eval-time ablations on one checkpoint per variant, not retrained models: they
  show how a trained model reacts to losing experts, not what a model trained without them would
  reach. One checkpoint per variant, so no seed spread.
- Single GPU: no real expert parallelism. E6 is an analytical model (no topology, overlap,
  contention or kernel efficiency) on 25 evaluation steps of real routing traces.
- E5 profiles one FFN layer with random inputs (near-uniform routing except the skew sweep), not
  the full model.

Earlier plumbing observations (before the main runs; kept for the record):
- At initialization Switch at cf = 1.25 drops almost nothing; drops appeared in training and
  fell again as routing balanced (E1 routing logs).
- DeepSeek batched dispatch was initially not bit-deterministic on multithreaded CPU (parallel
  `index_add` with k = 3 targets per row); fixed with unique-index scatters plus a fixed-order
  k-sum (Section 11).
- The E3 "jump then plateau" seen on a 60-step smoke checkpoint is still present on the
  trained main checkpoints (above), so it was not just undertraining.

---

## Questions for lab meeting

1. **Expert granularity vs memory bandwidth.** Fine-grained experts lower each GEMM's
   arithmetic intensity (AI ≈ tokens-per-expert in bf16 when tokens are few). On an
   accelerator with a fixed ridge point, is there an optimal expert width w for a given batch
   size? Should the accelerator, rather than the model designer, pick m?
2. **Weight streaming in decode.** At inference batch sizes, an MoE layer reads (up to) all E
   experts' weights while doing the FLOPs of one dense FFN. Does that make MoE inference
   fundamentally a memory-capacity and bandwidth problem, and would HBM vs SRAM-heavy designs
   (e.g. wafer-scale on-chip SRAM) change which MoE shape is best?
3. **All-to-all on a mesh (WSC-LLM bridge).** On a 2D mesh of dies, all-to-all cost scales
   with hop count and bisection bandwidth, not one link speed. Should expert placement be
   topology-aware (put frequently co-selected experts close together, using routing traces)?
   How does that interact with DeepSeek's device-level balance loss, which ignores topology?
4. **Capacity padding as wasted compute.** Switch-style static capacity buys static shapes
   with padded FLOPs, memory, and all-to-all bytes. Is dynamic-shape hardware/software
   (grouped GEMM, ragged tensors) worth its complexity, or is a little dropping cheaper overall?
5. **Load imbalance as tail latency.** In expert parallelism the step time is set by the most
   loaded device (straggler slowdown = max/mean). Is an auxiliary loss the right tool, or
   should balancing happen in the system (replicating hot experts, token rebalancing, or
   bias-based "aux-loss-free" balancing as in later DeepSeek models)?
6. **Router precision on low-precision hardware.** Switch needed an fp32 router. On hardware
   built around FP8/FP4 datapaths, what is the cheapest way to keep routing stable: a small
   fp32 unit, higher-precision accumulation only for the softmax, or quantization-aware routing?
7. **Kernel launch and host sync overheads.** Our loop path pays one host sync per expert.
   How much MoE speedup on real systems comes from better kernels (fused permute + grouped
   GEMM) rather than from the model itself? Would a hardware dispatch unit (token routing in
   the NoC or DMA engine) remove the permute cost entirely?
8. **Shared experts and hardware.** A shared expert is a dense GEMM on every token, so it has
   high arithmetic intensity and is easy to overlap with the routed experts' all-to-all. Is
   this part of why shared experts work well in practice, beyond the model-quality argument?
9. **Dropping is not causal.** With capacity limits, whether a token is dropped depends on
   other tokens in the batch. What does that mean for serving with continuous batching, where
   batch composition changes request to request (same prompt, different outputs)?
10. **Counting FLOPs honestly.** "Matched compute" here means matched analytic FFN FLOPs. For
    an accelerator paper, what should the comparison metric be instead: energy per token,
    bytes moved per token, or achieved latency at a fixed batch size?
11. **Expert specialization vs placement.** If routing traces show experts specialize by
    token type (e.g. punctuation, names), could a scheduler predict routing one layer ahead
    and prefetch expert weights or pre-position tokens to hide all-to-all latency?

---

## Resume-ready facts

Each line is accurate as written and traceable to `RESULTS.md` (generated from `results/main/`
by `scripts/make_report.py`). Keep the qualifiers: they are what makes each claim defensible.
Scale for all of them: 6-layer, d = 384 decoder, 13.8M active / 63M total parameters, 49.2M
TinyStories tokens, one RTX 5070 Ti Laptop GPU.

1. Implemented Switch Transformer (top-1, capacity factor, load-balancing loss, fp32 router),
   GShard-style top-2, and DeepSeekMoE (fine-grained + shared experts) MoE layers in PyTorch,
   each with a readable per-expert loop path and a batched path that tests check produce the
   same outputs and gradients. (`moe/`, `tests/`)
2. At matched activated FFN compute (model forward FLOPs/token within 0.5%), DeepSeekMoE reached
   val loss 1.658 vs 1.708 for the dense baseline (−0.050 nats; mean of 2 seeds, seed spread
   ≤ 0.004). (`RESULTS.md` › E1)
3. Switch Transformer (8 experts, top-1) matched but did not beat the FLOP-matched dense model at
   this scale (−0.004 nats, within seed noise); I report this as a non-replication of the
   paper's headline. (`RESULTS.md` › E1)
4. Measured the cost side: MoE training throughput was 54–63% of dense and peak memory up to
   2.9× (DeepSeekMoE) at matched FLOPs, wall-clock on a laptop GPU. (`RESULTS.md` › E1)
5. Capacity-factor sweep (1 seed per point): raising Switch's capacity factor from 0.75 to 2.0
   cut the validation token-drop rate from 25.2% to 0.0% and val loss from 1.838 to 1.698, with
   diminishing returns above 1.25. (`RESULTS.md` › E2)
6. Eval-time ablation (one checkpoint): removing DeepSeekMoE's shared expert raised val loss by
   0.34 nats, and activating an extra routed expert in its place recovered almost none of it
   (+0.33). (`RESULTS.md` › E3/E4)
7. Built a roofline for the GPU with a measured ceiling (57.3 TFLOP/s bf16 GEMM, 481.5 GB/s copy)
   next to a cited/derived spec ceiling (62.0 TFLOP/s derived, 672 GB/s); the best FFN layer
   reached 38.2 TFLOP/s (dense, 67% of measured), MoE layers 6–9 TFLOP/s forward at the training
   batch. (`RESULTS.md` › E5; per-variant TFLOP/s from `bench.jsonl`)
8. Batched expert dispatch was 1.5–2.9× faster than a per-expert loop in the forward pass at the
   training batch (8,192 tokens), up to 5.9× at 1,024 tokens, shrinking to 0.9–1.1× at 65,536
   tokens; the gain is largest for fine-grained experts. (`RESULTS.md` › E5)
9. Wrote an analytical expert-parallel placement model driven by real routing traces: device
   straggler slowdown at 8 devices was 1.13–1.25×, and greedy (LPT) placement cut GShard's from
   1.251× to 1.129×; it is a simple model, not a simulator. (`RESULTS.md` › E6)
10. Built an 8-page Streamlit dashboard and a self-contained HTML report in which every number
    is computed from result files (no hard-coded results), including an interactive view of
    per-token routing decisions in trained checkpoints. (`dashboard/`, `results/main/report.html`)

Do **not** claim: that MoE "beat dense" in general (Switch did not), that DeepSeekMoE matched the
dense upper bound (our dense ×8 point is unreliable, see `docs/E1_DENSE_X8_NOTE.md`), any
multi-GPU result (E6 is analytical), or statistical significance (1–2 seeds).
