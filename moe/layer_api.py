"""Common layer API shared by every FFN variant (dense, dense_x8, switch, gshard, deepseek).

This file is the *contract* between the backbone (``moe/model.py``, builder) and the FFN
variants (``moe/ffn.py``, builder; ``moe/moe_*.py``, architect). ``model.py`` must only rely
on what is written here. It never looks inside routing.

=====================================================================================
CONTRACT
=====================================================================================

Configs are plain nested dicts (``moe.utils.load_config`` returns a dict; key names are
fixed by ``docs/CONFIG_SCHEMA.md``). ``cfg`` below means the full resolved config;
``model_cfg = cfg["model"]`` and ``moe_cfg = cfg["moe"]`` (the WHOLE ``moe`` section, so a
layer can read the shared keys ``dispatch`` / ``renormalize_gates`` plus its own sub-dict).

1. Classes, module paths and constructors (all are ``torch.nn.Module``)::

    moe.ffn.DenseFFN(d_model: int, d_ff: int, model_cfg: dict)            # builder, T5
    moe.moe_switch.SwitchMoE(d_model: int, d_ff: int, moe_cfg: dict, model_cfg: dict)
    moe.moe_gshard.GShardMoE(d_model: int, d_ff: int, moe_cfg: dict, model_cfg: dict)
    moe.moe_deepseek.DeepSeekMoE(d_model: int, d_ff: int, moe_cfg: dict, model_cfg: dict)

   ``d_ff`` is always the BASE dense width (``model.d_ff``). Each MoE layer derives its own
   expert width and counts through :func:`ffn_spec` (the same function ``flops.py`` uses, so
   the analytic table and the real modules cannot disagree).
   ``dense_x8`` has no class of its own: it is ``DenseFFN(d_model, width_mult * d_ff, model_cfg)``.

   ``model.py`` should not call these constructors directly; it calls
   :func:`build_ffn` ``(variant, layer_idx, cfg)`` defined below, which also applies
   ``model.moe_every`` (non-MoE layers get a plain ``DenseFFN(d_model, d_ff)``).

2. Forward (identical for all five)::

    forward(x: Tensor[B, T, d_model], return_routing: bool = False) -> tuple[Tensor[B, T, d_model], MoEAux]

   * ``y`` has the same shape as ``x``. Its dtype follows autocast (bf16 on CUDA, fp32 on
     CPU). ``model.py`` adds it to the residual stream itself: ``h = h + y``. The FFN does
     NOT apply the pre-norm or the residual add.
   * A dropped token (Switch/GShard over capacity) gets ``y == 0`` exactly at that row, so
     it passes through on the residual only.
   * ``aux`` is a :class:`MoEAux`. ``DenseFFN`` returns ``MoEAux.empty(x.device)``.
   * The MoE aux loss is computed in both train and eval mode (eval logs it). ``model.py``
     returns ``sum(aux.aux_loss for all layers)`` and ``train.py`` adds it to CE in training
     only: ``loss = ce + aux_total`` (each layer's aux already includes its alpha, PLAN §7).

3. Train / eval mode (``module.training``):

   * Router jitter (Switch App. C: router *input* times ``U(1-eps, 1+eps)``) is applied only
     when ``self.training and jitter_eps > 0``.
   * Capacity factor: ``capacity_factor`` in train mode; in eval mode
     ``eval_capacity_factor`` if it is not None, else ``capacity_factor``.
   * Nothing else depends on the mode (no dropout anywhere in this repo).

4. Eval-time ablation switches (runtime attributes, NOT config keys; ``train.py`` never sets
   them). Set them with :func:`set_eval_overrides` ``(model, ...)``, which walks all modules
   and raises if a switch is not supported by any layer (so an ablation can never silently
   no-op). Supported attributes per class (``ABLATION_ATTRS`` class attribute):

   ==================== ========================= =======================================
   attribute            classes                    meaning
   ==================== ========================= =======================================
   eval_capacity_factor SwitchMoE, GShardMoE       cf used in eval mode; ``NO_DROP`` (=inf)
                                                   means capacity = N (no token dropped)
   n_disable_top_routed SwitchMoE, GShardMoE,      per token, mask the n highest-prob routed
                        DeepSeekMoE                experts, then take top-k of the rest; gate
                                                   = original softmax prob, no renorm
                                                   (DeepSeekMoE Sec. 4.5 / Fig. 4). Default 0.
   disable_shared       DeepSeekMoE                drop the shared-expert term of Eq. 9.
   k_routed_override    DeepSeekMoE                use this k instead of k_routed (Fig. 5).
   ==================== ========================= =======================================

   Each MoE class therefore declares ``ABLATION_ATTRS: tuple[str, ...]`` and exposes the
   read-only attributes ``spec`` (:class:`FFNSpec`), ``n_routed``, ``top_k`` and ``width``.

   These switches apply whenever they are set (in either mode); ``eval_capacity_factor``
   by definition only affects eval mode. ``disable_top_routed_ratio = r`` (master-prompt
   name) is accepted by :func:`set_eval_overrides` and converted to a count with
   ``floor(r * n_routed + 0.5)``; ablation scripts should prefer explicit counts
   (PLAN §7 E3 table).

5. Init: every weight matrix is initialised with :func:`init_weight_` ``(param, fan_in,
   model_cfg)``. ``fan_in`` = the input dimension of the matmul: ``d_model`` for
   Wq/Wk/Wv/Wo, FFN/expert ``W_in`` and the router; the FFN/expert width ``w`` for
   ``W_out``; ``d_model`` for the tied embedding (it is also the output head, whose input
   dim is d_model). RMSNorm weights are ones. No biases anywhere (PLAN §2).

6. Weight decay (PLAN §4): decay only weight matrices. Excluded: every parameter with
   ``ndim < 2`` (norm weights), the token embedding (tied with the output head), and router
   weights. The embedding and router are tagged with :func:`mark_no_weight_decay`
   (``model.py`` tags ``tok_emb.weight``; ``moe/router.py`` tags its own weight). Build the
   AdamW groups with :func:`build_param_groups` ``(model, weight_decay)``.

7. Parameter storage (needed for exact param matching): MoE layers store stacked expert
   weights ``W_in[E, d, w]`` and ``W_out[E, w, d]``; dense FFN stores ``W_in[d, w]`` and
   ``W_out[w, d]`` (either as nn.Parameter or as ``nn.Linear(bias=False)``, the count is
   ``2*d*w`` either way). DenseFFN must have exactly ``2*d_model*d_ff`` params.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import nn

# ----------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------

ALL_VARIANTS: tuple[str, ...] = ("dense", "switch", "deepseek", "gshard", "dense_x8")
MOE_VARIANTS: tuple[str, ...] = ("switch", "deepseek", "gshard")

#: eval_capacity_factor value meaning "capacity = N, nothing is dropped" (E3 uses this).
NO_DROP: float = float("inf")

#: DeepSeekMoE init: "all learnable parameters are randomly initialized with a standard
#: deviation of 0.006" (DeepSeekMoE Sec. 4.1 training settings). Not a config key; see
#: CONFIG_SCHEMA note in the T3 report.
DEEPSEEK_INIT_STD: float = 0.006


# ----------------------------------------------------------------------------------------
# MoEAux: what every FFN variant returns next to y
# ----------------------------------------------------------------------------------------

@dataclass
class MoEAux:
    """Auxiliary outputs of one FFN layer (PLAN §7).

    All tensors stay on the layer's device; nothing here forces a host sync. Statistics are
    detached; only ``aux_loss`` carries gradient.

    Fields:
        aux_loss: 0-dim fp32 tensor, ALREADY multiplied by its coefficient
            (Switch Eq. 4: ``alpha * N * sum_i f_i P_i``; DeepSeekMoE Eqs. 12-14:
            ``alpha1 * sum_i f_i P_i``, plus the device-level term Eqs. 15-17 if enabled).
            Zero for dense layers.
        n_experts: number of ROUTED experts E (0 for dense). Shared experts are not counted.
        expert_counts: int64 [E], assignments routed to each expert BEFORE capacity dropping,
            summed over all k choices (GShard: sums to 2N; DeepSeek: sums to k*N).
        kept_counts: int64 [E], assignments actually processed (= expert_counts when there
            is no capacity, i.e. DeepSeekMoE).
        drop_fraction: 0-dim fp32, dropped (token, slot) assignments / all assignments.
        router_entropy: 0-dim fp32, mean over tokens of the router softmax entropy (nats).
        topk_idx: int64 [N, k] routed-expert ids, N = B*T in row-major (b, t) order.
            Only filled when ``return_routing=True``.
        gates: fp32 [N, k] gate value used for each chosen expert (``return_routing`` only).
        kept: bool [N, k], False if that assignment was dropped (``return_routing`` only).
        top_k: experts chosen per token (k actually used, incl. ``k_routed_override``); 0 for dense.
        n_tokens: N = B*T tokens seen by this layer in this forward; 0 for dense.
        extra: dict of extra tensors. **0-dim entries are logged** (averaged into the
            ``extra`` field of routing_log.jsonl by ``moe.metrics``; e.g. DeepSeek's split
            ``aux_expert`` / ``aux_device``). **Entries with ndim >= 1 appear only with
            ``return_routing=True`` and are never logged.** Those are (DASHBOARD_SPEC R1-R3):
              ``router_probs``           fp32 [N, E] full router softmax (all MoE variants)
              ``routed_expert_out_norm`` fp32 [N, k] L2 norm of each GATED expert
                                         contribution g * E_i(x) (0 if dropped)
              ``shared_out_norm``        fp32 [N]    DeepSeek: L2 norm of the shared-expert sum
              ``routed_out_norm``        fp32 [N]    DeepSeek: L2 norm of the gated routed sum
    """

    aux_loss: torch.Tensor
    n_experts: int = 0
    expert_counts: Optional[torch.Tensor] = None
    kept_counts: Optional[torch.Tensor] = None
    drop_fraction: Optional[torch.Tensor] = None
    router_entropy: Optional[torch.Tensor] = None
    topk_idx: Optional[torch.Tensor] = None
    gates: Optional[torch.Tensor] = None
    kept: Optional[torch.Tensor] = None
    top_k: int = 0
    n_tokens: int = 0
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def empty(cls, device: torch.device | str | None = None) -> "MoEAux":
        """The dense / non-MoE case: zero aux loss, no routing statistics."""
        return cls(aux_loss=torch.zeros((), dtype=torch.float32, device=device))

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0


# ----------------------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------------------

def is_moe_layer(layer_idx: int, moe_every: int) -> bool:
    """Whether block ``layer_idx`` (0-based) uses the variant's FFN.

    ``moe_every=1``: every block. ``moe_every=2``: blocks 1, 3, 5, ... (every other FFN
    layer, the last one of each group of ``moe_every``; Switch Transformer places experts at
    every other FFN layer). The other blocks use a plain ``DenseFFN(d_model, d_ff)``.
    Applies to dense_x8 too (its wide FFN sits only at "MoE" positions), so total FFN
    params stay matched to the MoE variants for any ``moe_every``.
    """
    if moe_every < 1:
        raise ValueError(f"moe_every must be >= 1, got {moe_every}")
    return layer_idx % moe_every == moe_every - 1


@dataclass(frozen=True)
class FFNSpec:
    """Shape of one FFN layer for a variant (single source for modules and flops.py).

    A layer has ``n_shared`` always-on experts plus ``n_routed`` routed experts, all of width
    ``width``; each token activates the shared ones plus ``top_k`` routed ones.
    Dense / dense_x8 are ``n_shared=1, n_routed=0``.
    """

    variant: str
    width: int
    n_shared: int
    n_routed: int
    top_k: int

    @property
    def n_experts_total(self) -> int:
        return self.n_shared + self.n_routed

    @property
    def is_moe(self) -> bool:
        return self.n_routed > 0


def ffn_spec(variant: str, cfg: Mapping[str, Any], *, moe_layer: bool = True) -> FFNSpec:
    """Expert counts/width for ``variant`` (MASTER §4.3 table, PLAN §5, CONFIG_SCHEMA keys).

    ``moe_layer=False`` returns the plain dense spec used at non-MoE positions.
    Asserts the activated-width matching rule (activated FFN width == d_ff) for the MoE
    variants, as required by MASTER §4.5 for DeepSeekMoE.
    """
    d_ff = int(cfg["model"]["d_ff"])
    moe = cfg.get("moe", {})
    if not moe_layer or variant == "dense":
        return FFNSpec("dense", d_ff, 1, 0, 0)
    if variant == "dense_x8":
        mult = int(moe["dense_x8"]["width_mult"])
        return FFNSpec("dense_x8", mult * d_ff, 1, 0, 0)
    if variant == "switch":
        s = moe["switch"]
        return FFNSpec("switch", d_ff, 0, int(s["n_experts"]), 1)
    if variant == "gshard":
        g = moe["gshard"]
        div, k = int(g["width_divisor"]), int(g["top_k"])
        if d_ff % div:
            raise ValueError(f"d_ff={d_ff} not divisible by gshard.width_divisor={div}")
        spec = FFNSpec("gshard", d_ff // div, 0, int(g["n_experts"]), k)
    elif variant == "deepseek":
        s = moe["deepseek"]
        m = int(s["m"])
        if d_ff % m:
            raise ValueError(f"d_ff={d_ff} not divisible by deepseek.m={m}")
        spec = FFNSpec("deepseek", d_ff // m, int(s["n_shared"]), int(s["n_routed"]), int(s["k_routed"]))
    else:
        raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")
    # Activated FFN width must equal the dense model's (MASTER §4.3 / §4.5).
    act_width = (spec.n_shared + spec.top_k) * spec.width
    assert act_width == d_ff, (
        f"{variant}: activated width (n_shared+k)*w = {act_width} != d_ff = {d_ff}")
    return spec


def expert_capacity(n_tokens: int, n_experts: int, capacity_factor: float, top_k: int = 1) -> int:
    """Expert capacity C (Switch Eq. 3, generalised to top-k).

    Switch Eq. 3: ``C = floor(tokens_per_batch / n_experts * capacity_factor)``.
    For top-k (GShard) the expected load per expert is ``k*N/E``, so we use
    ``C = floor(k * N / E * cf)`` (GShard Algorithm 1 uses C = 2N/E for top-2). For k=1 this
    is exactly Switch Eq. 3. C is clamped to N, because one expert can receive at most one
    assignment per token. ``capacity_factor = NO_DROP`` (inf) gives C = N.
    """
    if math.isinf(capacity_factor):
        return n_tokens
    return min(n_tokens, int(math.floor(top_k * n_tokens / n_experts * capacity_factor)))


# ----------------------------------------------------------------------------------------
# Init (Switch Sec. 2.4, DeepSeekMoE Sec. 4.1)
# ----------------------------------------------------------------------------------------

@torch.no_grad()
def init_weight_(param: torch.Tensor, fan_in: int, model_cfg: Mapping[str, Any]) -> torch.Tensor:
    """Initialise ``param`` in place according to ``model_cfg["init"]``.

    * ``init: switch`` (default; Switch Sec. 2.4 "reduced initialization scale"): truncated
      normal, mean 0, ``sigma = sqrt(s / fan_in)`` with ``s = model_cfg["init_scale"]``
      (0.1), truncated at +-2 sigma (values outside are redrawn). As in TF/Mesh-TF
      ``truncated_normal``, sigma is the std of the normal BEFORE truncation; the realised
      std is about 0.88 sigma.
    * ``init: deepseek``: plain normal with std 0.006 (DeepSeekMoE Sec. 4.1). The paper does
      not mention truncation, so none is applied.

    ``fan_in`` is the input dimension of the matmul this weight performs (for a stacked
    expert tensor ``W_in[E, d, w]`` it is d, not E*d). Works for any shape and dtype.
    """
    scheme = model_cfg.get("init", "switch")
    if scheme == "switch":
        s = float(model_cfg.get("init_scale", 0.1))
        sigma = math.sqrt(s / fan_in)
        nn.init.trunc_normal_(param, mean=0.0, std=sigma, a=-2.0 * sigma, b=2.0 * sigma)
    elif scheme == "deepseek":
        nn.init.normal_(param, mean=0.0, std=DEEPSEEK_INIT_STD)
    else:
        raise ValueError(f"unknown model.init {scheme!r} (expected 'switch' or 'deepseek')")
    return param


# ----------------------------------------------------------------------------------------
# Weight decay grouping
# ----------------------------------------------------------------------------------------

_NO_WD_ATTR = "_no_weight_decay"


def mark_no_weight_decay(param: nn.Parameter) -> nn.Parameter:
    """Tag a parameter so :func:`build_param_groups` excludes it from weight decay.

    Used for the token embedding (model.py) and router weights (router.py). The tag is a
    Python attribute on the Parameter object; ``load_state_dict`` copies into the same
    object, so it survives checkpoint resume.
    """
    setattr(param, _NO_WD_ATTR, True)
    return param


def wants_weight_decay(param: nn.Parameter) -> bool:
    """True for weight matrices (ndim >= 2) that are not tagged no-decay."""
    return param.ndim >= 2 and not getattr(param, _NO_WD_ATTR, False)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW param groups: matrices get ``weight_decay``; norms, embedding, router get 0.

    ``named_parameters`` de-duplicates shared tensors, so the tied embedding/head appears once.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if wants_weight_decay(p) else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ----------------------------------------------------------------------------------------
# FFN factory
# ----------------------------------------------------------------------------------------

def build_ffn(variant: str, layer_idx: int, cfg: Mapping[str, Any]) -> nn.Module:
    """Construct the FFN for block ``layer_idx`` (0-based). The one place ``model.py`` uses.

    Honors ``model.moe_every`` via :func:`is_moe_layer`. Imports are lazy so this file has no
    import-time dependency on modules written later (ffn.py, moe_*.py).
    """
    model_cfg = cfg["model"]
    d_model, d_ff = int(model_cfg["d_model"]), int(model_cfg["d_ff"])
    moe_layer = is_moe_layer(layer_idx, int(model_cfg.get("moe_every", 1)))

    if variant == "dense" or (variant in ALL_VARIANTS and not moe_layer):
        from moe.ffn import DenseFFN
        return DenseFFN(d_model, d_ff, model_cfg)
    if variant == "dense_x8":
        from moe.ffn import DenseFFN
        return DenseFFN(d_model, ffn_spec("dense_x8", cfg).width, model_cfg)
    if variant == "switch":
        from moe.moe_switch import SwitchMoE
        return SwitchMoE(d_model, d_ff, cfg["moe"], model_cfg)
    if variant == "gshard":
        from moe.moe_gshard import GShardMoE
        return GShardMoE(d_model, d_ff, cfg["moe"], model_cfg)
    if variant == "deepseek":
        from moe.moe_deepseek import DeepSeekMoE
        return DeepSeekMoE(d_model, d_ff, cfg["moe"], model_cfg)
    raise ValueError(f"unknown variant {variant!r}; expected one of {ALL_VARIANTS}")


# ----------------------------------------------------------------------------------------
# Eval-time ablation switches
# ----------------------------------------------------------------------------------------

_ABLATION_KEYS = ("eval_capacity_factor", "n_disable_top_routed", "disable_shared", "k_routed_override")
_UNSET = object()


def set_eval_overrides(
    model: nn.Module,
    *,
    eval_capacity_factor: Any = _UNSET,
    n_disable_top_routed: Any = _UNSET,
    disable_top_routed_ratio: Any = _UNSET,
    disable_shared: Any = _UNSET,
    k_routed_override: Any = _UNSET,
) -> int:
    """Set ablation attributes on every FFN layer that supports them (contract item 4).

    Only the keywords you pass are changed. Pass the neutral value to clear one
    (``eval_capacity_factor=None``, ``n_disable_top_routed=0``, ``disable_shared=False``,
    ``k_routed_override=None``). ``disable_top_routed_ratio=r`` is converted per layer to
    ``floor(r * n_routed + 0.5)``. Raises ``ValueError`` if a requested switch is supported by
    no layer of ``model``. Returns the number of layers touched.
    """
    requested: dict[str, Any] = {}
    for key, val in (("eval_capacity_factor", eval_capacity_factor),
                     ("n_disable_top_routed", n_disable_top_routed),
                     ("disable_shared", disable_shared),
                     ("k_routed_override", k_routed_override)):
        if val is not _UNSET:
            requested[key] = val
    if disable_top_routed_ratio is not _UNSET:
        if "n_disable_top_routed" in requested:
            raise ValueError("pass n_disable_top_routed or disable_top_routed_ratio, not both")
        requested["n_disable_top_routed"] = ("ratio", float(disable_top_routed_ratio))

    applied = {k: False for k in requested}
    touched = 0
    for mod in model.modules():
        attrs = getattr(type(mod), "ABLATION_ATTRS", ())
        if not attrs:
            continue
        hit = False
        for key, val in requested.items():
            if key not in attrs:
                continue
            if isinstance(val, tuple) and val and val[0] == "ratio":
                val = int(math.floor(val[1] * mod.n_routed + 0.5))
            setattr(mod, key, val)
            applied[key] = hit = True
        touched += int(hit)
    missing = [k for k, ok in applied.items() if not ok]
    if missing:
        raise ValueError(f"no layer in the model supports eval override(s) {missing}")
    return touched
