"""Dense FFN expert: the building block (MASTER §3).

``DenseFFN`` is one two-matrix GELU feed-forward network, ``y = GELU(x @ W_in) @ W_out``.
It is used (a) as the plain dense baseline FFN, (b) as ``dense_x8`` with width
``width_mult * d_ff`` (``layer_api.build_ffn``), and (c) at non-MoE positions when
``model.moe_every > 1``. The MoE variants stack many of these experts (width ``w``) in one
tensor and add routing; this class is the un-routed reference for their math.

Contract: ``moe/layer_api.py`` items 2, 5, 7. No biases, no pre-norm, no residual add (the
Block does those), no dropout (PLAN §2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from moe.layer_api import MoEAux, init_weight_


class DenseFFN(nn.Module):
    """Dense FFN expert, the building block (MASTER §3). ``2 * d_model * d_ff`` parameters.

    Args:
        d_model: input / output width.
        d_ff: hidden width (``model.d_ff``, or ``width_mult * model.d_ff`` for dense_x8).
        model_cfg: the ``model`` config section (read by ``init_weight_``).
    """

    def __init__(self, d_model: int, d_ff: int, model_cfg: dict) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_in = nn.Parameter(torch.empty(d_model, d_ff))
        self.W_out = nn.Parameter(torch.empty(d_ff, d_model))
        init_weight_(self.W_in, fan_in=d_model, model_cfg=model_cfg)   # input dim of W_in is d_model
        init_weight_(self.W_out, fan_in=d_ff, model_cfg=model_cfg)     # input dim of W_out is d_ff

    def forward(self, x: torch.Tensor, return_routing: bool = False) -> tuple[torch.Tensor, MoEAux]:
        """x [B, T, d_model] -> (y [B, T, d_model], empty MoEAux). ``return_routing`` is ignored."""
        h = F.gelu(x @ self.W_in)       # [B, T, d_ff]
        y = h @ self.W_out              # [B, T, d_model]
        return y, MoEAux.empty(x.device)
