"""Stacked expert weights shared by the loop and batched dispatch paths (PLAN §2, §7).

Every MoE layer keeps ALL of its experts in two tensors:

    W_in  [E, d, w]   (expert e's first matrix is W_in[e])
    W_out [E, w, d]

Expert e computes the same FFN as ``moe.ffn.DenseFFN`` of width w:

    FFN_e(x) = GELU(x @ W_in[e]) @ W_out[e]

The loop path calls :meth:`ExpertBank.expert_forward` once per expert on that expert's own
tokens. The batched path calls :meth:`ExpertBank.batched_forward` on a padded buffer
``[E, R, d]`` with two ``torch.bmm``. Both read the SAME parameters, so the loop==batched
tests compare math, not two parameter copies.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from moe.layer_api import init_weight_


class ExpertBank(nn.Module):
    """``n_experts`` GELU FFN experts of width ``width``, stored stacked. 2*d*w*E params."""

    def __init__(self, n_experts: int, d_model: int, width: int, model_cfg: Mapping[str, Any]) -> None:
        super().__init__()
        self.n_experts, self.d_model, self.width = n_experts, d_model, width
        self.W_in = nn.Parameter(torch.empty(n_experts, d_model, width))   # [E, d, w]
        self.W_out = nn.Parameter(torch.empty(n_experts, width, d_model))  # [E, w, d]
        # Switch Sec. 2.4 init: fan_in is the matmul input dim (d for W_in, w for W_out), not E*d.
        init_weight_(self.W_in, fan_in=d_model, model_cfg=model_cfg)
        init_weight_(self.W_out, fan_in=width, model_cfg=model_cfg)

    def expert_forward(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Expert e on its tokens: x [n, d] -> [n, d]."""
        h = F.gelu(x @ self.W_in[e])      # [n, w]
        return h @ self.W_out[e]          # [n, d]

    def batched_forward(self, buf: torch.Tensor) -> torch.Tensor:
        """All experts at once on a padded buffer: buf [E, R, d] -> [E, R, d].

        Row r of buf[e] is processed by expert e. Padding rows are zeros; they cost real
        FLOPs (that is the capacity-padding overhead E5 measures) and produce GELU(0)@W = 0.
        """
        h = F.gelu(torch.bmm(buf, self.W_in))   # [E, R, w]
        return torch.bmm(h, self.W_out)         # [E, R, d]
