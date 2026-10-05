"""Decoder-only, pre-norm transformer backbone shared by all variants (MASTER §4.2, PLAN §2).

Only the FFN differs between variants; it is always created by
``layer_api.build_ffn(variant, layer_idx, cfg)`` and called as ``y, aux = ffn(x, return_routing)``.
The model never looks inside routing. CE and aux loss are returned separately; ``train.py``
computes ``loss = ce + aux`` and logs both (MASTER §4.8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from moe import flops, layer_api
from moe.layer_api import MoEAux, init_weight_


class RMSNorm(nn.Module):
    """RMSNorm with a learned scale and no bias: ``x / rms(x) * weight`` (PLAN §2)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # normalise in fp32 for stability under autocast
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) tables for ``seq_len`` positions.

    cos/sin have shape [seq_len, head_dim]; they are non-persistent buffers, so they are
    not in the state_dict and not counted as parameters.

    Convention (rotate-half, as in GPT-NeoX/LLaMA): the head dim is split into two halves
    (x1, x2); frequency i rotates the pair (x1[i], x2[i]), and
    ``rotate_half(x) = cat(-x2, x1)``, so  out = x * cos + rotate_half(x) * sin.
    The angle table is cat(freqs, freqs) so both halves of a pair share one angle.
    """

    def __init__(self, head_dim: int, seq_len: int, base: float) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.outer(pos, inv_freq)               # [T, head_dim/2]
        angles = torch.cat([angles, angles], dim=-1)      # [T, head_dim]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, head_dim] -> same shape, rotated by position."""
        T = x.shape[-2]
        cos = self.cos[:T].to(x.dtype)
        sin = self.sin[:T].to(x.dtype)
        return x * cos + self.rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE. Wq, Wk, Wv, Wo: 4*d^2 params, no biases."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        d, H = int(model_cfg["d_model"]), int(model_cfg["n_heads"])
        assert d % H == 0, "d_model must be divisible by n_heads"
        self.n_heads, self.head_dim = H, d // H
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        for lin in (self.wq, self.wk, self.wv, self.wo):
            init_weight_(lin.weight, fan_in=d, model_cfg=model_cfg)
        self.rope = RotaryEmbedding(self.head_dim, int(model_cfg["seq_len"]), float(model_cfg["rope_base"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        def split(t: torch.Tensor) -> torch.Tensor:  # [B,T,d] -> [B,H,T,hd]
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        q, k = self.rope.apply(q), self.rope.apply(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    """Pre-norm block: ``h += attn(norm1(h)); h += ffn(norm2(h))`` (PLAN §2)."""

    def __init__(self, variant: str, layer_idx: int, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        self.norm1 = RMSNorm(int(m["d_model"]), float(m["norm_eps"]))
        self.attn = CausalSelfAttention(m)
        self.norm2 = RMSNorm(int(m["d_model"]), float(m["norm_eps"]))
        self.ffn = layer_api.build_ffn(variant, layer_idx, cfg)  # the only way FFNs are created

    def forward(self, h: torch.Tensor, return_routing: bool = False) -> tuple[torch.Tensor, MoEAux]:
        h = h + self.attn(self.norm1(h))
        y, aux = self.ffn(self.norm2(h), return_routing)
        return h + y, aux


@dataclass
class ModelOutput:
    """Result of ``MoETransformer.forward``.

    logits: [B, T, V] (dtype follows autocast). ce_loss: fp32 scalar or None without targets.
    aux_loss: 0-dim fp32, sum over layers of ``aux.aux_loss`` (already scaled by alpha).
    aux_per_layer: list of per-layer aux_loss tensors. layer_aux: list of MoEAux, one per block.
    """

    logits: torch.Tensor
    ce_loss: Optional[torch.Tensor]
    aux_loss: torch.Tensor
    aux_per_layer: list[torch.Tensor]
    layer_aux: list[MoEAux]


class MoETransformer(nn.Module):
    """Decoder-only LM; ``cfg["variant"]`` selects the FFN type (MASTER §4.2)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = cfg["variant"]
        m = cfg["model"]
        self.vocab_size = int(cfg["data"]["vocab_size"])
        self.d_model = int(m["d_model"])
        self.seq_len = int(m["seq_len"])
        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        init_weight_(self.tok_emb.weight, fan_in=self.d_model, model_cfg=m)
        layer_api.mark_no_weight_decay(self.tok_emb.weight)  # tied with the head; no WD
        self.blocks = nn.ModuleList([Block(self.variant, i, cfg) for i in range(int(m["n_layers"]))])
        self.final_norm = RMSNorm(self.d_model, float(m["norm_eps"]))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                return_routing: bool = False) -> ModelOutput:
        h = self.tok_emb(idx)
        layer_aux: list[MoEAux] = []
        for block in self.blocks:
            h, aux = block(h, return_routing)
            layer_aux.append(aux)
        h = self.final_norm(h)
        logits = F.linear(h, self.tok_emb.weight)  # tied output head

        ce = None
        if targets is not None:
            # CE in fp32 (MASTER §4.8). Not added to aux here; train.py does loss = ce + aux.
            ce = F.cross_entropy(logits.float().view(-1, self.vocab_size), targets.reshape(-1))
        aux_per_layer = [a.aux_loss for a in layer_aux]
        aux_total = torch.stack([a.float() for a in aux_per_layer]).sum()
        return ModelOutput(logits, ce, aux_total, aux_per_layer, layer_aux)

    # ------------------------------------------------------------------ helpers
    def num_params(self) -> int:
        """Total parameters (tied embedding counted once)."""
        return sum(p.numel() for p in self.parameters())

    def param_table_str(self) -> str:
        """Analytic param/FLOP table for all variants under this config (startup printout)."""
        return flops.format_param_table(flops.param_table(self.cfg))

    def verify_param_count(self) -> int:
        """Assert the real parameter count equals flops.py's analytic ``total_params``."""
        expected = flops.param_row(self.variant, self.cfg)["total_params"]
        actual = self.num_params()
        assert actual == expected, (
            f"{self.variant}: real params {actual:,} != analytic {expected:,}")
        return actual

    def configure_optimizer(self, train_cfg: dict) -> torch.optim.AdamW:
        """AdamW; decay only matrices, not norms/embedding/router (PLAN §4, layer_api item 6)."""
        groups = layer_api.build_param_groups(self, float(train_cfg["weight_decay"]))
        on_cuda = next(self.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=float(train_cfg["peak_lr"]),
                                 betas=tuple(train_cfg["betas"]), fused=on_cuda)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling; context is cropped to the last ``seq_len`` tokens."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.seq_len:]).logits[:, -1, :].float()
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
