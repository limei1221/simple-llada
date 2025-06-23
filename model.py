# Modified from https://github.com/ML-GSAI/SMDM/blob/main/lit_gpt/diffmodel.py


import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RoPECache = Tuple[torch.Tensor, torch.Tensor]


def _build_rope_cache(
    seq_len: int, dim: int, device: torch.device, base: int = 10_000
) -> RoPECache:
    """Pre-computes cos/sin tables used by RoPE.

    Returned tensors have shape (seq_len, dim).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
    return emb.cos(), emb.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary embeddings *in-place*.

    Args:
        x:   (..., D) where D is *even*.
        cos: (T, D)
        sin: (T, D)
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[..., ::2], sin[..., ::2]  # (T, D/2)

    x_rope_even = x1 * cos - x2 * sin
    x_rope_odd = x1 * sin + x2 * cos
    return torch.stack((x_rope_even, x_rope_odd), dim=-1).flatten(-2)


class MLP(nn.Module):
    """GELU MLP used in the original GPT and LLaMA models."""

    def __init__(self, n_embd: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, n_embd, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return self.fc2(F.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    """Multi-head self-attention with Group Query Attention (GQA) and optional RoPE."""

    def __init__(
        self, n_embd: int, n_head: int, n_query_groups: int, bias: bool
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_dim = n_embd // n_head
        self.scale = 1 / math.sqrt(self.head_dim)

        assert n_head % n_query_groups == 0
        self.n_kv_head = n_head // n_query_groups

        self.q_proj = nn.Linear(n_embd, n_head * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, self.n_kv_head * self.head_dim, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x: torch.Tensor, rope: RoPECache, is_causal: bool) -> torch.Tensor:  # (B, T, C)
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)  # (B,T,H,D)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)  # (B,T,KV,D)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)  # (B,T,KV,D)

        q = q.transpose(1, 2)  # (B,H,T,D)
        k = k.transpose(1, 2)  # (B,KV,T,D)
        v = v.transpose(1, 2)  # (B,KV,T,D)

        cos, sin = rope
        cos, sin = cos[:T], sin[:T]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        k = k.repeat_interleave(self.n_query_groups, dim=1)  # (B,H,T,D)
        v = v.repeat_interleave(self.n_query_groups, dim=1)  # (B,H,T,D)

        # attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        # attn = attn.softmax(dim=-1)
        # out = attn @ v  # (B,H,T,D)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.scale, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B,T,C)
        return self.proj(out)


class Block(nn.Module):
    """Transformer block: norm → attn → residual → norm → MLP → residual."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_query_groups: int,
        intermediate_size: int,
        norm_eps: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, eps=norm_eps)
        self.attn = SelfAttention(n_embd, n_head, n_query_groups, bias)
        self.ln2 = nn.LayerNorm(n_embd, eps=norm_eps)
        self.mlp = MLP(n_embd, intermediate_size, bias)

    def forward(self, x: torch.Tensor, rope: RoPECache, is_causal: bool) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope, is_causal)
        x = x + self.mlp(self.ln2(x))
        return x


class TransEncoder(nn.Module):
    """A *very* small LLM to demonstrate masked-token diffusion training."""

    def __init__(self, config, is_causal: bool = False) -> None:
        super().__init__()
        self.config = config
        self.is_causal = is_causal

        # +1 for the special *mask* token
        self.embed_tokens = nn.Embedding(config.vocab_size + 1, config.n_embd)
        self.blocks = nn.ModuleList(
            Block(
                n_embd=config.n_embd,
                n_head=config.n_head,
                n_query_groups=config.n_query_groups,
                intermediate_size=config.intermediate_size,
                norm_eps=config.norm_eps,
                bias=config.bias,
            )
            for _ in range(config.n_layer)
        )
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.register_buffer("_rope_cos", None, persistent=False)
        self.register_buffer("_rope_sin", None, persistent=False)

        self.apply(self._init_weights)

    @property
    def device(self):
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        self._maybe_create_rope_cache(
            seq_len=T, device=idx.device, dtype=self.embed_tokens.weight.dtype
        )

        x = self.embed_tokens(idx)  # (B,T,C)
        rope = (self._rope_cos, self._rope_sin)
        for block in self.blocks:
            x = block(x, rope, self.is_causal)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        return logits

    def _maybe_create_rope_cache(
        self, *, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._rope_cos is None
            or self._rope_cos.size(0) < seq_len
            or self._rope_cos.device != device
        ):
            cos, sin = _build_rope_cache(seq_len, self.config.head_dim, device)
            self._rope_cos = cos.to(dtype)
            self._rope_sin = sin.to(dtype)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
