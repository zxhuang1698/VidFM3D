# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    A cross-attention module that takes query tokens (x) and context tokens (context),
    then attends 'x' to 'context' (keys/values). Similar structure to Attention, but
    we separate Q from X, and K,V from context.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
    ):
        """
        Args:
            dim: Dimension of the query tokens (x).
            context_dim: Dimension of the context tokens. Defaults to `dim` if None.
            num_heads, qkv_bias, etc.: as in your Attention class.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        if context_dim is None:
            context_dim = dim

        # We'll define separate linear layers for Q (from x) and K,V (from context).
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        # Norms for Q,K if qk_norm is True
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x is shape [B, N_q, dim], context is shape [B, N_ctx, context_dim].
        """
        B, N_q, C = x.shape
        _, N_ctx, C_ctx = context.shape

        # Project Q from x, and K,V from context
        q = self.q_proj(x)  # [B, N_q, dim]
        kv = self.kv_proj(context)  # [B, N_ctx, 2*dim]
        q = q.reshape(B, N_q, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # [B, heads, N_q, head_dim]
        kv = kv.reshape(B, N_ctx, 2, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )  # [2, B, heads, N_ctx, head_dim]
        k, v = kv.unbind(0)  # [B, heads, N_ctx, head_dim]

        # Optionally apply norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # scaled dot‐product attention
        if self.fused_attn:
            # Torch 2.x fused sdp
            x = F.scaled_dot_product_attention(
                q,  # [B, heads, N_q, head_dim]
                k,  # [B, heads, N_ctx, head_dim]
                v,  # [B, heads, N_ctx, head_dim]
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            # manual attn
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, heads, N_q, N_ctx]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # [B, heads, N_q, head_dim]

        # Merge heads
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
