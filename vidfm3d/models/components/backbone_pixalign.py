
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from vidfm3d.models.components.patchify import PatchifyTokenizer
from vidfm3d.vggt.layers.block import Block
from vidfm3d.vggt.layers.rope import PositionGetter, RotaryPositionEmbedding2D

logger = logging.getLogger(__name__)


class BackbonePA(nn.Module):
    """
    The Backbone that process the input video features.
    Follows the design of alternating attention in VGGT.

    Args:
        in_channels (int): Number of input channels.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        in_channels,
        embed_dim=512,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        gradient_checkpointing=False,
        with_mask=False,
    ):
        super().__init__()

        # Initialize rotary position embedding if frequency > 0
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.gradient_checkpointing = gradient_checkpointing

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Input projection with LayerNorm and Linear layers
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, embed_dim),
        )

        if with_mask:
            # Patchify tokenizer for foreground mask
            self.mask_tokenizer = PatchifyTokenizer(
                patch_size=14,
                in_chans=1,
                embed_dim=embed_dim // 4,
                normalize=False,
            )
            # Fuse layer for mask and video tokens
            self.fuse_layer = nn.Linear(embed_dim + embed_dim // 4, embed_dim)

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

    def forward(
        self,
        video_tokens: torch.Tensor,
        fg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            video_tokens (torch.Tensor): Input images with shape [B, S, C, H, W].
                B: batch size, S: sequence length, C: feature channels, H: height, W: width
            fg_mask (Optional[torch.Tensor]): Foreground mask with shape [B, S, 1, Hm, Wm].

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        assert (
            video_tokens.ndim == 5
        ), f"Input must be 5D tensor [B, S, C, H, W], got shape {video_tokens.shape}"
        B, S, C_in, H, W = video_tokens.shape

        if fg_mask is not None:
            assert (
                fg_mask.ndim == 5
            ), f"Foreground mask must be 5D tensor [B, S, 1, Hm, Wm], got shape {fg_mask.shape}"
            assert (
                fg_mask.shape[0] == B and fg_mask.shape[1] == S
            ), f"Foreground mask must have same batch and sequence length as video tokens, got {fg_mask.shape} vs {video_tokens.shape}"
            # Tokenize the foreground mask
            if fg_mask.shape[3] != H * 14 or fg_mask.shape[4] != W * 14:
                fg_mask = F.interpolate(
                    fg_mask.squeeze(2),
                    size=(H * 14, W * 14),
                    mode="bilinear",
                    align_corners=False,
                ).unsqueeze(
                    2
                )  # [B, S, 1, H*14, W*14]
            mask_tokens = self.mask_tokenizer(fg_mask)  # [B, S, C, H, W]

        # Reshape to [B*S, H*W, C]
        video_tokens = video_tokens.view(B * S, C_in, H * W).permute(0, 2, 1)
        _, P, C = video_tokens.shape

        # Input projection
        video_tokens = self.input_proj(video_tokens)  # [B*S, P, C]

        if fg_mask is not None:
            mask_tokens = mask_tokens.view(
                B * S, video_tokens.shape[-1] // 4, H * W
            ).permute(
                0, 2, 1
            )  # [B*S, P, C]
            video_tokens = torch.cat(
                [video_tokens, mask_tokens], dim=-1
            )  # [B*S, P, C+C/4]
            video_tokens = self.fuse_layer(video_tokens)  # [B*S, P, C]

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)  # (B*S, 1, C)
        register_token = slice_expand_and_flatten(
            self.register_token, B, S
        )  # (B*S, 4, C)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, video_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H, W, device=video_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.patch_start_idx, 2)
                .to(video_tokens.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    (
                        tokens,
                        frame_idx,
                        frame_intermediates,
                    ) = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    (
                        tokens,
                        global_idx,
                        global_intermediates,
                    ) = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_intermediates[i], global_intermediates[i]], dim=-1
                )
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.gradient_checkpointing:
                tokens = checkpoint(
                    self.frame_blocks[frame_idx], tokens, pos, use_reentrant=False
                )
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.gradient_checkpointing:
                tokens = checkpoint(
                    self.global_blocks[global_idx], tokens, pos, use_reentrant=False
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
