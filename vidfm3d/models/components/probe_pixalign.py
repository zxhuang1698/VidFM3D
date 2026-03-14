
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vidfm3d.models.components.backbone_pixalign import BackbonePA
from vidfm3d.models.components.dpt_head import DPTHead
from vidfm3d.vggt.heads.camera_head import CameraHead


class ProbeModelPA(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        video_channels=1024,
        embed_dim=512,
        backbone_depth=8,
        gradient_checkpointing=False,
        dpt_dim=128,
        dpt_stage_channels=[128, 256, 512, 512],
        dpt_intermediate_layer_idx=[1, 3, 5, 7],
        active_heads=["depth", "point"],
        with_mask=False,
    ):
        super().__init__()

        self.backbone = BackbonePA(
            in_channels=video_channels,
            embed_dim=embed_dim,
            depth=backbone_depth,
            gradient_checkpointing=gradient_checkpointing,
            with_mask=with_mask,
        )
        self.camera_head = (
            CameraHead(dim_in=embed_dim * 2) if "camera" in active_heads else None
        )
        self.point_head = (
            DPTHead(
                dim_in=embed_dim * 2,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                features=dpt_dim,
                out_channels=dpt_stage_channels,
                intermediate_layer_idx=dpt_intermediate_layer_idx,
            )
            if "point" in active_heads
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=embed_dim * 2,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                features=dpt_dim,
                out_channels=dpt_stage_channels,
                intermediate_layer_idx=dpt_intermediate_layer_idx,
            )
            if "depth" in active_heads
            else None
        )

    def forward(
        self,
        video_features: torch.Tensor,
        video_shape: tuple,
        fg_mask: torch.Tensor = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            video_features (torch.Tensor): Video features with shape [B, S, C, Hf, Wf].
                B: batch size, S: sequence length, C: input channels, Hf: height, Wf: width
            video_shape (tuple): Shape of the input video (B, S, 3, H, W). For interpolation in DPT heads.
                B: batch size, S: sequence length, 3: RGB channels, H: raw height, W: raw width
            fg_mask (torch.Tensor, optional): Foreground mask with shape [B, S, 1, H, W]. Default is None.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        assert len(video_shape) == 5, "video_shape must be a 5D tensor (B, S, 3, H, W)"
        aggregated_tokens_list, patch_start_idx = self.backbone(video_features, fg_mask)
        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    n_patches=(video_features.shape[3], video_features.shape[4]),
                    frames_shape=video_shape,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    n_patches=(video_features.shape[3], video_features.shape[4]),
                    frames_shape=video_shape,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        return predictions
