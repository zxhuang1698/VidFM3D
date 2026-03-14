import torch
import torch.nn as nn

from vidfm3d.utils.typing import *
from vidfm3d.vggt.layers import PatchEmbed


class PatchifyTokenizer(nn.Module):
    def __init__(
        self, img_size=224, patch_size=14, in_chans=3, embed_dim=768, normalize=True
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.normalize = normalize
        if normalize:
            self.register_buffer(
                "image_mean",
                torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "image_std",
                torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )

    def forward(
        self,
        images: Float[Tensor, "B *N C H W"],
    ) -> Float[Tensor, "B *N Ct Nt"]:
        if images.ndim == 4:  # pack singleton sequence dim
            images = images.unsqueeze(1)
            packed = True
        else:
            packed = False
        B, S, C, H, W = images.shape

        if self.normalize:
            # check range of images
            if images.min() < 0 or images.max() > 1:
                raise ValueError(
                    f"Image values should be in range [0, 1], but got min: {images.min()}, max: {images.max()}"
                )
            images = (images - self.image_mean) / self.image_std

        # (B,S,C,H,W) ➜ (B*S,C,H,W) ➜ patchify ➜ (B*S,N,D)
        tokens = self.patch_embed(images.reshape(B * S, C, H, W))

        # reshape back to spatial maps
        Ht, Wt = (
            H // self.patch_embed.patch_size[0],
            W // self.patch_embed.patch_size[1],
        )
        tokens = tokens.reshape(B, S, Ht, Wt, self.patch_embed.embed_dim).permute(
            0, 1, 4, 2, 3
        )  # (B,S,D,Ht,Wt)

        if packed:
            tokens = tokens.squeeze(1)  # (B,D,Ht,Wt)
        return tokens
