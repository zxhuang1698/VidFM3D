#!/usr/bin/env python3
"""
opensora_feature.py
===================

Single‑step **feature extractor** for the Open‑Sora video diffusion model.
It reproduces the *training‑time* noise injection formula and captures a
chosen FLUX transformer block’s hidden state in one forward pass.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from colossalai.utils import set_seed
from mmengine.config import Config
from opensora.models.mmdit.model import MMDiTModel
from opensora.utils.cai import get_booster, init_inference_environment

# Open‑Sora helpers --------------------------------------------------------- #
from opensora.utils.config import merge_args, parse_alias
from opensora.utils.logger import create_logger
from opensora.utils.misc import to_torch_dtype
from opensora.utils.sampling import (
    SamplingOption,
    get_res_lin_function,
    pack,
    prepare,
    prepare_models,
    sanitize_sampling_option,
    time_shift,
    unpack,
)
from PIL import Image

# -------------------------------------------------------------------------- #
# Utility functions
# -------------------------------------------------------------------------- #


def load_cfg(cfg_path: str, extra_cli: list[str]) -> Config:
    """Load config (.py / .yml) and apply CLI overrides + alias expansion."""
    cfg = Config.fromfile(cfg_path)
    if len(extra_cli) > 0:
        cfg = merge_args(cfg, extra_cli)  # emulate parse_configs()
    cfg.config_path = cfg_path

    # hard-coded for spatial compression
    if cfg.get("ae_spatial_compression", None) is not None:
        import os

        os.environ["AE_SPATIAL_COMPRESSION"] = str(cfg.ae_spatial_compression)
    return parse_alias(cfg)


@lru_cache(maxsize=None)  # key = (cfg_path, device, dtype)
def _get_models(cfg_path: str, device: str, dtype: torch.dtype):
    cfg = load_cfg(cfg_path, [])
    if torch.cuda.device_count() == 1:
        cfg.plugin = cfg.plugin_ae = "zero2"
        cfg.plugin_config = cfg.plugin_config_ae = {}

    model, model_ae, model_t5, model_clip, _ = prepare_models(
        cfg, device, dtype, offload_model=False  # cfg.get("offload_model", False)
    )
    return cfg, model, model_ae, model_t5, model_clip


def frames_to_tensor(frames: list[Image.Image], size: tuple[int, int]) -> torch.Tensor:
    """Convert a list of PIL images to a tensor of shape (1, C, T, H, W)."""
    frames = [frame.resize(size, Image.BICUBIC) for frame in frames]  # resize to (H,W)
    frames_tensor = torch.stack(
        [torchvision.transforms.functional.to_tensor(frame) for frame in frames], dim=0
    )  # (T,C,H,W)
    normalize = torchvision.transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
    )
    frames_tensor = normalize(frames_tensor)  # normalize to [-1,1]
    frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (C,T,H,W)
    return frames_tensor.unsqueeze(0)  # add batch dimension: (1,C,T,H,W)


def prepare_visual_condition_causal(
    x: torch.Tensor, model_ae: torch.nn.Module
) -> torch.Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (torch.Tensor): The input video tensor.
        model_ae (torch.nn.Module): The video encoder module.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = 16
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = torch.zeros(B, 1, T, H, W).to(
        x.device, x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = torch.zeros(B, C, T, H, W).to(x.device, x.dtype)
    masks[:, :, 0, :, :] = 1
    x_0 = model_ae.encode(x)
    # condition: encode the image only
    latent[:, :, :1, :, :] = model_ae.encode(x[:, :, :1, :, :])

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = torch.cat((masks, latent), dim=1)
    return x_0, cond


@torch.no_grad()
def pca_to_rgb(
    feat_a: torch.Tensor,  # (Hf,Wf,C)
    out_h: int = 480,
    out_w: int = 848,
) -> list[np.ndarray]:
    """
    Robust PCA → RGB visualiser for a pair of VFM feature maps.
    """
    T, Ha, Wa, C = feat_a.shape
    # --------- flatten THEN concatenate -------------------------------
    feat_a = feat_a.reshape(-1, C).float()
    f = feat_a - feat_a.mean(0, keepdim=True)  # centre

    # --------- PCA (top-3 right singular vectors) ----------------------
    _, _, Vt = torch.linalg.svd(f, full_matrices=False)  # (C,C)
    proj = f @ Vt[:3].T  # (N,3)

    # --------- robust per-channel scaling (5-95 %) ---------------------
    q05 = torch.quantile(proj, 0.01, dim=0, keepdim=True)
    q95 = torch.quantile(proj, 0.99, dim=0, keepdim=True)
    proj = (proj - q05) / (q95 - q05 + 1e-8)
    proj = proj.clamp_(0, 1)

    # --------- reshape back -------------------------------------------
    proj = proj.reshape(T, Ha, Wa, 3)

    # --------- upsample -----------------------------------------------
    imgs = []
    for i in range(T):
        img = proj[i].permute(2, 0, 1)[None]  # (1,3,Hf,Wf)
        img = torch.nn.functional.interpolate(img, size=(out_h, out_w), mode="nearest")[
            0
        ]
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        imgs.append(img)
    return imgs


# -------------------------------------------------------------------------- #
# Main
# -------------------------------------------------------------------------- #


@torch.inference_mode()
def extract_feature(
    frames: list[Image.Image],
    layer_indices: list[int] = [10],
    timestep: float = 0.25,
    seed: int = 42,
    config_path: str = "configs/diffusion/inference/640px.py",
):
    # Config & device ---------------------------------------------------- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype("bf16")  # keep whatever logic you prefer
    set_seed(seed)
    init_inference_environment()
    logger = create_logger()

    # Build models ------------------------------------------------------- #
    model: MMDiTModel
    cfg, model, model_ae, model_t5, model_clip = _get_models(config_path, device, dtype)
    model.eval()
    model_ae.eval()
    model_t5.eval()
    model_clip.eval()
    # logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    booster = get_booster(cfg)
    booster_ae = get_booster(cfg, ae=True)
    plugin_config = cfg.get("plugin_config", {})
    seq_align = plugin_config.get("sp_size", 1)

    if booster:
        model, _, _, _, _ = booster.boost(model=model)
        model = model.unwrap()
    if booster_ae:
        model_ae, _, _, _, _ = booster_ae.boost(model=model_ae)
        model_ae = model_ae.unwrap()
    model.eval()
    model_ae.eval()
    model_t5.eval()
    model_clip.eval()

    for layer_idx in layer_indices:
        if not (0 <= layer_idx < len(model.single_blocks)):
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {len(model.single_blocks)})"
            )

    # Data prep ---------------------------------------------------------- #
    opt = sanitize_sampling_option(SamplingOption(**cfg.sampling_option))
    # video.shape: [1, 3, 129, 480, 848]
    video = frames_to_tensor(frames, (opt.width, opt.height)).to(device).to(dtype)
    # x_0.shape: [1, 16, 21, 60, 106]
    x_0, cond = prepare_visual_condition_causal(video, model_ae)
    cond = pack(cond, patch_size=cfg.get("patch_size", 2))
    inp = dict()
    # == prepare condition ==
    inp["cond"] = cond
    inp_ = prepare(
        model_t5,
        model_clip,
        x_0,
        prompt="",
        seq_align=seq_align,
        patch_size=cfg.get("patch_size", 2),
    )
    inp.update(inp_)

    # == prepare timestep ==
    # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024px
    shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
    # add temporal influence
    shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
    assert 0 <= timestep <= 1, f"timestep {timestep} out of range (0,1]"
    # larger t, more noise
    t = torch.tensor([timestep], device=device, dtype=dtype).expand(x_0.shape[0])  # (B)
    t = time_shift(shift_alpha, t).to(dtype)
    x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))

    # == prepare input ==
    sigma_min = cfg.get("sigma_min", 1e-5)
    x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
    t_rev = 1 - t
    x_t = (
        t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
    )
    inp["img"] = x_t
    inp["timesteps"] = t.to(dtype)
    inp[
        "guidance"
    ] = None  # torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)

    # Capture activations -------------------------------------------- #
    # 1. prepare a place to stash activations
    saved_activations = {}

    # 2. make a hook factory that captures idx
    def make_hook(idx):
        def hook(module, inputs, output):
            # detach so you don’t keep the graph
            saved_activations[idx] = output.detach()[0, 512:].view(-1, 30, 53, 3072)

        return hook

    # 3. register the hook on the SingleStreamBlock
    handles = []
    for layer_idx in layer_indices:
        handle = model.single_blocks[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        handles.append(handle)

    # --- forward pass ---
    out = model(**inp)
    # ------------------------------------

    # afterwards you can grab:
    # print("Block output:", saved_activations[layer_idx].shape)
    for handle in handles:
        handle.remove()

    return saved_activations


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    frame_path = "/path/to/DL3DV/scene/images_4"
    frames = [
        Image.open(Path(frame_path) / f) for f in sorted(Path(frame_path).glob("*.png"))
    ][:129]
    features = extract_feature(
        frames=frames,
        layer_indices=[10],
        timestep=0.6,
        config_path="configs/diffusion/inference/640px.py",
    )[
        10
    ]  # get the feature for layer 10

    # visualise the feature
    pca_imgs = pca_to_rgb(features, out_h=480, out_w=848)
    os.makedirs("samples/viz", exist_ok=True)
    for i, pca_img in enumerate(pca_imgs):
        frame_idx = max((i - 1) * 4 + 1, 0)
        frame_rgb = frames[frame_idx].resize((848, 480), Image.BICUBIC)
        pca_rgb = Image.fromarray(pca_img)
        # stack vertically
        combined = Image.new("RGB", (848, 960))
        combined.paste(frame_rgb, (0, 0))
        combined.paste(pca_rgb, (0, 480))
        combined.save(f"samples/viz/feat_layer10_t0.6_{i:02d}.png")
