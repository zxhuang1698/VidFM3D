#!/usr/bin/env python3
"""
Extract one 81-frame feature window from a DL3DV scene using Wan-T2V and
save **one safetensor per requested layer**:

    <out_dir>/feature_t{start_idx}_layer{layer_id}.sft

Each file holds a single key  ``"feat"``  whose value is shaped
**(T, H, W, C)** and stored in **FP16**.

Typical wrapper call
--------------------
python -m features.wan.extract_features \
       --scene-dir   vidfm3d/data/DL3DV/DL3DV-10K/1K/<hash>/images_4 \
       --data-sft    vidfm3d/data/DL3DV/DL3DV-processed/1K/<hash>.sft \
       --out-dir     vidfm3d/features/wan/1K/<hash> \
       --model-id    Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
       --prompt      "" \
       --t           1 \
       --output-layers 10 15 20
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .wan_feature import WanFeaturizer, get_wan_featurizer
from .wan_feature_i2v import WanFeaturizerI2V, get_wan_featurizer_i2v

# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #


def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame PNG paths (frame_00001.png, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_frames(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p) for p in paths]


def forward_wan(
    model: WanFeaturizer | WanFeaturizerI2V,
    frames: List[Image.Image],
    prompt: str,
    t: int,
    layer_ids: List[int],
    ensemble: int,
) -> Dict[int, torch.Tensor]:
    with torch.no_grad():
        return model.forward(
            video=frames,
            prompt=prompt,
            t=t,
            output_layer_indices=layer_ids,
            ensemble_size=ensemble,
        )


def reshape_to_t_h_w_c(raw: torch.Tensor) -> torch.Tensor:
    """
    Wan (1, N_tokens, C) → (T, H, W, C)

    With 832x480 inputs Wan produces (H_p=30, W_p=52) spatial tokens per
    *output* frame and temporal stride = 4 ⇒ T = 80/4+1 = 21 frames.

    """
    t_tokens, h_tokens, w_tokens = 21, 30, 52
    assert raw.ndim == 3, f"Expected 3D tensor, got {raw.shape}"
    assert raw.shape[0] == 1, f"Expected batch size 1, got {raw.shape[0]}"
    assert (
        raw.shape[1] == t_tokens * h_tokens * w_tokens
    ), f"Expected {t_tokens * h_tokens * w_tokens} tokens, got {raw.shape[1]}"
    return (
        raw.squeeze(0)  # (N_tokens, C)
        .reshape(t_tokens, h_tokens, w_tokens, raw.shape[-1])
        .contiguous()
    )


# ---------------------------------------------------------------------------- #
# Main                                                                         #
# ---------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="WAN feature extractor (one window, one file per layer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-dir", required=True, help="Image folder")
    parser.add_argument("--data-sft", required=True, help="Processed .sft file")
    parser.add_argument("--out-dir", required=True, help="Directory to save features")
    parser.add_argument("--image-ext", default="png", help="Image file extension")

    # Wan-specific
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--t", type=int, default=499, choices=range(0, 1001))
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[15],
        help="Transformer block indices to extract",
    )
    parser.add_argument("--ensemble", type=int, default=1)
    args, unknown = parser.parse_known_args(argv)

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}: [{levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    if unknown:
        logging.debug(f"[warn] ignored unknown args: {unknown}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 0 . Determine which layers are still missing (resume support)      #
    # ------------------------------------------------------------------ #
    fname_prefix = "feature_i2v" if "I2V" in args.model_id else "feature"
    if args.model_id.endswith("14B-Diffusers") and "I2V" not in args.model_id:
        fname_prefix = "feature_t2v_14b"
    missing_layers = [
        l
        for l in args.output_layers
        if not os.path.exists(
            os.path.join(out_dir, f"{fname_prefix}_t{args.t}_layer{l}.sft")
        )
    ]

    if not missing_layers:
        logging.info("All requested layer files already exist - nothing to do.")
        return

    # --------------------------------------------------------------------- #
    # 1 . Pick the temporal window                                          #
    # --------------------------------------------------------------------- #
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())  # 0-based index ↔ frame_00001.png
    window_size = 81

    frame_paths = list_frames(args.scene_dir, ext=args.image_ext)
    if start_idx + window_size > len(frame_paths):
        raise RuntimeError(
            f"Need {start_idx+window_size} frames but only "
            f"{len(frame_paths)} exist in {args.scene_dir}"
        )

    window_paths = frame_paths[start_idx : start_idx + window_size]
    frames = load_frames(window_paths)

    # --------------------------------------------------------------------- #
    # 2 . Forward pass                                                     #
    # --------------------------------------------------------------------- #
    logging.debug(f"[WAN] loading model {args.model_id}")
    if "T2V" in args.model_id:
        wan = get_wan_featurizer(model_id=args.model_id)
    else:
        wan = get_wan_featurizer_i2v(model_id=args.model_id)

    feats = forward_wan(
        wan,
        frames,
        prompt=args.prompt,
        t=args.t,
        layer_ids=missing_layers,
        ensemble=args.ensemble,
    )

    # Warn about missing layers
    missing = set(missing_layers) - set(feats.keys())
    if missing:
        logging.warning(
            f"[warn] requested layers {sorted(missing)} not returned by WAN model"
        )

    # --------------------------------------------------------------------- #
    # 3 . Save one .sft per layer                                           #
    # --------------------------------------------------------------------- #
    saved = 0
    for layer_id, raw_feat in feats.items():
        reshaped = reshape_to_t_h_w_c(raw_feat)
        out_path = os.path.join(
            out_dir, f"{fname_prefix}_t{args.t}_layer{layer_id}.sft"
        )
        save_file({"feat": reshaped.half()}, out_path)
        logging.debug(
            f"[WAN] saved layer {layer_id} → {out_path} "
            f"shape {tuple(reshaped.shape)}"
        )
        saved += 1

    if saved == 0:
        logging.error("[err] no layers saved - aborting")
    else:
        logging.info(f"[WAN] done - {saved} layer files written to {out_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
