#!/usr/bin/env python3
"""
Extract one 81-frame feature window from a DL3DV scene with Open-Sora
and save one safetensor per requested layer**:

    <out_dir>/feature_t{start_idx}_layer{layer_id}.sft

Each file stores a single key  ``"feat"`` whose value is shaped
(T, H, W, C) and saved in **FP16**.

Typical call
------------
python -m features.opensora.extract_features \
       --scene-dir   vidfm3d/data/DL3DV/DL3DV-10K/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_4 \
       --data-sft    vidfm3d/data/DL3DV/DL3DV-processed/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3.sft \
       --out-dir     features/opensora/samples/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3 \
       --config-path features/opensora/configs/diffusion/inference/640px.py \
       --t 0.6 \
       --output-layers 15
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

# ------------------------------------------------- Open-Sora helpers -------
from .opensora_features import extract_feature

# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame images (`*.png` etc.)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_frames(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p) for p in paths]


def forward_opensora(
    frames: List[Image.Image],
    cfg_path: str,
    timestep: float,
    layer_ids: List[int],
    seed: int,
) -> Dict[int, torch.Tensor]:
    """
    Run a single forward pass *per layer* (keeps code simple & symmetric
    with the WAN wrapper).  Returns a dict {layer_id: feat(T,H,W,C)}.
    """
    feats: Dict[int, torch.Tensor]
    feats = extract_feature(
        frames=frames,
        layer_indices=layer_ids,
        timestep=timestep,
        seed=seed,
        config_path=cfg_path,
    )
    return feats


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Open-Sora feature extractor (one window, one file per layer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-dir", required=True, help="Image folder")
    parser.add_argument("--data-sft", required=True, help="Processed .sft file")
    parser.add_argument("--out-dir", required=True, help="Directory to save features")
    parser.add_argument("--image-ext", default="png", help="Image file extension")

    # Open-Sora-specific
    parser.add_argument(
        "--config-path",
        default="features/opensora/configs/diffusion/inference/640px.py",
        help="Open-Sora inference config",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.25,
        help="Noise-schedule timestep (0-1, higher → noisier)",
    )
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[15],
        help="Transformer block indices to extract",
    )
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args(argv)

    # --------------------------------------------------------------------- #
    # Logging
    # --------------------------------------------------------------------- #
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

    # --------------------------------------------------------------------- #
    # 0 . Determine which layers are still missing (resume support)
    # --------------------------------------------------------------------- #
    missing_layers = [
        l
        for l in args.output_layers
        if not os.path.exists(os.path.join(out_dir, f"feature_t{args.t}_layer{l}.sft"))
    ]

    if not missing_layers:
        logging.info("All requested layer files already exist – nothing to do.")
        return

    # --------------------------------------------------------------------- #
    # 1 . Pick the temporal window
    # --------------------------------------------------------------------- #
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())  # 0-based index ↔ first frame

    window_size = 129
    frame_paths = list_frames(args.scene_dir, ext=args.image_ext)
    if start_idx + window_size > len(frame_paths):
        raise RuntimeError(
            f"Need {start_idx+window_size} frames "
            f"but only {len(frame_paths)} exist in {args.scene_dir}"
        )

    window_paths = frame_paths[start_idx : start_idx + window_size]
    frames = load_frames(window_paths)

    # --------------------------------------------------------------------- #
    # 2 . Forward pass
    # --------------------------------------------------------------------- #
    feats = forward_opensora(
        frames,
        cfg_path=args.config_path,
        timestep=args.t,
        layer_ids=missing_layers,
        seed=args.seed,
    )

    # Warn about layers not returned (shouldn’t happen unless out-of-range)
    missing = set(missing_layers) - set(feats.keys())
    if missing:
        logging.warning(
            f"[warn] requested layers {sorted(missing)} not returned by Open-Sora"
        )

    # --------------------------------------------------------------------- #
    # 3 . Save one .sft per layer
    # --------------------------------------------------------------------- #
    saved = 0
    for layer_id, feat in feats.items():
        out_path = os.path.join(out_dir, f"feature_t{args.t}_layer{layer_id}.sft")
        save_file({"feat": feat.half()[:21]}, out_path)
        logging.debug(
            f"[Open-Sora] saved layer {layer_id} → {out_path} "
            f"shape {tuple(feat.shape)} (saved first 21)"
        )
        saved += 1

    if saved == 0:
        logging.error("[err] no layers saved – aborting")
    else:
        logging.info(f"[Open-Sora] done – {saved} layer files written to {out_dir}")


if __name__ == "__main__":
    main()
