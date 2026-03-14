#!/usr/bin/env python3
"""
Extract one 81-frame feature window from a DL3DV scene using Aether and
save **one safetensor per requested layer**:

    <out_dir>/feature_t{start_idx}_layer{layer_id}.sft

Each file holds a single key  ``"feat"``  whose value is shaped
**(T, H, W, C)** and stored in **FP16**.

Typical wrapper call
--------------------
python -m features.aether.extract_features \
       --scene-dir   vidfm3d/data/DL3DV/DL3DV-raw/DL3DV-10K/1K/<hash>/images_4 \
       --data-sft    vidfm3d/data/DL3DV/DL3DV-processed/1K/<hash>.sft \
       --out-dir     vidfm3d/features/aether/1K/<hash> \
       --t           499 \
       --output-layers 10 15 20 \
       --task        videogen
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
from typing import Dict, List

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .aether_feature import AetherFeaturizer, get_aether_featurizer

# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #


def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame PNG paths (frame_00001.png, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_frames(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p) for p in paths]


def forward_aether(
    model: AetherFeaturizer,
    frames: List[Image.Image],
    t: int,
    layer_ids: List[int],
) -> Dict[int, torch.Tensor]:
    # 1.  Build 2 clips, each 41 frames:
    #     clip_0 = [0, 1, 3, 5, ..., 77]
    #     clip_1 = [0, 2, 4, 6, ..., 78]
    clip_indices = [
        [0] + [i + 2 * k for k in range(40)] for i in range(1, 3)  # 1≤i≤2
    ]  # list[2][41]

    # get the frames for each clip
    clips = [[frames[i] for i in indices] for indices in clip_indices]

    all_features = []
    for i, clip in enumerate(clips):
        if len(clip) != 41:
            raise RuntimeError(f"Clip {i} has {len(clip)} frames, expected 41.")

        with torch.no_grad():
            feat = model.forward(
                video=clip,
                t=t,
                output_layer_indices=layer_ids,
            )
            all_features.append(feat)

    # 2. Merge the feature dictionaries from both clips:
    merged_features = {
        layer_id: torch.cat(
            [all_features[0][layer_id], all_features[1][layer_id]], dim=0
        )
        for layer_id in all_features[0].keys()
    }
    return merged_features


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

    # Aether-specific
    parser.add_argument("--t", type=int, default=499, choices=range(0, 1001))
    parser.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[15],
        help="Transformer block indices to extract",
    )
    parser.add_argument("--task", default="videogen", help="Task for Aether model")
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
    fname_prefix = "feature" if args.task == "videogen" else "feature_recon"
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
    window_size = 41 * 2 - 1

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
    aether = get_aether_featurizer()

    feats = forward_aether(
        aether,
        frames,
        t=args.t,
        layer_ids=missing_layers,
    )

    # Warn about missing layers
    missing = set(missing_layers) - set(feats.keys())
    if missing:
        logging.warning(
            f"[warn] requested layers {sorted(missing)} not returned by Aether model"
        )

    # --------------------------------------------------------------------- #
    # 3 . Save one .sft per layer                                           #
    # --------------------------------------------------------------------- #
    saved = 0
    for layer_id, feat in feats.items():
        out_path = os.path.join(
            out_dir, f"{fname_prefix}_t{args.t}_layer{layer_id}.sft"
        )
        save_file({"feat": feat.half()}, out_path)
        logging.debug(
            f"[WAN] saved layer {layer_id} → {out_path} " f"shape {tuple(feat.shape)}"
        )
        saved += 1

    if saved == 0:
        logging.error("[err] no layers saved - aborting")
    else:
        logging.info(f"[WAN] done - {saved} layer files written to {out_dir}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
