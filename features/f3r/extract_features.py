#!/usr/bin/env python3
"""
Extract Fast3R spatial features:

Saves **one safetensor per requested layer**:

    <out_dir>/feature_l{L}.sft   # key "feat": (T, 18, 32, 1024) FP16

Typical wrapper call (run_co3d):
--------------------------------
python -m features.run_co3d \
       --vfm vidfm3d \
       --subset all \
       --output-layers 14 19 24
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

# add sys.path to import f3r
import sys
import time
from functools import lru_cache
from typing import List, Sequence

import torch
from safetensors.torch import load_file, save_file

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from f3r.dust3r.inference_multiview import inference
from f3r.dust3r.utils.image import load_images
from f3r.models.vidfm3d import Fast3R

# # add sys.path to import vidfm3d
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Constants / Hardcoding to match your demo                                    #
# --------------------------------------------------------------------------- #

LONG_SIDE = 512
HTOK = 288 // 16  # 18
WTOK = 512 // 16  # 32
C = 1024  # channel dim in your demo reshape


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame paths (frame_00001.<ext>, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


@lru_cache(maxsize=None)
def load_model(model_id: str, device: str) -> Fast3R:
    """Load Fast3R once (cached across calls)."""
    logging.debug("Loading model %s on %s …", model_id, device)
    model = Fast3R.from_pretrained(model_id).to(device).eval()
    return model


@torch.no_grad()
def vidfm3d_forward(filelist: Sequence[str], model: Fast3R, device: torch.device):
    """Exactly your demo path: load_images → inference(bfloat16) → return."""
    imgs = load_images(filelist, size=LONG_SIDE, verbose=False)
    ret = inference(
        imgs,
        model,
        device,
        dtype=torch.bfloat16,  # stay hardcoded
        verbose=False,
        profiling=False,
    )
    # Handle accidentally returning (features, profiling)
    if isinstance(ret, tuple):
        ret = ret[0]
    return ret  # expected: list[Tensor] where each is (T*HTOK*WTOK*C) compatible


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Fast3R feature extractor (demo-hardcoded)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene-dir", required=True, help="Directory with frame_*.png|jpg")
    p.add_argument(
        "--data-sft",
        required=True,
        help="Processed .sft (contains 'start_idx' and 'images')",
    )
    p.add_argument("--out-dir", required=True, help="Where to save features")
    p.add_argument("--image-ext", default="png", help="Image file extension")
    p.add_argument("--model-id", default="jedyang97/Fast3R_ViT_Large_512")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-layers",
        nargs="+",
        type=int,
        default=[14, 19, 24],  # indices into the returned 'features' list
        help="Which indices from the returned features[] to save.",
    )
    args, unknown = p.parse_known_args(argv)

    # Logging like DINO/WAN
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    if unknown:
        logging.warning("[warn] ignored unknown args: %s", unknown)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- read meta ----------
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())
    T = int(meta["images"].shape[0])

    # ---------- collect frame paths ----------
    all_paths = list_frames(args.scene_dir, ext=args.image_ext)
    filelist = all_paths[start_idx : start_idx + T]
    if len(filelist) != T:
        raise RuntimeError(
            f"Scene folder missing frames: need {T} from index {start_idx}, found {len(filelist)}."
        )

    # ---------- resume check ----------
    target = {
        L: os.path.join(args.out_dir, f"feature_l{L}.sft") for L in args.output_layers
    }
    missing = [L for L, pth in target.items() if not os.path.exists(pth)]
    if not missing:
        logging.info("All requested layer files already exist - nothing to do.")
        return
    else:
        logging.debug("Will compute missing layers only: %s", missing)
        args.output_layers = missing  # narrow down

    # ---------- load model (cached) ----------
    device = torch.device(args.device)
    logging.debug("Loading %s …", args.model_id)
    model = load_model(args.model_id, str(device))

    # ---------- forward ----------
    t0 = time.time()
    feats = vidfm3d_forward(filelist, model, device)  # expected: list[Tensor]
    elapsed = time.time() - t0

    # Normalize to list
    if isinstance(feats, torch.Tensor):
        feats = [feats]

    if not isinstance(feats, (list, tuple)):
        raise RuntimeError(
            "Expected inference(...) to return a Tensor or list of Tensors."
        )

    # ---------- basic sanity & layer filter ----------
    num_layers_avail = len(feats)
    logging.debug(
        "Fast3R returned %d feature tensors (elapsed %.1fs).", num_layers_avail, elapsed
    )

    bad = [L for L in args.output_layers if L < 0 or L >= num_layers_avail]
    if bad:
        raise RuntimeError(
            f"Requested --output-layers {bad} out of range [0, {num_layers_avail-1}]"
        )

    # ---------- reshape + save one file per requested layer ----------
    total_saved = 0
    expected_numel = T * HTOK * WTOK * C
    for L in args.output_layers:
        raw = feats[L]
        if raw.numel() != expected_numel:
            raise RuntimeError(
                f"Layer {L}: numel={raw.numel()} != T*Ht*Wt*C={expected_numel} — reshape will fail."
            )
        feat_spatial = raw.reshape(T, HTOK, WTOK, C).contiguous()
        out_path = target[L]
        save_file({"feat": feat_spatial.half()}, out_path)
        logging.debug(
            "[FAST3R] saved layer %d → %s  shape=%s",
            L,
            out_path,
            tuple(feat_spatial.shape),
        )
        total_saved += 1

    logging.info(
        "[FAST3R] done - wrote %d/%d requested layers to %s",
        total_saved,
        len(args.output_layers),
        args.out_dir,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
