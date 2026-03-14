#!/usr/bin/env python3
"""
Extract per‑frame DINO‑v2 features for a DL3DV scene and save **one**
safetensor:

    <out_dir>/feature_t{start_idx}.sft   # key "feat": (T, 30, 52, C) FP16

Typical wrapper call
--------------------
python -m features.dino.extract_features \
       --scene-dir  vidfm3d/DL3DV/DL3DV-10K/1K/<hash>/images_4 \
       --data-sft   vidfm3d/data/DL3DV/DL3DV-processed/1K/<hash>.sft \
       --out-dir    vidfm3d/features/dino/1K/<hash> \
       --model-id   facebook/dinov2-large \
       --batch      64
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from typing import List

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoImageProcessor, Dinov2Model


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame PNG paths (frame_00001.png, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_and_resize(paths: List[str], size=(728, 420)) -> List[Image.Image]:
    pil_frames = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        pil_frames.append(im.resize(size, Image.BICUBIC))
    return pil_frames


@torch.no_grad()
def dino_forward(model, proc, pil_batch, device):
    """Return (B, 1+N, C) where N = 30*52."""
    inputs = proc(
        images=pil_batch,
        return_tensors="pt",
        do_center_crop=False,
        do_resize=False,  # already resized
    ).to(device)
    return model(**inputs).last_hidden_state  # (B,1+N,C)


def reshape_tokens(tok: torch.Tensor, hp=30, wp=52) -> torch.Tensor:
    """(1+N,C) -> (hp,wp,C) dropping CLS."""
    return tok[1:].reshape(hp, wp, -1).contiguous()


from functools import lru_cache


@lru_cache(maxsize=None)
def load_model(model_id: str, device: str) -> Dinov2Model:
    model = Dinov2Model.from_pretrained(model_id).to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="DINO‑v2 scene feature extractor (resume‑safe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--data-sft", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-ext", default="png", help="Image file extension")
    p.add_argument("--model-id", default="facebook/dinov2-large")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args, unknown = p.parse_known_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    if unknown:
        logging.warning("[warn] ignored unknown args: %s", unknown)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---------- resume check -------------------------------------------------
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())
    num_frames = meta["images"].shape[0]  # length S in the safetensor
    out_path = os.path.join(out_dir, "feature.sft")

    if os.path.exists(out_path):
        try:
            buf = load_file(out_path)
            feat = buf["feat"]  # (T, 30, 52, C)
            ok = (
                feat.ndim == 4
                and feat.shape[0] == num_frames
                and feat.shape[1] == 30
                and feat.shape[2] == 52
            )
            if ok:
                logging.info(
                    "Found existing feature file with correct shape %s – skipping.",
                    tuple(feat.shape),
                )
                return
            else:
                logging.warning(
                    "Existing feature file has shape %s, expected (%d,30,52,C) – "
                    "re‑computing.",
                    tuple(feat.shape),
                    num_frames,
                )
        except Exception as e:
            logging.warning(
                "Could not read existing feature file (%s), re‑creating.  %s",
                out_path,
                e,
            )

    # ---------- gather full‑res frame paths ----------------------------------
    frame_paths_all = list_frames(args.scene_dir, ext=args.image_ext)
    needed = frame_paths_all[start_idx : start_idx + num_frames]
    if len(needed) != num_frames:
        raise RuntimeError("Scene folder missing required frames.")

    # ---------- load model ---------------------------------------------------
    logging.info("Loading %s …", args.model_id)
    proc = AutoImageProcessor.from_pretrained(args.model_id)
    model = load_model(args.model_id, args.device)

    # ---------- extract in batches ------------------------------------------
    feats_list = []
    hp, wp = 30, 52
    t0 = time.time()
    for i in range(0, num_frames, args.batch_size):
        batch_paths = needed[i : i + args.batch_size]
        pil_batch = load_and_resize(batch_paths)
        tok = dino_forward(model, proc, pil_batch, args.device)  # (B,1+N,C)
        # reshape each frame
        for t in tok:
            feats_list.append(reshape_tokens(t, hp, wp))  # (30,52,C)

    feat_tensor = torch.stack(feats_list)  # (T,30,52,C)
    logging.info(
        "DINO extracted %d frames in %.1fs, feat shape %s",
        num_frames,
        time.time() - t0,
        tuple(feat_tensor.shape),
    )

    # ---------- save ---------------------------------------------------------
    save_file({"feat": feat_tensor.half()}, out_path)
    logging.info("Saved features → %s", out_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
