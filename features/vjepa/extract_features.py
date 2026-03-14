#!/usr/bin/env python3
"""
Extract V-JEPA features from a DL3DV scene and save **one** safetensor:

    <out_dir>/feature.sft            # key "feat": (T, H, W, C) — FP16

Example
-------
python -m features.vjepa.extract_features \
       --scene-dir  vidfm3d/data/DL3DV/DL3DV-10K/1K/<hash>/images_4 \
       --data-sft   vidfm3d/data/DL3DV/DL3DV-processed/1K/<hash>.sft \
       --out-dir    vidfm3d/features/vjepa/1K/<hash> \
       --checkpoint path/to/vjepa.ckpt
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

# --------------------------------------------------------------------------- #
# 0.  Build / load the V-JEPA backbone                                        #
# --------------------------------------------------------------------------- #
from .vjepa import build_model as build_vjepa


class VJEPAFeaturizer_Spaced(nn.Module):
    """
    Groups everything we need to turn a list-of-PIL frames into
    a (T,H,W,C) token tensor.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        img_size: tuple[int, int] = (832, 480),  # WxH fed to encoder
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # backbone ---------------------------------------------------
        self.backbone = build_vjepa(ckpt)
        self.backbone.eval().requires_grad_(False).to(self.device)

        # preprocessing ---------------------------------------------
        self.transform = T.Compose(
            [
                T.Resize(img_size[::-1]),  # PIL expects H,W
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_size = img_size

    # -------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, frames: List[Image.Image], interval=5) -> torch.Tensor:
        """
        Args
        ----
        frames : *exactly* 76 PIL RGB images (frame-0 … frame-60)

        Returns
        -------
        feat   : (4, 8, H=img_h//16, W=img_w//16, C=1280)
        """
        assert (
            len(frames) == interval * 15 + 1
        ), f"Need {interval * 15 + 1} frames, got {len(frames)}"

        # ------------------------------------------------------ #
        # 1.  Build 4 clips, each 16 frames:                    #
        #     clip_i = [0] + [i  + k*4  for k=0..14]            #
        # ------------------------------------------------------ #
        clip_indices = [
            [0] + [i + interval * k for k in range(15)]
            for i in range(1, 1 + interval)  # 1≤i≤5
        ]  # list[N][16]

        # stack into (N,C,16,H,W)
        clips = []
        for ids in clip_indices:
            clip = torch.stack([self.transform(frames[j]) for j in ids], dim=1)
            clips.append(clip)  # (C,16,H,W)
        batch = torch.stack(clips).to(self.device)  # (N,C,16,H,W)

        # ------------------------------------------------------ #
        # 2.  Forward & reshape                                 #
        # ------------------------------------------------------ #
        tokens = self.backbone(batch)  # (N, Ntok, 1280)
        tokens = rearrange(
            tokens,
            "b (t h w) c -> b t h w c",
            t=8,
            h=self.img_size[1] // 16,
            w=self.img_size[0] // 16,
        )  # (N,8,Ht,Wt,1280)
        return tokens


class VJEPAFeaturizer_Chunked(nn.Module):
    """
    Groups everything we need to turn a list-of-PIL frames into
    a (T,H,W,C) token tensor.
    """

    def __init__(
        self,
        ckpt: str | None = None,
        img_size: tuple[int, int] = (832, 480),  # WxH fed to encoder
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

        # backbone ---------------------------------------------------
        self.backbone = build_vjepa(ckpt)
        self.backbone.eval().requires_grad_(False).to(self.device)

        # preprocessing ---------------------------------------------
        self.transform = T.Compose(
            [
                T.Resize(img_size[::-1]),  # PIL expects H,W
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_size = img_size

    # -------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, frames: List[Image.Image], n_chunks=5) -> torch.Tensor:
        """
        Args
        ----
        frames : PIL RGB images

        Returns
        -------
        feat   : (4, 8, H=img_h//16, W=img_w//16, C=1280)
        """
        assert (
            len(frames) == n_chunks * 16
        ), f"Need {n_chunks * 16} frames, got {len(frames)}"

        # ------------------------------------------------------ #
        # 1.  Build n_chunks clips, each 16 frames          #
        # ------------------------------------------------------ #
        clip_indices = [
            [i * 16 + j for j in range(16)] for i in range(n_chunks)  # 0≤i<n_chunks
        ]  # list[N][16]

        # stack into (N,C,16,H,W)
        clips = []
        for ids in clip_indices:
            clip = torch.stack([self.transform(frames[j]) for j in ids], dim=1)
            clips.append(clip)  # (C,16,H,W)
        batch = torch.stack(clips).to(self.device)  # (N,C,16,H,W)

        # ------------------------------------------------------ #
        # 2.  Forward & reshape                                 #
        # ------------------------------------------------------ #
        tokens = self.backbone(batch)  # (N, Ntok, 1280)
        tokens = rearrange(
            tokens,
            "b (t h w) c -> b t h w c",
            t=8,
            h=self.img_size[1] // 16,
            w=self.img_size[0] // 16,
        )  # (N,8,Ht,Wt,1280)
        return tokens


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def list_frames(scene_dir: str, ext: str = "png") -> List[str]:
    """Return sorted list of frame PNG paths (frame_00001.png, …)."""
    return sorted(glob.glob(os.path.join(scene_dir, f"frame_*.{ext}")))


def load_pil(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="V-JEPA scene feature extractor (resume-safe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--data-sft", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-ext", default="png", help="Image file extension")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--partition", default="spaced", choices=["spaced", "chunked"])
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
    out_path = os.path.join(
        out_dir, "feature.sft" if args.partition == "spaced" else "feature_chunked.sft"
    )

    # ---------- resume ------------------------------------------------------
    if os.path.isfile(out_path):
        try:
            buf = load_file(out_path)["feat"]  # (5,T,H,W,C)
            if buf.dim() == 5 and buf.shape[0] == 5 and buf.shape[1] == 8:
                logging.info("Found existing feature file %s - skipping.", out_path)
                return
            else:
                logging.warning(
                    "Existing file has shape %s — re-computing.", tuple(buf.shape)
                )
        except Exception as e:
            logging.warning(
                "Could not read existing feature file (%s): %s — re-creating.",
                out_path,
                e,
            )

    # ---------- pick 81-frame window ---------------------------------------
    meta = load_file(args.data_sft)
    start_idx = int(meta["start_idx"].item())  # 0-based
    window_size = 76 if args.partition == "spaced" else 80

    frame_paths = list_frames(args.scene_dir, ext=args.image_ext)
    if start_idx + window_size > len(frame_paths):
        raise RuntimeError(
            f"Need {start_idx+window_size} frames but only {len(frame_paths)} exist in {args.scene_dir}"
        )

    window_paths = frame_paths[start_idx : start_idx + window_size]
    frames_pil = load_pil(window_paths)

    # ---------- model -------------------------------------------------------
    logging.info("[VJEPA] loading backbone …")
    if args.partition == "spaced":
        model = VJEPAFeaturizer_Spaced(args.checkpoint, device=args.device)
    elif args.partition == "chunked":
        model = VJEPAFeaturizer_Chunked(args.checkpoint, device=args.device)
    else:
        raise ValueError(f"Unknown partition type: {args.partition}")

    t0 = time.time()
    feats = model(frames_pil)  # (5,8,H,W,C)
    logging.info(
        "[VJEPA] extracted tokens %s in %.1fs", tuple(feats.shape), time.time() - t0
    )

    # ---------- save --------------------------------------------------------
    save_file({"feat": feats.half().contiguous()}, out_path)
    logging.info("Saved features → %s", out_path)


if __name__ == "__main__":
    main()
