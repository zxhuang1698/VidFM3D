#!/usr/bin/env python3
"""
extract_frames.py
────────────────
Convert raw extractor output into DL3DV‑style sequences with:
  • exact *A:R* crops (default 16 : 9) – **no padding, always inside**
  • stride / frame‑count sampling
  • per‑sequence acceptance based on mean truncation ≤ τ

**Crop pipeline** (per sampling window)
1. Compute the union of all tight masks.
2. *Expand* that union to the requested aspect‑ratio (never smaller).
3. If the expanded crop is inside ⇒ truncation = 0.
4. Otherwise compute one **uniform** scale so the crop *shrinks* until
   the worst‑violating side is exactly flush with the image border.
   (Aspect is preserved.)
5. Round to integers while keeping the exact ratio, anchor on the union
   centre, write frames – reject the whole sequence if even the *best*
   window’s mean truncation exceeds `--trunc_thresh`.

Usage example:
```
python extract_frames.py \
    --raw_root ./data --out_root ./CO3D-raw \
    --stride 1 --num_frames 81 \
    --trunc_thresh 0.25 --resize_to 960 540
```
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

# ──────────────── CLI ───────────────────────────────────────────────────

def get_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--search_step", type=int, default=4,
                   help="step size for candidate window starts (≥1)")
    p.add_argument("--trunc_thresh", type=float, default=0.25)
    p.add_argument("--bbox_border", type=float, default=0.50)
    p.add_argument("--resize_to", nargs=2, type=int, default=[960, 540], metavar=("W", "H"))
    p.add_argument("--aspect", type=str, default="16:9", help="target aspect ratio W:H, e.g. 4:3")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ──────────────── geometry helpers ──────────────────────────────────────

def upsample_factor(crop: list[int], target_w: int, target_h: int) -> float:
    """Return max resize factor ( >1 means up-sampling )."""
    cw = crop[2] - crop[0] + 1
    ch = crop[3] - crop[1] + 1
    return max(target_w / cw, target_h / ch)

def tight_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return None if xs.size == 0 else [xs.min(), ys.min(), xs.max(), ys.max()]


def _parse_aspect(ar: str) -> float:
    """Parse *W:H* string → float aspect (W/H)."""
    try:
        w, h = map(float, ar.split(":"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid aspect '{ar}', use W:H") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Aspect parts must be > 0")
    return w / h


def crop_aspect_pp_locked(
    img_w: int,
    img_h: int,
    x0u: int,
    y0u: int,
    x1u: int,
    y1u: int,
    ar: float,
) -> list[int]:
    """
    Return an integer crop [x0, y0, x1, y1] such that

      • the crop is *centred* on the principal point  →  centre = ((w-1)/2,(h-1)/2)
      • the crop has exact aspect ratio `ar` (W/H)
      • the crop is **as large as possible** while staying inside the sensor
        – may uniformly shrink if the perfect container would overflow
      • the crop *tries* to contain the union box; if that is impossible
        the excess will be left to the later `trunc_loss()` test.

    The function always returns a valid crop (never None) so the caller
    can rely on `trunc_loss()` to decide whether to keep or reject
    the sequence.
    """
    # ── principal point (fixed centre) ────────────────────────────────
    cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0        # float

    # ── minimum half-sizes needed to cover the union ──────────────────
    half_w_req = max(cx - x0u, x1u - cx)
    half_h_req = max(cy - y0u, y1u - cy)

    # ── start with a width-limited crop of the right AR ───────────────
    crop_w = 2 * half_w_req + 1
    crop_h = crop_w / ar
    if crop_h < 2 * half_h_req + 1:          # height was the tighter side
        crop_h = 2 * half_h_req + 1
        crop_w = crop_h * ar

    # ── round to integers, preserving exact AR ────────────────────────
    crop_w = int(np.ceil(crop_w))
    crop_h = int(np.ceil(crop_h))
    if crop_w / crop_h > ar:                 # still a hair too wide
        crop_w = int(round(crop_h * ar))
    else:
        crop_h = int(round(crop_w / ar))

    # ── uniform *shrink* if the crop still sticks out of the sensor ───
    if crop_w > img_w or crop_h > img_h:
        s = min(img_w / crop_w, img_h / crop_h)          # 0 < s ≤ 1
        crop_w = int(np.floor(crop_w * s))
        crop_h = int(round(crop_w / ar))                 # keep ratio
        crop_w = min(crop_w, img_w)                      # safety

    # ── final integer bounds, centred on PP ───────────────────────────
    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x1 = x0 + crop_w - 1
    y1 = y0 + crop_h - 1

    # ── clip in case rounding pushed us outside by ±1 px ──────────────
    x0 = max(0, x0);  y0 = max(0, y0)
    x1 = min(img_w - 1, x1)
    y1 = min(img_h - 1, y1)
    return [x0, y0, x1, y1]

def crop_aspect_inside(
    img_w: int,
    img_h: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    ar: float,
) -> list[int]:
    """Return `[x0,y0,x1,y1]` (ints) – exact `ar` and fully inside.

    * Expand union → exact `ar`  (never shrinks the union)
    * Uniform‑shrink until the crop *touches* the worst border.
    * Round to ints while preserving the ratio, then translate so the
      crop is centred on the union (small shift if border‑limited).
    """
    # centre + size of union box (float)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # ── 1. expand to desired AR ───────────────────────────────────────
    if bw / bh < ar:      # too tall ⇒ widen
        crop_h = bh
        crop_w = crop_h * ar
    else:                 # too wide ⇒ heighten
        crop_w = bw
        crop_h = crop_w / ar

    # ── 2. uniform shrink until inside (touching at least one side) ───
    s = min(1.0, img_w / crop_w, img_h / crop_h)
    crop_w *= s
    crop_h *= s

    # ── 3. round to ints & preserve aspect exactly ────────────────────
    crop_w = int(np.floor(crop_w))
    crop_h = int(np.floor(crop_h))
    # adjust the less‑restrictive dimension to keep exact ratio
    if crop_w / crop_h > ar:           # still slightly too wide
        crop_w = int(round(crop_h * ar))
    else:                              # too tall / ok
        crop_h = int(round(crop_w / ar))
    crop_w = min(crop_w, img_w)
    crop_h = min(crop_h, img_h)

    # ── 4. anchor centre (translation only) ───────────────────────────
    # for example, if the crop_w is the limiting factor, then it is the image size
    # which means the new center horizontally shift to the center of the image
    x0c = int(np.clip(round(cx - crop_w / 2), 0, img_w - crop_w))
    y0c = int(np.clip(round(cy - crop_h / 2), 0, img_h - crop_h))
    return [x0c, y0c, x0c + crop_w - 1, y0c + crop_h - 1]


def trunc_loss(boxes: list[list[int]], crop: list[int]) -> float:
    """Mean truncated area fraction for *boxes* inside *crop*."""
    x0, y0, x1, y1 = crop
    lost = total = 0
    for bx0, by0, bx1, by1 in boxes:
        area = (bx1 - bx0 + 1) * (by1 - by0 + 1)
        total += area
        ix0, iy0 = max(bx0, x0), max(by0, y0)
        ix1, iy1 = min(bx1, x1), min(by1, y1)
        keep = max(ix1 - ix0 + 1, 0) * max(iy1 - iy0 + 1, 0)
        lost += area - keep
    return lost / total if total else 0.0


def process_sequence(
        seq_dir: Path, out_root: Path, nF: int, 
        stride: int, search_step: int, resize_to: Tuple[int, int], 
        trunc_thresh: float, ar: float, bbox_border: float,
    ) -> Tuple[int, str]:            # returns True if accepted
    cat = seq_dir.parts[-2]
    seq = seq_dir.name
    
    imgs = sorted((seq_dir / "images").glob("*.jpg"))
    masks = sorted((seq_dir / "masks").glob("*.png"))
    
    # ─ output path ───────────────────────────────────────────────
    out_img = out_root / cat / seq / "images"
    out_msk = out_root / cat / seq / "masks"
    
    # ---- resume: skip if already complete ----------------------------------
    if out_img.exists():
        done_imgs = len(list(out_img.glob("frame_*.jpg")))
        done_masks = len(list(out_msk.glob("frame_*.png")))
        if done_imgs == nF and done_masks == nF:
            return 1, "already processed, skip"                # jump to next sequence
        else:
            # remove incomplete output
            shutil.rmtree(out_img, ignore_errors=True)
            shutil.rmtree(out_msk, ignore_errors=True)

    if len(imgs) == 0 or len(imgs) != len(masks):
        return 0, f"error in input data {seq_dir}, detected {len(imgs)} images and {len(masks)} masks"

    img_w, img_h = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]
    best = {"trunc": 1.0, "crop": None, "start": None}

    for start in range(0, len(imgs) - stride * (nF - 1), search_step):
        idxs = range(start, start + stride * nF, stride)
        bbs = []
        for i in idxs:
            mask = cv2.imread(str(masks[i]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return 0, f"corrupt mask file {masks[i].name}"
            if mask.shape != (img_h, img_w):
                return 0, (f"shape mismatch in {masks[i].name}: "
                        f"{mask.shape} vs RGB {(img_h, img_w)}")
            b = tight_bbox(mask)
            if b is None:
                break              # empty mask – discard this window
            bbs.append(b)
        else:                      # runs only when *no* break occurred
            ux0 = min(b[0] for b in bbs)
            uy0 = min(b[1] for b in bbs)
            ux1 = max(b[2] for b in bbs)
            uy1 = max(b[3] for b in bbs)
            
            # ── add α-fractional border (clamped to image) ───────────
            bw, bh = ux1 - ux0 + 1, uy1 - uy0 + 1
            pad_x = int(round(bbox_border * bw / 2))
            pad_y = int(round(bbox_border * bh / 2))
            ex0 = max(0, ux0 - pad_x)
            ey0 = max(0, uy0 - pad_y)
            ex1 = min(img_w - 1, ux1 + pad_x)
            ey1 = min(img_h - 1, uy1 + pad_y)
            crop = crop_aspect_pp_locked(img_w, img_h, ex0, ey0, ex1, ey1, ar)
            tl = trunc_loss(bbs, crop)
            if tl < best["trunc"]:
                best.update({"trunc": tl, "crop": crop, "start": start})

    # accept / reject sequence (truncation + up-sample check)
    ups = np.inf if best["crop"] is None else upsample_factor(
        best["crop"], *resize_to)
    if best["crop"] is None or best["trunc"] > trunc_thresh or ups > 1.5:
        return 0, f"rejected - trunc={best['trunc']:.4f}  upx={ups:.2f}"
    
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)
    
    w_out, h_out = resize_to
    x0, y0, x1, y1 = best["crop"]

    for j, i in enumerate(range(best["start"], best["start"] + stride * nF, stride), 1):
        rgb = cv2.imread(str(imgs[i]), cv2.IMREAD_UNCHANGED)[y0 : y1 + 1, x0 : x1 + 1]
        msk = cv2.imread(str(masks[i]), cv2.IMREAD_UNCHANGED)[y0 : y1 + 1, x0 : x1 + 1]
        if rgb.shape[:2] != msk.shape[:2]:
            raise ValueError("shape mismatch")
        rgb = cv2.resize(rgb, (w_out,h_out), interpolation=cv2.INTER_CUBIC)
        msk = cv2.resize(msk, (w_out,h_out), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_img/f"frame_{j:05d}.jpg"), rgb)
        cv2.imwrite(str(out_msk/f"frame_{j:05d}.png"), msk)
    return 1, f"accepted - trunc={best['trunc']:.4f}  upx={ups:.2f}"

# ──────────────── main ---------------------------------------------------

def main() -> None:
    args = get_cli()
    ar = _parse_aspect(args.aspect)

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = list(raw_root.glob("*/*"))  # cat / seq
    sequences = [sdir for sdir in sequences if sdir.is_dir() and (sdir / "images").is_dir() and (sdir / "masks").is_dir()]
    if not sequences:
        raise FileNotFoundError(f"No sequences found in {raw_root}")
    print(f"Found {len(sequences)} sequences in {raw_root}")
    sequences.sort()  # sort for reproducibility
    n_total = len(sequences)
    n_ok = 0
    t0 = time.time()

    stride, nF = args.stride, args.num_frames
    
    process_func = partial(
        process_sequence,
        out_root=out_root,
        nF=nF,
        stride=stride,
        search_step=args.search_step,
        resize_to=tuple(args.resize_to),
        trunc_thresh=args.trunc_thresh,
        ar=ar,
        bbox_border=args.bbox_border,
    )
    
    # ───────── parallel driver ─────────────────────────────────────────
    pool = ProcessPoolExecutor(max_workers=6)
    try:
        futures = {pool.submit(process_func, sdir) for sdir in sequences}
        n_total = len(futures); n_ok = 0; t0 = time.time()
        for done, thread in enumerate(as_completed(futures), 1):
            result, msg = thread.result()
            if result: n_ok += 1
            rate = n_ok / done
            eta  = (time.time() - t0) / done * (n_total - done)
            print(f"Progress:{done}/{n_total}, keep rate={rate:.2%}, ETA={eta/3600:.1f}H, {msg}", flush=True)
    except KeyboardInterrupt:
        print("\n✗ Interrupted – cancelling outstanding jobs, don't cancel again...", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)  
        raise
    else:
        pool.shutdown()
        print(f"\n✓ finished - accepted {n_ok}/{n_total} sequences")

if __name__ == "__main__":
    main()