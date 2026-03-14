#!/usr/bin/env python3
"""
Quick‑start
-----------
WAN:
python -m features.run_co3d \
       --vfm wan \
       --subset all \
       --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
       --prompt "" \
       --output-layers 20 \
       --t 749 \

DINO:
python -m features.run_co3d \
       --vfm dino \
       --subset all \
       --batch-size 64 \
       --model-id facebook/dinov2-large

VJEPA:
python -m features.run_co3d \
       --vfm vjepa \
       --subset all

Opensora:
python -m features.run_co3d \
       --vfm opensora \
       --subset all \
       --t 0.25 \
       --output-layers 20

CogVideoX:
python -m features.run_co3d \
       --vfm cogvideox \
       --subset all \
       --t 749 \
       --output-layers 20

Aether:
python -m features.run_co3d \
       --vfm aether \
       --subset all \
       --t 749 \
       --output-layers 20 \
       --task videogen

Fast3R:
python -m features.run_co3d \
       --vfm f3r \
       --subset all \
       --output-layers 24
"""

import argparse
import importlib
import logging
import os
import sys
import time
from datetime import timedelta


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def list_hash_dirs(subset_dir):
    """Return the list of immediate sub‑dirs (scene hashes) under subset_dir."""
    return sorted(
        d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))
    )


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")  # Redirect to null device

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def call_extractor(extractor_main, argv):
    """Execute extractor_main(argv) and report (elapsed, success)."""
    start = time.time()
    try:
        with HiddenPrints():
            extractor_main(argv)  # argv is a normal sys.argv[1:]
        return time.time() - start, True
    except SystemExit as exc:  # argparse uses this
        return time.time() - start, exc.code == 0
    except Exception as exc:
        logging.exception("Extractor crashed: %s", exc)
        return time.time() - start, False


def nice_td(seconds):
    return str(timedelta(seconds=int(seconds)))


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="CO3D → feature extraction wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset locations (all have reasonable defaults)
    parser.add_argument(
        "--co3d-root",
        default="vidfm3d/data/CO3D/CO3D-raw",
        help="Root that contains CO3D-10K/hydrant/…",
    )
    parser.add_argument(
        "--processed-root",
        default="vidfm3d/data/CO3D/CO3D-processed",
        help="Root where extract_points.py wrote .sft",
    )
    parser.add_argument(
        "--out-root",
        default="vidfm3d/data/CO3D/FEAT",
        help="Root to store extracted features <out_root>/<vfm>/<subset>/<hash>",
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        help="Image file extension to look for in the scene directories",
    )

    # Which part of the dataset
    parser.add_argument(
        "--subset", default="all", help="'toy', 'kite', …, or 'all' (default)"
    )

    # Optinally start from specified index and end at specified index
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start processing from this index in the scene list (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End processing at this index in the scene list (default: None, process all)",
    )

    # Which video foundation model
    parser.add_argument(
        "--vfm",
        default="wan",
        choices=["wan", "dino", "vjepa", "opensora", "cogvideox", "aether", "f3r"],
        help="Which extractor module to invoke",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG‑level logs in the terminal (file log is always DEBUG)",
    )

    # Parse known vs unknown flags
    args, extractor_args = parser.parse_known_args()
    extractor_mod = importlib.import_module(f"features.{args.vfm}.extract_features")
    extractor_main = extractor_mod.main  # we’ll call this repeatedly

    # ------------------------------------------------------------------ #
    # Logging                                                            #
    # ------------------------------------------------------------------ #
    log_dir = os.path.join(args.out_root, args.vfm)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"processing-{args.subset}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Wrapper started – VFM=%s  subset=%s", args.vfm, args.subset)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            h.setLevel(logging.DEBUG)  # file shows everything
        elif isinstance(h, logging.StreamHandler):
            h.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # ------------------------------------------------------------------ #
    # Gather all scenes first (for progress/ETA)                         #
    # ------------------------------------------------------------------ #
    if args.subset.lower() == "all":
        subsets = sorted(
            d
            for d in os.listdir(args.co3d_root)
            if os.path.isdir(os.path.join(args.co3d_root, d))
        )
    elif args.subset.lower() == "ablate":
        subsets = [
            "apple",
            "backpack",
            "bench",
            "bowl",
            "chair",
            "handbag",
            "hydrant",
            "motorcycle",
            "plant",
            "teddybear",
            "toytruck",
        ]
    else:
        subsets = [args.subset]

    scenes = []  # list of (subset, hash_dir, images, data_sft)
    for subset in subsets:
        subset_dir = os.path.join(args.co3d_root, subset)
        if not os.path.isdir(subset_dir):
            continue
        for hash_dir in list_hash_dirs(subset_dir):
            img_dir = os.path.join(subset_dir, hash_dir, "images")
            data_sft = os.path.join(args.processed_root, subset, f"{hash_dir}.sft")
            if os.path.isdir(img_dir) and os.path.isfile(data_sft):
                scenes.append((subset, hash_dir, img_dir, data_sft))

    if args.start > 0 or args.end is not None:
        start_idx = args.start
        end_idx = args.end if args.end is not None else len(scenes)
        logging.info(f"Starting from scene index {start_idx}: {scenes[start_idx]}")
        logging.info(
            f"Ending at scene index {end_idx} (exclusive): {scenes[end_idx] if end_idx <= len(scenes)-1 else 'end of list'}"
        )
        scenes = scenes[start_idx:end_idx]

    total_scenes = len(scenes)
    if total_scenes == 0:
        logging.error("No valid scenes found – exiting.")
        return

    extractor_module = f"features.{args.vfm}.extract_features"
    wrapper_start = time.time()
    done = 0

    # ------------------------------------------------------------------ #
    # Process each scene                                                 #
    # ------------------------------------------------------------------ #
    for subset, hash_dir, scene_images, data_sft in scenes:
        out_dir = os.path.join(args.out_root, args.vfm, subset, hash_dir)
        os.makedirs(out_dir, exist_ok=True)

        argv = [
            "--scene-dir",
            scene_images,
            "--data-sft",
            data_sft,
            "--out-dir",
            out_dir,
            "--image-ext",
            args.image_ext,
            *extractor_args,  # whatever extra flags you passed through
        ]
        logging.debug("Extractor argv: %s", " ".join(argv))
        elapsed, ok = call_extractor(extractor_main, argv)

        # elapsed, ok = call_extractor(cmd)
        done += 1
        eta = (time.time() - wrapper_start) / done * (total_scenes - done)

        status = "OK" if ok else "FAILED"
        logging.info(
            "[ %d / %d | %s | ETA %s ] Scene %s … %s",
            done,
            total_scenes,
            nice_td(elapsed),
            nice_td(eta),
            hash_dir[:8],
            status,
        )

    logging.info(
        "Finished %d scenes in %s (log saved to %s)",
        total_scenes,
        nice_td(time.time() - wrapper_start),
        log_path,
    )


if __name__ == "__main__":
    main()
