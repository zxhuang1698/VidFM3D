#!/usr/bin/env python3
"""
Create a train/val split list for CO3D.

Each entry in the json is:  ["toy", "f0e3c1d6"]   # [subset, scene_name]

Example:
python vidfm3d/data/processing/co3d/create_split.py \
    --root vidfm3d/data/CO3D/CO3D-processed \
    --out-dir vidfm3d/data/CO3D/CO3D-processed
"""
import argparse
import json
import os
import random
from pathlib import Path


def collect_scenes(root: Path, subset: str):
    subdirs = (
        [subset]
        if subset.lower() != "all"
        else [d for d in os.listdir(root) if (root / d).is_dir()]
    )
    scenes = []
    for sub in sorted(subdirs):
        sub_path = root / sub
        for h in sorted(os.listdir(sub_path)):
            if not h.endswith(".sft") or not (sub_path / h).is_file():
                continue
            scenes.append((sub, h.split(".")[0]))
    return scenes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--subset", default="all")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    root = Path(args.root)
    scenes = collect_scenes(root, args.subset)
    random.seed(args.seed)
    random.shuffle(scenes)

    n_val = max(1, int(len(scenes) * args.val_ratio))
    val_scenes, train_scenes = scenes[:n_val], scenes[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train.json", "w") as f:
        json.dump(train_scenes, f, indent=2)
    with open(out_dir / "val.json", "w") as f:
        json.dump(val_scenes, f, indent=2)
    print(f"Wrote {len(train_scenes)} train / {len(val_scenes)} val scenes ➜ {out_dir}")


if __name__ == "__main__":
    main()
