#!/usr/bin/env python3
"""
Create a train/val split list for DL3DV WITHOUT using glob.

Each entry in the json is:  ["1K", "f0e3c1d6"]   # [subset, scene_hash]

Example:
  python vidfm3d/data/processing/dl3dv/create_split_dl3dv.py \
      --root vidfm3d/data/DL3DV/DL3DV-processed \
      --out-dir vidfm3d/data/DL3DV/DL3DV-processed
"""
import argparse
import json
import os
import random
from pathlib import Path

outliers = [
    ("4K", "e9b49b6588acea0bee7b8ecd0ff34ea24dcb9a8e84e10a124ee8b8bab0983f11.sft"),
    ("4K", "cc40802589f94594383488ff89249d7be1b0609a3b9cfc980698bf2854406eff.sft"),
    ("4K", "88e8c9b9d50305d4c35e9f918527b4b4c2a5f69651fe63678278aa4c0fa122c9.sft"),
    ("4K", "2cbef5aa4038c2674847e25d77d43a7401554f518ef4ff9f36038c6aac7c1e2b.sft"),
    ("4K", "22657d48067798e7286d409f0c62cba3667f90a3e71f7ad7ba70ab54ed6f885b.sft"),
    ("4K", "5b93293a6d926ce06f6efa8dbf5429df90d14077f12456c1c48402d007fe9191.sft"),
    ("4K", "5d6481db3d3f5d79fba3092a4a8dcb04ba9cb20ce20f66b0dfbd5ca85edfb734.sft"),
    ("3K", "6da29bd9d9b51888ac93d545aeedaf7c72a41b56f09f90ff1da3ea01ee200176.sft"),
    ("4K", "023633c21f5b4a633a836e5c8eb8e5464bd133db28c3c2fc7324124eeb031105.sft"),
    ("3K", "785fc27e01679d03934ac0601acf26c54f474ccc23bb313a111f56ff6d4cbdc3.sft"),
    ("3K", "b917b26a4b0387001b6598ca28d7c21f003eb1285bb82fc11cf297e06128c625.sft"),
    ("4K", "a8a848e57e3e74a18107d8b0166323ede27e80bd9a7ec83cca17bd77bcaf4588.sft"),
    ("3K", "eb42750070f9e8d5b5569aaffb1a97626a9dcf4ac588f5092824bf183e0e3fa7.sft"),
]


def collect_scenes(root: Path, subset: str):
    subdirs = (
        [subset]
        if subset.lower() != "all"
        else [d for d in os.listdir(root) if (root / d).is_dir() and d.endswith("K")]
    )
    scenes = []
    for sub in sorted(subdirs):
        sub_path = root / sub
        for h in sorted(os.listdir(sub_path)):
            if not h.endswith(".sft") or not (sub_path / h).is_file():
                continue
            if (sub, h) in outliers:
                print(f"Skipping outlier: {sub}/{h}")
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
