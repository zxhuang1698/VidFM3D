#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import time

import cv2
import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from safetensors.torch import save_file

from vidfm3d.vggt.models.vggt import VGGT
from vidfm3d.vggt.utils.geometry import unproject_depth_map_to_point_map
from vidfm3d.vggt.utils.load_fn import load_and_preprocess_images
from vidfm3d.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vidfm3d.vggt.visual_util import download_file_from_url, segment_sky

# ---------------- Helper Functions --------------------


def get_image_file_list(scene_dir):
    """
    Returns a sorted list of PNG files from the scene directory.
    """
    pattern = os.path.join(scene_dir, "**/*.png")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No PNG images found in {scene_dir}")
    return files


# ---------------- Main Processing Function --------------------


def process_scene(
    model,
    scene_dir,
    output_path,
    num_frames=150,
):
    # Get list of image file names
    image_files = get_image_file_list(scene_dir)
    logging.debug(f"Found {len(image_files)} images in {scene_dir}")

    # Raise error if less than required images
    if num_frames > 0 and len(image_files) < num_frames:
        raise ValueError(
            f"Scene directory must contain at least {num_frames} images. Found {len(image_files)}."
        )

    # Subsample consecutive frames if num_frames is specified
    if num_frames > 0 and len(image_files) > num_frames:
        start_idx = np.random.randint(0, len(image_files) - num_frames)
        image_files = image_files[start_idx : start_idx + num_frames]
        logging.debug(
            f"Subsampling {num_frames} images from {len(image_files)} total images."
        )
    else:
        start_idx = 0
        logging.debug(f"Using all {len(image_files)} images.")

    # Preprocess images using VGGT's utility. This returns a tensor of shape (N, C, H, W)
    images = load_and_preprocess_images(image_files).to("cuda")

    # Run inference with mixed precision
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    logging.debug("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = (
                predictions[key].cpu().numpy().squeeze(0)
            )  # remove batch dimension

    # Generate world points from depth map
    logging.debug("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    # predictions (dict): Dictionary containing model predictions with keys:
    # - world_points: 3D point coordinates (S, H, W, 3)
    # - world_points_conf: Confidence scores (S, H, W)
    # - images: Input images (S, H, W, 3)
    # - intrinsic: Camera intrinsic matrices (S, 3, 3)
    # - extrinsic: Camera extrinsic matrices (S, 3, 4)
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get(
        "depth_conf", np.ones_like(pred_world_points[..., 0])
    )
    pred_world_points_conf = np.expand_dims(
        pred_world_points_conf, axis=-1
    )  # (S, H, W, 1)

    # Prepare output dictionary for saving as safetenors
    output_dict = {
        "images": torch.from_numpy(predictions["images"] * 255).to(
            torch.uint8
        ),  # (S, 3, H, W)
        "depthmaps": torch.from_numpy(predictions["depth"]).float(),  # (S, H, W, 1)
        "pointmaps": torch.from_numpy(pred_world_points).float(),  # (S, H, W, 3)
        "confmaps": torch.from_numpy(pred_world_points_conf).float(),  # (S, H, W, 1)
        "intrinsic": torch.from_numpy(predictions["intrinsic"]).float(),  # (S, 3, 3)
        "extrinsic": torch.from_numpy(predictions["extrinsic"]).float(),  # (S, 3, 4)
        "start_idx": torch.tensor(start_idx, dtype=torch.int32),  # (1,)
    }

    # Print the shape and size of each field in bytes
    for key, val in output_dict.items():
        logging.debug(f"{key} shape: {val.shape}")
        logging.debug(f"{key} size: {val.element_size() * val.nelement()} bytes")

    # Save output as safetensors
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(output_dict, output_path)

    # Clean up
    torch.cuda.empty_cache()

    logging.debug(f"Scene processing complete. Results saved at {output_path}")


# ---------------- Main --------------------


def main():
    parser = argparse.ArgumentParser(
        description="Process a DL3DV scene using VGGT to predict camera parameters, depth maps, point maps, and a global point cloud."
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        required=True,
        help="Path to the scene images directory (e.g., .../DL3DV-10K/1001/abc123/images_8)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output safetensors file"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=150,
        help="Number of sampled frames to process",
    )
    args = parser.parse_args()

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime}: [{levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Load VGGT model
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to("cuda")
    model.eval()

    process_scene(
        model,
        args.scene_dir,
        args.output,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
