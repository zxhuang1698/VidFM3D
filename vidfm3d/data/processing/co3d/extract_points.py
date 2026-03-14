#!/usr/bin/env python3
import argparse
import glob
import logging
import os

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from vidfm3d.vggt.models.vggt import VGGT
from vidfm3d.vggt.utils.geometry import unproject_depth_map_to_point_map
from vidfm3d.vggt.utils.load_fn import load_and_preprocess_images
from vidfm3d.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ---------------- Helper Functions --------------------


def get_image_and_mask_lists(scene_dir):
    """
    CO3D structure:
        …/scene/images/frame_00000.jpg
        …/scene/masks/frame_00000.png
    Returns two equal-length, sorted lists.
    """
    img_dir = os.path.join(scene_dir, "images")

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    mask_files = [
        image_file.replace("/images/", "/masks/").replace(".jpg", ".png")
        for image_file in image_files
    ]

    if not image_files:
        raise FileNotFoundError(f"No images found in {img_dir}")

    return image_files, mask_files


# ---------------- Main Processing Function --------------------


def process_scene(
    model,
    scene_dir,
    output_path,
    num_frames=81,
    masked_input=False,
    mask_threshold=0.5,
):
    # Get lists of images & masks
    image_files, mask_files = get_image_and_mask_lists(scene_dir)
    logging.debug(f"Found {len(image_files)} images in {scene_dir}")

    if num_frames > 0 and len(image_files) < num_frames:
        raise ValueError(
            f"Scene directory must contain at least {num_frames} images. "
            f"Found {len(image_files)}."
        )

    # Optional consecutive-frame subsampling (affects images and masks identically)
    if num_frames > 0 and len(image_files) > num_frames:
        start_idx = 0
        image_files = image_files[start_idx : start_idx + num_frames]
        mask_files = mask_files[start_idx : start_idx + num_frames]
        logging.debug(f"Subsampling {num_frames} images.")
    elif num_frames <= 0 or len(image_files) == num_frames:
        start_idx = 0
        logging.debug(f"Using all {len(image_files)} images.")
    else:
        raise ValueError(
            f"Scene directory must contain at least {num_frames} images. "
            f"Found {len(image_files)}."
        )

    # Mask the images if requested
    if masked_input:
        input_images = []
        for image_file, mask_file in zip(image_files, mask_files):
            # Load image and mask
            image = Image.open(image_file).convert("RGB")
            mask = Image.open(mask_file).convert("L")

            # Make sure mask has the same size as the image
            assert (
                image.size == mask.size
            ), f"Image and mask sizes do not match: {image.size} vs {mask.size}"
            # Convert mask to binary
            mask_np = np.array(mask) / 255.0
            mask_np = np.clip(mask_np, 0, 1)
            mask_np = np.where(mask_np > mask_threshold, 1.0, 0.0)
            # Composite an RGBA image
            image_np = np.array(image) / 255.0
            image_rgba = np.concatenate(
                [image_np, np.expand_dims(mask_np, axis=-1)], axis=-1
            )
            # Convert to PIL Image
            image_rgba = Image.fromarray(
                (image_rgba * 255).astype(np.uint8), mode="RGBA"
            )
            input_images.append(image_rgba)
    else:
        input_images = image_files

    # Preprocess images using VGGT's utility. This returns a tensor of shape (N, C, H, W)
    images = load_and_preprocess_images(input_images).to("cuda")
    masks = load_and_preprocess_images(mask_files)[:, :1]
    masks = masks > mask_threshold

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
        "masks": masks.permute(0, 2, 3, 1),  # (S, H, W, 1)
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
        help="Path to the scene directory (containing both images and masks)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save output safetensors file"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Number of sampled frames to process",
    )
    parser.add_argument(
        "--masked-input",
        action="store_true",
        help="Feed the model RGB images composited on white using the foreground mask",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.99,
        help="Foreground threshold (after preprocessing, in [0, 1])",
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
        masked_input=args.masked_input,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    main()
