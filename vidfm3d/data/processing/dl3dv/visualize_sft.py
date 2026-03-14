#!/usr/bin/env python3
import argparse
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from safetensors.torch import load_file


def load_sft(sft_path):
    """
    Loads the SFT file (a safetensors file) and converts the tensors to numpy arrays.
    Expected keys: "images", "vertices", "colors", "extrinsic", "intrinsic"
    """
    data = load_file(sft_path)
    out = {}
    for key, tensor in data.items():
        # Convert to numpy array and squeeze batch dimensions if needed.
        arr = tensor.cpu().numpy()
        out[key] = arr
    return out


def save_image(image_array, save_path):
    """
    Saves an image given a numpy array of shape (H, W, 3) with values in [0, 255].
    """
    # If the array is not uint8, convert it.
    if image_array.dtype != "uint8":
        image_array = (image_array).astype(np.uint8)
    Image.fromarray(image_array).save(save_path)


def transform_point_cloud(vertices, extrinsic):
    """
    Given a global point cloud (vertices of shape (N, 3)) in world coordinates and an extrinsic matrix
    (3x4) for a view, transforms the points into the camera coordinate system.

    We first add a homogeneous coordinate, then multiply by the transpose of extrinsic.
    """
    N = vertices.shape[0]
    vertices_hom = np.concatenate([vertices, np.ones((N, 1))], axis=1)  # (N, 4)
    # extrinsic: (3, 4). Transform: X_cam = extrinsic @ X_world_homogeneous (if extrinsic is world-to-camera)
    transformed = vertices_hom @ extrinsic.T  # (N, 3)
    # flip y and z axes to match the camera coordinate system
    transformed[:, [1, 2]] *= -1
    return transformed


def save_ply_point_cloud(vertices, colors, save_path):
    """
    Uses trimesh to export a point cloud as an ASCII PLY.
    vertices: (N, 3) numpy array
    colors: (N, 3) numpy array (assumed to be in [0, 255])
    """
    # Create a trimesh PointCloud
    pc = trimesh.points.PointCloud(vertices, colors=colors)
    pc.export(save_path)


def get_point_cloud(
    pred_world_points,
    pred_world_points_conf,
    images,
    conf_thres=0.0,
    max_points=10000,
    fps=False,
):
    """
    Given the predicted world points and their confidence scores, this function
    generates a point cloud. It applies a confidence threshold and subsamples the points
    if needed. The function also handles different image formats (NCHW or NHWC).
    Args:
        pred_world_points: Predicted world points of shape (S, H, W, 3)
        pred_world_points_conf: Confidence scores of shape (S, H, W, 1)
        images: Input images of shape (S, C, H, W) or (S, H, W, C)
        conf_thres: Confidence threshold for filtering points
        max_points: Maximum number of points to keep in the point cloud
        fps: If True, use farthest point sampling for subsampling
    Returns:
        vertices_3d: 3D points of shape (N, 3)
        colors_rgb: RGB colors of shape (N, 3)
    """
    # Build the point cloud
    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = colors_rgb.reshape(-1, 3)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres * 100)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    # Apply confidence mask
    vertices_3d = vertices_3d[conf_mask]  # (N, 3)
    colors_rgb = colors_rgb[conf_mask]  # (N, 3)

    # Subsample points if needed
    if max_points > 0 and len(vertices_3d) > max_points:
        if fps:
            vertices_3d = torch.from_numpy(vertices_3d).to("cuda")
            colors_rgb = torch.from_numpy(colors_rgb).to("cuda")
            # subsample to 500K points first
            sample_indices = torch.randperm(vertices_3d.shape[0])[:500000]
            vertices_3d = vertices_3d[sample_indices].unsqueeze(0)
            colors_rgb = colors_rgb[sample_indices].unsqueeze(0)
            # then subsample to max_points
            _, sample_indices = sample_farthest_points(vertices_3d, K=max_points)
            vertices_3d = masked_gather(vertices_3d, sample_indices)[0].cpu().numpy()
            colors_rgb = masked_gather(colors_rgb, sample_indices)[0].cpu().numpy()
        else:
            sample_indices = np.random.choice(
                len(vertices_3d), max_points, replace=False
            )
            vertices_3d = vertices_3d[sample_indices]
            colors_rgb = colors_rgb[sample_indices]

    return vertices_3d, colors_rgb


def visualize_sft(sft_data, output_dir):
    """
    For each view in the sft_data, creates a folder in output_dir:
      - Saves the view image (from sft_data["images"][i]) as a PNG.
      - Transforms the global point cloud (sft_data["vertices"]) using the i-th extrinsic and
        saves the transformed point cloud (with sft_data["colors"]) as a PLY.
    """
    images = sft_data["images"]  # (S, 3, H, W)
    pointmaps = sft_data["pointmaps"]  # (S, H, W, 3)
    confmaps = sft_data["confmaps"]  # (S, H, W, 1)
    depthmaps = sft_data["depthmaps"]  # (S, H, W, 1)
    extrinsics = sft_data["extrinsic"]  # (S, 3, 4)

    # Extract the point cloud and colors
    vertices, colors = get_point_cloud(
        pointmaps, confmaps, images, conf_thres=0.0, max_points=10000, fps=False
    )

    num_views = images.shape[0]
    print(f"[INFO] Found {num_views} views in the SFT file.")

    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depthmaps"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)

    # Normalize the depth maps to [0, 1] and colorize with viridis colormap
    depth_min, depth_max = depthmaps.min(), depthmaps.max()
    depthmaps_norm = (depthmaps - depth_min) / (depth_max - depth_min + 1e-8)
    # Apply colormap per frame
    depth_viridis = []
    for i in range(num_views):
        frame = (depthmaps_norm[i].squeeze() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(frame, cv2.COLORMAP_VIRIDIS)
        # Convert to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        depth_viridis.append(colored)
    depth_viridis = np.stack(depth_viridis, axis=0)  # (S, H, W, 3)

    for i in range(num_views):
        # Save the i-th image
        image_path = os.path.join(output_dir, f"images/{i:05d}.png")
        # Ensure image is uint8
        img = images[i].transpose(1, 2, 0)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        save_image(img, image_path)

        # Save the depth map with normalization
        depth_path = os.path.join(output_dir, f"depthmaps/{i:05d}.png")
        depth_image = depth_viridis[i]  # already (H, W, 3)
        save_image(depth_image, depth_path)

        # Transform the global point cloud using view i's extrinsic
        extr_i = extrinsics[i]  # (3, 4)
        transformed_points = transform_point_cloud(vertices, extr_i)
        print(extr_i)
        # Save as a PLY file
        ply_path = os.path.join(output_dir, f"pointclouds/{i:05d}.ply")
        save_ply_point_cloud(transformed_points, colors, ply_path)
        # print(f"[INFO] Saved transformed point cloud to {ply_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SFT output by extracting images, global point cloud, and per-view camera parameters."
    )
    parser.add_argument(
        "--sft",
        type=str,
        required=True,
        help="Path to the SFT file (safetensors format) generated by point_processing.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save visualizations (subfolders per view).",
    )
    args = parser.parse_args()

    sft_data = load_sft(args.sft)
    os.makedirs(args.output, exist_ok=True)
    visualize_sft(sft_data, args.output)
    print(f"[INFO] Visualization complete. Check {args.output}")


# run in bash:
# for file in vidfm3d/data/DL3DV/DL3DV-processed/1K/0*.sft; do hash=$(basename "$file" .sft); CUDA_VISIBLE_DEVICES=0 python -m vidfm3d.data.processing.visualize_sft --sft "$file" --output vidfm3d/data/DL3DV/DL3DV-10K/test_output/"$hash"; done
if __name__ == "__main__":
    main()
