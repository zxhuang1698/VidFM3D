# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import numpy as np
import torch


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    )  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(
    depth_map: np.ndarray, intrinsic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# --------------------------------------------------------------------------- #
#  Plücker <‑‑> camera–matrix conversion utilities
# --------------------------------------------------------------------------- #


def _build_pixel_grid(H: int, W: int, device=None, dtype=torch.float32):
    """(H,W,3) homogeneous pixel grid   [[u,v,1]]."""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    return torch.stack((xs, ys, ones), dim=-1)  # (H,W,3)


@torch.no_grad()
def mat2plucker(
    intr: torch.Tensor,  # (B,3,3)
    extr: torch.Tensor,  # (B,3,4)   world→cam   [R|t]
    image_size: tuple[int, int],  # (H,W)
    layout: str = "spatial",  #  "spatial" → (B,6,H,W)   "flat" → (B,H*W,6)
    normalize_moment: bool = True,
    moments_rescale: float = 1.0,
) -> torch.Tensor:
    """
    Convert a batch of camera matrices to a Plücker‑ray image.

    Returns
    -------
    torch.Tensor
        (B,6,H,W) if layout=="spatial" else (B,H*W,6)
        ordered as  (d_x,d_y,d_z, m_x,m_y,m_z)
    """
    assert intr.ndim == 3 and intr.shape[-2:] == (3, 3)
    assert extr.ndim == 3 and extr.shape[-2:] == (3, 4)
    B, H, W = intr.shape[0], *image_size
    device, dtype = intr.device, intr.dtype

    # homogeneous pixel grid – reused for the whole batch
    pix_h = _build_pixel_grid(H, W, device, dtype).reshape(-1, 3).t()  # (3,H*W)

    pl = torch.empty((B, 6, H * W), dtype=dtype, device=device)

    for i in range(B):
        K = intr[i]  # (3,3)
        Rwc = extr[i, :, :3]  # world→cam
        twc = extr[i, :, 3:]  # (3,1)

        C = -Rwc.T @ twc  # camera centre in world  (3,1)

        # directions in *camera* space, then to *world* space
        d_cam = torch.linalg.inv(K) @ pix_h  # (3,H*W)
        d_cam = d_cam / torch.linalg.norm(d_cam, dim=0, keepdim=True)
        d_world = Rwc.T @ d_cam  # (3,H*W)

        # moments:  m = C × d
        C_rep = C.expand_as(d_world)  # (3,H*W)
        m_world = torch.cross(C_rep.t(), d_world.t(), dim=-1).t()  # (3,H*W)

        if normalize_moment and moments_rescale != 1.0:
            m_world = m_world / moments_rescale

        pl[i] = torch.cat((d_world, m_world), dim=0)  # (6,H*W)

    if layout == "spatial":
        pl = pl.reshape(B, 6, H, W)
    elif layout == "flat":
        pl = pl.permute(0, 2, 1)  # (B,H*W,6)
    else:
        raise ValueError("layout must be 'spatial' or 'flat'")

    return pl


# ---------- helper: camera centre from many Plücker rays ------------------- #
def _camera_center_from_plucker(d: torch.Tensor, m: torch.Tensor):
    """
    Solve  (I - d dᵀ) C = d × m   in least–squares sense.
    d: (P,3) unit          m: (P,3)           returns C  (3,)
    """
    A = torch.eye(3, device=d.device, dtype=d.dtype)[None] - torch.einsum(
        "pi,pj->pij", d, d
    )  # (P,3,3)
    b = torch.cross(d, m, dim=-1)  # (P,3)
    ATA = (A.transpose(-1, -2) @ A).sum(0)  # (3,3)
    ATb = (A.transpose(-1, -2) @ b.unsqueeze(-1)).sum(0)  # (3,1)
    C = torch.linalg.solve(ATA, ATb).squeeze(-1)  # (3,)
    return C


@torch.no_grad()
def plucker2mat(
    plucker: torch.Tensor,  # (B,6,H,W) or (B,H*W,6)
    image_size: tuple[int, int] | None = None,
    layout: str = "spatial",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Re‑estimate (intrinsic, extrinsic) from a batch of Plücker‑ray images.
    The maths is exact for data coming from `mat2plucker`, so this is chiefly
    for internal consistency checks.

    Returns
    -------
    K : torch.Tensor  (B,3,3)
    E : torch.Tensor  (B,3,4)   world→cam
    """
    if layout == "flat":
        B, P, _ = plucker.shape
        H = W = int(math.sqrt(P))
        assert H * W == P, "need square image to auto‑infer H,W"
        pl = plucker.permute(0, 2, 1).reshape(B, 6, H, W)  # (B,6,H,W)
    else:
        B, _, H, W = plucker.shape
        pl = plucker

    Ks, Es = [], []

    y_pix, x_pix = torch.meshgrid(
        torch.arange(H, device=pl.device, dtype=pl.dtype),
        torch.arange(W, device=pl.device, dtype=pl.dtype),
        indexing="ij",
    )
    u = x_pix.reshape(-1)  # (P,)  ---> horizontal coord
    v = y_pix.reshape(-1)  # (P,)  ---> vertical   coord

    for i in range(B):
        d = pl[i, :3].reshape(3, -1).t()  # (P,3)
        m = pl[i, 3:].reshape(3, -1).t()  # (P,3)

        # --- camera centre ---------------------------------------------------
        C = _camera_center_from_plucker(d, m)

        # --- camera axes -----------------------------------------------------
        idx_c = (H // 2) * W + (W // 2)  # centre
        idx_rx = idx_c + 1 if (idx_c + 1) < d.shape[0] else idx_c
        idx_ry = idx_c + W if (idx_c + W) < d.shape[0] else idx_c

        z_axis = torch.nn.functional.normalize(d[idx_c], dim=0)
        x_tmp = torch.nn.functional.normalize(d[idx_rx], dim=0)
        x_axis = torch.nn.functional.normalize(
            x_tmp - torch.dot(x_tmp, z_axis) * z_axis, dim=0
        )
        y_axis = torch.cross(z_axis, x_axis)
        Rwc = torch.stack((x_axis, y_axis, z_axis), dim=0)  # (3,3)

        # --- intrinsics ------------------------------------------------------
        d_cam = (Rwc @ d.t()).t()  # (P,3)
        x_n = d_cam[:, 0] / d_cam[:, 2]  # (P,)
        y_n = d_cam[:, 1] / d_cam[:, 2]

        A_x = torch.stack((x_n, torch.ones_like(x_n)), dim=1)  # (P,2)
        A_y = torch.stack((y_n, torch.ones_like(y_n)), dim=1)

        sol_x = torch.linalg.lstsq(A_x, u[:, None]).solution  # (2,1)
        sol_y = torch.linalg.lstsq(A_y, v[:, None]).solution

        fx, cx = sol_x[:2, 0]
        fy, cy = sol_y[:2, 0]

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=pl.dtype, device=pl.device
        )

        # --- extrinsic -------------------------------------------------------
        twc = -Rwc @ C[:, None]  # (3,1)
        E = torch.cat((Rwc, twc), dim=1)  # (3,4)

        Ks.append(K)
        Es.append(E)

    return torch.stack(Ks, dim=0), torch.stack(Es, dim=0)
