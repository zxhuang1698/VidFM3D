import os
from typing import Optional, Tuple, Union

import numpy as np
import pytorch3d
import roma
import torch
import trimesh
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


# normalize the pointcloud with per-axis std
@torch.no_grad()
def normalize_pc(points, scale=1.0):
    """
    args:
        points: [N, 3]
        scale: float, linearly scale the size of the pointcloud
    return:
        points_normalized: [N, 3]
    """
    points_std = points[torch.isfinite(points.sum(dim=-1))].std(dim=0)  # [3]
    if points_std.abs().min() < 1.0e-7:
        return points
    points = points / points_std.mean()
    points_mean = points[torch.isfinite(points.sum(dim=-1))].mean(dim=0)  # [3]
    points_normalized = points - points_mean
    return points_normalized * scale


# dump two pointclouds for comparison
def dump_pointclouds_compare(pc1, pc2, path="./pc.ply"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pc1 = pc1.cpu().numpy()  # [N1, 3]
    pc1[pc1 == np.inf] = 0
    pc2 = pc2.cpu().numpy()  # [N2, 3]
    pc2[pc2 == np.inf] = 0
    color1 = np.zeros(pc1.shape).astype(np.uint8)
    color1[:, 0] = 255
    color2 = np.zeros(pc2.shape).astype(np.uint8)
    color2[:, 1] = 255
    pc_vertices = np.vstack([pc1, pc2])
    colors = np.vstack([color1, color2])
    pc_color = trimesh.points.PointCloud(vertices=pc_vertices, colors=colors)
    pc_color.export(path)


# https://github.com/facebookresearch/pytorch3d
def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


@torch.no_grad()
def chamfer_dist(x, y, x_lengths=None, y_lengths=None, norm=2):
    """
    Calculate the euclidean chamfer distance between two pointclouds.
    args:
        x: [B, N1, 3]
        y: [B, N2, 3]
    return:
        cham_x: [B, N1], accuracy
        cham_y: [B, N2], completeness
        idx_x: [B, N1], nearest indices for each point of x in y
        idx_y: [B, N2], nearest indices for each point of y in x
    """
    x, x_lengths, _ = _handle_pointcloud_input(x, x_lengths, None)
    y, y_lengths, _ = _handle_pointcloud_input(y, y_lengths, None)

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    idx_x = x_nn.idx[..., 0]  # (N, P1)
    idx_y = y_nn.idx[..., 0]  # (N, P2)

    return cham_x.sqrt(), cham_y.sqrt(), idx_x, idx_y


# calculate the F-score based on the chamfer distance
@torch.no_grad()
def compute_fscore(dist1, dist2, thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    """
    args:
        dist1: [B, N1]
        dist2: [B, N2]
        thresholds: list
    return:
        fscores: [B, len(thresholds)]
    """
    fscores = []
    for threshold in thresholds:
        precision = torch.mean((dist1 < threshold).float(), dim=1)  # [B, ]
        recall = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore)] = 0
        fscores.append(fscore)
    fscores = torch.stack(fscores, dim=1)
    return fscores


def azim_to_rotation_matrix(azim, representation="angle"):
    """Azim is angle with vector +X, rotated in XZ plane"""
    if representation == "rad":
        # [B, ]
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == "angle":
        # [B, ]
        azim = azim * np.pi / 180
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == "trig":
        # [B, 2]
        cos, sin = azim[:, 0], azim[:, 1]
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R


def elev_to_rotation_matrix(elev, representation="angle"):
    """Angle with vector +Z in YZ plane"""
    if representation == "rad":
        # [B, ]
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == "angle":
        # [B, ]
        elev = elev * np.pi / 180
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == "trig":
        # [B, 2]
        cos, sin = elev[:, 0], elev[:, 1]
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    R[:, 1, 1:] = torch.stack([cos, -sin], dim=-1)
    R[:, 2, 1:] = torch.stack([sin, cos], dim=-1)
    return R


def roll_to_rotation_matrix(roll, representation="angle"):
    """Angle with vector +X in XY plane"""
    if representation == "rad":
        # [B, ]
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == "angle":
        # [B, ]
        roll = roll * np.pi / 180
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == "trig":
        # [B, 2]
        cos, sin = roll[:, 0], roll[:, 1]
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R


def get_rotation_sphere(
    azim_sample=4, elev_sample=4, roll_sample=4, scales=[1.0], device="cuda"
):
    rotations = []
    azim_range = [0, 360]
    elev_range = [0, 360]
    roll_range = [0, 360]
    azims = np.linspace(azim_range[0], azim_range[1], num=azim_sample, endpoint=False)
    elevs = np.linspace(elev_range[0], elev_range[1], num=elev_sample, endpoint=False)
    rolls = np.linspace(roll_range[0], roll_range[1], num=roll_sample, endpoint=False)
    for scale in scales:
        for azim in azims:
            for elev in elevs:
                for roll in rolls:
                    Ry = azim_to_rotation_matrix(torch.tensor([azim]))
                    Rx = elev_to_rotation_matrix(torch.tensor([elev]))
                    Rz = roll_to_rotation_matrix(torch.tensor([roll]))
                    R_permute = (
                        torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
                        .float()
                        .to(Ry.device)
                        .unsqueeze(0)
                        .expand_as(Ry)
                    )
                    R = scale * Rz @ Rx @ Ry @ R_permute
                    rotations.append(R.to(device).float())
    return torch.cat(rotations, dim=0)


def eval_pc(
    pred, gt, f_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5], rgb_pred=None, rgb_gt=None
):
    # pred: [B, N1, 3]
    # gt: [B, N2, 3]
    # rgb_pred: [B, N1, 3]
    # rgb_gt: [B, N2, 3]
    dist_acc, dist_comp, idx_pred, _ = chamfer_dist(pred, gt)
    f_score = compute_fscore(dist_acc, dist_comp, f_thresholds)  # [B, n_threshold]
    if rgb_pred is not None and rgb_gt is not None:
        nearest_rgb = knn_gather(rgb_gt, idx_pred.unsqueeze(-1)).squeeze(
            -2
        )  # [B, N1, 3]
        l2_dist = (
            ((rgb_pred - nearest_rgb) ** 2).view(pred.shape[0], -1).mean(dim=1)
        )  # [B]
        psnr = 20 * torch.log10(1.0 / (l2_dist**0.5)).clamp(max=2)
    else:
        psnr = None
    return dist_acc.mean(dim=1), dist_comp.mean(dim=1), f_score, psnr


def evaluate_sample(
    pc_pred,
    pc_gt,
    f_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5],
    device="cuda",
    icp_align=True,
):
    pc_pred = pc_pred.to(device).unsqueeze(0).float()
    pc_gt = pc_gt.to(device).unsqueeze(0).float()
    xyz_pred, rgb_pred = pc_pred[..., :3], pc_pred[..., 3:]
    xyz_gt, rgb_gt = pc_gt[..., :3], pc_gt[..., 3:]
    xyz_gt = normalize_pc(xyz_gt)

    # brute force evaluation
    # get the best CD by iterating over the pose
    best_cd = np.inf
    rotations = get_rotation_sphere(
        azim_sample=24, elev_sample=24, roll_sample=12, scales=[1.0]
    )
    batch_size = 48

    # process the pointcloud in batches
    for i in range(0, len(rotations), batch_size):
        rotation_batch = rotations[i : i + batch_size].to(device)
        xyz_pred_rotated = (
            rotation_batch
            @ xyz_pred.repeat(rotation_batch.shape[0], 1, 1).permute(0, 2, 1)
        ).permute(0, 2, 1)
        xyz_pred_rotated = normalize_pc(xyz_pred_rotated).contiguous()
        acc, comp, f_score, psnr = eval_pc(
            xyz_pred_rotated,
            xyz_gt.repeat(rotation_batch.shape[0], 1, 1),
            f_thresholds,
            rgb_pred,
            rgb_gt.repeat(rotation_batch.shape[0], 1, 1),
        )
        cd = (acc + comp) / 2
        # save the best cd
        for j in range(len(cd)):
            if cd[j] < best_cd:
                best_xyz_pred = xyz_pred_rotated[j].clone()
                best_acc = acc[j]
                best_comp = comp[j]
                best_cd = cd[j]
                best_fscore = f_score[j]
                best_psnr = psnr[j]
                best_rotation = rotation_batch[j].clone()

    if icp_align:
        # further register the pointcloud with ICP
        _, best_xyz_pred, _ = trimesh.registration.icp(
            best_xyz_pred.cpu().numpy(), xyz_gt[0].cpu().numpy(), scale=False
        )
        best_xyz_pred = torch.tensor(best_xyz_pred).float().to(best_rotation)
        # re-calculate the CD
        best_acc, best_comp, best_fscore, best_psnr = eval_pc(
            best_xyz_pred.unsqueeze(0),
            xyz_gt,
            f_thresholds,
            rgb_pred,
            rgb_gt,
        )
        best_acc, best_comp, best_fscore = best_acc[0], best_comp[0], best_fscore[0]
    best_pc_pred = torch.cat([best_xyz_pred.unsqueeze(0), rgb_pred], dim=-1).squeeze(
        0
    )  # [N, 3+3]
    pc_gt = torch.cat([xyz_gt, rgb_gt], dim=-1).squeeze(0)  # [N, 3+3]
    return best_acc, best_comp, best_fscore, best_psnr, best_pc_pred, pc_gt


@torch.no_grad()
def build_mask(
    conf: torch.Tensor,  # [S,1,H,W]         – confidence
    fg_masks: Optional[torch.Tensor] = None,  # [S,1,H,W] or None
    thresh: float = 0.75,
) -> torch.Tensor:
    """
    Returns a single boolean mask [S,1,H,W] that keeps:
       • pixels whose confidence is within the top `thresh` quantile
       • AND (optionally) inside an external foreground mask
    """
    q = np.percentile(conf.view(-1).cpu().numpy(), thresh * 100)
    mask = conf >= q
    if fg_masks is not None:
        mask = mask & (fg_masks > 0)
    return mask  # bool tensor, same shape as conf


@torch.no_grad()
def pmaps_to_pc(
    pmaps: torch.Tensor,  # [S,3,H,W]
    images: torch.Tensor,  # [S,3,H,W]  (RGB in [0,1])
    mask: torch.Tensor,  # [S,1,H,W]  boolean
    max_points: int = 4096,
) -> torch.Tensor:
    """
    Returns an (N,6) tensor [x,y,z,r,g,b] with at most `max_points` rows.
    """
    S, _, H, W = pmaps.shape
    assert images.shape == (S, 3, H, W)
    assert mask.shape == (S, 1, H, W)

    # flatten & filter
    pts = pmaps.permute(0, 2, 3, 1)[mask.squeeze(1)].reshape(-1, 3)  # [M,3]
    rgb = images.permute(0, 2, 3, 1)[mask.squeeze(1)].reshape(-1, 3)  # [M,3]

    if max_points > 0 and pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0], device=pts.device)[:max_points]
        pts, rgb = pts[idx], rgb[idx]

    return torch.cat([pts, rgb], dim=-1)  # [N,6] on same device


def align_pmaps(
    pred_pmaps: torch.Tensor,  # [S,3,H,W]  in world-coords
    gt_pmaps: torch.Tensor,  # [S,3,H,W]
    mask: Optional[torch.Tensor] = None,  # [S,1,H,W] or None
) -> torch.Tensor:
    """
    Returns the aligned predicted point-map to GT.

    We solve for a similarity transform  (R,t,s)  via
    roma.rigid_points_registration and apply it to *all* predicted
    points before computing the MSE.

    A simple binary mask (confidence ⋅ fg) can be provided to discard
    low-quality pixels when estimating the transform.
    """
    pred_pmaps = pred_pmaps.float()  # [S,3,H,W]
    gt_pmaps = gt_pmaps.float()  # [S,3,H,W]
    device = pred_pmaps.device
    S, _, H, W = pred_pmaps.shape

    # (S,3,H,W) → (N,3)
    pred_flat = pred_pmaps.permute(0, 2, 3, 1).reshape(-1, 3)
    gt_flat = gt_pmaps.permute(0, 2, 3, 1).reshape(-1, 3)

    if mask is not None:
        m = mask.bool().view(-1)  # [N]
    else:
        m = torch.ones(pred_flat.shape[0], dtype=torch.bool, device=device)

    # Use only masked points to estimate R|t|s – but transform *everything*
    src = pred_flat[m]  # (M,3)
    tgt = gt_flat[m]  # (M,3)

    if src.shape[0] < 3:  # degeneracy guard
        aligned = pred_flat
    else:
        with torch.amp.autocast("cuda", enabled=False):
            R, t, s = roma.rigid_points_registration(src, tgt, compute_scaling=True)
        aligned = s * (pred_flat @ R.T) + t  # (N,3)

    aligned_map = aligned.reshape(S, H, W, 3).permute(0, 3, 1, 2)  # back to [S,3,H,W]

    return aligned_map
