

import torch
import torch.nn.functional as F
from torch.nn.functional import huber_loss

from vidfm3d.utils.metrics import batched_se3_to_relative_pose_error, calculate_auc
from vidfm3d.vggt.utils.pose_enc import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor,
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def camera_loss(
    pred_pose_enc_list,
    gt_intrinsic,  # (B, S, 3, 3)
    gt_extrinsic,  # (B, S, 3, 4)
    image_hw,  # last two dims of image
    loss_type="huber",
    gamma=0.6,
    pose_encoding_type="absT_quaR_FoV",
    weight_T=1.0,
    weight_R=1.0,
    weight_fl=0.5,
    return_metrics=False,
):
    # Extract predicted and ground truth components
    num_predictions = len(pred_pose_enc_list)
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsic, gt_intrinsic, image_hw, pose_encoding_type=pose_encoding_type
    )  # (B, S, 9)

    # We predict a list of pose encodings, later predictions are more important
    loss_T = loss_R = loss_fl = 0
    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)
        cur_pred_pose_enc = pred_pose_enc_list[i]

        # Compute the camera loss
        loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(
            cur_pred_pose_enc.clone(), gt_pose_encoding.clone(), loss_type=loss_type
        )
        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_dict = {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl,
    }

    if not return_metrics:
        return loss_dict

    # Computer the metrics: Rotation and Translation accuracy and AUC
    with torch.no_grad():
        # compute auc
        last_pred_pose_enc = pred_pose_enc_list[-1]  # (B, S, 9)
        last_pred_extrinsic, _ = pose_encoding_to_extri_intri(
            last_pred_pose_enc.detach(),
            image_hw,
            pose_encoding_type=pose_encoding_type,
            build_intrinsics=False,
        )  # (B, S, 3, 4)

        rel_rangle_deg, rel_tangle_deg = batched_se3_to_relative_pose_error(
            last_pred_extrinsic.float(),  # (B, S, 3, 4)
            gt_extrinsic.float(),  # (B, S, 3, 4)
        )

        thresholds = [5, 15]
        for threshold in thresholds:
            loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
            loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

        _, normalized_histogram = calculate_auc(
            rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
        )

        auc_thresholds = [30, 10, 5, 3]
        for auc_threshold in auc_thresholds:
            cur_auc = torch.cumsum(normalized_histogram[:auc_threshold], dim=0).mean()
            loss_dict[f"Auc_{auc_threshold}"] = cur_auc

    return loss_dict


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(
            dim=-1, keepdim=True
        )
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(
            cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3], reduction="none"
        )
        loss_R = huber_loss(
            cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7], reduction="none"
        )
        loss_fl = huber_loss(
            cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:], reduction="none"
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    assert pts3d.shape[:-1] == valid_mask.shape
    assert valid_mask.ndim == 4
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


def depth_loss(pred, conf, gt, mask=None, gradient_loss=None):
    """
    pred: B, S, H, W, 1
    conf: B, S, H, W
    gt: B, S, H, W, 1
    mask: B, S, H, W
    """
    conf_loss_dict = conf_loss(
        pred,
        conf,
        gt,
        mask,
        normalize_pred=False,
        normalize_gt=False,
        gradient_loss=gradient_loss,
        postfix="_depth",
    )

    return conf_loss_dict


def point_loss(
    pred,
    conf,
    gt,
    mask=None,
    normalize_pred=False,
    normalize_gt=False,
    gradient_loss=None,
):
    """
    pred: B, S, H, W, 3
    conf: B, S, H, W
    gt: B, S, H, W, 3
    mask: B, S, H, W
    """
    conf_loss_dict = conf_loss(
        pred,
        conf,
        gt,
        mask,
        normalize_pred=normalize_pred,
        normalize_gt=normalize_gt,
        gradient_loss=gradient_loss,
        postfix="_point",
    )
    return conf_loss_dict


def conf_loss(
    pts3d,
    pts3d_conf,
    gt_pts3d,
    valid_mask=None,
    normalize_gt=True,
    normalize_pred=True,
    gradient_loss=None,
    postfix="",
):
    if valid_mask is None:
        valid_mask = torch.ones_like(pts3d_conf)
    # normalize
    if normalize_gt:
        gt_pts3d, _ = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, _ = normalize_pointcloud(pts3d, valid_mask)

    (
        loss_reg_first_frame,
        loss_reg_other_frames,
        loss_grad_first_frame,
        loss_grad_other_frames,
    ) = reg_loss(
        pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss, conf=pts3d_conf
    )

    first_frame_conf = pts3d_conf[:, 0:1, ...]
    other_frames_conf = pts3d_conf[:, 1:, ...]
    first_frame_mask = valid_mask[:, 0:1, ...] > 0
    other_frames_mask = valid_mask[:, 1:, ...] > 0

    conf_loss_first_frame = loss_reg_first_frame * first_frame_conf[first_frame_mask]
    conf_loss_other_frames = (
        loss_reg_other_frames * other_frames_conf[other_frames_mask]
    )

    conf_loss_first_frame = check_and_fix_inf_nan(
        conf_loss_first_frame, f"conf_loss_first_frame{postfix}"
    )
    conf_loss_other_frames = check_and_fix_inf_nan(
        conf_loss_other_frames, f"conf_loss_other_frames{postfix}"
    )

    all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
    conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

    # for logging only
    conf_loss_first_frame = (
        conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
    )
    conf_loss_other_frames = (
        conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    )

    loss_dict = {
        f"loss{postfix}": conf_loss,
        f"loss_first{postfix}": conf_loss_first_frame,
        f"loss_other{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad_first{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad_other{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None, conf=None):
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...] > 0

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...] > 0

    if conf is not None:
        first_frame_conf = conf[:, 0:1, ...]
        other_frames_conf = conf[:, 1:, ...]

    loss_reg_first_frame = torch.norm(
        first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask],
        dim=-1,
    )
    loss_reg_other_frames = torch.norm(
        other_frames_gt_pts3d[other_frames_mask]
        - other_frames_pts3d[other_frames_mask],
        dim=-1,
    )

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(
            first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
            first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
            first_frame_mask.reshape(bb * ss, hh, ww),
            conf=first_frame_conf.reshape(bb * ss, hh, ww)
            if conf is not None
            else None,
        )
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(
            other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
            other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
            other_frames_mask.reshape(bb * ss, hh, ww),
            conf=other_frames_conf.reshape(bb * ss, hh, ww)
            if conf is not None
            else None,
        )
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(
            first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
            first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
            first_frame_mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=first_frame_conf.reshape(bb * ss, hh, ww)
            if conf is not None
            else None,
        )
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(
            other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
            other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
            other_frames_mask.reshape(bb * ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=other_frames_conf.reshape(bb * ss, hh, ww)
            if conf is not None
            else None,
        )
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(
        loss_reg_first_frame, "loss_reg_first_frame"
    )
    loss_reg_other_frames = check_and_fix_inf_nan(
        loss_reg_other_frames, "loss_reg_other_frames"
    )

    return (
        loss_reg_first_frame,
        loss_reg_other_frames,
        loss_grad_first_frame,
        loss_grad_other_frames,
    )


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = loss * conf
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.autocast("cuda", enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        pts = F.pad(
            point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0
        ).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Direction vectors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = (
            padded_mask[:, :-2, 1:-1]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 1:-1, :-2]
        )
        v2 = (
            padded_mask[:, 1:-1, :-2]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 2:, 1:-1]
        )
        v3 = (
            padded_mask[:, 2:, 1:-1]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, 1:-1, 2:]
        )
        v4 = (
            padded_mask[:, 1:-1, 2:]
            & padded_mask[:, 1:-1, 1:-1]
            & padded_mask[:, :-2, 1:-1]
        )

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def gradient_loss(prediction, target, mask, conf=None):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W
    # conf: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = grad_x * conf_x
        grad_y = grad_y * conf_y

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_loss_multi_scale(
    prediction, target, mask, scales=4, gradient_loss_fn=gradient_loss, conf=None
):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )

    total = total / scales
    return total
