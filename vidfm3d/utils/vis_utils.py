import os

import cv2
import matplotlib
import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation


def vfm_pca_images(
    feat_1d: torch.Tensor,
    tp: int,
    hp: int,
    wp: int,
    hi: int,
    wi: int,
    return_pil: bool = False,
):
    """
    Build ONE global PCA basis from *all* video-features in `feat_1d`
    and convert every frame’s tokens into an RGB map.

    Args
    ----
    feat_1d : (N, C)   where N  = tp * hp * wp   (flattened tokens of the clip)
    tp      : int      number of temporal positions (frames kept by the VFM)
    hp, wp  : int      token grid height / width   (= H//patch, W//patch)
    hi, wi  : int      original image height / width (for up-sampling)
    return_pil : bool  if True, returns list[ PIL.Image ]; otherwise list[ np.uint8 ]

    Returns
    -------
    list of length `tp`, each element an RGB visualisation (H, W, 3) uint8
    """
    assert feat_1d.ndim == 2, "feat_1d must be (N,C)"
    with torch.no_grad():
        X = feat_1d.float()  # promote if fp16
        X -= X.mean(dim=0, keepdim=True)  # centre tokens

        # PCA via SVD  – X = U @ S @ Vᵀ   →  take first 3 columns of V
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: (C,C)
        proj = X @ Vh[:3].T  # (N,3)

        # map each channel to [0,1] jointly
        q05 = torch.quantile(proj, 0.05, dim=0, keepdim=True)
        q95 = torch.quantile(proj, 0.95, dim=0, keepdim=True)
        proj = (proj - q05) / (q95 - q05 + 1e-8)

        # reshape to (T, 3, hp, wp)
        proj = proj.reshape(tp, hp, wp, 3).permute(0, 3, 1, 2).cpu().clamp(0, 1)

        out = []
        to_pil = torchvision.transforms.ToPILImage()
        for i in range(tp):
            im = to_pil(proj[i])
            im = im.resize((wi, hi), Image.NEAREST)  # keep blocky look
            out.append(im if return_pil else np.asarray(im))
    return out


# ----------------------------------------------------------------- #
#  main public entry point
# ----------------------------------------------------------------- #
def save_scene_glb(
    pmaps: torch.Tensor,  # [S,3,H,W]  – xyz in *world* coords
    images: torch.Tensor,  # [S,3,H,W]  – rgb in [0,1]
    conf: torch.Tensor,  # [S,1,H,W]  – confidence
    extrinsics: torch.Tensor,  # [S,3,4]    – world→cam
    save_path: str,
    conf_percentile: float = 75.0,  # drop the lowest X % confidences
    max_points: int = 16384,
) -> None:
    """
    Writes <save_path>.glb containing coloured points **and** camera frustums.

    Nothing is returned.  Any existing file is silently overwritten.
    """

    # 1) flatten the tensors -------------------------------------------------
    pmaps = pmaps.detach().cpu()  # (S,3,H,W)
    images = images.detach().cpu()  # (S,3,H,W)
    conf = conf.detach().cpu()  # (S,1,H,W)

    verts = pmaps.permute(0, 2, 3, 1).reshape(-1, 3).numpy()  # (N,3)
    colors = images.permute(0, 2, 3, 1).reshape(-1, 3).numpy()  # (N,3)
    confs = conf.permute(0, 2, 3, 1).reshape(-1).numpy()  # (N,)

    # 2) confidence & random sub-sampling -----------------------------------
    thr = np.percentile(confs, conf_percentile)
    keep = (confs >= thr) & (confs > 1e-6)
    verts, colors = verts[keep], colors[keep]

    if 0 < max_points < len(verts):
        idx = np.random.choice(len(verts), max_points, replace=False)
        verts, colors = verts[idx], colors[idx]

    # 3) build the trimesh.Scene --------------------------------------------
    scene = trimesh.Scene()
    if len(verts) == 0:  # edge-case safeguard
        verts = np.array([[0.0, 0.0, 0.0]])
        colors = np.array([[1.0, 1.0, 1.0]])

    colors_255 = (colors * 255).clip(0, 255).astype(np.uint8)
    scene.add_geometry(trimesh.PointCloud(vertices=verts, colors=colors_255))

    # 4) add cameras ---------------------------------------------------------
    extr = extrinsics.detach().cpu().numpy()  # (S,3,4)
    num_cam = extr.shape[0]
    cm = matplotlib.colormaps.get_cmap("gist_rainbow")

    scene_scale = np.linalg.norm(
        np.percentile(verts, 95, axis=0) - np.percentile(verts, 5, axis=0)
    )
    scene_scale = scene_scale if scene_scale > 0 else 1.0

    for i in range(num_cam):
        w2c = np.eye(4)
        w2c[:3, :4] = extr[i]
        c2w = np.linalg.inv(w2c)

        rgba = cm(i / num_cam)
        rgb = tuple(int(255 * x) for x in rgba[:3])

        _integrate_camera(scene, c2w, rgb, scene_scale)

    # (optional) align everything w.r.t. the first camera’s view
    scene.apply_transform(_scene_alignment(extr[0]))

    # 5) dump to disk --------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scene.export(save_path)  # format inferred from extension
    print(f"[save_scene_glb] wrote {save_path}")


# =======================================================================
#                     -------- helpers --------
# =======================================================================


def _integrate_camera(scene, c2w, rgb, scale):
    """
    Adds a small cone that roughly looks like a camera frustum.
    `c2w` is a 4×4 camera-to-world matrix.
    """
    cam_w = scale * 0.05
    cam_h = scale * 0.10

    cone = trimesh.creation.cone(radius=cam_w, height=cam_h, sections=4)
    # make the cone point along -Z like a pin-hole camera
    tip_rot = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    tip_tf = np.eye(4)
    tip_tf[:3, :3] = tip_rot
    tip_tf[2, 3] = -cam_h
    cone.apply_transform(tip_tf)

    # send to world frame
    cone.apply_transform(c2w @ _opengl_mat())

    cone.visual.face_colors[:, :3] = rgb
    scene.add_geometry(cone)


def _scene_alignment(first_extrinsic_3x4):
    """Rotate axes so +y is up and camera-0 sits at origin looking forward."""
    T = np.eye(4)
    T[:3, :4] = first_extrinsic_3x4
    c2w = np.linalg.inv(T)

    flip = _opengl_mat()  # (y,z) axis swap for OpenGL
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    return c2w @ flip @ rot


def _opengl_mat():
    """NDC → OpenGL handedness flip (y, z)."""
    m = np.eye(4)
    m[1, 1] = -1
    m[2, 2] = -1
    return m


def dump_viser_artifact(
    pmaps: torch.Tensor,  # [S,3,H,W]  world XYZ
    images: torch.Tensor,  # [S,3,H,W]  RGB in [0,1]
    conf: torch.Tensor,  # [S,1,H,W]  confidence
    extrinsics: torch.Tensor,  # [S,3,4]    world→cam
    out_dir: str,
    max_pts: int = 50_000,
    drop_percent: float = 75.0,  # drop lowest X % confidence (unused now, kept for compatibility)
):
    """Write <out_dir>/{points.npz,cameras.npy,images/*.png}."""
    os.makedirs(out_dir, exist_ok=True)

    # ---------- prepare point cloud -------------------------------------------------
    pts = pmaps.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    cols = (images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy() * 255).astype(
        np.uint8
    )
    confs = conf.permute(0, 2, 3, 1).reshape(-1).cpu().numpy()

    # Only subsample if we have too many points, but keep all confidence values
    if 0 < max_pts < len(pts):
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        cols = cols[idx]
        confs = confs[idx]

    # Save as NPZ with xyz, colors, and confidence
    np.savez_compressed(
        os.path.join(out_dir, "points.npz"),
        xyz=pts.astype(np.float32),
        colors=cols,
        confidence=confs.astype(np.float32),
    )

    # ---------- cameras & images -----------------------------------------------------
    np.save(
        os.path.join(out_dir, "cameras.npy"),
        extrinsics.cpu().numpy().astype(np.float32),
    )

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(images.cpu()):
        cv2.imwrite(
            f"{img_dir}/frame_{i:03d}.png",
            cv2.cvtColor(
                (img * 255).byte().permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR
            ),
        )
