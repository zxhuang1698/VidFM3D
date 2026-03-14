# vidfm3d/utils/viser_view.py
import argparse
import glob
import math
import os
import time

import cv2
import numpy as np
import trimesh
import viser
import viser.transforms as vtf

# ——————————————————————— presets (position only: y = +2 is “above ground”) ——————————————————————
_PRESET_POS = {
    "Front": (-0.0, -4.0, -4.0),
    "Back": (-0.0, -4.0, 4.0),
    "Left": (-4.0, -4.0, 0.0),
    "Right": (4.0, -4.0, 0.0),
    "Iso-": (-4.0, -4.0, 4.0),
    "Iso+": (4.0, -4.0, 4.0),
}


def _quat_look_at(
    pos: np.ndarray,
    target: np.ndarray = np.zeros(3),
    up: np.ndarray = np.array([0.0, 1.0, 0.0]),
):
    """Quaternion that makes -Z face *target* with +Y = *up* (OpenGL‑style)."""
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    R = np.stack([r, u, f], axis=1)  # world→cam
    return vtf.SO3.from_matrix(R).wxyz


# ──────────────────────────────────────────────────────────────────────────────
def launch_viser_from_artifact(pkg_dir: str, port: int = 8080):
    # Load point cloud data
    points_file = os.path.join(pkg_dir, "points.npz")
    if os.path.exists(points_file):
        # Load NPZ with confidence
        data = np.load(points_file)
        pts_full = data["xyz"].astype(np.float64)  # Ensure float64 for better precision
        colors_full = data["colors"]
        confidence_full = data["confidence"].astype(np.float64)
    else:
        # Fallback to PLY (legacy)
        cloud = trimesh.load(os.path.join(pkg_dir, "points.ply"))
        pts_full, colors_full = cloud.vertices.astype(np.float64), cloud.colors[:, :3]
        confidence_full = np.ones(len(pts_full), dtype=np.float64)  # dummy confidence

    # Ensure colors are uint8 in [0,255] range (as expected by viser)
    if colors_full.dtype != np.uint8:
        if colors_full.max() <= 1.0:
            colors_full = (colors_full * 255).astype(np.uint8)
        else:
            colors_full = colors_full.astype(np.uint8)

    extr = np.load(os.path.join(pkg_dir, "cameras.npy"))
    imgs = sorted(glob.glob(os.path.join(pkg_dir, "images/*.png")))
    imgs_rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in imgs]
    S, H, W = len(imgs_rgb), imgs_rgb[0].shape[0], imgs_rgb[0].shape[1]

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(control_layout="collapsible")

    # Compute scene center and recenter points (similar to reference code)
    scene_center = np.mean(pts_full, axis=0)
    pts_full_centered = pts_full - scene_center

    # Confidence statistics
    conf_min, conf_max = float(confidence_full.min()), float(confidence_full.max())
    default_percentile = 50.0

    # Initialize with default filtering
    def _filter_points(percentile: float):
        """Filter points based on confidence percentile (similar to reference)"""
        threshold = np.percentile(confidence_full, percentile)
        mask = (confidence_full >= threshold) & (confidence_full > 1e-5)
        return pts_full_centered[mask], colors_full[mask], mask.sum(), threshold

    # Create initial point cloud
    pts, colors, n_visible, threshold = _filter_points(default_percentile)

    # Add confidence controls
    with server.gui.add_folder("Point Cloud"):
        conf_slider = server.gui.add_slider(
            "Confidence %ile",
            min=0.0,
            max=100.0,
            step=0.1,  # Finer step like in reference
            initial_value=default_percentile,
        )
        points_counter = server.gui.add_text(
            "points shown", f"{n_visible:,} / {len(pts_full):,}"
        )
        conf_range = server.gui.add_text(
            "conf range", f"{conf_min:.4f} - {conf_max:.4f}", disabled=True
        )
        threshold_display = server.gui.add_text(
            "threshold", f"{threshold:.4f}", disabled=True
        )

    # Create point cloud object with proper parameters
    pcd = server.scene.add_point_cloud(
        name="pts",
        points=pts,
        colors=colors,
        point_size=0.0025,  # Smaller point size like reference
        point_shape="circle",
    )

    # Update point cloud when slider changes - use direct assignment like reference
    @conf_slider.on_update
    def _(event):
        pts_new, colors_new, n_visible, threshold_val = _filter_points(
            conf_slider.value
        )
        # Direct assignment like in reference code
        pcd.points = pts_new
        pcd.colors = colors_new
        points_counter.value = f"{n_visible:,} / {len(pts_full):,}"
        threshold_display.value = f"{threshold_val:.4f}"

    # ───────── per‑client GUI ─────────
    @server.on_client_connect
    def _(cli: viser.ClientHandle):
        dd = cli.gui.add_dropdown("preset view", list(_PRESET_POS), "Front")

        with cli.gui.add_folder("Camera pose"):
            pos_lbl = cli.gui.add_text("position", "", disabled=True)
            rot_lbl = cli.gui.add_text("rotation (w x y z)", "", disabled=True)
            pose_in = cli.gui.add_text("pose string (editable)", "")

        lock = {"busy": False}

        # helpers ------------------------------------------------------------
        def _update_labels():
            lock["busy"] = True  # ➊ suppress callbacks while we write
            p = cli.camera.position
            q = cli.camera.wxyz
            pos_lbl.value = f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}"
            rot_lbl.value = f"{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"
            pose_in.value = (
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"
            )
            lock["busy"] = False  # ➋ re‑enable callbacks

        def _apply_pose(pos: np.ndarray, quat: np.ndarray):
            quat /= np.linalg.norm(quat)
            with cli.atomic():
                cli.camera.position = pos
                cli.camera.wxyz = tuple(quat)
                cli.camera.look_at = (0, 0, 0)

        # preset dropdown
        @dd.on_update
        def _(_e):
            pos = np.array(_PRESET_POS[dd.value], float)
            _apply_pose(pos, _quat_look_at(pos))
            _update_labels()

        # textbox → camera
        @pose_in.on_update
        def _(_e):
            if lock["busy"]:
                return
            try:
                vals = [float(v) for v in pose_in.value.replace(",", " ").split()]
                if len(vals) == 7:
                    _apply_pose(np.array(vals[:3]), np.array(vals[3:]))
                    _update_labels()
            except ValueError:
                pass  # ignore malformed input

        # camera → labels / textbox
        @cli.camera.on_update
        def _(_e):
            lock["busy"] = True
            _update_labels()
            lock["busy"] = False

        # initialise on first connect
        init_pos = np.array(_PRESET_POS["Front"], float)
        _apply_pose(init_pos, _quat_look_at(init_pos))
        _update_labels()

    # ───────── scene objects (cameras) ─────────
    gui_cam = server.gui.add_checkbox("show cameras", True)

    frames, frust = [], []
    for i in range(S):
        E = extr[i] if extr[i].shape == (4, 4) else np.vstack([extr[i], [0, 0, 0, 1]])
        # Adjust camera positions relative to scene center
        c2w = vtf.SE3.from_matrix(np.linalg.inv(E))
        cam_pos = c2w.translation() - scene_center

        frm = server.scene.add_frame(
            f"cam{i}",
            wxyz=c2w.rotation().wxyz,
            position=cam_pos,
            axes_length=0.05,
            axes_radius=0.002,
        )
        frames.append(frm)

        fov = 2 * math.atan2(H / 2, 1.1 * H)
        fr = server.scene.add_camera_frustum(
            f"cam{i}/frustum",
            fov=fov,
            aspect=W / H,
            scale=0.05,
            image=imgs_rgb[i],
            line_width=1.0,
        )
        frust.append(fr)

        @fr.on_click
        def _(evt, frm=frm):
            for cli in server.get_clients().values():
                cli.camera.wxyz, cli.camera.position = frm.wxyz, frm.position

    gui_cam.on_update(
        lambda _: [setattr(o, "visible", gui_cam.value) for o in (*frames, *frust)]
    )

    print(f"Viser up on http://localhost:{port} (CTRL‑C to quit)")
    while True:
        time.sleep(0.05)


# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--port", type=int, default=8080)
    launch_viser_from_artifact(p.parse_args().scene, port=p.parse_args().port)
