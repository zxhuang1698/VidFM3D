import json
import logging
import math

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from vidfm3d.dust3r.datasets.base.easy_dataset import EasyDataset
from vidfm3d.vggt.utils.geometry import closed_form_inverse_se3, mat2plucker

logger = logging.getLogger(__name__)


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (
        np.float32,
        torch.float32,
        bool,
        np.int32,
        np.int64,
        np.uint8,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    ):
        return False, f"bad {v.dtype=}"
    return True, None


def to_hom_4x4(E_3x4):
    """Convert a (3,4) extrinsic to a (4,4) homogeneous matrix."""
    E_4x4 = torch.eye(4, dtype=E_3x4.dtype, device=E_3x4.device)
    E_4x4[:3, :4] = E_3x4
    return E_4x4


@torch.no_grad()
def invert_pose_ref_and_scale(
    extrinsics_3x4: torch.Tensor,  # (S,3,4)
    pointmaps_world: torch.Tensor,  # (S,H,W,3)
    depthmaps: torch.Tensor = None,  # (S,H,W,1)
    ref_idx: int = 0,
    scale_by_points: bool = True,
):
    """
    (1) Make extrinsics_3x4[ref_idx] become identity by right-multiplying each E_i by E_ref^-1.
    (2) Transform the 'world' pointmaps by E_ref so they live in the reference camera’s coords.
    (3) Optionally scale everything so that the average distance of valid points to the origin is ~1 (in ref frame).

    Returns:
      new_ex_3x4:  (S,3,4)
      new_points:  (S,H,W,3)
    """
    assert extrinsics_3x4.ndim == 3 and extrinsics_3x4.shape[1:] == (3, 4)
    assert pointmaps_world.ndim == 4 and pointmaps_world.shape[-1] == (3)
    S = extrinsics_3x4.shape[0]

    # 1) Convert (3,4)->(4,4)
    E_4x4_all = []
    for i in range(S):
        E_4x4_all.append(to_hom_4x4(extrinsics_3x4[i]))
    E_4x4_all = torch.stack(E_4x4_all, dim=0)  # (S,4,4)

    # 2) Invert ref camera’s extrinsic
    E_ref_4x4 = E_4x4_all[ref_idx]  # shape (4,4)
    E_ref_inv_4x4 = closed_form_inverse_se3(E_ref_4x4[None])[0]
    # or simply torch.inverse(E_ref_4x4) if you prefer

    # 3) Re-base all extrinsics => E'_i = E_i * E_ref^-1
    new_ex_4x4 = E_4x4_all @ E_ref_inv_4x4  # (S,4,4)
    new_ex_3x4 = new_ex_4x4[:, :3, :4]  # back to shape (S,3,4)

    # 4) Transform pointmaps from old-world coords into ref camera coords:
    #    X'_i = E_ref_4x4 * X, i.e. multiply by E_ref, since E_ref mapped oldWorld->camRef.
    new_points = []
    for i in range(S):
        pm = pointmaps_world[i]  # (H,W,3)
        pm_flat = pm.reshape(-1, 3)
        ones = torch.ones(pm_flat.shape[0], 1, dtype=pm.dtype, device=pm.device)
        pm_h = torch.cat([pm_flat, ones], dim=1)  # (N,4)
        # apply E_ref_4x4
        pm_trans_h = (E_ref_4x4 @ pm_h.T).T  # (N,4)
        pm_trans = pm_trans_h[:, :3].reshape(pm.shape)
        new_points.append(pm_trans)
    new_points = torch.stack(new_points, dim=0)  # (S,H,W,3)

    if not scale_by_points:
        return new_ex_3x4, new_points

    # 5) Optionally, scale the entire scene so that the average distance in ref frame is ~1
    #    We do this by computing mean distance and we scaling the extrinsics and the new_points accordingly.
    dist = new_points[ref_idx].norm(dim=-1)
    dist_mean = dist.mean()  # average distance

    # clamp to avoid blow-ups if the scene is basically empty
    if dist_mean < 1e-3 or dist_mean > 1e4:
        return new_ex_3x4, new_points

    scale = 1.0 / dist_mean

    # scale the extrinsics translation
    new_ex_3x4[:, :, 3] *= scale
    # scale the pointmaps
    new_points *= scale

    if depthmaps is not None:
        # scale the depthmaps
        depthmaps = depthmaps * scale

    return new_ex_3x4, new_points, depthmaps


class VideoProbeDataset(EasyDataset):
    def __init__(
        self,
        root: str,
        root_vfm: str,
        subset: str = "1K",
        split: str = "train",
        vfm_name: str = "wan",
        feat_postfix: str = "_t0_layer15",
        feat_pixalign: bool = False,
        seed: int = None,
        debug: bool = False,
        num_views: int = 4,
        min_view_interval: int = None,
        plucker_rescale: float = 0.2,
        context_len: int = 81,
        seen_ratio: float = 1.0,
        query_idx_divisor: int = None,
        use_mask: bool = False,
        **kwargs,
    ):
        assert vfm_name in [
            "wan",
            "dino",
            "f3r",
            "vjepa",
            "opensora",
            "cogvideox",
            "aether",
        ], f"Unknown vfm_name {vfm_name}"
        self.vfm_name = vfm_name
        self.feat_postfix = feat_postfix
        self.feat_pixalign = (
            feat_pixalign  # if true, only use the feature of the corresponding frame
        )
        self.num_views = num_views
        self.subset = subset
        self.min_view_interval = min_view_interval
        self.split = split
        self.seed = seed
        self.debug = debug
        self.plucker_rescale = plucker_rescale
        self.root_vfm = root_vfm
        self.use_mask = use_mask
        self.context_len = context_len  # coverage of each context window
        self.seen_ratio = seen_ratio  # ratio of seen frames in the context window
        self.query_idx_divisor = query_idx_divisor  # floor the query indices to 1+the nearest multiple of this divisor
        assert (
            0.0 <= self.seen_ratio <= 1.0
        ), f"seen_ratio {self.seen_ratio} out of range"
        self.kwargs = kwargs

        # convert subset to list
        if isinstance(subset, str) and subset.lower() == "ablate":
            subset = [
                "apple",
                "backpack",
                "bench",
                "bowl",
                "chair",
                "handbag",
                "hydrant",
                "motorcycle",
                "plant",
                "teddybear",
                "toytruck",
            ]
        elif isinstance(subset, str) and subset.lower() != "all":
            subset = [subset]
        elif isinstance(subset, str) and subset.lower() == "all":
            subset = [
                x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))
            ]
        if not isinstance(subset, list):
            raise ValueError(
                f"subset must be a string or a list of strings, got {type(subset)}"
            )

        # collect all scene paths
        split = "val" if split == "test" else split
        split_file = os.path.join(root, f"{split}.json")
        with open(split_file, "r") as f:
            pairs = json.load(f)  # list of [subset, hash]
            self.scenes = []
            for sub, h in pairs:
                if sub not in subset:
                    continue
                sft_path = os.path.join(root, sub, f"{h}.sft")
                if os.path.isfile(sft_path):
                    self.scenes.append(sft_path)
        logger.info(f"Found {len(self.scenes)} scenes in {split_file}")

        if self.debug:
            # try to load all the scenes and check the shape of the images
            image_shape = (294, 518)
            pointmap_shape = (294, 518, 3)
            confmap_shape = (294, 518, 1)
            depthmap_shape = (294, 518, 1)
            intrinsic_shape = (3, 3)
            extrinsic_shape = (3, 4)
            for scene in self.scenes:
                data = load_file(scene)
                images = data["images"].float()
                pointmaps = data["pointmaps"].float()
                confmaps = data["confmaps"].float()
                depthmaps = data["depthmaps"].float()
                intrinsics = data["intrinsic"].float()
                extrinsics = data["extrinsic"].float()
                if (
                    images.shape[2:] != image_shape
                    or pointmaps.shape[1:] != pointmap_shape
                    or confmaps.shape[1:] != confmap_shape
                    or depthmaps.shape[1:] != depthmap_shape
                    or intrinsics.shape[1:] != intrinsic_shape
                    or extrinsics.shape[1:] != extrinsic_shape
                ):
                    logger.warning(
                        f"Skipping scene {scene} with shape {images.shape} "
                        f"{pointmaps.shape} {confmaps.shape} {depthmaps.shape} "
                        f"{intrinsics.shape} {extrinsics.shape}"
                    )
                    self.scenes.remove(scene)

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} scenes"

    def __repr__(self):
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _sample_query_frames(
        self, rng, n, win_len, local_start: int = 0
    ) -> torch.Tensor:
        """
        Return `num_views` distinct frame indices in
            [local_start, local_start + win_len)
        such that consecutive indices differ by **at least**
        `self.min_view_interval`.

        The algorithm is O(num_views):
        1.  Check feasibility once.
        2.  Compute how many “slack” frames are left after enforcing the
            minimum gap:  slack = win_len-1 - (num_views-1)*min_gap.
        3.  Randomly split that slack over the (num_views-1) gaps.
        4.  Walk left→right, adding min_gap + slack_i each step.
        """
        min_gap = self.min_view_interval or 0

        # no constraint
        if min_gap <= 0:
            return (
                torch.linspace(
                    local_start, local_start + win_len - 1, n, dtype=torch.float32
                )
                .round()
                .to(torch.long)
            )

        # feasibility check
        needed = (n - 1) * min_gap + 1
        if needed > win_len:
            raise ValueError(
                f"Cannot sample {n} views with min_gap={min_gap} "
                f"inside a window of length={win_len}"
            )

        # random slack allocation
        slack = win_len - needed
        # Draw n-1 sorted cut points in [0, slack]
        # e.g. slack = 4 and n = 4, we may get cuts = [1, 1, 4]
        cuts = np.sort(rng.integers(0, slack + 1, size=n - 1, dtype=int))
        # add 0 and slack to the cuts, cuts is now [0, 1, 1, 4, 4]
        # then diff gives [1, 0, 3, 0], which is the step offset (always positive)
        # and the sum of this diff array is slack
        extras = np.diff(np.concatenate(([0], cuts, [slack])))

        # build indices left→right
        idxs = [local_start]
        for extra in extras[
            :-1
        ]:  # get rid of the last one so we don't always land on the last frame
            # add the minimum gap and the step offset
            idxs.append(idxs[-1] + min_gap + int(extra))
        # if everything works as expected, we should have n indices
        # and the last one should be exactly local_start + win_len - 1
        assert len(idxs) == n, (
            f"Expected {n} indices, got {len(idxs)}: {idxs} "
            f"with local_start={local_start}, win_len={win_len}, "
            f"min_gap={min_gap}, slack={slack}"
        )
        assert idxs[-1] <= local_start + win_len - 1, (
            f"Last index {idxs[-1]} is larger than "
            f"local_start + win_len - 1 = {local_start + win_len - 1}"
        )
        return torch.as_tensor(idxs, dtype=torch.long)

    def __getitem__(self, idx):
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        output = {}
        data = load_file(self.scenes[idx])

        # read them from the dict
        images = data["images"].float()  # (S, 3, H, W)
        masks = data.get("masks", None) if self.use_mask else None  # (S, H, W, 1)
        pointmaps = data["pointmaps"].float()  # (S, H, W, 3)
        confmaps = data["confmaps"].float()  # (S, H, W, 1)
        depthmaps = data["depthmaps"].float()  # (S, H, W, 1)
        intrinsics = data["intrinsic"].float()  # (S, 3, 3)
        extrinsics = data["extrinsic"].float()  # (S, 3, 4)

        # get the vfm feature path
        scene_hash = Path(self.scenes[idx]).stem
        current_subset = self.scenes[idx].split("/")[-2]
        vfm_feat_path = os.path.join(
            self.root_vfm,
            self.vfm_name,
            current_subset,
            scene_hash,
            f"feature{self.feat_postfix}.sft",
        )

        # seen: pick the query frames inside the context window
        # unseen: pick the query frames outside the context window
        total_frames = images.shape[0]
        if self.feat_pixalign:
            n_seen = self.num_views
            n_unseen = 0
        else:
            n_seen = int(self.num_views * self.seen_ratio)
            n_unseen = self.num_views - n_seen
        sel = []
        sel_seen = (
            self._sample_query_frames(self._rng, n_seen, self.context_len, 0)
            if n_seen > 0
            else None
        )
        unseen_start = (
            self._rng.integers(
                self.context_len,
                total_frames
                - self.num_views
                * (self.min_view_interval if self.min_view_interval else 1),
            )
            if n_unseen > 0
            else 0
        )
        sel_unseen = (
            self._sample_query_frames(
                self._rng, n_unseen, total_frames - int(unseen_start), int(unseen_start)
            )
            if n_unseen > 0
            else None
        )
        if sel_seen is not None:
            sel.append(sel_seen)
        if sel_unseen is not None:
            sel.append(sel_unseen)
        # concatenate the two selections
        sel = torch.cat(sel, dim=0)
        if self.query_idx_divisor is not None:
            # floor the indices to 1+the nearest multiple of query_idx_divisor
            sel = (
                torch.floor((sel - 1) / self.query_idx_divisor) * self.query_idx_divisor
                + 1
            )
            sel = sel.clamp(min=0).long()  # ensure indices are non-negative integers

        # feature loading -- two steps:
        # load the desired feature as a single tensor
        # record the mapping from each queried frame to the corresponding feature frame id
        if self.vfm_name in ["wan", "opensora"]:
            # load WAN/opensora features
            vfm_feat = load_file(vfm_feat_path)["feat"]  # (T, H, W, C)
            assert vfm_feat.shape[0] == 21
            if self.context_len < 81:
                end_idx = math.ceil(float(self.context_len - 1) / (81 - 1) * (21 - 1))
                vfm_feat = vfm_feat[: end_idx + 1]  # (T, H, W, C)
            # calculate the query-to-key mapping
            # the WAN features are composed of a single first frame feature + T // 4 features
            vfm_idx = (
                torch.floor((sel - 1).float() / (81 - 1) * (21 - 1)).long() + 1
            )  # e.g. 0 -> 0, [1-4] -> 1, ...
            if self.feat_pixalign:
                vfm_feat = vfm_feat[vfm_idx]
                vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)
        elif self.vfm_name in ["cogvideox", "aether"]:
            # load CogVideoX/Aether features
            # CogVideoX: two chunks, 49 frames each, shared first frame
            # Aether: two chunks, 41 frames each, shared first frame
            vfm_feat_5d = load_file(vfm_feat_path)["feat"]  # (N,T,H,W,C)
            n_frames = vfm_feat_5d.shape[1]
            assert n_frames in [
                11,
                13,
            ], f"Expected 11 or 13 feature frames, got {n_frames} in {vfm_feat_path}"
            n_chunks = vfm_feat_5d.shape[0]
            vfm_feat = vfm_feat_5d.view(-1, *vfm_feat_5d.shape[2:])  # (N*T,H,W,C)

            T = vfm_feat.shape[0]

            # map each queried frame-index ‘sel’ → token-index in [0,T)
            vfm_idx = torch.zeros_like(sel)  # placeholder
            mask0 = sel == 0  # frame-0 → token-0
            vfm_idx[mask0] = 0

            mask_rest = ~mask0
            if mask_rest.any():
                offs = sel[mask_rest] - 1  # shift so frame-1 → 0
                clip_idx = offs % n_chunks  # 0,1  ←→  2 chunks
                pos_in_clip = offs // n_chunks + 1  # 1…T-1  (0 is frame-0)
                token_id = (pos_in_clip - 1) // 4 + 1
                vfm_idx[mask_rest] = (
                    clip_idx * n_frames + token_id
                )  # final index ∈ [0,31]

            if self.feat_pixalign:
                vfm_feat = vfm_feat[vfm_idx]
                vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)
            else:
                raise NotImplementedError
        elif self.vfm_name == "vjepa":
            vfm_feat_5d = load_file(vfm_feat_path)["feat"]  # (N,8,H,W,C)
            n_chunks = vfm_feat_5d.shape[0]
            vfm_feat = vfm_feat_5d.view(-1, *vfm_feat_5d.shape[2:])  # (N*8,H,W,C)
            if self.feat_postfix == "_chunked":
                assert self.context_len <= 80
                clip_idx = sel // 16  # 0,1,2,3,4  ←→  5 chunks
                token_id = (sel % 16) // 2  # stride-2 → 0…7
                vfm_idx = clip_idx * 8 + token_id
            else:
                if self.context_len < 76:
                    raise NotImplementedError
                T = vfm_feat.shape[0]

                # map each queried frame-index ‘sel’ → token-index in [0,T)
                vfm_idx = torch.zeros_like(sel)  # placeholder
                mask0 = sel == 0  # frame-0 → token-0
                vfm_idx[mask0] = 0

                mask_rest = ~mask0
                if mask_rest.any():
                    offs = sel[mask_rest] - 1  # shift so frame-1 → 0
                    clip_idx = offs % n_chunks  # 0,1,2,3,4  ←→  5 chunks
                    pos_in_clip = offs // n_chunks + 1  # 1…15  (0 is frame-0)
                    token_id = pos_in_clip // 2  # stride-2 → 0…7
                    vfm_idx[mask_rest] = clip_idx * 8 + token_id  # final index ∈ [0,31]

            if self.feat_pixalign:
                vfm_feat = vfm_feat[vfm_idx]
                vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)
        elif self.vfm_name in ["dino", "f3r"]:
            # load DINO/Fast3R features
            assert (
                "fold_stride" in self.kwargs
            ), "fold_stride is required for DINO/Fast3R features"
            assert (
                self.kwargs["fold_stride"] >= 1
            ), "fold_stride must be >= 1 for DINO/Fast3R features"
            if self.kwargs["fold_stride"] > 1:
                # end index should be 1 + the nearest multiple of fold_stride
                end_idx = (
                    math.ceil(float(self.context_len - 1) / self.kwargs["fold_stride"])
                    * self.kwargs["fold_stride"]
                )
                vfm_feat = load_file(vfm_feat_path)["feat"][
                    : end_idx + 1
                ]  # (T, H, W, C)
            else:
                vfm_feat = load_file(vfm_feat_path)["feat"][
                    : self.context_len
                ]  # (T, H, W, C)

            T, _, _, C = vfm_feat.shape
            # subsample the features to only include self.total_sampled_frame_feature key frames
            # except for the first frame, fold/pack the rest of the frames by stride
            # for first frame, repeat the channels by fold_stride
            # for the rest, convert from (T, H, W, C) to (T // fold_stride, H, W, C * fold_stride)
            fold_stride = self.kwargs["fold_stride"]
            if fold_stride > 1:
                assert (
                    T - 1
                ) % fold_stride == 0, (
                    f"fold_stride {fold_stride} does not divide T-1 {T-1}"
                )
                first_feat = vfm_feat[:1, :, :, :].repeat(1, 1, 1, fold_stride)
                remaining_feat = (
                    vfm_feat[1:, :, :, :]
                    .permute(0, 3, 1, 2)
                    .reshape(
                        (T - 1) // fold_stride,
                        C * fold_stride,
                        vfm_feat.shape[1],
                        vfm_feat.shape[2],
                    )
                    .permute(0, 2, 3, 1)
                )  # (T // fold_stride, H, W, C * fold_stride)
                vfm_feat = torch.cat([first_feat, remaining_feat], dim=0)
                T = vfm_feat.shape[0]

            # calculate the query-to-key mapping
            # the DINO/Fast3R features are composed of a single first frame feature + T // fold_stride features
            vfm_idx = (
                torch.floor((sel - 1).float() / (self.context_len - 1) * (T - 1)).long()
                + 1
            )
            if self.feat_pixalign:
                vfm_feat = vfm_feat[vfm_idx]
                vfm_idx = torch.arange(vfm_feat.shape[0], device=vfm_feat.device)
        else:
            raise ValueError(f"Unknown vfm_name {self.vfm_name}")

        # transform and scale the point maps so the extrinsics of start is identity rotation and zero translation
        extrinsics, pointmaps, depthmaps = invert_pose_ref_and_scale(
            extrinsics[sel],  # (num_views, 3, 4)
            pointmaps[sel],  # (num_views, H, W, 3)
            depthmaps=depthmaps[sel],  # (num_views, H, W, 1)
            ref_idx=0,
            scale_by_points=True,
        )

        image = images[sel] / 255.0  # (num_views, 3, H, W)
        pointmap = pointmaps  # (num_views, H, W, 3)
        confmap = confmaps[sel]  # (num_views, H, W, 1)
        depthmap = depthmaps  # (num_views, H, W, 1)
        intrinsic = intrinsics[sel]  # (num_views, 3, 3)
        extrinsic = extrinsics  # (num_views, 3, 4)
        if masks is not None:
            masks = masks[sel]  # (num_views, H, W, 1)

        plucker = mat2plucker(
            intrinsic,  # (num_views,3,3)
            extrinsic,  # (num_views,3,4)
            image.shape[-2:],  # (H,W)
            layout="spatial",
            moments_rescale=self.plucker_rescale,
        )

        # prepare the output
        output["image"] = image
        if masks is not None:
            output["masks"] = masks.float().permute(0, 3, 1, 2)  # (num_views, 1, H, W)
        output["pmaps"] = pointmap.permute(0, 3, 1, 2)  # (num_views, 3, H, W)
        output["cmaps"] = confmap.permute(0, 3, 1, 2)  # (num_views, 1, H, W)
        output["dmaps"] = depthmap.permute(0, 3, 1, 2)  # (num_views, 1, H, W)
        output["intrinsics"] = intrinsic
        output["extrinsics"] = extrinsic
        output["plucker"] = plucker  # (num_views,6,H,W)
        output["vfm_feat"] = vfm_feat  # (T, N, C)
        output["vfm_idx"] = vfm_idx  # (num_views,)

        # check all datatypes
        for key, val in output.items():
            res, err_msg = is_good_type(key, val)
            assert res, f"{err_msg} with {key}={val}, scene={self.scenes[idx]}"

        # this allows to check whether the RNG is is the same state each time
        output["rng"] = int.from_bytes(self._rng.bytes(4), "big")
        output["scene_path"] = str(self.scenes[idx])
        output["vfm_name"] = self.vfm_name
        return output
