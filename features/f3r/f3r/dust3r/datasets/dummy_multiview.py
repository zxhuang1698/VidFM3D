# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np

from f3r.dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset


class DummyMultiview(BaseStereoViewDataset):
    def __init__(
        self, num_views=4, dataset_size=1000, *args, split=None, ROOT=None, **kwargs
    ):
        """
        Args:
            num_views: Number of views per sample
            dataset_size: Total number of samples in the dataset
            split: Ignored, kept for API compatibility
            ROOT: Ignored, kept for API compatibility
        """
        super().__init__(*args, **kwargs)
        self.num_views = num_views
        self.dataset_size = dataset_size

        # Pre-generate random indices to mimic real dataset behavior
        self.combinations = [(i,) * num_views for i in range(dataset_size)]

    def __len__(self):
        return self.dataset_size

    def _get_views(self, idx, resolution, rng):
        """
        Generate dummy views with random tensors in the same shape as ARKitScenes.
        """
        views = []

        for view_idx in range(self.num_views):
            # Generate random tensors with appropriate shapes
            h, w = resolution

            # Random RGB image (0-255)
            rgb_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            # Random depth map (0-10 range, typical for indoor scenes)
            depthmap = np.random.uniform(0, 10, (h, w)).astype(np.float32)

            # Random camera pose (4x4 transformation matrix)
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :] += np.random.uniform(-0.1, 0.1, (3, 4))

            # Random intrinsics (3x3 matrix)
            intrinsics = np.array(
                [[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="dummy",
                    label=f"dummy_scene_{idx}_{view_idx}",
                    instance=f"{str(idx)}_{str(view_idx)}",
                )
            )

        return views
