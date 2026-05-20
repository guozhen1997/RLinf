# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Libero_sim / oxe_droid: multi-view video concat for patched DreamTransform."""

import numpy as np
from einops import rearrange
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    DreamTransform as DreamTransformBase,
)

class DreamTransform(DreamTransformBase):
    """Adds LIBERO_SIM horizontal two-view concat (exterior | wrist)."""

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        if images.shape[0] > 1:
            v, t, c, h, w = images.shape

            # For DROID embodiment: 2x2 grid where the wrist view spans the full top row,
            # and the two exterior views occupy the bottom row.
            if self.embodiment_tag == EmbodimentTag.OXE_DROID and v >= 3:
                left_exterior = images[0]  # (t, c, h, w)
                right_exterior = images[1]  # (t, c, h, w)
                wrist_image = images[2]  # (t, c, h, w)

                concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

                wrist_wide = np.repeat(wrist_image, 2, axis=-1)  # (t, c, h, 2w)
                concat_images[0, :, :, :h, :] = wrist_wide

                concat_images[0, :, :, h:, :w] = left_exterior
                concat_images[0, :, :, h:, w:] = right_exterior

                return concat_images

            if self.embodiment_tag == EmbodimentTag.LIBERO_SIM and v >= 2:
                concat_images = np.zeros((1, t, c, h, 2 * w), dtype=images.dtype)
                concat_images[0, :, :, :, :w] = images[0]
                concat_images[0, :, :, :, w:] = images[1]
                return concat_images

            # For other embodiments: use 2x2 grid layout
            concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)

            if v > 0:
                concat_images[0, :, :, :h, :w] = images[0]

            if v > 1:
                concat_images[0, :, :, h:, :w] = images[1]

            if v > 2:
                concat_images[0, :, :, :h, w:] = images[2]

            return concat_images

        return images
