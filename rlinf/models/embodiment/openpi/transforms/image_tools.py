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

"""Pure PIL/NumPy image utilities matching ``openpi_client.image_tools``."""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a floating-point image in [0, 1] to uint8."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    method: int = Image.BILINEAR,
) -> np.ndarray:
    """Resize images to ``(height, width)`` with zero padding preserving aspect ratio.

    Matches ``openpi_client.image_tools.resize_with_pad`` for ``[..., H, W, C]`` arrays.
    """
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat_images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_pad_pil(Image.fromarray(im), height, width, method=method)
            for im in flat_images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
