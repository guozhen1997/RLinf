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

"""Image and text processors for the STEAM value model.

Thin subclasses of the shared base in
:mod:`rlinf.models.embodiment.common.image_text_processing`. Versus the
value-model flavour, STEAM:

    * emits raw BCHW ``[0, 1]`` float images (not ``[-1, 1]``) so the
      downstream ``SteamBackbone`` can apply its own SigLIP-style mean/std
      normalisation — keeping the vision encoder swappable;
    * uses the vision encoder's native resolution (default 384x384), not a
      fixed 224x224;
    * uses the ``Task: {prompt}\\nValue: `` template.

The camera-view time axis (frame_t vs frame_{t+k}) is handled outside the
processor by the pair collator, which runs ``process_images`` once per frame
and stacks the results along a new ``num_frames`` dimension.
"""

from typing import ClassVar, Optional

import torch

from ..common.image_text_processing import (
    IMAGE_KEYS,
    BaseMultiViewImageProcessor,
    BaseValueTextProcessor,
    normalize_image_to_range,
    resize_with_pad,
    resolve_image_size,
    resolve_vision_image_size,
)

# Default SigLIP vision-encoder resolution (siglip-so400m-patch14-384).
IMAGE_RESOLUTION = (384, 384)


def normalize_image_to_steam_format(
    img: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Convert any image format to BCHW ``[0, 1]`` float (STEAM backbone format).

    The backbone expects raw ``[0, 1]`` floats; the model itself applies SigLIP
    mean/std normalisation internally.
    """
    return normalize_image_to_range(img, "unit", device=device, dtype=dtype)


class SteamImageProcessor(BaseMultiViewImageProcessor):
    """STEAM image processor: raw BCHW ``[0, 1]`` float at native resolution."""

    output_range: str = "unit"
    DEFAULT_IMAGE_SIZE: ClassVar[tuple[int, int]] = IMAGE_RESOLUTION


class SteamProcessor(BaseValueTextProcessor):
    """STEAM value model processor. Text template: ``Task: {prompt}\\nValue: ``."""

    image_processor_class = "SteamImageProcessor"
    _default_image_processor_cls: ClassVar[Optional[type]] = SteamImageProcessor
    text_template: str = "Task: {prompt}\nValue: "


__all__ = [
    "SteamImageProcessor",
    "SteamProcessor",
    "normalize_image_to_steam_format",
    "resolve_image_size",
    "resolve_vision_image_size",
    "resize_with_pad",
    "IMAGE_KEYS",
    "IMAGE_RESOLUTION",
]
