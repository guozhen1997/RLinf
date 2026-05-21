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

from typing import Any, Protocol, runtime_checkable

import numpy as np
from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform


@runtime_checkable
class DreamZeroEmbodimentTransform(Protocol):
    """Static interface implemented by each embodiment module."""

    TAG: str
    DEFAULT_TAG_MAPPING: dict[str, int]
    DEFAULT_ACTION_HORIZON: int

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]: ...

    @staticmethod
    def get_transform(
        *,
        tokenizer_path: str,
        cfg: Any,
        embodiment_tag_mapping: dict[str, int],
    ) -> ComposedModalityTransform: ...

    @staticmethod
    def format_training_prompt(instruction: str) -> str: ...

    @staticmethod
    def concat_multiview_video(images: np.ndarray) -> np.ndarray:
        """Concat multi-view frames ``(v, t, c, h, w)`` to ``(1, t, c, H, W)``."""
        ...
