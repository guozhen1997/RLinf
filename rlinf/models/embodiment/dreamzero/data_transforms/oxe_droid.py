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

from typing import Any

from groot.vla.data.dataset.lerobot import ModalityConfig
from groot.vla.data.transform.base import ComposedModalityTransform
from groot.vla.data.transform.concat import ConcatTransform
from groot.vla.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from groot.vla.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform


_VIDEO_KEYS = [
    "video.exterior_image_1_left",
    "video.exterior_image_2_left",
    "video.wrist_image_left",
]
_STATE_KEYS = ["state.joint_position", "state.gripper_position"]
_ACTION_KEYS = ["action.joint_position", "action.gripper_position"]

_VIDEO_BACKEND = "torchvision"


class OxeDroidDataTransform:
    """Provides modality config and composed transform for oxe_droid.
    """

    # ------------------------------------------------------------------
    # Modality config
    # ------------------------------------------------------------------

    @staticmethod
    def get_modality_config() -> dict[str, ModalityConfig]:
        """Return modality config dict for oxe_droid (25 video delta, 24 action delta)."""
        return {
            "video": ModalityConfig(
                delta_indices=list(range(25)),
                eval_delta_indices=[0],
                modality_keys=list(_VIDEO_KEYS),
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=list(_STATE_KEYS),
            ),
            "action": ModalityConfig(
                delta_indices=list(range(24)),
                modality_keys=list(_ACTION_KEYS),
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "annotation.language.language_instruction",
                    "annotation.language.language_instruction_2",
                    "annotation.language.language_instruction_3",
                ],
            ),
            "lapa_action": ModalityConfig(
                delta_indices=[0],
                modality_keys=["lapa_action"],
            ),
        }

    # ------------------------------------------------------------------
    # Composed transform
    # ------------------------------------------------------------------

    @staticmethod
    def get_transform(
        tokenizer_path: str,
        state_horizon: int = 1,
        action_horizon: int = 24,
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        max_length: int = 512,
        default_instruction: str = "Perform the default behavior.",
        language_dropout_prob: float = 0.0,
        always_use_default_instruction: bool = False,
        embodiment_tag_mapping: dict[str, int] | None = None,
    ) -> ComposedModalityTransform:
        """Build the full ``ComposedModalityTransform`` chain for oxe_droid.

        The chain is: VideoToTensor -> VideoCrop(0.95) -> VideoResize(176,320)
        -> VideoColorJitter(0.3,0.4,0.5,0.08) -> VideoToNumpy ->
        StateActionToTensor(state) -> StateActionTransform(state, q99) ->
        StateActionToTensor(action) -> StateActionTransform(action, q99) ->
        ConcatTransform -> DreamTransform.
        """
        if embodiment_tag_mapping is None:
            from rlinf.models.embodiment.dreamzero.data_transforms import (
                DEFAULT_EMBODIMENT_TAG_MAPPING,
            )

            embodiment_tag_mapping = DEFAULT_EMBODIMENT_TAG_MAPPING["oxe_droid"]

        vk = list(_VIDEO_KEYS)
        state_k = list(_STATE_KEYS)
        action_k = list(_ACTION_KEYS)

        transforms: list[Any] = [
            VideoToTensor(apply_to=vk, backend=_VIDEO_BACKEND),
            VideoCrop(apply_to=vk, backend=_VIDEO_BACKEND, scale=0.95),
            VideoResize(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                height=176,
                width=320,
                interpolation="linear",
            ),
            VideoColorJitter(
                apply_to=vk,
                backend=_VIDEO_BACKEND,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=vk, backend=_VIDEO_BACKEND),
            StateActionToTensor(apply_to=state_k),
            StateActionTransform(
                apply_to=state_k,
                normalization_modes={
                    "state.joint_position": "q99",
                    "state.gripper_position": "q99",
                },
            ),
            StateActionToTensor(apply_to=action_k),
            StateActionTransform(
                apply_to=action_k,
                normalization_modes={
                    "action.joint_position": "q99",
                    "action.gripper_position": "q99",
                },
            ),
            ConcatTransform(
                apply_to=[],
                video_concat_order=vk,
                state_concat_order=state_k,
                action_concat_order=action_k,
            ),
            DreamTransform(
                default_instruction=default_instruction,
                language_dropout_prob=language_dropout_prob,
                always_use_default_instruction=always_use_default_instruction,
                max_state_dim=max_state_dim,
                max_action_dim=max_action_dim,
                max_length=max_length,
                state_horizon=state_horizon,
                action_horizon=action_horizon,
                tokenizer_path=tokenizer_path,
                embodiment_tag_mapping=embodiment_tag_mapping,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)