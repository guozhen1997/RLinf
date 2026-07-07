# Copyright 2025 The RLinf Authors.
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

"""Reward models for embodied RL."""

from typing import TYPE_CHECKING

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

if TYPE_CHECKING:
    from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        HistoryVLMRewardModel,
        VLMRewardModel,
    )

__all__ = [
    "BaseRewardModel",
    "ResNetRewardModel",
    "VLMRewardModel",
    "HistoryVLMRewardModel",
    "get_reward_model_class",
    "resolve_reward_model_backend",
]

reward_model_registry = {
    "resnet": "resnet",
    "vlm": "vlm",
    "history_vlm": "history_vlm",
}

_HISTORY_VLM_MODEL_TYPE = "history_vlm"
_HISTORY_VLM_TRANSFORMERS_BACKEND = "hf"
_HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES = {"hf", "transformers"}
_HISTORY_VLM_SUPPORTED_BACKENDS = _HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES


def _load_reward_model_class(reward_model_type: str):
    if reward_model_type == "resnet":
        from rlinf.models.embodiment.reward.resnet_reward_model import (
            ResNetRewardModel,
        )

        return ResNetRewardModel
    if reward_model_type == "vlm":
        from rlinf.models.embodiment.reward.vlm_reward_model import VLMRewardModel

        return VLMRewardModel
    if reward_model_type == "history_vlm":
        from rlinf.models.embodiment.reward.vlm_reward_model import (
            HistoryVLMRewardModel,
        )

        return HistoryVLMRewardModel
    raise ValueError(f"Unsupported reward model type: {reward_model_type}")


def __getattr__(name: str):
    if name == "ResNetRewardModel":
        return _load_reward_model_class("resnet")
    if name == "VLMRewardModel":
        return _load_reward_model_class("vlm")
    if name == "HistoryVLMRewardModel":
        return _load_reward_model_class("history_vlm")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _normalize_backend(inference_backend: str | None) -> str | None:
    if inference_backend is None or inference_backend == "":
        return None
    backend = str(inference_backend).lower()
    if backend in _HISTORY_VLM_TRANSFORMERS_BACKEND_ALIASES:
        return _HISTORY_VLM_TRANSFORMERS_BACKEND
    return backend


def resolve_reward_model_backend(
    reward_model_type: str,
    inference_backend: str | None = None,
) -> tuple[str, str | None]:
    backend = _normalize_backend(inference_backend)

    if reward_model_type not in reward_model_registry:
        raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    if reward_model_type != _HISTORY_VLM_MODEL_TYPE:
        if backend is not None:
            raise ValueError(
                "reward.model.inference_backend is only supported for "
                "reward.model.model_type='history_vlm'."
            )
        return reward_model_type, None

    if backend is not None and backend not in _HISTORY_VLM_SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported reward.model.inference_backend for history_vlm: "
            f"{inference_backend!r}. Supported backend values are 'hf' "
            "(alias 'transformers') or unset. Use reward.worker_type='api' "
            "with an OpenAI-compatible reward.api for API reward inference."
        )
    return reward_model_type, backend


def get_reward_model_class(
    reward_model_type: str,
    inference_backend: str | None = None,
):
    reward_model_type, inference_backend = resolve_reward_model_backend(
        reward_model_type,
        inference_backend,
    )

    return _load_reward_model_class(reward_model_registry[reward_model_type])
