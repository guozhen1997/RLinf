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

"""DreamZero policy configuration.

:class:`DreamZeroConfig` holds the VLA policy fields loaded from checkpoint
``config.json`` or Hydra ``actor.model`` (see ``load_dreamzero_config_dict``).
SFT temporal fields (``action_horizon``, ``num_chunks``, etc.) are read directly
from ``actor.model`` by the dataset and ``data_transforms`` builders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.model.dreamzero.base_vla import VLAConfig
from transformers.configuration_utils import PretrainedConfig


def load_dreamzero_config_dict(cfg: Any) -> dict[str, Any]:
    """Load architecture from ``model_path/config.json`` or Hydra ``actor.model``."""
    model_path = cfg.get("model_path", None)

    if model_path is not None:
        json_path = Path(model_path) / "config.json"
        if not json_path.is_file():
            raise FileNotFoundError(
                f"DreamZero model_path is set but config.json is missing: {json_path}"
            )
        base = json.loads(json_path.read_text(encoding="utf-8"))
        from rlinf.utils.logging import get_logger

        get_logger().warning(
            "DreamZero: loading architecture from %s (actor.model YAML ignored).",
            json_path,
        )
    else:
        yaml_dict = OmegaConf.to_container(cfg, resolve=True)
        nullish = {"", "null", "none", "~"}
        for key in (
            "tokenizer_path",
            "diffusion_model_pretrained_path",
            "image_encoder_pretrained_path",
            "text_encoder_pretrained_path",
            "vae_pretrained_path",
        ):
            v = cfg.get(key)
            if v is None or str(v).strip().lower() in nullish:
                raise ValueError(
                    f"DreamZero: model_path unset; actor.model.{key} must be set (non-null)."
                )
        base = yaml_dict

    return base


@dataclass
class DreamZeroConfig(VLAConfig):
    model_type = "dreamzero"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})

    env_action_dim: int = field(
        default=None, metadata={"help": "Environment action dimension."}
    )
    num_action_chunks: int = field(
        default=16, metadata={"help": "Number of action chunks."}
    )

    relative_action: bool = field(default=False, metadata={"help": "Relative action."})
    relative_action_per_horizon: bool = field(
        default=False, metadata={"help": "Relative action per horizon."}
    )
    relative_action_keys: list = field(
        default_factory=list, metadata={"help": "Relative action keys."}
    )

    data_transforms: ComposedModalityTransform = field(
        default=None,
        metadata={
            "help": "Transforming data modalities, e.g. video frame augmentation or action normalization."
        },
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
