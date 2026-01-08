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

import os

import torch

from rlinf.config import SupportedModel


def get_model_save_helper(model_type: str):
    model_type = SupportedModel(model_type)

    _MODEL_SAVE_HELPER_REGISTRY = {
        SupportedModel.OPENVLA_OFT: openvla_oft_save_helper,
    }

    if model_type in _MODEL_SAVE_HELPER_REGISTRY:
        return _MODEL_SAVE_HELPER_REGISTRY[model_type]
    else:
        return None


def openvla_oft_save_helper(model_state_dict, model_config, save_path, **kwargs):
    global_step = kwargs.get("global_step", 0)
    if model_config.get("use_film", False):
        vision_sd = {
            k.replace("vision_backbone.", "", 1): v
            for k, v in model_state_dict.items()
            if k.startswith("vision_backbone.")
        }
        torch.save(
            vision_sd,
            os.path.join(save_path, f"vision_backbone--{global_step}_checkpoint.pt"),
        )
    if model_config.get("use_proprio", False):
        proprio_sd = {
            k.replace("proprio_projector.", "", 1): v
            for k, v in model_state_dict.items()
            if k.startswith("proprio_projector.")
        }
        torch.save(
            proprio_sd,
            os.path.join(save_path, f"proprio_projector--{global_step}_checkpoint.pt"),
        )
