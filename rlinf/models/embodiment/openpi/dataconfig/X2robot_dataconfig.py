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
import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np
import openpi.models.model as _model
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import arx_policy


def _default_repack() -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {
                        "left_wrist_view": "left_wrist_view",
                        "face_view": "face_view",
                        "right_wrist_view": "right_wrist_view",
                    },
                    "state": "state",
                    "actions": "actions",
                    "actions_is_pad": "actions_is_pad",
                    "prompt": "task",
                }
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class LeRobotX2robotDataConfig(DataConfigFactory):
    """Data configuration for X2Robot/ARX LeRobot datasets."""

    mode: str | None = None
    use_delta_actions: bool = False
    mask_history_slave_states: bool = False
    action_dim: int = 14
    state_history_size: int = 0
    state_future_size: int = 0
    state_step: int = 1
    slave_state_dim: int = 14
    random_drop_master: float = 0.0
    random_drop_history: float = 0.0
    random_drop_future: float = 0.0
    random_pos_offset: float = 0.0
    only_right_obs: bool = False
    unified_input: bool = False
    individual_keys: bool = False

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_default_repack
    )

    @property
    def state_sequence_length(self) -> int:
        return self.state_history_size + 1 + self.state_future_size

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        assert self.mode in ["s2s", "s2m", "sm2m", "sm2sm"], (
            f"Invalid mode: {self.mode}"
        )
        if self.individual_keys:
            raise NotImplementedError(
                "LeRobotX2robotDataConfig.individual_keys requires the component "
                "key contract from the source dataset. TODO(agent): wire this once "
                "the dataset schema is available in RLinf."
            )

        data_transforms = _transforms.Group(
            inputs=[
                arx_policy.ArxInputs(
                    mode=self.mode,
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_history_size=self.state_history_size,
                    state_future_size=self.state_future_size,
                    slave_state_dim=self.slave_state_dim,
                    mask_history_slave_states=self.mask_history_slave_states,
                    random_drop_master=self.random_drop_master,
                    random_drop_history=self.random_drop_history,
                    random_drop_future=self.random_drop_future,
                    random_pos_offset=self.random_pos_offset,
                    only_right_obs=self.only_right_obs,
                    unified_input=self.unified_input,
                    individual_keys=self.individual_keys,
                )
            ],
            outputs=[arx_policy.ArxOutputs(action_dim=self.action_dim)],
        )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)

        if (
            self.random_drop_master > 0.0 or self.random_drop_future > 0.0
        ) and base_config.norm_stats is not None:
            norm_stats = dict(base_config.norm_stats)
            state_stats = norm_stats["state"]
            new_std = np.array(state_stats.std, copy=True)
            zero_var_indices = np.where(new_std == 0)[0]
            if len(zero_var_indices) > 0:
                new_std[zero_var_indices] = 1.0
                logging.info(
                    "Fixed %s zero-variance state dimensions: %s",
                    len(zero_var_indices),
                    zero_var_indices.tolist(),
                )
                norm_stats["state"] = _normalize.NormStats(
                    mean=state_stats.mean,
                    std=new_std,
                    q01=state_stats.q01,
                    q99=state_stats.q99,
                )
                base_config = dataclasses.replace(base_config, norm_stats=norm_stats)

        replace_kwargs: dict[str, Any] = {
            "repack_transforms": self.repack_transforms,
            "data_transforms": data_transforms,
            "model_transforms": model_transforms,
        }
        data_config_fields = {field.name for field in dataclasses.fields(DataConfig)}
        if "state_history_size" in data_config_fields:
            replace_kwargs["state_history_size"] = self.state_history_size
        if "state_future_size" in data_config_fields:
            replace_kwargs["state_future_size"] = self.state_future_size

        return dataclasses.replace(base_config, **replace_kwargs)
