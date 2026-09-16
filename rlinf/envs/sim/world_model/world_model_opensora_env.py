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

"""The OpenSora world model as an env: :class:`WorldModelEnv` + :class:`OpenSoraBackend`."""

from __future__ import annotations

from omegaconf import OmegaConf
from opensora.registry import MODELS, build_module

from rlinf.envs.sim.world_model.backend import WorldModelBackend
from rlinf.envs.sim.world_model.opensora_backend import OpenSoraBackend
from rlinf.envs.sim.world_model.world_model_env import WorldModelEnv

__all__ = ["OpenSoraEnv"]


class OpenSoraEnv(WorldModelEnv):
    supports_kir = False

    def _build_backend(self) -> WorldModelBackend:
        return OpenSoraBackend(self.cfg, self._get_runtime_device())

    def _load_reward_model(self):
        rm_cfg = OmegaConf.to_container(
            self.cfg.world_model_cfg.reward_model, resolve=True
        )
        return build_module(rm_cfg, MODELS)
