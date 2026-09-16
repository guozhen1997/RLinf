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

"""The Wan world model as an env: :class:`WorldModelEnv` plus :class:`WanBackend`."""

from __future__ import annotations

from typing import Optional

from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel

from rlinf.envs.sim.world_model.backend import WorldModelBackend
from rlinf.envs.sim.world_model.wan_backend import WanBackend
from rlinf.envs.sim.world_model.world_model_env import WorldModelEnv

__all__ = ["WanEnv"]


class WanEnv(WorldModelEnv):
    def _build_backend(self) -> WorldModelBackend:
        return WanBackend(self.cfg, self._get_runtime_device())

    def _load_reward_model(self):
        if self.cfg.reward_model.type == "ResnetRewModel":
            return ResnetRewModel(self.cfg.reward_model.from_pretrained)
        elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
            return TaskEmbedResnetRewModel(
                checkpoint_path=self.cfg.reward_model.from_pretrained,
                task_suite_name=self.cfg.task_suite_name,
            )
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")

    def _reward_instructions(self) -> Optional[list[str]]:
        if self.cfg.reward_model.type != "TaskEmbedResnetRewModel":
            return None
        # One instruction per scored frame, so each description repeats over its chunk
        instructions = []
        for env_idx in range(self.num_envs):
            instructions.extend([self.task_descriptions[env_idx]] * self.chunk)
        return instructions
