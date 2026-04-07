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

from omnigibson.learning.eval import Evaluator
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

from rlinf.envs.behavior.instance_loader import load_cached_activity_instance


def apply() -> None:
    if getattr(ControllableObjectViewAPI, "__rlinf_patched__", False):
        return

    def _get_pattern_from_prim_path(cls, prim_path):
        scene_id, robot_name = prim_path.split("/")[2:4]
        if not scene_id.startswith("scene_"):
            raise ValueError(f"Unexpected prim path: {prim_path}")
        prefix, robot_type, _ = robot_name.split("__")
        return prim_path.replace(f"/{robot_name}", f"/{prefix}__{robot_type}__*")

    def _load_task_instance(self, instance_id: int) -> None:
        load_cached_activity_instance(
            self.env,
            instance_id=instance_id,
            reset_scene=True,
        )

    ControllableObjectViewAPI._get_pattern_from_prim_path = classmethod(
        _get_pattern_from_prim_path
    )
    ControllableObjectViewAPI.__rlinf_patched__ = True
    Evaluator.load_task_instance = _load_task_instance
