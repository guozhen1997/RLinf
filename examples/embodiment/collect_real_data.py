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


import copy
import os
import pickle as pkl

import gymnasium as gym
import hydra
import numpy as np

from rlinf.envs.realworld.common.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
    RelativeFrame,
    SpacemouseIntervention,
)
from rlinf.scheduler import Cluster


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    Cluster(cluster_cfg=cfg.cluster)

    success_needed = 20
    success_cnt = 0
    total_cnt = 0
    env = gym.make(
        "PegInsertionEnv-v1",
        override_cfg={
            "robot_ip": "ROBOT_IP",
            "camera_serials": [
                "CAMERA_SERIAL_NUMBER",
            ],
            "target_ee_pose": "TARGET_EE_POSE_AS_LIST",  # pos and euler angles
        },
    )

    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)

    transitions = []

    obs, _ = env.reset()
    print("Start collecting data...")
    while success_cnt < success_needed:
        action = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        transition = copy.deepcopy(
            {
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": rew,
                "masks": 1.0 - done,
                "dones": done,
            }
        )
        transitions.append(transition)

        obs = next_obs

        if done:
            success_cnt += rew
            total_cnt += 1
            print(
                f"{rew}\tGot {success_cnt} successes of {total_cnt} trials. {success_needed} successes needed."
            )
            obs, _ = env.reset()

    save_file_path = os.path.join(cfg.runner.logger.log_path, "data.pkl")
    with open(save_file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {save_file_path}")

    env.close()


if __name__ == "__main__":
    main()
