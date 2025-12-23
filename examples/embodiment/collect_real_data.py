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

import hydra
import numpy as np

from rlinf.envs.realworld.realworld_env import RealworldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.success_needed = 20
        self.total_cnt = 0
        self.env = RealworldEnv(
            cfg.env.train, num_envs=1, seed_offset=0, total_num_processes=1
        )

        self.transitions = []

    def run(self):
        obs, _ = self.env.reset()
        success_cnt = 0
        self.log_info("Start collecting data...")
        while success_cnt < self.success_needed:
            action = np.zeros((6,))
            next_obs, rew, done, truncated, info = self.env.step(action)
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
            self.transitions.append(transition)

            obs = next_obs

            if done:
                success_cnt += rew
                self.total_cnt += 1
                self.log_info(
                    f"{rew}\tGot {success_cnt} successes of {self.total_cnt} trials. {self.success_needed} successes needed."
                )
                obs, _ = self.env.reset()
        save_file_path = os.path.join(self.cfg.runner.logger.log_path, "data.pkl")
        with open(save_file_path, "wb") as f:
            pkl.dump(self.transitions, f)
            self.log_info(f"Saved {self.success_needed} demos to {save_file_path}")

        self.env.close()


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
