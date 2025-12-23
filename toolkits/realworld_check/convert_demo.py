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
import pickle as pkl
from typing import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

demo_path = "data.pkl"
tgt_path = "torch_data.pkl"

if not os.path.exists(demo_path):
    raise FileNotFoundError(f"File {demo_path} not found")

with open(demo_path, "rb") as f:
    trajs = pkl.load(f)

obs_key_map = {
    "front": "images/base_camera",
    "wrist": "images/wrist_camera",
    "wrist_1": "images/wrist_1",
    "state": "states",
}


def convert_data():
    torch_trajs = []
    for traj in tqdm(trajs):
        torch_traj = {}

        torch_traj["transitions"] = {
            "obs": {"images": {}, "states": None},
            "next_obs": {"images": {}, "states": None},
        }
        # observations
        for key, value in traj["observations"]["frames"].items():
            print(f"{key=}, {value.shape=}")
            tgt_value = torch.from_numpy(value)
            tgt_value = tgt_value.permute(2, 0, 1).float() / 255
            torch_traj["transitions"]["obs"]["images"][key] = tgt_value

        raw_states = OrderedDict(sorted(traj["observations"]["state"].items()))
        full_states = []
        for key, value in raw_states.items():
            full_states.append(value)
        full_states = np.concatenate(full_states, axis=-1)
        torch_traj["transitions"]["obs"]["states"] = torch.from_numpy(full_states)

        # next observations
        for key, value in traj["next_observations"]["frames"].items():
            tgt_value = torch.from_numpy(value)
            tgt_value = tgt_value.permute(2, 0, 1).float() / 255
            torch_traj["transitions"]["next_obs"]["images"][key] = tgt_value

        raw_states = OrderedDict(sorted(traj["next_observations"]["state"].items()))
        full_states = []
        for key, value in raw_states.items():
            full_states.append(value)
        full_states = np.concatenate(full_states, axis=-1)
        torch_traj["transitions"]["next_obs"]["states"] = torch.from_numpy(full_states)

        torch_traj["action"] = torch.from_numpy(traj["actions"].flatten())

        for key in ["rewards", "dones"]:
            value = traj[key]
            if isinstance(value, np.ndarray):
                if len(value.shape) == 0:
                    value = np.array(
                        [
                            value,
                        ]
                    )
                torch_traj[key] = torch.from_numpy(value)
            else:
                torch_traj[key] = torch.tensor(
                    [
                        value,
                    ]
                )

        torch_traj["terminations"] = torch_traj["dones"].clone()
        torch_traj["truncations"] = torch.zeros_like(torch_traj["dones"])
        torch_trajs.append(torch_traj)

    with open(tgt_path, "wb") as f:
        pkl.dump(torch_trajs, f)


if __name__ == "__main__":
    convert_data()
