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

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput
    from vllm.outputs import RequestOutput as VllmRequestOutput

from rlinf.data.utils import batch_pad_to_fixed_len
from rlinf.scheduler import Channel
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    merge_list,
    merge_tensor,
    split_list,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)

@dataclass(kw_only=True)
class EnvOutput:
    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.intervene_actions = (
            self.intervene_actions.cpu().contiguous()
            if self.intervene_actions is not None
            else None
        )
        self.intervene_flags = (
            self.intervene_flags.cpu().contiguous()
            if self.intervene_flags is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"]) if "task_descriptions" in obs else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self):
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags

        return env_output_dict


@dataclass(kw_only=True)
class ChunkStepResult:
    obs: dict[str, Any] = field(default_factory=dict)
    final_obs: dict[str, Any] = field(default_factory=dict)
    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.obs is not None and len(self.obs) > 0:
            self.obs = put_tensor_device(self.obs, "cpu")
        if self.final_obs is not None and len(self.final_obs) > 0:
            self.final_obs = put_tensor_device(self.final_obs, "cpu")

@dataclass
class Trajectory:
    """
    trajectory contains multiple episodes.
    """
    max_episode_length: int = 0 # max episode length

    obs: dict[str, Any] = field(default_factory=dict)
    curr_obs_idx: torch.Tensor = None
    next_obs_idx: torch.Tensor = None
    actions: torch.Tensor = None 
    intervene_flags: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    forward_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    """
    collect trajectories for rollout.
    """

    collect_obs: bool = True
    max_episode_length: int = 0
    obs_pointer: torch.Tensor = None
    
    obs: list[dict[str, Any]] = field(default_factory=list) # trajectory_length + (number of dones == true)
    curr_obs_idx: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    next_obs_idx: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    actions: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    intervene_flags: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    rewards: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    terminations: list[torch.Tensor] = field(default_factory=list) # trajectory_length + rollout_epoch
    truncations: list[torch.Tensor] = field(default_factory=list) # trajectory_length + rollout_epoch
    dones: list[torch.Tensor] = field(default_factory=list) # trajectory_length + rollout_epoch
    prev_logprobs: list[torch.Tensor] = field(default_factory=list) # trajectory_length
    prev_values: list[torch.Tensor] = field(default_factory=list) # trajectory_length + rollout_epoch
    forward_inputs: list[dict[str, Any]] = field(default_factory=list) # trajectory_length

    def append_step_result(self, result: ChunkStepResult, is_last_step: bool = False):
        if self.obs_pointer is None:
            self.obs_pointer = torch.zeros(result.dones.shape[0], dtype=torch.int32)
            self.obs_pointer = self.obs_pointer - 1

        if self.collect_obs and result.obs is not None:
            if (result.dones.any() or is_last_step):
                assert result.final_obs is not None, "final_obs is None when dones.any() or is_last_step, please check the environment implementation."

                self.obs.append(result.final_obs)
                self.obs.append(result.obs)
                
                self.curr_obs_idx.append(self.obs_pointer + 1)
                self.next_obs_idx.append(self.obs_pointer + 2)
                self.obs_pointer = self.obs_pointer + 2
            else:
                self.obs.append(result.obs)
                self.curr_obs_idx.append(self.obs_pointer)
                self.next_obs_idx.append(self.obs_pointer + 1)
                self.obs_pointer = self.obs_pointer + 1
                    
        if result.actions is not None:
            self.actions.append(result.actions)
            self.intervene_flags.append(torch.zeros(1, dtype=torch.bool))
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.forward_inputs is not None:
            self.forward_inputs.append(result.forward_inputs)
    
    def update_last_actions(self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor):
        if self.actions and len(self.actions) > 0:
            self.actions[-1] = intervene_actions * intervene_flags[..., None] + self.actions[-1] * (~intervene_flags[..., None])
            self.intervene_flags[-1] = intervene_flags

    def to_trajectory(self) -> Trajectory:
        # return [trajectory_length, B, ...]
        trajectory = Trajectory(max_episode_length=self.max_episode_length)
        if self.collect_obs and len(self.obs) > 0:
            trajectory.obs = stack_list_of_dict_tensor(self.obs)
            for key in trajectory.obs.keys():
                trajectory.obs[key] = trajectory.obs[key].cpu().contiguous()
        if len(self.curr_obs_idx) > 0:
            trajectory.curr_obs_idx = torch.stack(self.curr_obs_idx, dim=0).cpu().contiguous()
        if len(self.next_obs_idx) > 0:
            trajectory.next_obs_idx = torch.stack(self.next_obs_idx, dim=0).cpu().contiguous()
        if len(self.actions) > 0:
            trajectory.actions = torch.stack(self.actions, dim=0).cpu().contiguous()
        if len(self.intervene_flags) > 0:
            trajectory.intervene_flags = torch.stack(self.intervene_flags, dim=0).cpu().contiguous()
        if len(self.rewards) > 0:
            trajectory.rewards = torch.stack(self.rewards, dim=0).cpu().contiguous()
        if len(self.terminations) > 0:
            trajectory.terminations = torch.stack(self.terminations, dim=0).cpu().contiguous()
        if len(self.truncations) > 0:
            trajectory.truncations = torch.stack(self.truncations, dim=0).cpu().contiguous()
        if len(self.dones) > 0:
            trajectory.dones = torch.stack(self.dones, dim=0).cpu().contiguous()
        if len(self.prev_logprobs) > 0:
            trajectory.prev_logprobs = torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
        if len(self.prev_values) > 0:
            trajectory.prev_values = torch.stack(self.prev_values, dim=0).cpu().contiguous()
        if len(self.forward_inputs) > 0:
            trajectory.forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
            for key in trajectory.forward_inputs.keys():
                trajectory.forward_inputs[key] = trajectory.forward_inputs[key].cpu().contiguous()
        return trajectory

    def to_splited_trajectories(self, split_size: int) -> list[Trajectory]:
        all_trajectory: Trajectory = self.to_trajectory()
        splited_trajectories: list[Trajectory] = [Trajectory() for _ in range(split_size)]

        if len(all_trajectory.obs) > 0:
            splited_obs = split_dict_to_chunk(all_trajectory.obs, split_size, dim=1)
            for i in range(split_size):
                splited_trajectories[i].obs = splited_obs[i]

        if all_trajectory.forward_inputs is not None and len(all_trajectory.forward_inputs) > 0:
            splited_forward_inputs = split_dict_to_chunk(
                all_trajectory.forward_inputs, split_size, dim=1
            )
            for i in range(split_size):
                splited_trajectories[i].forward_inputs = splited_forward_inputs[i]

        for field in all_trajectory.__dataclass_fields__.keys():
            if field in ["obs", "final_obs", "forward_inputs", "max_episode_length"]:
                continue
            value = getattr(all_trajectory, field)
            if value is None:
                continue

            chunks = torch.chunk(value, split_size, dim=1)
            for i in range(split_size):
                setattr(splited_trajectories[i], field, chunks[i])

        return splited_trajectories


def pad_and_stack_trajectories(trajectories: list[Trajectory]) -> dict[str, torch.Tensor]:
    """
    Pad and stack a list of trajectories with different lengths.
    
    Args:
        trajectories: List of Trajectory objects with potentially different lengths
        
    Returns:
        Dictionary with stacked tensors, all padded to the maximum length.
        Shape: [T, B, ...] where T is max trajectory length, B is batch size.
    """
    if not trajectories:
        return {}
    
    batch = {}
    
    # Helper function to pad a list of tensors to the same length
    def pad_tensor_list(tensor_list: list[torch.Tensor]) -> torch.Tensor:
        if not tensor_list:
            return None
        # Find max length
        max_len = max(t.shape[0] for t in tensor_list)
        # Get shape info from first tensor to determine pad value
        first_tensor = tensor_list[0]
        if first_tensor.dtype == torch.bool:
            pad_value = False
        elif first_tensor.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            pad_value = 0
        else:
            pad_value = 0.0
        
        # Pad each tensor
        padded_list = []
        for t in tensor_list:
            if t.shape[0] < max_len:
                pad_size = max_len - t.shape[0]
                # Pad along the first dimension
                pad_shape = list(t.shape)
                pad_shape[0] = pad_size
                pad_tensor = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                padded_tensor = torch.cat([t, pad_tensor], dim=0)
            else:
                padded_tensor = t
            padded_list.append(padded_tensor)
        
        # Stack along batch dimension: [B, T, ...], then transpose to [T, B, ...]
        stacked = torch.stack(padded_list, dim=0)  # [B, T, ...]
        if stacked.dim() >= 2:
            stacked = stacked.transpose(0, 1)  # [T, B, ...]
        return stacked
    
    # Helper function to pad and stack dict of tensors
    def pad_dict_tensor_list(dict_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not dict_list:
            return {}
        # Get all keys
        all_keys = set()
        for d in dict_list:
            all_keys.update(d.keys())
        
        result = {}
        for key in all_keys:
            tensor_list = [d[key] for d in dict_list if key in d]
            if tensor_list:
                result[key] = pad_tensor_list(tensor_list)
        return result
    
    # Process obs (dict[str, Tensor])
    if trajectories[0].obs:
        obs_batch = pad_dict_tensor_list([traj.obs for traj in trajectories])
        for key, value in obs_batch.items():
            value = value.squeeze(2)
            batch["obs/" + key] = value
    
    # Process simple tensor fields
    # Fields with length T (trajectory_length)
    for field_name in ["actions", "intervene_flags", "rewards", "prev_logprobs"]:
        field_list = [getattr(traj, field_name) for traj in trajectories if getattr(traj, field_name) is not None]
        if field_list:
            batch[field_name] = pad_tensor_list(field_list)
    
    # Fields with length T+rollout_epoch (trajectory_length + rollout_epoch)
    for field_name in ["dones", "terminations", "truncations", "prev_values"]:
        field_list = [getattr(traj, field_name) for traj in trajectories if getattr(traj, field_name) is not None]
        if field_list:
            batch[field_name] = pad_tensor_list(field_list)
    
    # Process forward_inputs (dict[str, Tensor])
    if trajectories[0].forward_inputs:
        forward_inputs_batch = pad_dict_tensor_list([traj.forward_inputs for traj in trajectories])
        for key, value in forward_inputs_batch.items():
            value = value.squeeze(2)
            batch["forward_inputs/" + key] = value

    return batch


def convert_trajectories_to_batch(
    trajectories: list[Trajectory],
) -> dict[str, torch.Tensor]:
    """
    convert a list of trajectories to a batch dict, the shape of the batch is [T, B_total, ...].
    donot handle obs and final_obs
    """
    if not trajectories:
        return {}

    batch: dict[str, torch.Tensor] = {}

    # -------- obs / forward_inputs: dict[str, Tensor] --------
    if trajectories[0].obs and trajectories[0].curr_obs_idx is not None and trajectories[0].next_obs_idx is not None:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.obs.keys())

        curr_obs_data = {key: [] for key in all_keys}
        next_obs_data = {key: [] for key in all_keys}

        for traj in trajectories:
            if not traj.obs or traj.curr_obs_idx is None or traj.next_obs_idx is None:
                continue

            for key in all_keys:
                if key not in traj.obs:
                    continue

                obs_tensor = traj.obs[key]  # [obs_length, B, ...]

                # Select cur_obs and next_obs using indices
                curr_obs_selected = obs_tensor[traj.curr_obs_idx]  # [T, B, ...]
                next_obs_selected = obs_tensor[traj.next_obs_idx]  # [T, B, ...]

                curr_obs_data[key].append(curr_obs_selected)
                next_obs_data[key].append(next_obs_selected)

        # Concatenate along batch dimension (dim=1)
        batch["curr_obs"] = {}
        batch["next_obs"] = {}
        for key, tensors in curr_obs_data.items():
            if tensors:
                batch["curr_obs"][key] = torch.cat(tensors, dim=1)

        for key, tensors in next_obs_data.items():
            if tensors:
                batch["next_obs"][key] = torch.cat(tensors, dim=1)

    if trajectories[0].forward_inputs:
        all_keys: set[str] = set()
        for traj in trajectories:
            all_keys.update(traj.forward_inputs.keys())
        batch["forward_inputs"] = {}
        for key in all_keys:
            tensors = [traj.forward_inputs[key] for traj in trajectories if key in traj.forward_inputs]
            if tensors:
                batch["forward_inputs"][key] = torch.cat(tensors, dim=1)

    # -------- tensor fields --------
    for field_name in trajectories[0].__dataclass_fields__.keys():
        if field_name in ["obs", "curr_obs_idx", "next_obs_idx", "forward_inputs", "max_episode_length"]:
            continue
        field_list = [
            getattr(traj, field_name)
            for traj in trajectories
            if getattr(traj, field_name) is not None
        ]
        if field_list:
            batch[field_name] = torch.cat(field_list, dim=1)

    return batch
