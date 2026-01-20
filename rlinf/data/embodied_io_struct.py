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

def put_tensor_cpu(data_dict):
    if data_dict is None:
        return None

    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict

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
    # required
    obs: dict[str, Any] = field(default_factory=dict)
    actions: torch.Tensor = None  # [B, action_dim]
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)


    final_obs: dict[str, Any] = field(default_factory=dict)

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
        if self.obs:
            self.obs = put_tensor_device(self.obs, "cpu")
        if self.final_obs:
            self.final_obs = put_tensor_device(self.final_obs, "cpu")


@dataclass(kw_only=True)
class EmbodiedRolloutResult_old:
    # required
    rollout_epoch: int = None
    prev_logprobs: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    prev_values: list[torch.Tensor] = field(
        default_factory=list
    )  # lens is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    dones: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    terminations: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    truncations: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    rewards: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    forward_inputs: list[dict[str, list[torch.Tensor]]] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    transitions: list[tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=list
    )

    def append_result(self, result: ChunkStepResult):
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.forward_inputs:
            self.forward_inputs.append(result.forward_inputs)

    def add_transition(self, obs, next_obs):
        self.transitions.append(
            {
                "obs": put_tensor_device(obs, "cpu"),
                "next_obs": put_tensor_device(next_obs, "cpu"),
            }
        )

    def to_dict(self):
        rollout_result_dict = {}
        rollout_result_dict["prev_logprobs"] = (
            torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            if len(self.prev_logprobs) > 0
            else None
        )
        rollout_result_dict["prev_values"] = (
            torch.stack(self.prev_values, dim=0).cpu().contiguous()
            if len(self.prev_values) > 0
            else None
        )
        rollout_result_dict["dones"] = (
            torch.stack(self.dones, dim=0).cpu().contiguous()
            if len(self.dones) > 0
            else None
        )
        rollout_result_dict["terminations"] = (
            torch.stack(self.terminations, dim=0).cpu().contiguous()
            if len(self.terminations) > 0
            else None
        )
        rollout_result_dict["truncations"] = (
            torch.stack(self.truncations, dim=0).cpu().contiguous()
            if len(self.truncations) > 0
            else None
        )
        rollout_result_dict["rewards"] = (
            torch.stack(self.rewards, dim=0).cpu().contiguous()
            if len(self.rewards) > 0
            else None
        )

        merged_forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
        for k in merged_forward_inputs.keys():
            assert k not in [
                "dones",
                "terminations",
                "truncations",
                "rewards",
                "prev_logprobs",
                "prev_values",
            ]
            rollout_result_dict[k] = merged_forward_inputs[k]

        transition_dict = stack_list_of_dict_tensor(self.transitions)
        if len(transition_dict) > 0:
            rollout_result_dict["transitions"] = transition_dict

        assert len(rollout_result_dict["dones"]) == len(
            rollout_result_dict["prev_values"]
        ), "dones and prev_values must have the same length"
        assert (
            len(rollout_result_dict["dones"])
            == len(rollout_result_dict["rewards"]) + self.rollout_epoch
        ), "dones length must be the length of rewards plus rollout_epoch"

        return rollout_result_dict

    def to_splitted_dict(self, split_size) -> list[dict[str, Any]]:
        return split_dict_to_chunk(self.to_dict(), split_size, dim=1)

class Episode:
    obs: dict[str, Any] = {}
    actions: torch.Tensor = None
    intervene_flags: torch.Tensor = None
    rewards: torch.Tensor = None
    terminations: torch.Tensor = None
    truncations: torch.Tensor = None
    dones: torch.Tensor = None
    prev_logprobs: torch.Tensor = None
    prev_values: torch.Tensor = None
    forward_inputs: dict[str, Any] = {}
    loss_mask: torch.Tensor = None
    loss_mask_sum: torch.Tensor = None


@dataclass(kw_only=True)
class SingleEmbodiedRolloutResult:
    """
    collect rollout results for a single episode.
    """

    collect_obs: bool = True
    is_completed: bool = False
    max_episode_length: int = -1
    
    obs: list[dict[str, Any]] = field(default_factory=list) # episode_length + 1
    actions: list[torch.Tensor] = field(default_factory=list) # episode_length
    intervene_flags: list[torch.Tensor] = field(default_factory=list) # episode_length
    rewards: list[torch.Tensor] = field(default_factory=list) # episode_length
    terminations: list[torch.Tensor] = field(default_factory=list) # episode_length + 1
    truncations: list[torch.Tensor] = field(default_factory=list) # episode_length + 1
    dones: list[torch.Tensor] = field(default_factory=list) # episode_length + 1
    prev_logprobs: list[torch.Tensor] = field(default_factory=list) # episode_length
    prev_values: list[torch.Tensor] = field(default_factory=list) # episode_length + 1
    forward_inputs: list[dict[str, Any]] = field(default_factory=list) # episode_length

    def append_single_result(self, result: ChunkStepResult):

        if self.collect_obs and result.obs is not None:
            self.obs.append(result.obs)
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
            if result.dones.any() or len(self.dones) - 1 >= self.max_episode_length:
                if self.collect_obs and result.final_obs is not None:
                    self.obs[-1] = result.final_obs
                self.is_completed = True
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

    def to_episode(self) -> Episode:
        episode = Episode()
        if self.collect_obs and len(self.obs) > 0:
            episode.obs = stack_list_of_dict_tensor(self.obs)
            for key in episode.obs.keys():
                episode.obs[key] = episode.obs[key].cpu().contiguous()
        if len(self.actions) > 0:
            episode.actions = torch.stack(self.actions, dim=0).cpu().contiguous()
        if len(self.intervene_flags) > 0:
            episode.intervene_flags = torch.stack(self.intervene_flags, dim=0).cpu().contiguous()
        if len(self.rewards) > 0:
            episode.rewards = torch.stack(self.rewards, dim=0).cpu().contiguous()
        if len(self.terminations) > 0:
            episode.terminations = torch.stack(self.terminations, dim=0).cpu().contiguous()
        if len(self.truncations) > 0:
            episode.truncations = torch.stack(self.truncations, dim=0).cpu().contiguous()
        if len(self.dones) > 0:
            episode.dones = torch.stack(self.dones, dim=0).cpu().contiguous()
        if len(self.prev_logprobs) > 0:
            episode.prev_logprobs = torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
        if len(self.prev_values) > 0:
            episode.prev_values = torch.stack(self.prev_values, dim=0).cpu().contiguous()
        if len(self.forward_inputs) > 0:
            episode.forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
            for key in episode.forward_inputs.keys():
                episode.forward_inputs[key] = episode.forward_inputs[key].cpu().contiguous()
        return episode

@dataclass(kw_only=True)
class BatchEmbodiedRolloutResult:
    """
    collect rollout results for a batch of episodes, each episode is a list of chunk step results, 
    the first dimension is batch_size, the second dimension is episode_length.
    episode_length is the length of the episode, which may be different in each batch.
    """
    batch_size: int = 0
    max_episode_length: int = -1
    collect_obs: bool = True

    batch_episode_results: list[SingleEmbodiedRolloutResult] = field(default_factory=list) # batch_size

    def update_last_actions(self, intervene_actions: torch.Tensor, intervene_flags: torch.Tensor):
        """
        update the last actions and intervene flags if intervention is triggered.
        """
        for batch_size_idx in range(self.batch_size):
            self.batch_episode_results[batch_size_idx].update_last_actions(intervene_actions[batch_size_idx], intervene_flags[batch_size_idx])

    async def append_batch_result(self, batch_result: ChunkStepResult):
        batch_size = batch_result.dones.shape[0]
        # Initialize batch_size and batch_episode_results on first call
        if self.batch_size == 0:
            self.batch_size = batch_size
            self.batch_episode_results = [
                SingleEmbodiedRolloutResult(collect_obs=self.collect_obs, max_episode_length=self.max_episode_length) 
                for _ in range(self.batch_size)
            ]
        assert batch_size == self.batch_size, "batch_size of the result must be the same as the batch_size of the batch episode result"
        if self.collect_obs and batch_result.obs is not None:
            obs_list_of_dict = split_dict_to_chunk(batch_result.obs, batch_size, dim=0)
        else:
            obs_list_of_dict = None
        if self.collect_obs and batch_result.final_obs is not None:
            final_obs_list_of_dict = split_dict_to_chunk(batch_result.final_obs, batch_size, dim=0)
        else:
            final_obs_list_of_dict = None

        if batch_result.forward_inputs is not None:
            forward_inputs_list_of_dict = split_dict_to_chunk(batch_result.forward_inputs, batch_size, dim=0)
        else:
            forward_inputs_list_of_dict = None

        for batch_size_idx in range(batch_size):
            single_result = ChunkStepResult(
                obs=obs_list_of_dict[batch_size_idx] if obs_list_of_dict is not None else None,
                final_obs=final_obs_list_of_dict[batch_size_idx] if final_obs_list_of_dict is not None else None,
                rewards=batch_result.rewards[batch_size_idx] if batch_result.rewards is not None else None,
                terminations=batch_result.terminations[batch_size_idx],
                truncations=batch_result.truncations[batch_size_idx],
                dones=batch_result.dones[batch_size_idx],
                prev_logprobs=batch_result.prev_logprobs[batch_size_idx] if batch_result.prev_logprobs is not None else None,
                prev_values=batch_result.prev_values[batch_size_idx] if batch_result.prev_values is not None else None,
                forward_inputs=forward_inputs_list_of_dict[batch_size_idx] if forward_inputs_list_of_dict is not None else None,
            )
            self.batch_episode_results[batch_size_idx].append_single_result(single_result)

    def reset_specific_episode(self, idx, collect_obs):
        # Reset the episode with same configuration
        self.batch_episode_results[idx] = None
        self.batch_episode_results[idx] = SingleEmbodiedRolloutResult(
            collect_obs=collect_obs,
            max_episode_length=self.max_episode_length
        )
        self.batch_episode_results[idx].is_completed = False

    def get_completed_episodes(self) -> list[SingleEmbodiedRolloutResult]:
        completed_episodes = []
        for batch_size_idx in range(self.batch_size):
            if self.batch_episode_results[batch_size_idx].is_completed:
                completed_episodes.append(self.batch_episode_results[batch_size_idx])
                # reset
                collect_obs = self.batch_episode_results[batch_size_idx].collect_obs
                self.reset_specific_episode(batch_size_idx, collect_obs)
        return completed_episodes

    def force_complete_all_episodes(self) -> list[SingleEmbodiedRolloutResult]:
        """Force complete all remaining episodes (used when epoch ends and episodes cannot span epochs)."""
        completed_episodes = []
        for batch_size_idx in range(self.batch_size):
            # Only process non-empty episodes that are not yet completed
            if len(self.batch_episode_results[batch_size_idx].dones) > 0:
                if not self.batch_episode_results[batch_size_idx].is_completed:
                    # Force mark as completed
                    self.batch_episode_results[batch_size_idx].is_completed = True
                # Save collect_obs before resetting
                completed_episodes.append(self.batch_episode_results[batch_size_idx])

            collect_obs = self.batch_episode_results[batch_size_idx].collect_obs
            self.reset_specific_episode(batch_size_idx, collect_obs)

        return completed_episodes

from collections import deque


def pad_and_stack_episodes(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    """
    Pad and stack a list of episodes with different lengths.
    
    Args:
        episodes: List of Episode objects with potentially different lengths
        
    Returns:
        Dictionary with stacked tensors, all padded to the maximum length.
        Shape: [T, B, ...] where T is max episode length, B is batch size.
    """
    if not episodes:
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
    if episodes[0].obs:
        obs_batch = pad_dict_tensor_list([ep.obs for ep in episodes])
        for key, value in obs_batch.items():
            value = value.squeeze(2)
            batch["obs/" + key] = value
    
    # Process simple tensor fields
    # Fields with length T (episode_length)
    for field_name in ["actions", "intervene_flags", "rewards", "prev_logprobs"]:
        field_list = [getattr(ep, field_name) for ep in episodes if getattr(ep, field_name) is not None]
        if field_list:
            batch[field_name] = pad_tensor_list(field_list)
    
    # Fields with length T+1 (episode_length + 1)
    for field_name in ["dones", "terminations", "truncations", "prev_values", "loss_mask"]:
        field_list = [getattr(ep, field_name) for ep in episodes if getattr(ep, field_name) is not None]
        if field_list:
            batch[field_name] = pad_tensor_list(field_list)
    
    # Process loss_mask_sum (usually [1] or scalar)
    loss_mask_sum_list = [ep.loss_mask_sum for ep in episodes if ep.loss_mask_sum is not None]
    if loss_mask_sum_list:
        # loss_mask_sum might have different shapes, pad to max
        batch["loss_mask_sum"] = pad_tensor_list(loss_mask_sum_list)
    
    # Process forward_inputs (dict[str, Tensor])
    if episodes[0].forward_inputs:
        forward_inputs_batch = pad_dict_tensor_list([ep.forward_inputs for ep in episodes])
        for key, value in forward_inputs_batch.items():
            value = value.squeeze(2)
            batch["forward_inputs/" + key] = value

    return batch


class EmbodiedRolloutEpisodeResultHandler:
    """
     for storing episode results before sending to actor workers.
    
    This class maintains a buffer of completed episode results and automatically sends
    them to the replay channel when the buffer size reaches the specified threshold.
    When the buffer contains at least actor_split_num results, it sends them one by one,
    sending exactly actor_split_num episodes in total.
    
    Also computes loss_mask and loss_mask_sum for each episode when compute_mask=True.
    """
    def __init__(
        self, 
        actor_split_num: int,
        actor_world_size: int,
        channel: Channel,
        compute_mask: bool = False,
        reward_type: str = "chunk_level"
    ):
        """
        Args:
            actor_split_num: Number of episodes to send at once
            channel: Channel to send episodes to
            compute_mask: Whether to compute loss_mask and loss_mask_sum for episodes
            reward_type: Reward type, used to determine mask processing ("chunk_level" or other)
        """
        self.actor_split_num = actor_split_num
        self.actor_world_size = actor_world_size
        self.buffer: deque[Episode] = deque()
        self.channel = channel
        self.compute_mask = compute_mask
        self.reward_type = reward_type

    def _compute_loss_mask_for_episode(self, dones: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute loss_mask and loss_mask_sum for a single episode.
        
        Args:
            dones: [episode_length + 1, 1]
            
        Returns:
            Tuple of (loss_mask, loss_mask_sum)
        """                
        from rlinf.utils.metric_utils import compute_loss_mask
        dones = dones.unsqueeze(1)
        loss_mask, loss_mask_sum = compute_loss_mask(dones)
        
        # Handle chunk_level reward_type (same as original logic)
        if self.reward_type == "chunk_level":
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]

        loss_mask = loss_mask.squeeze(1)
        loss_mask_sum = loss_mask_sum.squeeze(1)
        return loss_mask, loss_mask_sum

    def append_episode_results(self, episode_results: list[SingleEmbodiedRolloutResult]) -> None:
        if not episode_results:
            return
        episodes = [episode_result.to_episode() for episode_result in episode_results]
        # Compute loss_mask and loss_mask_sum for each episode only if compute_mask=True
        if self.compute_mask:
            for episode in episodes:
                loss_mask, loss_mask_sum = self._compute_loss_mask_for_episode(episode.dones)
                episode.loss_mask = loss_mask.cpu().contiguous()
                episode.loss_mask_sum = loss_mask_sum.cpu().contiguous()

        self.buffer.extend(episodes)

    async def append_and_send_eposide_results(self, rollout_results: list[SingleEmbodiedRolloutResult]) -> None:
        if not rollout_results:
            return
        episodes = [rollout_result.to_episode() for rollout_result in rollout_results]
        # Compute loss_mask and loss_mask_sum for each episode only if compute_mask=True
        if self.compute_mask:
            for episode in episodes:
                loss_mask, loss_mask_sum = self._compute_loss_mask_for_episode(episode.dones)
                episode.loss_mask = loss_mask.cpu().contiguous()
                episode.loss_mask_sum = loss_mask_sum.cpu().contiguous()
        
        self.buffer.extend(episodes)

    async def send(self) -> None:
        episode_num = len(self.buffer) // self.actor_split_num
        await self.channel.put(episode_num, async_op=True).async_wait()
        while len(self.buffer) >= self.actor_split_num:
            for _ in range(self.actor_split_num):
                await self.channel.put(self.buffer.popleft(), async_op=True).async_wait()