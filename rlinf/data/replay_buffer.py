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


import json
import os
import pickle as pkl
import tempfile
import uuid
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor


def split_episode_by_dones(episode: Trajectory) -> list[Trajectory]:
    """
    根据 dones 将一个包含多个 episode 的 Trajectory 对象拆分成多个独立的 Episode。

    在 auto_reset=True 的情况下，dones 的长度是 episode_length + num_rollout_epoch，
    dones 中为 True 的位置表示 epoch 结束时的完整 episode。

    Args:
        episode: 输入的 Trajectory 对象，形状为 [T, B, ...]，其中 T 可能包含多个 episode

    Returns:
        拆分后的独立 Trajectory 列表，每个 Trajectory 形状为 [t_i, B, ...]

    示例：
    dones: [F, F, F, T, F, F, T, F, T, F, F, F]
    episode 1: indices [0,1,2,3] (dones[3]=T)
    episode 2: indices [4,5,6] (dones[6]=T)
    episode 3: indices [7,8] (dones[9]=T，最后一个未完成的episode被丢弃)

    final obs append是发生在done=True的时候，所以final obs的T实际等于done=True的个数
    每次取episode数据时，需要在obs的T维度上再加上final obs，让obs 长度变成T + 1
    """
    if episode.dones is None:
        return [episode]

    # dones 形状: [T + n, B, ...]，其中 T = episode_length, n = num_rollout_epoch
    dones = episode.dones
    B = dones.shape[1] if len(dones.shape) > 1 else 1

    split_episodes = []

    # 对每个 batch 分别处理
    for b in range(B):
        # 获取当前 batch 的 dones: [T+1]
        batch_dones = dones[:, b] if len(dones.shape) > 1 else dones

        # 找到 dones 为 True 的位置（episode 结束位置）
        done_indices = torch.where(batch_dones)[0]

        if len(done_indices) == 0:
            # 如果没有 done，整个序列作为一个 episode（没有 final_obs）
            single_episode = Trajectory()
            _copy_episode_slice(episode, single_episode, 0, None, b, B, final_obs_idx=None)
            split_episodes.append(single_episode)
            continue

        # 处理每个完成的 episode
        start_idx = 0
        for i, done_idx in enumerate(done_indices):
            # done_idx 是 dones 中为 True 的索引
            # episode 包含从 start_idx 到 done_idx（包括 done_idx）的所有数据
            # actions 等字段的长度是 done_idx - start_idx + 1（从 start_idx 到 done_idx）
            episode_length = done_idx.item() - start_idx + 1

            if episode_length > 0:  # 确保有实际的数据
                single_episode = Trajectory()
                # 对于 actions 等字段（长度 T），我们取 [start_idx:start_idx + episode_length]
                # 对于 dones 等字段（长度 T+1），我们取 [start_idx:start_idx + episode_length + 1]
                # final_obs_idx = i，表示这是第 i 个完成的 episode，对应的 final_obs 索引是 i
                _copy_episode_slice(episode, single_episode, start_idx, episode_length, b, B, final_obs_idx=i)
                split_episodes.append(single_episode)

            # 更新起始位置到下一个 episode 开始
            start_idx = done_idx.item() + 1

        # 注意：最后一个未完成（没有对应 done=True）的部分会被丢弃，不作为episode

    return split_episodes


def _copy_episode_slice(
    source_episode: Trajectory,
    target_episode: Trajectory,
    start_idx: int,
    episode_length: Optional[int],  # episode 的实际长度（actions 等字段的长度）
    batch_idx: int,
    total_batches: int,
    final_obs_idx: Optional[int] = None,  # 该 episode 对应的 final_obs 索引（第几个完成的 episode）
):
    """
    从源 episode 中复制指定时间范围和 batch 的数据到目标 episode。

    Args:
        source_episode: 源 Trajectory 对象
        target_episode: 目标 Trajectory 对象
        start_idx: 时间维度的开始索引
        episode_length: episode 的长度（对于 actions 等字段），None 表示到结尾
        batch_idx: batch 维度的索引
        total_batches: 总 batch 数量，用于判断是否需要 squeeze
    """
    fields_T = ["actions", "intervene_flags", "rewards", "prev_logprobs", "forward_inputs"]  # 长度 T
    fields_T1 = ["dones", "terminations", "truncations", "prev_values"]  # 长度 T+1

    end_idx = episode_length if episode_length is not None else None
    end_idx_T1 = episode_length + 1 if episode_length is not None else None

    # 处理 obs：需要结合普通 obs 和 final_obs
    # final obs append 是发生在 done=True 的时候，所以 final obs 的 T 实际等于 done=True 的个数
    # 每次取 episode 数据时，需要在 obs 的 T 维度上再加上 final obs，让 obs 长度变成 T + 1
    if source_episode.obs or (hasattr(source_episode, 'final_obs') and source_episode.final_obs):
        target_episode.obs = {}

        # 首先处理普通 obs（时间步 0 到 T）
        if source_episode.obs:
            for key, tensor in source_episode.obs.items():
                if tensor is not None and len(tensor.shape) > 1:
                    # 取 episode 期间的 obs: [start_idx : start_idx + episode_length]
                    obs_slice = tensor[start_idx:start_idx + episode_length, batch_idx:batch_idx+1]
                    if total_batches == 1:
                        obs_slice = obs_slice.squeeze(1)
                    target_episode.obs[key] = obs_slice
                elif tensor is not None:
                    target_episode.obs[key] = tensor[start_idx:start_idx + episode_length]

        # 然后添加对应的 final_obs（episode 结束时的状态）
        if hasattr(source_episode, 'final_obs') and source_episode.final_obs and final_obs_idx is not None:
            for key, tensor in source_episode.final_obs.items():
                if tensor is not None and len(tensor.shape) > 1:
                    # final_obs 形状通常是 [num_done_true, B, ...]，我们取第 final_obs_idx 个
                    final_obs_slice = tensor[final_obs_idx:final_obs_idx+1, batch_idx:batch_idx+1]
                    if total_batches == 1:
                        final_obs_slice = final_obs_slice.squeeze(1)

                    # 将 final_obs 添加到 obs 的末尾，使 obs 长度变成 T + 1
                    if key in target_episode.obs:
                        target_episode.obs[key] = torch.cat([target_episode.obs[key], final_obs_slice], dim=0)
                    else:
                        target_episode.obs[key] = final_obs_slice
                elif tensor is not None:
                    final_obs_val = tensor[final_obs_idx]
                    if key in target_episode.obs:
                        # 如果 obs 是一维的，需要扩展维度后拼接
                        target_obs = target_episode.obs[key]
                        if len(target_obs.shape) == 1 and len(final_obs_val.shape) == 0:
                            # 都转换为 1D tensor 后拼接
                            target_episode.obs[key] = torch.cat([target_obs, final_obs_val.unsqueeze(0)])
                        else:
                            target_episode.obs[key] = torch.cat([target_obs.unsqueeze(0), final_obs_val.unsqueeze(0)], dim=0)
                    else:
                        target_episode.obs[key] = final_obs_val

    # 处理 forward_inputs（长度 T）
    if source_episode.forward_inputs:
        target_episode.forward_inputs = {}
        for key, tensor in source_episode.forward_inputs.items():
            if tensor is not None and len(tensor.shape) > 1:
                sliced = tensor[start_idx:end_idx, batch_idx:batch_idx+1]
                if total_batches == 1:
                    sliced = sliced.squeeze(1)
                target_episode.forward_inputs[key] = sliced
            elif tensor is not None:
                target_episode.forward_inputs[key] = tensor[start_idx:end_idx]

    # 处理普通张量字段
    for field_name in fields_T:
        source_tensor = getattr(source_episode, field_name)
        if source_tensor is not None:
            if len(source_tensor.shape) > 1:
                sliced = source_tensor[start_idx:end_idx, batch_idx:batch_idx+1]
                if total_batches == 1:
                    sliced = sliced.squeeze(1)
                setattr(target_episode, field_name, sliced)
            else:
                setattr(target_episode, field_name, source_tensor[start_idx:end_idx])

    for field_name in fields_T1:
        source_tensor = getattr(source_episode, field_name)
        if source_tensor is not None:
            if len(source_tensor.shape) > 1:
                sliced = source_tensor[start_idx:end_idx_T1, batch_idx:batch_idx+1]
                if total_batches == 1:
                    sliced = sliced.squeeze(1)
                setattr(target_episode, field_name, sliced)
            else:
                setattr(target_episode, field_name, source_tensor[start_idx:end_idx_T1])


def process_nested_dict_for_replay_buffer(nested_dict, rm_extra_done=True):
    ret_dict = {}
    num_data = None
    for key, value in nested_dict.items():
        if key in ["dones", "truncations", "terminations"] and rm_extra_done:
            value = value[1:]
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:]).cpu()
            if num_data is not None:
                assert num_data == ret_dict[key].shape[0], (
                    f"{key=}, {num_data=}, {ret_dict[key].shape[0]=}"
                )
            num_data = ret_dict[key].shape[0]
        elif isinstance(value, dict):
            ret_dict[key], num_data = process_nested_dict_for_replay_buffer(value)
    if len(ret_dict) > 0:
        assert num_data is not None
    return ret_dict, num_data


def get_zero_nested_dict(flattened_batch, capacity, with_batch_dim=True):
    buffer = {}
    for key, value in flattened_batch.items():
        if isinstance(value, torch.Tensor):
            if with_batch_dim:
                tgt_shape = (capacity, *value.shape[1:])
            else:
                tgt_shape = (capacity, *value.shape)
            buffer[key] = torch.zeros(tgt_shape, dtype=value.dtype, device="cpu")
        elif isinstance(value, dict):
            buffer[key] = get_zero_nested_dict(value, capacity, with_batch_dim)
        else:
            raise NotImplementedError
    return buffer


def truncate_nested_dict_by_capacity(nested_dict, capacity):
    ret_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, torch.Tensor):
            ret_dict[key] = val[-capacity:]
        elif isinstance(val, dict):
            ret_dict[key] = truncate_nested_dict_by_capacity(nested_dict, capacity)
        else:
            raise NotImplementedError
    return ret_dict


def sample_nested_batch(nested_dict, sample_ids):
    sample_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            sample_dict[key] = value[sample_ids]
        elif isinstance(value, dict):
            sample_dict[key] = sample_nested_batch(value, sample_ids)
        else:
            raise NotImplementedError
    return sample_dict


def insert_nested_batch(nested_dict, tgt_dict, insert_ids):
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            tgt_dict[key][insert_ids] = value
        elif isinstance(value, dict):
            tgt_dict[key] = insert_nested_batch(value, tgt_dict[key], insert_ids)
        else:
            raise NotImplementedError
    return tgt_dict


def shuffle_and_split_dict_to_chunk(data: dict, split_size, indice_ids):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = torch.chunk(value[indice_ids], split_size)
        elif isinstance(value, dict):
            split_vs = shuffle_and_split_dict_to_chunk(value, split_size, indice_ids)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = split_vs[split_id]
    return splited_list


def clone_dict_and_get_size(nested_dict):
    ret_dict = {}
    size = None
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.clone()
            size = value.shape[0]
        elif isinstance(value, dict):
            ret_dict[key], size = clone_dict_and_get_size(value)
        else:
            raise NotImplementedError
    return ret_dict, size


class SACReplayBuffer:
    """
    Replay buffer for SAC algorithm using pre-allocated torch tensors.
    Implements a circular buffer for efficient memory usage.
    """

    def __init__(self, capacity: int, device: str = "cpu", seed: Optional[int] = None):
        """
        Initialize replay buffer.
        Args:
            capacity: Maximum number of transitions to store
            device: Device to output samples on (storage is always on CPU to save GPU memory)
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.device = device
        self.start = False

        # Storage: dictionary of pre-allocated tensors
        # Will be initialized lazily on first insertion
        self.buffer: dict[str, torch.Tensor] = {}

        self.pos = 0  # Next insertion index
        self.size = 0  # Current number of elements

        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.random_generator = torch.Generator()
            self.random_generator.manual_seed(seed)
        else:
            self.random_generator = None

    @classmethod
    def create_from_demo(cls, demo_path, seed=None):
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"File {demo_path} not found")

        if demo_path.endswith(".pkl"):
            with open(demo_path, "rb") as f:
                data_ls = pkl.load(f)
        elif demo_path.endswith(".pt"):
            data_ls = torch.load(demo_path)

        # TODO: Possibly need to convert from jax to torch.
        instance = cls(capacity=len(data_ls), seed=seed)
        for data in data_ls:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            instance.add(data)
        return instance

    @classmethod
    def create_from_buffer(cls, buffer, seed):
        instance = cls(capacity=None, seed=seed)
        instance.buffer, size = clone_dict_and_get_size(buffer)
        instance.size = size
        instance.capacity = size
        return instance

    def _initialize_storage(
        self, flattened_batch: dict[str, torch.Tensor], with_batch_dim=True
    ):
        self.buffer = get_zero_nested_dict(
            flattened_batch, self.capacity, with_batch_dim
        )

    def add(self, data):
        if not self.buffer:
            self._initialize_storage(data, with_batch_dim=False)

        insert_nested_batch(data, self.buffer, self.pos)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _preprocess_rollout_batch(self, rollout_batch):
        if hasattr(self, "cfg"):
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                raise NotImplementedError

            # filter data by rewards
            if self.cfg.algorithm.get("filter_rewards", False):
                raise NotImplementedError

        flattened_batch, num_to_add = process_nested_dict_for_replay_buffer(
            rollout_batch
        )
        return flattened_batch, num_to_add

    def add_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor], extra_preprocess=True
    ):
        """
        Add a batch of transitions to the buffer.
        Handles flattening [T, B, ...] -> [T*B, ...] and circular insertion.
        """
        # 1. Flatten the batch: [n-chunk-steps, actor-bsz, ...] -> [num_samples, ...]

        if "prev_logprobs" in rollout_batch:
            rollout_batch.pop("prev_logprobs")
        if "prev_values" in rollout_batch:
            rollout_batch.pop("prev_values")

        if extra_preprocess:
            flattened_batch, num_to_add = self._preprocess_rollout_batch(rollout_batch)
        else:
            flattened_batch = rollout_batch
            num_to_add = flattened_batch["rewards"].shape[0]
        assert num_to_add > 0

        # 2. Lazy initialization of storage tensors on first call
        if not self.buffer:
            self._initialize_storage(flattened_batch)

        # 3. Handle case where incoming batch is larger than the entire capacity
        if num_to_add >= self.capacity:
            # Just take the last 'capacity' elements
            print(
                f"Warning: Adding batch size {num_to_add} >= capacity {self.capacity}. Overwriting entire buffer."
            )

            self.buffer = truncate_nested_dict_by_capacity(flattened_batch)
            self.pos = 0
            self.size = self.capacity
            return

        # 4. Circular buffer insertion
        start_idx = self.pos
        end_idx = start_idx + num_to_add

        # Use mod operation (%) to get circulated index.
        # [0, 1, 2, ..., capacity-1, capacity, capacity+1, ...]
        # -> [0, 1, 2, ..., capacity-1, 0, 1, ...]
        indices = torch.arange(start_idx, end_idx) % self.capacity

        # 5. Insert the batch
        insert_nested_batch(flattened_batch, self.buffer, indices)

        # 6. Update position and size
        self.pos = end_idx % self.capacity
        self.size = min(self.size + num_to_add, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Random sampling indices
        transition_ids = torch.randint(
            low=0, high=self.size, size=(batch_size,), generator=self.random_generator
        )

        batch = sample_nested_batch(self.buffer, transition_ids)
        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    async def is_ready_async(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        """Clear the buffer (reset pointers, keep memory allocated)."""
        self.pos = 0
        self.size = 0

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics."""
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity if self.capacity > 0 else 0.0,
        }

        # Calculate reward statistics if available and buffer is not empty
        if self.size > 0 and "rewards" in self.buffer:
            # Only calculate stats on currently valid data
            valid_rewards = self.buffer["rewards"][: self.size]
            stats.update(
                {
                    "mean_reward": valid_rewards.mean().item(),
                    "std_reward": valid_rewards.std().item(),
                    "min_reward": valid_rewards.min().item(),
                    "max_reward": valid_rewards.max().item(),
                }
            )

        return stats

    def split_to_dict(self, num_splits, is_sequential=False):
        assert self.capacity % num_splits == 0

        all_ids = torch.arange(self.size).to(self.device)
        if not is_sequential:
            all_ids = torch.randperm(self.size, generator=self.random_generator).to(
                self.device
            )

        res_ls = shuffle_and_split_dict_to_chunk(
            self.buffer, split_size=num_splits, indice_ids=all_ids
        )
        return res_ls

    async def run(self, cfg, data_channel: Channel, split_num):
        self.start = True
        self.cfg = cfg
        while True:
            recv_list = []
            for _ in range(split_num):
                recv_list.append(await data_channel.get(async_op=True).async_wait())
            rollout_batch = cat_list_of_dict_tensor(recv_list)
            self.add_rollout_batch(rollout_batch, extra_preprocess=False)

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self.buffer, f)


from rlinf.data.embodied_io_struct import Trajectory

class TrajectoryCache:
    """FIFO cache for storing loaded trajectories."""

    def __init__(self, max_size: int = 5):
        self.cache: OrderedDict[str, Trajectory] = OrderedDict()
        self.max_size = max_size

    def get(self, trajectory_uuid: str) -> Optional[Trajectory]:
        return self.cache.get(trajectory_uuid)

    def put(self, trajectory_uuid: str, trajectory: Trajectory):
        if trajectory_uuid not in self.cache:
            self.cache[trajectory_uuid] = trajectory
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        else:
            # Update existing without changing position
            self.cache[trajectory_uuid] = trajectory

    def clear(self):
        self.cache.clear()

class TrajectoryReplayBuffer:
    """
    Simplified trajectory-based replay buffer.
    Directly stores batched trajectories (shape: [T, B, ...]) without splitting.
    Supports chunk-level sampling with caching.
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: Optional[int] = 1234,
        storage_dir: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 5,
        storage_format: str = "pt",
    ):
        """
        Initialize trajectory-based replay buffer.

        Args:
            device: Device to output samples on
            seed: Random seed for reproducibility
            storage_dir: Directory to store trajectories (None uses temp directory)
            enable_cache: Whether to enable trajectory caching
            cache_size: Maximum number of trajectories to cache in memory
            storage_format: Storage format ("pt", "pkl")
        """
        self.device = device
        self.storage_format = storage_format
        self.enable_cache = enable_cache

        # Storage directory
        if storage_dir is None:
            storage_dir = tempfile.mkdtemp(prefix="trajectory_replay_buffer_")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # Trajectory index: dict mapping trajectory_uuid to trajectory metadata
        # Each entry: {
        #   "uuid": str,
        #   "num_samples": int,  # T * B (total samples in this trajectory)
        #   "path": str,  # file path
        #   "trajectory_id": int,  # sequential ID
        #   "max_episode_length": int,  # max episode length
        #   "shape": tuple,  # (T, B, ...)
        # }
        self._trajectory_index: dict[str, dict] = {}
        self._trajectory_uuid_list: list[str] = []  # Ordered list of trajectory UUIDs
        self._trajectory_counter = 0  # Next trajectory ID to use

        # Trajectory cache for storing loaded trajectories in memory
        self._trajectory_cache = TrajectoryCache(cache_size) if enable_cache else None

        # Buffer state
        self.size = 0  # Current number of trajectories
        self._total_samples = 0  # Total number of samples across all trajectories

        # Random seed
        self.seed = seed
        self.random_generator: Optional[torch.Generator] = None

        self._load_metadata()
        self._init_random_generator(self.seed)

    def _init_random_generator(self, seed):
        """(Re)initialize numpy and torch RNGs from self.seed."""
        np.random.seed(seed)
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)

    def _get_trajectory_path(self, trajectory_uuid: str) -> str:
        """Get file path for a trajectory."""
        ext = ".pt" if self.storage_format == "pt" else ".pkl"
        return os.path.join(self.storage_dir, f"trajectory_{self._trajectory_counter}_{trajectory_uuid}{ext}")

    def _get_metadata_path(self) -> str:
        """Get path to metadata file."""
        return os.path.join(self.storage_dir, "metadata.json")

    def _get_trajectory_index_path(self) -> str:
        """Get path to trajectory index file."""
        return os.path.join(self.storage_dir, "trajectory_index.json")

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata = {
            "storage_format": self.storage_format,
            "size": self.size,
            "total_samples": self._total_samples,
            "trajectory_counter": self._trajectory_counter,
            "seed": self.seed,
        }
        with open(self._get_metadata_path(), "w") as f:
            json.dump(metadata, f)

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = self._get_metadata_path()
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.storage_format = metadata.get("storage_format", self.storage_format)
            self.size = metadata.get("size", 0)
            self._total_samples = metadata.get("total_samples", 0)
            self._trajectory_counter = metadata.get("trajectory_counter", 0)
            self.seed = metadata.get("seed", self.seed)

            # Load trajectory index if exists
            index_path = self._get_trajectory_index_path()
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                self._trajectory_index = index_data.get("trajectory_index", {})
                self._trajectory_uuid_list = index_data.get("trajectory_uuid_list", [])

    def _save_trajectory_index(self):
        """Save trajectory index to disk."""
        index_data = {
            "trajectory_index": self._trajectory_index,
            "trajectory_uuid_list": self._trajectory_uuid_list,
        }
        with open(self._get_trajectory_index_path(), "w") as f:
            json.dump(index_data, f)

    def _save_trajectory(self, trajectory: Trajectory, trajectory_uuid: str):
        """Save a single episode to disk as a dictionary."""
        trajectory_path = self._get_trajectory_path(trajectory_uuid)

        # Convert Trajectory to dictionary for more stable storage
        trajectory_dict = {}
        for field_name in trajectory.__dataclass_fields__.keys():
            value = getattr(trajectory, field_name, None)
            if value is not None:
                trajectory_dict[field_name] = value

        if self.storage_format == "pt":
            torch.save(trajectory_dict, trajectory_path)
        else:
            with open(trajectory_path, "wb") as f:
                pkl.dump(trajectory_dict, f)

    def _load_trajectory(self, trajectory_uuid: str) -> Trajectory:
        """Load a trajectory from disk and reconstruct Trajectory object (with caching)."""
        # Check cache first
        if self._trajectory_cache is not None:
            cached = self._trajectory_cache.get(trajectory_uuid)
            if cached is not None:
                return cached

        # Get trajectory info from index
        if trajectory_uuid not in self._trajectory_index:
            raise ValueError(f"Trajectory {trajectory_uuid} not found in index")

        trajectory_info = self._trajectory_index[trajectory_uuid]
        trajectory_path = trajectory_info["path"]

        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file not found at {trajectory_path}")

        # Load trajectory dictionary
        if self.storage_format == "pt":
            trajectory_dict = torch.load(trajectory_path, map_location="cpu")
        else:
            with open(trajectory_path, "rb") as f:
                trajectory_dict = pkl.load(f)

        # Reconstruct Trajectory object from dictionary
        trajectory = Trajectory(max_episode_length=trajectory_info["max_episode_length"])
        for field_name, value in trajectory_dict.items():
            setattr(trajectory, field_name, value)

        return trajectory

    def add_trajectories(self, trajectories: list[Trajectory]):
        """
        Add trajectories to the buffer.
        Each trajectory is directly stored as-is (shape: [T, B, ...]).

        Args:
            trajectories: List of Trajectory objects, each with shape [T, B, ...]
                     where T*B is the total number of samples in the trajectory.
        """
        if not trajectories:
            return

        for trajectory in trajectories:
            # Generate UUID for this trajectory
            trajectory_uuid = str(uuid.uuid4())

            # Calculate total samples: T * B
            if trajectory.prev_logprobs is not None:
                T, B = trajectory.prev_logprobs.shape[:2]
                num_samples = T * B
                trajectory_shape = trajectory.prev_logprobs.shape
            else:
                continue  # Skip empty trajectories

            # Save trajectory to disk
            trajectory_path = self._get_trajectory_path(trajectory_uuid)
            self._save_trajectory(trajectory, trajectory_uuid)

            # Add to index
            trajectory_info = {
                "uuid": trajectory_uuid,
                "num_samples": num_samples,
                "path": trajectory_path,
                "trajectory_id": self._trajectory_counter,
                "max_episode_length": trajectory.max_episode_length,
                "shape": tuple(trajectory_shape),
            }
            self._trajectory_index[trajectory_uuid] = trajectory_info
            self._trajectory_uuid_list.append(trajectory_uuid)

            # Update counters
            self._trajectory_counter += 1
            self.size += 1
            self._total_samples += num_samples

            # Cache the trajectory if enabled
            if self._trajectory_cache is not None:
                self._trajectory_cache.put(trajectory_uuid, trajectory)

        # Save metadata
        self._save_metadata()
        self._save_trajectory_index()

    def sample(
        self,
        num_chunks: int = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.

        Args:
            num_chunks: Minimum number of chunks (transitions) to return

        Returns:
            Dictionary with rollout batch format [B, ...]
        """
        assert num_chunks is not None
        return self.sample_chunks(num_chunks)

    def sample_chunks(self, num_chunks: int) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.
        Each chunk is a single transition from any trajectory.

        Args:
            num_chunks: Number of chunks (transitions) to sample

        Returns:
            Dictionary with batch format [B, ...] where B = num_chunks
        """
        if self._total_samples == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        if num_chunks > self._total_samples:
            num_chunks = self._total_samples

        # Sample chunk indices directly from total samples
        sample_ids = torch.randint(
            low=0, high=self._total_samples, size=(num_chunks,), generator=self.random_generator
        )

        # Convert global sample indices to (trajectory_idx, local_sample_idx) pairs
        chunks = []
        for sample_id in sample_ids:
            # Find which trajectory this sample belongs to
            trajectory_start = 0
            trajectory_uuid = None
            local_sample_idx = None

            for uuid in self._trajectory_uuid_list:
                trajectory_info = self._trajectory_index[uuid]
                trajectory_samples = trajectory_info["num_samples"]
                trajectory_end = trajectory_start + trajectory_samples

                if trajectory_start <= sample_id < trajectory_end:
                    trajectory_uuid = uuid
                    local_sample_idx = sample_id - trajectory_start
                    break

                trajectory_start = trajectory_end

            if trajectory_uuid is None:
                continue

            # Try to get from cache first
            trajectory = None
            if self._trajectory_cache is not None:
                trajectory = self._trajectory_cache.get(trajectory_uuid)

            # If not in cache, load from disk
            if trajectory is None:
                trajectory = self._load_trajectory(trajectory_uuid)

            # Convert local sample index to (t, b) coordinates
            trajectory_info = self._trajectory_index[trajectory_uuid]
            _, B = trajectory_info["shape"][:2]
            t_idx = local_sample_idx // B
            b_idx = local_sample_idx % B

            # Extract the chunk at (t_idx, b_idx)
            chunk = self._extract_chunk_from_trajectory(trajectory, t_idx, b_idx)
            if chunk:
                chunks.append(chunk)

        # Merge chunks into batch [B, ...]
        batch = self._merge_chunks_to_batch(chunks)
        return batch

    def _extract_chunk_from_trajectory(self, trajectory: Trajectory, t_idx: int, b_idx: int) -> dict:
        """
        Extract a single chunk (time step) from a trajectory at position (t_idx, b_idx).

        Args:
            trajectory: Trajectory object with shape [T, B, ...]
            t_idx: Time index
            b_idx: Batch index within the trajectory

        Returns:
            Dictionary containing the chunk data
        """
        chunk = {}

        # Extract tensor fields
        tensor_fields = trajectory.__dataclass_fields__.keys()
        for field in tensor_fields:
            if field in ["obs", "curr_obs_idx", "next_obs_idx", "forward_inputs"]:
                continue
            tensor = getattr(trajectory, field)
            if tensor is not None:
                chunk[field] = tensor[t_idx, b_idx]

        # Extract obs (dict of tensors)
        if trajectory.obs and trajectory.curr_obs_idx is not None and trajectory.next_obs_idx is not None:
            chunk["curr_obs"] = {}
            chunk["next_obs"] = {}
            for key, tensor in trajectory.obs.items():
                if tensor is not None:
                    curr_obs_idx = trajectory.curr_obs_idx[t_idx, b_idx]
                    next_obs_idx = trajectory.next_obs_idx[t_idx, b_idx]
                    chunk["curr_obs"][key] = tensor[curr_obs_idx, b_idx]
                    chunk["next_obs"][key] = tensor[next_obs_idx, b_idx]

        # Extract forward_inputs (dict of tensors)
        if trajectory.forward_inputs:
            chunk["forward_inputs"] = {}
            for key, tensor in trajectory.forward_inputs.items():
                if tensor is not None:
                    chunk["forward_inputs"][key] = tensor[t_idx, b_idx]

        return chunk

    def _merge_chunks_to_batch(self, chunks: list[dict]) -> dict[str, torch.Tensor]:
        """
        Merge a list of chunks into a batch dictionary.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Batch dictionary with shape [B, ...] where B = len(chunks)
        """
        if not chunks:
            return {}

        batch = {}
        first_chunk = chunks[0]

        for key, value in first_chunk.items():
            if isinstance(value, torch.Tensor):
                tensors = [chunk[key] for chunk in chunks if key in chunk]
                if tensors:
                    batch[key] = torch.stack(tensors, dim=0)  # [B, ...]
            elif isinstance(value, dict):
                # Handle nested dicts (obs, forward_inputs)
                nested_dicts = [chunk[key] for chunk in chunks if key in chunk]
                if nested_dicts:
                    all_keys = set()
                    for d in nested_dicts:
                        all_keys.update(d.keys())

                    nested_batch = {}
                    for nested_key in all_keys:
                        nested_tensors = [
                            d[nested_key] for d in nested_dicts
                            if nested_key in d and isinstance(d[nested_key], torch.Tensor)
                        ]
                        if nested_tensors:
                            nested_batch[nested_key] = torch.stack(nested_tensors, dim=0)  # [B, ...]
                    if nested_batch:
                        batch[key] = nested_batch

        return batch

    def __len__(self) -> int:
        """Return current buffer size (number of trajectories)."""
        return self.size

    @property
    def total_samples(self) -> int:
        """Return total number of samples across all trajectories."""
        return self._total_samples

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    async def is_ready_async(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size

    def clear(self):
        # Clear index
        self._trajectory_index.clear()
        self._trajectory_uuid_list.clear()

        # Clear cache
        if self._trajectory_cache is not None:
            self._trajectory_cache.clear()

        # Reset state
        self.size = 0
        self._total_samples = 0
        self._trajectory_counter = 0

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics."""
        # Calculate samples per episode (all episodes have the same number)
        samples_per_trajectory = (
            self._trajectory_index[self._trajectory_uuid_list[0]]["num_samples"]
            if self._trajectory_uuid_list else 0
        )

        stats = {
            "num_trajectories": self.size,
            "total_samples": self._total_samples,
            "samples_per_trajectory": samples_per_trajectory,
            "cache_size": len(self._trajectory_cache.cache) if self._trajectory_cache else 0,
        }

        return stats

    def save(self, save_path: str):
        """
        Save buffer state (metadata and indices) to save_path.
        """
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "storage_dir": self.storage_dir,
                    "storage_format": self.storage_format,
                    "size": self.size,
                    "total_samples": self._total_samples,
                    "trajectory_counter": self._trajectory_counter,
                    "seed": self.seed,
                },
                f,
            )
        
        # Save trajectory index and uuid list
        index_path = os.path.join(save_path, "trajectory_index.json")
        index_data = {
            "trajectory_index": self._trajectory_index,
            "trajectory_uuid_list": self._trajectory_uuid_list,
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f)

    def load(
        self, 
        load_path: str,
        partial_load: bool = False,
        load_rank: int = 0, 
        load_split_num: int = 1,
    ):
        """
        Load buffer state from saved metadata.
        
        Args:
            load_path: Path to the directory containing metadata.json and trajectory_index.json
            partial_load: If True, only load a portion of trajectories based on load_rank and load_split_num
            load_rank: Rank index (0-based) for partial loading. Only used when partial_load=True
            load_split_num: Number of splits to divide trajectories into. Only used when partial_load=True
        """
        metadata_path = os.path.join(load_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update instance attributes from metadata
        self.storage_dir = metadata["storage_dir"]
        self.storage_format = metadata.get("storage_format", "pt")
        if "seed" in metadata:
            self.seed = metadata["seed"]
            self._init_random_generator(self.seed)

        # Load trajectory index and uuid list from save_path
        index_path = os.path.join(load_path, "trajectory_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Trajectory index not found at {index_path}")
        
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        full_trajectory_index = index_data.get("trajectory_index", {})
        full_trajectory_uuid_list = index_data.get("trajectory_uuid_list", [])
        
        # Handle partial loading
        if partial_load:
            if load_rank < 0 or load_rank >= load_split_num:
                raise ValueError(
                    f"load_rank ({load_rank}) must be in range [0, {load_split_num})"
                )
            if load_split_num <= 0:
                raise ValueError(f"load_split_num ({load_split_num}) must be > 0")
            
            # Split trajectory_uuid_list into load_split_num parts
            total_trajectories = len(full_trajectory_uuid_list)
            trajectories_per_split = total_trajectories // load_split_num
            remainder = total_trajectories % load_split_num
            
            # Calculate start and end indices for this rank
            start_idx = load_rank * trajectories_per_split + min(load_rank, remainder)
            end_idx = start_idx + trajectories_per_split + (1 if load_rank < remainder else 0)
            
            # Extract the portion for this rank
            self._trajectory_uuid_list = full_trajectory_uuid_list[start_idx:end_idx]
            
            # Filter trajectory_index to only include trajectories in this rank's portion
            self._trajectory_index = {
                uuid: full_trajectory_index[uuid] 
                for uuid in self._trajectory_uuid_list 
                if uuid in full_trajectory_index
            }
                
            # Update size, total_samples, and trajectory_counter based on loaded portion
            self.size = len(self._trajectory_uuid_list)
            self._total_samples = sum(
                trajectory_info.get("num_samples", 0)
                for trajectory_info in self._trajectory_index.values()
            )
            # trajectory_counter should be set to the max trajectory_id in the loaded portion + 1
            if self._trajectory_index:
                max_trajectory_id = max(
                    trajectory_info.get("trajectory_id", 0) 
                    for trajectory_info in self._trajectory_index.values()
                )
                self._trajectory_counter = max_trajectory_id + 1
            else:
                self._trajectory_counter = 0
        else:
            # Full load
            self._trajectory_index = full_trajectory_index
            self._trajectory_uuid_list = full_trajectory_uuid_list
            self.size = metadata.get("size", 0)
            self._total_samples = metadata.get("total_samples", 0)
            self._trajectory_counter = metadata.get("trajectory_counter", 0)

    def clear_cache(self):
        """Clear trajectory cache."""
        if self._trajectory_cache is not None:
            self._trajectory_cache.clear()
