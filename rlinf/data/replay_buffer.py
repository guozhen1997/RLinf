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

from rlinf.scheduler import Channel
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor


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


from rlinf.data.embodied_io_struct import Episode, pad_and_stack_episodes

class EpisodeCache:
    """FIFO cache for storing loaded episodes."""

    def __init__(self, max_size: int = 5):
        self.cache: OrderedDict[str, Episode] = OrderedDict()
        self.max_size = max_size

    def get(self, episode_uuid: str) -> Optional[Episode]:
        return self.cache.get(episode_uuid)

    def put(self, episode_uuid: str, episode: Episode):
        if episode_uuid not in self.cache:
            self.cache[episode_uuid] = episode
            # Evict oldest if cache is full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
        else:
            # Update existing without changing position
            self.cache[episode_uuid] = episode

    def clear(self):
        self.cache.clear()

class EpisodeReplayBuffer:
    """
    Episode-based replay buffer for SAC algorithm.
    Supports episode-level and chunk-level sampling.
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: Optional[int] = 1234,
        storage_dir: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 5,
        storage_format: str = "pt",
        sample_window_size: Optional[int] = None,
    ):
        """
        Initialize episode-based replay buffer.

        Args:
            device: Device to output samples on
            seed: Random seed for reproducibility
            storage_dir: Directory to store episodes (None uses temp directory)
            enable_cache: Whether to enable episode caching
            cache_size: Maximum number of episodes to cache
            storage_format: Storage format ("pt", "pkl")
            sample_window_size: Number of most recent episodes to sample from.
                If None or <= 0, sample from all stored episodes.
        """
        # Capacity is intentionally not stored/used anymore.
        self.device = device
        self.storage_format = storage_format
        self.sample_window_size = sample_window_size
        self.start = False

        # Storage directory
        if storage_dir is None:
            storage_dir = tempfile.mkdtemp(prefix="episode_replay_buffer_")
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        # Episode index: dict mapping episode_uuid to episode metadata
        # Each entry: {
        #   "uuid": str,
        #   "length": int,  # number of transitions
        #   "path": str,  # file path
        #   "episode_id": int  # sequential ID
        # }
        self._episode_index: dict[str, dict] = {}
        self._episode_uuid_list: list[str] = []  # Ordered list of episode UUIDs
        self._episode_counter = 0  # Next episode ID to use

        # Episode cache for storing loaded episodes
        self.enable_cache = enable_cache
        self._episode_cache = EpisodeCache(cache_size) if enable_cache else None

        # Buffer state
        self.size = 0  # Current number of episodes
        self._total_transitions = 0  # Total number of transitions across all episodes

        # Random seed (may be overridden by metadata)
        self.seed = seed
        self.random_generator: Optional[torch.Generator] = None

        self._load_metadata()

        self._init_random_generator(self.seed)

    def _init_random_generator(self, seed):
        """(Re)initialize numpy and torch RNGs from self.seed."""
        np.random.seed(seed)
        self.random_generator = torch.Generator()
        self.random_generator.manual_seed(seed)

    def _get_episode_path(self, episode_uuid: str) -> str:
        """Get file path for an episode."""
        ext = ".pt" if self.storage_format == "pt" else ".pkl"
        return os.path.join(self.storage_dir, f"episode_{episode_uuid}{ext}")

    def _get_metadata_path(self) -> str:
        """Get path to metadata file."""
        return os.path.join(self.storage_dir, "metadata.json")

    def _get_episode_index_path(self) -> str:
        """Get path to episode index file."""
        return os.path.join(self.storage_dir, "episode_index.json")

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata = {
            "storage_format": self.storage_format,
            "size": self.size,
            "total_transitions": self._total_transitions,
            "episode_counter": self._episode_counter,
            "seed": self.seed,
            "sample_window_size": self.sample_window_size,
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
            self._total_transitions = metadata.get("total_transitions", 0)
            self._episode_counter = metadata.get("episode_counter", 0)
            self.seed = metadata.get("seed", self.seed)
            self.sample_window_size = metadata.get(
                "sample_window_size", self.sample_window_size
            )

            # Load episode index if exists
            index_path = self._get_episode_index_path()
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                self._episode_index = index_data.get("episode_index", {})
                self._episode_uuid_list = index_data.get("episode_uuid_list", [])

    def _save_episode_index(self):
        """Save episode index to disk."""
        index_data = {
            "episode_index": self._episode_index,
            "episode_uuid_list": self._episode_uuid_list,
        }
        with open(self._get_episode_index_path(), "w") as f:
            json.dump(index_data, f)

    def _count_episode_length(self, episode: Episode) -> int:
        """Count the number of steps in an episode."""
        # Episode.dones has shape [T+1, ...] where T is the number of transitions/steps
        if episode.dones is not None:
            return episode.dones.shape[0] - 1
        elif episode.rewards is not None:
            return episode.rewards.shape[0]
        else:
            raise ValueError("Episode has no dones or rewards")

    def _save_episode(self, episode: Episode, episode_uuid: str):
        """Save a single episode to disk."""
        episode_path = self._get_episode_path(episode_uuid)
        if self.storage_format == "pt":
            torch.save(episode, episode_path)
        else:
            with open(episode_path, "wb") as f:
                pkl.dump(episode, f)

    def _load_episode(self, episode_uuid: str) -> Episode:
        """Load an episode from disk (with caching)."""
        # Check cache first
        if self._episode_cache is not None:
            cached = self._episode_cache.get(episode_uuid)
            if cached is not None:
                return cached

        # Get episode info from index
        if episode_uuid not in self._episode_index:
            raise ValueError(f"Episode {episode_uuid} not found in index")
        
        episode_info = self._episode_index[episode_uuid]
        episode_path = episode_info["path"]

        if not os.path.exists(episode_path):
            raise FileNotFoundError(f"Episode file not found at {episode_path}")

        # Load episode
        if self.storage_format == "pt":
            episode = torch.load(episode_path, map_location="cpu")
        else:
            with open(episode_path, "rb") as f:
                episode = pkl.load(f)

        # Cache it
        if self._episode_cache is not None:
            self._episode_cache.put(episode_uuid, episode)

        return episode

    def _get_sample_index_range(self) -> tuple[int, int]:
        """
        Get the [start, end) index range to use for sampling.

        If sample_window_size is set, only the most recent `sample_window_size`
        episodes are considered; otherwise, all episodes are used.
        """
        if self.size == 0:
            return 0, 0

        if self.sample_window_size is None or self.sample_window_size <= 0:
            return 0, self.size

        window = min(self.sample_window_size, self.size)
        start = self.size - window
        end = self.size
        return start, end

    def add_episodes(self, episodes: list[Episode]):
        """
        Add episodes to the buffer.
        Each episode is assigned a UUID and stored with its metadata.
        """
        if not episodes:
            return
        
        for episode in episodes:
            # Generate UUID for this episode
            episode_uuid = str(uuid.uuid4())
            episode_length = self._count_episode_length(episode)
            
            if episode_length == 0:
                continue  # Skip empty episodes
            
            # Save episode to disk
            episode_path = self._get_episode_path(episode_uuid)
            self._save_episode(episode, episode_uuid)
            
            # Add to index
            episode_info = {
                "uuid": episode_uuid,
                "length": episode_length,
                "path": episode_path,
                "episode_id": self._episode_counter,
            }
            self._episode_index[episode_uuid] = episode_info
            self._episode_uuid_list.append(episode_uuid)
            
            # Update counters
            self._episode_counter += 1
            self.size += 1
            self._total_transitions += episode_length
            
            # Cache the episode if enabled
            if self._episode_cache is not None:
                self._episode_cache.put(episode_uuid, episode)
        # Capacity-based truncation is disabled; we only update metadata.
        self._save_metadata()
        self._save_episode_index()

    def _extract_chunks_from_episode(
        self, episode: Episode, start_idx: int, num_chunks: int
    ) -> list[dict]:
        """
        Extract chunks from an episode starting at start_idx.
        
        Args:
            episode: Episode object
            start_idx: Starting index in the episode
            num_chunks: Number of chunks to extract
            
        Returns:
            List of chunk dictionaries (each chunk is a transition)
        """
        chunks = []
        episode_length = self._count_episode_length(episode)
        
        for i in range(start_idx, min(start_idx + num_chunks, episode_length)):
            chunk = {}
            
            # Extract obs (obs is dict[str, Tensor] with shape [T+1, ...])
            if episode.obs:
                obs_dict = {}
                next_obs_dict = {}
                for key, tensor in episode.obs.items():
                    if i < tensor.shape[0] - 1:
                        obs_dict[key] = tensor[i]
                        next_obs_dict[key] = tensor[i + 1]
                if obs_dict:
                    chunk["obs"] = obs_dict
                    chunk["next_obs"] = next_obs_dict
            
            # Extract rewards (shape [T, ...])
            if episode.rewards is not None and i < episode.rewards.shape[0]:
                chunk["rewards"] = episode.rewards[i]
            
            # Extract terminations, truncations, dones (shape [T+1, ...])
            if episode.terminations is not None and i < episode.terminations.shape[0]:
                chunk["terminations"] = episode.terminations[i]
            if episode.truncations is not None and i < episode.truncations.shape[0]:
                chunk["truncations"] = episode.truncations[i]
            if episode.dones is not None and i < episode.dones.shape[0]:
                chunk["dones"] = episode.dones[i]
            
            # Extract prev_logprobs (shape [T, ...])
            if episode.prev_logprobs is not None and i < episode.prev_logprobs.shape[0]:
                chunk["prev_logprobs"] = episode.prev_logprobs[i]
            
            # Extract prev_values (shape [T+1, ...])
            if episode.prev_values is not None and i < episode.prev_values.shape[0]:
                chunk["prev_values"] = episode.prev_values[i]
            
            # Extract forward_inputs (dict[str, Tensor] with shape [T, ...])
            if episode.forward_inputs:
                forward_dict = {}
                for key, tensor in episode.forward_inputs.items():
                    if i < tensor.shape[0]:
                        forward_dict[key] = tensor[i]
                if forward_dict:
                    chunk.update(forward_dict)
            
            chunks.append(chunk)
        
        return chunks

    def sample(
        self,
        mode: str = "episode",
        num_episodes: int = None,
        num_chunks: int = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample episodes or chunks from the buffer.

        Args:
            num_episodes: Number of episodes to sample
            num_chunks: Minimum number of chunks (transitions) to return
            mode: Sampling mode ("episode" or "chunk")

        Returns:
            Dictionary with rollout batch format [T, B, ...]
        """
        if mode == "episode":
            assert num_episodes is not None
            # If min_chunks is not provided, default to 0 (no minimum).
            min_chunks = 0 if num_chunks is None else num_chunks
            return self.sample_episodes(num_episodes, min_chunks)
        elif mode == "chunk":
            assert num_chunks is not None
            return self.sample_chunks(num_chunks)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def sample_episodes(self, num_episodes: int, min_chunks: int) -> dict[str, torch.Tensor]:
        """
        Sample complete episodes from the buffer.
        Guarantees at least min_chunks transitions in total.
        
        Args:
            num_episodes: Number of episodes to sample
            min_chunks: Minimum number of chunks (transitions) to return
            
        Returns:
            Dictionary with rollout batch format [T, B, ...]
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Determine sampling window [start, end)
        start_idx, end_idx = self._get_sample_index_range()
        window_size = end_idx - start_idx
        if window_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        if num_episodes > window_size:
            num_episodes = window_size

        # Randomly sample episode indices within the window
        relative_indices = torch.randperm(
            window_size, generator=self.random_generator
        )[:num_episodes]
        episode_indices = (relative_indices + start_idx).tolist()
        
        sampled_episodes = []
        total_chunks = 0
        
        for idx in episode_indices:
            episode_uuid = self._episode_uuid_list[idx]
            episode = self._load_episode(episode_uuid)
            sampled_episodes.append(episode)
            total_chunks += self._count_episode_length(episode)
        
        # If we don't have enough chunks, sample more episodes inside the window
        while total_chunks < min_chunks and len(sampled_episodes) < window_size:
            # Candidate indices within the window that haven't been used yet
            all_window_indices = list(range(start_idx, end_idx))
            used_set = set(episode_indices)
            remaining_indices = [i for i in all_window_indices if i not in used_set]
            if not remaining_indices:
                break
            
            additional_idx = remaining_indices[
                torch.randint(0, len(remaining_indices), (1,), generator=self.random_generator).item()
            ]
            episode_uuid = self._episode_uuid_list[additional_idx]
            episode = self._load_episode(episode_uuid)
            sampled_episodes.append(episode)
            total_chunks += self._count_episode_length(episode)
            episode_indices.append(additional_idx)
        
        # Convert episodes to rollout batch format
        rollout_batch = pad_and_stack_episodes(sampled_episodes)
        
        return rollout_batch

    def sample_chunks(self, num_chunks: int) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.
        Chunks may come from different episodes and don't need to be complete episodes.
        
        Args:
            num_chunks: Number of chunks (transitions) to sample
            
        Returns:
            Dictionary with batch format [B, ...] (flattened)
        """
        if self._total_transitions == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        if num_chunks > self._total_transitions:
            num_chunks = self._total_transitions

        # Determine sampling window [start, end)
        start_idx, end_idx = self._get_sample_index_range()
        window_size = end_idx - start_idx
        if window_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        # Sample chunks randomly across episodes in the window
        chunks = []
        chunks_sampled = 0

        # Create a list of (episode_idx, episode_length) for weighted sampling
        window_indices = list(range(start_idx, end_idx))
        episode_weights = [
            self._episode_index[self._episode_uuid_list[i]]["length"]
            for i in window_indices
        ]
        total_weight = sum(episode_weights)
        
        while chunks_sampled < num_chunks:
            # Sample an episode (weighted by length)
            rand_val = torch.rand(1, generator=self.random_generator).item() * total_weight
            cumsum = 0
            selected_window_pos = 0
            for i, weight in enumerate(episode_weights):
                cumsum += weight
                if rand_val <= cumsum:
                    selected_window_pos = i
                    break
            
            # Map window position back to global episode index
            episode_global_idx = window_indices[selected_window_pos]
            episode_uuid = self._episode_uuid_list[episode_global_idx]
            episode = self._load_episode(episode_uuid)
            episode_length = self._count_episode_length(episode)
            
            # Sample a random chunk from this episode
            chunk_idx = torch.randint(0, episode_length, (1,), generator=self.random_generator).item()
            episode_chunks = self._extract_chunks_from_episode(episode, chunk_idx, 1)
            
            if episode_chunks:
                chunks.append(episode_chunks[0])
                chunks_sampled += 1
        
        # Merge chunks into batch
        # For chunk sampling, we stack chunks along batch dimension [B, ...]
        # This is different from episode sampling which returns [T, B, ...]
        batch = {}
        if not chunks:
            return batch
        
        first_chunk = chunks[0]
        
        for key, value in first_chunk.items():
            if isinstance(value, torch.Tensor):
                tensors = [chunk[key] for chunk in chunks if key in chunk]
                if tensors:
                    batch[key] = torch.stack(tensors, dim=0)  # [B, ...]
            elif isinstance(value, dict):
                # Handle nested dicts (like obs, forward_inputs)
                nested_dicts = [chunk[key] for chunk in chunks if key in chunk]
                if nested_dicts:
                    # Collect all keys from all dicts
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
            else:
                batch[key] = [chunk[key] for chunk in chunks if key in chunk]
        
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
        # Clear index
        self._episode_index.clear()
        self._episode_uuid_list.clear()

        # Clear cache
        if self._episode_cache is not None:
            self._episode_cache.clear()

        # Reset state
        self.size = 0
        self._total_transitions = 0
        self._episode_counter = 0

    def get_stats(self) -> dict[str, float]:
        """Get buffer statistics."""
        stats = {
            "num_episodes": self.size,
            "total_transitions": self._total_transitions,
            "avg_episode_length": (
                self._total_transitions / self.size if self.size > 0 else 0.0
            ),
            "cache_size": len(self._episode_cache.cache) if self._episode_cache else 0,
        }

        # Calculate disk usage
        total_size = 0
        for episode_uuid in self._episode_uuid_list:
            episode_info = self._episode_index.get(episode_uuid)
            if episode_info:
                episode_path = episode_info["path"]
                if os.path.exists(episode_path):
                    total_size += os.path.getsize(episode_path)
        stats["disk_size_mb"] = total_size / (1024 * 1024)

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
                    "total_transitions": self._total_transitions,
                    "episode_counter": self._episode_counter,
                    "sample_window_size": self.sample_window_size,
                    "seed": self.seed,
                },
                f,
            )
        
        # Save episode index and uuid list
        index_path = os.path.join(save_path, "episode_index.json")
        index_data = {
            "episode_index": self._episode_index,
            "episode_uuid_list": self._episode_uuid_list,
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
            load_path: Path to the directory containing metadata.json and episode_index.json
            partial_load: If True, only load a portion of episodes based on load_rank and load_split_num
            load_rank: Rank index (0-based) for partial loading. Only used when partial_load=True
            load_split_num: Number of splits to divide episodes into. Only used when partial_load=True
        """
        metadata_path = os.path.join(load_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update instance attributes from metadata
        self.storage_dir = metadata["storage_dir"]
        self.storage_format = metadata.get("storage_format", "pt")
        self.sample_window_size = metadata.get("sample_window_size")
        if "seed" in metadata:
            self.seed = metadata["seed"]
            self._init_random_generator(self.seed)

        # Load episode index and uuid list from save_path
        index_path = os.path.join(load_path, "episode_index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Episode index not found at {index_path}")
        
        with open(index_path, "r") as f:
            index_data = json.load(f)
        
        full_episode_index = index_data.get("episode_index", {})
        full_episode_uuid_list = index_data.get("episode_uuid_list", [])
        
        # Handle partial loading
        if partial_load:
            if load_rank < 0 or load_rank >= load_split_num:
                raise ValueError(
                    f"load_rank ({load_rank}) must be in range [0, {load_split_num})"
                )
            if load_split_num <= 0:
                raise ValueError(f"load_split_num ({load_split_num}) must be > 0")
            
            # Split episode_uuid_list into load_split_num parts
            total_episodes = len(full_episode_uuid_list)
            episodes_per_split = total_episodes // load_split_num
            remainder = total_episodes % load_split_num
            
            # Calculate start and end indices for this rank
            start_idx = load_rank * episodes_per_split + min(load_rank, remainder)
            end_idx = start_idx + episodes_per_split + (1 if load_rank < remainder else 0)
            
            # Extract the portion for this rank
            self._episode_uuid_list = full_episode_uuid_list[start_idx:end_idx]
            
            # Filter episode_index to only include episodes in this rank's portion
            self._episode_index = {
                uuid: full_episode_index[uuid] 
                for uuid in self._episode_uuid_list 
                if uuid in full_episode_index
            }
            
            # Update size, total_transitions, and episode_counter based on loaded portion
            self.size = len(self._episode_uuid_list)
            self._total_transitions = sum(
                episode_info.get("length", 0) 
                for episode_info in self._episode_index.values()
            )
            # episode_counter should be set to the max episode_id in the loaded portion + 1
            if self._episode_index:
                max_episode_id = max(
                    episode_info.get("episode_id", 0) 
                    for episode_info in self._episode_index.values()
                )
                self._episode_counter = max_episode_id + 1
            else:
                self._episode_counter = 0
        else:
            # Full load
            self._episode_index = full_episode_index
            self._episode_uuid_list = full_episode_uuid_list
            self.size = metadata.get("size", 0)
            self._total_transitions = metadata.get("total_transitions", 0)
            self._episode_counter = metadata.get("episode_counter", 0)

    def clear_cache(self):
        """Clear episode cache."""
        if self._episode_cache is not None:
            self._episode_cache.clear()
