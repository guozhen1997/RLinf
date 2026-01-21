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


import bisect
import json
import os
import pickle as pkl
import tempfile
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

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
        seed: Optional[int] = 1234,
        storage_dir: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 5,
        storage_format: str = "pt",
        sample_window_size: int = 100,
    ):
        """
        Initialize trajectory-based replay buffer.

        Args:
            seed: Random seed for reproducibility
            storage_dir: Directory to store trajectories (None uses temp directory)
            enable_cache: Whether to enable trajectory caching
            cache_size: Maximum number of trajectories to cache in memory
            storage_format: Storage format ("pt", "pkl")
        """
        self.storage_format = storage_format
        self.enable_cache = enable_cache
        self.sample_window_size = sample_window_size

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

    def _get_trajectory_path(
        self, trajectory_uuid: str, trajectory_id: Optional[int] = None
    ) -> str:
        """Get file path for a trajectory."""
        if trajectory_id is None:
            trajectory_id = self._trajectory_counter
        ext = ".pt" if self.storage_format == "pt" else ".pkl"
        return os.path.join(
            self.storage_dir, f"trajectory_{trajectory_id}_{trajectory_uuid}{ext}"
        )

    def _get_metadata_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to metadata file."""
        base_dir = base_dir or self.storage_dir
        return os.path.join(base_dir, "metadata.json")

    def _get_trajectory_index_path(self, base_dir: Optional[str] = None) -> str:
        """Get path to trajectory index file."""
        base_dir = base_dir or self.storage_dir
        return os.path.join(base_dir, "trajectory_index.json")

    def _save_metadata(self, save_path: Optional[str] = None):
        """Save metadata to disk."""
        metadata = {
            "storage_format": self.storage_format,
            "size": self.size,
            "total_samples": self._total_samples,
            "trajectory_counter": self._trajectory_counter,
            "seed": self.seed,
        }
        with open(self._get_metadata_path(save_path), "w") as f:
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

    def _save_trajectory_index(self, save_path: Optional[str] = None):
        """Save trajectory index to disk."""
        index_data = {
            "trajectory_index": self._trajectory_index,
            "trajectory_uuid_list": self._trajectory_uuid_list,
        }
        with open(self._get_trajectory_index_path(save_path), "w") as f:
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
        trajectory = Trajectory(
            max_episode_length=trajectory_info["max_episode_length"]
        )
        for field_name, value in trajectory_dict.items():
            setattr(trajectory, field_name, value)

        # Put into cache for future reuse
        if self._trajectory_cache is not None:
            self._trajectory_cache.put(trajectory_uuid, trajectory)

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
            elif trajectory.rewards is not None:
                T, B = trajectory.rewards.shape[:2]
                num_samples = T * B
                trajectory_shape = trajectory.rewards.shape
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
        num_chunks: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Sample chunks (transitions) from the buffer.

        Args:
            num_chunks: Minimum number of chunks (transitions) to return

        Returns:
            Dictionary with rollout batch format [B, ...]
        """
        assert num_chunks > 0
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

        # Sample from the most recent trajectories (windowed)
        window_size = max(0, int(self.sample_window_size))
        if window_size > 0:
            window_uuids = self._trajectory_uuid_list[-window_size:]
        else:
            window_uuids = self._trajectory_uuid_list

        if not window_uuids:
            return {}

        window_total_samples = sum(
            self._trajectory_index[uuid]["num_samples"] for uuid in window_uuids
        )
        if window_total_samples == 0:
            return {}

        if num_chunks > window_total_samples:
            num_chunks = window_total_samples

        # Sample chunk indices directly from total samples
        sample_ids = torch.randint(
            low=0,
            high=window_total_samples,
            size=(num_chunks,),
            generator=self.random_generator,
        )

        # Convert global sample indices to per-trajectory local indices
        chunks = []
        if not window_uuids:
            return {}

        # Build cumulative end offsets for fast lookup
        cumulative_ends = []

        running = 0
        for single_uuid in window_uuids:
            running += self._trajectory_index[single_uuid]["num_samples"]
            cumulative_ends.append(running)

        grouped_indices: dict[str, list[tuple[int, int]]] = {}
        for idx_in_batch, sample_id in enumerate(sample_ids):
            sample_id = int(sample_id)
            idx = bisect.bisect_right(cumulative_ends, sample_id)
            if idx >= len(window_uuids):
                continue
            trajectory_uuid = window_uuids[idx]
            start = 0 if idx == 0 else cumulative_ends[idx - 1]
            local_sample_idx = sample_id - start
            grouped_indices.setdefault(trajectory_uuid, []).append(
                (idx_in_batch, local_sample_idx)
            )

        # Load each trajectory once and extract multiple chunks
        ordered_chunks: list[dict | None] = [None] * len(sample_ids)
        for trajectory_uuid, local_indices in grouped_indices.items():
            # Try to get from cache first
            trajectory = None
            if self._trajectory_cache is not None:
                trajectory = self._trajectory_cache.get(trajectory_uuid)

            # If not in cache, load from disk
            if trajectory is None:
                trajectory = self._load_trajectory(trajectory_uuid)

            trajectory_info = self._trajectory_index[trajectory_uuid]
            _, B = trajectory_info["shape"][:2]

            for idx_in_batch, local_sample_idx in local_indices:
                t_idx = int(local_sample_idx // B)
                b_idx = int(local_sample_idx % B)

                chunk = self._extract_chunk_from_trajectory(trajectory, t_idx, b_idx)
                ordered_chunks[idx_in_batch] = chunk

        for chunk in ordered_chunks:
            if chunk:
                chunks.append(chunk)

        # Merge chunks into batch [B, ...]
        batch = self._merge_chunks_to_batch(chunks)
        return batch

    def _extract_chunk_from_trajectory(
        self, trajectory: Trajectory, t_idx: int, b_idx: int
    ) -> dict:
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
            if field in [
                "curr_obs",
                "next_obs",
                "forward_inputs",
                "max_episode_length",
            ]:
                continue
            tensor = getattr(trajectory, field)
            if tensor is not None:
                chunk[field] = tensor[t_idx, b_idx]

        # Extract curr_obs / next_obs (dict of tensors)
        if trajectory.curr_obs:
            chunk["curr_obs"] = {}
            for key, tensor in trajectory.curr_obs.items():
                if tensor is not None:
                    chunk["curr_obs"][key] = tensor[t_idx, b_idx]
        if trajectory.next_obs:
            chunk["next_obs"] = {}
            for key, tensor in trajectory.next_obs.items():
                if tensor is not None:
                    chunk["next_obs"][key] = tensor[t_idx, b_idx]

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
                            d[nested_key]
                            for d in nested_dicts
                            if nested_key in d
                            and isinstance(d[nested_key], torch.Tensor)
                        ]
                        if nested_tensors:
                            nested_batch[nested_key] = torch.stack(
                                nested_tensors, dim=0
                            )  # [B, ...]
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
        stats = {
            "num_trajectories": self.size,
            "total_samples": self._total_samples,
            "cache_size": len(self._trajectory_cache.cache)
            if self._trajectory_cache
            else 0,
        }
        return stats

    def save(self, save_path: str):
        """
        Save buffer state (metadata and indices) to save_path.
        """
        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Save metadata and trajectory index into the specified directory
        self._save_metadata(save_path)
        self._save_trajectory_index(save_path)

    def load(
        self,
        load_path: str,
        is_distributed: bool = False,
        load_rank: int = 0,
        load_split_num: int = 1,
    ):
        """
        Load buffer state from saved metadata.

        Args:
            load_path: Path to the directory containing metadata.json and trajectory_index.json
            is_distributed: If True, only load a portion of trajectories based on load_rank and load_split_num
            load_rank: Rank index (0-based) for partial loading. Only used when is_distributed=True
            load_split_num: Number of splits to divide trajectories into. Only used when is_distributed=True
        """
        metadata_path = os.path.join(load_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update instance attributes from metadata
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

        # Handle distributed loading
        if is_distributed:
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
            end_idx = (
                start_idx + trajectories_per_split + (1 if load_rank < remainder else 0)
            )

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
