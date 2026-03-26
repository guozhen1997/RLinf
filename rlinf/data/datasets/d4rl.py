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

"""D4RL offline RL dataset with actor-local asynchronous batch sampling."""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    import d4rl
except ImportError:  # pragma: no cover
    d4rl = None  # type: ignore[assignment]

try:
    import gym  # type: ignore
except ImportError:  # pragma: no cover
    gym = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    import gym as _gym

    GymEnv = _gym.Env
else:
    GymEnv = Any


class D4RLDataset(Dataset):
    """Single source of truth for D4RL offline loading and sampling.

    This class owns:
    - D4RL transition loading (from env or hdf5)
    - PyTorch Dataset interface (__len__/__getitem__)
    - DataLoader + DistributedSampler build
    - Async prefetch queue (next_batch/state_dict/load_state_dict)
    """

    def __init__(
        self,
        env: GymEnv,
        env_name: str,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ):
        if d4rl is None or gym is None:  # pragma: no cover
            missing = []
            if gym is None:
                missing.append("gym")
            if d4rl is None:
                missing.append("d4rl")
            raise ImportError(
                "D4RLDataset requires optional dependencies: "
                + ", ".join(missing)
                + ". Please install them to use D4RL offline datasets."
            )

        raw = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            raw["actions"] = np.clip(raw["actions"], -lim, lim)

        dones_float = self._compute_dones_float(
            raw["observations"], raw["next_observations"], raw["terminals"]
        )
        self.observations = raw["observations"].astype(np.float32)
        self.actions = raw["actions"].astype(np.float32)
        self.rewards = raw["rewards"].astype(np.float32)
        self.masks = 1.0 - raw["terminals"].astype(np.float32)
        self.dones_float = dones_float.astype(np.float32)
        self.next_observations = raw["next_observations"].astype(np.float32)
        self.size = len(self.observations)

        self._apply_reward_postprocess(env_name)
        self._init_runtime_fields()

    def _init_runtime_fields(self) -> None:
        self._torch_observations: torch.Tensor | None = None
        self._torch_actions: torch.Tensor | None = None
        self._torch_rewards: torch.Tensor | None = None
        self._torch_masks: torch.Tensor | None = None
        self._torch_next_observations: torch.Tensor | None = None

        self._dataloader: DataLoader | None = None
        self.prefetch_batches: int = 2
        self._queue: queue.Queue | None = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._iterator = None
        self._epoch = 0
        self._produced_in_epoch = 0
        self._consumed_in_epoch = 0
        self._lock = threading.Lock()

    def _ensure_torch_cache(self) -> None:
        if self._torch_observations is None:
            self._torch_observations = torch.from_numpy(self.observations).float()
            self._torch_actions = torch.from_numpy(self.actions).float()
            self._torch_rewards = torch.from_numpy(self.rewards).float()
            self._torch_masks = torch.from_numpy(self.masks).float()
            self._torch_next_observations = torch.from_numpy(
                self.next_observations
            ).float()

    def __len__(self) -> int:
        return int(self.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_torch_cache()
        assert self._torch_observations is not None
        assert self._torch_actions is not None
        assert self._torch_rewards is not None
        assert self._torch_masks is not None
        assert self._torch_next_observations is not None
        return {
            "observations": self._torch_observations[idx],
            "actions": self._torch_actions[idx],
            "rewards": self._torch_rewards[idx],
            "masks": self._torch_masks[idx],
            "next_observations": self._torch_next_observations[idx],
        }

    @staticmethod
    def _compute_dones_float(
        observations: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        dones_float = np.zeros_like(terminals)
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(observations[i + 1] - next_observations[i]) > 1e-6
                or terminals[i] == 1.0
            ):
                dones_float[i] = 1.0
            else:
                dones_float[i] = 0.0
        if len(dones_float) > 0:
            dones_float[-1] = 1.0
        return dones_float

    def _apply_reward_postprocess(self, env_name: str) -> None:
        if "antmaze" in env_name:
            self.rewards -= 1.0
        elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
            self._normalize_rewards_for_mujoco()

    @staticmethod
    def _default_dataset_path(env_name: str) -> str:
        if gym is None:  # pragma: no cover
            raise ImportError(
                "D4RLDataset.from_path requires 'gym' (or install the embodied env deps)."
            )
        try:
            canonical_env_id = gym.spec(env_name).id
        except Exception:
            canonical_env_id = env_name
        return str(Path.home() / ".d4rl" / "datasets" / f"{canonical_env_id}.hdf5")

    @staticmethod
    def infer_obs_action_dims_from_env(env_name: str) -> tuple[int, int]:
        """Infer flat obs/action dims from a gym env's spaces."""
        if gym is None:  # pragma: no cover
            raise RuntimeError(
                "Failed to infer D4RL obs/action dims: missing 'gym' dependency."
            )
        env = gym.make(env_name)
        try:
            obs_shape = getattr(env.observation_space, "shape", None)
            act_shape = getattr(env.action_space, "shape", None)
            if obs_shape is None or act_shape is None:
                raise RuntimeError(
                    f"Env {env_name!r} does not expose Box-like observation/action shape."
                )
            return int(np.prod(obs_shape)), int(np.prod(act_shape))
        finally:
            env.close()

    @classmethod
    def from_path(
        cls,
        dataset_path: str | os.PathLike[str] | None,
        env_name: str,
        *,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
    ) -> D4RLDataset:
        path = (
            str(dataset_path)
            if dataset_path is not None
            else cls._default_dataset_path(env_name)
        )

        if d4rl is None or gym is None:  # pragma: no cover
            missing = []
            if gym is None:
                missing.append("gym")
            if d4rl is None:
                missing.append("d4rl")
            raise ImportError(
                "D4RLDataset.from_path requires optional dependencies: "
                + ", ".join(missing)
                + ". Please install them to load D4RL datasets."
            )

        try:
            import h5py  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading D4RL dataset from path requires 'h5py'. "
                "Install it or use env-based construction."
            ) from exc

        if not os.path.exists(path):
            env = gym.make(env_name)
            try:
                return cls(env=env, env_name=env_name, clip_to_eps=clip_to_eps, eps=eps)
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        with h5py.File(path, "r") as f:
            observations = np.asarray(f["observations"], dtype=np.float32)
            actions = np.asarray(f["actions"], dtype=np.float32)
            rewards = np.asarray(f["rewards"], dtype=np.float32)
            terminals = np.asarray(f["terminals"], dtype=np.float32)
            next_observations = np.asarray(f["next_observations"], dtype=np.float32)

        if clip_to_eps:
            lim = 1 - eps
            actions = np.clip(actions, -lim, lim)

        ds = cls.from_arrays(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=1.0 - terminals,
            dones_float=cls._compute_dones_float(
                observations, next_observations, terminals
            ),
            next_observations=next_observations,
        )
        ds._apply_reward_postprocess(env_name)
        return ds

    @classmethod
    def from_arrays(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ) -> D4RLDataset:
        self = cls.__new__(cls)
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = len(observations)
        self._init_runtime_fields()
        return self

    @classmethod
    def build_offline_actor_batch_provider(
        cls,
        cfg: Any,
        per_rank_batch_size: int,
    ) -> D4RLDataset:
        dataset_cfg = cfg.get("dataset", {})
        dataset_type = str(
            dataset_cfg.get("dataset_type", cfg.env.get("dataset_type", "d4rl"))
        ).lower()
        if dataset_type != "d4rl":
            raise NotImplementedError(
                f"Offline IQL currently only supports dataset_type='d4rl', got {dataset_type!r}."
            )

        dataset_env_name = dataset_cfg.get(
            "env_name",
            cfg.env.eval.get("env_name", cfg.env.get("env_name", None)),
        )
        if not dataset_env_name:
            raise ValueError("Offline dataset requires dataset.env_name.")
        dataset_path = dataset_cfg.get(
            "dataset_path", cfg.env.get("dataset_path", None)
        )

        dataset_init_kwargs_cfg = dataset_cfg.get("dataset_init_kwargs", {})
        if OmegaConf.is_config(dataset_init_kwargs_cfg):
            dataset_init_kwargs = OmegaConf.to_container(
                dataset_init_kwargs_cfg, resolve=True
            )
        else:
            dataset_init_kwargs = dict(dataset_init_kwargs_cfg)
        dataset_init_kwargs.setdefault("dataset_path", dataset_path)
        dataset_init_kwargs.setdefault("env_name", str(dataset_env_name))

        dataset = cls.from_path(**dataset_init_kwargs)
        dataset._init_dataloader_and_prefetch(
            per_rank_batch_size=int(per_rank_batch_size),
            dataset_cfg=dataset_cfg,
        )
        return dataset

    def _init_dataloader_and_prefetch(
        self,
        per_rank_batch_size: int,
        dataset_cfg: Any,
    ) -> None:
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                self,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=bool(dataset_cfg.get("shuffle", True)),
                seed=int(dataset_cfg.get("seed", 42)),
                drop_last=True,
            )
        else:
            sampler = None

        self._dataloader = DataLoader(
            self,
            batch_size=int(per_rank_batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            drop_last=True,
            pin_memory=bool(dataset_cfg.get("pin_memory", True)),
            persistent_workers=(
                int(dataset_cfg.get("num_workers", 0)) > 0
                and bool(dataset_cfg.get("persistent_workers", True))
            ),
        )
        if len(self) < int(per_rank_batch_size):
            raise ValueError(
                "Dataset size "
                f"({len(self)}) must be >= batch_size ({int(per_rank_batch_size)})."
            )

        self.prefetch_batches = int(dataset_cfg.get("prefetch_batches", 2))
        self._queue = queue.Queue(maxsize=max(1, self.prefetch_batches))
        self._stop_event = threading.Event()
        self._iterator = None
        self._epoch = 0
        self._produced_in_epoch = 0
        self._consumed_in_epoch = 0
        self._start_prefetch_thread()

    def _set_sampler_epoch(self, epoch: int) -> None:
        assert self._dataloader is not None
        sampler = getattr(self._dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(epoch))

    def _build_iterator(self, epoch: int) -> None:
        assert self._dataloader is not None
        self._set_sampler_epoch(epoch)
        self._iterator = iter(self._dataloader)
        self._produced_in_epoch = 0

    def _next_with_metadata(self) -> tuple[int, int, dict[str, torch.Tensor]]:
        if self._iterator is None:
            self._build_iterator(self._epoch)
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._build_iterator(self._epoch)
            batch = next(self._iterator)
        self._produced_in_epoch += 1
        return self._epoch, self._produced_in_epoch, batch

    def _prefetch_loop(self) -> None:
        assert self._queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                payload = self._next_with_metadata()
                self._queue.put(payload, timeout=0.1)
            except queue.Full:
                continue
            except Exception as exc:
                self._queue.put(exc)
                return

    def _start_prefetch_thread(self) -> None:
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    def _stop_prefetch_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self._queue is None:
            raise RuntimeError("D4RLDataset provider is not initialized.")
        item = self._queue.get()
        if isinstance(item, Exception):
            raise RuntimeError("Async batch prefetch failed.") from item
        epoch, consumed_in_epoch, batch = item
        with self._lock:
            self._epoch = int(epoch)
            self._consumed_in_epoch = int(consumed_in_epoch)
        return batch

    def state_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "data_epoch": int(self._epoch),
                "data_iter_offset": int(self._consumed_in_epoch),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._dataloader is None:
            raise RuntimeError("D4RLDataset provider is not initialized.")

        target_epoch = int(state.get("data_epoch", 0))
        target_offset = int(state.get("data_iter_offset", 0))
        self._stop_prefetch_thread()

        self._queue = queue.Queue(maxsize=max(1, int(self.prefetch_batches)))
        self._stop_event = threading.Event()
        self._epoch = max(0, target_epoch)
        self._consumed_in_epoch = 0
        self._build_iterator(self._epoch)
        for _ in range(max(0, target_offset)):
            try:
                next(self._iterator)
                self._produced_in_epoch += 1
            except StopIteration:
                self._epoch += 1
                self._build_iterator(self._epoch)
                next(self._iterator)
                self._produced_in_epoch += 1
        self._consumed_in_epoch = max(0, target_offset)
        self._start_prefetch_thread()

    def get_obs_action_dims(self) -> tuple[int, int]:
        return int(self.observations.shape[-1]), int(self.actions.shape[-1])

    def get_dataset_size(self) -> int:
        return int(self.size)

    def _normalize_rewards_for_mujoco(self) -> None:
        trajs = self._split_into_trajectories(
            self.observations,
            self.actions,
            self.rewards,
            self.masks,
            self.dones_float,
            self.next_observations,
        )

        def compute_returns(traj: list[tuple[np.ndarray, ...]]) -> float:
            return float(sum(rew for _, _, rew, _, _, _ in traj))

        trajs.sort(key=compute_returns)
        denom = compute_returns(trajs[-1]) - compute_returns(trajs[0])
        if abs(denom) < 1e-12:
            return
        self.rewards /= denom
        self.rewards *= 1000.0

    @staticmethod
    def _split_into_trajectories(
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
    ) -> list[list[tuple[np.ndarray, ...]]]:
        trajs: list[list[tuple[np.ndarray, ...]]] = [[]]
        for i in tqdm.tqdm(range(len(observations))):
            trajs[-1].append(
                (
                    observations[i],
                    actions[i],
                    rewards[i],
                    masks[i],
                    dones_float[i],
                    next_observations[i],
                )
            )
            if dones_float[i] == 1.0 and i + 1 < len(observations):
                trajs.append([])
        return trajs
