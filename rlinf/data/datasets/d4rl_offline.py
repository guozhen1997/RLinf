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
import time
from typing import Callable

import gym
import numpy as np
import tqdm

try:
    import d4rl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "d4rl is required for D4RL offline datasets. Please install d4rl."
    ) from exc


def _compat_reset(env: gym.Env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _compat_step(env: gym.Env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        if truncated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info
    return out


class EpisodeMonitor(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, done, info = _compat_step(self.env, action)
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}
        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self, "get_normalized_score"):
                info["episode"]["return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )
        return observation, reward, done, info

    def reset(self):
        self._reset_stats()
        return _compat_reset(self.env)


class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if isinstance(self.observation_space, gym.spaces.Box):
            obs_space = self.observation_space
            self.observation_space = gym.spaces.Box(
                obs_space.low, obs_space.high, obs_space.shape
            )
        elif isinstance(self.observation_space, gym.spaces.Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = gym.spaces.Box(v.low, v.high, v.shape)
            self.observation_space = gym.spaces.Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation):
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        if isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
        return observation


def split_into_trajectories(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    dones_float: np.ndarray,
    next_observations: np.ndarray,
):
    trajs = [[]]
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


class Dataset:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
        dones_float: np.ndarray,
        next_observations: np.ndarray,
        size: int,
    ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(self.size, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "masks": self.masks[indices],
            "next_observations": self.next_observations[indices],
        }


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)
        if clip_to_eps:
            lim = 1 - eps
            dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
        dones_float = np.zeros_like(dataset["rewards"])
        for i in range(len(dones_float) - 1):
            if (
                np.linalg.norm(
                    dataset["observations"][i + 1] - dataset["next_observations"][i]
                )
                > 1e-6
                or dataset["terminals"][i] == 1.0
            ):
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1
        super().__init__(
            dataset["observations"].astype(np.float32),
            actions=dataset["actions"].astype(np.float32),
            rewards=dataset["rewards"].astype(np.float32),
            masks=1.0 - dataset["terminals"].astype(np.float32),
            dones_float=dones_float.astype(np.float32),
            next_observations=dataset["next_observations"].astype(np.float32),
            size=len(dataset["observations"]),
        )


def normalize(dataset: D4RLDataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations,
    )

    def compute_returns(traj):
        return sum(rew for _, _, rew, _, _, _ in traj)

    trajs.sort(key=compute_returns)
    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


class PartitionedDataset:
    """Wrapper that exposes only a rank's partition for data-parallel sampling."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        n = dataset.size
        per_rank = n // world_size
        start = rank * per_rank
        end = start + per_rank if rank < world_size - 1 else n
        self._indices = np.arange(start, end, dtype=np.int64)
        self._parent = dataset
        self.size = len(self._indices)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        indices = np.random.randint(self.size, size=batch_size)
        real_indices = self._indices[indices]
        return {
            "observations": self._parent.observations[real_indices],
            "actions": self._parent.actions[real_indices],
            "rewards": self._parent.rewards[real_indices],
            "masks": self._parent.masks[real_indices],
            "next_observations": self._parent.next_observations[real_indices],
        }

    @property
    def observations(self) -> np.ndarray:
        return self._parent.observations[self._indices]

    @property
    def actions(self) -> np.ndarray:
        return self._parent.actions[self._indices]

    @property
    def rewards(self) -> np.ndarray:
        return self._parent.rewards[self._indices]

    @property
    def masks(self) -> np.ndarray:
        return self._parent.masks[self._indices]

    @property
    def next_observations(self) -> np.ndarray:
        return self._parent.next_observations[self._indices]


def partition_dataset_for_rank(
    dataset: Dataset, rank: int, world_size: int
) -> PartitionedDataset:
    """Create a partition of the dataset for data-parallel training.

    Each rank gets a non-overlapping subset so that together they cover the
    full dataset without redundancy during sampling.
    """
    return PartitionedDataset(dataset, rank, world_size)


def create_partition_dataset(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    """Create a Dataset containing only the rank's partition.

    This is useful for data-parallel training where each rank should hold only a
    shard of the offline dataset to reduce memory duplication.
    """
    n = dataset.size
    per_rank = n // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else n
    indices = np.arange(start, end, dtype=np.int64)
    return Dataset(
        observations=dataset.observations[indices].copy(),
        actions=dataset.actions[indices].copy(),
        rewards=dataset.rewards[indices].copy(),
        masks=dataset.masks[indices].copy(),
        dones_float=dataset.dones_float[indices].copy(),
        next_observations=dataset.next_observations[indices].copy(),
        size=len(indices),
    )


def make_d4rl_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env = EpisodeMonitor(env)
    env = SinglePrecision(env)
    if hasattr(env, "seed"):
        env.seed(seed)
    else:
        env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_d4rl_env_and_dataset(env_name: str, seed: int) -> tuple[gym.Env, D4RLDataset]:
    env = make_d4rl_env(env_name, seed)

    dataset = D4RLDataset(env)
    if "antmaze" in env_name:
        dataset.rewards -= 1.0
    elif any(name in env_name for name in ("halfcheetah", "walker2d", "hopper")):
        normalize(dataset)
    return env, dataset


def evaluate_policy(
    sample_fn: Callable[[np.ndarray, float], np.ndarray],
    env: gym.Env,
    num_episodes: int,
) -> dict[str, float]:
    """Evaluate a policy given a sampling function (obs, temperature) -> action."""
    stats = {"return": [], "length": []}
    for _ in range(num_episodes):
        observation, done = _compat_reset(env), False
        while not done:
            action = sample_fn(observation, 0.0)
            observation, _, done, info = _compat_step(env, action)
        for key in stats:
            stats[key].append(info["episode"][key])
    return {k: float(np.mean(v)) for k, v in stats.items()}
