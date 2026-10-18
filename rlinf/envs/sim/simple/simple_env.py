# Copyright 2026 The RLinf Authors.
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

"""SIMPLE Teleop runtime and Ray proxy for Psi0."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import ray
import torch
from filelock import FileLock
from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rlinf.envs.sim.simple.utils import (
    SimpleController,
    SimpleTeleopController,
    extract_psi0_state,
)

_SUPPORTED_TASK_IDS = frozenset(
    {
        "simple/G1WholebodyXMovePickTeleop-v0",
        "simple/G1WholebodyHandoverTeleop-v0",
        "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
        "simple/G1WholebodyXMoveBendPickTeleop-v0",
        "simple/G1WholebodyCloseDoorTeleop-v0",
        "simple/G1WholebodyOpenOvenTeleop-v0",
        "simple/G1WholebodyOpenFaucetTeleop-v0",
        "simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0",
    }
)


def _install_data_path_lock() -> None:
    """Guard the upstream resolver before importing SIMPLE runtime modules.

    Hugging Face locks only the download. SIMPLE subsequently extracts and
    deletes the archive, so callers must share a lock for the whole operation.
    Even an existing target must be checked under the lock: another worker may
    have created that directory without finishing extraction yet.

    This changes only the SIMPLE resolver in the current worker process. All
    RLinf SIMPLE workers sharing its data directory use the same lock file.
    """
    import simple.utils as simple_utils

    original = simple_utils.resolve_data_path
    if getattr(original, "_rlinf_data_path_lock", False):
        return

    lock_path = str(
        Path(simple_utils.get_data_dir()) / ".cache" / "rlinf" / "resolve-data.lock"
    )

    def keep_log(record: logging.LogRecord) -> bool:
        # Suppress only this lock's DEBUG chatter, not other locks or warnings.
        return not (
            record.levelno == logging.DEBUG
            and isinstance(record.args, tuple)
            and lock_path in record.args
        )

    logging.getLogger("filelock").addFilter(keep_log)

    @wraps(original)
    def resolve_data_path(
        rel_path: str | None = None,
        create_if_not_exist: bool = False,
        auto_download: bool = False,
    ) -> str:
        if not rel_path:
            return original(rel_path, create_if_not_exist, auto_download)

        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        # Separate from HF's download lock to avoid recursively acquiring it.
        # Never unlink this file: waiters must keep locking the same inode.
        with FileLock(lock_path).acquire(poll_interval=0.5):
            resolved = original(rel_path, create_if_not_exist, auto_download)
            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"SIMPLE asset preparation did not create {resolved}."
                )
            return resolved

    resolve_data_path._rlinf_data_path_lock = True
    simple_utils.resolve_data_path = resolve_data_path


def _install_articulation_reuse_guard(native_env: Any) -> None:
    """Keep SIMPLE from rebuilding an unchanged Isaac articulation on reset."""
    isaac = native_env.isaac
    original_add = isaac.add_articulated_object
    asset_keys: dict[str, tuple[str, str]] = {}

    def add_articulated_object_once(obj_name: str, obj_info: Any) -> None:
        asset_name = str(obj_info.asset.name)
        asset_key = (str(obj_info.asset.uid), str(obj_info.asset.usd_path))
        if asset_name in isaac.articulated_objects:
            if asset_keys.get(asset_name) != asset_key:
                raise RuntimeError(
                    f"Cannot replace SIMPLE articulated object {asset_name!r} "
                    "without rebuilding its Isaac physics views."
                )
            return
        original_add(obj_name, obj_info)
        asset_keys[asset_name] = asset_key

    isaac.add_articulated_object = add_articulated_object_once


def _resolve_hssd_scene_dir(state: dict[str, Any]) -> Path | None:
    """Resolve the HSSD scene directory used by relative USD references."""
    scene = state.get("dr_state_dict", {}).get("scene", {})
    if not str(scene.get("uid", "")).startswith("hssd:"):
        return None

    data_dir = scene.get("data_dir")
    if not isinstance(data_dir, str):
        raise ValueError("SIMPLE HSSD reset state is missing scene.data_dir.")

    from simple.utils import resolve_data_path

    scene_dir = Path(resolve_data_path(data_dir, auto_download=True))
    props_dir = scene_dir / "props"
    if not props_dir.is_dir():
        raise FileNotFoundError(
            f"SIMPLE HSSD material assets are missing under {props_dir}."
        )
    return scene_dir


def _anchor_hssd_material_references(scene_dir: Path) -> Any:
    """Anchor HSSD material references to their containing scene layer."""
    from pxr import Sdf

    scene_path = scene_dir / f"{scene_dir.name}.usd"
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if layer is None:
        raise FileNotFoundError(f"Cannot open SIMPLE HSSD scene layer {scene_path}.")
    props_dir = scene_dir / "props"
    material_files = {
        path.relative_to(props_dir).as_posix().casefold(): path.relative_to(
            props_dir
        ).as_posix()
        for path in props_dir.rglob("*")
        if path.is_file()
    }
    material_refs = [
        asset_path
        for asset_path in layer.GetCompositionAssetDependencies()
        if "/props/" in asset_path
    ]
    if not material_refs:
        raise ValueError(f"SIMPLE HSSD scene has no material references: {scene_path}.")
    for asset_path in material_refs:
        requested_path = asset_path.rsplit("/props/", 1)[1]
        actual_path = material_files.get(requested_path.casefold())
        if actual_path is None:
            raise FileNotFoundError(
                f"SIMPLE HSSD material {requested_path} is missing under {props_dir}."
            )
        anchored_path = f"./props/{actual_path}"
        if not layer.UpdateCompositionAssetDependency(asset_path, anchored_path):
            raise RuntimeError(
                f"Cannot update SIMPLE HSSD material reference {asset_path}."
            )
    return layer


class SimpleLeRobotResetDataset:
    """Load fixed SIMPLE reset states from LeRobot episode metadata."""

    def __init__(
        self,
        path: str | Path,
        *,
        dr_level: int,
        episode_start: int,
        num_episodes: int,
    ):
        dataset_path = Path(path).expanduser()
        if dataset_path.suffix == ".zip":
            raise ValueError("Extract the SIMPLE reset-state zip before loading it.")
        if episode_start < 0 or num_episodes <= 0:
            raise ValueError("episode_start must be >= 0 and num_episodes must be > 0.")

        level_name = f"level-{dr_level}"
        metadata_files = [
            path
            for path in dataset_path.rglob("meta/episodes.jsonl")
            if path.parent.parent.name == level_name
        ]
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected one {level_name}/meta/episodes.jsonl under "
                f"{dataset_path}, found {len(metadata_files)}."
            )
        with metadata_files[0].open() as file:
            episodes = [json.loads(line) for line in file if line.strip()]
        episodes.sort(key=lambda episode: int(episode["episode_index"]))
        self._episodes = episodes[episode_start : episode_start + num_episodes]
        if len(self._episodes) != num_episodes:
            raise ValueError(
                f"Requested {num_episodes} reset episodes from index {episode_start}, "
                f"but found {len(self._episodes)} in {metadata_files[0]}."
            )

    def __len__(self) -> int:
        return len(self._episodes)

    def load(self, index: int) -> tuple[dict[str, Any], Any]:
        episode = self._episodes[index]
        encoded_config = episode.get("environment_config")
        if not isinstance(encoded_config, str):
            raise ValueError(
                "SIMPLE LeRobot episode metadata is missing environment_config."
            )
        return json.loads(encoded_config), episode


@ray.remote(
    num_cpus=1, num_gpus=0, max_concurrency=1, max_restarts=0, max_task_retries=0
)
class SimpleProcess(gym.Env):
    """Own one simulator and controller on a synchronous Ray actor's main thread."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self, cfg: DictConfig, num_envs: int, seed_offset: int, total_num_processes: int
    ) -> None:
        if num_envs != 1:
            raise ValueError("SIMPLE requires one env per EnvWorker.")
        if total_num_processes < 1:
            raise ValueError("total_num_processes must be positive.")
        task_id = str(cfg.init_params.task_id)
        if task_id not in _SUPPORTED_TASK_IDS:
            raise ValueError(f"Unsupported Psi0 SIMPLE task: {task_id}.")
        if cfg.auto_reset:
            raise ValueError(
                "Psi0 SIMPLE requires auto_reset=false to preserve one episode "
                "per rollout epoch."
            )

        self.cfg = cfg
        self.num_envs = 1
        self.seed = int(cfg.seed) + int(seed_offset)
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.is_start = True
        self._elapsed_steps = 0
        self._return = 0.0
        self._last_observation: dict[str, Any] | None = None
        self._terminal_episode: dict[str, torch.Tensor] | None = None
        if cfg.reset_dataset.get("format", "lerobot") != "lerobot":
            raise ValueError("The SIMPLE reset dataset uses LeRobot format.")
        self._dataset = SimpleLeRobotResetDataset(
            cfg.reset_dataset.path,
            dr_level=int(cfg.reset_dataset.dr_level),
            episode_start=int(cfg.reset_dataset.episode_start),
            num_episodes=int(cfg.reset_dataset.num_episodes),
        )
        initial_dataset_index = int(seed_offset) % len(self._dataset)
        self._dataset_index = initial_dataset_index - 1
        initial_state, _ = self._dataset.load(initial_dataset_index)
        # Install before any SIMPLE asset modules capture the resolver by import.
        _install_data_path_lock()
        self._hssd_scene_dir = _resolve_hssd_scene_dir(initial_state)
        self._hssd_scene_layer = None
        self._raw_env, self._native_env, self._controller = self._create_runtime()
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(36,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})

    def ready(self) -> None:
        """Confirm initialization without advancing the reset dataset cursor."""

    @staticmethod
    def _make_sonic_config() -> dict[str, Any]:
        import tyro
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        config = tyro.cli(
            SimLoopConfig,
            config=(tyro.conf.ConsolidateSubcommandArgs,),
            args=[],
        )
        sonic_config = config.load_wbc_yaml()
        sonic_config["ENV_NAME"] = "simple"
        return sonic_config

    def _create_runtime(self) -> tuple[Any, Any, SimpleController]:
        """Initialize native libraries before Isaac and create one runtime."""
        import simple.envs  # noqa: F401

        sonic_config = self._make_sonic_config()
        sim_mode = str(self.cfg.init_params.sim_mode)
        original_argv = sys.argv
        try:
            if "isaac" in sim_mode:
                # SimulationApp reads Kit CLI options during gym.make(). Give
                # each process its own writable caches/config/logs; keep SIMPLE
                # source assets shared. Preserve logs after a crash for diagnosis.
                runtime_dir = tempfile.mkdtemp(prefix="rlinf-simple-isaac-")
                sys.argv = [*original_argv, "--portable-root", runtime_dir]
            raw_env = gym.make(
                str(self.cfg.init_params.task_id),
                sim_mode=sim_mode,
                headless=bool(self.cfg.init_params.headless),
                sonic_config=sonic_config,
                render_hz=int(self.cfg.init_params.render_hz),
                dr_level=int(self.cfg.reset_dataset.dr_level),
            )
        finally:
            sys.argv = original_argv
        native_env = raw_env.unwrapped
        if sim_mode == "mujoco_isaac":
            _install_articulation_reuse_guard(native_env)
            if not native_env.task.metadata.get("debug", False):
                # SIMPLE otherwise renders and discards the MuJoCo camera frames.
                native_env._render_frame = native_env.isaac.render
        task_horizon = int(native_env.task.metadata["max_episode_steps"])
        configured_horizon = int(self.cfg.max_episode_steps)
        if configured_horizon <= 0 or configured_horizon > task_horizon:
            raise ValueError(
                "SIMPLE Teleop horizon must not exceed the task metadata: "
                f"config={configured_horizon}, task={task_horizon}."
            )
        raw_env = TimeLimit(raw_env, max_episode_steps=configured_horizon)
        controller = SimpleTeleopController(native_env.task.robot, sonic_config)
        return raw_env, native_env, controller

    def _wrap_observation(
        self,
        observation: dict[str, Any],
        *,
        reset: bool,
    ) -> dict[str, Any]:
        image = np.asarray(observation["head_stereo_left"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                "SIMPLE head_stereo_left must be an RGB/HWC uint8 image, "
                f"got {image.shape}/{image.dtype}."
            )
        state = extract_psi0_state(
            observation["joint_qpos"], self._controller.last_base_height
        )
        instruction = str(self._native_env.task.instruction)
        return {
            "main_images": torch.from_numpy(np.ascontiguousarray(image))[None],
            "wrist_images": None,
            "extra_view_images": None,
            "states": torch.from_numpy(state)[None, None],
            "task_descriptions": [instruction],
            "reset_mask": torch.tensor([reset], dtype=torch.bool),
        }

    def _next_dataset_index(self) -> int:
        self._dataset_index = (self._dataset_index + 1) % len(self._dataset)
        return self._dataset_index

    def update_reset_state_ids(self) -> None:
        """Keep reset-state ownership local to the SIMPLE dataset cursor."""

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed = int(seed)
        state, _ = self._dataset.load(self._next_dataset_index())
        scene_dir = _resolve_hssd_scene_dir(state)
        if scene_dir != self._hssd_scene_dir:
            raise ValueError(
                "A SIMPLE worker cannot switch HSSD scenes after USD startup: "
                f"initial={self._hssd_scene_dir}, requested={scene_dir}."
            )
        if scene_dir is not None and self._hssd_scene_layer is None:
            self._hssd_scene_layer = _anchor_hssd_material_references(scene_dir)
        observation, info = self._raw_env.reset(options={"state_dict": state})
        self._controller.reset()

        control_dt = self._controller.control_dt
        for _ in range(int(self.cfg.init_params.stabilization_steps)):
            if self._controller.is_stabilized():
                break
            step_start = time.monotonic()
            low_level_action = self._controller.stabilize(observation)
            observation, *_, info = self._native_env.step(low_level_action)
            self._native_env.update_viewer()
            self._native_env.update_reward()
            remaining = control_dt - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)
        self._controller.finish_stabilization()

        self._elapsed_steps = 0
        self._return = 0.0
        self._terminal_episode = None
        self.is_start = False
        self._last_observation = observation
        return self._wrap_observation(observation, reset=True), info

    def chunk_step(self, chunk_actions):
        """Synchronously execute a valid prefix of one 24-step Psi0 response."""
        actions = np.asarray(chunk_actions, dtype=np.float32)
        if actions.shape != (1, 24, 36):
            raise ValueError(
                f"SIMPLE chunk actions must have shape (1, 24, 36), got {actions.shape}."
            )
        if self._last_observation is None:
            raise RuntimeError("SIMPLE environment must be reset before chunk_step().")

        if self._terminal_episode is not None:
            terminal_observation = self._wrap_observation(
                self._last_observation, reset=False
            )
            infos_list = [{} for _ in range(24)]
            infos_list[-1] = {
                "episode": self._terminal_episode,
                "executed_mask": torch.zeros((1, 24), dtype=torch.bool),
            }
            return (
                [terminal_observation] * 24,
                torch.zeros((1, 24), dtype=torch.float32),
                torch.zeros((1, 24), dtype=torch.bool),
                torch.zeros((1, 24), dtype=torch.bool),
                infos_list,
            )

        obs_list = []
        infos_list = []
        rewards = torch.zeros((1, 24), dtype=torch.float32)
        raw_terminations = torch.zeros((1, 24), dtype=torch.bool)
        raw_truncations = torch.zeros((1, 24), dtype=torch.bool)
        executed_mask = torch.zeros((1, 24), dtype=torch.bool)
        terminal_observation = None
        terminal_info = None

        observation = self._last_observation
        for index, high_level_action in enumerate(actions[0]):
            low_level_action = self._controller.action(observation, high_level_action)
            observation, reward, terminated, truncated, info = self._raw_env.step(
                low_level_action
            )
            self._elapsed_steps += 1
            self._return += float(reward)
            executed_mask[0, index] = True
            rewards[0, index] = float(reward)
            raw_terminations[0, index] = bool(terminated)
            raw_truncations[0, index] = bool(truncated)
            obs_list.append(self._wrap_observation(observation, reset=False))
            infos_list.append(dict(info))
            if terminated or truncated:
                terminal_observation = obs_list[-1]
                terminal_info = dict(info)
                break

        done = terminal_observation is not None
        if done:
            success = bool(self._native_env._success)
            episode = {
                "success": torch.tensor([success]),
                "return": torch.tensor([self._return], dtype=torch.float32),
                "episode_len": torch.tensor([self._elapsed_steps]),
            }
            self._terminal_episode = episode
            terminal_info["episode"] = episode
            terminal_info["executed_mask"] = executed_mask
            while len(obs_list) < 24:
                obs_list.append(terminal_observation)
                infos_list.append({})
            infos_list[-1] = terminal_info
        else:
            infos_list[-1]["executed_mask"] = executed_mask

        self._last_observation = observation
        terminations = torch.zeros_like(raw_terminations)
        truncations = torch.zeros_like(raw_truncations)
        terminations[:, -1] = raw_terminations.any(dim=1)
        truncations[:, -1] = raw_truncations.any(dim=1)
        return obs_list, rewards, terminations, truncations, infos_list

    def capture_image(self) -> np.ndarray | None:
        if self._last_observation is None:
            return None
        return np.asarray(self._last_observation["head_stereo_left"])[None]

    def close(self):
        self._raw_env.close()
        super().close()


class SimpleEnv(gym.Env):
    """Proxy one SIMPLE runtime while keeping native imports out of EnvWorker."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        del worker_info
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed = int(cfg.seed) + int(seed_offset)
        self.auto_reset = bool(cfg.auto_reset)
        self.ignore_terminations = bool(cfg.ignore_terminations)
        self.is_start = True
        self._process = None
        self._image: np.ndarray | None = None
        self.action_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(36,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({})

        # RLinf assigns GPU visibility independently of Ray GPU resources.
        env_vars = {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "MUJOCO_EGL_DEVICE_ID",
                "EGL_DEVICE_ID",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
            )
            if key in os.environ
        }
        env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        self._process = SimpleProcess.options(
            # Native children follow their owner, outside Cluster's named-worker cleanup.
            namespace=f"{ray.get_runtime_context().namespace}.simple",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(), soft=False
            ),
            # Ray inherits the parent's Python/runtime environment and merges env_vars.
            runtime_env={"env_vars": env_vars},
        ).remote(cfg, num_envs, seed_offset, total_num_processes)
        try:
            ray.get(self._process.ready.remote())
        except BaseException:
            # The constructor may have only partially initialized native libraries.
            self._kill_process()
            raise

    def _kill_process(self) -> None:
        process, self._process = self._process, None
        if process is not None and ray.is_initialized():
            try:
                ray.kill(process, no_restart=True)
            except ray.exceptions.RayActorError:
                pass

    def _require_process(self) -> Any:
        if self._process is None:
            raise RuntimeError("SIMPLE environment is closed.")
        return self._process

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset once in the simulator process and cache its initial image."""
        observation, info = ray.get(
            self._require_process().reset.remote(seed=seed, options=options)
        )
        if seed is not None:
            self.seed = int(seed)
        self.is_start = False
        self._image = observation["main_images"].numpy()
        return observation, info

    def chunk_step(
        self, chunk_actions: np.ndarray | torch.Tensor
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        """Execute a whole chunk in one RPC, including any terminal padding."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        result = ray.get(self._require_process().chunk_step.remote(chunk_actions))
        observations, _, _, _, infos = result
        executed_mask = infos[-1].get("executed_mask")
        if (
            not isinstance(executed_mask, torch.Tensor)
            or executed_mask.dtype != torch.bool
            or executed_mask.shape != (1, 24)
        ):
            raise RuntimeError(
                "SIMPLE must return a bool executed_mask of shape [1, 24]."
            )
        self._image = observations[-1]["main_images"].numpy()
        return result

    def capture_image(self) -> np.ndarray | None:
        """Return the latest frame without an additional simulator RPC."""
        return self._image

    def update_reset_state_ids(self) -> None:
        """Leave reset cursor advancement to the process's next reset call."""

    def close(self) -> None:
        """Release the child once; Isaac requires process termination for teardown."""
        if self._process is None:
            return
        try:
            if ray.is_initialized() and "isaac" not in str(
                self.cfg.init_params.sim_mode
            ):
                try:
                    ray.get(self._process.close.remote())
                except ray.exceptions.RayActorError:
                    pass
        finally:
            self._kill_process()
            self._image = None
