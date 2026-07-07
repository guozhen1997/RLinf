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

from typing import Optional, OrderedDict, Union

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf
from rlinf.envs.maniskill.peg_insertion_side_variants import (
    RLT_OPENPI_JOINT_WRAP_MODE,
    init_peg_insertion_event_state,
    is_peg_insertion_side_env_id,
    maybe_augment_peg_insertion_info,
    patch_rlt_openpi_joint_env_args,
    reset_peg_insertion_event_state,
    resolve_maniskill_task_descriptions,
    restore_peg_insertion_event_state,
    snapshot_peg_insertion_event_state,
    wrap_rlt_openpi_joint_obs,
)
from rlinf.envs.maniskill.rlt_policy_switch import (
    apply_rlt_policy_switch_info,
    attach_rlt_policy_switch_obs,
    build_maniskill_rlt_policy_switch_controller,
)

__all__ = ["ManiskillEnv"]

def _to_bool_tensor(value, *, num_envs, device):
    if isinstance(value, bool):
        return torch.full((num_envs,), value, dtype=torch.bool, device=device)
    if isinstance(value, torch.Tensor):
        value = value.to(device=device, dtype=torch.bool)
    else:
        value = torch.as_tensor(value, device=device, dtype=torch.bool)
    if value.ndim == 0:
        value = value.reshape(1).repeat(num_envs)
    return value


def extract_termination_from_info(info, num_envs, device, fallback=None):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(
                _to_bool_tensor(info["success"], num_envs=num_envs, device=device),
                _to_bool_tensor(info["fail"], num_envs=num_envs, device=device),
            )
        else:
            terminated = _to_bool_tensor(
                info["success"], num_envs=num_envs, device=device
            )
    else:
        if "fail" in info:
            terminated = _to_bool_tensor(info["fail"], num_envs=num_envs, device=device)
        else:
            if fallback is None:
                return torch.zeros(num_envs, dtype=bool, device=device)
            terminated = _to_bool_tensor(fallback, num_envs=num_envs, device=device)
    return terminated


def _shape_str(value):
    return "None" if value is None else str(tuple(getattr(value, "shape", ())))


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self._has_seeded_reset = False
        self.task_id = getattr(cfg.init_params, "id", None)
        self._is_peg_insertion_side = is_peg_insertion_side_env_id(self.task_id)
        self.rlt_policy_switch_cfg = getattr(cfg, "rlt_policy_switch", None)
        self.rlt_policy_switch_controller = None

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        env_args = patch_rlt_openpi_joint_env_args(
            env_args,
            wrap_obs_mode=getattr(cfg, "wrap_obs_mode", "default"),
        )
        self.env: BaseEnv = gym.make(**env_args)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        if self._is_peg_insertion_side:
            self.peg_event_state = init_peg_insertion_event_state(
                num_envs=self.num_envs,
                device=self.device,
            )
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()
        self._init_persistent_done_state()
        self._init_rlt_policy_switch_controller()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        return resolve_maniskill_task_descriptions(
            self.env.unwrapped,
            num_envs=self.num_envs,
            is_peg_insertion_side=self._is_peg_insertion_side,
        )

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return infos["extracted_obs"]

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return {"states": raw_obs}
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    state = common.flatten_state_dict(
                        raw_obs, use_torch=True, device=self.device
                    )

                main_images = sensor_data["base_camera"]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera")
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "states": state,
                }

        if wrap_obs_mode == RLT_OPENPI_JOINT_WRAP_MODE:
            if self.env.unwrapped.obs_mode != "rgb":
                raise ValueError(
                    "wrap_obs_mode='rlt_openpi_joint' requires ManiSkill obs_mode='rgb'."
                )
            obs = wrap_rlt_openpi_joint_obs(
                raw_obs,
                infos=infos,
                task_descriptions=self.instruction,
                num_envs=self.num_envs,
                device=self.device,
                is_peg_insertion_side=self._is_peg_insertion_side,
            )
            return attach_rlt_policy_switch_obs(
                controller=self.rlt_policy_switch_controller,
                obs=obs,
                device=self.device,
            )

        # Default
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        return {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _init_rlt_policy_switch_controller(self):
        self.rlt_policy_switch_controller = (
            build_maniskill_rlt_policy_switch_controller(
                switch_cfg=self.rlt_policy_switch_cfg,
                is_peg_insertion_side=self._is_peg_insertion_side,
                env=self.env.unwrapped,
                batch_size=self.num_envs,
            )
        )

    def _reset_rlt_policy_switch_controller(self, env_idx=None):
        if self.rlt_policy_switch_controller is not None:
            self.rlt_policy_switch_controller.reset(env_idx=env_idx)

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _init_persistent_done_state(self):
        self._persistent_done_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._persistent_done_obs = None
        self._persistent_done_infos = None

    def _reset_persistent_done_state(self, env_idx=None):
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
            return

        if env_idx is None:
            self._persistent_done_mask.zero_()
            self._persistent_done_obs = None
            self._persistent_done_infos = None
            return

        self._persistent_done_mask[env_idx] = False

    def _update_persistent_done_state(self, dones, extracted_obs, infos):
        if self.auto_reset or not dones.any():
            return

        newly_done = dones & (~self._persistent_done_mask)
        if not newly_done.any():
            return

        if self._persistent_done_obs is None:
            self._persistent_done_obs = torch_clone_dict(extracted_obs)
        else:
            self._persistent_done_obs = self._restore_frozen_values(
                self._persistent_done_obs, extracted_obs, newly_done
            )

        if self._persistent_done_infos is None:
            self._persistent_done_infos = torch_clone_dict(infos)
        else:
            self._persistent_done_infos = self._restore_frozen_values(
                self._persistent_done_infos, infos, newly_done
            )

        self._persistent_done_mask |= newly_done

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        for key in (
            "rlt_switch_flags",
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in infos:
                value = infos[key]
                if isinstance(value, torch.Tensor):
                    episode_info[key] = value.reshape(self.num_envs, -1)[:, -1].clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
            if seed is None:
                if self.use_fixed_reset_state_ids or not self._has_seeded_reset:
                    seed = self.seed
        if seed is not None:
            self._has_seeded_reset = True
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            if self._is_peg_insertion_side:
                reset_peg_insertion_event_state(
                    self.peg_event_state,
                    env_idx=env_idx,
                )
            self._reset_metrics(env_idx)
        else:
            if self._is_peg_insertion_side:
                reset_peg_insertion_event_state(self.peg_event_state)
            self._reset_metrics()
        self._reset_persistent_done_state(options.get("env_idx"))
        self._reset_rlt_policy_switch_controller(options.get("env_idx"))
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        return extracted_obs, infos

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        infos = maybe_augment_peg_insertion_info(
            env=self.env.unwrapped,
            infos=infos,
            event_state=getattr(self, "peg_event_state", None),
            device=self.device,
            is_peg_insertion_side=self._is_peg_insertion_side,
        )
        infos["elapsed_steps"] = self.elapsed_steps.clone()
        terminations = extract_termination_from_info(
            infos,
            num_envs=self.num_envs,
            device=self.device,
            fallback=terminations,
        )
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        step_reward = self._calc_step_reward(_reward, infos)

        if self.record_metrics:
            infos = self._record_metrics(step_reward, infos)
        self._attach_rlt_policy_switch_info(infos)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def _attach_rlt_policy_switch_info(self, infos):
        if self.rlt_policy_switch_controller is None:
            return
        policy_info = self.rlt_policy_switch_controller.export_info(device=self.device)
        infos["policy_info"] = policy_info
        for key, value in policy_info.items():
            infos[key] = value

    def _snapshot_episode_state(self):
        state = {
            "prev_step_reward": self.prev_step_reward.clone(),
        }
        if self.record_metrics:
            state.update(
                {
                    "success_once": self.success_once.clone(),
                    "fail_once": self.fail_once.clone(),
                    "returns": self.returns.clone(),
                }
            )
        if self._is_peg_insertion_side:
            state["peg_event_state"] = snapshot_peg_insertion_event_state(
                self.peg_event_state
            )
        return state

    def _restore_episode_state(self, state, mask):
        self.prev_step_reward[mask] = state["prev_step_reward"][mask]
        if self.record_metrics:
            self.success_once[mask] = state["success_once"][mask]
            self.fail_once[mask] = state["fail_once"][mask]
            self.returns[mask] = state["returns"][mask]
        if self._is_peg_insertion_side:
            restore_peg_insertion_event_state(
                self.peg_event_state,
                state["peg_event_state"],
                mask,
            )

    def _zero_frozen_actions(self, actions, mask):
        if not mask.any():
            return actions
        if isinstance(actions, torch.Tensor):
            frozen_mask = mask.to(device=actions.device)
            actions = actions.clone()
            actions[frozen_mask] = 0.0
            return actions

        frozen_mask = mask.detach().cpu().numpy()
        actions = np.asarray(actions).copy()
        actions[frozen_mask] = 0.0
        return actions

    def _restore_frozen_values(self, values, previous_values, mask):
        if not isinstance(values, dict) or not isinstance(previous_values, dict):
            return values

        restored = torch_clone_dict(values)
        for key, prev_value in previous_values.items():
            if key not in restored:
                continue
            value = restored[key]
            if isinstance(value, torch.Tensor) and isinstance(prev_value, torch.Tensor):
                if value.ndim > 0 and value.shape[0] == self.num_envs:
                    value_mask = mask.to(device=value.device)
                    value[value_mask] = prev_value.to(value.device)[value_mask]
            elif isinstance(value, dict) and isinstance(prev_value, dict):
                restored[key] = self._restore_frozen_values(
                    value, prev_value, mask
                )
            elif isinstance(value, list) and isinstance(prev_value, list):
                mask_cpu = mask.detach().cpu().numpy()
                for env_idx, should_restore in enumerate(mask_cpu):
                    if (
                        should_restore
                        and env_idx < len(value)
                        and env_idx < len(prev_value)
                    ):
                        value[env_idx] = prev_value[env_idx]
        return restored

    def _restore_frozen_info_values(self, infos, previous_infos, mask):
        return self._restore_frozen_values(infos, previous_infos, mask)

    def _validate_chunk_actions(self, chunk_actions) -> None:
        if not hasattr(chunk_actions, "shape") or len(chunk_actions.shape) != 3:
            raise ValueError(
                "ManiskillEnv.chunk_step expected action chunk shape "
                f"[num_envs, chunk_steps, action_dim], got {_shape_str(chunk_actions)}. "
                "Refuse to execute malformed actions."
            )

        if int(chunk_actions.shape[0]) != self.num_envs:
            raise ValueError(
                "ManiskillEnv.chunk_step action batch mismatch: expected "
                f"num_envs={self.num_envs}, got shape {_shape_str(chunk_actions)}. "
                "Refuse to execute actions for the wrong env batch."
            )

        expected_action_dim = getattr(self.env.unwrapped.single_action_space, "shape", None)
        expected_action_dim = expected_action_dim[-1] if expected_action_dim else None
        if expected_action_dim is not None and int(chunk_actions.shape[2]) != int(
            expected_action_dim
        ):
            raise ValueError(
                "ManiskillEnv.chunk_step action dim mismatch before env.step: "
                f"expected action_dim={expected_action_dim}, got shape "
                f"{_shape_str(chunk_actions)}. Check actor.model.action_dim and "
                "ManiSkill control_mode."
            )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        self._validate_chunk_actions(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
        frozen_dones = (
            self._persistent_done_mask.clone()
            if not self.auto_reset
            else torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        )
        initially_frozen_dones = frozen_dones.clone()
        last_extracted_obs = (
            torch_clone_dict(self._persistent_done_obs)
            if frozen_dones.any() and self._persistent_done_obs is not None
            else None
        )
        last_infos = (
            torch_clone_dict(self._persistent_done_infos)
            if frozen_dones.any() and self._persistent_done_infos is not None
            else None
        )
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            if (
                frozen_dones.all()
                and last_extracted_obs is not None
                and last_infos is not None
            ):
                extracted_obs = torch_clone_dict(last_extracted_obs)
                infos = torch_clone_dict(last_infos)
                step_reward = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float32
                )
                terminations = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
                truncations = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
            else:
                state_before_step = self._snapshot_episode_state()
                actions = self._zero_frozen_actions(actions, frozen_dones)
                extracted_obs, step_reward, terminations, truncations, infos = self.step(
                    actions, auto_reset=False
                )
                if frozen_dones.any():
                    self._restore_episode_state(state_before_step, frozen_dones)
                    if last_extracted_obs is not None:
                        extracted_obs = self._restore_frozen_values(
                            extracted_obs, last_extracted_obs, frozen_dones
                        )
                    step_reward = step_reward.clone()
                    step_reward[frozen_dones] = 0.0
                    terminations = terminations.clone()
                    truncations = truncations.clone()
                    terminations[frozen_dones] = False
                    truncations[frozen_dones] = False
                    if last_infos is not None:
                        infos = self._restore_frozen_info_values(
                            infos, last_infos, frozen_dones
                        )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)
            frozen_dones |= torch.logical_or(terminations, truncations)
            last_extracted_obs = torch_clone_dict(extracted_obs)
            last_infos = torch_clone_dict(infos)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        policy_switch_chunk_dones = torch.logical_or(
            raw_chunk_terminations,
            raw_chunk_truncations,
        )
        if initially_frozen_dones.any():
            policy_switch_chunk_dones = policy_switch_chunk_dones | (
                initially_frozen_dones[:, None].expand_as(policy_switch_chunk_dones)
            )
        apply_rlt_policy_switch_info(
            controller=self.rlt_policy_switch_controller,
            infos_list=infos_list,
            chunk_dones=policy_switch_chunk_dones,
        )
        self._sync_rlt_policy_switch_episode_info(infos_list[-1])
        obs_list[-1] = attach_rlt_policy_switch_obs(
            controller=self.rlt_policy_switch_controller,
            obs=obs_list[-1],
            device=self.device,
        )

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )
        elif past_dones.any():
            self._update_persistent_done_state(past_dones, obs_list[-1], infos_list[-1])

        chunk_terminations = raw_chunk_terminations
        chunk_truncations = raw_chunk_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _sync_rlt_policy_switch_episode_info(self, infos):
        if not isinstance(infos, dict) or "episode" not in infos:
            return
        for key in (
            "rlt_switch_flags",
            "entered_actor_phase_once",
            "actor_switch_step",
            "actor_switch_step_nonzero",
        ):
            if key in infos:
                infos["episode"][key] = infos[key].reshape(self.num_envs, -1)[
                    :, -1
                ].clone()

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) == 3:
            img = img[None]

        if infos is not None:
            for i in range(len(img)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                img[i] = put_info_on_image(img[i], info_item)
        if len(img.shape) > 3:
            if len(img) == 1:
                img = img[0]
            else:
                img = tile_images(img, nrows=int(np.sqrt(self.num_envs)))
        return img

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()
