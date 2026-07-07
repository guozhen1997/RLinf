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

"""ManiSkill-specific RLT policy switch integration.

This is intentionally a composition helper for ``ManiskillEnv`` instead of a
``gym.Wrapper``. The switch touches chunk-level rollout state, metrics, reset
bookkeeping, and RLT OpenPI observations that live in RLinf's ManiSkill adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from mani_skill.utils.common import torch_clone_dict
from omegaconf import DictConfig

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


class ManiSkillRLTPolicySwitchWrapper:
    """RLT policy switch helper owned by ``ManiskillEnv``."""

    POLICY_INFO_KEYS = (
        "rlt_switch_flags",
        "entered_actor_phase_once",
        "actor_switch_step",
        "actor_switch_step_nonzero",
    )

    def __init__(
        self,
        *,
        owner,
        switch_cfg: DictConfig | None,
        is_peg_insertion_side: bool,
    ) -> None:
        self.owner = owner
        self.switch_cfg = switch_cfg
        self.is_peg_insertion_side = is_peg_insertion_side
        self.peg_event_state = (
            init_peg_insertion_event_state(
                num_envs=self.owner.num_envs,
                device=self.owner.device,
            )
            if self.is_peg_insertion_side
            else None
        )
        self.controller = self._build_controller()
        self._init_persistent_done_state()

    @staticmethod
    def is_peg_insertion_side_env_id(env_id: str | None) -> bool:
        return is_peg_insertion_side_env_id(env_id)

    @staticmethod
    def patch_env_args(
        env_args: dict[str, Any],
        *,
        wrap_obs_mode: str,
    ) -> dict[str, Any]:
        return patch_rlt_openpi_joint_env_args(
            env_args,
            wrap_obs_mode=wrap_obs_mode,
        )

    def _build_controller(self) -> "ManiSkillRLTPolicySwitchController | None":
        if self.switch_cfg is None:
            return None
        if not bool(self.switch_cfg.get("enable", False)):
            return None
        if not self.is_peg_insertion_side:
            raise ValueError(
                "ManiSkill RLT policy switch is only supported for peg-insertion tasks."
            )
        return ManiSkillRLTPolicySwitchController(
            cfg=self.switch_cfg,
            batch_size=self.owner.num_envs,
            hole_radii=getattr(self.owner.env.unwrapped, "box_hole_radii", None),
        )

    def resolve_task_descriptions(self, env):
        return resolve_maniskill_task_descriptions(
            env,
            num_envs=self.owner.num_envs,
            is_peg_insertion_side=self.is_peg_insertion_side,
        )

    def handles_obs_mode(self, wrap_obs_mode: str) -> bool:
        return wrap_obs_mode == RLT_OPENPI_JOINT_WRAP_MODE

    def wrap_obs(
        self,
        raw_obs: dict[str, Any],
        *,
        infos: dict[str, Any] | None,
        task_descriptions,
    ) -> dict[str, Any]:
        if self.owner.env.unwrapped.obs_mode != "rgb":
            raise ValueError(
                "wrap_obs_mode='rlt_openpi_joint' requires ManiSkill obs_mode='rgb'."
            )
        obs = wrap_rlt_openpi_joint_obs(
            raw_obs,
            infos=infos,
            task_descriptions=task_descriptions,
            num_envs=self.owner.num_envs,
            device=self.owner.device,
            is_peg_insertion_side=self.is_peg_insertion_side,
        )
        return self.attach_obs(obs)

    def attach_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.controller is None:
            return obs
        obs.update(self.controller.export_obs(device=self.owner.device))
        return obs

    def reset_episode_event_state(self, env_idx=None) -> None:
        if self.peg_event_state is None:
            return
        reset_peg_insertion_event_state(
            self.peg_event_state,
            env_idx=env_idx,
        )

    def reset_rollout_state(self, env_idx=None) -> None:
        self._reset_persistent_done_state(env_idx)
        if self.controller is not None:
            self.controller.reset(env_idx=env_idx)

    def augment_step_info(self, infos: dict[str, Any]) -> dict[str, Any]:
        infos = maybe_augment_peg_insertion_info(
            env=self.owner.env.unwrapped,
            infos=infos,
            event_state=self.peg_event_state,
            device=self.owner.device,
            is_peg_insertion_side=self.is_peg_insertion_side,
        )
        infos["elapsed_steps"] = self.owner.elapsed_steps.clone()
        return infos

    def record_metrics(
        self,
        episode_info: dict[str, torch.Tensor],
        infos: dict[str, Any],
    ) -> None:
        for key in self.POLICY_INFO_KEYS:
            if key in infos:
                value = infos[key]
                if isinstance(value, torch.Tensor):
                    episode_info[key] = value.reshape(self.owner.num_envs, -1)[
                        :, -1
                    ].clone()

    def attach_step_info(self, infos: dict[str, Any]) -> None:
        if self.controller is None:
            return
        policy_info = self.controller.export_info(device=self.owner.device)
        infos["policy_info"] = policy_info
        for key, value in policy_info.items():
            infos[key] = value

    def begin_chunk_step(self) -> dict[str, Any]:
        owner = self.owner
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
        frozen_dones = (
            self._persistent_done_mask.clone()
            if not owner.auto_reset
            else torch.zeros(owner.num_envs, device=owner.device, dtype=torch.bool)
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
        return {
            "frozen_dones": frozen_dones,
            "initially_frozen_dones": initially_frozen_dones,
            "last_extracted_obs": last_extracted_obs,
            "last_infos": last_infos,
        }

    def step_chunk_action(self, actions, chunk_state: dict[str, Any]):
        owner = self.owner
        frozen_dones = chunk_state["frozen_dones"]
        last_extracted_obs = chunk_state["last_extracted_obs"]
        last_infos = chunk_state["last_infos"]

        if (
            frozen_dones.all()
            and last_extracted_obs is not None
            and last_infos is not None
        ):
            extracted_obs = torch_clone_dict(last_extracted_obs)
            infos = torch_clone_dict(last_infos)
            step_reward = torch.zeros(
                owner.num_envs, device=owner.device, dtype=torch.float32
            )
            terminations = torch.zeros(
                owner.num_envs, device=owner.device, dtype=torch.bool
            )
            truncations = torch.zeros(
                owner.num_envs, device=owner.device, dtype=torch.bool
            )
        else:
            state_before_step = self._snapshot_episode_state()
            actions = self._zero_frozen_actions(actions, frozen_dones)
            extracted_obs, step_reward, terminations, truncations, infos = owner.step(
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

        frozen_dones |= torch.logical_or(terminations, truncations)
        chunk_state["last_extracted_obs"] = torch_clone_dict(extracted_obs)
        chunk_state["last_infos"] = torch_clone_dict(infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def finalize_chunk_step(
        self,
        *,
        chunk_state: dict[str, Any],
        obs_list: list[dict[str, Any]],
        infos_list: list[dict[str, Any]],
        raw_chunk_terminations: torch.Tensor,
        raw_chunk_truncations: torch.Tensor,
    ) -> None:
        if not obs_list or not infos_list:
            return

        initially_frozen_dones = chunk_state["initially_frozen_dones"]
        policy_switch_chunk_dones = torch.logical_or(
            raw_chunk_terminations,
            raw_chunk_truncations,
        )
        if initially_frozen_dones.any():
            policy_switch_chunk_dones = policy_switch_chunk_dones | (
                initially_frozen_dones[:, None].expand_as(policy_switch_chunk_dones)
            )
        self.apply_policy_switch_info(
            infos_list=infos_list,
            chunk_dones=policy_switch_chunk_dones,
        )
        self.sync_episode_info(infos_list[-1])
        obs_list[-1] = self.attach_obs(obs_list[-1])

    def update_persistent_done_state(
        self,
        dones: torch.Tensor,
        extracted_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> None:
        self._update_persistent_done_state(dones, extracted_obs, infos)

    def apply_policy_switch_info(
        self,
        *,
        infos_list: list[dict[str, Any]],
        chunk_dones: torch.Tensor,
    ) -> None:
        if self.controller is None:
            return
        infos_last = infos_list[-1]
        policy_info = self.controller.update(
            infos=infos_last,
            chunk_dones=chunk_dones,
        )
        infos_last["policy_info"] = policy_info
        for key, value in policy_info.items():
            infos_last[key] = value
        infos_list[-1] = infos_last

    def sync_episode_info(self, infos: dict[str, Any]) -> None:
        if not isinstance(infos, dict) or "episode" not in infos:
            return
        for key in self.POLICY_INFO_KEYS:
            if key in infos:
                infos["episode"][key] = (
                    infos[key].reshape(self.owner.num_envs, -1)[:, -1].clone()
                )

    def _init_persistent_done_state(self) -> None:
        self._persistent_done_mask = torch.zeros(
            self.owner.num_envs, device=self.owner.device, dtype=torch.bool
        )
        self._persistent_done_obs = None
        self._persistent_done_infos = None

    def _reset_persistent_done_state(self, env_idx=None) -> None:
        if not hasattr(self, "_persistent_done_mask"):
            self._init_persistent_done_state()
            return

        if env_idx is None:
            self._persistent_done_mask.zero_()
            self._persistent_done_obs = None
            self._persistent_done_infos = None
            return

        self._persistent_done_mask[env_idx] = False

    def _update_persistent_done_state(
        self,
        dones: torch.Tensor,
        extracted_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> None:
        if self.owner.auto_reset or not dones.any():
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

    def _snapshot_episode_state(self) -> dict[str, Any]:
        state = {
            "prev_step_reward": self.owner.prev_step_reward.clone(),
        }
        if self.owner.record_metrics:
            state.update(
                {
                    "success_once": self.owner.success_once.clone(),
                    "fail_once": self.owner.fail_once.clone(),
                    "returns": self.owner.returns.clone(),
                }
            )
        if self.peg_event_state is not None:
            state["peg_event_state"] = snapshot_peg_insertion_event_state(
                self.peg_event_state
            )
        return state

    def _restore_episode_state(self, state: dict[str, Any], mask: torch.Tensor) -> None:
        self.owner.prev_step_reward[mask] = state["prev_step_reward"][mask]
        if self.owner.record_metrics:
            self.owner.success_once[mask] = state["success_once"][mask]
            self.owner.fail_once[mask] = state["fail_once"][mask]
            self.owner.returns[mask] = state["returns"][mask]
        if self.peg_event_state is not None:
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
                if value.ndim > 0 and value.shape[0] == self.owner.num_envs:
                    value_mask = mask.to(device=value.device)
                    value[value_mask] = prev_value.to(value.device)[value_mask]
            elif isinstance(value, dict) and isinstance(prev_value, dict):
                restored[key] = self._restore_frozen_values(value, prev_value, mask)
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


class ManiSkillRLTPolicySwitchController:
    """State machine that switches from reference actions to the RLT actor."""

    REQUIRED_INFO_KEYS = (
        "consecutive_grasp_current",
        "success_current",
        "peg_head_hole_x",
        "peg_head_hole_abs_y",
        "peg_head_hole_abs_z",
    )

    FULL_TASK = "full_task"
    CRITICAL_PHASE = "critical_phase"

    AUTO_TRIGGER = "auto"
    ALWAYS_ON_TRIGGER = "always_on"
    INTERVENTION_GATE_TRIGGER = "intervention_gate"
    STALLED_PROGRESS_TRIGGER = "stalled_progress"

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        hole_radii: torch.Tensor | None = None,
    ) -> None:
        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.hole_radii = hole_radii
        self.state = self.init_state(self.batch_size)

    def init_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        task_mode = str(self.cfg.get("task_mode", self.FULL_TASK))
        start_active = task_mode == self.CRITICAL_PHASE
        return {
            "rlt_switch_flags": torch.full(
                (batch_size,),
                start_active,
                dtype=torch.bool,
            ),
            "entered_actor_phase_once": torch.full(
                (batch_size,),
                start_active,
                dtype=torch.bool,
            ),
            "actor_switch_step": torch.zeros(batch_size, dtype=torch.float32),
            "expert_takeover_active": torch.zeros(batch_size, dtype=torch.bool),
            "expert_progress_guard": torch.zeros(batch_size, dtype=torch.bool),
            "progress_initialized": torch.zeros(batch_size, dtype=torch.bool),
            "best_progress_x": torch.zeros(batch_size, dtype=torch.float32),
            "best_progress_yz": torch.zeros(batch_size, dtype=torch.float32),
            "best_progress_score": torch.zeros(batch_size, dtype=torch.float32),
            "stalled_progress_chunks": torch.zeros(batch_size, dtype=torch.float32),
        }

    def reset(self, env_idx=None) -> None:
        new_state = self.init_state(self.batch_size)
        if env_idx is None:
            self.state = new_state
            return
        for key, value in new_state.items():
            state_value = self.state[key]
            index = env_idx
            if isinstance(index, torch.Tensor):
                index = index.to(device=state_value.device)
            value = value.to(device=state_value.device)
            self.state[key][index] = value[index]

    def export_obs(self, *, device: torch.device) -> dict[str, torch.Tensor]:
        state = self._state_to(device)
        return {
            "rlt_switch_flags": state["rlt_switch_flags"][:, None],
        }

    def update(
        self,
        *,
        infos: dict[str, Any],
        chunk_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        infos = self.select_info_source(infos)
        device = infos["peg_head_hole_x"].device
        state = self._state_to(device)

        task_mode = str(self.cfg.get("task_mode", self.FULL_TASK))
        trigger_mode = str(self.cfg.get("trigger_mode", self.AUTO_TRIGGER))
        if task_mode == self.CRITICAL_PHASE or trigger_mode == self.ALWAYS_ON_TRIGGER:
            enter_actor = torch.ones(self.batch_size, dtype=torch.bool, device=device)
        elif task_mode == self.FULL_TASK and trigger_mode == self.AUTO_TRIGGER:
            enter_actor = self._auto_enter_actor(infos, device)
        else:
            raise ValueError(
                "rlt_policy_switch supports task_mode in "
                f"{self.FULL_TASK, self.CRITICAL_PHASE} and trigger_mode in "
                f"{self.AUTO_TRIGGER, self.ALWAYS_ON_TRIGGER}, got "
                f"{task_mode=} {trigger_mode=}."
            )

        previous_switch_flags = state["rlt_switch_flags"]
        latch_until_done = bool(self.cfg.get("latch_until_done", True))
        if latch_until_done:
            switch_flags = previous_switch_flags | enter_actor
        else:
            switch_flags = enter_actor

        switched_now = (~previous_switch_flags) & switch_flags
        elapsed_steps = self._elapsed_steps(infos, device)
        state["actor_switch_step"] = torch.where(
            switched_now,
            elapsed_steps,
            state["actor_switch_step"],
        )
        state["entered_actor_phase_once"] = (
            state["entered_actor_phase_once"] | switch_flags | switched_now
        )
        state["rlt_switch_flags"] = switch_flags
        self._update_expert_takeover_state(infos=infos, device=device)

        return self.export_info(device=device, infos=infos, chunk_dones=chunk_dones)

    def export_info(
        self,
        *,
        device: torch.device,
        infos: dict[str, Any] | None = None,
        chunk_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self._state_to(device)
        expert_takeover = self._expert_takeover_mask(infos=infos, device=device)
        record_transition = state["rlt_switch_flags"]
        if chunk_dones is not None:
            chunk_done_mask = torch.as_tensor(
                chunk_dones,
                device=device,
                dtype=torch.bool,
            ).reshape(self.batch_size, -1)
            record_transition = record_transition & (~chunk_done_mask.any(dim=1))
        return {
            "rlt_switch_flags": state["rlt_switch_flags"][:, None],
            "entered_actor_phase_once": state["entered_actor_phase_once"][:, None],
            "actor_switch_step": state["actor_switch_step"][:, None],
            "actor_switch_step_nonzero": torch.where(
                state["entered_actor_phase_once"],
                state["actor_switch_step"],
                torch.zeros_like(state["actor_switch_step"]),
            )[:, None],
            "in_critical_phase": state["rlt_switch_flags"][:, None],
            # Match real-world RLT: only the latched critical phase is stored
            # for Stage2 replay. This must not depend on learner warmup.
            "record_transition": record_transition[:, None],
            "expert_takeover": expert_takeover[:, None],
            "expert_takeover_active": state["expert_takeover_active"][:, None],
            "expert_progress_guard": state["expert_progress_guard"][:, None],
            "expert_stalled_progress_chunks": state["stalled_progress_chunks"][:, None],
            "intervention_phase": expert_takeover.to(torch.float32)[:, None],
        }

    def _expert_takeover_mask(
        self,
        *,
        infos: dict[str, Any] | None,
        device: torch.device,
    ) -> torch.Tensor:
        expert_cfg = self.cfg.get("expert_takeover", {})
        if not bool(expert_cfg.get("enable", False)):
            return torch.zeros(self.batch_size, dtype=torch.bool, device=device)

        trigger_mode = str(expert_cfg.get("trigger_mode", "critical_phase"))
        state = self._state_to(device)
        if trigger_mode == "critical_phase":
            return state["rlt_switch_flags"].clone()
        if trigger_mode == self.INTERVENTION_GATE_TRIGGER:
            return state["rlt_switch_flags"] & self._expert_intervention_gate(
                infos=infos,
                expert_cfg=expert_cfg,
                device=device,
            )
        if trigger_mode == self.STALLED_PROGRESS_TRIGGER:
            takeover = state["rlt_switch_flags"] & state["expert_takeover_active"]
            if infos is not None:
                takeover = takeover & (
                    ~self._info_bool(infos, "success_current", device)
                )
            return takeover
        if trigger_mode == "always_on":
            return torch.ones(self.batch_size, dtype=torch.bool, device=device)
        raise ValueError(
            "rlt_policy_switch.expert_takeover.trigger_mode supports "
            "'critical_phase', 'intervention_gate', 'stalled_progress', and "
            f"'always_on', got {trigger_mode!r}."
        )

    def _update_expert_takeover_state(
        self,
        *,
        infos: dict[str, Any],
        device: torch.device,
    ) -> None:
        expert_cfg = self.cfg.get("expert_takeover", {})
        state = self._state_to(device)
        if not bool(expert_cfg.get("enable", False)):
            state["expert_takeover_active"].zero_()
            state["expert_progress_guard"].zero_()
            state["progress_initialized"].zero_()
            state["stalled_progress_chunks"].zero_()
            return

        trigger_mode = str(expert_cfg.get("trigger_mode", "critical_phase"))
        if trigger_mode == self.STALLED_PROGRESS_TRIGGER:
            self._update_stalled_progress_expert_takeover(
                infos=infos,
                expert_cfg=expert_cfg,
                device=device,
            )
            return

        # Non-stateful modes compute their mask directly in _expert_takeover_mask.
        state["expert_takeover_active"].zero_()
        state["expert_progress_guard"].zero_()
        state["progress_initialized"].zero_()
        state["stalled_progress_chunks"].zero_()

    def _update_stalled_progress_expert_takeover(
        self,
        *,
        infos: dict[str, Any],
        expert_cfg: DictConfig | dict,
        device: torch.device,
    ) -> None:
        state = self._state_to(device)
        gate_cfg = expert_cfg.get("gate", {})

        in_critical_phase = state["rlt_switch_flags"]
        success = self._info_bool(infos, "success_current", device)
        active_before = state["expert_takeover_active"] & in_critical_phase & (~success)
        progress_guard = self._stalled_progress_guard(
            infos=infos,
            gate_cfg=gate_cfg,
            device=device,
        )
        state["expert_progress_guard"] = progress_guard

        eligible = in_critical_phase & progress_guard & (~success)
        if bool(gate_cfg.get("require_grasp", False)):
            eligible = eligible & self._info_bool(
                infos,
                "consecutive_grasp_current",
                device,
            )

        # Progress is judged once per action chunk. Any meaningful forward x
        # movement, lateral/vertical alignment improvement, or combined score
        # improvement resets the stall counter.
        hole_x = self._info_float(infos, "peg_head_hole_x", device)
        abs_y = self._info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._info_float(infos, "peg_head_hole_abs_z", device)
        yz_dist = torch.sqrt(torch.square(abs_y) + torch.square(abs_z))
        yz_weight = float(gate_cfg.get("progress_yz_weight", 1.0))
        progress_score = hole_x - yz_weight * yz_dist

        initialized = state["progress_initialized"] & eligible & (~active_before)
        should_initialize = eligible & (~active_before) & (~initialized)
        monitor_progress = eligible & (~active_before) & initialized

        min_x_progress = float(gate_cfg.get("min_x_progress", 0.003))
        min_yz_progress = float(gate_cfg.get("min_yz_progress", 0.0015))
        min_score_progress = float(gate_cfg.get("min_score_progress", 0.002))
        x_improved = hole_x > (state["best_progress_x"] + min_x_progress)
        yz_improved = yz_dist < (state["best_progress_yz"] - min_yz_progress)
        score_improved = progress_score > (
            state["best_progress_score"] + min_score_progress
        )
        improved = monitor_progress & (x_improved | yz_improved | score_improved)

        no_progress = monitor_progress & (~improved)
        stalled_chunks = torch.where(
            no_progress,
            state["stalled_progress_chunks"] + 1.0,
            state["stalled_progress_chunks"],
        )
        stalled_chunks = torch.where(
            improved | should_initialize | (~eligible),
            torch.zeros_like(stalled_chunks),
            stalled_chunks,
        )

        stuck_chunks_before_takeover = max(
            1,
            int(gate_cfg.get("stuck_chunks_before_takeover", 3)),
        )
        trigger_now = no_progress & (
            stalled_chunks >= float(stuck_chunks_before_takeover)
        )

        update_best = should_initialize | improved
        state["best_progress_x"] = torch.where(
            should_initialize,
            hole_x,
            torch.where(
                update_best,
                torch.maximum(state["best_progress_x"], hole_x),
                state["best_progress_x"],
            ),
        )
        state["best_progress_yz"] = torch.where(
            should_initialize,
            yz_dist,
            torch.where(
                update_best,
                torch.minimum(state["best_progress_yz"], yz_dist),
                state["best_progress_yz"],
            ),
        )
        state["best_progress_score"] = torch.where(
            should_initialize,
            progress_score,
            torch.where(
                update_best,
                torch.maximum(state["best_progress_score"], progress_score),
                state["best_progress_score"],
            ),
        )

        # Once intervention starts, the expert keeps control until success,
        # termination/truncation reset, or a full env reset. We do not hand
        # control back mid-episode because that flickers and makes labels noisy.
        state["expert_takeover_active"] = active_before | trigger_now
        state["progress_initialized"] = torch.where(
            eligible & (~active_before),
            state["progress_initialized"] | should_initialize,
            torch.zeros_like(state["progress_initialized"]),
        )
        state["stalled_progress_chunks"] = torch.where(
            active_before,
            state["stalled_progress_chunks"],
            stalled_chunks,
        )

    def _stalled_progress_guard(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        hole_x = self._info_float(infos, "peg_head_hole_x", device)
        abs_y = self._info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._hole_radii(abs_y, gate_cfg, device)

        near_hole_x_min = float(gate_cfg.get("near_hole_x_min", -0.10))
        yz_margin = float(gate_cfg.get("near_hole_yz_margin", 2.0))
        guard = (
            (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )
        if bool(gate_cfg.get("require_prealigned", False)):
            guard = guard & self._prealigned_mask(
                infos=infos,
                gate_cfg=gate_cfg,
                device=device,
            )
        return guard

    def _expert_intervention_gate(
        self,
        *,
        infos: dict[str, Any] | None,
        expert_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if infos is None:
            return torch.zeros(self.batch_size, dtype=torch.bool, device=device)

        gate_cfg = expert_cfg.get("gate", {})
        if "partial_insert_current" in infos and not gate_cfg:
            gate = self._info_bool(infos, "partial_insert_current", device)
        else:
            gate = self._threshold_gate(
                infos=infos,
                gate_cfg=gate_cfg,
                device=device,
            )

        if bool(gate_cfg.get("require_grasp", False)):
            gate = gate & self._info_bool(
                infos,
                "consecutive_grasp_current",
                device,
            )
        if bool(gate_cfg.get("require_not_success", True)):
            gate = gate & (~self._info_bool(infos, "success_current", device))
        return gate

    def _threshold_gate(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        prealigned = self._prealigned_mask(
            infos=infos,
            gate_cfg=gate_cfg,
            device=device,
        )

        hole_x = self._info_float(infos, "peg_head_hole_x", device)
        abs_y = self._info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._hole_radii(abs_y, gate_cfg, device)

        near_hole_x_min = float(gate_cfg.get("near_hole_x_min", -0.05))
        yz_margin = float(gate_cfg.get("near_hole_yz_margin", 1.25))
        return (
            prealigned
            & (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )

    def _prealigned_mask(
        self,
        *,
        infos: dict[str, Any],
        gate_cfg: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if "prealigned_current" in infos:
            return self._info_bool(infos, "prealigned_current", device)

        yz_threshold = float(gate_cfg.get("prealign_yz_threshold", 0.01))
        return (
            self._info_float(infos, "peg_head_goal_yz_dist", device) < yz_threshold
        ) & (self._info_float(infos, "peg_body_goal_yz_dist", device) < yz_threshold)

    def _info_bool(
        self,
        infos: dict[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        return self._info_tensor(
            infos,
            key,
            dtype=torch.bool,
            device=device,
        )

    def _info_float(
        self,
        infos: dict[str, Any],
        key: str,
        device: torch.device,
    ) -> torch.Tensor:
        return self._info_tensor(
            infos,
            key,
            dtype=torch.float32,
            device=device,
        )

    def _info_tensor(
        self,
        infos: dict[str, Any],
        key: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        value = torch.as_tensor(infos[key], device=device).to(dtype=dtype)
        if value.numel() == 1:
            return value.reshape(1).repeat(self.batch_size)
        if value.numel() == self.batch_size:
            return value.reshape(self.batch_size)
        return value.reshape(self.batch_size, -1)[:, -1]

    @classmethod
    def select_info_source(cls, infos: dict[str, Any]) -> dict[str, Any]:
        if all(key in infos for key in cls.REQUIRED_INFO_KEYS):
            return infos
        final_info = infos.get("final_info")
        if isinstance(final_info, dict) and all(
            key in final_info for key in cls.REQUIRED_INFO_KEYS
        ):
            return final_info
        missing = [key for key in cls.REQUIRED_INFO_KEYS if key not in infos]
        raise RuntimeError(
            "RLT policy switch is enabled, but ManiSkill info is missing "
            f"required keys {missing}. This usually means the env wrapper is not "
            "using the aligned peg-insertion info path."
        )

    def _auto_enter_actor(
        self,
        infos: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        auto_gate = self.cfg.get("auto_gate", {})
        grasp = self._info_bool(infos, "consecutive_grasp_current", device)
        success = self._info_bool(infos, "success_current", device)
        hole_x = self._info_float(infos, "peg_head_hole_x", device)
        abs_y = self._info_float(infos, "peg_head_hole_abs_y", device)
        abs_z = self._info_float(infos, "peg_head_hole_abs_z", device)
        hole_radii = self._hole_radii(abs_y, auto_gate, device)

        near_hole_x_min = float(auto_gate.get("near_hole_x_min", -0.16))
        yz_margin = float(auto_gate.get("near_hole_yz_margin", 1.5))
        near_hole = (
            (hole_x >= near_hole_x_min)
            & (abs_y <= yz_margin * hole_radii)
            & (abs_z <= yz_margin * hole_radii)
        )

        enter_actor = near_hole
        if bool(auto_gate.get("require_grasp", True)):
            enter_actor = enter_actor & grasp
        if bool(auto_gate.get("require_not_success", True)):
            enter_actor = enter_actor & (~success)
        return enter_actor

    def _hole_radii(
        self,
        like: torch.Tensor,
        auto_gate: DictConfig | dict,
        device: torch.device,
    ) -> torch.Tensor:
        if self.hole_radii is not None:
            return self.hole_radii.to(device, dtype=torch.float32)
        fallback_hole_radius = auto_gate.get("fallback_hole_radius", 0.03)
        return torch.full_like(like, float(fallback_hole_radius))

    def _elapsed_steps(
        self,
        infos: dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        elapsed_steps = infos.get("elapsed_steps", infos.get("_elapsed_steps"))
        if elapsed_steps is None:
            return torch.zeros(self.batch_size, dtype=torch.float32, device=device)
        elapsed_steps = torch.as_tensor(
            elapsed_steps,
            dtype=torch.float32,
            device=device,
        )
        if elapsed_steps.numel() == 1:
            return elapsed_steps.reshape(1).repeat(self.batch_size)
        return elapsed_steps.reshape(self.batch_size, -1)[:, -1]

    def _state_to(self, device: torch.device) -> dict[str, torch.Tensor]:
        for key, value in self.state.items():
            self.state[key] = value.to(device)
        return self.state
