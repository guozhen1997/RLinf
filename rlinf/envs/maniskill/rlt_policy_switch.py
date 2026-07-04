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

"""Automatic RLT policy switch controller for ManiSkill peg insertion."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig


def build_maniskill_rlt_policy_switch_controller(
    *,
    switch_cfg: DictConfig | None,
    is_peg_insertion_side: bool,
    env,
    batch_size: int,
) -> "ManiSkillRLTPolicySwitchController | None":
    if switch_cfg is None:
        return None
    if not bool(switch_cfg.get("enable", False)):
        return None
    if not is_peg_insertion_side:
        raise ValueError(
            "ManiSkill RLT policy switch is only supported for peg-insertion tasks."
        )
    return ManiSkillRLTPolicySwitchController(
        cfg=switch_cfg,
        batch_size=batch_size,
        hole_radii=getattr(env, "box_hole_radii", None),
    )


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
            "rlt_use_actor": torch.full(
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
            "rlt_use_actor": state["rlt_use_actor"][:, None],
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

        previous_use_actor = state["rlt_use_actor"]
        latch_until_done = bool(self.cfg.get("latch_until_done", True))
        if latch_until_done:
            use_actor = previous_use_actor | enter_actor
        else:
            use_actor = enter_actor

        switched_now = (~previous_use_actor) & use_actor
        elapsed_steps = self._elapsed_steps(infos, device)
        state["actor_switch_step"] = torch.where(
            switched_now,
            elapsed_steps,
            state["actor_switch_step"],
        )
        state["entered_actor_phase_once"] = (
            state["entered_actor_phase_once"] | use_actor | switched_now
        )
        state["rlt_use_actor"] = use_actor

        return self.export_info(device=device, chunk_dones=chunk_dones)

    def export_info(
        self,
        *,
        device: torch.device,
        chunk_dones: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self._state_to(device)
        expert_takeover = self._expert_takeover_mask(device=device)
        record_transition = state["rlt_use_actor"]
        if chunk_dones is not None:
            chunk_done_mask = torch.as_tensor(
                chunk_dones,
                device=device,
                dtype=torch.bool,
            ).reshape(self.batch_size, -1)
            record_transition = record_transition & (~chunk_done_mask.any(dim=1))
        return {
            "rlt_use_actor": state["rlt_use_actor"][:, None],
            "entered_actor_phase_once": state["entered_actor_phase_once"][:, None],
            "actor_switch_step": state["actor_switch_step"][:, None],
            "actor_switch_step_nonzero": torch.where(
                state["entered_actor_phase_once"],
                state["actor_switch_step"],
                torch.zeros_like(state["actor_switch_step"]),
            )[:, None],
            "in_critical_phase": state["rlt_use_actor"][:, None],
            # Match real-world RLT: only the latched critical phase is stored
            # for Stage2 replay. This must not depend on learner warmup.
            "record_transition": record_transition[:, None],
            "expert_takeover": expert_takeover[:, None],
            "intervention_phase": torch.zeros(
                (self.batch_size, 1),
                dtype=torch.float32,
                device=device,
            ),
        }

    def _expert_takeover_mask(self, *, device: torch.device) -> torch.Tensor:
        expert_cfg = self.cfg.get("expert_takeover", {})
        if not bool(expert_cfg.get("enable", False)):
            return torch.zeros(self.batch_size, dtype=torch.bool, device=device)

        trigger_mode = str(expert_cfg.get("trigger_mode", "critical_phase"))
        state = self._state_to(device)
        if trigger_mode == "critical_phase":
            return state["rlt_use_actor"].clone()
        if trigger_mode == "always_on":
            return torch.ones(self.batch_size, dtype=torch.bool, device=device)
        raise ValueError(
            "rlt_policy_switch.expert_takeover.trigger_mode supports "
            f"'critical_phase' and 'always_on', got {trigger_mode!r}."
        )

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
        grasp = infos["consecutive_grasp_current"].to(torch.bool)
        success = infos["success_current"].to(torch.bool)
        hole_x = infos["peg_head_hole_x"].to(torch.float32)
        abs_y = infos["peg_head_hole_abs_y"].to(torch.float32)
        abs_z = infos["peg_head_hole_abs_z"].to(torch.float32)
        hole_radii = self._hole_radii(abs_y, auto_gate, device)

        near_hole_x_min = float(auto_gate.get("near_hole_x_min", -0.16))
        yz_margin = float(auto_gate.get("near_hole_yz_margin", 1.5))
        near_hole = (hole_x >= near_hole_x_min) & (
            abs_y <= yz_margin * hole_radii
        ) & (abs_z <= yz_margin * hole_radii)

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


def attach_rlt_policy_switch_obs(
    *,
    controller: ManiSkillRLTPolicySwitchController | None,
    obs: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if controller is None:
        return obs
    obs.update(controller.export_obs(device=device))
    return obs


def apply_rlt_policy_switch_info(
    *,
    controller: ManiSkillRLTPolicySwitchController | None,
    infos_list: list[dict[str, Any]],
    chunk_dones: torch.Tensor,
) -> None:
    if controller is None:
        return
    infos_last = infos_list[-1]
    policy_info = controller.update(
        infos=infos_last,
        chunk_dones=chunk_dones,
    )
    infos_last["policy_info"] = policy_info
    for key, value in policy_info.items():
        infos_last[key] = value
    infos_list[-1] = infos_last
