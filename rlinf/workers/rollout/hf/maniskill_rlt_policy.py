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
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models import get_model


class ManiSkillRLTPolicyMixin:
    """ManiSkill-specific RLT action routing and expert takeover helpers."""

    def _defer_rlt_expert_model(self) -> bool:
        """Let ManiSkill RLT own its expert model without touching DAgger state."""
        return self._defer_maniskill_rlt_expert_model()

    def _use_rlt_maniskill_route(self) -> bool:
        train_env_cfg = self.cfg.env.get("train", None)
        return (
            str(self.cfg.algorithm.get("loss_type", "")) == "rlt_ac"
            and train_env_cfg is not None
            and str(train_env_cfg.get("env_type", "")) == "maniskill"
        )

    def _build_rlt_expert_model_config(self):
        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is None:
            raise ValueError(
                "ManiSkill RLT expert requires rollout.rlt_feature_model so the "
                "expert can reuse the same OpenPI/RL-token config."
            )
        expert_model_config = copy.deepcopy(rlt_feature_model_config)
        expert_overrides = OmegaConf.to_container(
            self.cfg.rollout.expert_model,
            resolve=True,
        )
        expert_overrides = {} if expert_overrides is None else dict(expert_overrides)
        return OmegaConf.merge(expert_model_config, expert_overrides)

    def _defer_maniskill_rlt_expert_model(self) -> bool:
        if not self._use_rlt_maniskill_route():
            return False
        self._rlt_expert_model = None
        self._rlt_expert_model_config = self._build_rlt_expert_model_config()
        return True

    def _ensure_rlt_expert_model_loaded(self):
        rlt_expert_model = getattr(self, "_rlt_expert_model", None)
        if rlt_expert_model is not None:
            return rlt_expert_model
        expert_model_config = getattr(self, "_rlt_expert_model_config", None)
        if expert_model_config is None:
            raise RuntimeError(
                "ManiSkill RLT expert takeover was requested, but "
                "rollout.expert_model is not configured."
            )
        self._rlt_expert_model = get_model(copy.deepcopy(expert_model_config))
        self._rlt_expert_model.eval()
        self._rlt_expert_model.requires_grad_(False)
        return self._rlt_expert_model

    def _offload_rlt_models(self) -> None:
        rlt_expert_model = getattr(self, "_rlt_expert_model", None)
        if rlt_expert_model is not None:
            rlt_expert_model.to("cpu")

    def _reload_rlt_models(self) -> None:
        rlt_expert_model = getattr(self, "_rlt_expert_model", None)
        if rlt_expert_model is not None:
            rlt_expert_model.to(self.device)

    def _rlt_schedule_cfg(self):
        return self.cfg.algorithm.get("rlt_schedule", {})

    def _rlt_schedule_value(self, key: str, default):
        schedule_cfg = self._rlt_schedule_cfg()
        return (
            schedule_cfg.get(key, default)
            if schedule_cfg is not None and key in schedule_cfg
            else self.cfg.algorithm.get(key, default)
        )

    def _use_rlt_schedule(self) -> bool:
        if str(self.cfg.algorithm.get("loss_type", "")) != "rlt_ac":
            return False
        schedule_cfg = self._rlt_schedule_cfg()
        if schedule_cfg is not None and "enable" in schedule_cfg:
            return bool(schedule_cfg.get("enable", False))
        return any(
            key in self.cfg.algorithm
            for key in (
                "warmup_post_collect_updates",
                "train_every_transitions",
                "train_every_episodes",
                "max_updates_per_train_step",
            )
        )

    def _rlt_ready_for_online(self) -> bool:
        return not self._use_rlt_schedule() or int(self.version) >= int(
            self._rlt_schedule_value("warmup_post_collect_updates", 0)
        )

    def _predict_with_rlt_features(
        self,
        env_obs: dict[str, Any],
        final_obs: dict[str, Any] | None,
        mode: Literal["train", "eval"],
        *,
        env_infos: dict[str, Any] | None = None,
        allow_expert: bool = True,
        rlt_switch_flags: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            rlt_obs = self.rlt_feature_model.extract_rlt_obs(env_obs)
            actions, result = self.hf_model.predict_action_batch(
                env_obs=rlt_obs,
                mode=mode,
                return_obs=True,
            )
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)

            if self._use_rlt_maniskill_route():
                actions, result = self._route_rlt_maniskill_actions(
                    env_obs=env_obs,
                    rlt_obs=rlt_obs,
                    student_actions=actions,
                    result=result,
                    mode=mode,
                    env_infos=env_infos,
                    allow_expert=allow_expert,
                )
            else:
                actions = self._route_rlt_reference_actions(
                    env_obs=env_obs,
                    actions=actions,
                    result=result,
                    rlt_switch_flags=rlt_switch_flags,
                )

            transition_obs = rlt_obs
            if final_obs is not None:
                transition_obs = self.rlt_feature_model.extract_rlt_obs(final_obs)
            for key in ("z_rl", "proprio", "ref_chunk"):
                result["forward_inputs"][f"rlt_transition_{key}"] = transition_obs[key]

        result["expert_label_flag"] = bool(result.get("expert_label_flag", False))
        return actions, result

    @staticmethod
    def _merge_maniskill_env_infos(
        env_infos_list: list[dict[str, Any] | None],
        merge_dicts,
    ) -> dict[str, Any] | None:
        if not any(env_infos is not None for env_infos in env_infos_list):
            return None
        merged_env_infos = {}
        keys = set()
        for env_infos in env_infos_list:
            if isinstance(env_infos, dict):
                keys.update(env_infos.keys())
        for key in keys:
            values = [
                env_infos[key]
                for env_infos in env_infos_list
                if isinstance(env_infos, dict) and key in env_infos
            ]
            if not values:
                continue
            first = values[0]
            if isinstance(first, dict):
                merged_env_infos[key] = merge_dicts(values)
            elif isinstance(first, torch.Tensor):
                merged_env_infos[key] = torch.cat(values, dim=0)
            else:
                merged_env_infos[key] = values
        return merged_env_infos

    @staticmethod
    def _flatten_action_chunk(actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() <= 2:
            return actions
        return actions.reshape(actions.shape[0], -1)

    @staticmethod
    def _policy_info_bool(
        policy_info: dict[str, Any] | None,
        key: str,
        *,
        batch_size: int,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        if policy_info is None or key not in policy_info:
            return torch.full(
                (batch_size,), bool(default), dtype=torch.bool, device=device
            )
        value = torch.as_tensor(policy_info[key], device=device)
        if value.numel() == 1:
            return torch.full(
                (batch_size,),
                bool(value.reshape(-1)[0].item()),
                dtype=torch.bool,
                device=device,
            )
        return value.reshape(batch_size, -1).to(torch.bool).any(dim=1)

    @staticmethod
    def _policy_info_float(
        policy_info: dict[str, Any] | None,
        key: str,
        *,
        batch_size: int,
        device: torch.device,
        default: float,
    ) -> torch.Tensor:
        if policy_info is None or key not in policy_info:
            return torch.full(
                (batch_size,), float(default), dtype=torch.float32, device=device
            )
        value = torch.as_tensor(policy_info[key], device=device)
        if value.numel() == 1:
            return torch.full(
                (batch_size,),
                float(value.reshape(-1)[0].item()),
                dtype=torch.float32,
                device=device,
            )
        return value.reshape(batch_size, -1)[:, -1].to(torch.float32)

    def _rlt_base_actions(
        self,
        ref_chunk: torch.Tensor,
        *,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ref_chunk.dim() == 2:
            ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
        return ref_chunk[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)

    def _route_rlt_reference_actions(
        self,
        *,
        env_obs: dict[str, Any],
        actions: torch.Tensor,
        result: dict[str, Any],
        rlt_switch_flags: torch.Tensor | None,
    ) -> torch.Tensor:
        if rlt_switch_flags is None:
            rlt_switch_flags = env_obs.get("rlt_switch_flags", None)
        if rlt_switch_flags is None:
            rlt_switch_flags = torch.full(
                (actions.shape[0], actions.shape[1]),
                self._rlt_ready_for_online() if self._use_rlt_schedule() else False,
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            rlt_switch_flags = torch.as_tensor(
                rlt_switch_flags, device=actions.device
            ).bool()
        if rlt_switch_flags.dim() == 1:
            rlt_switch_flags = rlt_switch_flags[:, None]
        if rlt_switch_flags.shape[1] > 1:
            rlt_switch_flags = rlt_switch_flags[:, -1:]
        if actions.shape[1] > 1:
            rlt_switch_flags = rlt_switch_flags.expand(-1, actions.shape[1])
        rlt_switch_flags = rlt_switch_flags.reshape(
            actions.shape[0], actions.shape[1], 1
        )
        ref_actions = result["forward_inputs"]["ref_chunk"].to(
            device=actions.device, dtype=actions.dtype
        )
        routed_actions = torch.where(
            rlt_switch_flags,
            actions,
            ref_actions[:, : actions.shape[1], : actions.shape[2]],
        ).contiguous()
        result["forward_inputs"]["action"] = routed_actions.reshape(
            routed_actions.shape[0], -1
        ).contiguous()
        return routed_actions

    def _rlt_expert_actions(
        self,
        env_obs: dict[str, Any],
        *,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expert_model = self._ensure_rlt_expert_model_loaded()
        with torch.no_grad():
            expert_actions, _ = expert_model.predict_action_batch(
                env_obs=env_obs,
                mode="eval",
                compute_values=False,
            )
        if isinstance(expert_actions, np.ndarray):
            expert_actions = torch.from_numpy(expert_actions)
        if expert_actions.dim() == 2:
            expert_actions = expert_actions.reshape(
                expert_actions.shape[0], -1, action_dim
            )
        return expert_actions[:, :chunk_len, :action_dim].to(
            device=device,
            dtype=dtype,
        )

    def _route_rlt_maniskill_actions(
        self,
        *,
        env_obs: dict[str, Any],
        rlt_obs: dict[str, torch.Tensor],
        student_actions: torch.Tensor,
        result: dict[str, Any],
        mode: Literal["train", "eval"],
        env_infos: dict[str, Any] | None,
        allow_expert: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions = student_actions
        batch_size, chunk_len, action_dim = actions.shape
        policy_info = None
        if isinstance(env_infos, dict) and isinstance(
            env_infos.get("policy_info"), dict
        ):
            policy_info = env_infos["policy_info"]

        ready_for_online = self._rlt_ready_for_online()
        switch_flags = env_obs.get("rlt_switch_flags", None)
        if switch_flags is None:
            fallback_critical_phase = torch.zeros(
                (batch_size,),
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            fallback_critical_phase = torch.as_tensor(
                switch_flags,
                device=actions.device,
            ).bool().reshape(batch_size, -1).any(dim=1)
        in_critical_phase = self._policy_info_bool(
            policy_info,
            "in_critical_phase",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        if policy_info is None or "in_critical_phase" not in policy_info:
            in_critical_phase = fallback_critical_phase
        requested_expert_takeover = self._policy_info_bool(
            policy_info,
            "expert_takeover",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        expert_takeover = (
            requested_expert_takeover
            & ready_for_online
            & bool(allow_expert)
            & (mode == "train")
        )
        if switch_flags is None:
            actor_control = (
                torch.full(
                    (batch_size,),
                    bool(ready_for_online),
                    dtype=torch.bool,
                    device=actions.device,
                )
                & in_critical_phase
            )
        else:
            actor_control = torch.as_tensor(
                switch_flags, device=actions.device
            ).bool().reshape(batch_size, -1).any(dim=1)
            if self._use_rlt_schedule():
                actor_control = actor_control & torch.full(
                    (batch_size,),
                    bool(ready_for_online),
                    dtype=torch.bool,
                    device=actions.device,
                )

        base_actions = self._rlt_base_actions(
            rlt_obs["ref_chunk"],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        routed_actions = torch.where(
            actor_control[:, None, None],
            actions,
            base_actions,
        ).contiguous()

        intervention_flags = torch.zeros(
            (batch_size, chunk_len),
            dtype=torch.bool,
            device=actions.device,
        )
        if expert_takeover.any():
            expert_actions = self._rlt_expert_actions(
                env_obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                device=actions.device,
                dtype=actions.dtype,
            )
            routed_actions = torch.where(
                expert_takeover[:, None, None],
                expert_actions,
                routed_actions,
            ).contiguous()
            intervention_flags[expert_takeover] = True

        forward_inputs = result["forward_inputs"]
        forward_inputs["action"] = self._flatten_action_chunk(routed_actions).detach()
        forward_inputs["intervention_flags"] = intervention_flags
        forward_inputs["student_control"] = actor_control[:, None]
        forward_inputs["intervention_requested"] = requested_expert_takeover[:, None]
        forward_inputs["in_critical_phase"] = in_critical_phase[:, None]
        record_transition = self._policy_info_bool(
            policy_info,
            "record_transition",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        if policy_info is None or "record_transition" not in policy_info:
            record_transition = in_critical_phase
        forward_inputs["record_transition"] = record_transition[:, None]
        forward_inputs["intervention_phase"] = self._policy_info_float(
            policy_info,
            "intervention_phase",
            batch_size=batch_size,
            device=actions.device,
            default=0.0,
        )[:, None]
        forward_inputs["ready_for_online"] = torch.full(
            (batch_size, 1),
            bool(ready_for_online),
            dtype=torch.bool,
            device=actions.device,
        )
        result["expert_label_flag"] = bool(expert_takeover.any().item())
        return routed_actions, result
