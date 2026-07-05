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
        self._rlt_expert_model_config = self._build_rlt_expert_model_config()
        return True

    def _ensure_rlt_expert_model_loaded(self):
        if self.expert_model is not None:
            return self.expert_model
        expert_model_config = getattr(self, "_rlt_expert_model_config", None)
        if expert_model_config is None:
            raise RuntimeError(
                "ManiSkill RLT expert takeover was requested, but "
                "rollout.expert_model is not configured."
            )
        self.expert_model = get_model(copy.deepcopy(expert_model_config))
        self.expert_model.eval()
        self.expert_model.requires_grad_(False)
        return self.expert_model

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
        use_actor = env_obs.get("rlt_use_actor", None)
        if use_actor is None:
            fallback_critical_phase = torch.zeros(
                (batch_size,),
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            fallback_critical_phase = torch.as_tensor(
                use_actor,
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
        if use_actor is None:
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
                use_actor, device=actions.device
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
