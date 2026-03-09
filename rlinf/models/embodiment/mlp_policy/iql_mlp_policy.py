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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.modules.utils import get_act_func


class IQLMLPPolicy(MLPPolicy):
    """IQL-specific policy derived from the generic MLPPolicy."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_action_chunks,
        add_value_head,
        add_q_head,
        q_head_type="default",
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            add_value_head=add_value_head,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
        )
        self.iql_kind = None

    def configure_iql(self, iql_config: dict) -> None:
        hidden_dims = iql_config.get("hidden_dims", None)
        if hidden_dims is None:
            raise ValueError("hidden_dims must be provided in iql_config.")
        self.iql_kind = iql_config.get("kind", None)

        self._init_iql(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=iql_config.get("dropout_rate", None),
            log_std_min=float(iql_config.get("log_std_min", -5.0)),
            log_std_max=float(iql_config.get("log_std_max", 2.0)),
            state_dependent_std=bool(iql_config.get("state_dependent_std", False)),
        )

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        dropout_rate: float | None = None,
        activate_final: bool = False,
    ) -> nn.Sequential:
        act = get_act_func("relu")
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        if activate_final:
            layers.append(act())
        return nn.Sequential(*layers)

    def _init_iql(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        dropout_rate: float | None,
        log_std_min: float,
        log_std_max: float,
        state_dependent_std: bool,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if self.iql_kind == "actor":
            self.backbone = self._build_mlp(
                input_dim=obs_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
                dropout_rate=dropout_rate,
                activate_final=True,
            )
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.state_dependent_std = state_dependent_std
            if state_dependent_std:
                self.actor_logstd = nn.Linear(hidden_dims[-1], action_dim)
            else:
                self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
            self.logstd_range = (log_std_min, log_std_max)
        elif self.iql_kind == "critic":
            self.net = self._build_mlp(
                input_dim=obs_dim + action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif self.iql_kind == "value":
            self.net = self._build_mlp(
                input_dim=obs_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        else:
            raise ValueError(f"Unsupported iql_kind: {self.iql_kind}")

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.IQL:
            return self.iql_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def iql_forward(self, **kwargs) -> torch.Tensor:
        observations = kwargs.get("observations")
        if observations is None:
            raise ValueError("IQL forward expects observations.")

        actions = kwargs.get("actions")
        temperature = float(kwargs.get("temperature", 1.0))
        kind = kwargs.get("kind", None)
        kind = self.iql_kind if kind is None else kind

        if kind == "actor":
            feat = self.backbone(observations)
            action_mean_raw = self.actor_mean(feat)
            if self.state_dependent_std:
                action_logstd = self.actor_logstd(feat)
            else:
                # (1, action_dim) broadcasts to (batch, action_dim)
                action_logstd = self.actor_logstd.unsqueeze(0)
            action_logstd = torch.clamp(
                action_logstd, self.logstd_range[0], self.logstd_range[1]
            )
            action_std = torch.exp(action_logstd)

            if actions is not None:
                raw_actions = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
                log_prob = (
                    Normal(action_mean_raw, action_std).log_prob(raw_actions).sum(-1)
                )
                # Numerically stable: -log(1-tanh²(x)) = 2*(log(2) - x - softplus(-2x))
                log_prob -= (
                    2 * (math.log(2) - raw_actions - F.softplus(-2 * raw_actions))
                ).sum(-1)
                return log_prob

            mode = "eval" if float(temperature) == 0.0 else "train"
            if mode == "train":
                sampling_std = action_std * max(float(temperature), 1e-6)
                raw_action = Normal(action_mean_raw, sampling_std).rsample()
            elif mode == "eval":
                raw_action = action_mean_raw.clone()
            else:
                raise NotImplementedError(f"{mode=}")
            return torch.tanh(raw_action)
        if kind == "critic":
            if actions is None:
                raise ValueError("IQL critic expects actions.")
            x = torch.cat([observations, actions], dim=-1)
            return self.net(x).squeeze(-1)
        if kind == "value":
            return self.net(observations).squeeze(-1)
        raise RuntimeError(f"Unsupported iql_kind: {kind}")
