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

from __future__ import annotations

import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import rlinf.config as config_module
from rlinf.config import validate_embodied_cfg


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture(autouse=True)
def patch_cluster(monkeypatch):
    class _FakePlacement:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def get_world_size(self, component_name):
            del component_name
            return 1

    monkeypatch.setattr(config_module, "Cluster", lambda *args, **kwargs: None)
    monkeypatch.setattr(config_module, "HybridComponentPlacement", _FakePlacement)


def _make_embodied_eval_sglang_reward_cfg():
    return OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied_eval",
                "only_eval": True,
                "val_check_interval": -1,
            },
            "reward": {
                "use_reward_model": True,
                "worker_type": "api",
                "reward_mode": "history_buffer",
                "api": {
                    "api_base": "http://reward-api:30000",
                    "model": "Qwen3-VL-4B-Instruct",
                    "sampling_params": {"max_tokens": 16},
                },
                "model": {
                    "model_type": "history_vlm",
                    "model_path": "/models/Qwen3-VL-4B-Instruct",
                },
            },
            "rollout": {
                "pipeline_stage_num": 1,
                "model": {
                    "model_type": "mlp_policy",
                    "num_action_chunks": 1,
                    "policy_setup": "panda-ee-dpos",
                    "action_dim": 4,
                },
            },
            "env": {
                "eval": {
                    "env_type": "maniskill",
                    "total_num_envs": 8,
                    "group_size": 1,
                    "max_steps_per_rollout_epoch": 8,
                    "init_params": {},
                }
            },
        }
    )


def test_embodied_eval_config_without_algorithm_validates():
    cfg = _make_embodied_eval_sglang_reward_cfg()

    validated_cfg = validate_embodied_cfg(cfg)

    assert "algorithm" not in validated_cfg
    assert validated_cfg.runner.only_eval is True
    assert validated_cfg.reward.worker_type == "api"
    assert validated_cfg.reward.api.api_base == "http://reward-api:30000"
    assert validated_cfg.env.eval.init_params.control_mode == "pd_ee_delta_pos"


def test_embodied_eval_config_rejects_removed_sglang_public_args():
    cfg = _make_embodied_eval_sglang_reward_cfg()
    cfg.reward.model.sglang_server_args = {"tp_size": 2}

    with pytest.raises(
        AssertionError,
        match="reward.model.sglang_server_args is no longer supported",
    ):
        validate_embodied_cfg(cfg)


def test_embodied_eval_config_rejects_removed_sglang_engine_args():
    cfg = _make_embodied_eval_sglang_reward_cfg()
    cfg.reward.model.sglang_engine_args = {"max_running_requests": 32}

    with pytest.raises(
        AssertionError,
        match="reward.model.sglang_engine_args is no longer supported",
    ):
        validate_embodied_cfg(cfg)


def test_embodied_eval_config_rejects_sglang_inference_backend():
    cfg = _make_embodied_eval_sglang_reward_cfg()
    cfg.reward.model.inference_backend = "sglang"

    with pytest.raises(
        AssertionError,
        match="reward.model.inference_backend='sglang' is no longer supported",
    ):
        validate_embodied_cfg(cfg)
