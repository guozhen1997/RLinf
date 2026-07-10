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

import sys
import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from examples.embodiment import train_embodied_agent
from rlinf.workers.reward.api_reward_worker import EmbodiedAPIRewardWorker


class _FakeProcessor:
    video_token = "<video>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del messages, tokenize, add_generation_prompt
        return "rendered"


class _FakeRewardParser:
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        return torch.tensor([float(output) if output else 0.0 for output in outputs])


class _FakeHistoryInputBuilder:
    def __init__(self):
        self.calls: list[list[int]] = []

    def get_valid_input_ids(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
    ) -> list[int]:
        del observations
        history_window = history_input["history_window"]["main_images"]
        return [
            env_idx for env_idx, frames in enumerate(history_window) if len(frames) > 0
        ]

    def prepare_inputs(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
        valid_input_ids: list[int],
    ) -> dict[str, object]:
        del history_input
        slot_ids = observations["slot_ids"][valid_input_ids].tolist()
        self.calls.append(slot_ids)
        return {
            "prompt_texts_list": [[f"prompt-{slot_id}"] for slot_id in slot_ids],
            "videos_list": [
                [
                    [torch.zeros((1, 1, 3), dtype=torch.uint8)],
                    [torch.ones((1, 1, 3), dtype=torch.uint8)],
                ]
                for _ in slot_ids
            ],
        }


class _TestEmbodiedAPIRewardWorker(EmbodiedAPIRewardWorker):
    def __init__(self, cfg, api_cfg=None):
        self.model_cfg = cfg
        if api_cfg is None:
            api_cfg = OmegaConf.create(
                {
                    "api_base": "http://router:30000",
                    "model": "reward-model",
                    "sampling_params": {"max_tokens": 32, "temperature": 0.0},
                }
            )
        self.api_cfg = api_cfg
        self.setup_api_reward()

    def setup_processor(self) -> None:
        self._processor = _FakeProcessor()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeHistoryInputBuilder()

    def setup_reward_parser(self) -> None:
        self.reward_parser = _FakeRewardParser()

    def _generate(
        self, payloads: list[dict[str, object]]
    ) -> tuple[list[str], list[int]]:
        outputs = []
        for payload in payloads:
            content = payload["messages"][0]["content"]
            outputs.append(content[-1]["text"].removeprefix("prompt-"))
        return outputs, [1] * len(payloads)


class _FakeHandle:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value


class _FakeWorkerGroup:
    def __init__(self):
        self.shutdown_calls = 0

    def get_server_url(self):
        return _FakeHandle(["http://server-0:31000"])

    def get_router_url(self):
        return _FakeHandle(["http://router:32000"])

    def shutdown(self):
        self.shutdown_calls += 1
        return _FakeHandle()


class _FakeComponentPlacement:
    def get_strategy(self, component_name):
        assert component_name == "reward_server"
        return "reward-server-placement"


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "precision": "bf16",
            "input_builder_name": "history_vlm_input_builder",
            "reward_parser_name": "base_reward_parser",
            "history_buffers": {
                "history_window": {
                    "history_size": 10,
                    "input_interval": 10,
                    "history_keys": ["main_images"],
                    "input_on_done": False,
                }
            },
        }
    )


def _make_reward_input(
    slot_ids: list[int], valid_env_ids: list[int] | None = None
) -> dict[str, object]:
    valid_env_ids = valid_env_ids or list(range(len(slot_ids)))
    valid_env_id_set = set(valid_env_ids)
    return {
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "main_images": torch.zeros((len(slot_ids), 1, 1, 1), dtype=torch.uint8),
        "history_input": {
            "history_window": {
                "main_images": [
                    [f"frame-{slot_id}"] if env_idx in valid_env_id_set else []
                    for env_idx, slot_id in enumerate(slot_ids)
                ]
            }
        },
    }


def test_history_vlm_api_worker_writes_sparse_valid_envs_back_to_slots():
    worker = _TestEmbodiedAPIRewardWorker(_make_cfg())

    rewards = worker.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert worker.input_builder.calls == [[21, 23]]


def test_history_vlm_api_worker_builds_openai_payload_with_images():
    worker = object.__new__(EmbodiedAPIRewardWorker)
    worker.model_name = "reward-model"
    worker.sampling_params = {"max_tokens": 16, "temperature": 0.0}
    worker.image_format = "jpeg"
    worker.jpeg_quality = 95
    chw_frame = torch.zeros((3, 2, 2), dtype=torch.uint8)
    pil_frame = Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8))

    payloads = worker._build_chat_payloads(
        {
            "prompt_texts_list": [["judge progress"]],
            "videos_list": [[[chw_frame], [pil_frame]]],
        }
    )

    assert payloads[0]["model"] == "reward-model"
    assert "lora_path" not in payloads[0]
    assert payloads[0]["max_tokens"] == 16
    content = payloads[0]["messages"][0]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert content[-1] == {"type": "text", "text": "judge progress"}


def test_history_vlm_api_worker_base_url_and_response_contracts():
    with pytest.raises(ValueError, match="reward.api.api_base must be set"):
        _TestEmbodiedAPIRewardWorker(
            _make_cfg(),
            api_cfg=OmegaConf.create({}),
        )

    worker = object.__new__(EmbodiedAPIRewardWorker)
    text, token_count = worker._extract_text_and_token_count(
        {
            "choices": [{"message": {"content": "positive"}}],
            "usage": {"completion_tokens": 3},
        }
    )
    assert text == "positive"
    assert token_count == 3


def test_managed_sglang_reward_api_launches_router_and_sets_api_base(monkeypatch):
    cfg = OmegaConf.create(
        {
            "reward": {
                "use_reward_model": True,
                "worker_type": "api",
                "api": {"api_base": None},
                "model": {
                    "model_path": "/models/QwenTrend",
                    "model_type": "history_vlm",
                },
            },
            "router_server_args": {
                "model_path": "/models/QwenTrend",
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "server": {
                    "model_path": "/models/QwenTrend",
                    "tp_size": 2,
                    "pp_size": 1,
                    "enable_multimodal": True,
                },
                "router": {"policy": "cache_aware"},
            },
        }
    )
    launch_calls = []

    def _fake_launch_sglang_router_and_server(**kwargs):
        launch_calls.append(kwargs)
        return _FakeWorkerGroup(), _FakeWorkerGroup()

    monkeypatch.setitem(
        sys.modules,
        "rlinf.workers.rollout.sglang_server",
        types.SimpleNamespace(
            launch_sglang_router_and_server=_fake_launch_sglang_router_and_server
        ),
    )

    stack = train_embodied_agent.launch_managed_sglang_reward_api(
        cfg,
        cluster=object(),
        component_placement=_FakeComponentPlacement(),
    )

    assert launch_calls[0]["router_server_args"] is cfg.router_server_args
    assert launch_calls[0]["placement_strategy"] == "reward-server-placement"
    assert cfg.reward.api.api_base == "http://router:32000"

    server_group, router_group = stack
    assert router_group.shutdown_calls == 0
    assert server_group.shutdown_calls == 0
