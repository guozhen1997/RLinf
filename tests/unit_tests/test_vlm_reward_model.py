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

# ruff: noqa: E402

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]

if "peft" not in sys.modules:
    peft_stub = types.ModuleType("peft")

    class _FakeLoraConfig:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    peft_stub.LoraConfig = _FakeLoraConfig
    peft_stub.get_peft_model = lambda model, *_args, **_kwargs: model
    peft_stub.set_peft_model_state_dict = lambda *_args, **_kwargs: None
    sys.modules["peft"] = peft_stub

from examples.embodiment.train_embodied_agent import (
    launch_managed_sglang_reward_api,
)
from rlinf.models.embodiment.reward import (
    get_reward_model_class,
    resolve_reward_model_backend,
    reward_model_registry,
)
from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
)
from rlinf.workers.reward.api_reward_worker import EmbodiedAPIRewardWorker


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def generate(
        self, input_ids: torch.Tensor, reward_ids: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del kwargs
        return torch.cat([input_ids, reward_ids.unsqueeze(-1)], dim=-1)


class _FakeProcessor:
    def batch_decode(self, output_ids: torch.Tensor, skip_special_tokens: bool = True):
        del skip_special_tokens
        return [str(int(token.item())) for token in output_ids[:, 0]]


class _FakeSGLangProcessor:
    video_token = "<video>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return "rendered"


class _FakeRewardParser:
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        return torch.tensor([float(output) if output else 0.0 for output in outputs])


class _FakeHistoryInputBuilder(HistoryVLMInputBuilder):
    def __init__(self, history_buffer_names: list[str]):
        super().__init__(
            _processor=None,
            history_buffer_names=history_buffer_names,
        )
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
    ) -> dict[str, torch.Tensor]:
        del history_input
        reward_ids = observations["slot_ids"][valid_input_ids].to(dtype=torch.long)
        self.calls.append(reward_ids.tolist())
        return {
            "input_ids": torch.zeros((len(valid_input_ids), 1), dtype=torch.long),
            "reward_ids": reward_ids,
        }

    def process_inputs(
        self, prepared_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return prepared_inputs


class _FakeSGLangHistoryInputBuilder:
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


class _TestHistoryVLMRewardModel(HistoryVLMRewardModel):
    def setup_processor(self) -> None:
        self._processor = _FakeProcessor()

    def setup_model(self) -> None:
        self._model = _FakeModel()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeHistoryInputBuilder(
            history_buffer_names=self.history_buffer_names
        )

    def setup_reward_parser(self) -> None:
        self.reward_parser = _FakeRewardParser()


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
        self._processor = _FakeSGLangProcessor()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeSGLangHistoryInputBuilder()

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
        self.registered_urls = []
        self.registration_kwargs = []
        self.shutdown_calls = 0

    def init_server(self):
        return _FakeHandle()

    def init_router(self):
        return _FakeHandle()

    def get_server_url(self):
        return _FakeHandle(["http://server-0:31000"])

    def get_router_url(self):
        return _FakeHandle(["http://router:32000"])

    def register_server(self, server_url, **kwargs):
        self.registered_urls.append(server_url)
        self.registration_kwargs.append(kwargs)
        return _FakeHandle()

    def shutdown(self):
        self.shutdown_calls += 1
        return _FakeHandle()


class _FakeComponentPlacement:
    def get_hardware_ranks(self, component_name):
        assert component_name == "reward_server"
        return [0, 1]

    def get_world_size(self, component_name):
        assert component_name == "reward_server"
        return 1

    def get_strategy(self, component_name):
        assert component_name == "reward_server"
        return "reward-server-placement"


def _make_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "precision": "bf16",
            "infer_micro_batch_size": 2,
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


def test_history_vlm_backend_contracts_and_yaml_defaults():
    hf_cfg = OmegaConf.load(
        REPO_ROOT / "examples/embodiment/config/maniskill_ppo_mlp_qwentrend_reward.yaml"
    )
    sglang_cfg = OmegaConf.load(
        REPO_ROOT
        / "examples/embodiment/config/maniskill_ppo_mlp_qwentrend_sglang_reward.yaml"
    )

    assert "history_vlm_sglang" not in reward_model_registry
    assert resolve_reward_model_backend("history_vlm") == ("history_vlm", None)
    assert resolve_reward_model_backend("history_vlm", "hf") == ("history_vlm", "hf")
    assert resolve_reward_model_backend("history_vlm", "transformers") == (
        "history_vlm",
        "hf",
    )
    assert get_reward_model_class("history_vlm").__name__ == "HistoryVLMRewardModel"
    with pytest.raises(ValueError, match="Unsupported reward.model.inference_backend"):
        resolve_reward_model_backend("history_vlm", "sglang")
    with pytest.raises(ValueError, match="Unsupported reward.model.inference_backend"):
        resolve_reward_model_backend("history_vlm", "vllm")

    assert hf_cfg.reward.model.get("inference_backend") in (None, "hf")
    assert sglang_cfg.reward.worker_type == "api"
    assert sglang_cfg.reward.api.model == "Qwen3-VL-4B-Instruct"
    assert sglang_cfg.cluster.component_placement.reward_server == "0-1:0"
    assert "inference_backend" not in sglang_cfg.reward.model
    assert "sglang_server_args" not in sglang_cfg.reward.model
    assert "sglang_router_args" not in sglang_cfg.reward.model
    assert "sglang_engine_args" not in sglang_cfg.reward.model
    assert sglang_cfg.router_server_args.server.trust_remote_code is True
    assert "max_running_requests" not in sglang_cfg.router_server_args.server


def test_history_vlm_transformers_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMRewardModel(_make_cfg())

    rewards = model.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21], [23]]


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


def test_history_vlm_api_worker_runtime_base_url_and_response_contracts():
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


def test_managed_sglang_reward_api_launches_router_and_injects_runtime_api_base(
    monkeypatch,
):
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
                "group_name": "SGLangRewardServerGroup",
                "launch_server": True,
                "server": {
                    "model_path": "/models/QwenTrend",
                    "tp_size": 2,
                    "pp_size": 1,
                    "enable_multimodal": True,
                    "max_running_requests": 16,
                },
                "router_group_name": "SGLangRewardRouterGroup",
                "launch_router": True,
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

    stack = launch_managed_sglang_reward_api(
        cfg,
        cluster=object(),
        component_placement=_FakeComponentPlacement(),
    )

    assert launch_calls[0]["router_server_args"] is cfg.router_server_args
    assert launch_calls[0]["placement_strategy"] == "reward-server-placement"
    assert cfg.reward.api._runtime_api_base == "http://router:32000"
    assert "api_base" not in cfg.reward.api or cfg.reward.api.api_base is None
    assert "_runtime_" + "sglang_api_base" not in cfg.reward.model

    server_group, router_group = stack
    assert router_group.shutdown_calls == 0
    assert server_group.shutdown_calls == 0
