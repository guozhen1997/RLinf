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

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from transformers import AutoProcessor

from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
    get_input_builder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    get_reward_parser,
)
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.utils.logging import get_logger
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

logger = get_logger()

_DEFAULT_REQUEST_TIMEOUT = 120.0
_IMAGE_FORMAT = "jpeg"
_JPEG_QUALITY = 95


class APIRewardTimingRecorder:
    """Records timing metrics for one API-backed reward inference call."""

    _BASE_KEYS = (
        "prepare_inputs_ms",
        "image_encode_ms",
        "http_request_ms",
        "parse_ms",
        "total_ms",
    )

    def __init__(self) -> None:
        self.timings = dict.fromkeys(self._BASE_KEYS, 0.0)
        self._total_start = 0.0

    def __enter__(self):
        self._total_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        self.timings["total_ms"] = (time.perf_counter() - self._total_start) * 1000

    @contextmanager
    def record(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] += (time.perf_counter() - start) * 1000

    def metrics(self) -> dict[str, float]:
        return {
            **self.timings,
            "media_convert_ms": self.timings["image_encode_ms"],
            "api_request_ms": self.timings["http_request_ms"],
            "generate_ms": self.timings["http_request_ms"],
        }


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


def _derive_model_name(model_path: Any) -> str:
    name = str(model_path).rstrip("/").split("/")[-1]
    return name or "history_vlm_reward"


class EmbodiedAPIRewardWorker(EmbodiedRewardWorker):
    """Embodied reward worker that calls an OpenAI-compatible reward API."""

    def init_worker(self):
        """Initialize API reward inference without loading a local reward model."""
        if self._standalone_realworld:
            raise ValueError(
                "standalone_realworld reward workers do not support API reward "
                "inference."
            )

        assert self.train_batch_size % self._world_size == 0, (
            f"train_batch_size ({self.train_batch_size}) must be divisible by "
            f"world_size ({self._world_size})."
        )
        self.local_num_train_envs = self.train_batch_size // self._world_size

        self.model_cfg = self.cfg.reward.model
        self.api_cfg = self.cfg.reward.get("api", {})
        if self.model_cfg.get("model_type") != "history_vlm":
            raise ValueError(
                "EmbodiedAPIRewardWorker currently supports only "
                "reward.model.model_type='history_vlm'."
            )

        self.setup_api_reward()

    def setup_api_reward(self) -> None:
        self.request_timeout = _DEFAULT_REQUEST_TIMEOUT
        self.image_format = _IMAGE_FORMAT
        self.jpeg_quality = _JPEG_QUALITY

        self.model_path = self.model_cfg.get("model_path")
        if not self.model_path:
            raise ValueError(
                "reward.model.model_path must be set for API reward inference."
            )

        self.api_base = str(self.api_cfg.get("api_base") or "").rstrip("/")
        if not self.api_base:
            raise ValueError(
                "reward.api.api_base must be set for API reward inference. "
                "When using Ray-managed SGLang serving, the entrypoint injects "
                "reward.api.api_base at runtime."
            )
        client_base_url = (
            self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
        )
        self.http_client = InferenceHTTPClient(
            client_base_url,
            connect_timeout=self.request_timeout,
        )

        self.history_buffer_names = list(self.model_cfg.history_buffers.keys())
        self.interval_reward = float(self.model_cfg.get("interval_reward", 0.0))
        self.gt_success_bonus = float(self.model_cfg.get("gt_success_bonus", 0.0))
        self.model_name = str(
            self.api_cfg.get("model") or _derive_model_name(self.model_path)
        )
        self.sampling_params = self._build_sampling_params(
            self.api_cfg,
        )

        self.last_timing_ms: dict[str, float] = {}
        self.last_generation_stats: dict[str, float] = {}
        self.last_outputs: list[str] = []

        self.setup_processor()
        self.setup_input_builder()
        self.setup_reward_parser()

    def setup_processor(self) -> None:
        self._processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        subprocessor_kwargs = self.model_cfg.get("subprocessor_kwargs", {})
        for subprocessor_name, kwargs in subprocessor_kwargs.items():
            subprocessor = getattr(self._processor, subprocessor_name, None)
            if subprocessor is None:
                continue
            for key, value in dict(kwargs).items():
                if hasattr(subprocessor, key):
                    setattr(subprocessor, key, value)

    def setup_input_builder(self) -> None:
        self.input_builder = get_input_builder(
            self.model_cfg.get("input_builder_name", "history_vlm_input_builder")
        )(
            **self.model_cfg.get("input_builder_params", {}),
            _processor=self._processor,
            history_buffer_names=self.history_buffer_names,
        )
        assert isinstance(self.input_builder, HistoryVLMInputBuilder), (
            "EmbodiedAPIRewardWorker only supports HistoryVLMInputBuilder."
        )

    def setup_reward_parser(self) -> None:
        self.reward_parser = get_reward_parser(
            self.model_cfg.get("reward_parser_name", "base_reward_parser")
        )(**self.model_cfg.get("reward_parser_params", {}))

    def _build_sampling_params(
        self,
        api_cfg: DictConfig,
    ) -> dict[str, Any]:
        return _to_plain_dict(api_cfg.get("sampling_params", {}))

    def _frame_to_numpy(self, frame: Any) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        elif isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))
        else:
            frame = np.asarray(frame)

        if (
            frame.ndim == 3
            and frame.shape[0] in (1, 3, 4)
            and frame.shape[-1] not in (1, 3, 4)
        ):
            frame = np.moveaxis(frame, 0, -1)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(frame[..., :3])

    def _frame_to_data_url(self, frame: Any) -> str:
        image = Image.fromarray(self._frame_to_numpy(frame), mode="RGB")
        image_buffer = io.BytesIO()
        if self.image_format == "jpeg":
            image.save(image_buffer, format="JPEG", quality=self.jpeg_quality)
            mime_type = "image/jpeg"
        else:
            image.save(image_buffer, format="PNG")
            mime_type = "image/png"
        encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _build_content_items(
        self,
        prompt_texts: list[str],
        videos: list[list[Any]],
    ) -> list[dict[str, Any]]:
        prompt_text = prompt_texts[0] if prompt_texts else ""
        content: list[dict[str, Any]] = []
        for video in videos:
            for frame in video:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._frame_to_data_url(frame)},
                    }
                )
        content.append({"type": "text", "text": prompt_text})
        return content

    def _build_chat_payloads(
        self,
        prepared_inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        prompt_texts_list = prepared_inputs.get("prompt_texts_list") or []
        videos_list = prepared_inputs.get("videos_list") or []

        payloads: list[dict[str, Any]] = []
        for prompt_texts, videos in zip(prompt_texts_list, videos_list):
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": self._build_content_items(prompt_texts, videos),
                    }
                ],
            }
            payload.update(self.sampling_params)
            payloads.append(payload)
        return payloads

    def _extract_text_and_token_count(
        self, response: dict[str, Any]
    ) -> tuple[str, int]:
        choices = response.get("choices") or []
        text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            if isinstance(content, list):
                text = "".join(str(item.get("text", "")) for item in content)
            else:
                text = str(content)

        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = usage.get("output_tokens")
        return text, int(completion_tokens or 0)

    @staticmethod
    def _summarize_generation_stats(completion_tokens: list[int]) -> dict[str, float]:
        completion_tokens = [count for count in completion_tokens if count > 0]
        if not completion_tokens:
            return {}
        return {
            "generated_tokens_mean": float(np.mean(completion_tokens)),
            "generated_tokens_min": float(min(completion_tokens)),
            "generated_tokens_max": float(max(completion_tokens)),
        }

    def _chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload)
        messages = payload.pop("messages")
        model = payload.pop("model")
        return self.http_client.chat_completion(
            messages=messages,
            model=model,
            **payload,
        )

    def _generate(self, payloads: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
        if not payloads:
            return [], []

        def _generate_one(payload: dict[str, Any]) -> tuple[str, int]:
            text, token_count = self._extract_text_and_token_count(
                self._chat_completion(payload)
            )
            return text, token_count

        with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
            results = list(executor.map(_generate_one, payloads))

        outputs = [text for text, _ in results]
        completion_tokens = [token_count for _, token_count in results]
        return outputs, completion_tokens

    def _set_timing_metrics(
        self,
        timing_recorder: APIRewardTimingRecorder,
        generation_stats: dict[str, float],
    ) -> None:
        timings = timing_recorder.metrics()
        self.last_timing_ms = timings
        self.last_generation_stats = dict(generation_stats)
        logger.debug("%s timing_ms=%s", self.__class__.__name__, self.last_timing_ms)

    def apply_gt_success_bonus(
        self, rewards: torch.Tensor, reward_input: dict[str, Any]
    ) -> torch.Tensor:
        if rewards is None or self.gt_success_bonus == 0.0:
            return rewards
        env_infos = (
            reward_input.get("env_infos") if isinstance(reward_input, dict) else None
        )
        if not isinstance(env_infos, dict):
            return rewards

        success = None
        final_info = env_infos.get("final_info", {})
        for info_dict in (
            env_infos,
            env_infos.get("episode"),
            final_info,
            final_info.get("episode") if isinstance(final_info, dict) else None,
        ):
            if not isinstance(info_dict, dict):
                continue
            for key in ("success", "success_at_end", "success_once"):
                value = info_dict.get(key)
                if value is not None:
                    success = torch.as_tensor(value).reshape(-1).bool()
                    break
            if success is not None:
                break

        if success is None or success.shape[0] != rewards.shape[0]:
            return rewards
        bonus = success.to(device=rewards.device, dtype=rewards.dtype)
        return rewards + (bonus * self.gt_success_bonus).view(
            -1, *([1] * (rewards.dim() - 1))
        )

    @torch.no_grad()
    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        with APIRewardTimingRecorder() as timing_recorder:
            history_input: dict[str, dict[str, list[list[Any]]]] = reward_input[
                "history_input"
            ]
            input_batch_size = len(
                next(iter(next(iter(history_input.values())).values()))
            )
            observations = {
                key: value
                for key, value in reward_input.items()
                if key != "history_input"
            }

            all_outputs: list[str] = []
            generated_token_counts: list[int] = []
            rewards = torch.full(
                (input_batch_size,),
                fill_value=self.interval_reward,
                dtype=torch.float32,
            )

            valid_input_ids = self.input_builder.get_valid_input_ids(
                observations,
                history_input,
            )
            if len(valid_input_ids) > 0:
                with timing_recorder.record("prepare_inputs_ms"):
                    prepared_inputs = self.input_builder.prepare_inputs(
                        observations,
                        history_input,
                        valid_input_ids,
                    )

                with timing_recorder.record("image_encode_ms"):
                    payloads = self._build_chat_payloads(prepared_inputs)

                with timing_recorder.record("http_request_ms"):
                    outputs, token_counts = self._generate(payloads)
                generated_token_counts.extend(token_counts)
                all_outputs.extend(outputs)

                if len(outputs) != len(valid_input_ids):
                    logger.warning(
                        "API reward output count mismatch: outputs=%d valid_inputs=%d",
                        len(outputs),
                        len(valid_input_ids),
                    )
                    outputs += [""] * (len(valid_input_ids) - len(outputs))
                    outputs = outputs[: len(valid_input_ids)]

                with timing_recorder.record("parse_ms"):
                    parsed_rewards = self.reward_parser.parse_rewards(outputs).to(
                        dtype=torch.float32
                    )

                rewards[valid_input_ids] = parsed_rewards

        self.last_outputs = all_outputs
        self._set_timing_metrics(
            timing_recorder,
            self._summarize_generation_stats(generated_token_counts),
        )
        return self.apply_gt_success_bonus(rewards, observations)

    @torch.no_grad()
    def compute_image_rewards(
        self, observations: dict[str, Any]
    ) -> torch.Tensor | np.ndarray:
        rewards = self.compute_reward(observations)
        if rewards is not None and rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        return rewards.detach().cpu()
