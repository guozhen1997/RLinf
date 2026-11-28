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

"""General-purpose multimodal SFT dataset for Qwen3.8 / Qwen-VL models.

Supports single-turn and multi-turn multimodal conversations stored in JSON,
JSONL, or Parquet files across three common schemas:

1. OpenAI-style ``messages`` (with ``system``/``user``/``assistant``, optional
   ``reasoning_content``, inline ``{"type": "image", ...}`` items or ``<image>``
   placeholders paired with top-level ``images``/``image``).
2. ShareGPT-style ``conversations`` (``from``/``value`` + ``images``/``image``).
3. Simple QA pairs (``prompt``/``question`` + ``response``/``answer`` +
   optional ``choices``, ``reasoning_content``, ``image``/``images``).
"""

from __future__ import annotations

import ast
import inspect
import os
import re
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry
from rlinf.utils.logging import get_logger

logger = get_logger()

_SHAREGPT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "observation": "tool",
    "tool": "tool",
}


def _extract_image_entries(value: Any) -> list[Any]:
    """Flatten a single image entry, list, tuple, or ndarray into raw entries."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_extract_image_entries(item))
        return out
    if isinstance(value, np.ndarray):
        return _extract_image_entries(value.tolist())
    return [value]


def _resolve_image_path(
    path: str,
    image_root: Optional[str],
    record_dir: Optional[str],
) -> str:
    """Resolve an image path against explicit ``image_root`` or the record file's dir."""
    if os.path.isfile(path):
        return path
    for base in (image_root, record_dir):
        if base:
            candidate = os.path.join(base, path)
            if os.path.isfile(candidate):
                return candidate
    return path


def _load_pil_image(
    image: Any,
    *,
    image_root: Optional[str] = None,
    record_dir: Optional[str] = None,
) -> Image.Image:
    """Convert PIL Image, bytes, HF parquet dict, or file path into an RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(image))) as img:
            return img.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return _load_pil_image(
                image["bytes"], image_root=image_root, record_dir=record_dir
            )
        if image.get("path") is not None:
            return _load_pil_image(
                image["path"], image_root=image_root, record_dir=record_dir
            )
        if image.get("url") is not None:
            return _load_pil_image(
                image["url"], image_root=image_root, record_dir=record_dir
            )
    if isinstance(image, str):
        resolved = _resolve_image_path(image, image_root, record_dir)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(
                f"Multimodal SFT image file not found: {image!r} "
                f"(resolved: {resolved!r}, image_root={image_root!r})"
            )
        with Image.open(resolved) as img:
            return img.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            if float(arr.max()) <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise ValueError(
        f"Unsupported image type {type(image).__name__} in MultimodalSFTDataset."
    )


def _split_text_with_image_placeholders(
    text: str,
    image_queue: list[Image.Image],
    *,
    prepend_remaining_if_no_placeholder: bool = False,
) -> list[dict[str, Any]]:
    """Build HF multimodal content items from text containing optional ``<image>`` tags."""
    parts = re.split(r"<image>", text)
    content: list[dict[str, Any]] = []
    if len(parts) == 1:
        if prepend_remaining_if_no_placeholder:
            while image_queue:
                content.append({"type": "image", "image": image_queue.pop(0)})
        if text:
            content.append({"type": "text", "text": text})
        return content

    for idx, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if idx < len(parts) - 1:
            if not image_queue:
                raise ValueError(
                    "Found more <image> placeholders than provided images in record."
                )
            content.append({"type": "image", "image": image_queue.pop(0)})
    return content


def _extract_thinking_from_text(text: str) -> tuple[str, Optional[str]]:
    """Split leading ``<think>...</think>`` into ``(content, reasoning_content)``."""
    match = re.match(
        r"^\s*<think>(.*?)</think>\s*(.*)$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(2), match.group(1).strip()
    return text, None


def _assistant_text_from_message(msg: dict[str, Any]) -> str:
    """Extract plain assistant text from a normalized message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "".join(parts)
    return str(content)


def _supports_kwarg(fn: Any, kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    if kwarg in sig.parameters:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


@VLMDatasetRegistry.register("multimodal_sft")
class MultimodalSFTDataset(VLMBaseDataset):
    """Flexible multimodal SFT dataset supporting Qwen3.8 / Qwen3-VL / Qwen2.5-VL."""

    def __init__(
        self,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
        eval_dataset: bool = False,
    ) -> None:
        self.messages_key = config.data.get("messages_key", "messages")
        self.conversations_key = config.data.get("conversations_key", "conversations")
        self.response_key = config.data.get("response_key", "response")
        self.reasoning_key = config.data.get("reasoning_key", "reasoning_content")
        self.image_root = config.data.get("image_root") or config.data.get("data_root")
        self.enable_thinking = config.data.get("enable_thinking", None)
        self.reasoning_effort = config.data.get("reasoning_effort", None)
        self.max_seq_length = config.data.get("max_seq_length", None)
        if self.max_seq_length is not None:
            self.max_seq_length = int(self.max_seq_length)
        self.eval_mode = str(config.actor.get("eval_mode", "generate")).lower()
        self.trust_remote_code = bool(
            config.actor.model.get("trust_remote_code", True)
        )
        super().__init__(data_paths, config, tokenizer, eval_dataset)
        if self.max_seq_length is None:
            self.max_seq_length = self.max_prompt_length

    def __getitem__(self, idx: int) -> SftDatasetItem:
        record_dir: Optional[str] = None
        if self.lazy_loading:
            path, fmt, key = self._indices[idx]
            record_dir = os.path.dirname(path) if path else None
            raw = self._load_single_lazy(path, fmt, key)
        else:
            raw = self._records[idx]
            if len(self.data_paths) == 1:
                record_dir = os.path.dirname(self.data_paths[0])
        return self._process_raw_record(raw, idx, record_dir=record_dir)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.actor.model.model_path,
                trust_remote_code=self.trust_remote_code,
            )
        return self._processor

    def _chat_template_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = bool(self.enable_thinking)
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(self.reasoning_effort)
        return kwargs

    def _collect_top_level_images(
        self, raw: dict[str, Any], record_dir: Optional[str]
    ) -> list[Image.Image]:
        raw_images: list[Any] = []
        keys_to_check: list[str] = []
        for k in list(self.image_keys) + ["images", "image"]:
            if k not in keys_to_check:
                keys_to_check.append(k)
        for key in keys_to_check:
            if key in raw and raw[key] is not None:
                raw_images.extend(_extract_image_entries(raw[key]))
                if raw_images:
                    break
        return [
            _load_pil_image(img, image_root=self.image_root, record_dir=record_dir)
            for img in raw_images
            if img is not None
        ]

    def _normalize_messages(
        self, raw: dict[str, Any], record_dir: Optional[str]
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Normalize any supported record schema into HF ``messages`` + PIL images."""
        top_images = self._collect_top_level_images(raw, record_dir)
        image_queue = list(top_images)
        ordered_images: list[Image.Image] = []
        messages: list[dict[str, Any]] = []

        if self.messages_key in raw and raw[self.messages_key] is not None:
            raw_msgs = raw[self.messages_key]
            if isinstance(raw_msgs, np.ndarray):
                raw_msgs = raw_msgs.tolist()
            has_inline_placeholder = any(
                (
                    isinstance(m.get("content"), str)
                    and "<image>" in m.get("content", "")
                )
                or (
                    isinstance(m.get("content"), (list, tuple, np.ndarray))
                    and any(
                        isinstance(item, dict)
                        and (
                            item.get("type") == "image"
                            or "image" in item
                            or "image_url" in item
                        )
                        for item in m.get("content", [])
                    )
                )
                for m in raw_msgs
            )
            first_user_seen = False
            for msg in raw_msgs:
                role = str(msg.get("role", "")).lower()
                content = msg.get("content", "")
                norm_msg: dict[str, Any] = {"role": role}
                if role == "assistant":
                    reasoning = msg.get(self.reasoning_key)
                    if isinstance(content, str):
                        if reasoning is None:
                            content, extracted = _extract_thinking_from_text(content)
                            if extracted is not None:
                                reasoning = extracted
                        norm_msg["content"] = content
                    elif isinstance(content, (list, tuple, np.ndarray)):
                        norm_msg["content"] = list(content)
                    else:
                        norm_msg["content"] = str(content)
                    if reasoning is not None:
                        norm_msg["reasoning_content"] = str(reasoning)
                    messages.append(norm_msg)
                    continue

                if isinstance(content, str):
                    prepend = (
                        role == "user"
                        and not first_user_seen
                        and not has_inline_placeholder
                        and bool(image_queue)
                    )
                    if role == "user":
                        first_user_seen = True
                    items = _split_text_with_image_placeholders(
                        content,
                        image_queue,
                        prepend_remaining_if_no_placeholder=prepend,
                    )
                    for item in items:
                        if item.get("type") == "image":
                            ordered_images.append(item["image"])
                    norm_msg["content"] = items if (
                        role == "user" and any(it.get("type") == "image" for it in items)
                    ) else content
                    messages.append(norm_msg)
                elif isinstance(content, (list, tuple, np.ndarray)):
                    if role == "user":
                        first_user_seen = True
                    items = []
                    for item in content:
                        if isinstance(item, dict) and (
                            item.get("type") == "image"
                            or "image" in item
                            or "image_url" in item
                        ):
                            raw_img = item.get("image", item.get("image_url"))
                            if raw_img is None:
                                if not image_queue:
                                    raise ValueError(
                                        "Image content item has no inline image and no remaining top-level images."
                                    )
                                pil_img = image_queue.pop(0)
                            else:
                                pil_img = _load_pil_image(
                                    raw_img,
                                    image_root=self.image_root,
                                    record_dir=record_dir,
                                )
                            ordered_images.append(pil_img)
                            items.append({"type": "image", "image": pil_img})
                        elif isinstance(item, dict) and "text" in item:
                            items.append({"type": "text", "text": str(item["text"])})
                        else:
                            items.append(item)
                    norm_msg["content"] = items
                    messages.append(norm_msg)
                else:
                    norm_msg["content"] = str(content)
                    messages.append(norm_msg)

        elif self.conversations_key in raw and raw[self.conversations_key] is not None:
            convs = raw[self.conversations_key]
            if isinstance(convs, np.ndarray):
                convs = convs.tolist()
            has_placeholder = any(
                "<image>" in str(turn.get("value", turn.get("content", "")))
                for turn in convs
            )
            first_user_seen = False
            for turn in convs:
                raw_role = str(turn.get("from", turn.get("role", ""))).lower()
                role = _SHAREGPT_ROLE_MAP.get(raw_role, raw_role)
                text = str(turn.get("value", turn.get("content", "")))
                if role == "assistant":
                    reasoning = turn.get(self.reasoning_key)
                    if reasoning is None:
                        text, reasoning = _extract_thinking_from_text(text)
                    msg_dict: dict[str, Any] = {"role": "assistant", "content": text}
                    if reasoning is not None:
                        msg_dict["reasoning_content"] = str(reasoning)
                    messages.append(msg_dict)
                elif role == "user":
                    prepend = (
                        not first_user_seen
                        and not has_placeholder
                        and bool(image_queue)
                    )
                    first_user_seen = True
                    items = _split_text_with_image_placeholders(
                        text,
                        image_queue,
                        prepend_remaining_if_no_placeholder=prepend,
                    )
                    for item in items:
                        if item.get("type") == "image":
                            ordered_images.append(item["image"])
                    messages.append(
                        {
                            "role": "user",
                            "content": items
                            if any(it.get("type") == "image" for it in items)
                            else text,
                        }
                    )
                else:
                    messages.append({"role": role, "content": text})

        else:
            # Simple QA schema: prompt/question (+ optional choices) -> response/answer
            prompt_text = raw.get(self.prompt_key)
            if prompt_text is None:
                prompt_text = raw.get("prompt", raw.get("question"))
            if prompt_text is None:
                raise KeyError(
                    f"Record missing {self.messages_key!r}, {self.conversations_key!r}, "
                    f"and {self.prompt_key!r}."
                )
            prompt_text = str(prompt_text)

            choices = raw.get(self.choice_key) if self.choice_key else None
            if choices is not None:
                if isinstance(choices, str):
                    try:
                        choices = ast.literal_eval(choices)
                    except (ValueError, SyntaxError):
                        choices = [choices]
                if isinstance(choices, np.ndarray):
                    choices = choices.tolist()
                if isinstance(choices, (list, tuple)) and choices:
                    prompt_text = f"{prompt_text}\nChoices:\n" + "\n".join(
                        f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                    )

            response = raw.get(self.response_key)
            if response is None and self.answer_key:
                response = raw.get(self.answer_key)
            if response is None:
                response = raw.get("answer", raw.get("response"))
            if response is None:
                raise KeyError(
                    f"Record missing target response ({self.response_key!r} / {self.answer_key!r})."
                )
            if (
                choices is not None
                and isinstance(choices, (list, tuple))
                and isinstance(response, (int, np.integer))
                and 0 <= int(response) < len(choices)
            ):
                response_text = f"{chr(65 + int(response))}. {choices[int(response)]}"
            else:
                response_text = str(response)

            reasoning = raw.get(self.reasoning_key)
            if reasoning is None:
                response_text, reasoning = _extract_thinking_from_text(response_text)

            items = _split_text_with_image_placeholders(
                prompt_text,
                image_queue,
                prepend_remaining_if_no_placeholder=bool(image_queue),
            )
            for item in items:
                if item.get("type") == "image":
                    ordered_images.append(item["image"])
            messages.append(
                {
                    "role": "user",
                    "content": items
                    if any(it.get("type") == "image" for it in items)
                    else prompt_text,
                }
            )
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response_text,
            }
            if reasoning is not None:
                assistant_msg["reasoning_content"] = str(reasoning)
            messages.append(assistant_msg)

        if self.system_prompt and (
            not messages or messages[0].get("role") != "system"
        ):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return messages, ordered_images

    def _encode_messages_single_pass(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        *,
        add_generation_prompt: bool,
        return_assistant_mask: bool,
    ) -> dict[str, Any]:
        processor = self._get_processor()
        template_kwargs = self._chat_template_kwargs()
        apply_fn = processor.apply_chat_template

        # Fast path: transformers >= 5 processor.apply_chat_template handles
        # images + chat template (+ assistant_masks) in a single consistent call.
        if _supports_kwarg(apply_fn, "return_dict"):
            call_kwargs: dict[str, Any] = {
                "tokenize": True,
                "add_generation_prompt": add_generation_prompt,
                "return_dict": True,
                "return_tensors": "pt",
                **template_kwargs,
            }
            if return_assistant_mask and _supports_kwarg(
                apply_fn, "return_assistant_tokens_mask"
            ):
                call_kwargs["return_assistant_tokens_mask"] = True
            try:
                out = apply_fn(messages, **call_kwargs)
                if isinstance(out, dict) or hasattr(out, "keys"):
                    return dict(out)
            except (TypeError, ValueError, KeyError) as exc:
                logger.debug(
                    "Falling back to two-step processor encoding: %s", exc
                )

        rendered = apply_fn(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        proc_kwargs: dict[str, Any] = {
            "text": [rendered],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            proc_kwargs["images"] = images
        return dict(processor(**proc_kwargs))

    def _compute_fallback_label_mask(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mask everything before the final assistant turn when template lacks generation tags."""
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        if last_assistant_idx is None:
            return torch.ones_like(input_ids, dtype=torch.bool)

        prefix_messages = messages[:last_assistant_idx]
        prefix_images: list[Image.Image] = []
        for msg in prefix_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        prefix_images.append(item["image"])
        prefix_out = self._encode_messages_single_pass(
            prefix_messages,
            prefix_images,
            add_generation_prompt=True,
            return_assistant_mask=False,
        )
        prefix_len = int(prefix_out["input_ids"].reshape(-1).numel())
        label_mask = torch.zeros(input_ids.numel(), dtype=torch.bool)
        label_mask[: min(prefix_len, input_ids.numel())] = True
        return label_mask

    def _safe_truncate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_mask: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Truncate trailing text tokens if sequence exceeds ``max_seq_length``."""
        if self.max_seq_length is None or input_ids.numel() <= self.max_seq_length:
            return input_ids, attention_mask, label_mask, multi_modal_inputs

        mm_types = multi_modal_inputs.get("mm_token_type_ids")
        if torch.is_tensor(mm_types):
            flat_mm = mm_types.reshape(-1)
            # Only cut if no vision tokens exist in the truncated tail; cutting
            # vision tokens would desynchronize pixel_values / image_grid_thw.
            if bool((flat_mm[self.max_seq_length :] != 0).any()):
                logger.warning(
                    "Sample length %d exceeds max_seq_length %d and contains "
                    "vision tokens past the cutoff; keeping full sequence to "
                    "preserve image token alignment.",
                    int(input_ids.numel()),
                    self.max_seq_length,
                )
                return input_ids, attention_mask, label_mask, multi_modal_inputs
            multi_modal_inputs["mm_token_type_ids"] = flat_mm[: self.max_seq_length]

        return (
            input_ids[: self.max_seq_length],
            attention_mask[: self.max_seq_length],
            label_mask[: self.max_seq_length],
            multi_modal_inputs,
        )

    def _process_raw_record(
        self,
        raw: dict[str, Any],
        idx: int,
        *,
        record_dir: Optional[str] = None,
    ) -> SftDatasetItem:
        messages, images = self._normalize_messages(raw, record_dir=record_dir)

        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        answer = (
            _assistant_text_from_message(messages[last_assistant_idx])
            if last_assistant_idx is not None
            else None
        )

        # In generate-eval mode, feed prompt up to the last assistant turn.
        if self.eval_dataset and self.eval_mode == "generate":
            prompt_messages = (
                messages[:last_assistant_idx]
                if last_assistant_idx is not None
                else messages
            )
            prompt_images: list[Image.Image] = []
            for msg in prompt_messages:
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            prompt_images.append(item["image"])
            encoded = self._encode_messages_single_pass(
                prompt_messages,
                prompt_images,
                add_generation_prompt=True,
                return_assistant_mask=False,
            )
            input_ids = encoded.pop("input_ids").reshape(-1).to(dtype=torch.long)
            attention_mask = (
                encoded.pop("attention_mask").reshape(-1).to(dtype=torch.long)
            )
            encoded.pop("assistant_masks", None)
            label_mask = attention_mask.to(dtype=torch.bool)
            multi_modal_inputs = {k: v for k, v in encoded.items() if v is not None}
            return SftDatasetItem(
                prompt=input_ids,
                length=int(input_ids.numel()),
                idx=idx,
                image_data=prompt_images or None,
                answer=answer,
                prompt_text=prompt_messages,
                attention_mask=attention_mask,
                label_mask=label_mask,
                meta=None,
                multi_modal_inputs=multi_modal_inputs,
            )

        encoded = self._encode_messages_single_pass(
            messages,
            images,
            add_generation_prompt=False,
            return_assistant_mask=True,
        )
        input_ids = encoded.pop("input_ids").reshape(-1).to(dtype=torch.long)
        attention_mask = (
            encoded.pop("attention_mask").reshape(-1).to(dtype=torch.long)
        )
        assistant_masks = encoded.pop("assistant_masks", None)

        if assistant_masks is not None:
            asst_tensor = (
                assistant_masks
                if isinstance(assistant_masks, torch.Tensor)
                else torch.as_tensor(assistant_masks)
            ).reshape(-1).to(dtype=torch.bool)
            if asst_tensor.numel() == input_ids.numel() and bool(asst_tensor.any()):
                # label_mask=True means masked out (-100); False means supervised.
                label_mask = ~asst_tensor
            else:
                label_mask = self._compute_fallback_label_mask(
                    messages, images, input_ids
                )
        else:
            label_mask = self._compute_fallback_label_mask(
                messages, images, input_ids
            )

        multi_modal_inputs = {k: v for k, v in encoded.items() if v is not None}
        input_ids, attention_mask, label_mask, multi_modal_inputs = (
            self._safe_truncate(
                input_ids, attention_mask, label_mask, multi_modal_inputs
            )
        )

        return SftDatasetItem(
            prompt=input_ids,
            length=int(input_ids.numel()),
            idx=idx,
            image_data=images or None,
            answer=answer,
            prompt_text=messages,
            attention_mask=attention_mask,
            label_mask=label_mask,
            meta=None,
            multi_modal_inputs=multi_modal_inputs,
        )
