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

"""Unit tests for Qwen3.8-27B multimodal SFT support."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from rlinf.config import SupportedModel
from rlinf.data.datasets.vlm import MultimodalSFTDataset, collate_fn
from rlinf.data.datasets.vlm.robo2vlm import Robo2VLMSFTDataset
from rlinf.workers.sft.fsdp_vlm_sft_worker import FSDPVlmSftWorker
from rlinf.workers.sft.utils import vlm_extract_answer


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [10 + (ord(ch) % 50) for ch in text]


class _FakeQwen38Processor:
    """Minimal processor mimicking Qwen3.8's single-pass apply_chat_template."""

    def __init__(self, emit_assistant_masks: bool = True):
        self.tokenizer = _FakeTokenizer()
        self.emit_assistant_masks = emit_assistant_masks
        self.calls: list[dict] = []

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        return_assistant_tokens_mask: bool = False,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "return_assistant_tokens_mask": return_assistant_tokens_mask,
                "kwargs": kwargs,
            }
        )
        # Support Robo2VLM's batch-of-1 tuple convention as well as flat messages list.
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], (list, tuple))
        ):
            text = self._render_text(list(messages[0]), add_generation_prompt)
            return [text]

        if not tokenize:
            return self._render_text(messages, add_generation_prompt)

        ids: list[int] = []
        mm_types: list[int] = []
        asst_mask: list[int] = []
        images: list[Image.Image] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            # role prefix token
            ids.append(100 if role == "user" else (101 if role == "assistant" else 102))
            mm_types.append(0)
            asst_mask.append(0)

            if role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    r_toks = self.tokenizer.encode(reasoning)
                    ids.extend(r_toks)
                    mm_types.extend([0] * len(r_toks))
                    asst_mask.extend([1] * len(r_toks))
                text = content if isinstance(content, str) else "".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
                t_toks = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
                ids.extend(t_toks)
                mm_types.extend([0] * len(t_toks))
                asst_mask.extend([1] * len(t_toks))
            else:
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            images.append(item["image"])
                            # 4 synthetic vision pad tokens per image
                            ids.extend([248056] * 4)
                            mm_types.extend([1] * 4)
                            asst_mask.extend([0] * 4)
                        elif "text" in item:
                            t_toks = self.tokenizer.encode(item["text"])
                            ids.extend(t_toks)
                            mm_types.extend([0] * len(t_toks))
                            asst_mask.extend([0] * len(t_toks))
                else:
                    t_toks = self.tokenizer.encode(str(content))
                    ids.extend(t_toks)
                    mm_types.extend([0] * len(t_toks))
                    asst_mask.extend([0] * len(t_toks))

        if add_generation_prompt:
            ids.append(101)
            mm_types.append(0)
            asst_mask.append(0)

        out = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([mm_types], dtype=torch.long),
        }
        if images:
            out["pixel_values"] = torch.randn(4 * len(images), 16)
            out["image_grid_thw"] = torch.tensor(
                [[1, 2, 2] for _ in images], dtype=torch.long
            )
        if return_assistant_tokens_mask and self.emit_assistant_masks:
            out["assistant_masks"] = torch.tensor([asst_mask], dtype=torch.long)
        return out

    def _render_text(self, messages: list[dict], add_generation_prompt: bool) -> str:
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text = "".join(
                    "<image>" if item.get("type") == "image" else item.get("text", "")
                    for item in content
                )
            else:
                text = str(content)
            parts.append(f"<|im_start|>{msg['role']}\n{text}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, images=None, return_tensors="pt", padding=True):
        s = text[0]
        ids = self.tokenizer.encode(s)
        out = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "mm_token_type_ids": torch.zeros((1, len(ids)), dtype=torch.long),
        }
        if images:
            out["pixel_values"] = torch.randn(4 * len(images), 16)
            out["image_grid_thw"] = torch.tensor(
                [[1, 2, 2] for _ in images], dtype=torch.long
            )
        return out


def _make_sft_cfg(data_path: str, *, eval_mode: str = "loss", max_seq_length: int = 2048):
    return OmegaConf.create(
        {
            "data": {
                "type": "vlm",
                "dataset_name": "multimodal_sft",
                "apply_chat_template": True,
                "use_chat_template": True,
                "train_data_paths": data_path,
                "prompt_key": "prompt",
                "response_key": "response",
                "answer_key": "answer",
                "choice_key": "choices",
                "image_keys": ["images", "image"],
                "enable_thinking": True,
                "reasoning_effort": "low",
                "max_prompt_length": max_seq_length,
                "max_seq_length": max_seq_length,
                "lazy_loading": True,
            },
            "actor": {
                "eval_mode": eval_mode,
                "model": {
                    "model_type": "qwen3_8",
                    "model_path": "/nonexistent/qwen3_8",
                    "trust_remote_code": True,
                },
            },
        }
    )


def test_supported_model_and_extract_answer_support_qwen3_8():
    assert SupportedModel("qwen3_8") == SupportedModel.QWEN3_8_SFT
    assert SupportedModel("qwen3_5") == SupportedModel.QWEN3_5_SFT

    raw = "<think>\nCheck the mug color.\n</think>\n\nThe answer is: red mug.<|im_end|>"
    assert vlm_extract_answer(raw, "qwen3_8") == "red mug"
    assert vlm_extract_answer(raw, "qwen3_5") == "red mug"


def test_multimodal_sft_dataset_supports_multiturn_sharegpt_and_qa(tmp_path: Path):
    img1_path = tmp_path / "cam1.png"
    img2_path = tmp_path / "cam2.png"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(img1_path)
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(img2_path)

    records = [
        # 1) Multi-turn OpenAI messages with inline images + reasoning_content
        {
            "messages": [
                {"role": "system", "content": "Be precise."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "cam1.png"},
                        {"type": "text", "text": "Step 1?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Move left.",
                    "reasoning_content": "Target is left.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img2_path)},
                        {"type": "text", "text": "Step 2?"},
                    ],
                },
                {"role": "assistant", "content": "Grasp."},
            ]
        },
        # 2) ShareGPT conversations with <image> placeholder and <think> tag
        {
            "conversations": [
                {"from": "human", "value": "<image>\nWhat color?"},
                {"from": "gpt", "value": "<think>Look closely</think>\nRed"},
            ],
            "images": ["cam1.png"],
        },
        # 3) Simple text-only QA record (mixed into same batch!)
        {
            "prompt": "Say ok.",
            "response": "ok",
        },
    ]
    jsonl_path = tmp_path / "train.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    cfg = _make_sft_cfg(str(jsonl_path))
    ds = MultimodalSFTDataset(str(jsonl_path), cfg, _FakeTokenizer(), eval_dataset=False)
    ds._processor = _FakeQwen38Processor(emit_assistant_masks=True)

    items = [ds[0], ds[1], ds[2]]
    assert len(items) == 3

    # Multi-turn item has 2 images and masks user/system tokens while supervising assistant turns
    assert items[0].multi_modal_inputs["image_grid_thw"].shape == (2, 3)
    assert items[0].multi_modal_inputs["pixel_values"].shape == (8, 16)
    assert bool((~items[0].label_mask).any())
    assert bool(items[0].label_mask.any())
    assert items[0].answer == "Grasp."

    # ShareGPT item has 1 image and parsed answer without <think>
    assert items[1].multi_modal_inputs["image_grid_thw"].shape == (1, 3)
    assert items[1].answer == "Red"

    # Text-only item has no pixel_values
    assert "pixel_values" not in items[2].multi_modal_inputs

    # Collate mixed image + text-only batch cleanly
    batch = collate_fn(items)
    assert batch["prompt"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["prompt"].shape
    assert batch["label_mask"].shape == batch["prompt"].shape
    assert batch["multi_modal_inputs"]["mm_token_type_ids"].shape == batch["prompt"].shape
    assert len(batch["multi_modal_inputs"]["pixel_values"]) == 2
    assert batch["multi_modal_inputs"]["image_grid_thw"].shape == (3, 3)


def test_multimodal_sft_dataset_fallback_mask_when_template_lacks_generation_tags(
    tmp_path: Path,
):
    img_path = tmp_path / "a.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path)
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "prompt": "Describe it.",
                "response": "A dark square.",
                "image": str(img_path),
            }
        )
        + "\n"
    )

    cfg = _make_sft_cfg(str(jsonl_path))
    ds = MultimodalSFTDataset(str(jsonl_path), cfg, _FakeTokenizer(), eval_dataset=False)
    ds._processor = _FakeQwen38Processor(emit_assistant_masks=False)

    item = ds[0]
    assert item.label_mask.shape == item.prompt.shape
    # Prefix is masked (True) and suffix is supervised (False)
    assert bool(item.label_mask[0].item()) is True
    assert bool(item.label_mask[-1].item()) is False


def test_robo2vlm_sft_uses_add_generation_prompt_false_for_full_prompt(tmp_path: Path):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=(0, 0, 255)).save(buf, format="PNG")
    img_path = tmp_path / "r.png"
    img_path.write_bytes(buf.getvalue())

    jsonl_path = tmp_path / "robo.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "question": "Pick object?",
                "choices": ["cube", "mug"],
                "correct_answer": "A",
                "image": str(img_path),
            }
        )
        + "\n"
    )

    cfg = _make_sft_cfg(str(jsonl_path))
    cfg.data.dataset_name = "robo2vlmsft"
    cfg.data.prompt_key = "question"
    cfg.data.choice_key = "choices"
    cfg.data.answer_key = "correct_answer"
    cfg.data.image_keys = ["image"]
    ds = Robo2VLMSFTDataset(str(jsonl_path), cfg, _FakeTokenizer(), eval_dataset=False)
    proc = _FakeQwen38Processor()
    ds._processor = proc

    item = ds[0]
    assert item.answer == "A. cube"
    # First call is full prompt (add_generation_prompt=False), second is label prefix (True)
    assert len(proc.calls) == 2
    assert proc.calls[0]["add_generation_prompt"] is False
    assert proc.calls[1]["add_generation_prompt"] is True
    assert proc.calls[0]["kwargs"]["enable_thinking"] is True
    assert proc.calls[0]["kwargs"]["reasoning_effort"] == "low"


def test_chunked_fp32_head_sft_loss_matches_standard_cross_entropy():
    torch.manual_seed(0)
    vocab, hidden = 64, 16

    class _MiniBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.proj = nn.Linear(hidden, hidden)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return (self.proj(self.embed(input_ids)),)

    class _MiniLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MiniBackbone()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            logits_to_keep=0,
            return_dict=True,
            use_cache=False,
            **kwargs,
        ):
            hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            slice_idx = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_idx, :])
            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous().float()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            if not return_dict:
                return (logits,) if loss is None else (loss, logits)
            return SimpleNamespace(loss=loss, logits=logits)

    model_ref = _MiniLM()
    model_chunked = _MiniLM()
    model_chunked.load_state_dict(model_ref.state_dict())

    batch_ids = torch.tensor([[0, 0, 5, 8, 12, 19, 2], [3, 7, 11, 15, 20, 25, 2]])
    attn_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    label_mask = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False],
        ]
    )
    batch = {
        "prompt": batch_ids,
        "attention_mask": attn_mask,
        "label_mask": label_mask,
        "multi_modal_inputs": {},
    }

    from contextlib import nullcontext

    def _make_worker(model, use_fp32_head: bool):
        w = object.__new__(FSDPVlmSftWorker)
        w.model = model
        w.device = torch.device("cpu")
        w.amp_context = nullcontext()
        w._use_fp32_head = use_fp32_head
        w._activation_cpu_offload = False
        w._activation_cpu_offload_pin_memory = False
        w._activation_cpu_offload_gpu_limit_bytes = 0
        return w

    w_ref = _make_worker(model_ref, use_fp32_head=False)
    w_chunked = _make_worker(model_chunked, use_fp32_head=True)

    loss_ref, _ = w_ref.get_train_model_output(batch)
    loss_chunked, _ = w_chunked.get_train_model_output(batch)
    assert torch.allclose(loss_chunked, loss_ref, atol=1e-5, rtol=1e-5)

    loss_ref.backward()
    loss_chunked.backward()
    for (n1, p1), (n2, p2) in zip(
        model_ref.named_parameters(), model_chunked.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p2.grad, p1.grad, atol=1e-5, rtol=1e-5), f"grad mismatch on {n1}"
