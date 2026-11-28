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

"""Pure SentencePiece tokenizers matching ``openpi.models.tokenizer``."""

from __future__ import annotations

import logging

import numpy as np
import sentencepiece

from rlinf.models.embodiment.openpi.transforms import download

_PALIGEMMA_TOKENIZER_URI = "gs://big_vision/paligemma_tokenizer.model"


class PaligemmaTokenizer:
    """PaliGemma prompt tokenizer for Pi0 and Pi0.5 (JAX-free)."""

    def __init__(self, max_len: int = 48) -> None:
        self._max_len = max_len
        path = download.maybe_download(_PALIGEMMA_TOKENIZER_URI, gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # Pi0.5 format: state is discretized into 256 bins in [-1, 1] and embedded in text.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format: tokenize "\n" separately as the "start of answer" token.
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating. "
                    "Consider increasing `max_token_len` in your model config if this happens frequently.",
                    len(tokens),
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    """FAST action tokenizer with lazy ``transformers`` import."""

    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ) -> None:
        self._max_len = max_len
        path = download.maybe_download(_PALIGEMMA_TOKENIZER_URI, gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        from transformers import AutoProcessor

        self._fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )
        self._fast_skip_tokens = 128

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating. "
                    "Consider increasing `max_token_len` in your model config if this happens frequently.",
                    len(tokens),
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(
                decoded_tokens.split("Action: ")[1].split("|")[0].strip()
            )
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )[0]

    def _act_tokens_to_paligemma_tokens(
        self, tokens: np.ndarray | list[int]
    ) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return (
            self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
        )
