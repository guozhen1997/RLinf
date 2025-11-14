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

import json
import os
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


class SFTDataset(Dataset):
    def __init__(self, cfg: DictConfig, tokenizer):
        self.tokenizer = tokenizer
        self.cfg = cfg
        #self._load_data()
        self.data = self._load_data()

        self.prompt_key = cfg.prompt_key
        self.response_key = cfg.response_key

        self.max_length = cfg.max_length


    def _expand_data_paths(self, inputs: list[str]) -> list[str]:
        exts = {".jsonl", ".json", ".parquet"}
        files: list[str] = []
        for p in inputs:
            if os.path.isdir(p):
                for root, _, fnames in os.walk(p):
                    for fn in fnames:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext in exts:
                            files.append(os.path.join(root, fn))
            else:
                files.append(p)
        files = sorted(set(files))
        return files

    # TODO(tyr): Add lazy load in future.
    def _eager_load_all(self) -> None:
        merged: list[dict[str, Any]] = []
        for path in self.data_paths:
            fmt = os.path.splitext(path)[1].lower()
            if fmt == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    merged.extend(json.loads(l) for l in f)
            elif fmt == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        merged.extend(content)
                    else:
                        merged.append(content)
            elif fmt == ".parquet":
                try:
                    merged.extend(pd.read_parquet(path).to_dict(orient="records"))
                except Exception as e:
                    raise RuntimeError(f"Failed to load parquet eagerly: {path}: {e}")
            else:
                logging.warning(f"Unsupported format {fmt} for path {path}, skipping.")
        self._records = merged
        # Build indices for consistency
        self._indices = [("", "eager", i) for i in range(len(self._records))]

    # Eager load
    def _load_data_in(self):
        """Load data from dirs or paths"""
        raw_paths = [self.cfg.data_paths] if isinstance(self.cfg.data_paths, str) else self.cfg.data_paths
        self.data_paths = self._expand_data_paths(raw_paths)
        self._eager_load_all()

    def _load_data(self):
        """Load data from specified paths"""
        data = []
        for data_path in self.cfg.data_paths:
            with open(data_path, "r") as f:
                if data_path.endswith(".jsonl"):
                    # Load JSONL format (one JSON per line)
                    for line in f:
                        data.append(json.loads(line.strip()))
                else:
                    # Load regular JSON format
                    data.extend(json.load(f))
        return data

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        prompt = self.data[idx][self.prompt_key]
        response = self.data[idx][self.response_key]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = self.tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_chat_str = response + self.tokenizer.eos_token

        # tokenize
        prompt_ids_output = self.tokenizer(
            prompt_chat_str, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = self.tokenizer(
            response_chat_str, return_tensors="pt", add_special_tokens=False
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat(
            (prompt_attention_mask, response_attention_mask), dim=-1
        )

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(
                    size=(self.max_length - sequence_length,), dtype=input_ids.dtype
                )
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(
                size=(self.max_length - sequence_length,), dtype=attention_mask.dtype
            )

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            raise NotImplementedError(
                f"{sequence_length=} is larger than {self.max_length=}"
            )

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
