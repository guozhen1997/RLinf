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

import logging

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.reasoning.collate_rl import collate_fn
from rlinf.data.datasets.reasoning.collate_sft import sft_collate_fn
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.datasets.reasoning.rstar2 import Rstar2Dataset
from rlinf.data.datasets.reasoning.wideseek_r1 import WideSeekR1Dataset
from rlinf.data.datasets.vlm import VLMDatasetRegistry

TEXT_DATASET_TYPE_MAP = {
    "reasoning": ReasoningDataset,
    "math": ReasoningDataset,
    "wideseek_r1": WideSeekR1Dataset,
    "rstar2": Rstar2Dataset,
}


def create_rl_dataset(
    config: DictConfig, tokenizer: AutoTokenizer
) -> tuple[Dataset | None, Dataset | None]:
    """Create train/val datasets according to ``config.data.type``."""
    if config.data.type in TEXT_DATASET_TYPE_MAP:
        dataset_cls = TEXT_DATASET_TYPE_MAP[config.data.type]
        logging.info(f"Using dataset class: {dataset_cls.__name__}")

        train_dataset, val_dataset = None, None
        if config.runner.task_type != "reasoning_eval":
            train_dataset = dataset_cls(
                data_paths=config.data.train_data_paths,
                config=config,
                tokenizer=tokenizer,
            )

        if config.data.get("val_data_paths", None) is not None:
            val_dataset = dataset_cls(
                data_paths=config.data.val_data_paths,
                config=config,
                tokenizer=tokenizer,
            )
        return train_dataset, val_dataset

    if config.data.type == "vision_language":
        dataset_name = getattr(config.data, "dataset_name", None)
        lazy_loading = bool(getattr(config.data, "lazy_loading", False))

        logging.info(
            f"Using VLM dataset: name={dataset_name}, lazy_loading={lazy_loading}"
        )

        train_dataset = VLMDatasetRegistry.create(
            dataset_name,
            data_paths=config.data.train_data_paths,
            config=config,
            tokenizer=tokenizer,
        )
        val_dataset = None
        if config.data.get("val_data_paths", None) is not None:
            val_dataset = VLMDatasetRegistry.create(
                dataset_name,
                data_paths=config.data.val_data_paths,
                config=config,
                tokenizer=tokenizer,
            )
        return train_dataset, val_dataset

    raise NotImplementedError(
        "Unsupported dataset type "
        f"{config.data.type}, only support "
        f"{sorted(TEXT_DATASET_TYPE_MAP.keys()) + ['vision_language']}"
    )


__all__ = [
    "ReasoningDataset",
    "Rstar2Dataset",
    "TEXT_DATASET_TYPE_MAP",
    "WideSeekR1Dataset",
    "collate_fn",
    "create_rl_dataset",
    "sft_collate_fn",
]
