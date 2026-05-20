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

"""DreamZero ``ComposedModalityTransform`` built from per-embodiment modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform.base import ComposedModalityTransform

from rlinf.models.embodiment.dreamzero.data_transforms.libero_sim import (
    LiberoSimDataTransform,
)
from rlinf.models.embodiment.dreamzero.data_transforms.oxe_droid import (
    OxeDroidDataTransform,
)

# Projector indices for built-in embodiments (must match DreamTransform / collate).
DEFAULT_EMBODIMENT_TAG_MAPPING: dict[str, dict[str, int]] = {
    "libero_sim": {"libero_sim": 21},
    "oxe_droid": {"oxe_droid": 17},
}

__all__ = [
    "DEFAULT_EMBODIMENT_TAG_MAPPING",
    "build_dreamzero_composed_transform",
    "collect_dreamzero_dataset_keys",
    "embodiment_tag_mapping_for_embodiment",
    "language_keys_for_embodiment",
    "load_dreamzero_dataset_metadata",
]


def embodiment_tag_mapping_for_embodiment(
    tag: str,
    override: dict[str, int] | None = None,
) -> dict[str, int]:
    """Return embodiment tag -> projector id mapping for collate / DreamTransform."""
    if override is not None:
        return dict(override)
    try:
        return dict(DEFAULT_EMBODIMENT_TAG_MAPPING[tag])
    except KeyError:
        raise ValueError(
            f"Unsupported embodiment_tag {tag!r}; set actor.model.embodiment_tag_mapping. "
            f"Built-in tags: {list(DEFAULT_EMBODIMENT_TAG_MAPPING)}."
        ) from None


def collect_dreamzero_dataset_keys(
    data_transform: Any,
    embodiment_tag: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Collect video/state/action keys from the transform chain and language keys from embodiment config."""
    video_keys: list[str] = []
    state_keys: list[str] = []
    action_keys: list[str] = []
    for transform in getattr(data_transform, "transforms", []):
        video_keys.extend(getattr(transform, "video_concat_order", []) or [])
        state_keys.extend(getattr(transform, "state_concat_order", []) or [])
        action_keys.extend(getattr(transform, "action_concat_order", []) or [])
    language_keys = language_keys_for_embodiment(embodiment_tag)
    return video_keys, state_keys, action_keys, language_keys


def language_keys_for_embodiment(tag: str) -> list[str]:
    """Return language modality keys from the embodiment transform config."""
    if tag == "oxe_droid":
        modality = OxeDroidDataTransform.get_modality_config()
    elif tag == "libero_sim":
        modality = LiberoSimDataTransform.get_modality_config()
    else:
        raise ValueError(f"Unsupported embodiment_tag {tag!r} for built-in DreamZero transforms. "
        "Currently only oxe_droid and libero_sim are supported. "
        "To add a new embodiment, create a module under "
        "rlinf.models.embodiment.dreamzero.data_transforms.")
    language_cfg = modality.get("language")
    if language_cfg is None:
        raise KeyError(f"Missing language ModalityConfig for {tag!r}")
    return [str(k) for k in language_cfg.modality_keys]


def load_dreamzero_dataset_metadata(cfg: Any) -> DatasetMetadata:
    """Load :class:`DatasetMetadata` for ``embodiment_tag``."""
    tag = cfg.embodiment_tag
    if cfg.get("metadata_json_path", None):
        path = Path(str(cfg["metadata_json_path"])).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"metadata_json_path is not a file: {path}")
    else:
        model_path = cfg.get("model_path", None)
        path = (
            Path(model_path) / "experiment_cfg" / "metadata.json"
            if model_path is not None
            else None
        )
        if path is None or not path.is_file():
            raise FileNotFoundError(
                "DreamZero metadata.json not found. This file is generated from "
                "the dataset and its path must be specified.\n"
                "Set metadata_json_path in your config to the path of the "
                "metadata.json file, or ensure it exists at "
                "model_path/experiment_cfg/metadata.json."
            )

    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    if tag not in blob:
        raise KeyError(
            f"embodiment_tag {tag!r} not found in {path} (keys: {list(blob.keys())})."
        )
    return DatasetMetadata.model_validate(blob[tag])


def build_dreamzero_composed_transform(
    cfg: Any,
    tokenizer_path: str,
) -> ComposedModalityTransform:
    """Construct ``ComposedModalityTransform`` for the current ``embodiment_tag``."""
    tag = cfg.embodiment_tag
    embodiment_tag_mapping = embodiment_tag_mapping_for_embodiment(
        tag, cfg.get("embodiment_tag_mapping")
    )

    if tag == "oxe_droid":
        return OxeDroidDataTransform.get_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=cfg.get("state_horizon", 1),
            action_horizon=cfg.get("action_horizon", 24),
            max_state_dim=cfg.get("max_state_dim", 64),
            max_action_dim=cfg.get("max_action_dim", 32),
            max_length=cfg.get("max_seq_len", 512),
            default_instruction=cfg.get(
                "default_instruction", "Perform the default behavior."
            ),
            language_dropout_prob=cfg.get("language_dropout_prob", 0.0),
            always_use_default_instruction=cfg.get(
                "always_use_default_instruction", False
            ),
            embodiment_tag_mapping=embodiment_tag_mapping,
        )

    if tag == "libero_sim":
        return LiberoSimDataTransform.get_transform(
            tokenizer_path=tokenizer_path,
            state_horizon=cfg.get("state_horizon", 1),
            action_horizon=cfg.get("action_horizon", 16),
            max_state_dim=cfg.get("max_state_dim", 64),
            max_action_dim=cfg.get("max_action_dim", 32),
            max_length=cfg.get("max_seq_len", 512),
            default_instruction=cfg.get(
                "default_instruction", "Perform the default behavior."
            ),
            language_dropout_prob=cfg.get("language_dropout_prob", 0.0),
            always_use_default_instruction=cfg.get(
                "always_use_default_instruction", False
            ),
            embodiment_tag_mapping=embodiment_tag_mapping,
        )

    raise ValueError(
        f"Unsupported embodiment_tag {tag!r} for built-in DreamZero transforms. "
        "Currently only oxe_droid and libero_sim are supported. "
        "To add a new embodiment, create a module under "
        "rlinf.models.embodiment.dreamzero.data_transforms."
    )
