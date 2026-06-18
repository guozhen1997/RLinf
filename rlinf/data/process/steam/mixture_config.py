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

"""``meta/mixture_config.yaml`` per-tag read/write for the advantage pipeline.

Both the advantage computation and the CPU relabel tool record a per-tag entry
under ``tags[<tag>]`` of each dataset's ``meta/mixture_config.yaml``. Top-level
fields (``advantage_tag``, ``datasets``, ``unified_threshold``, …) are treated
as read-only — only ``tags`` is ever mutated. PyYAML is the only dependency so
the pure-CPU relabel tool can import this without torch / LeRobot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml

PathLike = Union[str, Path]


def mixture_config_path(dataset_path: PathLike) -> Path:
    """Return ``<dataset_path>/meta/mixture_config.yaml``."""
    return Path(dataset_path) / "meta" / "mixture_config.yaml"


def read_mixture_config(dataset_path: PathLike) -> dict[str, Any]:
    """Load ``meta/mixture_config.yaml`` as a dict (``{}`` when absent/empty)."""
    p = mixture_config_path(dataset_path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"mixture_config.yaml at {p} is not a mapping; refusing to read"
        )
    return loaded


def write_mixture_config_tag(
    dataset_path: PathLike, tag: str, entry: dict[str, Any]
) -> Path:
    """Merge ``tags[tag] = entry`` into ``meta/mixture_config.yaml``.

    Only the ``tags`` sub-mapping is written; every other top-level field is
    preserved verbatim. Returns the config path that was written.
    """
    p = mixture_config_path(dataset_path)
    existing = read_mixture_config(dataset_path)
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(f"mixture_config.yaml 'tags' field at {p} is not a mapping")
    tags[str(tag)] = entry
    existing["tags"] = tags
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return p


def update_advantage_tag(
    *,
    dataset_path: PathLike,
    tag: str,
    positive_threshold: float,
    ensemble_size: int,
    num_bins: int,
    total_samples: int,
    num_positive: int,
    dataset_type: str,
    label_mode: str,
    rollout_quantile: Optional[float] = None,
    expert_quantile: Optional[float] = None,
) -> Path:
    """Write the per-tag advantage entry produced by ``compute_ensemble_advantages``.

    ``label_mode`` records which rule produced the bool label: ``"threshold"``
    (fixed ``positive_threshold``) or ``"quantile"``. In quantile mode
    ``rollout_quantile`` (required) and ``expert_quantile`` (optional) record the
    per-pool top-fractions, while ``positive_threshold`` holds the threshold
    actually applied to THIS dataset (the rollout-pool percentile for ``rollout``
    datasets, the expert-pool percentile for ``sft`` datasets filtered by
    ``expert_quantile``). ``dataset_type`` is recorded so downstream recompute /
    viz tooling can read it without re-parsing the parquet schema.
    """
    if label_mode not in ("threshold", "quantile"):
        raise ValueError(
            f"label_mode must be 'threshold' or 'quantile', got {label_mode!r}"
        )
    if label_mode == "quantile" and rollout_quantile is None:
        raise ValueError("label_mode='quantile' requires rollout_quantile")

    entry: dict[str, Any] = {
        "positive_threshold": float(positive_threshold),
        "label_mode": str(label_mode),
        "ensemble_size": int(ensemble_size),
        "num_bins": int(num_bins),
        "total_samples": int(total_samples),
        "num_positive": int(num_positive),
        "dataset_type": str(dataset_type),
    }
    if label_mode == "quantile":
        entry["rollout_quantile"] = float(rollout_quantile)
        if expert_quantile is not None:
            entry["expert_quantile"] = float(expert_quantile)
    return write_mixture_config_tag(dataset_path, tag, entry)


__all__ = [
    "mixture_config_path",
    "read_mixture_config",
    "write_mixture_config_tag",
    "update_advantage_tag",
]
