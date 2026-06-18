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

"""Pure-CPU relabel of an existing STEAM advantages parquet.

Rewrites only the boolean ``advantage`` column of
``meta/advantages_{source_tag}.parquet`` under a new threshold or quantile and
writes ``meta/advantages_{new_tag}.parquet`` plus a new ``tags[new_tag]`` entry —
``advantage_continuous`` is never recomputed, so no GPU inference is needed.

Dependency-light (pandas / numpy / PyYAML); reuses the labelling math in
:mod:`~rlinf.data.process.steam.labelling` and metadata I/O in
:mod:`~rlinf.data.process.steam.mixture_config` so threshold semantics stay
identical to ``compute_ensemble_advantages``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from rlinf.data.process.steam.labelling import apply_boolean_label, quantile_threshold
from rlinf.data.process.steam.mixture_config import (
    read_mixture_config,
    write_mixture_config_tag,
)

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_source_parquet(
    dataset_path: Path, source_tag: str
) -> tuple[pd.DataFrame, Path]:
    """Load ``meta/advantages_{source_tag}.parquet`` and validate its schema."""
    meta_dir = dataset_path / "meta"
    source_path = meta_dir / f"advantages_{source_tag}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source parquet not found: {source_path}. Did you pass the "
            "correct source_tag?"
        )
    df = pd.read_parquet(source_path)
    required = {"episode_index", "frame_index", "advantage_continuous"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_path} is missing required columns {sorted(missing)}; "
            "this only relabels the bool 'advantage' from an existing "
            "'advantage_continuous' — it cannot reconstruct it."
        )
    if df.empty:
        raise ValueError(f"{source_path} is empty — nothing to relabel")
    if df.duplicated(subset=["episode_index", "frame_index"]).any():
        raise ValueError(
            f"{source_path} has duplicated (episode_index, frame_index) rows"
        )
    return df, source_path


def resolve_dataset_type(
    dataset_path: Path,
    source_tag: str,
    cli_type: Optional[str],
) -> str:
    """Pick dataset_type from explicit value > mixture_config tag metadata.

    Raises otherwise — never silently falls back to rollout/sft.
    """
    if cli_type is not None:
        t = str(cli_type).lower()
        if t not in ("sft", "rollout"):
            raise ValueError(
                f"dataset_type must be 'sft' or 'rollout', got {cli_type!r}"
            )
        return t
    mix = read_mixture_config(dataset_path)
    tags = mix.get("tags") or {}
    source_meta = tags.get(source_tag) if isinstance(tags, dict) else None
    if isinstance(source_meta, dict):
        meta_type = source_meta.get("dataset_type")
        if meta_type is not None:
            t = str(meta_type).lower()
            if t not in ("sft", "rollout"):
                raise ValueError(
                    f"mixture_config tag {source_tag!r} at {dataset_path} "
                    f"has dataset_type={meta_type!r}; must be 'sft' or 'rollout'"
                )
            return t
    raise ValueError(
        f"dataset_type for {dataset_path} tag {source_tag!r} is not recorded "
        "in mixture_config.yaml tags[source_tag].dataset_type, and no explicit "
        "dataset_type was provided. Pass dataset_types to specify."
    )


def build_new_tag_entry(
    source_meta: dict[str, Any],
    *,
    source_tag: str,
    recorded_threshold: float,
    label_mode: str,
    rollout_quantile: Optional[float],
    expert_quantile: Optional[float],
    dataset_type: str,
    total_samples: int,
    num_positive: int,
) -> dict[str, Any]:
    """Inherit source metadata, then override the fields this tool owns.

    ``recorded_threshold`` is the threshold applied to THIS dataset (the
    rollout-pool percentile for rollout datasets, the expert-pool percentile for
    sft datasets filtered by ``expert_quantile``; for all-True sft the caller
    passes the rollout threshold for provenance).
    """
    entry = dict(source_meta) if isinstance(source_meta, dict) else {}
    entry["positive_threshold"] = float(recorded_threshold)
    entry["label_mode"] = str(label_mode)
    entry["dataset_type"] = str(dataset_type)
    entry["total_samples"] = int(total_samples)
    entry["num_positive"] = int(num_positive)
    entry["derived_from_tag"] = str(source_tag)
    # Always clear the deprecated single-quantile key so a stale value from the
    # source tag can never be misread.
    entry.pop("positive_quantile", None)
    if label_mode == "quantile":
        if rollout_quantile is None:
            raise ValueError("label_mode='quantile' requires rollout_quantile")
        entry["rollout_quantile"] = float(rollout_quantile)
        if expert_quantile is not None:
            entry["expert_quantile"] = float(expert_quantile)
        else:
            entry.pop("expert_quantile", None)
    else:
        # Threshold mode — drop stale quantiles inherited from the source tag so
        # the new entry cannot be misread as quantile-derived.
        entry.pop("rollout_quantile", None)
        entry.pop("expert_quantile", None)
    return entry


def _validate_relabel_args(
    *,
    source_tag: str,
    new_tag: str,
    mode: str,
    positive_threshold: Optional[float],
    rollout_quantile: Optional[float],
    expert_quantile: Optional[float],
    dataset_paths: list[Path],
    dataset_types: Optional[list[str]],
) -> None:
    """Hard-fail on inconsistent relabel parameters before touching any data."""
    if new_tag == source_tag:
        raise ValueError("new_tag must differ from source_tag")
    if mode not in ("threshold", "quantile"):
        raise ValueError(f"mode must be 'threshold' or 'quantile', got {mode!r}")
    if mode == "threshold":
        if positive_threshold is None:
            raise ValueError("positive_threshold is required when mode='threshold'")
        if not (-1.0 <= positive_threshold <= 1.0):
            raise ValueError(
                "positive_threshold must be in [-1, 1] (it is a signed-score "
                f"threshold); got {positive_threshold}"
            )
        if rollout_quantile is not None or expert_quantile is not None:
            raise ValueError("quantiles are only valid with mode='quantile'")
    else:  # quantile
        if rollout_quantile is None:
            raise ValueError("rollout_quantile is required when mode='quantile'")
        if not (0.0 < rollout_quantile < 1.0):
            raise ValueError(
                "rollout_quantile must be in (0, 1) — fraction of top rollout "
                f"frames; got {rollout_quantile}"
            )
        if expert_quantile is not None and not (0.0 < expert_quantile < 1.0):
            raise ValueError(
                "expert_quantile must be in (0, 1) — fraction of top sft frames; "
                f"got {expert_quantile}"
            )
        if positive_threshold is not None:
            raise ValueError("positive_threshold is only valid with mode='threshold'")
    if dataset_types is not None and len(dataset_types) != len(dataset_paths):
        raise ValueError(
            f"dataset_types length {len(dataset_types)} does not match "
            f"dataset_paths length {len(dataset_paths)}"
        )


def relabel_advantages(
    dataset_paths: list[PathLike],
    *,
    source_tag: str,
    new_tag: str,
    mode: str,
    positive_threshold: Optional[float] = None,
    rollout_quantile: Optional[float] = None,
    expert_quantile: Optional[float] = None,
    dataset_types: Optional[list[str]] = None,
) -> None:
    """Relabel ``advantage`` across ``dataset_paths`` under a new threshold/quantile.

    Quantile mode scores rollout and sft over independent pools. When
    ``expert_quantile`` is omitted, sft frames stay all-True (the historical
    convention) but still record the rollout threshold for provenance.
    """
    resolved_paths = [Path(p) for p in dataset_paths]
    _validate_relabel_args(
        source_tag=source_tag,
        new_tag=new_tag,
        mode=mode,
        positive_threshold=positive_threshold,
        rollout_quantile=rollout_quantile,
        expert_quantile=expert_quantile,
        dataset_paths=resolved_paths,
        dataset_types=dataset_types,
    )

    # ---- Phase 1: load every source parquet + resolve dataset_type ----
    # Resolve types up-front (not lazily) so a missing metadata entry fails loud
    # before we pool scores or start writing outputs.
    loaded: list[tuple[Path, pd.DataFrame, Path, str, dict[str, Any]]] = []
    for i, ds_path in enumerate(resolved_paths):
        ds_path = ds_path.resolve()
        if not ds_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {ds_path}")
        cli_type = dataset_types[i] if dataset_types is not None else None
        dataset_type = resolve_dataset_type(ds_path, source_tag, cli_type)
        mix = read_mixture_config(ds_path)
        tags = mix.get("tags") or {}
        source_meta = tags.get(source_tag) if isinstance(tags, dict) else {}
        if not isinstance(source_meta, dict):
            raise RuntimeError(
                f"mixture_config tag {source_tag!r} at {ds_path} is not a mapping"
            )
        df, src_path = load_source_parquet(ds_path, source_tag)
        loaded.append((ds_path, df, src_path, dataset_type, source_meta))
        logger.info(
            "Loaded %s: %d rows, dataset_type=%s, advantage_continuous "
            "range=[%.4f, %.4f], mean=%.4f",
            src_path,
            len(df),
            dataset_type,
            float(df["advantage_continuous"].min()),
            float(df["advantage_continuous"].max()),
            float(df["advantage_continuous"].mean()),
        )

    # ---- Phase 2: pick the rollout and (optional) expert thresholds ----
    expert_threshold: Optional[float] = None
    if mode == "threshold":
        rollout_threshold = float(positive_threshold)
        logger.info(
            "Mode: threshold; rollout_threshold=%.4f; sft frames all-True.",
            rollout_threshold,
        )
    else:
        rollout_scores = [
            df["advantage_continuous"].values
            for _, df, _, dt, _ in loaded
            if dt == "rollout"
        ]
        if not rollout_scores:
            raise ValueError(
                "mode='quantile' requires at least one rollout dataset; every "
                "passed dataset resolved to type='sft'."
            )
        combined = np.concatenate(rollout_scores)
        rollout_threshold = quantile_threshold(combined, float(rollout_quantile))
        n_positive_rollout = int((combined > rollout_threshold).sum())
        logger.info(
            "Mode: quantile rollout_quantile=%.3f (top %.1f%% of %d rollout "
            "frames) → rollout_threshold=%.4f; %d/%d rollout frames > threshold "
            "(range=[%.4f, %.4f])",
            rollout_quantile,
            rollout_quantile * 100.0,
            len(combined),
            rollout_threshold,
            n_positive_rollout,
            len(combined),
            float(combined.min()),
            float(combined.max()),
        )
        if expert_quantile is not None:
            sft_scores = [
                df["advantage_continuous"].values
                for _, df, _, dt, _ in loaded
                if dt == "sft"
            ]
            if sft_scores:
                combined_sft = np.concatenate(sft_scores)
                expert_threshold = quantile_threshold(
                    combined_sft, float(expert_quantile)
                )
                n_positive_sft = int((combined_sft > expert_threshold).sum())
                logger.info(
                    "expert_quantile=%.3f (top %.1f%% of %d sft frames) → "
                    "expert_threshold=%.4f; %d/%d sft frames > threshold "
                    "(range=[%.4f, %.4f])",
                    expert_quantile,
                    expert_quantile * 100.0,
                    len(combined_sft),
                    expert_threshold,
                    n_positive_sft,
                    len(combined_sft),
                    float(combined_sft.min()),
                    float(combined_sft.max()),
                )
            else:
                logger.warning(
                    "expert_quantile=%.3f set but no sft datasets passed; "
                    "expert quantile is ignored.",
                    expert_quantile,
                )
        else:
            logger.info("expert_quantile not set → every sft frame labelled positive.")

    # ---- Phase 3: relabel + write each dataset ----
    for ds_path, df, src_path, dataset_type, source_meta in loaded:
        is_sft = dataset_type == "sft"
        # Threshold actually applied to the bool label. None ⇒ force every frame
        # True (sft with no expert quantile).
        label_threshold = expert_threshold if is_sft else rollout_threshold
        new_df = apply_boolean_label(df, positive_threshold=label_threshold)
        out_path = ds_path / "meta" / f"advantages_{new_tag}.parquet"
        if out_path.exists():
            logger.warning("Overwriting existing %s", out_path)
        new_df.to_parquet(out_path, index=False)
        num_positive = int(new_df["advantage"].sum())
        total_samples = int(len(new_df))

        # For all-True sft fall back to the rollout threshold so the entry still
        # carries a numeric value (matches compute_ensemble_advantages).
        recorded_threshold = (
            label_threshold if label_threshold is not None else rollout_threshold
        )
        new_entry = build_new_tag_entry(
            source_meta,
            source_tag=source_tag,
            recorded_threshold=recorded_threshold,
            label_mode=mode,
            rollout_quantile=(
                float(rollout_quantile) if mode == "quantile" else None
            ),
            expert_quantile=(
                float(expert_quantile)
                if (mode == "quantile" and expert_quantile is not None)
                else None
            ),
            dataset_type=dataset_type,
            total_samples=total_samples,
            num_positive=num_positive,
        )
        mix_path = write_mixture_config_tag(ds_path, new_tag, new_entry)
        logger.info(
            "  wrote %s (type=%s, rows=%d, positive=%d/%d, %.1f%%); updated %s",
            out_path,
            dataset_type,
            total_samples,
            num_positive,
            total_samples,
            100.0 * num_positive / max(total_samples, 1),
            mix_path,
        )

    logger.info("Done.")


__all__ = [
    "load_source_parquet",
    "resolve_dataset_type",
    "build_new_tag_entry",
    "relabel_advantages",
]
