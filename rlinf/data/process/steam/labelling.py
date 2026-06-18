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

"""Signed-score → boolean-advantage labelling for the STEAM advantage pipeline.

Pure pandas / numpy so both the GPU advantage computation and the CPU relabel
tool share one definition of ``advantage_continuous`` (the raw ensemble signed
score) and the boolean ``advantage`` label derived from it under a threshold or
quantile rule.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Canonical column order for an advantages parquet emitted by the pipeline.
CANONICAL_OUTPUT_COLS: list[str] = [
    "episode_index",
    "frame_index",
    "advantage",
    "advantage_continuous",
    # ``ensemble_signed_score`` is kept as an explicit column so the raw
    # ensemble signal is never lost — recompute / viz read this when they
    # need the unmodified value.
    "ensemble_signed_score",
    "p_progress_mean",
    "p_progress_min",
    "p_progress_variance",
    "member_values",
    "expected_stride_normalized",
    "entropy_aggregated",
    "entropy_member_mean",
    "entropy_member_variance",
]


def resolve_quantile_alias(
    rollout_quantile: Optional[float],
    positive_quantile: Optional[float],
) -> Optional[float]:
    """Resolve ``rollout_quantile`` from itself and its deprecated alias.

    ``positive_quantile`` is a deprecated alias for ``rollout_quantile``; setting
    both is an error. Returns ``None`` when neither is set.
    """
    if rollout_quantile is not None and positive_quantile is not None:
        raise ValueError(
            "Set only one of rollout_quantile and positive_quantile (the latter "
            "is a deprecated alias for the former)."
        )
    return rollout_quantile if rollout_quantile is not None else positive_quantile


def quantile_threshold(scores: np.ndarray, quantile: float) -> float:
    """Threshold above which the top ``quantile`` fraction of ``scores`` lies.

    ``quantile=0.3`` returns the 70th percentile, so ``score > threshold`` keeps
    the top 30%.
    """
    return float(
        np.percentile(np.asarray(scores), (1.0 - float(quantile)) * 100.0)
    )


def compute_advantage_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Write ``advantage_continuous`` (= ``ensemble_signed_score``) on a copy.

    Kept separate from :func:`apply_boolean_label` so the two-phase pipeline can
    compute continuous scores for every dataset first, then decide on a single
    (possibly quantile-derived) threshold before emitting the bool label.
    """
    if df.empty:
        raise RuntimeError(
            "Empty DataFrame — no predictions were produced for this dataset"
        )
    out = df.copy()
    out["advantage_continuous"] = out["ensemble_signed_score"]
    return out


def apply_boolean_label(
    df: pd.DataFrame,
    *,
    positive_threshold: Optional[float],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Add the boolean ``advantage`` column from ``advantage_continuous``.

    ``positive_threshold`` is the effective signed-score threshold for THIS
    dataset: ``advantage = advantage_continuous > positive_threshold``. Pass
    ``None`` to force ``advantage=True`` for every frame — used for ``sft``
    datasets when no expert quantile/threshold is configured (expert demos are
    positive by construction). The caller owns the per-dataset policy (which
    threshold, or ``None``) so quantile mode can apply distinct expert/rollout
    thresholds to the two data types.

    ``columns`` optionally selects/reorders the output columns (e.g.
    :data:`CANONICAL_OUTPUT_COLS`); when ``None`` every input column is kept.
    """
    if "advantage_continuous" not in df.columns:
        raise ValueError(
            "apply_boolean_label requires 'advantage_continuous' column; run "
            "compute_advantage_continuous first."
        )
    out = df.copy()
    if positive_threshold is None:
        out["advantage"] = True
    else:
        out["advantage"] = out["advantage_continuous"] > float(positive_threshold)
    if columns is not None:
        return out[list(columns)]
    return out


__all__ = [
    "CANONICAL_OUTPUT_COLS",
    "resolve_quantile_alias",
    "quantile_threshold",
    "compute_advantage_continuous",
    "apply_boolean_label",
]
