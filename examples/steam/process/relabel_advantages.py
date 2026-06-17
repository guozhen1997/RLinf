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

"""Relabel the bool ``advantage`` column on an existing advantages parquet
under a new threshold or quantile — pure CPU, no GPU inference.

Reads ``meta/advantages_{source_tag}.parquet`` for each dataset, rewrites only
the boolean ``advantage`` column (``advantage_continuous`` and every other
column pass through untouched), and writes
``meta/advantages_{new_tag}.parquet`` plus a new ``tags[new_tag]`` entry in
``meta/mixture_config.yaml``.

Two mutually-exclusive modes:

    * ``--mode threshold --positive_threshold T``
          rollout frames: ``advantage = advantage_continuous > T``
          sft frames:     ``advantage = True``

    * ``--mode quantile --rollout_quantile Q [--expert_quantile E]``
          (Q, E in (0, 1); ``--positive_quantile`` is a deprecated alias for
          ``--rollout_quantile``)
          rollout: pool ``advantage_continuous`` across every rollout dataset
              in ``--dataset_paths``, take the ``(1 - Q)``-th percentile
              ``rollout_threshold``, then
              ``advantage = advantage_continuous > rollout_threshold``.
          sft: when ``--expert_quantile E`` is given, pool
              ``advantage_continuous`` across every sft dataset, take the
              ``(1 - E)``-th percentile ``expert_threshold``, then
              ``advantage = advantage_continuous > expert_threshold``;
              otherwise every sft frame is labelled True (historical default).
          The two pools are scored independently.

``dataset_type`` per path is resolved in this order: ``--dataset_types``
(CLI) → ``tags[source_tag].dataset_type`` in ``mixture_config.yaml``. If
neither is available the script raises — we never silently default to
``rollout`` or ``sft``.

The new tag's metadata inherits everything from ``tags[source_tag]`` (so
ensemble_size, num_bins, etc. survive) and then overrides
``positive_threshold`` (the threshold applied to THIS dataset), ``label_mode``,
``total_samples``, ``num_positive``, ``dataset_type``, ``derived_from_tag`` —
plus ``rollout_quantile`` / ``expert_quantile`` when ``--mode=quantile``.

Example — threshold-only relabel on two datasets::

    python relabel_advantages.py \\
        --dataset_paths /p1 /p2 \\
        --source_tag existing_tag --new_tag tight_threshold \\
        --mode threshold --positive_threshold 0.6

Example — top-30% rollout + top-90% expert quantile::

    python relabel_advantages.py \\
        --dataset_paths /sft_ds /rollout_ds_a /rollout_ds_b \\
        --dataset_types sft rollout rollout \\
        --source_tag existing_tag --new_tag q30 \\
        --mode quantile --rollout_quantile 0.3 --expert_quantile 0.9
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixture config I/O — kept self-contained so this script doesn't pull in
# the GPU-heavy compute_advantages_ensemble module.
# ---------------------------------------------------------------------------


def _mixture_config_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "mixture_config.yaml"


def _read_mixture_config(dataset_path: Path) -> dict[str, Any]:
    p = _mixture_config_path(dataset_path)
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


def _write_mixture_config_tag(
    dataset_path: Path, tag: str, new_entry: dict[str, Any]
) -> Path:
    """Merge ``tags[tag] = new_entry`` into ``meta/mixture_config.yaml``."""
    p = _mixture_config_path(dataset_path)
    existing = _read_mixture_config(dataset_path)
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(f"mixture_config.yaml at {p} has non-mapping 'tags'")
    tags[str(tag)] = new_entry
    existing["tags"] = tags
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return p


# ---------------------------------------------------------------------------
# Source parquet + dataset_type resolution
# ---------------------------------------------------------------------------


def _load_source_parquet(
    dataset_path: Path, source_tag: str
) -> tuple[pd.DataFrame, Path]:
    meta_dir = dataset_path / "meta"
    source_path = meta_dir / f"advantages_{source_tag}.parquet"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Source parquet not found: {source_path}. Did you pass the "
            "correct --source_tag?"
        )
    df = pd.read_parquet(source_path)
    required = {"episode_index", "frame_index", "advantage_continuous"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_path} is missing required columns {sorted(missing)}; "
            "this script only relabels the bool 'advantage' from an existing "
            "'advantage_continuous' — it cannot reconstruct it."
        )
    if df.empty:
        raise ValueError(f"{source_path} is empty — nothing to relabel")
    if df.duplicated(subset=["episode_index", "frame_index"]).any():
        raise ValueError(
            f"{source_path} has duplicated (episode_index, frame_index) rows"
        )
    return df, source_path


def _resolve_dataset_type(
    dataset_path: Path,
    source_tag: str,
    cli_type: Optional[str],
) -> str:
    """Pick dataset_type from CLI > mixture_config tag metadata. Raise
    otherwise — never silently fall back to rollout/sft."""
    if cli_type is not None:
        t = str(cli_type).lower()
        if t not in ("sft", "rollout"):
            raise ValueError(
                f"--dataset_types entry must be 'sft' or 'rollout', got {cli_type!r}"
            )
        return t
    mix = _read_mixture_config(dataset_path)
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
        "in mixture_config.yaml tags[source_tag].dataset_type, and "
        "--dataset_types was not provided. Pass --dataset_types to specify."
    )


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def _relabel(
    df: pd.DataFrame,
    *,
    positive_threshold: Optional[float],
) -> pd.DataFrame:
    """Rewrite ``advantage`` on a copy of ``df``; keep every other column.

    ``positive_threshold`` is the effective signed-score threshold for THIS
    dataset: ``advantage = advantage_continuous > positive_threshold``. Pass
    ``None`` to force ``advantage=True`` for every frame — used for ``sft``
    datasets when no expert quantile is configured. The caller picks the
    threshold (expert vs rollout) per dataset so the two pools stay
    independent.
    """
    out = df.copy()
    if positive_threshold is None:
        out["advantage"] = True
    else:
        out["advantage"] = out["advantage_continuous"] > float(positive_threshold)
    return out


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------


def _build_new_tag_entry(
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
    """Inherit source metadata, then override the fields this script owns.

    ``recorded_threshold`` is the threshold applied to THIS dataset (the
    rollout-pool percentile for rollout datasets, the expert-pool percentile
    for sft datasets filtered by ``--expert_quantile``; for all-True sft the
    caller passes the rollout threshold for provenance).
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
        # Threshold mode — drop stale quantiles inherited from the source tag
        # so the new entry cannot be misread as quantile-derived.
        entry.pop("rollout_quantile", None)
        entry.pop("expert_quantile", None)
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel the bool advantage column on an existing advantages "
            "parquet using a new threshold or a cross-rollout quantile. "
            "Pure CPU; advantage_continuous is not recomputed."
        )
    )
    parser.add_argument(
        "--dataset_paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more LeRobot dataset roots (each containing a meta/ dir).",
    )
    parser.add_argument(
        "--source_tag",
        required=True,
        help="Tag of the existing advantages parquet to read as the baseline.",
    )
    parser.add_argument(
        "--new_tag",
        required=True,
        help="Output tag; written to meta/advantages_{new_tag}.parquet.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("threshold", "quantile"),
        help=(
            "Labelling mode. 'threshold' uses --positive_threshold as-is "
            "(sft frames stay True); 'quantile' labels the top "
            "--rollout_quantile fraction of rollout frames and, when "
            "--expert_quantile is set, the top --expert_quantile fraction of "
            "sft frames (scored over independent pools)."
        ),
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=None,
        help=(
            "Signed-score threshold (required when --mode=threshold). Must "
            "be in [-1, 1] to match advantage_continuous's ensemble range; "
            "the lower bound is a soft guideline — we raise only on "
            "egregiously out-of-range input."
        ),
    )
    parser.add_argument(
        "--rollout_quantile",
        type=float,
        default=None,
        help=(
            "Top fraction of rollout frames to label True (required when "
            "--mode=quantile). Must be in (0, 1); e.g. 0.3 ⇒ top 30%%."
        ),
    )
    parser.add_argument(
        "--expert_quantile",
        type=float,
        default=None,
        help=(
            "Top fraction of sft (expert) frames to label True (optional, "
            "--mode=quantile only). Must be in (0, 1). Omit to label every "
            "sft frame True (historical default)."
        ),
    )
    parser.add_argument(
        "--positive_quantile",
        type=float,
        default=None,
        help="Deprecated alias for --rollout_quantile.",
    )
    parser.add_argument(
        "--dataset_types",
        nargs="+",
        default=None,
        choices=("sft", "rollout"),
        help=(
            "One entry per --dataset_paths, overriding mixture_config. If "
            "omitted, dataset_type is read from tags[source_tag].dataset_type."
        ),
    )
    args = parser.parse_args(argv)

    if args.new_tag == args.source_tag:
        parser.error("--new_tag must differ from --source_tag")
    if args.mode == "threshold":
        if args.positive_threshold is None:
            parser.error("--positive_threshold is required when --mode=threshold")
        if not (-1.0 <= args.positive_threshold <= 1.0):
            parser.error(
                "--positive_threshold must be in [-1, 1] (it is a signed-score "
                f"threshold); got {args.positive_threshold}"
            )
        for q_name in ("rollout_quantile", "expert_quantile", "positive_quantile"):
            if getattr(args, q_name) is not None:
                parser.error(f"--{q_name} is only valid with --mode=quantile")
    else:  # quantile
        # Resolve the deprecated --positive_quantile alias into rollout_quantile.
        if args.rollout_quantile is not None and args.positive_quantile is not None:
            parser.error(
                "set only one of --rollout_quantile / --positive_quantile "
                "(the latter is a deprecated alias for the former)"
            )
        if args.rollout_quantile is None and args.positive_quantile is not None:
            args.rollout_quantile = args.positive_quantile
        if args.rollout_quantile is None:
            parser.error(
                "--rollout_quantile is required when --mode=quantile "
                "(--positive_quantile accepted as a deprecated alias)"
            )
        if not (0.0 < args.rollout_quantile < 1.0):
            parser.error(
                "--rollout_quantile must be in (0, 1) — fraction of top "
                f"rollout frames; got {args.rollout_quantile}"
            )
        if args.expert_quantile is not None and not (0.0 < args.expert_quantile < 1.0):
            parser.error(
                "--expert_quantile must be in (0, 1) — fraction of top sft "
                f"frames; got {args.expert_quantile}"
            )
        if args.positive_threshold is not None:
            parser.error("--positive_threshold is only valid with --mode=threshold")
    if args.dataset_types is not None and len(args.dataset_types) != len(
        args.dataset_paths
    ):
        parser.error(
            f"--dataset_types length {len(args.dataset_types)} does not match "
            f"--dataset_paths length {len(args.dataset_paths)}"
        )
    return args


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    if args.mode == "quantile" and args.positive_quantile is not None:
        logger.warning(
            "--positive_quantile is a deprecated alias for --rollout_quantile; "
            "please use --rollout_quantile."
        )

    # ---- Phase 1: load every source parquet + resolve dataset_type ----
    # We resolve types up-front (not lazily) so a missing metadata entry
    # fails loud before we pool scores or start writing outputs.
    loaded: list[tuple[Path, pd.DataFrame, Path, str, dict[str, Any]]] = []
    for i, ds_path in enumerate(args.dataset_paths):
        ds_path = ds_path.resolve()
        if not ds_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {ds_path}")
        cli_type = args.dataset_types[i] if args.dataset_types is not None else None
        dataset_type = _resolve_dataset_type(ds_path, args.source_tag, cli_type)
        mix = _read_mixture_config(ds_path)
        tags = mix.get("tags") or {}
        source_meta = tags.get(args.source_tag) if isinstance(tags, dict) else {}
        if not isinstance(source_meta, dict):
            raise RuntimeError(
                f"mixture_config tag {args.source_tag!r} at {ds_path} is not a mapping"
            )
        df, src_path = _load_source_parquet(ds_path, args.source_tag)
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
    # Quantile mode scores rollout and sft over independent pools. When
    # --expert_quantile is omitted, sft frames stay all-True (the historical
    # convention) but still record the rollout threshold for provenance.
    expert_threshold: Optional[float] = None
    if args.mode == "threshold":
        rollout_threshold = float(args.positive_threshold)
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
                "--mode=quantile requires at least one rollout dataset; every "
                "passed dataset resolved to type='sft'."
            )
        combined = np.concatenate(rollout_scores)
        rollout_threshold = float(
            np.percentile(combined, (1.0 - args.rollout_quantile) * 100.0)
        )
        n_positive_rollout = int((combined > rollout_threshold).sum())
        logger.info(
            "Mode: quantile rollout_quantile=%.3f (top %.1f%% of %d rollout "
            "frames) → rollout_threshold=%.4f; %d/%d rollout frames > threshold "
            "(range=[%.4f, %.4f])",
            args.rollout_quantile,
            args.rollout_quantile * 100.0,
            len(combined),
            rollout_threshold,
            n_positive_rollout,
            len(combined),
            float(combined.min()),
            float(combined.max()),
        )
        if args.expert_quantile is not None:
            sft_scores = [
                df["advantage_continuous"].values
                for _, df, _, dt, _ in loaded
                if dt == "sft"
            ]
            if sft_scores:
                combined_sft = np.concatenate(sft_scores)
                expert_threshold = float(
                    np.percentile(combined_sft, (1.0 - args.expert_quantile) * 100.0)
                )
                n_positive_sft = int((combined_sft > expert_threshold).sum())
                logger.info(
                    "expert_quantile=%.3f (top %.1f%% of %d sft frames) → "
                    "expert_threshold=%.4f; %d/%d sft frames > threshold "
                    "(range=[%.4f, %.4f])",
                    args.expert_quantile,
                    args.expert_quantile * 100.0,
                    len(combined_sft),
                    expert_threshold,
                    n_positive_sft,
                    len(combined_sft),
                    float(combined_sft.min()),
                    float(combined_sft.max()),
                )
            else:
                logger.warning(
                    "--expert_quantile=%.3f set but no sft datasets passed; "
                    "expert quantile is ignored.",
                    args.expert_quantile,
                )
        else:
            logger.info(
                "--expert_quantile not set → every sft frame labelled positive."
            )

    # ---- Phase 3: relabel + write each dataset ----
    for ds_path, df, src_path, dataset_type, source_meta in loaded:
        is_sft = dataset_type == "sft"
        # Threshold actually applied to the bool label. None ⇒ force every
        # frame True (sft with no expert quantile).
        label_threshold = expert_threshold if is_sft else rollout_threshold
        new_df = _relabel(df, positive_threshold=label_threshold)
        out_path = ds_path / "meta" / f"advantages_{args.new_tag}.parquet"
        if out_path.exists():
            logger.warning("Overwriting existing %s", out_path)
        new_df.to_parquet(out_path, index=False)
        num_positive = int(new_df["advantage"].sum())
        total_samples = int(len(new_df))

        # For all-True sft fall back to the rollout threshold so the entry
        # still carries a numeric value (matches compute_advantages_ensemble).
        recorded_threshold = (
            label_threshold if label_threshold is not None else rollout_threshold
        )
        new_entry = _build_new_tag_entry(
            source_meta,
            source_tag=args.source_tag,
            recorded_threshold=recorded_threshold,
            label_mode=args.mode,
            rollout_quantile=(
                float(args.rollout_quantile) if args.mode == "quantile" else None
            ),
            expert_quantile=(
                float(args.expert_quantile)
                if (args.mode == "quantile" and args.expert_quantile is not None)
                else None
            ),
            dataset_type=dataset_type,
            total_samples=total_samples,
            num_positive=num_positive,
        )
        mix_path = _write_mixture_config_tag(ds_path, args.new_tag, new_entry)
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


if __name__ == "__main__":
    main()
