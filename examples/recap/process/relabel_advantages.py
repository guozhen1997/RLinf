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

    * ``--mode quantile --positive_quantile Q``  (Q in (0, 1))
          Pool ``advantage_continuous`` across every rollout dataset passed
          in ``--dataset_paths``, compute the ``(1 - Q)``-th percentile
          ``unified_threshold`` on that pool, then:
              rollout: ``advantage = advantage_continuous > unified_threshold``
              sft:     ``advantage = True``

``dataset_type`` per path is resolved in this order: ``--dataset_types``
(CLI) → ``tags[source_tag].dataset_type`` in ``mixture_config.yaml``. If
neither is available the script raises — we never silently default to
``rollout`` or ``sft``.

The new tag's metadata inherits everything from ``tags[source_tag]`` (so
ensemble_size, inference_mode, etc. survive) and then overrides
``positive_threshold``, ``label_mode``, ``total_samples``, ``num_positive``,
``dataset_type``, ``derived_from_tag`` — plus ``positive_quantile`` when
``--mode=quantile``.

Example — threshold-only relabel on two datasets::

    python relabel_advantages.py \\
        --dataset_paths /p1 /p2 \\
        --source_tag existing_tag --new_tag tight_threshold \\
        --mode threshold --positive_threshold 0.6

Example — top-30% quantile across rollout datasets (sft stays True)::

    python relabel_advantages.py \\
        --dataset_paths /sft_ds /rollout_ds_a /rollout_ds_b \\
        --dataset_types sft rollout rollout \\
        --source_tag existing_tag --new_tag q30 \\
        --mode quantile --positive_quantile 0.3
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
    dataset_type: str,
    threshold: float,
) -> pd.DataFrame:
    """Rewrite ``advantage`` on a copy of ``df``; keep every other column."""
    out = df.copy()
    if dataset_type == "sft":
        out["advantage"] = True
    elif dataset_type == "rollout":
        out["advantage"] = out["advantage_continuous"] > float(threshold)
    else:  # unreachable — _resolve_dataset_type guards both sides
        raise ValueError(f"unknown dataset_type {dataset_type!r}")
    return out


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------


def _build_new_tag_entry(
    source_meta: dict[str, Any],
    *,
    source_tag: str,
    unified_threshold: float,
    label_mode: str,
    positive_quantile: Optional[float],
    dataset_type: str,
    total_samples: int,
    num_positive: int,
) -> dict[str, Any]:
    """Inherit source metadata, then override the fields this script owns."""
    entry = dict(source_meta) if isinstance(source_meta, dict) else {}
    entry["positive_threshold"] = float(unified_threshold)
    entry["label_mode"] = str(label_mode)
    entry["dataset_type"] = str(dataset_type)
    entry["total_samples"] = int(total_samples)
    entry["num_positive"] = int(num_positive)
    entry["derived_from_tag"] = str(source_tag)
    if label_mode == "quantile":
        if positive_quantile is None:
            raise ValueError("label_mode='quantile' requires positive_quantile")
        entry["positive_quantile"] = float(positive_quantile)
    else:
        # Threshold mode — drop a stale positive_quantile inherited from the
        # source tag so the new entry cannot be misread as quantile-derived.
        entry.pop("positive_quantile", None)
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
            "Labelling mode. 'threshold' uses --positive_threshold as-is; "
            "'quantile' pools advantage_continuous across rollout datasets "
            "and labels the top --positive_quantile fraction as True."
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
        "--positive_quantile",
        type=float,
        default=None,
        help=(
            "Top fraction of rollout frames to label True (required when "
            "--mode=quantile). Must be in (0, 1); e.g. 0.3 ⇒ top 30%%."
        ),
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
        if args.positive_quantile is not None:
            parser.error("--positive_quantile is only valid with --mode=quantile")
    else:  # quantile
        if args.positive_quantile is None:
            parser.error("--positive_quantile is required when --mode=quantile")
        if not (0.0 < args.positive_quantile < 1.0):
            parser.error(
                "--positive_quantile must be in (0, 1) — fraction of top "
                f"rollout frames; got {args.positive_quantile}"
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

    # ---- Phase 2: pick the unified threshold ----
    # Quantile mode pools over rollout only — sft frames are ignored when
    # picking the cut-off (they are labelled True by convention) but still
    # receive the resulting unified_threshold in their tag metadata.
    if args.mode == "threshold":
        unified_threshold = float(args.positive_threshold)
        logger.info("Mode: threshold; unified_threshold=%.4f", unified_threshold)
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
        unified_threshold = float(
            np.percentile(combined, (1.0 - args.positive_quantile) * 100.0)
        )
        n_positive_rollout = int((combined > unified_threshold).sum())
        logger.info(
            "Mode: quantile (top %.1f%% of %d rollout frames) → "
            "unified_threshold=%.4f; %d/%d rollout frames > threshold "
            "(range=[%.4f, %.4f])",
            args.positive_quantile * 100.0,
            len(combined),
            unified_threshold,
            n_positive_rollout,
            len(combined),
            float(combined.min()),
            float(combined.max()),
        )

    # ---- Phase 3: relabel + write each dataset ----
    for ds_path, df, src_path, dataset_type, source_meta in loaded:
        new_df = _relabel(df, dataset_type=dataset_type, threshold=unified_threshold)
        out_path = ds_path / "meta" / f"advantages_{args.new_tag}.parquet"
        if out_path.exists():
            logger.warning("Overwriting existing %s", out_path)
        new_df.to_parquet(out_path, index=False)
        num_positive = int(new_df["advantage"].sum())
        total_samples = int(len(new_df))

        new_entry = _build_new_tag_entry(
            source_meta,
            source_tag=args.source_tag,
            unified_threshold=unified_threshold,
            label_mode=args.mode,
            positive_quantile=(
                float(args.positive_quantile) if args.mode == "quantile" else None
            ),
            dataset_type=dataset_type,
            total_samples=total_samples,
            num_positive=num_positive,
        )
        mix_path = _write_mixture_config_tag(ds_path, args.new_tag, new_entry)
        logger.info(
            "  wrote %s (rows=%d, positive=%d/%d, %.1f%%); updated %s",
            out_path,
            total_samples,
            num_positive,
            total_samples,
            100.0 * num_positive / max(total_samples, 1),
            mix_path,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
