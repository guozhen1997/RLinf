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

"""End-to-end advantage computation for STEAM CFG-RL training.

Loads a trained ensemble :class:`SteamCriticModel`, scores every anchor frame to
a signed advantage in ``[-1, 1]`` (the worst-of-N ensemble expectation), then
labels frames positive/negative under a ``threshold`` or ``quantile`` rule and
writes ``meta/advantages_{tag}.parquet`` plus a per-tag ``mixture_config.yaml``
entry per dataset. Multi-GPU sharding runs over ``torchrun``; threshold
selection and writes happen on rank 0.

See the STEAM pipeline docs for the column schema, ``label_mode`` semantics, and
launch commands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from rlinf.data.process.distributed import (
    cleanup_distributed,
    setup_distributed,
)
from rlinf.data.process.steam.inference import run_inference_for_dataset
from rlinf.data.process.steam.labelling import (
    CANONICAL_OUTPUT_COLS,
    apply_boolean_label,
    compute_advantage_continuous,
    quantile_threshold,
    resolve_quantile_alias,
)
from rlinf.data.process.steam.mixture_config import update_advantage_tag
from rlinf.models.embodiment.value.steam.ensemble_modeling_critic import coerce_to_ensemble
from rlinf.models.embodiment.value.steam.modeling_critic import SteamCriticModel

logger = logging.getLogger(__name__)


def _resolve_quantiles(
    advantage_cfg: DictConfig,
) -> tuple[Optional[float], Optional[float]]:
    """Resolve ``(rollout_quantile, expert_quantile)`` for quantile label_mode.

    ``rollout_quantile`` is the fraction of top rollout frames labelled True;
    ``advantage.positive_quantile`` is accepted as a deprecated alias for it.
    ``expert_quantile`` is the fraction of top ``sft`` (expert) frames labelled
    True — ``None`` when unset, which keeps the historical "every sft frame is
    positive" behaviour. Range validation lives in :func:`validate_advantage_cfg`.
    """
    rollout_quantile = resolve_quantile_alias(
        advantage_cfg.get("rollout_quantile"),
        advantage_cfg.get("positive_quantile"),
    )
    expert = advantage_cfg.get("expert_quantile")
    return (
        None if rollout_quantile is None else float(rollout_quantile),
        None if expert is None else float(expert),
    )


def validate_advantage_cfg(cfg: DictConfig) -> None:
    """Hard-fail on configuration mistakes — no silent fallbacks."""
    if "advantage" not in cfg:
        raise ValueError("Config missing 'advantage' section")
    if "data" not in cfg:
        raise ValueError("Config missing 'data' section")

    ckpt = cfg.advantage.get("value_checkpoint")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"value_checkpoint does not exist: {ckpt!r}")

    label_mode = cfg.advantage.get("label_mode")
    if label_mode is None:
        raise ValueError(
            "advantage.label_mode is required; must be 'threshold' or 'quantile'. "
            "'threshold' labels advantage=True when advantage_continuous > "
            "advantage.positive_threshold; 'quantile' labels the top "
            "advantage.rollout_quantile fraction of rollout frames as True (and "
            "the top advantage.expert_quantile fraction of sft frames when set)."
        )
    label_mode = str(label_mode).lower()
    if label_mode not in ("threshold", "quantile"):
        raise ValueError(
            "advantage.label_mode must be 'threshold' or 'quantile'; got "
            f"{label_mode!r}"
        )

    if label_mode == "threshold":
        threshold = cfg.advantage.get("positive_threshold")
        if threshold is None:
            raise ValueError(
                "advantage.positive_threshold is required when "
                "advantage.label_mode='threshold'"
            )
        threshold = float(threshold)
        # ``advantage_continuous`` is a signed bin-weighted expectation in
        # ``[-1, 1]``; ``positive_threshold`` is applied directly on it, so it
        # must live in the same range — NOT in ``[0, 1]``.
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError(
                f"positive_threshold must be in [-1, 1] (it is a signed-score "
                f"threshold matching ensemble_signed_score's range); got {threshold}"
            )
    else:  # label_mode == "quantile"
        rollout_quantile, expert_quantile = _resolve_quantiles(cfg.advantage)
        if rollout_quantile is None:
            raise ValueError(
                "advantage.rollout_quantile is required when "
                "advantage.label_mode='quantile' (e.g. 0.3 ⇒ top 30% of rollout "
                "samples by advantage_continuous are labelled True). "
                "advantage.positive_quantile is accepted as a deprecated alias."
            )
        if not (0.0 < rollout_quantile < 1.0):
            raise ValueError(
                "rollout_quantile must be a fraction in (0, 1) — fraction of "
                f"top rollout samples labelled True; got {rollout_quantile}"
            )
        # expert_quantile is optional: unset ⇒ every sft frame is labelled True.
        if expert_quantile is not None and not (0.0 < expert_quantile < 1.0):
            raise ValueError(
                "expert_quantile must be a fraction in (0, 1) — fraction of top "
                f"sft (expert) samples labelled True; got {expert_quantile}"
            )

    tag = cfg.advantage.get("tag")
    if not tag:
        raise ValueError("advantage.tag is required")

    k = int(cfg.data.get("k", 0))
    if k < 1:
        raise ValueError(f"data.k must be >= 1, got {k}")

    train_paths = cfg.data.get("train_data_paths")
    if not train_paths:
        raise ValueError("data.train_data_paths is empty")
    for entry in train_paths:
        ds_type = entry.get("type")
        if ds_type not in ("sft", "rollout"):
            raise ValueError(
                f"train_data_paths entry has invalid 'type'={ds_type!r}; "
                "must be 'sft' or 'rollout'"
            )
        ds_path = entry.get("dataset_path")
        if not ds_path or not Path(ds_path).exists():
            raise FileNotFoundError(f"dataset_path does not exist: {ds_path!r}")


def _save_advantages_parquet(df: pd.DataFrame, dataset_path: str, tag: str) -> Path:
    meta_dir = Path(dataset_path) / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Dataset meta dir does not exist: {meta_dir}")
    out_path = meta_dir / f"advantages_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def compute_ensemble_advantages(cfg: DictConfig) -> None:
    """Run the full STEAM advantage computation pipeline.

    Initialises ``torch.distributed`` for torchrun, loads the ensemble critic,
    runs two phases — Phase 1 scores every dataset to ``advantage_continuous`` on
    all ranks (rank 0 accumulates), Phase 2 picks thresholds and writes parquet +
    ``mixture_config.yaml`` on rank 0 — and tears down the process group.
    """
    rank, world_size, device = setup_distributed(cfg)

    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if rank == 0:
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    validate_advantage_cfg(cfg)

    precision = cfg.advantage.model.get("precision", None)
    label_mode = str(cfg.advantage.label_mode).lower()
    # Coerce the knobs the selected mode reads eagerly so any late YAML typo
    # (e.g. "positive_treshold") trips here rather than deep inside Phase 2.
    positive_threshold_cfg: Optional[float] = (
        float(cfg.advantage.positive_threshold) if label_mode == "threshold" else None
    )
    rollout_quantile_cfg: Optional[float] = None
    expert_quantile_cfg: Optional[float] = None
    if label_mode == "quantile":
        rollout_quantile_cfg, expert_quantile_cfg = _resolve_quantiles(cfg.advantage)
        if (
            rank == 0
            and cfg.advantage.get("positive_quantile") is not None
            and cfg.advantage.get("rollout_quantile") is None
        ):
            logger.warning(
                "advantage.positive_quantile is a deprecated alias for "
                "advantage.rollout_quantile; please rename it in your config."
            )
    tag = str(cfg.advantage.tag)

    if rank == 0:
        logger.info(
            "Loading ensemble checkpoint from %s", cfg.advantage.value_checkpoint
        )
    raw_model = SteamCriticModel.from_checkpoint(
        cfg.advantage.value_checkpoint,
        device=device,
        env_type=str(cfg.data.robot_type),
        model_type=str(cfg.data.model_type),
        precision=precision,
    )
    model = coerce_to_ensemble(raw_model)

    if rank == 0:
        logger.info(
            "Ensemble loaded: ensemble_size=%d, max_token_len=%d",
            int(model.config.ensemble_size),
            int(getattr(model.config, "max_token_len", 200)),
        )

    try:
        # ---- Phase 1: GPU inference + advantage_continuous ----
        # All ranks participate in inference; rank 0 accumulates the per-frame
        # continuous scores so Phase 2 can pick a unified threshold before
        # committing bool labels. Other ranks park at the per-dataset barrier so
        # the GPU memory / NCCL state stays in lockstep with rank 0.
        collected: list[tuple[Any, pd.DataFrame]] = []
        for ds_idx, entry in enumerate(cfg.data.train_data_paths):
            if rank == 0:
                logger.info(
                    "[%d/%d] Inference on %s (type=%s)",
                    ds_idx + 1,
                    len(cfg.data.train_data_paths),
                    entry.dataset_path,
                    entry.type,
                )
            df = run_inference_for_dataset(
                model=model,
                dataset_entry=entry,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                device=device,
            )

            if rank == 0:
                df_cont = compute_advantage_continuous(df)
                collected.append((entry, df_cont))

            if world_size > 1:
                dist.barrier()

        # ---- Phase 2: pick thresholds + write parquet + update meta ----
        # Runs only on rank 0. ``rollout_threshold`` is applied to rollout
        # datasets; ``expert_threshold`` (quantile mode + advantage.expert_quantile)
        # is applied to sft datasets. In quantile mode each threshold is the
        # ``(1 - quantile)``-th percentile of advantage_continuous over its OWN
        # pool. When expert_quantile is unset, sft frames stay all-True (the
        # historical convention) but still record the rollout threshold in their
        # tag metadata for provenance.
        if rank == 0:
            expert_threshold: Optional[float] = None
            if label_mode == "quantile":
                rollout_scores: list[np.ndarray] = [
                    d["advantage_continuous"].values
                    for e, d in collected
                    if str(e.type).lower() == "rollout"
                ]
                if not rollout_scores:
                    # No rollout pool. This is only well-defined when
                    # expert_quantile is set: sft frames are then labelled by
                    # their own quantile and no rollout threshold is needed.
                    # Without expert_quantile there is nothing to threshold on.
                    if expert_quantile_cfg is None:
                        raise ValueError(
                            "advantage.label_mode='quantile' requires either at "
                            "least one rollout dataset (type='rollout') to derive "
                            "the rollout threshold, or advantage.expert_quantile "
                            "set so sft frames are labelled by their own "
                            "quantile. Neither was provided."
                        )
                    rollout_threshold = None
                    logger.info(
                        "label_mode='quantile' with no rollout datasets; "
                        "labelling sft frames by expert_quantile=%.3f only.",
                        float(expert_quantile_cfg),
                    )
                else:
                    combined_rollout = np.concatenate(rollout_scores)
                    rollout_threshold = float(
                        np.percentile(
                            combined_rollout,
                            (1.0 - float(rollout_quantile_cfg)) * 100.0,
                        )
                    )
                    logger.info(
                        "label_mode='quantile' rollout_quantile=%.3f (top %.1f%% of "
                        "%d rollout samples) → rollout_threshold=%.4f "
                        "(advantage_continuous range [%.4f, %.4f])",
                        float(rollout_quantile_cfg),
                        float(rollout_quantile_cfg) * 100.0,
                        len(combined_rollout),
                        rollout_threshold,
                        float(combined_rollout.min()),
                        float(combined_rollout.max()),
                    )
                if expert_quantile_cfg is not None:
                    sft_scores: list[np.ndarray] = [
                        d["advantage_continuous"].values
                        for e, d in collected
                        if str(e.type).lower() == "sft"
                    ]
                    if sft_scores:
                        combined_sft = np.concatenate(sft_scores)
                        expert_threshold = quantile_threshold(
                            combined_sft, float(expert_quantile_cfg)
                        )
                        logger.info(
                            "expert_quantile=%.3f (top %.1f%% of %d sft samples) "
                            "→ expert_threshold=%.4f (advantage_continuous range "
                            "[%.4f, %.4f])",
                            float(expert_quantile_cfg),
                            float(expert_quantile_cfg) * 100.0,
                            len(combined_sft),
                            expert_threshold,
                            float(combined_sft.min()),
                            float(combined_sft.max()),
                        )
                    else:
                        logger.warning(
                            "advantage.expert_quantile=%.3f set but no sft "
                            "datasets in train_data_paths; expert quantile is "
                            "ignored.",
                            float(expert_quantile_cfg),
                        )
                else:
                    logger.info(
                        "advantage.expert_quantile not set → every sft frame is "
                        "labelled positive (historical behaviour)."
                    )
            else:
                rollout_threshold = float(positive_threshold_cfg)
                logger.info(
                    "label_mode='threshold'; rollout_threshold=%.4f; sft frames "
                    "labelled all-True.",
                    rollout_threshold,
                )

            for entry, df_cont in collected:
                is_sft = str(entry.type).lower() == "sft"
                # Threshold actually applied to the bool label. ``None`` ⇒ force
                # every frame True (sft with no expert quantile).
                label_threshold = expert_threshold if is_sft else rollout_threshold
                final_df = apply_boolean_label(
                    df_cont,
                    positive_threshold=label_threshold,
                    columns=CANONICAL_OUTPUT_COLS,
                )
                # Threshold recorded in the tag metadata: for all-True sft fall
                # back to the rollout threshold so the entry still carries a
                # numeric value (matches the previous single-threshold layout).
                recorded_threshold = (
                    label_threshold
                    if label_threshold is not None
                    else rollout_threshold
                )
                out_path = _save_advantages_parquet(final_df, entry.dataset_path, tag)
                num_positive = int(final_df["advantage"].sum())
                total_samples = int(len(final_df))
                mix_path = update_advantage_tag(
                    dataset_path=entry.dataset_path,
                    tag=tag,
                    positive_threshold=recorded_threshold,
                    ensemble_size=int(model.config.ensemble_size),
                    num_bins=int(getattr(model.config, "num_bins", 2)),
                    total_samples=total_samples,
                    num_positive=num_positive,
                    dataset_type=str(entry.type),
                    label_mode=label_mode,
                    rollout_quantile=(
                        float(rollout_quantile_cfg)
                        if label_mode == "quantile"
                        else None
                    ),
                    expert_quantile=(
                        float(expert_quantile_cfg)
                        if (label_mode == "quantile" and expert_quantile_cfg is not None)
                        else None
                    ),
                )
                logger.info(
                    "Wrote %s (type=%s, rows=%d, positive=%d/%d, "
                    "raw_score_avg=%.4f, label_mode=%s). Updated %s",
                    out_path,
                    str(entry.type),
                    total_samples,
                    num_positive,
                    total_samples,
                    float(final_df["ensemble_signed_score"].mean()),
                    label_mode,
                    mix_path,
                )
    finally:
        # Drop the large CUDA modules before NCCL teardown; the process is
        # exiting, but NCCL may still need a little device memory to shut down.
        model = None
        raw_model = None
        cleanup_distributed()


__all__ = [
    "validate_advantage_cfg",
    "compute_ensemble_advantages",
]
