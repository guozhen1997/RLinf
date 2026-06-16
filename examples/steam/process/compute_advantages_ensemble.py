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

"""
Compute advantages for CFG-RL training using a trained ensemble
SteamCriticModel.

Per-frame ``ensemble_signed_score`` = ensemble-aggregated signed bin-weighted
expectation in ``[-1, 1]`` for the pair ``(frame_t, frame_{t+k})`` (see
:class:`rlinf.models.embodiment.steam.modeling_critic.
CriticOutput.predicted_values`). For binary (``num_bins=2``) it degenerates to
``2 · P(progress) - 1``; for multi-bin it is a signed bin expectation.

The final per-frame scalar written as ``advantage_continuous`` is the
``ensemble_signed_score``.

The boolean ``advantage`` label is derived from ``advantage_continuous``
under one of two mutually-exclusive modes selected via
``advantage.label_mode``:

    * ``label_mode: threshold`` — ``advantage = advantage_continuous >
      advantage.positive_threshold``, where ``positive_threshold`` is a
      signed-score threshold in ``[-1, 1]`` (NOT a probability). ``sft``
      frames are always labelled True (success demos by construction).
    * ``label_mode: quantile`` — separate top-fraction quantiles per data
      type, scored over independent pools:

        - ``advantage.rollout_quantile`` (required; ``positive_quantile`` is
          accepted as a deprecated alias) pools ``advantage_continuous``
          across every ``rollout`` dataset and labels the top
          ``rollout_quantile`` fraction True. e.g. ``0.3`` ⇒ top 30% rollout
          frames by ``advantage_continuous``.
        - ``advantage.expert_quantile`` (optional) pools
          ``advantage_continuous`` across every ``sft`` dataset and labels the
          top ``expert_quantile`` fraction True. When omitted, every ``sft``
          frame is labelled True (the historical behaviour).

      Both quantiles must be in ``(0, 1)``. The expert pool never sees rollout
      frames and vice-versa.

``advantage.label_mode`` is **required** — there is no default. Each dataset
records the threshold applied to it in ``tags[tag].positive_threshold``: the
rollout-pool percentile for ``rollout`` datasets, the expert-pool percentile
for ``sft`` datasets when ``expert_quantile`` is set (otherwise the
rollout-pool percentile, for provenance, while the frames stay all-True).

Output: ``meta/advantages_{tag}.parquet`` per dataset, with columns
``[episode_index, frame_index, advantage, advantage_continuous,
   ensemble_signed_score, p_progress_mean, p_progress_min, p_progress_variance,
   member_values, expected_stride_normalized, entropy_aggregated,
   entropy_member_mean, entropy_member_variance]``. Terminal frames are
backfilled with zero
``ensemble_signed_score``. ``mixture_config.yaml`` under ``meta/`` is updated
with a per-tag entry that now also records ``label_mode`` (and
``rollout_quantile`` / ``expert_quantile`` in quantile mode).

Usage:
    python compute_advantages_ensemble.py \\
        --config-name compute_advantages_ensemble

    # Multi-GPU
    torchrun --nproc_per_node=4 compute_advantages_ensemble.py \\
        --config-name compute_advantages_ensemble

    # Override at the CLI: quantile mode, top 30% rollout + top 90% expert
    torchrun --nproc_per_node=4 compute_advantages_ensemble.py \\
        --config-name compute_advantages_ensemble \\
        advantage.label_mode=quantile \\
        advantage.rollout_quantile=0.3 advantage.expert_quantile=0.9
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Make the rlinf package importable regardless of the cwd the user launched from.
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent.parent))

# Install the libav/libdav1d stderr filter before the heavy torch / torchcodec
# imports so the fd=2 redirect is in place before libav loads.
from rlinf.utils.logging import silence_libav_logs  # noqa: E402

silence_libav_logs()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from rlinf.data.datasets.steam.binning import expected_signed_stride  # noqa: E402
from rlinf.data.datasets.steam.pair_dataset import (  # noqa: E402
    BinaryPairDataCollator,
    BinaryPairInferenceDataset,
)
from rlinf.models.embodiment.steam.ensemble_modeling_critic import (  # noqa: E402
    EnsembleSteamCriticModel,
)
from rlinf.models.embodiment.steam.modeling_critic import (  # noqa: E402
    SteamCriticModel,
)
from rlinf.utils.dist_inference import (  # noqa: E402
    cleanup_distributed,
    gather_all_advantages,
    get_shard_indices,
    setup_distributed,
)


def _entropy_nats(probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy in nats; shape ``[..., num_bins]`` -> ``[...]``."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_quantiles(
    advantage_cfg: DictConfig,
) -> tuple[Optional[float], Optional[float]]:
    """Resolve ``(rollout_quantile, expert_quantile)`` for quantile label_mode.

    ``rollout_quantile`` is the fraction of top rollout frames labelled True;
    ``advantage.positive_quantile`` is accepted as a deprecated alias for it.
    ``expert_quantile`` is the fraction of top ``sft`` (expert) frames labelled
    True — ``None`` when unset, which keeps the historical "every sft frame is
    positive" behaviour. Range validation lives in :func:`_validate_cfg`; this
    only resolves the alias and coerces to ``float``.
    """
    rollout = advantage_cfg.get("rollout_quantile")
    positive = advantage_cfg.get("positive_quantile")
    if rollout is not None and positive is not None:
        raise ValueError(
            "Set only one of advantage.rollout_quantile and "
            "advantage.positive_quantile (the latter is a deprecated alias for "
            "the former)."
        )
    rollout_quantile = rollout if rollout is not None else positive
    expert = advantage_cfg.get("expert_quantile")
    return (
        None if rollout_quantile is None else float(rollout_quantile),
        None if expert is None else float(expert),
    )


def _validate_cfg(cfg: DictConfig) -> None:
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
        # ``[-1, 1]`` (see module docstring). ``positive_threshold`` is applied
        # directly on it, so it must live in the same range — NOT in ``[0, 1]``.
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


def _move_to_device(obj: Any, device: str):
    """Recursive .to(device) for tensors nested in dicts/lists."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def _coerce_inference_model(model) -> EnsembleSteamCriticModel:
    """Return an ensemble-shaped inference model, allowing ensemble_size=1."""
    if isinstance(model, EnsembleSteamCriticModel):
        return model

    if isinstance(model, SteamCriticModel):
        ensemble_size = int(getattr(model.config, "ensemble_size", 1))
        if ensemble_size == 1:
            wrapped = EnsembleSteamCriticModel(model.config, [model])
            processor = getattr(model, "processor", None)
            device = getattr(model, "_device", None)
            if processor is not None and device is not None:
                wrapped.attach_runtime_assets(processor, device)
            elif processor is not None:
                wrapped.processor = processor
            elif device is not None:
                wrapped._device = device
            wrapped.eval()
            return wrapped

    raise RuntimeError(
        "Expected an ensemble-compatible checkpoint, but "
        "SteamCriticModel.from_checkpoint returned "
        f"{type(model).__name__}. Check that the saved config.json has "
        "ensemble_size >= 1."
    )


def _build_dataloader(
    dataset: Dataset,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    collate_fn,
) -> tuple[DataLoader, int]:
    """Shard the dataset across ranks and wrap in a DataLoader."""
    total = len(dataset)
    if total == 0:
        raise RuntimeError("Inference dataset is empty")
    start, end = get_shard_indices(total, rank, world_size)
    shard_indices = list(range(start, end))
    shard = Subset(dataset, shard_indices)
    loader = DataLoader(
        shard,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return loader, len(shard_indices)


def _records_from_predict(
    out,
    batch: dict[str, Any],
    *,
    num_bins: int,
    stride_k: int,
) -> list[dict[str, Any]]:
    """Per-sample row dicts from a single CriticOutput + batch metadata.

    Always emits the same column set regardless of ``num_bins``:
    binary-mode output computes ``expected_stride_normalized`` and the
    entropy columns from the 2-wide softmax, so downstream readers don't
    need to branch on mode.
    """
    aggregated = out.predicted_values.detach().to("cpu", dtype=torch.float32)
    mean = out.prediction_mean.detach().to("cpu", dtype=torch.float32)
    minv = out.prediction_min.detach().to("cpu", dtype=torch.float32)
    var = out.prediction_variance.detach().to("cpu", dtype=torch.float32)
    members = out.member_predicted_values.detach().to("cpu", dtype=torch.float32)
    # Bin-level quantities. ``out.probs`` is the aggregated softmax
    # ([B, num_bins]); ``out.member_probs`` is [E, B, num_bins] and was
    # added to EnsembleCriticOutput in phase A. For ensembles of size 1
    # (single-model wrapper) E==1 and member-variance columns fall to 0.
    agg_probs_np = out.probs.detach().to("cpu", dtype=torch.float32).numpy()
    member_probs = out.member_probs
    if member_probs is None:
        raise RuntimeError(
            "compute_advantages_ensemble expects EnsembleCriticOutput.member_probs "
            "to be populated; got None. Ensure the checkpoint routes through the "
            "ensemble wrapper, including the single-member ensemble_size=1 case."
        )
    member_probs_np = member_probs.detach().to("cpu", dtype=torch.float32).numpy()

    es = expected_signed_stride(agg_probs_np, stride_k, num_bins) / float(stride_k)
    entropy_agg = _entropy_nats(agg_probs_np)  # [B]
    entropy_members = _entropy_nats(member_probs_np)  # [E, B]
    entropy_member_mean = entropy_members.mean(axis=0)  # [B]
    entropy_member_variance = entropy_members.var(axis=0, ddof=0)  # [B]

    episodes = batch["episode"].tolist()
    frame_t = batch["frame_idx_t"].tolist()

    rows: list[dict[str, Any]] = []
    bsize = aggregated.shape[0]
    for i in range(bsize):
        rows.append(
            {
                "episode_index": int(episodes[i]),
                "frame_index": int(frame_t[i]),
                # ``ensemble_signed_score`` = ``out.predicted_values`` — a
                # signed bin-weighted expectation in ``[-1, 1]`` (NOT a
                # probability). Renamed from the historical
                # ``p_progress_aggregated``, which was misleading because it is
                # not ``P(progress)`` for num_bins > 2; for num_bins == 2 it
                # degenerates to ``2 · P(progress) - 1``.
                "ensemble_signed_score": float(aggregated[i].item()),
                "p_progress_mean": float(mean[i].item()),
                "p_progress_min": float(minv[i].item()),
                "p_progress_variance": float(var[i].item()),
                "member_values": [float(x) for x in members[:, i].tolist()],
                # Multi-bin additive columns. Binary (num_bins=2) still
                # gets these: expected_stride_normalized degenerates to a
                # monotone function of ensemble_signed_score, entropy is
                # Bernoulli.
                "expected_stride_normalized": float(es[i]),
                "entropy_aggregated": float(entropy_agg[i]),
                "entropy_member_mean": float(entropy_member_mean[i]),
                "entropy_member_variance": float(entropy_member_variance[i]),
            }
        )
    return rows


def _build_terminal_frame_rows(
    *,
    episode_lengths: list[int],
    member_count: int,
) -> pd.DataFrame:
    """Build default-negative rows for each episode's terminal frame."""
    rows: list[dict[str, Any]] = []
    zero_members = [0.0] * max(1, int(member_count))
    for episode_index, episode_length in enumerate(episode_lengths):
        if int(episode_length) < 1:
            continue
        rows.append(
            {
                "episode_index": int(episode_index),
                "frame_index": int(episode_length) - 1,
                "ensemble_signed_score": 0.0,
                "p_progress_mean": 0.0,
                "p_progress_min": 0.0,
                "p_progress_variance": 0.0,
                "member_values": list(zero_members),
                # Terminal default: assume neutral signed stride (E[s]/K = 0)
                # and zero entropy — matches the "default neutral" intent of
                # the 0.0 ensemble_signed_score fill.
                "expected_stride_normalized": 0.0,
                "entropy_aggregated": 0.0,
                "entropy_member_mean": 0.0,
                "entropy_member_variance": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _append_missing_terminal_rows(
    df: pd.DataFrame,
    *,
    episode_lengths: list[int],
    member_count: int,
) -> tuple[pd.DataFrame, int]:
    """Append any missing terminal frames with zero progress defaults."""
    terminal_rows = _build_terminal_frame_rows(
        episode_lengths=episode_lengths,
        member_count=member_count,
    )
    if terminal_rows.empty:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        return df, 0

    if len(df) == 0:
        combined = terminal_rows.sort_values(
            [
                "episode_index",
                "frame_index",
            ]
        ).reset_index(drop=True)
        return combined, len(combined)

    existing_keys = set(
        map(
            tuple,
            df[["episode_index", "frame_index"]].astype(int).values.tolist(),
        )
    )
    missing_terminal_rows = terminal_rows[
        [
            (int(row.episode_index), int(row.frame_index)) not in existing_keys
            for row in terminal_rows.itertuples(index=False)
        ]
    ]
    combined = pd.concat([df, missing_terminal_rows], ignore_index=True)
    combined = combined.sort_values(["episode_index", "frame_index"]).reset_index(
        drop=True
    )
    return combined, len(missing_terminal_rows)


def _run_inference_for_dataset(
    *,
    model: EnsembleSteamCriticModel,
    dataset_entry: DictConfig,
    cfg: DictConfig,
    rank: int,
    world_size: int,
    device: str,
) -> pd.DataFrame:
    """Run ensemble inference on one dataset; return a sorted DataFrame on rank 0."""
    dataset = BinaryPairInferenceDataset(
        dataset_path=dataset_entry.dataset_path,
        camera_keys=list(cfg.data.camera_keys),
        k=int(cfg.data.k),
        prompt=cfg.data.get("prompt", None),
        dataset_type=dataset_entry.type,
    )

    collator = BinaryPairDataCollator(
        processor=model.members[0].processor,
        max_length=int(getattr(model.config, "max_token_len", 200)),
        train=False,
    )

    loader, shard_size = _build_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=int(cfg.advantage.batch_size),
        num_workers=int(cfg.advantage.num_dataloader_workers_per_gpu),
        prefetch_factor=int(cfg.advantage.prefetch_factor),
        collate_fn=collator,
    )

    if rank == 0:
        logger.info(
            "Dataset %s: total_anchors=%d, rank0 shard_size=%d, batch_size=%d, "
            "num_workers=%d, prefetch_factor=%d",
            dataset_entry.dataset_path,
            len(dataset),
            shard_size,
            int(cfg.advantage.batch_size),
            int(cfg.advantage.num_dataloader_workers_per_gpu),
            int(cfg.advantage.prefetch_factor),
        )

    num_bins = int(getattr(model.config, "num_bins", 2))
    stride_k = int(cfg.data.k)

    local_rows: list[dict[str, Any]] = []
    pbar = tqdm(
        loader,
        desc=f"[rank{rank}] {Path(dataset_entry.dataset_path).name}",
        disable=(rank != 0),
        total=len(loader),
    )
    first_batch_started_at = time.monotonic()
    if rank == 0:
        logger.info(
            "Waiting for first DataLoader batch from %s; large video batches can "
            "be silent until a full batch is decoded.",
            dataset_entry.dataset_path,
        )
    for batch_idx, batch in enumerate(pbar):
        if batch_idx == 0 and rank == 0:
            logger.info(
                "Received first DataLoader batch from %s after %.1fs",
                dataset_entry.dataset_path,
                time.monotonic() - first_batch_started_at,
            )
        observation = _move_to_device(batch["observation"], device)
        with torch.inference_mode():
            out = model.predict(observation)
        local_rows.extend(
            _records_from_predict(out, batch, num_bins=num_bins, stride_k=stride_k)
        )

    local_df = pd.DataFrame(local_rows)
    if world_size > 1:
        dist.barrier()
        df = gather_all_advantages(local_df, rank, world_size)
    else:
        df = local_df

    if rank == 0:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        episode_lengths = [
            dataset._source.episode_length(ep)
            for ep in range(dataset._source.num_episodes())
        ]
        df, num_appended = _append_missing_terminal_rows(
            df,
            episode_lengths=episode_lengths,
            member_count=int(model.config.ensemble_size),
        )
        if num_appended > 0:
            logger.info(
                "Appended %d terminal frames with default-negative scores for %s",
                num_appended,
                dataset_entry.dataset_path,
            )
    return df


_CANONICAL_OUTPUT_COLS: list[str] = [
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


def _compute_advantage_continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Write ``advantage_continuous`` onto a copy of ``df`` — no bool label.

    Kept separate from :func:`_apply_boolean_label` so two-phase main() can
    compute continuous scores for every dataset first, then decide on a
    single (possibly quantile-derived) threshold before emitting the bool.
    """
    if df.empty:
        raise RuntimeError(
            "Empty DataFrame after gather — no predictions were produced for this dataset"
        )
    out = df.copy()
    out["advantage_continuous"] = out["ensemble_signed_score"]
    return out


def _apply_boolean_label(
    df: pd.DataFrame,
    *,
    positive_threshold: Optional[float],
) -> pd.DataFrame:
    """Add the boolean ``advantage`` column and select canonical columns.

    ``positive_threshold`` is the effective signed-score threshold for THIS
    dataset: ``advantage = advantage_continuous > positive_threshold``. Pass
    ``None`` to force ``advantage=True`` for every frame — used for ``sft``
    datasets when no expert quantile/threshold is configured (expert demos are
    positive by construction). The caller owns the per-dataset policy (which
    threshold, or ``None``) so quantile mode can apply distinct expert/rollout
    thresholds to the two data types.
    """
    if "advantage_continuous" not in df.columns:
        raise ValueError(
            "_apply_boolean_label requires 'advantage_continuous' column; run "
            "_compute_advantage_continuous first."
        )
    out = df.copy()
    if positive_threshold is None:
        out["advantage"] = True
    else:
        out["advantage"] = out["advantage_continuous"] > float(positive_threshold)

    return out[list(_CANONICAL_OUTPUT_COLS)]


def _save_advantages_parquet(df: pd.DataFrame, dataset_path: str, tag: str) -> Path:
    meta_dir = Path(dataset_path) / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Dataset meta dir does not exist: {meta_dir}")
    out_path = meta_dir / f"advantages_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def _update_mixture_config(
    *,
    dataset_path: str,
    tag: str,
    positive_threshold: float,
    inference_mode: str,
    ensemble_size: int,
    num_bins: int,
    total_samples: int,
    num_positive: int,
    dataset_type: str,
    label_mode: str,
    rollout_quantile: Optional[float] = None,
    expert_quantile: Optional[float] = None,
) -> Path:
    """Merge a per-tag entry into ``<dataset>/meta/mixture_config.yaml``.

    Writes **only** under ``tags[tag]``; top-level fields (``advantage_tag``,
    ``datasets``, ``unified_threshold``, ``global_return_min/max``, …) are
    treated as read-only.

    ``label_mode`` records which rule produced the bool label:
    ``"threshold"`` (fixed ``positive_threshold``) or ``"quantile"``. In
    quantile mode ``rollout_quantile`` (required) and ``expert_quantile``
    (optional) record the per-pool top-fractions, while ``positive_threshold``
    holds the threshold actually applied to THIS dataset (the rollout-pool
    percentile for ``rollout`` datasets, the expert-pool percentile for
    ``sft`` datasets that were filtered by ``expert_quantile``).

    ``dataset_type`` is recorded so downstream recompute / viz tooling can
    read it without re-parsing the parquet schema.
    """
    if label_mode not in ("threshold", "quantile"):
        raise ValueError(
            f"label_mode must be 'threshold' or 'quantile', got {label_mode!r}"
        )
    if label_mode == "quantile" and rollout_quantile is None:
        raise ValueError("label_mode='quantile' requires rollout_quantile")

    meta_dir = Path(dataset_path) / "meta"
    cfg_path = meta_dir / "mixture_config.yaml"
    existing: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            loaded = yaml.safe_load(f)
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise RuntimeError(
                    f"mixture_config.yaml at {cfg_path} is not a mapping; refusing to overwrite"
                )
            existing = loaded
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(
            f"mixture_config.yaml 'tags' field at {cfg_path} is not a mapping"
        )
    entry: dict[str, Any] = {
        "positive_threshold": float(positive_threshold),
        "label_mode": str(label_mode),
        "inference_mode": str(inference_mode),
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
    tags[str(tag)] = entry
    existing["tags"] = tags
    with open(cfg_path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)
    return cfg_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_advantages_ensemble",
)
def main(cfg: DictConfig) -> None:
    rank, world_size, device = setup_distributed(cfg)

    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if rank == 0:
        logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

    _validate_cfg(cfg)

    inference_mode = str(cfg.advantage.model.get("inference_mode", "wco"))
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
        inference_mode=inference_mode,
        precision=precision,
    )
    model = _coerce_inference_model(raw_model)

    if rank == 0:
        logger.info(
            "Ensemble loaded: ensemble_size=%d, inference_mode=%s, "
            "max_token_len=%d",
            int(model.config.ensemble_size),
            str(model.config.inference_mode),
            int(getattr(model.config, "max_token_len", 200)),
        )

    try:
        # ---- Phase 1: GPU inference + advantage_continuous ----
        # All ranks participate in inference; rank 0 accumulates the per-frame
        # continuous scores so Phase 2 can pick a unified threshold before
        # committing bool labels. Other ranks park at the per-dataset barrier
        # so the GPU memory / NCCL state stays in lockstep with rank 0.
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
            df = _run_inference_for_dataset(
                model=model,
                dataset_entry=entry,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                device=device,
            )

            if rank == 0:
                df_cont = _compute_advantage_continuous(df)
                collected.append((entry, df_cont))

            if world_size > 1:
                dist.barrier()

        # ---- Phase 2: pick thresholds + write parquet + update meta ----
        # Runs only on rank 0. ``rollout_threshold`` is applied to rollout
        # datasets; ``expert_threshold`` (quantile mode + advantage.expert_quantile)
        # is applied to sft datasets. In quantile mode each threshold is the
        # ``(1 - quantile)``-th percentile of advantage_continuous over its OWN
        # pool (rollout frames vs sft frames). When expert_quantile is unset,
        # sft frames stay all-True (the historical convention) but still record
        # the rollout threshold in their tag metadata for provenance.
        if rank == 0:
            expert_threshold: Optional[float] = None
            if label_mode == "quantile":
                rollout_scores: list[np.ndarray] = [
                    d["advantage_continuous"].values
                    for e, d in collected
                    if str(e.type).lower() == "rollout"
                ]
                if not rollout_scores:
                    raise ValueError(
                        "advantage.label_mode='quantile' requires at least one "
                        "rollout dataset to derive the rollout threshold; none "
                        "of the configured train_data_paths has type='rollout'."
                    )
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
                        expert_threshold = float(
                            np.percentile(
                                combined_sft,
                                (1.0 - float(expert_quantile_cfg)) * 100.0,
                            )
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
                # Threshold actually applied to the bool label. ``None`` ⇒
                # force every frame True (sft with no expert quantile).
                label_threshold = expert_threshold if is_sft else rollout_threshold
                final_df = _apply_boolean_label(
                    df_cont,
                    positive_threshold=label_threshold,
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
                mix_path = _update_mixture_config(
                    dataset_path=entry.dataset_path,
                    tag=tag,
                    positive_threshold=recorded_threshold,
                    inference_mode=str(model.config.inference_mode),
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
                        if (
                            label_mode == "quantile"
                            and expert_quantile_cfg is not None
                        )
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


if __name__ == "__main__":
    main()
