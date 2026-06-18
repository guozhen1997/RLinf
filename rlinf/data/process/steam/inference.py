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

"""Ensemble inference loop for the STEAM advantage pipeline.

Shards a STEAM pair-inference dataset across ``torchrun`` ranks, runs the
ensemble critic over every anchor frame, and gathers per-frame records (signed
score, member stats, entropy, expected stride) back to rank 0 as a DataFrame.
Terminal frames carry no pair and are backfilled with neutral defaults.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from rlinf.data.datasets.steam.binning import entropy_nats, expected_signed_stride
from rlinf.data.datasets.steam.pair_dataset import (
    BinaryPairDataCollator,
    BinaryPairInferenceDataset,
)
from rlinf.data.process.distributed import (
    gather_dataframes_to_rank0,
    get_shard_indices,
)
from rlinf.models.embodiment.steam.ensemble_modeling_critic import (
    EnsembleSteamCriticModel,
)

logger = logging.getLogger(__name__)


def move_to_device(obj: Any, device: str):
    """Recursive ``.to(device)`` for tensors nested in dicts / lists."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return type(obj)(moved)
    return obj


def build_inference_dataloader(
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
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "shuffle": False,
        "collate_fn": collate_fn,
    }
    # prefetch_factor / persistent_workers are only valid with worker
    # processes; passing them when num_workers == 0 errors on older PyTorch.
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(shard, **loader_kwargs)
    return loader, len(shard_indices)


def records_from_predict(
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
    # ([B, num_bins]); ``out.member_probs`` is [E, B, num_bins]. For ensembles
    # of size 1 (single-model wrapper) E==1 and member-variance columns fall
    # to 0.
    agg_probs_np = out.probs.detach().to("cpu", dtype=torch.float32).numpy()
    member_probs = out.member_probs
    if member_probs is None:
        raise RuntimeError(
            "records_from_predict expects EnsembleCriticOutput.member_probs to be "
            "populated; got None. Ensure the checkpoint routes through the "
            "ensemble wrapper, including the single-member ensemble_size=1 case."
        )
    member_probs_np = member_probs.detach().to("cpu", dtype=torch.float32).numpy()

    es = expected_signed_stride(agg_probs_np, stride_k, num_bins) / float(stride_k)
    entropy_agg = entropy_nats(agg_probs_np)  # [B]
    entropy_members = entropy_nats(member_probs_np)  # [E, B]
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
                # probability). For num_bins == 2 it degenerates to
                # ``2 · P(progress) - 1``.
                "ensemble_signed_score": float(aggregated[i].item()),
                "p_progress_mean": float(mean[i].item()),
                "p_progress_min": float(minv[i].item()),
                "p_progress_variance": float(var[i].item()),
                "member_values": [float(x) for x in members[:, i].tolist()],
                # Multi-bin additive columns. Binary (num_bins=2) still gets
                # these: expected_stride_normalized degenerates to a monotone
                # function of ensemble_signed_score, entropy is Bernoulli.
                "expected_stride_normalized": float(es[i]),
                "entropy_aggregated": float(entropy_agg[i]),
                "entropy_member_mean": float(entropy_member_mean[i]),
                "entropy_member_variance": float(entropy_member_variance[i]),
            }
        )
    return rows


def build_terminal_frame_rows(
    *,
    episode_lengths: list[int],
    member_count: int,
) -> pd.DataFrame:
    """Build default-neutral rows for each episode's terminal frame."""
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


def append_missing_terminal_rows(
    df: pd.DataFrame,
    *,
    episode_lengths: list[int],
    member_count: int,
) -> tuple[pd.DataFrame, int]:
    """Append any missing terminal frames with zero progress defaults."""
    terminal_rows = build_terminal_frame_rows(
        episode_lengths=episode_lengths,
        member_count=member_count,
    )
    if terminal_rows.empty:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        return df, 0

    if len(df) == 0:
        combined = terminal_rows.sort_values(
            ["episode_index", "frame_index"]
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


def run_inference_for_dataset(
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

    loader, shard_size = build_inference_dataloader(
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
        observation = move_to_device(batch["observation"], device)
        with torch.inference_mode():
            out = model.predict(observation)
        local_rows.extend(
            records_from_predict(out, batch, num_bins=num_bins, stride_k=stride_k)
        )

    local_df = pd.DataFrame(local_rows)
    if world_size > 1:
        dist.barrier()
        df = gather_dataframes_to_rank0(local_df, rank, world_size)
    else:
        df = local_df

    if rank == 0:
        if len(df) > 0:
            df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        episode_lengths = [
            dataset._source.episode_length(ep)
            for ep in range(dataset._source.num_episodes())
        ]
        df, num_appended = append_missing_terminal_rows(
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


__all__ = [
    "move_to_device",
    "build_inference_dataloader",
    "records_from_predict",
    "build_terminal_frame_rows",
    "append_missing_terminal_rows",
    "run_inference_for_dataset",
]
