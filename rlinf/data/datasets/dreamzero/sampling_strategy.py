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

"""Temporal index sampling for DreamZero LeRobot SFT datasets.

Two modes:

- **dense**: fixed contiguous offsets from ``num_video_frames`` / ``num_chunks`` /
  ``action_horizon`` (legacy RLinf window).
- **sharded**: Groot ``lerobot_sharded`` language-aware multi-anchor expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SamplingMode = Literal["sharded", "dense"]


class EmptyShardedSampleError(ValueError):
    """Raised when sharded sampling yields no valid video/action/state indices."""


@dataclass(frozen=True)
class ShardedTemporalConfig:
    """Hyper-parameters for Groot-compatible sharded temporal sampling."""

    max_temporal_blocks: int
    macro_stride: int = 24
    action_subchunk_size: int = 24
    video_in_chunk_offsets: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21)


@dataclass(frozen=True)
class TemporalIndices:
    """Relative frame indices within an episode (before adding ``frame_in_ep``)."""

    video: np.ndarray
    state: np.ndarray
    action: np.ndarray
    num_video_chunks: int

    @property
    def is_empty(self) -> bool:
        return (
            self.video.size == 0
            or self.action.size == 0
            or self.state.size == 0
        )

def sample_video_indices(
    first_idx: int,
    language_annotations: np.ndarray,
    trajectory_length: int,
    cfg: ShardedTemporalConfig,
) -> tuple[np.ndarray, int]:
    """Match ``_uniform_sample_from_language_ranges`` in ``lerobot_sharded.py``."""
    if trajectory_length <= 0:
        return np.array([], dtype=np.int64), 0

    target_language = language_annotations[first_idx]
    max_frames = 8 * cfg.max_temporal_blocks + 1
    per_step_offsets = list(cfg.video_in_chunk_offsets)
    sampled_list: list[int] = []

    def add_step_set(anchor_index: int) -> None:
        if anchor_index < 0 or anchor_index + 23 >= trajectory_length:
            return
        if len(sampled_list) + len(per_step_offsets) > max_frames:
            return
        for offset in per_step_offsets:
            sampled_list.append(int(anchor_index + offset))

    add_step_set(first_idx)

    step = 1
    back_done = False
    fwd_done = False
    while len(sampled_list) < max_frames and (not back_done or not fwd_done):
        if not back_done:
            back_anchor = first_idx - cfg.macro_stride * step
            if back_anchor < 0:
                back_done = True
            elif language_annotations[back_anchor] != target_language:
                back_done = True
            else:
                add_step_set(back_anchor)
        if len(sampled_list) >= max_frames:
            break
        if not fwd_done:
            fwd_anchor = first_idx + cfg.macro_stride * step
            if fwd_anchor >= trajectory_length:
                fwd_done = True
            elif language_annotations[fwd_anchor] != target_language:
                fwd_done = True
            else:
                add_step_set(fwd_anchor)
        step += 1

    if not sampled_list:
        return np.array([], dtype=np.int64), 0

    unique_sorted = np.array(sorted(set(sampled_list)), dtype=np.int64)
    if unique_sorted.size > max_frames:
        unique_sorted = unique_sorted[:max_frames]

    if unique_sorted.size > 0:
        last_idx = int(unique_sorted[-1])
        additional_idx = last_idx + 3
        if additional_idx < trajectory_length and unique_sorted.size < max_frames:
            unique_sorted = np.append(unique_sorted, additional_idx)
        else:
            if unique_sorted.size <= 8:
                return np.array([], dtype=np.int64), 0
            unique_sorted = unique_sorted[:-7]

    if unique_sorted.size == 0 or unique_sorted.size % 8 != 1:
        return np.array([], dtype=np.int64), 0

    num_video_chunks = (unique_sorted.size - 1) // 8
    return unique_sorted.astype(np.int64, copy=False), int(num_video_chunks)


def sample_action_indices(
    first_idx: int,
    language_annotations: np.ndarray,
    trajectory_length: int,
    cfg: ShardedTemporalConfig,
    target_num_chunks: int | None,
) -> np.ndarray:
    """Match ``get_action`` language-aware branch in ``lerobot_sharded.py``."""
    if trajectory_length <= 0:
        return np.array([], dtype=np.int64)

    target_language = language_annotations[first_idx]
    max_frames = cfg.action_subchunk_size * cfg.max_temporal_blocks
    per_step_offsets = list(range(cfg.action_subchunk_size))
    sampled_list: list[int] = []

    def add_step_set(anchor_index: int) -> None:
        if anchor_index < 0 or anchor_index + cfg.action_subchunk_size >= trajectory_length:
            return
        if len(sampled_list) + cfg.action_subchunk_size > max_frames:
            return
        if (
            target_num_chunks is not None
            and len(sampled_list) // cfg.action_subchunk_size >= target_num_chunks
        ):
            return
        for offset in per_step_offsets:
            sampled_list.append(int(anchor_index + offset))

    add_step_set(first_idx)

    step = 1
    back_done = False
    fwd_done = False
    while len(sampled_list) < max_frames and (not back_done or not fwd_done):
        if target_num_chunks is not None and len(sampled_list) // cfg.action_subchunk_size >= target_num_chunks:
            break
        if not back_done:
            back_anchor = first_idx - cfg.macro_stride * step
            if back_anchor < 0:
                back_done = True
            elif language_annotations[back_anchor] != target_language:
                back_done = True
            else:
                add_step_set(back_anchor)
        if len(sampled_list) >= max_frames:
            break
        if not fwd_done:
            fwd_anchor = first_idx + cfg.macro_stride * step
            if fwd_anchor >= trajectory_length:
                fwd_done = True
            elif language_annotations[fwd_anchor] != target_language:
                fwd_done = True
            else:
                add_step_set(fwd_anchor)
        step += 1

    if not sampled_list:
        return np.array([], dtype=np.int64)

    unique_sorted = np.array(sorted(set(sampled_list)), dtype=np.int64)
    capped_size = min(unique_sorted.size, max_frames)
    divisible_size = (capped_size // cfg.action_subchunk_size) * cfg.action_subchunk_size
    return unique_sorted[:divisible_size]


def sample_state_indices(
    first_idx: int,
    language_annotations: np.ndarray,
    trajectory_length: int,
    cfg: ShardedTemporalConfig,
    target_num_chunks: int | None,
) -> np.ndarray:
    """Match ``get_state`` language-aware branch in ``lerobot_sharded.py``."""
    if trajectory_length <= 0:
        return np.array([], dtype=np.int64)

    target_language = language_annotations[first_idx]
    max_frames = cfg.max_temporal_blocks
    sampled_list: list[int] = []

    def add_anchor(anchor_index: int) -> None:
        if len(sampled_list) >= max_frames:
            return
        if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
            return
        if 0 <= anchor_index and anchor_index + cfg.macro_stride < trajectory_length:
            sampled_list.append(int(anchor_index))

    add_anchor(first_idx)

    step = 1
    back_done = False
    fwd_done = False
    while len(sampled_list) < max_frames and (not back_done or not fwd_done):
        if target_num_chunks is not None and len(sampled_list) >= target_num_chunks:
            break
        if not back_done:
            back_anchor = first_idx - cfg.macro_stride * step
            if back_anchor < 0:
                back_done = True
            elif language_annotations[back_anchor] != target_language:
                back_done = True
            else:
                add_anchor(back_anchor)
        if len(sampled_list) >= max_frames:
            break
        if not fwd_done:
            fwd_anchor = first_idx + cfg.macro_stride * step
            if fwd_anchor >= trajectory_length:
                fwd_done = True
            elif language_annotations[fwd_anchor] != target_language:
                fwd_done = True
            else:
                add_anchor(fwd_anchor)
        step += 1

    if not sampled_list:
        return np.array([], dtype=np.int64)
    return np.array(sorted(set(sampled_list)), dtype=np.int64)


def sample_temporal_indices(
    first_idx: int,
    language_annotations: np.ndarray,
    trajectory_length: int,
    cfg: ShardedTemporalConfig,
) -> TemporalIndices:
    """Sample video then action/state (video chunk count drives action/state)."""
    video, num_chunks = sample_video_indices(
        first_idx, language_annotations, trajectory_length, cfg
    )
    action = sample_action_indices(
        first_idx,
        language_annotations,
        trajectory_length,
        cfg,
        target_num_chunks=num_chunks if num_chunks > 0 else None,
    )
    state = sample_state_indices(
        first_idx,
        language_annotations,
        trajectory_length,
        cfg,
        target_num_chunks=num_chunks if num_chunks > 0 else None,
    )
    return TemporalIndices(
        video=video,
        state=state,
        action=action,
        num_video_chunks=num_chunks,
    )


def require_sharded_temporal_indices(
    frame_in_ep: int,
    language_annotations: np.ndarray,
    ep_len: int,
    cfg: ShardedTemporalConfig,
    *,
    episode_index: int | None = None,
) -> TemporalIndices:
    """Sharded sampling; raises :class:`EmptyShardedSampleError` when indices are empty."""
    temporal = sample_temporal_indices(
        frame_in_ep, language_annotations, ep_len, cfg
    )
    if temporal.is_empty:
        ep = episode_index if episode_index is not None else "?"
        raise EmptyShardedSampleError(
            f"Empty sharded temporal indices at frame {frame_in_ep} "
            f"episode {ep} len {ep_len}"
        )
    return temporal


def build_dense_offsets(
    num_video_frames: int,
    state_horizon: int,
    action_horizon: int,
    num_chunks: int,
) -> TemporalIndices:
    """Legacy contiguous offsets (``action_horizon * num_chunks`` window)."""
    return TemporalIndices(
        video=np.arange(num_video_frames, dtype=np.int64),
        state=np.asarray(
            [
                chunk_idx * action_horizon + state_idx
                for chunk_idx in range(num_chunks)
                for state_idx in range(state_horizon)
            ],
            dtype=np.int64,
        ),
        action=np.asarray(
            [
                chunk_idx * action_horizon + action_idx
                for chunk_idx in range(num_chunks)
                for action_idx in range(action_horizon)
            ],
            dtype=np.int64,
        ),
        num_video_chunks=num_chunks,
    )
