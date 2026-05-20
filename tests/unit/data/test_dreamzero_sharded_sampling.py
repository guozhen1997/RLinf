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

import numpy as np
import pytest

from rlinf.data.datasets.dreamzero.sampling_strategy import (
    ShardedTemporalConfig,
    sample_action_indices,
    sample_state_indices,
    sample_temporal_indices,
    sample_video_indices,
)


def test_video_indices_constant_language_produces_33_frames():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.zeros(120, dtype=np.int64)
    video, num_chunks = sample_video_indices(50, language, 120, cfg)
    assert video.size == 33
    assert video.size % 8 == 1
    assert num_chunks == 4


def test_action_indices_aligns_with_video_chunk_count():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.zeros(200, dtype=np.int64)
    video, num_chunks = sample_video_indices(80, language, 200, cfg)
    action = sample_action_indices(80, language, 200, cfg, target_num_chunks=num_chunks)
    assert action.size == 96
    assert action.size % 24 == 0


def test_state_indices_respect_target_num_chunks():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.zeros(200, dtype=np.int64)
    _, num_chunks = sample_video_indices(80, language, 200, cfg)
    state = sample_state_indices(80, language, 200, cfg, target_num_chunks=num_chunks)
    assert state.size == num_chunks


def test_language_change_stops_backward_expansion():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.array([0] * 50 + [1] * 50, dtype=np.int64)
    action = sample_action_indices(50, language, 100, cfg, target_num_chunks=None)
    assert action.min() >= 50
    assert action.size % 24 == 0


def test_temporal_indices_empty_near_trajectory_end():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.zeros(30, dtype=np.int64)
    temporal = sample_temporal_indices(28, language, 30, cfg)
    assert temporal.is_empty


def test_sample_temporal_indices_full_window():
    cfg = ShardedTemporalConfig(max_temporal_blocks=4)
    language = np.zeros(300, dtype=np.int64)
    temporal = sample_temporal_indices(100, language, 300, cfg)
    assert not temporal.is_empty
    assert temporal.video.size == 33
    assert temporal.action.size == 96
    assert temporal.state.size == 4
    assert temporal.num_video_chunks == 4
