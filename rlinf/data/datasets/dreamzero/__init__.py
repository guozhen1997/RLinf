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

"""DreamZero LeRobot SFT dataset package."""

from rlinf.data.datasets.dreamzero.dreamzero import (
    DEFAULT_MAX_TEMPORAL_BLOCKS,
    DreamZeroCollator,
    DreamZeroLeRobotDataset,
    build_dreamzero_sft_dataloader,
)
from rlinf.data.datasets.dreamzero.sampling_strategy import (
    EmptyShardedSampleError,
    SamplingMode,
    ShardedTemporalConfig,
    TemporalIndices,
    build_dense_offsets,
    require_sharded_temporal_indices,
    sample_action_indices,
    sample_state_indices,
    sample_temporal_indices,
    sample_video_indices,
)

__all__ = [
    "DEFAULT_MAX_TEMPORAL_BLOCKS",
    "DreamZeroCollator",
    "DreamZeroLeRobotDataset",
    "EmptyShardedSampleError",
    "SamplingMode",
    "ShardedTemporalConfig",
    "TemporalIndices",
    "build_dense_offsets",
    "build_dreamzero_sft_dataloader",
    "require_sharded_temporal_indices",
    "sample_action_indices",
    "sample_state_indices",
    "sample_temporal_indices",
    "sample_video_indices",
]
