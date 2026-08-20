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

import pytest
import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    flatten_leading_bt,
    segment_bounds,
    slice_batch_major,
)


def test_segment_bounds_cover_the_sequence():
    assert segment_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert segment_bounds(3, 8) == [(0, 3)]


def test_slice_batch_major_keeps_env_identity():
    num_timesteps = 4
    batch_size = 2
    state = torch.arange(batch_size * num_timesteps).reshape(
        batch_size, num_timesteps, 1
    )
    # Flatten batch-major: env0's 4 steps, then env1's.
    flat = state.reshape(batch_size * num_timesteps, 1)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    sliced = slice_batch_major(
        {"state": flat, "loss_mask": loss_mask, "num_timesteps": num_timesteps},
        t_start=1,
        t_end=3,
        num_timesteps=num_timesteps,
    )
    assert sliced["num_timesteps"] == 2
    # env0 steps 1,2 then env1 steps 1,2
    torch.testing.assert_close(
        sliced["state"].squeeze(-1), torch.tensor([1, 2, 5, 6])
    )
    torch.testing.assert_close(
        sliced["loss_mask"], torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    )


def test_slice_batch_major_rejects_bad_windows():
    batch = {"state": torch.zeros(4, 1)}
    with pytest.raises(ValueError):
        slice_batch_major(batch, 2, 2, 4)
    with pytest.raises(ValueError):
        slice_batch_major(batch, 0, 5, 4)


def test_flatten_leading_bt_is_batch_major():
    batch = {
        "state": torch.arange(12).reshape(2, 3, 2),
        "task": ["a", "b"],
        "nested": {"image": torch.arange(24).reshape(2, 3, 2, 2, 1)},
    }
    flat = flatten_leading_bt(batch, batch_size=2, num_timesteps=3)
    assert flat["num_timesteps"] == 3
    assert flat["task"] == ["a", "a", "a", "b", "b", "b"]
    # env0 steps then env1 steps
    torch.testing.assert_close(
        flat["state"],
        torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]),
    )
    assert tuple(flat["nested"]["image"].shape) == (6, 2, 2, 1)


def test_tbptt_segment_weights_sum_to_one():
    """Length-weighted segments reproduce (1/T) sum_t ell_t, including a short tail."""
    num_timesteps = 10
    weights = [
        (end - start) / num_timesteps
        for start, end in segment_bounds(num_timesteps, 8)
    ]
    assert weights == [0.8, 0.2]
    assert abs(sum(weights) - 1.0) < 1e-12
