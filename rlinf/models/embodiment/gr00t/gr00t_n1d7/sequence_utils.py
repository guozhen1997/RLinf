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

"""Helpers for batch-major ``[B * T, ...]`` GR00T sequence tensors."""

from __future__ import annotations

from typing import Any, Optional

import torch

# Fields that are flattened across images rather than the env batch, so they
# need the same batch-major slicing as the rest of the sequence once the
# collator has restored a leading ``[B * T]`` axis.
_SEQUENCE_BATCH_KEYS = {
    "state",
    "state_mask",
    "action",
    "action_mask",
    "embodiment_id",
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "image_sizes",
    "loss_mask",
}


def slice_batch_major(
    batch: dict[str, Any],
    t_start: int,
    t_end: int,
    num_timesteps: int,
) -> dict[str, Any]:
    """Take time range ``[t_start, t_end)`` from a batch-major sequence.

    Args:
        batch: tensors laid out as ``[B * T, ...]``, plus optional ``[B, T]``
            ``loss_mask``.
        t_start, t_end: half-open timestep window.
        num_timesteps: ``T`` of the incoming batch.

    Returns:
        A new dict whose tensors are ``[B * (t_end - t_start), ...]`` and whose
        ``num_timesteps`` is the window length.
    """
    if t_end <= t_start:
        raise ValueError(f"empty slice [{t_start}, {t_end})")
    if t_start < 0 or t_end > num_timesteps:
        raise ValueError(
            f"slice [{t_start}, {t_end}) is outside [0, {num_timesteps})"
        )
    window = t_end - t_start
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "num_timesteps":
            continue
        if not torch.is_tensor(value):
            sliced[key] = value
            continue
        if key == "loss_mask" and value.ndim >= 2 and value.shape[-1] == num_timesteps:
            sliced[key] = value[..., t_start:t_end]
            continue
        if key not in _SEQUENCE_BATCH_KEYS and value.shape[0] % num_timesteps != 0:
            sliced[key] = value
            continue
        if value.shape[0] % num_timesteps != 0:
            raise ValueError(
                f"{key} leading dim {value.shape[0]} is not divisible by "
                f"num_timesteps={num_timesteps}"
            )
        batch_size = value.shape[0] // num_timesteps
        reshaped = value.reshape(batch_size, num_timesteps, *value.shape[1:])
        sliced[key] = reshaped[:, t_start:t_end].reshape(
            batch_size * window, *value.shape[1:]
        )
    sliced["num_timesteps"] = window
    return sliced


def segment_bounds(num_timesteps: int, segment_length: int) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` windows covering ``num_timesteps``."""
    if segment_length <= 0:
        raise ValueError(f"segment_length must be positive, got {segment_length}")
    return [
        (start, min(start + segment_length, num_timesteps))
        for start in range(0, num_timesteps, segment_length)
    ]


def flatten_leading_bt(batch: dict[str, Any], batch_size: int, num_timesteps: int) -> dict[str, Any]:
    """Reshape ``[B, T, ...]`` fields to batch-major ``[B * T, ...]``.

    Used when a collator emits a time axis (online LeRobot chunk windows).
    """
    if num_timesteps <= 1:
        return batch
    flattened: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "num_timesteps":
            continue
        if key in {"task", "task_descriptions"} and isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                flattened[key] = [item for item in value for _ in range(num_timesteps)]
            else:
                flattened[key] = value
            continue
        if torch.is_tensor(value):
            if (
                value.ndim >= 2
                and value.shape[0] == batch_size
                and value.shape[1] == num_timesteps
            ):
                flattened[key] = value.reshape(
                    batch_size * num_timesteps, *value.shape[2:]
                )
            else:
                flattened[key] = value
            continue
        if isinstance(value, dict):
            flattened[key] = flatten_leading_bt(value, batch_size, num_timesteps)
            continue
        flattened[key] = value
    flattened["num_timesteps"] = num_timesteps
    return flattened


def dones_to_reset_mask(dones: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Flatten env-done flags to a ``[B]`` boolean reset mask."""
    if dones is None:
        return None
    return dones.reshape(-1).to(dtype=torch.bool)
