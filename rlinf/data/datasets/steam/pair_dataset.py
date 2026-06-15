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

"""Pair dataset + collator for ARM + ReWiND binary value learning.

Dataset contract (``__getitem__``):

    {
        "image_t":  {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t
        "image_tk": {cam_name: np.ndarray[H, W, 3] uint8, ...},  # frame at t+k
        "image_mask_t":  {cam_name: bool, ...},
        "image_mask_tk": {cam_name: bool, ...},
        "prompt": str,
        "state":    Optional[np.ndarray],  # proprio at t (for state-in-prompt)
        "state_tk": Optional[np.ndarray],  # proprio at t+k (reserved)
        "label": int,                      # long bin index in [0, num_bins); binary: 0 = regress, 1 = progress
        "episode": int,
        "frame_idx_t": int,
        "frame_idx_tk": int,
    }

Each ``cam_name`` is a camera **view** (e.g. ``"base_0_rgb"``,
``"left_wrist_0_rgb"``). The time axis — frame_t vs frame_{t+k} — is a
separate structural axis: the collator runs the
``SteamProcessor`` once for frame_t and once for frame_{t+k}, then
stacks the two per-camera image tensors along a new ``num_frames`` dim so
the backbone receives a ``[B, num_cameras, num_frames, 3, H, W]`` tensor
per camera key.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _isolate_hf_datasets_cache_for_process() -> None:
    """Optionally give each actor its own HuggingFace datasets cache directory."""
    if os.environ.get("RLINF_ISOLATE_HF_DATASETS_CACHE", "0").lower() in (
        "0",
        "false",
        "no",
    ):
        return

    if os.environ.get("RLINF_HF_DATASETS_CACHE_ISOLATED"):
        return

    base_cache = os.environ.get("HF_DATASETS_CACHE")
    if not base_cache:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        base_cache = str(Path(hf_home) / "datasets")

    rank = os.environ.get("RANK", "norank")
    cache_dir = Path(base_cache) / f"rlinf_rank_{rank}_pid_{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    os.environ["RLINF_HF_DATASETS_CACHE_ISOLATED"] = "1"

    try:
        import datasets

        datasets.config.HF_DATASETS_CACHE = str(cache_dir)
    except ImportError:
        pass


# Camera-view aliases tried against raw LeRobot sample dicts. Callers pass
# a plain camera key (e.g. ``image``) and the dataset probes the standard
# LeRobot path templates. Unlike the camera axis, state only has a single
# canonical field, so its alias list is shorter.
_IMAGE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
    "observation.images.{key}",
    "observation/images/{key}",
)
_STATE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
    "observation.state",
    "observation/state",
)


def _resolve_alias(sample: dict, key: str, aliases: Sequence[str]) -> Any:
    """Return ``sample[alias]`` for the first alias template that matches."""
    for template in aliases:
        resolved = template.format(key=key)
        if resolved in sample:
            return sample[resolved]
    raise KeyError(
        f"Could not resolve key={key!r} in sample. Tried: "
        f"{[t.format(key=key) for t in aliases]}. "
        f"Available: {sorted(sample.keys())}"
    )


def _scalar_item(value: Any) -> Any:
    """Return a Python scalar from tensor/array/list scalar-like values."""
    if isinstance(value, torch.Tensor):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected scalar-like list/tuple, got {value!r}")
        return _scalar_item(value[0])
    return value


def _to_uint8_hwc(frame: Any) -> np.ndarray:
    """Normalise a frame to ``(H, W, 3)`` uint8 numpy."""
    if hasattr(frame, "convert"):  # PIL.Image duck-typed
        frame = np.asarray(frame)
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)

    if arr.ndim != 3:
        raise ValueError(f"expected a rank-3 frame, got shape={arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if float(arr.max()) <= 1.5:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


def _to_float32_1d(state: Any, *, max_dim: Optional[int] = None) -> np.ndarray:
    """Normalise state to a rank-1 float32 array, optionally truncated/padded."""
    arr = _to_float32_array(state).reshape(-1)
    if max_dim is None:
        return arr
    if arr.shape[0] > max_dim:
        return arr[:max_dim]
    if arr.shape[0] < max_dim:
        padded = np.zeros((max_dim,), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr


def _to_float32_array(value: Any) -> np.ndarray:
    """Convert tensor/list/scalar-like state payloads to float32 numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _load_openpi_norm_stats(
    checkpoint_dir: Path, asset_id: str
) -> dict[str, dict[str, np.ndarray | None]]:
    """Load OpenPI norm stats as plain numpy arrays."""
    possible_paths = [
        checkpoint_dir / "norm_stats" / asset_id / "norm_stats.json",
        checkpoint_dir / "stats" / asset_id / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
    ]

    norm_stats_dir = checkpoint_dir / "norm_stats"
    if norm_stats_dir.exists():
        for subdir in norm_stats_dir.iterdir():
            if subdir.is_dir():
                candidate = subdir / "norm_stats.json"
                if candidate.exists() and candidate not in possible_paths:
                    possible_paths.append(candidate)

    for path in possible_paths:
        if not path.exists():
            continue
        logger.info("Loading norm stats from %s", path)
        with open(path) as f:
            data = json.load(f)
        if "norm_stats" in data:
            data = data["norm_stats"]

        stats: dict[str, dict[str, np.ndarray | None]] = {}
        for key, value in data.items():
            stats[key] = {
                "mean": np.asarray(value["mean"], dtype=np.float32),
                "std": np.asarray(value["std"], dtype=np.float32),
                "q01": (
                    np.asarray(value["q01"], dtype=np.float32)
                    if value.get("q01") is not None
                    else None
                ),
                "q99": (
                    np.asarray(value["q99"], dtype=np.float32)
                    if value.get("q99") is not None
                    else None
                ),
            }
        return stats

    raise FileNotFoundError(f"Could not find norm_stats.json in {checkpoint_dir}")


@dataclass(frozen=True)
class _OpenPIStateOnlyTransform:
    """State-only subset of the OpenPI input transform stack."""

    action_dim: int
    pre_pad_state: bool
    norm_stats: dict[str, dict[str, np.ndarray | None]] | None

    def __call__(self, state: Any) -> np.ndarray:
        state_arr = _to_float32_array(state)
        if self.pre_pad_state:
            state_arr = _pad_to_dim(state_arr, self.action_dim, axis=-1)
        if self.norm_stats is not None and "state" in self.norm_stats:
            stats = self.norm_stats["state"]
            q01 = stats.get("q01")
            q99 = stats.get("q99")
            if q01 is None or q99 is None:
                mean = stats["mean"]
                std = stats["std"]
                state_arr = (state_arr - mean[..., : state_arr.shape[-1]]) / (
                    std[..., : state_arr.shape[-1]] + 1e-6
                )
            else:
                state_arr = (state_arr - q01[..., : state_arr.shape[-1]]) / (
                    q99[..., : state_arr.shape[-1]]
                    - q01[..., : state_arr.shape[-1]]
                    + 1e-6
                ) * 2.0 - 1.0
        return _pad_to_dim(state_arr, self.action_dim, axis=-1).astype(
            np.float32,
            copy=False,
        )


def _pad_to_dim(
    x: np.ndarray,
    target_dim: int,
    *,
    axis: int = -1,
    value: float = 0.0,
) -> np.ndarray:
    """Pad an array to target_dim along axis using OpenPI-compatible semantics."""
    current_dim = x.shape[axis]
    if current_dim >= target_dim:
        return x
    pad_width = [(0, 0)] * len(x.shape)
    pad_width[axis] = (0, target_dim - current_dim)
    return np.pad(x, pad_width, constant_values=value)


# X2Robot camera-view repack mapping shared by every ``*_sm2sm``-style
# robot name (the views are fixed by the X2Robot platform, not the task).
_X2ROBOT_REPACK_KEYS = {
    "images": {
        "left_wrist_view": "left_wrist_view",
        "face_view": "face_view",
        "right_wrist_view": "right_wrist_view",
    },
    "state": "state",
    "actions": "actions",
    "prompt": "task",
}

_X2ROBOT_MODES = ("s2s", "s2m", "sm2m", "sm2sm")


def _get_x2robot_mode(robot_type: str) -> Optional[str]:
    """Return the X2Robot mode encoded in a robot/config name, if present."""
    robot = robot_type.lower()
    if robot in ("x2robot", "arx"):
        return "sm2sm"
    for mode in _X2ROBOT_MODES:
        if robot == mode or robot.endswith(f"_{mode}"):
            return mode
    return None


def build_openpi_state_transform(
    *,
    robot_type: str,
    model_type: str,
    action_dim: int,
    default_prompt: Optional[str],
    norm_stats_dir: Optional[str],
    asset_id: Optional[str],
):
    """Build a state-only OpenPI transform for compatibility supervision."""
    from rlinf.data.datasets.recap.value_model import _REPACK_KEYS

    del default_prompt, model_type
    robot = robot_type.lower()
    x2robot_mode = _get_x2robot_mode(robot)

    repack_keys = _REPACK_KEYS.get(robot)
    if repack_keys is None and x2robot_mode is not None:
        repack_keys = _X2ROBOT_REPACK_KEYS
    if repack_keys is None:
        raise ValueError(
            f"Unknown robot type: {robot_type}. Available: {list(_REPACK_KEYS.keys())}"
        )

    norm_stats = None
    if norm_stats_dir is not None:
        resolved_asset_id = asset_id or robot
        try:
            norm_stats = _load_openpi_norm_stats(
                Path(norm_stats_dir), asset_id=resolved_asset_id
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "PairDataset was given "
                f"norm_stats_dir={norm_stats_dir!r} asset_id={resolved_asset_id!r} "
                "but no norm_stats.json was found. State compatibility expects "
                "normalized state; fix norm_stats_dir/asset_id or explicitly set "
                "allow_raw_state_for_compatibility=true."
            ) from exc
    return _OpenPIStateOnlyTransform(
        action_dim=int(action_dim),
        pre_pad_state=(
            robot in ("franka", "franka_co_train") or x2robot_mode is not None
        ),
        norm_stats=norm_stats,
    )


# ---------------------------------------------------------------------------
# Signed-stride → bin mapping (multi-bin mode)
# ---------------------------------------------------------------------------


def _signed_stride_to_bin(stride: int, K: int, num_bins: int) -> int:
    """Map a signed stride in ``{-K,...,-1,1,...,K}`` to a bin index.

    Layout:
        * ``pos = stride + K``         if stride < 0  (pos ∈ [0, K))
        * ``pos = stride + K - 1``     if stride > 0  (pos ∈ [K, 2K))
        * ``bin_idx = (pos * num_bins) // (2 * K)``   in [0, num_bins)

    With ``num_bins`` even and ``2K % num_bins == 0`` (enforced at
    :class:`PairDataset` construction), the sign split lands exactly at
    ``num_bins // 2``: bins ``[0, num_bins // 2)`` are regressive and
    bins ``[num_bins // 2, num_bins)`` are progressive.

    Raises:
        ValueError: if ``stride == 0`` (sampling excludes zero strides)
            or if ``abs(stride) > K``.
    """
    if stride == 0:
        raise ValueError(
            "_signed_stride_to_bin does not accept stride == 0; the multi-bin "
            "sampling path skips i == 0."
        )
    if abs(stride) > K:
        raise ValueError(
            f"_signed_stride_to_bin requires |stride| <= K, got stride={stride}, K={K}."
        )
    pos = stride + K if stride < 0 else stride + K - 1
    return int((pos * num_bins) // (2 * K))


def _positive_stride_to_bin(stride: int, K: int, num_bins: int) -> int:
    """Map a positive stride in ``{1,...,K}`` to a positive-only bin index."""
    if stride < 1 or stride > K:
        raise ValueError(
            f"_positive_stride_to_bin requires 1 <= stride <= K, "
            f"got stride={stride}, K={K}."
        )
    if num_bins < 1 or num_bins > K or K % num_bins != 0:
        raise ValueError(
            f"positive-only bins require 1 <= num_bins <= K and K % num_bins == 0; "
            f"got K={K}, num_bins={num_bins}."
        )
    return int(((stride - 1) * num_bins) // K)


def _scaled_signed_stride_to_bin(scaled_stride: float, K: int, num_bins: int) -> int:
    """Bin a length-scaled signed stride over ``[-K, K]`` (bin width ``2K/num_bins``).

    The scaled stride ``signed_stride * L_max / L_ep`` is rounded to the nearest
    integer-equivalent stride, clamped to ``[-K, K] \\ {0}`` with the sign
    preserved, then mapped through :func:`_signed_stride_to_bin` — the *same*
    layout (and resolution) as the unscaled path. Consequences:

    * An episode of length ``L_max`` (``scale == 1``) reproduces the unscaled
      bins exactly.
    * Shorter episodes (``scale > 1``) push a fixed frame stride into a higher
      progress bin and saturate at ``±K`` earlier — a stride covering fraction
      ``K / L_max`` of its episode already hits the extreme bin.
    * Longer episodes (``scale < 1``) under-reach: their max stride lands below
      the extreme bin.

    Only the position on the ``±K`` axis is length-dependent; the regress/
    progress split at ``half`` is preserved, so the model loss and the
    bin-index decoders (``_predicted_signed_value``) stay unchanged.
    """
    s = int(round(float(scaled_stride)))
    if s == 0:  # a non-zero stride must keep a direction, never "no progress"
        s = 1 if scaled_stride > 0 else -1
    s = max(-int(K), min(int(K), s))
    return _signed_stride_to_bin(s, int(K), num_bins)


def _scaled_positive_stride_to_bin(scaled_stride: float, K: int, num_bins: int) -> int:
    """Positive-only analogue of :func:`_scaled_signed_stride_to_bin`.

    Rounds ``|scaled_stride|`` to the nearest integer stride, clamps to
    ``[1, K]``, then maps via :func:`_positive_stride_to_bin`.
    """
    s = int(round(abs(float(scaled_stride))))
    s = max(1, min(int(K), s))
    return _positive_stride_to_bin(s, int(K), num_bins)


def bin_centers(K: int, num_bins: int) -> np.ndarray:
    """Return the ``[num_bins]`` signed-stride centers for the bin layout.

    Each bin owns a contiguous set of ``strides_per_bin = 2K / num_bins``
    signed strides; the center is their arithmetic mean. By construction
    :func:`_signed_stride_to_bin` maps every stride into the bin whose
    center is closest (for even ``strides_per_bin`` the boundary ties
    are absorbed by the half-integer offsets).

    Examples:
        * ``K=8, num_bins=8``  → ``[-7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5]``
        * ``K=4, num_bins=4``  → ``[-3.5, -1.5, 1.5, 3.5]``
        * ``K=K, num_bins=2``  → ``[-K/2, K/2]`` — binary degenerate:
          :math:`E[s] / K = 2 \\cdot p_\\text{progress} - 1`, matching
          the existing ``2·P − 1`` signed-confidence derivation.

    Raises:
        ValueError: ``num_bins`` not even or ``2K % num_bins != 0``.
    """
    if num_bins < 2 or num_bins % 2 != 0:
        raise ValueError(f"num_bins must be >= 2 and even, got {num_bins}")
    if (2 * K) % num_bins != 0:
        raise ValueError(
            f"bin_centers requires 2*K to be a multiple of num_bins; "
            f"got K={K}, num_bins={num_bins} (2K={2 * K})."
        )
    strides_per_bin = (2 * K) // num_bins
    half = num_bins // 2
    # Regressive bins: cover signed strides [-K, -1] in order.
    # Progressive bins: cover signed strides [1, K] in order.
    # Center of a regressive bin b ∈ [0, half): midpoint of its
    # strides_per_bin consecutive strides starting at -K + b * strides_per_bin.
    # Center of a progressive bin b ∈ [half, num_bins): midpoint starting
    # at 1 + (b - half) * strides_per_bin.
    centers = np.empty(num_bins, dtype=np.float32)
    for b in range(num_bins):
        if b < half:
            low = -K + b * strides_per_bin
        else:
            low = 1 + (b - half) * strides_per_bin
        high = low + strides_per_bin - 1
        centers[b] = (low + high) / 2.0
    return centers


def positive_bin_centers(K: int, num_bins: int) -> np.ndarray:
    """Return ``[num_bins]`` positive-stride centers for positive-only mode."""
    if num_bins < 1 or num_bins > K or K % num_bins != 0:
        raise ValueError(
            f"positive_bin_centers requires 1 <= num_bins <= K and "
            f"K % num_bins == 0; got K={K}, num_bins={num_bins}."
        )
    strides_per_bin = K // num_bins
    centers = np.empty(num_bins, dtype=np.float32)
    for b in range(num_bins):
        low = 1 + b * strides_per_bin
        high = low + strides_per_bin - 1
        centers[b] = (low + high) / 2.0
    return centers


def expected_signed_stride(probs, K: int, num_bins: int):
    """Return ``E[s] = Σ_b probs[..., b] * bin_centers[b]``.

    Backend-polymorphic: if ``probs`` is a :class:`torch.Tensor` the
    computation stays on the input's device / dtype; otherwise falls
    back to numpy. The last dim of ``probs`` must equal ``num_bins``.

    For the binary degenerate case ``num_bins == 2``, equals
    ``K * (probs[..., 1] - probs[..., 0]) = K * (2·p_progress - 1)``.
    Dividing by ``K`` gives a ``[-1, 1]``-range signed confidence score
    consistent with the cumulative-progress integrator used in the
    visualize script.
    """
    centers_np = bin_centers(K, num_bins)
    if isinstance(probs, torch.Tensor):
        if probs.shape[-1] != num_bins:
            raise ValueError(
                f"probs last dim must be num_bins={num_bins}, got {tuple(probs.shape)}"
            )
        centers_t = torch.as_tensor(centers_np, dtype=probs.dtype, device=probs.device)
        return (probs * centers_t).sum(dim=-1)
    probs_np = np.asarray(probs)
    if probs_np.shape[-1] != num_bins:
        raise ValueError(
            f"probs last dim must be num_bins={num_bins}, got {probs_np.shape}"
        )
    return (probs_np * centers_np).sum(axis=-1)


# ---------------------------------------------------------------------------
# Trajectory sources
# ---------------------------------------------------------------------------


class TrajectorySource:
    """Minimal interface that any concrete trajectory source must satisfy."""

    def num_episodes(self) -> int:
        raise NotImplementedError

    def episode_length(self, episode: int) -> int:
        raise NotImplementedError

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        """Return a ``(H, W, 3)`` uint8 frame, or ``None`` if the camera is absent."""
        raise NotImplementedError

    def get_view_from_sample(
        self,
        sample: dict,
        camera_key: str,
    ) -> Optional[np.ndarray]:
        """Return a view from an already-loaded raw sample."""
        del sample, camera_key
        raise NotImplementedError

    def get_state(self, episode: int, frame: int, state_key: str) -> np.ndarray:
        raise NotImplementedError

    def get_state_from_sample(self, sample: dict, state_key: str) -> np.ndarray:
        """Return state from an already-loaded raw sample."""
        del sample, state_key
        raise NotImplementedError

    def get_raw_sample(self, episode: int, frame: int) -> dict:
        """Return the raw source sample for transform pipelines."""
        raise NotImplementedError

    def get_raw_pair(
        self,
        episode: int,
        frame_t: int,
        frame_tk: int,
        *,
        camera_keys: Sequence[str],
        video_transform: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> tuple[dict, dict]:
        """Return two raw samples, optionally optimized as one pair read."""
        del camera_keys, video_transform
        return (
            self.get_raw_sample(episode, frame_t),
            self.get_raw_sample(episode, frame_tk),
        )

    def get_prompt(self, episode: int, frame: int) -> Optional[str]:
        """Return the task / language instruction for a given frame.

        Implementations return ``None`` if no per-sample instruction is
        available; the caller is expected to fall back to a default.
        """
        return None

    def get_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame: int,
    ) -> Optional[str]:
        """Return prompt from an already-loaded raw sample."""
        del sample
        return self.get_prompt(episode, frame)

    def episode_is_success(self, episode: int) -> bool:
        """Return whether the episode should be treated as successful."""
        raise NotImplementedError


class _LeRobotSource(TrajectorySource):
    """LeRobot-backed source with lazy per-frame access."""

    def __init__(
        self,
        dataset_path: str,
        *,
        only_success: bool = True,
        dataset_type: str,
    ) -> None:
        _isolate_hf_datasets_cache_for_process()
        try:
            from lerobot.common.datasets.lerobot_dataset import (  # noqa: E501
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
            from lerobot.common.datasets.utils import hf_transform_to_torch
        except ImportError:  # pragma: no cover — older lerobot layout
            from lerobot.common.datasets.lerobot_dataset import (  # noqa: E501
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
            from lerobot.common.datasets.utils import hf_transform_to_torch
        from PIL import Image as PILImage

        local_path = Path(dataset_path).absolute()
        self._dataset_label = str(local_path)
        self.meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        self.base = LeRobotDataset(
            local_path.name, root=local_path, download_videos=False
        )
        self._only_success = bool(only_success)
        self.dataset_type = dataset_type
        self._state_column_cache: dict[str, Any] = {}

        eps = self.base.episode_data_index
        self._ep_starts = [int(x) for x in eps["from"].tolist()]
        self._ep_ends = [int(x) for x in eps["to"].tolist()]
        if self.dataset_type == "sft":
            self._episode_success = None
        else:
            self._episode_success = (
                self._scan_episode_successes() if only_success else None
            )

        def _decoding_transform(batch: dict) -> dict:
            for key in list(batch.keys()):
                vals = batch[key]
                if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
                    batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
            return hf_transform_to_torch(batch)

        self.base.hf_dataset.set_transform(_decoding_transform)

        self._tasks: dict[int, str] = self._load_tasks(local_path)

    @staticmethod
    def _load_tasks(dataset_path: Path) -> dict[int, str]:
        """Load task_index → instruction mapping from LeRobot meta."""
        meta = dataset_path / "meta"
        jsonl = meta / "tasks.jsonl"
        if jsonl.exists():
            tasks: dict[int, str] = {}
            with open(jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    tasks[int(d.get("task_index", len(tasks)))] = str(d.get("task", ""))
            return tasks
        parquet = meta / "tasks.parquet"
        if parquet.exists():
            import pandas as pd

            df = pd.read_parquet(parquet)
            if "task_index" in df.columns and "task" in df.columns:
                return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
        return {}

    def num_episodes(self) -> int:
        return len(self._ep_starts)

    def episode_length(self, episode: int) -> int:
        return self._ep_ends[episode] - self._ep_starts[episode]

    def _sample(self, episode: int, frame: int) -> dict:
        global_idx = self._ep_starts[episode] + int(frame)
        return self.base[global_idx]

    def _metadata_sample(self, episode: int, frame: int) -> dict:
        """Read non-video per-frame metadata from the HF table."""
        global_idx = self._ep_starts[episode] + int(frame)
        raw_dataset = self.base.hf_dataset
        sample = {
            key: raw_dataset.data.column(key)[global_idx].as_py()
            for key in raw_dataset.column_names
        }
        task_idx = sample.get("task_index")
        if task_idx is not None:
            sample["task"] = self.meta.tasks[int(_scalar_item(task_idx))]
        return sample

    @staticmethod
    def _coerce_success_flag(raw: Any) -> bool:
        """Normalise a raw per-frame success flag to Python bool."""
        return bool(_scalar_item(raw))

    def _scan_episode_successes(self) -> list[bool]:
        """Read one representative frame row per episode from ``is_success``."""
        raw_dataset = self.base.hf_dataset
        if "is_success" not in raw_dataset.column_names:
            raise ValueError(
                "PairDataset(dataset_type='rollout', only_success=True) "
                "requires the LeRobot dataset to contain an 'is_success' "
                "column on representative frame rows."
            )

        # Read is_success directly from the Arrow table, bypassing
        # hf_dataset.set_transform. The transform loads every column and runs
        # hf_transform_to_torch, which crashes on non-image dict columns
        # (e.g. lerobot Video struct dicts that lack a "bytes" key).
        is_success_column = raw_dataset.data.column("is_success")

        episode_success: list[bool] = [
            self._coerce_success_flag(is_success_column[int(start)].as_py())
            for start, _end in zip(self._ep_starts, self._ep_ends)
        ]

        num_success = sum(bool(v) for v in episode_success)
        logger.info(
            "Scanned %d episode(s) in %s via one frame per episode from is_success; "
            "%d marked successful",
            len(episode_success),
            self._dataset_label,
            num_success,
        )
        return episode_success

    def get_view(
        self, episode: int, frame: int, camera_key: str
    ) -> Optional[np.ndarray]:
        sample = self._sample(episode, frame)
        return self.get_view_from_sample(sample, camera_key)

    def get_view_from_sample(
        self,
        sample: dict,
        camera_key: str,
    ) -> Optional[np.ndarray]:
        try:
            raw = _resolve_alias(sample, camera_key, _IMAGE_KEY_ALIASES)
        except KeyError:
            return None
        return _to_uint8_hwc(raw)

    def get_state(self, episode: int, frame: int, state_key: str) -> np.ndarray:
        global_idx = self._ep_starts[episode] + int(frame)
        if state_key not in self._state_column_cache:
            self._state_column_cache[state_key] = None
            for template in _STATE_KEY_ALIASES:
                resolved = template.format(key=state_key)
                if resolved in self.base.hf_dataset.column_names:
                    self._state_column_cache[state_key] = (
                        self.base.hf_dataset.data.column(resolved)
                    )
                    break
        state_column = self._state_column_cache[state_key]
        if state_column is not None:
            raw = state_column[global_idx].as_py()
            return _to_float32_1d(raw)
        sample = self._sample(episode, frame)
        return self.get_state_from_sample(sample, state_key)

    def get_state_from_sample(self, sample: dict, state_key: str) -> np.ndarray:
        raw = _resolve_alias(sample, state_key, _STATE_KEY_ALIASES)
        return _to_float32_1d(raw)

    def get_raw_sample(self, episode: int, frame: int) -> dict:
        return self._sample(episode, frame)

    def get_raw_pair(
        self,
        episode: int,
        frame_t: int,
        frame_tk: int,
        *,
        camera_keys: Sequence[str],
        video_transform: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> tuple[dict, dict]:
        """Read a frame pair while decoding each camera video only once.

        ``LeRobotDataset.__getitem__`` decodes every video camera for a
        single timestamp. For pair training, calling it twice opens/seeks
        each camera video twice. This path reads metadata from Arrow and
        queries ``[t, tk]`` timestamps together per camera, which keeps the
        sample schema identical while reducing random video decode overhead.
        """
        raw_t = self._metadata_sample(episode, frame_t)
        raw_tk = self._metadata_sample(episode, frame_tk)

        video_keys = set(getattr(self.meta, "video_keys", []))
        camera_keys = tuple(camera_keys)
        requested_video_keys = [key for key in camera_keys if key in video_keys]
        if len(requested_video_keys) != len(camera_keys):
            return (
                self.get_raw_sample(episode, frame_t),
                self.get_raw_sample(episode, frame_tk),
            )
        if not requested_video_keys:
            return raw_t, raw_tk

        timestamps = [
            float(_scalar_item(raw_t["timestamp"])),
            float(_scalar_item(raw_tk["timestamp"])),
        ]
        ep_idx = int(_scalar_item(raw_t.get("episode_index", episode)))
        for camera_key in requested_video_keys:
            frames = self.base._query_videos({camera_key: timestamps}, ep_idx)[
                camera_key
            ]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            frame_t_tensor = frames[0]
            frame_tk_tensor = frames[1]
            raw_t[camera_key] = (
                video_transform(frame_t_tensor)
                if video_transform is not None
                else frame_t_tensor
            )
            raw_tk[camera_key] = (
                video_transform(frame_tk_tensor)
                if video_transform is not None
                else frame_tk_tensor
            )

        return raw_t, raw_tk

    def get_prompt(self, episode: int, frame: int) -> str:
        sample = self._sample(episode, frame)
        return self.get_prompt_from_sample(sample, episode, frame)

    def get_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame: int,
    ) -> str:
        task = sample.get("task")
        if isinstance(task, str) and task:
            return task
        ti = sample.get("task_index")
        if ti is None:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has no 'task' string and no "
                "'task_index' field; cannot resolve per-episode task instruction."
            )
        ti_int = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
        if not self._tasks:
            raise RuntimeError(
                f"PairDataset: sample for episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but the dataset "
                "has no meta/tasks.jsonl (or meta/tasks.parquet) to resolve the "
                "instruction."
            )
        prompt = self._tasks.get(int(ti_int))
        if not prompt:
            raise RuntimeError(
                f"PairDataset: episode={episode} frame={frame} in "
                f"{self._dataset_label!r} has task_index={ti_int} but it is not "
                f"present in meta/tasks.jsonl "
                f"(available indices: {sorted(self._tasks.keys())})."
            )
        return prompt

    def episode_is_success(self, episode: int) -> bool:
        if self.dataset_type == "sft":
            return True
        if self._episode_success is None:
            return True
        return bool(self._episode_success[episode])


# ---------------------------------------------------------------------------
# Pair dataset
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    """Yields ``(frame_t, frame_{t+k})`` pairs with multi-view per frame.

    Args:
        dataset_path: LeRobot dataset path.
        camera_keys: Camera view names to load per frame. These match the
            processor's ``image_keys`` — the collator feeds images under
            exactly these keys, and the processor fills any missing ones
            with zero placeholders (mask=False). Default:
            ``("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")``.
        k: Forward pair stride.
        include_state: If ``True``, samples carry ``state`` (proprio at
            ``t``) and ``state_tk`` (reserved).
        state_max_dim: Pad / truncate state to this dim.
        state_key: Fuzzy LeRobot state alias.
        dataset_type: Must be explicitly provided and be either ``"sft"``
            or ``"rollout"``. ``sft`` datasets are treated as all-success
            episodes, so they do not require an ``is_success`` column.
        only_success: Must be explicitly provided and currently must be
            ``True``. Keeps only episodes whose per-frame ``is_success``
            column marks the episode as successful. For LeRobot datasets
            this checks one representative frame row per episode rather
            than relying on episode-level metadata files.
        open_rewind: Whether to emit negative rewound pairs. When ``False``,
            the dataset becomes positive-only: each temporal anchor appears
            once and labels discretize positive strides ``1..k`` into
            ``num_bins`` bins.
        min_episode_length: Optional override for the minimum-length
            floor (default ``k + 1``).
        length_scale_enabled: If ``True`` (multi-bin modes only), the signed
            stride is length-normalized before binning:
            ``scaled = signed_stride * L_max / L_ep``, where ``L_ep`` is the
            current episode length and ``L_max`` is the reference length. Binning
            stays on the ``±K`` axis (bin width ``2K/num_bins``, same resolution
            as the unscaled path); ``L_max`` only sets the scale factor, and
            ``|scaled| > K`` saturates into the extreme bin. An episode of length
            ``L_max`` reproduces the unscaled layout (``scale == 1``); shorter
            episodes push a fixed frame stride into a higher bin (and saturate
            earlier), longer episodes under-reach. No-op in binary mode
            (``num_bins == 2``), where positive scaling preserves the stride sign.
        length_scale_percentile: Percentile of eligible-episode lengths used as
            the reference length ``L_max`` when no explicit
            ``length_scale_reference`` is supplied (default ``90``; ``100`` ⇒
            true max). Acts as the pivot length: episodes near ``L_max`` use the
            full bin range, shorter ones saturate, longer ones under-reach — so
            a lower percentile (e.g. the median) saturates fewer episodes.
        length_scale_reference: Explicit ``L_max``. When ``None`` and
            ``length_scale_enabled``, it is computed per-dataset from
            ``length_scale_percentile``; callers wanting one global ``L_max``
            across a mixture inject it via :meth:`set_length_scale_reference`.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        camera_keys: Sequence[str] = (
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        k: int = 4,
        include_state: bool = False,
        state_max_dim: Optional[int] = None,
        state_key: str = "state",
        dataset_type: Optional[str] = None,
        only_success: Optional[bool] = None,
        open_rewind: bool = True,
        min_episode_length: Optional[int] = None,
        num_bins: int = 2,
        length_scale_enabled: bool = False,
        length_scale_percentile: float = 90.0,
        length_scale_reference: Optional[float] = None,
        state_transform_enabled: bool = False,
        robot_type: str = "libero",
        model_type: str = "pi05",
        action_dim: int = 32,
        default_prompt: Optional[str] = None,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        allow_raw_state_for_compatibility: bool = False,
        compatibility_negative_enabled: bool = False,
        compatibility_num_same_episode_negatives: int = 1,
        compatibility_same_episode_negative_max_distance: Optional[int] = None,
    ) -> None:
        self.camera_keys: tuple[str, ...] = tuple(camera_keys)
        if not self.camera_keys:
            raise ValueError("camera_keys must be non-empty")
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        self.open_rewind = bool(open_rewind)
        # Mode switch. num_bins == 2 → legacy binary mode: fixed-stride k.
        # num_bins > 2 → multi-bin: sample i uniformly from [1, min(K, T-1-t)]
        # per-anchor at __getitem__ time. Both emit a long bin-index label
        # in [0, num_bins); the bin layout (_signed_stride_to_bin) places
        # regressive bins in [0, num_bins // 2) and progressive bins in
        # [num_bins // 2, num_bins), so binary degenerates to 0 = regress,
        # 1 = progress. 2K must be an integer multiple of num_bins so every
        # bin covers the same number of strides (uniform bin widths).
        self.num_bins = int(num_bins)
        if self.open_rewind and (self.num_bins < 2 or self.num_bins % 2 != 0):
            raise ValueError(f"num_bins must be >= 2 and even, got {self.num_bins}")
        if self.open_rewind and self.num_bins > 2 and (2 * self.k) % self.num_bins != 0:
            raise ValueError(
                f"For num_bins={self.num_bins} in multi-bin mode, 2*k must be a "
                f"multiple of num_bins; got k={self.k} (2*k={2 * self.k})."
            )
        if not self.open_rewind and (
            self.num_bins < 1 or self.num_bins > self.k or self.k % self.num_bins != 0
        ):
            raise ValueError(
                f"Positive-only mode requires 1 <= num_bins <= k and "
                f"k % num_bins == 0; got k={self.k}, num_bins={self.num_bins}."
            )
        self.length_scale_enabled = bool(length_scale_enabled)
        self.length_scale_percentile = float(length_scale_percentile)
        # Resolved per-dataset below once eligible episodes are known; a global
        # mixture-wide L_max can override it via set_length_scale_reference.
        self._length_scale_reference: Optional[float] = (
            None if length_scale_reference is None else float(length_scale_reference)
        )
        if self.length_scale_enabled:
            if not (0.0 < self.length_scale_percentile <= 100.0):
                raise ValueError(
                    "length_scale_percentile must be in (0, 100], got "
                    f"{self.length_scale_percentile}"
                )
            if self.open_rewind and self.num_bins == 2:
                logger.warning(
                    "PairDataset length_scale_enabled has no effect in binary "
                    "mode (num_bins == 2): scaling by L_max / L_ep preserves the "
                    "stride sign, so labels are unchanged. Set num_bins > 2 for "
                    "length-scaled multi-bin labels."
                )
        self.include_state = bool(include_state)
        self.state_max_dim = state_max_dim
        self.state_key = state_key
        self.state_transform_enabled = bool(state_transform_enabled)
        self.compatibility_negative_enabled = bool(compatibility_negative_enabled)
        self.compatibility_num_same_episode_negatives = max(
            0, int(compatibility_num_same_episode_negatives)
        )
        self.compatibility_same_episode_negative_max_distance = (
            None
            if compatibility_same_episode_negative_max_distance is None
            else int(compatibility_same_episode_negative_max_distance)
        )
        if (
            self.compatibility_same_episode_negative_max_distance is not None
            and self.compatibility_same_episode_negative_max_distance <= 0
        ):
            raise ValueError(
                "compatibility_same_episode_negative_max_distance must be null or > 0"
            )
        self.allow_raw_state_for_compatibility = bool(allow_raw_state_for_compatibility)
        self._state_transform = None
        self._rng: np.random.Generator | None = None
        if self.state_transform_enabled:
            if norm_stats_dir is None and not self.allow_raw_state_for_compatibility:
                raise ValueError(
                    "PairDataset state compatibility requires data.norm_stats_dir "
                    "unless allow_raw_state_for_compatibility=true."
                )
            self._state_transform = build_openpi_state_transform(
                robot_type=str(robot_type),
                model_type=str(model_type),
                action_dim=int(action_dim),
                default_prompt=default_prompt,
                norm_stats_dir=norm_stats_dir,
                asset_id=asset_id,
            )
        self.source_name = str(dataset_path)
        if dataset_type is None:
            raise ValueError(
                "PairDataset requires an explicit dataset_type argument "
                "('sft' or 'rollout')."
            )
        self.dataset_type = str(dataset_type).lower()
        if self.dataset_type not in ("sft", "rollout"):
            raise ValueError(
                f"PairDataset dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}."
            )
        if only_success is None:
            raise ValueError(
                "PairDataset requires an explicit only_success argument. "
                "Set only_success=true."
            )
        self.only_success = bool(only_success)
        if not self.only_success:
            raise ValueError(
                "PairDataset currently only supports only_success=True. "
                "Please remove the override or set only_success=true."
            )

        self._source = _LeRobotSource(
            dataset_path,
            only_success=self.only_success,
            dataset_type=self.dataset_type,
        )

        # Default floor: any episode with at least 2 frames can form a pair
        # (t=0, t+k clamped to T-1). Yamls can raise this if they want to
        # exclude episodes that can't supply at least one full stride-k pair.
        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)
        total_eps = self._source.num_episodes()
        self._eligible = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
            and (not self.only_success or self._source.episode_is_success(ep))
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes found with length >= {self._min_episode_length} "
                f"and only_success={self.only_success} "
                f"(dataset has {total_eps} episodes)."
            )

        # Per-eligible-episode count of valid pair start positions
        # t ∈ [0, T_ep - 1). For t in [0, T_ep - k) the pair is the regular
        # (t, t+k); for t in [T_ep - k, T_ep - 1) the second slot is clamped
        # to T_ep - 1 (boundary pair, stride < k). Cumulative sum lets
        # __getitem__ map a flat index to (eligible-slot, t) in
        # O(log |eligible|) via searchsorted.
        pair_positions_per_episode = np.array(
            [self._source.episode_length(ep) - 1 for ep in self._eligible],
            dtype=np.int64,
        )
        self._pair_position_ends = np.cumsum(pair_positions_per_episode)
        # Total number of distinct temporal anchors before we duplicate each
        # anchor into its positive and negative training samples.
        self._num_pair_positions = int(self._pair_position_ends[-1])

        # Resolve the per-dataset length-scale reference (L_max) as the
        # configured percentile of eligible-episode lengths, unless an explicit
        # (e.g. global mixture-wide) reference was supplied to the constructor.
        if self.length_scale_enabled and self._length_scale_reference is None:
            self._length_scale_reference = self._compute_length_scale_reference()

        logger.info(
            "PairDataset: dataset_path=%s, episodes=%d eligible=%d, k=%d, "
            "num_bins=%d (%s mode), total_positions=%d, include_state=%s, "
            "dataset_type=%s, only_success=%s, open_rewind=%s, camera_keys=%s",
            self.source_name,
            total_eps,
            len(self._eligible),
            self.k,
            self.num_bins,
            "binary"
            if self.open_rewind and self.num_bins == 2
            else ("multi-bin" if self.open_rewind else "positive-only"),
            self._num_pair_positions,
            self.include_state,
            self.dataset_type,
            self.only_success,
            self.open_rewind,
            self.camera_keys,
        )

    @property
    def source(self) -> TrajectorySource:
        return self._source

    @property
    def eligible_episodes(self) -> list[int]:
        return list(self._eligible)

    def set_epoch(self, epoch: int) -> None:
        del epoch  # no RNG state, retained for DataLoader wrapper compat

    @property
    def num_pair_positions(self) -> int:
        """Number of distinct ``(episode, t)`` anchors before label duplication."""
        return self._num_pair_positions

    @property
    def length_scale_reference(self) -> Optional[float]:
        """The resolved ``L_max`` used to length-scale strides (``None`` if off)."""
        return self._length_scale_reference

    def eligible_episode_lengths(self) -> list[int]:
        """Return the length of every eligible (trained-on) episode."""
        return [int(self._source.episode_length(ep)) for ep in self._eligible]

    def _compute_length_scale_reference(self) -> float:
        """Percentile of eligible-episode lengths, floored at 1."""
        lengths = self.eligible_episode_lengths()
        ref = float(
            np.percentile(
                np.asarray(lengths, dtype=np.float64),
                self.length_scale_percentile,
            )
        )
        return max(1.0, ref)

    def set_length_scale_reference(self, reference: float) -> None:
        """Override ``L_max`` (e.g. a global mixture-wide value).

        No-op when length scaling is disabled, so callers can apply it
        unconditionally across a heterogeneous mixture.
        """
        if not self.length_scale_enabled:
            return
        if reference <= 0:
            raise ValueError(f"length_scale_reference must be > 0, got {reference}")
        self._length_scale_reference = float(reference)

    @staticmethod
    def compute_global_length_scale_reference(
        datasets: Sequence["PairDataset"],
        percentile: float,
    ) -> float:
        """Pool eligible-episode lengths across datasets and return the percentile.

        Used to derive a single mixture-wide ``L_max`` so a fixed frame stride
        maps to the same bin regardless of which dataset the episode came from.
        """
        lengths: list[int] = []
        for ds in datasets:
            lengths.extend(ds.eligible_episode_lengths())
        if not lengths:
            raise ValueError(
                "compute_global_length_scale_reference got no episodes to pool."
            )
        ref = float(
            np.percentile(np.asarray(lengths, dtype=np.float64), float(percentile))
        )
        return max(1.0, ref)

    def __len__(self) -> int:
        if not self.open_rewind:
            return self._num_pair_positions
        # Each temporal anchor contributes two labeled samples:
        #   positive: (t, t+k)
        #   negative: (t+k, t)
        return 2 * self._num_pair_positions

    def _decode_sample_index(self, idx: int) -> tuple[int, bool]:
        """Map a flat dataset index to ``(pair_position, is_positive)``."""
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)

        if not self.open_rewind:
            return idx, True

        pair_position = idx // 2
        is_positive = (idx % 2) == 0
        return pair_position, is_positive

    def _resolve_pair_position(self, pair_position: int) -> tuple[int, int, int]:
        """Map a pair-position index to ``(episode, t, t_plus_k)``."""
        episode_slot = int(
            np.searchsorted(self._pair_position_ends, pair_position, side="right")
        )
        prev_episode_end = (
            int(self._pair_position_ends[episode_slot - 1]) if episode_slot > 0 else 0
        )
        episode = int(self._eligible[episode_slot])
        t = int(pair_position - prev_episode_end)
        # Boundary clamp: when t+k overruns the episode, use the last
        # available frame as the second slot. Stride degrades to T-1-t < k.
        t_plus_k = min(t + self.k, self._source.episode_length(episode) - 1)
        return episode, t, t_plus_k

    def _resolve_prompt(self, episode: int, frame_idx: int) -> str:
        """Return the per-sample task instruction; raises if missing."""
        return self._source.get_prompt(episode, frame_idx)

    def _resolve_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame_idx: int,
    ) -> str:
        """Return the per-sample task instruction from an already-loaded sample."""
        return self._source.get_prompt_from_sample(sample, episode, frame_idx)

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _load_state(self, episode: int, frame_idx: int) -> np.ndarray:
        """Load raw or OpenPI-normalized state and pad/truncate to max_state_dim."""
        state = self._source.get_state(episode, frame_idx, self.state_key)
        return self._normalize_loaded_state(state)

    def _load_state_from_sample(self, sample: dict) -> np.ndarray:
        """Load raw or OpenPI-normalized state from an already-loaded sample."""
        state = self._source.get_state_from_sample(sample, self.state_key)
        return self._normalize_loaded_state(state)

    def _normalize_loaded_state(self, state: Any) -> np.ndarray:
        """Normalize and pad/truncate a raw state payload."""
        if self._state_transform is not None:
            state = self._state_transform(state)
        return _to_float32_1d(state, max_dim=self.state_max_dim)

    def _sample_same_episode_negative_frame(
        self,
        episode: int,
        frame_idx: int,
    ) -> int | None:
        episode_length = self._source.episode_length(episode)
        if episode_length <= 1:
            return None
        max_distance = self.compatibility_same_episode_negative_max_distance
        if max_distance is not None:
            lo = max(0, int(frame_idx) - int(max_distance))
            hi = min(episode_length - 1, int(frame_idx) + int(max_distance))
            candidates = [idx for idx in range(lo, hi + 1) if idx != int(frame_idx)]
            if not candidates:
                return None
            rng = self._rng_for_worker()
            return int(candidates[int(rng.integers(low=0, high=len(candidates)))])
        # Sample from [0, T-2], then shift around the positive frame so the
        # negative is guaranteed to be from a different timestamp.
        draw = int(self._rng_for_worker().integers(low=0, high=episode_length - 1))
        if draw >= int(frame_idx):
            draw += 1
        return draw

    def _build_sample(
        self,
        *,
        episode: int,
        frame_idx_t: int,
        frame_idx_tk: int,
        prompt: str,
        label,
        raw_t: Optional[dict] = None,
        raw_tk: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Assemble the sample dict for a single labeled frame pair.

        ``label`` is a Python ``int`` bin index in ``[0, num_bins)``. The
        collator casts the batched column to ``torch.long``. Binary mode
        (``num_bins == 2``) degenerates to ``0 = regress``, ``1 =
        progress``, matching the multi-bin layout from
        :func:`_signed_stride_to_bin`.
        """
        views_t, mask_t = self._load_views(episode, frame_idx_t, sample=raw_t)
        views_tk, mask_tk = self._load_views(episode, frame_idx_tk, sample=raw_tk)

        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": label,
            "episode": int(episode),
            "frame_idx_t": int(frame_idx_t),
            "frame_idx_tk": int(frame_idx_tk),
            "source_name": self.source_name,
        }

        if self.include_state:
            state_t = (
                self._load_state_from_sample(raw_t)
                if raw_t is not None
                else self._load_state(episode, frame_idx_t)
            )
            state_tk = (
                self._load_state_from_sample(raw_tk)
                if raw_tk is not None
                else self._load_state(episode, frame_idx_tk)
            )
            sample["state"] = state_t  # consumed by state-in-prompt branch
            sample["state_tk"] = state_tk  # reserved for future extensions
            if self.compatibility_negative_enabled:
                neg_states_t: list[np.ndarray] = []
                neg_states_tk: list[np.ndarray] = []
                neg_dist_t: list[float] = []
                neg_dist_tk: list[float] = []
                for _ in range(self.compatibility_num_same_episode_negatives):
                    neg_t = self._sample_same_episode_negative_frame(
                        episode, frame_idx_t
                    )
                    neg_tk = self._sample_same_episode_negative_frame(
                        episode, frame_idx_tk
                    )
                    if neg_t is None or neg_tk is None:
                        continue
                    neg_states_t.append(self._load_state(episode, neg_t))
                    neg_states_tk.append(self._load_state(episode, neg_tk))
                    neg_dist_t.append(float(abs(int(neg_t) - frame_idx_t)))
                    neg_dist_tk.append(float(abs(int(neg_tk) - frame_idx_tk)))
                if neg_states_t and neg_states_tk:
                    sample["state_neg_t"] = np.stack(neg_states_t)
                    sample["state_neg_tk"] = np.stack(neg_states_tk)
                    sample["state_neg_distance_t"] = np.asarray(
                        neg_dist_t, dtype=np.float32
                    )
                    sample["state_neg_distance_tk"] = np.asarray(
                        neg_dist_tk, dtype=np.float32
                    )

        return sample

    def _load_views(
        self,
        episode: int,
        frame_idx: int,
        *,
        sample: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for camera_key in self.camera_keys:
            view = (
                self._source.get_view_from_sample(sample, camera_key)
                if sample is not None
                else self._source.get_view(episode, frame_idx, camera_key)
            )
            if view is None:
                masks[camera_key] = False
            else:
                views[camera_key] = view
                masks[camera_key] = True
        return views, masks

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair_position, is_positive = self._decode_sample_index(idx)
        episode, t, t_plus_k_binary = self._resolve_pair_position(pair_position)

        if self.open_rewind and self.num_bins == 2:
            # Binary path: fixed stride k with the existing boundary
            # clamp (t+k may degrade to T-1 near episode end). Labels are
            # long bin indices matching the multi-bin layout — 1 for
            # progress (positive stride), 0 for regress (negative stride).
            if is_positive:
                frame_idx_t, frame_idx_tk = t, t_plus_k_binary
                label: Any = 1
            else:
                frame_idx_t, frame_idx_tk = t_plus_k_binary, t
                label = 0
        else:
            # Multi-bin path: sample i uniformly from [1, min(K, T-1-t)]
            # at getitem time so every anchor gets exposure to every
            # valid stride over enough epochs. No boundary clamp — the
            # emitted bin always matches the true stride.
            episode_length = self._source.episode_length(episode)
            max_valid_stride = min(self.k, episode_length - 1 - t)
            if max_valid_stride < 1:
                # Should not happen: _pair_position_ends enumerates only
                # t ≤ T-2, so episode_length - 1 - t ≥ 1. Fail-loud per
                # 78bc04dd rather than silently handle.
                raise RuntimeError(
                    f"PairDataset: no valid stride for episode={episode} "
                    f"t={t} episode_length={episode_length} (bug in anchor "
                    "enumeration)."
                )
            i = int(self._rng_for_worker().integers(low=1, high=max_valid_stride + 1))
            if is_positive:
                frame_idx_t, frame_idx_tk = t, t + i
                signed_stride = i
            else:
                frame_idx_t, frame_idx_tk = t + i, t
                signed_stride = -i
            if self.length_scale_enabled and self._length_scale_reference is not None:
                # Length-normalize the stride so a fixed frame jump maps to a
                # higher progress bin in shorter episodes:
                #   scaled = signed_stride * L_max / L_ep
                # L_max only sets the scale factor; binning stays on the ±K axis
                # (bin width 2K/num_bins, same resolution as the unscaled path),
                # with |scaled| > K saturating into the extreme bin. An episode
                # of length L_max reproduces the unscaled layout (scale == 1).
                scale = self._length_scale_reference / float(episode_length)
                if self.open_rewind:
                    label = _scaled_signed_stride_to_bin(
                        signed_stride * scale, self.k, self.num_bins
                    )
                else:
                    label = _scaled_positive_stride_to_bin(
                        i * scale, self.k, self.num_bins
                    )
            elif self.open_rewind:
                label = _signed_stride_to_bin(signed_stride, self.k, self.num_bins)
            else:
                label = _positive_stride_to_bin(i, self.k, self.num_bins)

        raw_t, raw_tk = self._source.get_raw_pair(
            episode,
            frame_idx_t,
            frame_idx_tk,
            camera_keys=self.camera_keys,
        )
        # Preserve the previous behavior: language is resolved from the
        # anchor frame ``t`` even for reversed/negative pairs. One of the two
        # already-loaded frames is always the anchor, so this avoids an extra
        # source read.
        if frame_idx_t == t:
            prompt_sample = raw_t
        elif frame_idx_tk == t:
            prompt_sample = raw_tk
        else:
            prompt_sample = raw_t
        prompt = self._resolve_prompt_from_sample(prompt_sample, episode, t)

        return self._build_sample(
            episode=episode,
            frame_idx_t=frame_idx_t,
            frame_idx_tk=frame_idx_tk,
            prompt=prompt,
            label=label,
            raw_t=raw_t,
            raw_tk=raw_tk,
        )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


@dataclass
class BinaryPairDataCollator:
    """Collator that produces the backbone's observation dict for binary pairs.

    Parallel to :class:`~rlinf.models.embodiment.value_model.data_collator.\
ValueDataCollator`. Runs the :class:`SteamProcessor` **twice**
    — once for frame_t's multi-view images, once for frame_{t+k}'s — then
    stacks the per-camera outputs along a new ``num_frames`` axis. The
    backbone receives per-camera tensors of shape
    ``[B, num_frames, 3, H, W]``.

    Attributes:
        processor: :class:`SteamProcessor` with ``image_keys``
            matching the dataset's ``camera_keys``.
        max_length: Token padding length.
        train: If ``True``, the processor's image augmentations fire.
        num_bins: Matches the paired :class:`PairDataset`'s ``num_bins``.
            Used only for validation assertions — labels are always
            emitted as ``torch.long`` bin indices in ``[0, num_bins)``,
            so binary (``num_bins == 2``) and multi-bin (``num_bins >
            2``) share the same tensor dtype.
    """

    processor: Any
    max_length: int = 200
    train: bool = True
    num_bins: int = 2

    def _collect_per_camera(
        self,
        examples: list[dict[str, Any]],
        images_key: str,
        masks_key: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Gather per-camera image tensors at a single timestamp.

        Missing camera entries for a sample turn into zero placeholders so
        the processor always sees a rectangular camera dict; the returned
        mask records which samples actually had the view.
        """
        camera_keys = set()
        for ex in examples:
            camera_keys.update(ex[images_key].keys())
        if not camera_keys:
            return {}, {}

        bsize = len(examples)
        images_out: dict[str, torch.Tensor] = {}
        masks_out: dict[str, torch.Tensor] = {}
        for cam in sorted(camera_keys):
            frames: list[np.ndarray] = []
            mask_vec: list[bool] = []
            shapes: list[tuple[int, ...]] = []
            for ex in examples:
                v = ex[images_key].get(cam)
                if v is None:
                    frames.append(None)  # type: ignore[arg-type]
                    mask_vec.append(False)
                else:
                    frames.append(v)
                    shapes.append(tuple(int(dim) for dim in v.shape))
                    mask_vec.append(bool(ex[masks_key].get(cam, True)))

            unique_shapes = sorted(set(shapes))
            if len(unique_shapes) > 1:
                shape_examples = [
                    {
                        "source": ex.get("source_name", "unknown"),
                        "episode": ex.get("episode"),
                        "frame_idx_t": ex.get("frame_idx_t"),
                        "frame_idx_tk": ex.get("frame_idx_tk"),
                        "shape": None
                        if ex[images_key].get(cam) is None
                        else tuple(int(dim) for dim in ex[images_key][cam].shape),
                    }
                    for ex in examples
                ]
                raise ValueError(
                    "BinaryPairDataCollator saw incompatible raw image shapes "
                    f"for camera={cam!r} at {images_key!r}: {unique_shapes}. "
                    "PairDataset assumes camera tensors are already shape-aligned; "
                    "this usually means your train batch mixed datasets with "
                    "different raw resolutions for the same camera key. "
                    f"Examples: {shape_examples}"
                )

            if unique_shapes:
                h, w = unique_shapes[0][:2]
            else:
                h, w = 1, 1

            # Replace None entries with zero placeholders matching the first
            # real frame's spatial size. If no real frame exists for this
            # camera across the whole batch, fall back to 1x1.
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            stacked = torch.from_numpy(
                np.stack([f if f is not None else placeholder for f in frames])
            )
            images_out[cam] = stacked
            masks_out[cam] = torch.tensor(mask_vec, dtype=torch.bool)

        # Ensure every example yields the same bsize (torch stack invariant).
        for cam, t in images_out.items():
            if t.shape[0] != bsize:
                raise RuntimeError(f"Unexpected batch shape for cam={cam}: {t.shape}")
        return images_out, masks_out

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("BinaryPairDataCollator received an empty batch")

        prompts: list[str] = [ex["prompt"] for ex in examples]
        states_list: list[Optional[np.ndarray]] = [ex.get("state") for ex in examples]

        # Frame-t and frame-tk run through the processor independently so
        # image augmentations are sampled per frame and so the per-camera
        # masks can differ between the two timestamps (e.g. a wrist view
        # that blinks on/off mid-episode).
        images_t, masks_t = self._collect_per_camera(
            examples, "image_t", "image_mask_t"
        )
        images_tk, masks_tk = self._collect_per_camera(
            examples, "image_tk", "image_mask_tk"
        )

        processed_t = self.processor.image_processor(
            images=images_t,
            image_masks=masks_t,
            return_tensors="pt",
            train=self.train,
        )
        processed_tk = self.processor.image_processor(
            images=images_tk,
            image_masks=masks_tk,
            return_tensors="pt",
            train=self.train,
        )

        # After process_images, pixel_values is a dict[cam → [B, 3, H, W]]
        # covering the **processor's** image_keys (missing camera keys get
        # zero-filled with mask=False at that stage). Stacking along a new
        # dim=1 gives [B, num_frames, 3, H, W] per camera.
        pixel_values_t = processed_t["pixel_values"]
        pixel_values_tk = processed_tk["pixel_values"]
        camera_keys_out = sorted(set(pixel_values_t) | set(pixel_values_tk))
        if not camera_keys_out:
            raise RuntimeError(
                "Processor returned no camera views — check image_keys / "
                "camera_keys alignment between dataset and processor."
            )

        def _stack_over_time(cam: str) -> tuple[torch.Tensor, torch.Tensor]:
            v_t = pixel_values_t.get(cam)
            v_tk = pixel_values_tk.get(cam)
            if v_t is None or v_tk is None:
                raise RuntimeError(
                    f"Camera {cam!r} missing from one of the two per-frame "
                    "processor outputs — this indicates inconsistent batch "
                    "shapes; investigate the dataset sample schema."
                )
            img_stacked = torch.stack([v_t, v_tk], dim=1)  # [B, 2, 3, H, W]
            m_t = processed_t["image_masks"][cam]
            m_tk = processed_tk["image_masks"][cam]
            mask_stacked = torch.stack([m_t, m_tk], dim=1).to(torch.bool)
            return img_stacked, mask_stacked

        images_observation: dict[str, torch.Tensor] = {}
        masks_observation: dict[str, torch.Tensor] = {}
        for cam in camera_keys_out:
            img, mask = _stack_over_time(cam)
            images_observation[cam] = img
            masks_observation[cam] = mask

        any_state = any(s is not None for s in states_list)
        state_batch: Any = None
        if any_state:
            template = next((s for s in states_list if s is not None), None)
            state_dim = int(template.shape[0])
            state_batch = np.stack(
                [
                    (
                        np.asarray(s, dtype=np.float32).reshape(-1)
                        if s is not None
                        else np.zeros(state_dim, dtype=np.float32)
                    )
                    for s in states_list
                ]
            )

        text_states = (
            state_batch
            if bool(getattr(self.processor, "include_state_in_prompt", False))
            else None
        )
        processed_txt = self.processor.process_text(
            prompts=prompts,
            states=text_states,
            max_length=self.max_length,
            return_tensors="pt",
        )

        observation = {
            "images": images_observation,
            "image_masks": masks_observation,
            "tokenized_prompt": processed_txt["input_ids"],
            "tokenized_prompt_mask": processed_txt["attention_mask"].bool(),
        }
        if any_state:
            state_tk_list = [ex.get("state_tk") for ex in examples]
            if any(s is None for s in states_list) or any(
                s is None for s in state_tk_list
            ):
                raise ValueError(
                    "BinaryPairDataCollator received a mixed state batch. "
                    "When any sample has state, every sample must provide both "
                    "'state' and 'state_tk'."
                )
            observation["state_t"] = torch.tensor(state_batch, dtype=torch.float32)
            observation["state_tk"] = torch.tensor(
                np.stack(
                    [np.asarray(s, dtype=np.float32).reshape(-1) for s in state_tk_list]
                ),
                dtype=torch.float32,
            )

            neg_t_list = [ex.get("state_neg_t") for ex in examples]
            neg_tk_list = [ex.get("state_neg_tk") for ex in examples]
            if all(s is not None for s in neg_t_list) and all(
                s is not None for s in neg_tk_list
            ):
                observation["state_neg_t"] = torch.tensor(
                    np.stack([np.asarray(s, dtype=np.float32) for s in neg_t_list]),
                    dtype=torch.float32,
                )
                observation["state_neg_tk"] = torch.tensor(
                    np.stack([np.asarray(s, dtype=np.float32) for s in neg_tk_list]),
                    dtype=torch.float32,
                )
                observation["state_neg_distance_t"] = torch.tensor(
                    np.stack(
                        [
                            np.asarray(
                                ex.get("state_neg_distance_t", 0.0),
                                dtype=np.float32,
                            ).reshape(-1)
                            for ex in examples
                        ]
                    ),
                    dtype=torch.float32,
                )
                observation["state_neg_distance_tk"] = torch.tensor(
                    np.stack(
                        [
                            np.asarray(
                                ex.get("state_neg_distance_tk", 0.0),
                                dtype=np.float32,
                            ).reshape(-1)
                            for ex in examples
                        ]
                    ),
                    dtype=torch.float32,
                )

        episode = torch.tensor(
            [int(ex["episode"]) for ex in examples], dtype=torch.long
        )
        frame_idx_t = torch.tensor(
            [int(ex["frame_idx_t"]) for ex in examples], dtype=torch.long
        )
        frame_idx_tk = torch.tensor(
            [int(ex["frame_idx_tk"]) for ex in examples], dtype=torch.long
        )
        observation["episode"] = episode
        observation["frame_idx_t"] = frame_idx_t
        observation["frame_idx_tk"] = frame_idx_tk

        # Labels are always long bin indices in [0, num_bins). Binary
        # (num_bins == 2) uses 0 = regress, 1 = progress; multi-bin uses
        # the _signed_stride_to_bin layout. Both feed straight into
        # ``F.cross_entropy`` with no further remapping.
        labels = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)
        return {
            "observation": observation,
            "labels": labels,
            "episode": episode,
            "frame_idx_t": frame_idx_t,
            "frame_idx_tk": frame_idx_tk,
        }


class BinaryPairInferenceDataset(Dataset):
    """Yields one ``(frame_t, frame_{t+k})`` pair per anchor for inference.

    Differences vs :class:`PairDataset`:
        * No success-only filter — every episode contributes pairs (matches
          ``compute_advantages.py`` which scores every frame).
        * Forward direction only: ``image_t = frame_t``, ``image_tk = frame_{t+k}``,
          ``label = 0`` (placeholder so the existing collator works).
        * Boundary clamp identical to PairDataset: when ``t + k > T - 1`` the
          second slot is clamped to ``T - 1``.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        camera_keys: list[str],
        k: int,
        prompt: Optional[str],
        include_state: bool,
        state_max_dim: Optional[int],
        state_key: str,
        dataset_type: str,
        min_episode_length: Optional[int] = None,
        state_transform_enabled: bool = False,
        robot_type: str = "libero",
        model_type: str = "pi05",
        action_dim: int = 32,
        default_prompt: Optional[str] = None,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        allow_raw_state_for_compatibility: bool = False,
    ) -> None:
        if dataset_type not in ("sft", "rollout"):
            raise ValueError(
                "BinaryPairInferenceDataset.dataset_type must be 'sft' or 'rollout', "
                f"got {dataset_type!r}"
            )
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not camera_keys:
            raise ValueError("camera_keys must be non-empty")

        self.k = int(k)
        self.camera_keys = tuple(camera_keys)
        self.prompt = prompt
        self.include_state = bool(include_state)
        self.state_max_dim = state_max_dim
        self.state_key = str(state_key)
        self.dataset_type = dataset_type
        self.source_name = str(dataset_path)
        self.state_transform_enabled = bool(state_transform_enabled)
        self.allow_raw_state_for_compatibility = bool(allow_raw_state_for_compatibility)
        self._state_transform = None
        if self.state_transform_enabled:
            if norm_stats_dir is None and not self.allow_raw_state_for_compatibility:
                raise ValueError(
                    "BinaryPairInferenceDataset state compatibility requires "
                    "data.norm_stats_dir or per-dataset norm_stats_dir unless "
                    "data.allow_raw_state_for_compatibility=true."
                )
            self._state_transform = build_openpi_state_transform(
                robot_type=str(robot_type),
                model_type=str(model_type),
                action_dim=int(action_dim),
                default_prompt=default_prompt,
                norm_stats_dir=norm_stats_dir,
                asset_id=asset_id,
            )

        # Iterate every episode regardless of is_success; _LeRobotSource with
        # only_success=False skips the success scan and treats all episodes
        # as eligible.
        self._source = _LeRobotSource(
            dataset_path,
            only_success=False,
            dataset_type=dataset_type,
        )

        if min_episode_length is None:
            min_episode_length = 2
        self._min_episode_length = int(min_episode_length)

        total_eps = self._source.num_episodes()
        self._eligible: list[int] = [
            ep
            for ep in range(total_eps)
            if self._source.episode_length(ep) >= self._min_episode_length
        ]
        if not self._eligible:
            raise ValueError(
                f"No eligible episodes in {dataset_path!r} with length >= "
                f"{self._min_episode_length} (dataset has {total_eps} episodes)."
            )

        # One anchor per t in [0, T - 2]; total = sum(T_ep - 1).
        pair_positions_per_episode = np.array(
            [self._source.episode_length(ep) - 1 for ep in self._eligible],
            dtype=np.int64,
        )
        self._pair_position_ends = np.cumsum(pair_positions_per_episode)
        self._num_pair_positions = int(self._pair_position_ends[-1])

        logger.info(
            "BinaryPairInferenceDataset: source=%s, episodes=%d, k=%d, "
            "total_anchors=%d, include_state=%s, dataset_type=%s, "
            "camera_keys=%s",
            self.source_name,
            len(self._eligible),
            self.k,
            self._num_pair_positions,
            self.include_state,
            self.dataset_type,
            self.camera_keys,
        )

    def __len__(self) -> int:
        return self._num_pair_positions

    def _resolve_pair_position(self, idx: int) -> tuple[int, int, int]:
        if idx < 0 or idx >= self._num_pair_positions:
            raise IndexError(idx)
        episode_slot = int(np.searchsorted(self._pair_position_ends, idx, side="right"))
        prev_episode_end = (
            int(self._pair_position_ends[episode_slot - 1]) if episode_slot > 0 else 0
        )
        episode = int(self._eligible[episode_slot])
        t = int(idx - prev_episode_end)
        t_plus_k = min(t + self.k, self._source.episode_length(episode) - 1)
        return episode, t, t_plus_k

    def _resolve_prompt(self, episode: int, frame_idx: int) -> str:
        prompt = self._source.get_prompt(episode, frame_idx)
        if prompt:
            return prompt
        if self.prompt is None:
            raise RuntimeError(
                f"No per-episode task instruction for episode={episode} in "
                f"{self.source_name!r} and no fallback prompt was provided."
            )
        return self.prompt

    def _resolve_prompt_from_sample(
        self,
        sample: dict,
        episode: int,
        frame_idx: int,
    ) -> str:
        if hasattr(self._source, "get_prompt_from_sample"):
            prompt = self._source.get_prompt_from_sample(sample, episode, frame_idx)
        else:
            prompt = self._source.get_prompt(episode, frame_idx)
        if prompt:
            return prompt
        if self.prompt is None:
            raise RuntimeError(
                f"No per-episode task instruction for episode={episode} in "
                f"{self.source_name!r} and no fallback prompt was provided."
            )
        return self.prompt

    def _load_views(
        self,
        episode: int,
        frame_idx: int,
        *,
        sample: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for cam in self.camera_keys:
            view = (
                self._source.get_view_from_sample(sample, cam)
                if sample is not None and hasattr(self._source, "get_view_from_sample")
                else self._source.get_view(episode, frame_idx, cam)
            )
            if view is None:
                masks[cam] = False
            else:
                views[cam] = view
                masks[cam] = True
        return views, masks

    def _load_state(
        self,
        episode: int,
        frame_idx: int,
        *,
        sample: Optional[dict] = None,
    ) -> np.ndarray:
        if sample is not None and hasattr(self._source, "get_state_from_sample"):
            state = self._source.get_state_from_sample(sample, self.state_key)
            if self._state_transform is not None:
                state = self._state_transform(state)
            return _to_float32_1d(state, max_dim=self.state_max_dim)

        if self._state_transform is None:
            state = self._source.get_state(episode, frame_idx, self.state_key)
        else:
            sample = dict(self._source.get_raw_sample(episode, frame_idx))
            raw_state = self._source.get_state_from_sample(sample, self.state_key)
            state = self._state_transform(raw_state)
        return _to_float32_1d(state, max_dim=self.state_max_dim)

    def _build_sample_from_pair(
        self,
        *,
        episode: int,
        t: int,
        t_plus_k: int,
        raw_t: Optional[dict] = None,
        raw_tk: Optional[dict] = None,
    ) -> dict[str, Any]:
        prompt = (
            self._resolve_prompt_from_sample(raw_t, episode, t)
            if raw_t is not None
            else self._resolve_prompt(episode, t)
        )
        views_t, mask_t = self._load_views(episode, t, sample=raw_t)
        views_tk, mask_tk = self._load_views(
            episode,
            t_plus_k,
            sample=raw_tk,
        )
        sample: dict[str, Any] = {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": mask_t,
            "image_mask_tk": mask_tk,
            "prompt": prompt,
            "label": 0,  # placeholder; collator emits but inference ignores
            "episode": int(episode),
            "frame_idx_t": int(t),
            "frame_idx_tk": int(t_plus_k),
            "source_name": self.source_name,
        }

        if self.include_state:
            state_t = self._load_state(episode, t, sample=raw_t)
            state_tk = self._load_state(episode, t_plus_k, sample=raw_tk)
            sample["state"] = state_t
            sample["state_tk"] = state_tk

        return sample

    def _supports_batched_video_query(self) -> bool:
        if not hasattr(self._source, "base") or not hasattr(
            self._source.base,
            "_query_videos",
        ):
            return False
        if not hasattr(self._source, "_metadata_sample"):
            return False
        video_keys = set(getattr(self._source.meta, "video_keys", []))
        return bool(video_keys) and all(cam in video_keys for cam in self.camera_keys)

    @staticmethod
    def _scalar_item(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"Expected scalar-like list/tuple, got {value!r}")
            return BinaryPairInferenceDataset._scalar_item(value[0])
        return value

    def _getitems_batched_video(self, indices: list[int]) -> list[dict[str, Any]]:
        resolved: list[tuple[int, int, int, int]] = []
        by_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for out_idx, idx in enumerate(indices):
            episode, t, t_plus_k = self._resolve_pair_position(int(idx))
            resolved.append((out_idx, episode, t, t_plus_k))
            by_episode[episode].append((out_idx, t, t_plus_k))

        out: list[Optional[dict[str, Any]]] = [None] * len(indices)
        for episode, episode_entries in by_episode.items():
            frame_indices = sorted(
                {
                    frame
                    for _out_idx, t, t_plus_k in episode_entries
                    for frame in (t, t_plus_k)
                }
            )
            metadata_by_frame = {
                frame: self._source._metadata_sample(episode, frame)
                for frame in frame_indices
            }
            timestamps = [
                float(self._scalar_item(metadata_by_frame[frame]["timestamp"]))
                for frame in frame_indices
            ]
            ep_idx = int(
                self._scalar_item(
                    metadata_by_frame[frame_indices[0]].get("episode_index", episode)
                )
            )

            frames_by_camera = self._source.base._query_videos(
                dict.fromkeys(self.camera_keys, timestamps),
                ep_idx,
            )
            frame_lookup: dict[tuple[str, int], Any] = {}
            for cam in self.camera_keys:
                frames = frames_by_camera[cam]
                if frames.ndim == 3:
                    frames = frames.unsqueeze(0)
                if int(frames.shape[0]) != len(frame_indices):
                    raise RuntimeError(
                        "_query_videos returned an unexpected number of frames for "
                        f"camera={cam!r}: got {int(frames.shape[0])}, expected "
                        f"{len(frame_indices)}"
                    )
                for pos, frame_idx in enumerate(frame_indices):
                    frame_lookup[(cam, frame_idx)] = frames[pos]

            for out_idx, t, t_plus_k in episode_entries:
                raw_t = dict(metadata_by_frame[t])
                raw_tk = dict(metadata_by_frame[t_plus_k])
                for cam in self.camera_keys:
                    raw_t[cam] = frame_lookup[(cam, t)]
                    raw_tk[cam] = frame_lookup[(cam, t_plus_k)]
                out[out_idx] = self._build_sample_from_pair(
                    episode=episode,
                    t=t,
                    t_plus_k=t_plus_k,
                    raw_t=raw_t,
                    raw_tk=raw_tk,
                )

        if any(sample is None for sample in out):
            missing = [i for i, sample in enumerate(out) if sample is None]
            raise RuntimeError(
                f"Batched video loader missed sample positions: {missing}"
            )
        return [sample for sample in out if sample is not None]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        if self._supports_batched_video_query():
            return self._getitems_batched_video([int(idx) for idx in indices])
        return [self[int(idx)] for idx in indices]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, t, t_plus_k = self._resolve_pair_position(idx)

        if hasattr(self._source, "get_raw_pair"):
            raw_t, raw_tk = self._source.get_raw_pair(
                episode,
                t,
                t_plus_k,
                camera_keys=self.camera_keys,
            )
            return self._build_sample_from_pair(
                episode=episode,
                t=t,
                t_plus_k=t_plus_k,
                raw_t=raw_t,
                raw_tk=raw_tk,
            )
        return self._build_sample_from_pair(
            episode=episode,
            t=t,
            t_plus_k=t_plus_k,
        )


__all__ = [
    "BinaryPairDataCollator",
    "BinaryPairInferenceDataset",
    "PairDataset",
    "TrajectorySource",
    "build_openpi_state_transform",
    "positive_bin_centers",
]
