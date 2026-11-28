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

"""Pure NumPy/JSON normalization statistics matching ``openpi.shared.normalize``."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np


@dataclasses.dataclass
class NormStats:
    """Per-feature normalization statistics compatible with OpenPI ``norm_stats.json``."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean)
        self.std = np.asarray(self.std)
        if self.q01 is not None:
            self.q01 = np.asarray(self.q01)
        if self.q99 is not None:
            self.q99 = np.asarray(self.q99)


class RunningStats:
    """Compute running mean, std, and approximate 1st/99th quantiles of vectors."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        """Update running statistics with a batch of vectors shaped ``[..., D]``."""
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(
                    self._min[i] - 1e-10,
                    self._max[i] + 1e-10,
                    self._num_quantile_bins + 1,
                )
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    "The length of new vectors does not match the initialized vector length."
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """Compute and return accumulated :class:`NormStats`."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self) -> None:
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )
            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def _stats_to_jsonable(stats: NormStats) -> dict[str, Any]:
    return {
        "mean": np.asarray(stats.mean).tolist(),
        "std": np.asarray(stats.std).tolist(),
        "q01": np.asarray(stats.q01).tolist() if stats.q01 is not None else None,
        "q99": np.asarray(stats.q99).tolist() if stats.q99 is not None else None,
    }


def _stats_from_jsonable(data: dict[str, Any]) -> NormStats:
    return NormStats(
        mean=np.asarray(data["mean"]),
        std=np.asarray(data["std"]),
        q01=np.asarray(data["q01"]) if data.get("q01") is not None else None,
        q99=np.asarray(data["q99"]) if data.get("q99") is not None else None,
    )


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize normalization statistics to an OpenPI-compatible JSON string."""
    payload = {
        "norm_stats": {
            key: _stats_to_jsonable(stats) for key, stats in norm_stats.items()
        }
    }
    return json.dumps(payload, indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize normalization statistics from an OpenPI-compatible JSON string."""
    raw = json.loads(data)
    stats_map = raw["norm_stats"] if "norm_stats" in raw else raw
    return {key: _stats_from_jsonable(val) for key, val in stats_map.items()}


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save normalization statistics to ``<directory>/norm_stats.json``."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load normalization statistics from ``<directory>/norm_stats.json``."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
