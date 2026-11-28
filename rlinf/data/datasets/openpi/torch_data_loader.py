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

"""Pure PyTorch OpenPI LeRobot data loader (JAX/openpi-free)."""

from __future__ import annotations

import logging
import multiprocessing
import typing
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar

import numpy as np
import torch

from rlinf.models.embodiment.openpi.dataconfig import base as _config
from rlinf.models.embodiment.openpi.modules import model as _model
from rlinf.models.embodiment.openpi.transforms import core as _transforms

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a map-style dataset."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for an OpenPI-compatible data loader."""

    def data_config(self) -> _config.DataConfig:
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config."
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    """Apply a sequence of OpenPI transforms to each item of a map-style dataset."""

    def __init__(
        self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]
    ) -> None:
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    """Deterministic synthetic dataset matching an OpenPI model's input/action spec."""

    def __init__(self, model_config: Any, num_samples: int = 1024) -> None:
        self._model_config = model_config
        self._num_samples = num_samples

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        rng = np.random.default_rng(index.__index__())
        action_dim = int(self._model_config.action_dim)
        action_horizon = int(self._model_config.action_horizon)
        max_token_len = int(self._model_config.max_token_len)
        images = {
            key: rng.uniform(-1.0, 1.0, size=(*_model.IMAGE_RESOLUTION, 3)).astype(
                np.float32
            )
            for key in _model.IMAGE_KEYS
        }
        image_masks = {key: np.asarray(True, dtype=np.bool_) for key in _model.IMAGE_KEYS}
        return {
            "image": images,
            "image_mask": image_masks,
            "state": rng.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32),
            "tokenized_prompt": rng.integers(
                0, 2048, size=(max_token_len,), dtype=np.int32
            ),
            "tokenized_prompt_mask": np.ones((max_token_len,), dtype=np.bool_),
            "actions": rng.uniform(
                -1.0, 1.0, size=(action_horizon, action_dim)
            ).astype(np.float32),
        }

    def __len__(self) -> int:
        return self._num_samples


def _import_lerobot_dataset():
    try:  # lerobot >= 0.2 layout
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        return lerobot_dataset
    except ModuleNotFoundError:
        pass
    try:  # lerobot < 0.2 layout
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        return lerobot_dataset
    except ModuleNotFoundError as exc:
        raise ImportError(
            "LeRobot is required to load an OpenPI LeRobot dataset. "
            "Install `lerobot` in the current environment."
        ) from exc


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: Any,
) -> Dataset:
    """Create a map-style LeRobot (or synthetic) dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    lerobot_dataset = _import_lerobot_dataset()
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset


def transform_dataset(
    dataset: Dataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
) -> Dataset:
    """Wrap ``dataset`` with repack, data, normalize, and model transforms."""
    norm_stats: dict[str, Any] = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Make sure to compute norm stats "
                "using `toolkits/lerobot/calculate_norm_stats.py` or provide "
                "`openpi_data.norm_stats_path`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: Any = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["pytorch", "jax"] = "pytorch",
) -> DataLoader[tuple[_model.Observation, torch.Tensor]]:
    """Create a PyTorch OpenPI data loader from a :class:`TrainConfig`."""
    if framework != "pytorch":
        raise NotImplementedError(
            f"RLinf's built-in OpenPI data loader only supports framework='pytorch', got {framework!r}."
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info("data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError(
            "PyTorch RLDS data loader is not supported in RLinf's built-in OpenPI loader."
        )

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: Any,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: Any = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "pytorch",
) -> DataLoader[tuple[_model.Observation, torch.Tensor]]:
    """Create a PyTorch data loader with optional DistributedSampler."""
    if framework != "pytorch":
        raise NotImplementedError(
            f"RLinf's built-in OpenPI data loader only supports framework='pytorch', got {framework!r}."
        )
    del sharding

    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = batch_size

    logger.info("local_batch_size: %d", local_batch_size)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """PyTorch DataLoader matching ``openpi.training.data_loader.TorchDataLoader``."""

    def __init__(
        self,
        dataset: Dataset,
        local_batch_size: int,
        *,
        sharding: Any = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "pytorch",
    ) -> None:
        del sharding, framework
        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )

        self._num_batches = num_batches
        mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self) -> Iterator[dict[str, Any]]:
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield _to_torch_tree(batch)


def _collate_fn(items: Sequence[Any]) -> Any:
    """Recursively stack nested dicts/tuples/lists of leaves into batched numpy arrays."""
    first = items[0]
    if isinstance(first, Mapping):
        return {key: _collate_fn([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(_collate_fn([item[i] for item in items]) for i in range(len(first)))
    if isinstance(first, list):
        return [_collate_fn([item[i] for item in items]) for i in range(len(first))]
    return np.stack([np.asarray(x) for x in items], axis=0)


def _to_torch_tree(tree: Any) -> Any:
    """Recursively convert nested numpy leaves to PyTorch tensors."""
    if isinstance(tree, Mapping):
        return {key: _to_torch_tree(val) for key, val in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_torch_tree(val) for val in tree)
    if isinstance(tree, list):
        return [_to_torch_tree(val) for val in tree]
    return torch.as_tensor(tree)


class DataLoaderImpl(DataLoader):
    """Yield ``(Observation, actions)`` pairs from an underlying batch loader."""

    def __init__(
        self, data_config: _config.DataConfig, data_loader: TorchDataLoader
    ) -> None:
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple[_model.Observation, torch.Tensor]]:
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
