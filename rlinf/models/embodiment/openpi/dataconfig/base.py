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

"""Pure Python OpenPI data and training config dataclasses (JAX/tyro-free)."""

from __future__ import annotations

import abc
import dataclasses
import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from typing_extensions import override

from rlinf.models.embodiment.openpi.modules import model as _model
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.transforms import core as _transforms
from rlinf.models.embodiment.openpi.transforms import download as _download
from rlinf.models.embodiment.openpi.transforms import normalize as _normalize
from rlinf.models.embodiment.openpi.transforms import tokenizer as _tokenizer

logger = logging.getLogger(__name__)

_REMOTE_SCHEMES = ("gs://", "s3://", "http://", "https://")


def _join_asset_path(assets_dir: pathlib.Path | str, asset_id: str) -> str:
    raw_dir = str(assets_dir)
    if raw_dir.startswith(_REMOTE_SCHEMES):
        return f"{raw_dir.rstrip('/')}/{asset_id.lstrip('/')}"
    return str(pathlib.Path(raw_dir) / asset_id)


def load_norm_stats(
    assets_dir: pathlib.Path | str, asset_id: str
) -> dict[str, _normalize.NormStats]:
    """Load normalization stats from ``<assets_dir>/<asset_id>/norm_stats.json``."""
    norm_stats_dir = _join_asset_path(assets_dir, asset_id)
    norm_stats = _normalize.load(_download.maybe_download(norm_stats_dir))
    logger.info("Loaded norm stats from %s", norm_stats_dir)
    return norm_stats


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) for the data pipeline."""

    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    use_quantile_norm: bool = False

    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False

    # Retained for DROID RLDS config compatibility.
    rlds_data_dir: str | None = None
    action_space: Any = None
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: Any) -> _transforms.Group:
        """Create a transform group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Create model transforms for Pi0, Pi0.5, and Pi0-FAST models."""

    default_prompt: str | None = None

    def __call__(self, model_config: Any) -> _transforms.Group:
        model_type = _model.coerce_model_type(model_config.model_type)
        match model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=bool(
                                getattr(model_config, "discrete_state_input", True)
                            ),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                fast_cls = getattr(model_config, "fast_model_tokenizer", None)
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer if fast_cls is None else fast_cls
                )
                fast_kwargs = getattr(model_config, "fast_model_tokenizer_kwargs", None)
                tokenizer_kwargs = {} if fast_kwargs is None else fast_kwargs
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(
                                model_config.max_token_len, **tokenizer_kwargs
                            ),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | None = None
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: DataConfig | None = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path | str, model_config: Any
    ) -> DataConfig:
        """Create a data config."""

    def create_base_config(
        self, assets_dirs: pathlib.Path | str, model_config: Any
    ) -> DataConfig:
        repo_id = self.repo_id if isinstance(self.repo_id, str) else None
        asset_id = self.assets.asset_id or repo_id
        model_type = _model.coerce_model_type(model_config.model_type)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                self.assets.assets_dir or assets_dirs, asset_id
            ),
            use_quantile_norm=model_type != _model.ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: pathlib.Path | str, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        data_assets_dir = _join_asset_path(assets_dir, asset_id)
        try:
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logger.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logger.info("Norm stats not found in %s, skipping.", data_assets_dir)
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str | None = "fake"

    @override
    def create(
        self, assets_dirs: pathlib.Path | str, model_config: Any
    ) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    data_transforms: GroupFactory = dataclasses.field(
        default_factory=lambda: (lambda _model_config: _transforms.Group())
    )
    model_transforms: GroupFactory = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path | str, model_config: Any
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """OpenPI model + data training/evaluation configuration (JAX-free)."""

    name: str
    project: str = "openpi"
    exp_name: str | None = None

    model: Any = dataclasses.field(default_factory=Pi0Config)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"

    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    # Optional metadata retained for config compatibility.
    ema_decay: float | None = 0.99
    weight_loader: Any = None
    lr_schedule: Any = None
    optimizer: Any = None
    freeze_filter: Any = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (
            pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
