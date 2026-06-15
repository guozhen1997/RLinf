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
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils import _pytree
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.datasets.recap.cfg_model import CfgMixtureDataset
from rlinf.data.datasets.recap.common import BaseDataLoaderImpl
from rlinf.data.datasets.recap.utils import cast_image_features
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class SftPlainDataLoaderImpl(BaseDataLoaderImpl):
    """Yield OpenPI SFT ``(observation, actions)`` tuples from dict batches."""

    @property
    def sampler(self) -> Any:
        return getattr(self._data_loader, "sampler", None)

    @property
    def dataset(self) -> Any:
        return getattr(self._data_loader, "dataset", None)

    def __iter__(self):
        from openpi.models import model as openpi_model

        for batch in self._data_loader:
            observation = openpi_model.Observation.from_dict(batch)
            actions = batch["actions"]
            yield observation, actions


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            if self._uses_dataset_entries(data_paths):
                if eval_dataset:
                    raise NotImplementedError(
                        "eval is not supported for embodied OpenPI SFT with "
                        "list-style data.train_data_paths right now."
                    )
                return self._build_openpi_dataset_entries_dataloader(data_paths)

            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=self._openpi_data_kwargs(),
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    @staticmethod
    def _uses_dataset_entries(data_paths: Any) -> bool:
        """Return True when train_data_paths is a list of dataset config entries."""
        if not isinstance(data_paths, (list, ListConfig)) or len(data_paths) == 0:
            return False
        first = data_paths[0]
        return isinstance(first, (dict, DictConfig)) and "dataset_path" in first

    def _openpi_data_kwargs(self) -> Any:
        """Read OpenPI data overrides from the current and legacy config paths."""
        data_kwargs = getattr(self.cfg.actor.model, "openpi_data", None)
        if data_kwargs is None:
            data_kwargs = getattr(self.cfg.actor, "openpi_data", None)
        return data_kwargs

    @staticmethod
    def _fix_episode_data_index(dataset: Any, episodes: list[int]) -> None:
        """Fix LeRobotDataset episode indices after dataset-level filtering."""
        ep_idx_mapping = {ep: i for i, ep in enumerate(sorted(episodes))}
        max_ep_idx = max(episodes) + 1

        old_from = dataset.episode_data_index["from"]
        old_to = dataset.episode_data_index["to"]

        new_from = torch.full((max_ep_idx,), -1, dtype=old_from.dtype)
        new_to = torch.full((max_ep_idx,), -1, dtype=old_to.dtype)

        for orig_ep, new_idx in ep_idx_mapping.items():
            new_from[orig_ep] = old_from[new_idx]
            new_to[orig_ep] = old_to[new_idx]

        dataset.episode_data_index["from"] = new_from
        dataset.episode_data_index["to"] = new_to

    def _create_distributed_torch_dataloader(
        self, dataset: Any, *, batch_size: int, num_workers: int
    ) -> Any:
        """Create a torch DataLoader with a distributed sampler when needed."""
        sampler = None
        if torch.distributed.is_initialized():
            if batch_size % self._world_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"world_size ({self._world_size})"
                )
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                drop_last=True,
            )
            local_batch_size = batch_size // self._world_size
        else:
            local_batch_size = batch_size

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def _build_openpi_dataset_entries_dataloader(self, datasets_config: Any):
        """Build OpenPI SFT dataloader from one or more LeRobot dataset entries.

        Each entry carries ``dataset_path`` plus optional ``episodes`` /
        ``weight``.
        """
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
        import openpi.shared.download as download
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as transforms
        from openpi.training import checkpoints as _checkpoints

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        data_cfg = self.cfg.get("data", {})
        openpi_cfg = self.cfg.actor.model.openpi

        first_path = datasets_config[0]["dataset_path"]
        config = get_openpi_config(
            openpi_cfg.config_name,
            batch_size=self.cfg.actor.micro_batch_size * self._world_size,
            repo_id=first_path,
            asset_id=openpi_cfg.get("asset_id", None),
            data_kwargs=self._openpi_data_kwargs(),
        )
        data_config = config.data.create(config.assets_dirs, config.model)

        norm_stats = data_config.norm_stats
        if norm_stats is None and data_config.asset_id is not None:
            checkpoint_dir = download.maybe_download(
                str(self.cfg.actor.model.model_path)
            )
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir,
                data_config.asset_id,
            )
        norm_stats = norm_stats or {}

        state_history_size = getattr(
            data_config,
            "state_history_size",
            getattr(config.data, "state_history_size", 0),
        )
        state_future_size = getattr(
            data_config,
            "state_future_size",
            getattr(config.data, "state_future_size", 0),
        )
        state_step = getattr(
            data_config, "state_step", getattr(config.data, "state_step", 1)
        )

        def build_delta_timestamps(fps: float) -> dict[str, list[float]]:
            delta_timestamps = {
                key: [t / fps for t in range(config.model.action_horizon)]
                for key in data_config.action_sequence_keys
            }
            if state_history_size > 0 or state_future_size > 0:
                delta_timestamps["state"] = [
                    t * state_step / fps
                    for t in range(-state_history_size, state_future_size + 1)
                ]
            return delta_timestamps

        datasets_with_weights = []
        for ds_config in datasets_config:
            data_path = ds_config["dataset_path"]
            local_path = Path(data_path).absolute()
            episodes = ds_config.get("episodes")
            weight = float(ds_config.get("weight", 1.0))

            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
                local_path.name, root=local_path
            )
            base_dataset = lerobot_dataset.LeRobotDataset(
                local_path.name,
                root=local_path,
                episodes=episodes,
                delta_timestamps=build_delta_timestamps(dataset_meta.fps),
                download_videos=False,
            )
            base_dataset.hf_dataset = cast_image_features(base_dataset.hf_dataset)

            if episodes is not None:
                self._fix_episode_data_index(base_dataset, episodes)

            if data_config.prompt_from_task:
                base_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                )

            transformed_dataset = openpi_data_loader.TransformedDataset(
                base_dataset,
                [
                    *data_config.repack_transforms.inputs,
                    *data_config.data_transforms.inputs,
                    transforms.Normalize(
                        norm_stats, use_quantiles=data_config.use_quantile_norm
                    ),
                    *data_config.model_transforms.inputs,
                ],
            )

            final_dataset = transformed_dataset
            if self._rank == 0:
                self.log_info(
                    f"Loaded dataset: {data_path} "
                    f"({len(final_dataset)} samples, weight={weight})"
                )

            datasets_with_weights.append((final_dataset, weight))

        combined_dataset = CfgMixtureDataset(
            datasets=datasets_with_weights,
            mode="train",
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
            seed=data_cfg.get("seed", 42),
        )

        data_num_workers = int(data_cfg.get("num_workers", config.num_workers))
        torch_data_loader = self._create_distributed_torch_dataloader(
            combined_dataset,
            batch_size=config.batch_size,
            num_workers=data_num_workers,
        )

        data_loader = SftPlainDataLoaderImpl(data_config, torch_data_loader)
        return data_loader, data_loader.data_config()

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if isinstance(batch, (tuple, list)):
            # Dataset-entries OpenPI path yields (observation, actions) tuples.
            observation, actions = batch[:2]
            register_pytree_dataclasses(observation)
            observation = _pytree.tree_map(
                lambda x: (
                    torch.as_tensor(x, device=self.device).contiguous().clone()
                    if x is not None
                    else x
                ),
                observation,
            )
            actions = actions.to(torch.float32).to(self.device)
            batch = {"observation": observation, "actions": actions}

        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]
        if loss.dim() > 0:
            # OpenPI flow-matching SFT returns per-element losses.
            loss = loss.mean()

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict) and output.get("dynamics_loss", None) is not None:
            step_metrics.update(
                {
                    "dynamics_loss": output["dynamics_loss"].detach().item(),
                    "action_loss": output["action_loss"].detach().item(),
                }
            )
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        if isinstance(torch_data_loader, torch.utils.data.DataLoader):
            # SftPlainDataLoaderImpl wraps the torch DataLoader directly.
            return torch_data_loader
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
