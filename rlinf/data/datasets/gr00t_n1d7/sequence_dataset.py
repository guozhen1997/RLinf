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

"""Sequence SFT dataset for GR00T N1.7 RoboTTT.

The upstream GR00T factory only emits single-step samples. RoboTTT needs a
contiguous window of ``T`` steps so the TTT inner loop can walk the
trajectory. This dataset reuses ``LeRobotEpisodeLoader``,
``extract_step_data`` and ``Gr00tN1d7Processor`` to assemble those windows,
and the collator flattens them to ``[B * T, ...]`` in batch-major order.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    _batchify_gr00t_forward_input,
    _find_processor_dir,
    redirect_qwen3_backbone_to_local,
)
from rlinf.utils.logging import get_logger

logger = get_logger()

_DEFAULT_BACKBONE_MODEL_NAME = "nvidia/Cosmos-Reason2-2B"

_EMBODIMENT_TAG_BY_CFG = {
    "libero_sim": "libero_sim",
    "libero_panda": "libero_panda",
    "libero_franka": "libero_franka",
    "isaaclab_franka": "isaaclab_franka",
    "maniskill_widowx": "maniskill_widowx",
    "robocasa_panda_omron": "robocasa_panda_omron",
    "gr1": "gr1",
    "behavior_r1_pro": "behavior_r1_pro",
    "new_embodiment": "new_embodiment",
    "so101": "new_embodiment",
    "so100": "new_embodiment",
}


def _resolve_backbone_model_path(backbone_model_path: Optional[str]) -> Optional[str]:
    if backbone_model_path is None:
        return None
    resolved = str(Path(backbone_model_path).expanduser().resolve())
    if not Path(resolved).is_dir():
        raise FileNotFoundError(f"Backbone model path does not exist: {resolved}")
    return resolved


def _load_processor(model_path: str, backbone_model_path: Optional[str] = None):
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    processor_dir = _find_processor_dir(Path(model_path))
    if processor_dir is None:
        raise FileNotFoundError(
            f"No GR00T N1.7 processor files under {model_path}. "
            "Expected processor_config.json, statistics.json, embodiment_id.json."
        )
    with open(processor_dir / "processor_config.json", "r") as f:
        processor_cfg = json.load(f)["processor_kwargs"]
    with open(processor_dir / "statistics.json", "r") as f:
        processor_cfg["statistics"] = json.load(f)
    with open(processor_dir / "embodiment_id.json", "r") as f:
        processor_cfg["embodiment_id_mapping"] = json.load(f)
    resolved_backbone = _resolve_backbone_model_path(backbone_model_path)
    if resolved_backbone is not None:
        processor_cfg.setdefault("transformers_loading_kwargs", {})
        processor_cfg["transformers_loading_kwargs"]["local_files_only"] = True
        logger.info(
            "Loading processor backbone locally from %s (canonical model_name=%s)",
            resolved_backbone,
            processor_cfg.get("model_name", _DEFAULT_BACKBONE_MODEL_NAME),
        )
    canonical_name = str(
        processor_cfg.get("model_name", _DEFAULT_BACKBONE_MODEL_NAME)
    )
    with redirect_qwen3_backbone_to_local(canonical_name, resolved_backbone):
        processor = Gr00tN1d7Processor(**processor_cfg)
    processor.training = True
    return processor


def _resolve_embodiment_tag(tag_name: str):
    from rlinf.models.embodiment.gr00t.embodiment_tags import EmbodimentTag

    mapped = _EMBODIMENT_TAG_BY_CFG.get(tag_name, tag_name)
    return EmbodimentTag(mapped)


class Gr00tN1d7SequenceDataset(Dataset):
    """Sample contiguous ``T``-step sub-trajectories from a LeRobot dataset.

    Optionally prepends a human-video context segment (loss-masked) when
    ``human_video_path`` points at a paired LeRobot dataset. The human segment
    uses placeholder actions so TTT still updates, but no imitation loss is
    applied.
    """

    def __init__(
        self,
        dataset_path: str,
        processor: Any,
        embodiment_tag: Any,
        num_timesteps: int = 16,
        seed: int = 0,
        video_backend: str = "torchcodec",
        human_video_path: Optional[str] = None,
        human_video_timesteps: int = 0,
        samples_per_epoch: Optional[int] = None,
    ):
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self.processor = processor
        self.embodiment_tag = embodiment_tag
        self.num_timesteps = int(num_timesteps)
        self.human_video_timesteps = int(human_video_timesteps)
        self.rng = np.random.RandomState(seed)

        tag_value = embodiment_tag.value
        if tag_value not in processor.modality_configs:
            raise KeyError(
                f"Processor has no modality config for {tag_value}; "
                f"available: {list(processor.modality_configs)}"
            )
        self.modality_configs = processor.modality_configs[tag_value]
        self.action_horizon = len(self.modality_configs["action"].delta_indices)

        self.loader = LeRobotEpisodeLoader(
            dataset_path=dataset_path,
            modality_configs=self.modality_configs,
            video_backend=video_backend,
        )
        self._index = self._build_index(self.loader, self.num_timesteps)
        if not self._index:
            raise ValueError(
                f"No episode in {dataset_path} is long enough for "
                f"num_timesteps={self.num_timesteps} with action_horizon="
                f"{self.action_horizon}."
            )
        self.samples_per_epoch = (
            int(samples_per_epoch) if samples_per_epoch is not None else len(self._index)
        )

        self.human_loader = None
        self._human_rows: dict[int, tuple[int, int, int]] = {}
        if human_video_path and self.human_video_timesteps > 0:
            self.human_loader = LeRobotEpisodeLoader(
                dataset_path=human_video_path,
                modality_configs=self.modality_configs,
                video_backend=video_backend,
            )
            human_index = self._build_index(
                self.human_loader, self.human_video_timesteps
            )
            if not human_index:
                raise ValueError(
                    f"No episode in {human_video_path} is long enough for "
                    f"human_video_timesteps={self.human_video_timesteps} with "
                    f"action_horizon={self.action_horizon}."
                )
            human_by_id = {row[0]: row for row in human_index}
            # Pair by episode id when both datasets share ids; otherwise by order
            # so every robot window has a human prefix of fixed length T_h.
            n_id_matches = 0
            for i, (robot_ep, _, _) in enumerate(self._index):
                if robot_ep in human_by_id:
                    self._human_rows[robot_ep] = human_by_id[robot_ep]
                    n_id_matches += 1
                else:
                    self._human_rows[robot_ep] = human_index[i % len(human_index)]
            if n_id_matches == 0:
                logger.warning(
                    "human_video_path=%s has no episode ids overlapping the "
                    "robot dataset; pairing human videos by order.",
                    human_video_path,
                )
            elif n_id_matches < len(self._index):
                logger.warning(
                    "human_video_path=%s matched %d/%d robot episodes by id; "
                    "the rest are paired by order so collated windows stay "
                    "length T_h + T.",
                    human_video_path,
                    n_id_matches,
                    len(self._index),
                )

    def _build_index(self, loader, num_timesteps: int) -> list[tuple[int, int, int]]:
        """``(episode_id, max_start, length)`` for episodes that fit a window."""
        index = []
        for episode_id, length in enumerate(loader.episode_lengths):
            # Last step of the window is start+T-1 and still needs H action
            # frames, so start <= length - H - T + 1. Zero is a legal single
            # window at the start of the episode; negative means too short.
            max_start = int(length) - self.action_horizon - (num_timesteps - 1)
            if max_start >= 0:
                index.append((episode_id, max_start, int(length)))
        return index

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _process_step(self, episode_data, step_index: int, *, mask_loss: bool) -> dict:
        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

        step = extract_step_data(
            episode_data,
            step_index,
            self.modality_configs,
            self.embodiment_tag,
            allow_padding=True,
        )
        if mask_loss:
            zero_actions = {
                key: np.zeros_like(value) for key, value in step.actions.items()
            }
            step = replace(step, actions=zero_actions, is_demonstration=True)
        processed = self.processor([{"content": step}])
        return processed

    def _sample_window(self, loader, index_row, num_timesteps: int, *, mask_loss: bool):
        episode_id, max_start, _ = index_row
        start = int(self.rng.randint(0, max_start + 1))
        episode_data = loader[episode_id]
        steps = [
            self._process_step(episode_data, start + t, mask_loss=mask_loss)
            for t in range(num_timesteps)
        ]
        loss_mask = np.zeros(num_timesteps, dtype=np.float32) if mask_loss else np.ones(
            num_timesteps, dtype=np.float32
        )
        return steps, loss_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        del index  # windows are sampled, not enumerated
        robot_row = self._index[int(self.rng.randint(0, len(self._index)))]
        robot_steps, robot_mask = self._sample_window(
            self.loader, robot_row, self.num_timesteps, mask_loss=False
        )

        context_steps: list[dict[str, Any]] = []
        context_mask = np.zeros((0,), dtype=np.float32)
        episode_id = robot_row[0]
        if self.human_loader is not None:
            human_row = self._human_rows[episode_id]
            try:
                context_steps, context_mask = self._sample_window(
                    self.human_loader,
                    human_row,
                    self.human_video_timesteps,
                    mask_loss=True,
                )
            except (KeyError, AssertionError, ValueError) as exc:
                logger.warning(
                    "Retrying human-video context for episode %s: %s",
                    episode_id,
                    exc,
                )
                # Fall back to any valid human row so the collated length stays T_h + T.
                fallback = self._human_rows[next(iter(self._human_rows))]
                context_steps, context_mask = self._sample_window(
                    self.human_loader,
                    fallback,
                    self.human_video_timesteps,
                    mask_loss=True,
                )

        return {
            "steps": context_steps + robot_steps,
            "loss_mask": np.concatenate([context_mask, robot_mask], axis=0),
        }


class Gr00tN1d7SequenceCollator:
    """Collate B windows into a ``[B * T, ...]`` batch plus ``num_timesteps``."""

    def __init__(self, model_name: str, transformers_loading_kwargs: Optional[dict] = None):
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self._collator = Gr00tN1d7DataCollator(
            model_name=model_name,
            transformers_loading_kwargs=transformers_loading_kwargs or {},
        )

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        num_timesteps = len(samples[0]["steps"])
        for sample in samples:
            if len(sample["steps"]) != num_timesteps:
                raise ValueError(
                    "All windows in a batch must share the same length; "
                    f"got {len(sample['steps'])} vs {num_timesteps}."
                )
        # Batch-major flatten: sample 0's T steps, then sample 1's, ...
        flat_steps = [step for sample in samples for step in sample["steps"]]
        collated = self._collator(flat_steps)
        batch = dict(collated["inputs"] if "inputs" in collated else collated)
        batch_size = len(samples) * num_timesteps
        for key in ("pixel_values", "image_grid_thw", "image_sizes"):
            if key in batch:
                batch[key] = _batchify_gr00t_forward_input(key, batch[key], batch_size)
        batch["loss_mask"] = torch.from_numpy(
            np.stack([sample["loss_mask"] for sample in samples])
        )
        batch["num_timesteps"] = num_timesteps
        return batch


def build_gr00t_n1d7_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths,
    eval_dataset: bool = False,
):
    """Build the GR00T N1.7 sequence SFT dataloader."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.apply()

    model_cfg = cfg.actor.model
    data_cfg = cfg.data
    dataset_path = data_paths if isinstance(data_paths, str) else data_paths[0]
    backbone_model_path = _resolve_backbone_model_path(
        model_cfg.get("backbone_model_path", None)
    )
    processor = _load_processor(
        str(model_cfg.model_path),
        backbone_model_path=backbone_model_path,
    )
    embodiment_tag = _resolve_embodiment_tag(str(model_cfg.embodiment_tag))
    robottt_cfg = data_cfg.get("robottt", {}) or {}
    num_timesteps = int(robottt_cfg.get("num_timesteps", data_cfg.get("num_timesteps", 16)))
    dataset = Gr00tN1d7SequenceDataset(
        dataset_path=str(dataset_path),
        processor=processor,
        embodiment_tag=embodiment_tag,
        num_timesteps=num_timesteps,
        seed=int(model_cfg.get("seed", cfg.actor.get("seed", 0))) + rank,
        video_backend=str(data_cfg.get("video_backend", "torchcodec")),
        human_video_path=robottt_cfg.get("human_video_path", None),
        human_video_timesteps=int(robottt_cfg.get("human_video_timesteps", 0) or 0),
        samples_per_epoch=data_cfg.get("samples_per_epoch", None),
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    with redirect_qwen3_backbone_to_local(
        processor.model_name, backbone_model_path
    ):
        collator = Gr00tN1d7SequenceCollator(
            model_name=processor.model_name,
            transformers_loading_kwargs=getattr(
                processor, "transformers_loading_kwargs", {"trust_remote_code": True}
            ),
        )
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=int(cfg.actor.micro_batch_size),
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=not eval_dataset,
        collate_fn=collator,
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2))
        if int(data_cfg.get("num_workers", 4)) > 0
        else None,
    )
    logger.info(
        "GR00T N1.7 RoboTTT SFT dataset: %d windows, T=%d, from %s",
        len(dataset),
        num_timesteps,
        dataset_path,
    )
    return data_loader, {"num_samples": len(dataset)}
