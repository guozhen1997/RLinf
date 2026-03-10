# Copyright 2025 The RLinf Authors.
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

"""Reward Workers for RLinf.

This module provides:
- RewardWorker: For inference/computing rewards during RL training
- FSDPRewardWorker: For training reward models with FSDP (like FSDPSftWorker)
"""

import re as _re

import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models.embodiment.reward import ResNetRewardModel
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import clear_memory
from rlinf.utils.logging import get_logger

from rlinf.data.datasets.reward_model import (
    load_episodes_with_labels,
    balance_and_split_by_episode,
    RewardBinaryDataset,
)

from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.utils.down_sampling import down_sample_batch

class RewardWorker(Worker):
    """Reward Worker for inference during RL training."""

    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self.cfg = cfg

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )
        self.do_down_sampling = self.cfg.algorithm.get("down_sampling", {}).get("do_down_sampling", False)
        if self.do_down_sampling:
            self.down_sampling_config = self.cfg.algorithm.get("down_sampling", {}).get("down_sampling_config", {})

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            self.reward_model = get_reward_model_class(self.cfg.reward.model.model_type)(self.cfg.reward.model)
        else:
            self.rule_based_reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

        if self.cfg.reward.get("tokenizer", None) is not None:
            self.tokenizer = hf_tokenizer(self.cfg.reward.tokenizer.tokenizer_model)

    @Worker.timer("compute_rewards")
    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            if rollout_result.rewards is None:
                if self.cfg.reward.use_reward_model:
                    with self.device_lock:
                        batch = rollout_result.to_actor_batch(
                            self.cfg.data.max_prompt_length,
                            self.cfg.actor.model.encoder_seq_length,
                            self.tokenizer.eos_token_id,
                        )
                        rollout_result.rewards = (
                            self.compute_batch_rewards_with_model(batch)
                        )
                else:
                    rollout_result.rewards = self._compute_rule_based_rewards(
                        rollout_result
                    )
            if self.do_down_sampling:
                if rollout_result.response_texts is None:
                    rollout_result.response_texts = [
                        self.tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in rollout_result.response_ids
                    ]
                rollout_result = down_sample_batch(rollout_result, self.down_sampling_config)
            # answer is not needed in training
            rollout_result.answers = None

            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        # Decode only the generated tokens; response_ids are already the post-prompt tokens
        texts = rollout_result.response_texts
        if texts is None:
            texts = self.tokenizer.batch_decode(
                rollout_result.response_ids, skip_special_tokens=True
            )

        kwargs = {}
        if getattr(self.cfg.reward, "use_prompt", False):
            prompts = rollout_result.prompt_texts
            if prompts is None:
                prompts = self.tokenizer.batch_decode(
                    rollout_result.prompt_ids, skip_special_tokens=True
                )
            kwargs["prompts"] = prompts
        scores = self.rule_based_reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def compute_batch_rewards_with_model(self, batch: dict[str, torch.Tensor]):
        raise NotImplementedError("Reward model is not implemented yet.")


# =============================================================================
# FSDP Reward Worker for Training (like FSDPSftWorker)
# =============================================================================

class FSDPRewardWorker(FSDPModelManager, Worker):
    """FSDP-based worker for reward model training.

    This follows the same pattern as FSDPSftWorker:
    - Inherits FSDPModelManager for optimizer, scheduler, FSDP setup
    - Implements model_provider_func() to return ResNetRewardModel
    - Implements build_dataloader() for data loading
    - Implements run_training() for training loop
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.val_loader = self.build_dataloader()
        self.data_iter = iter(self.data_loader) if self.data_loader else None

        # Training step counter for validation interval
        self._training_step = 0
        self._val_interval = cfg.runner.get("val_check_interval", 50)

    def init_worker(self):
        """Initialize model and optimizer using base class."""
        self.tokenizer = None
        self.is_lora = False
        self.setup_model_and_optimizer()

        self.logger.info(
            f"Initialized FSDPRewardWorker with "
            f"{sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def _save_dataset_images(
        self, dataset: RewardBinaryDataset, save_dir: str, split: str
    ) -> None:
        """Save all images from dataset for debugging.

        Args:
            dataset: The dataset to save images from.
            save_dir: Base directory to save images.
            split: 'train' or 'val' to indicate which split.
        """
        from PIL import Image
        from tqdm import tqdm

        success_dir = os.path.join(save_dir, split, "success")
        fail_dir = os.path.join(save_dir, split, "fail")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)

        success_count = 0
        fail_count = 0

        self.logger.info(f"Saving {len(dataset)} {split} images to {save_dir}...")

        for idx in tqdm(range(len(dataset)), desc=f"Saving {split} images"):
            img_tensor, label = dataset[idx]

            # Convert tensor to numpy image
            # img_tensor is (C, H, W) float [0, 1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img_np)

            is_success = label.item() > 0.5
            if is_success:
                img_pil.save(os.path.join(success_dir, f"{split}_{idx:05d}.png"))
                success_count += 1
            else:
                img_pil.save(os.path.join(fail_dir, f"{split}_{idx:05d}.png"))
                fail_count += 1

        self.logger.info(f"Saved {split} images: {success_count} success, {fail_count} fail")

    def model_provider_func(self) -> torch.nn.Module:
        """Provide the ResNet reward model."""
        model_cfg = self.cfg.actor.model
        model = ResNetRewardModel(model_cfg)
        return model

    def build_dataloader(self) -> tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Build training and validation dataloaders.

        Uses per-frame 'is_obj_placed' labels from infos instead of episode-level labels.
        Splits by EPISODE to prevent data leakage (adjacent frames are too similar).
        Uses sparse sampling (start/middle/end) to reduce redundancy.
        """
        data_cfg = self.cfg.get("data", {})
        data_path = data_cfg.get("data_path")

        if not data_path or not os.path.exists(data_path):
            self.logger.warning(f"Data path not found: {data_path}")
            return None, None

        # Load episodes with configurable sampling
        num_samples = data_cfg.get("num_samples_per_episode", 5)
        self.logger.info(
            f"Loading episodes from {data_path} with {num_samples} samples per episode..."
        )
        episodes = load_episodes_with_labels(data_path, num_samples)

        if len(episodes) == 0:
            self.logger.warning("No episodes loaded from dataset")
            return None, None

        # Split by EPISODE (prevents data leakage), sample with fail:success ratio
        val_split = data_cfg.get("val_split", 0.2)
        fail_success_ratio = data_cfg.get("fail_success_ratio", 2.0)
        train_dataset, val_dataset = balance_and_split_by_episode(
            episodes, val_split, fail_success_ratio
        )

        if len(train_dataset) == 0:
            self.logger.warning("Training dataset is empty after balancing")
            return None, None

        # Debug: save training images to verify data pipeline
        debug_save_dir = data_cfg.get("debug_save_dir", None)
        if debug_save_dir and self._rank == 0:
            self._save_dataset_images(train_dataset, debug_save_dir, "train")
            self._save_dataset_images(val_dataset, debug_save_dir, "val")

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=False,
        )

        batch_size = self.cfg.actor.micro_batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        self.logger.info(
            f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val"
        )

        return train_loader, val_loader

    def run_training(self) -> dict[str, float]:
        """Run one training iteration with gradient accumulation.

        Follows the same pattern as FSDPSftWorker.run_training().
        """
        # Check if data loader is available
        if self.data_iter is None or self.data_loader is None:
            raise RuntimeError(
                "Data loader is not available. Please check that:\n"
                f"  1. Data path exists: {self.cfg.get('data', {}).get('data_path')}\n"
                "  2. Data path is an absolute path (not relative)\n"
                "  3. Data files (*_success.pkl and *_fail.pkl) are present"
            )

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()

            assert (
                self.cfg.actor.global_batch_size
                % (self.cfg.actor.micro_batch_size * self._world_size)
                == 0
            ), "global_batch_size is not divisible by micro_batch_size * world_size"

            self.gradient_accumulation = (
                self.cfg.actor.global_batch_size
                // self.cfg.actor.micro_batch_size
                // self._world_size
            )

            metrics = {}

            for idx in range(self.gradient_accumulation):
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                # Get batch (image, label)
                try:
                    images, labels = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    images, labels = next(self.data_iter)

                # Move to device: images shape is (B, C, H, W), labels shape is (B,)
                images = images.to(self.device)
                labels = labels.to(self.device)

                with self.amp_context:
                    # Forward pass - loss computed inside model
                    outputs = self.model(images, labels)
                    loss = outputs["loss"]

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

                # Accumulate metrics
                append_to_dict(
                    metrics,
                    {
                        "loss": outputs["loss"].item(),
                        "accuracy": outputs["accuracy"].item(),
                        "probabilities_mean": outputs["probabilities"].mean().item(),
                    },
                )

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            # Increment step counter and run validation at interval
            self._training_step += 1
            if self._training_step % self._val_interval == 0:
                val_metrics = self.run_validation()
                train_metrics.update(val_metrics)

            return train_metrics

    def run_validation(self) -> dict[str, float]:
        """Run validation over the entire validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with self.amp_context:
                    outputs = self.model(images, labels)

                append_to_dict(
                    metrics,
                    {
                        "val_loss": outputs["loss"].item(),
                        "val_accuracy": outputs["accuracy"].item(),
                        "val_probabilities_mean": outputs["probabilities"]
                        .mean()
                        .item(),
                    },
                )

        val_metrics = {key: np.mean(value) for key, value in metrics.items()}
        val_metrics = all_reduce_dict(val_metrics, op=torch.distributed.ReduceOp.AVG)

        return val_metrics


# =============================================================================
# Image Reward Worker for Inference (with Channel Communication)
# =============================================================================


class ImageRewardWorker(Worker):
    """Image-based Reward Worker for inference during RL training.

    This worker loads a trained ResNetRewardModel checkpoint and computes
    rewards for images received via channel communication.

    Usage:
        - Configure with checkpoint_path pointing to trained model
        - Connect input_channel (receives images) and output_channel (sends rewards)
        - Call compute_rewards() in the training loop
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        # Device setup
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()


        # Model will be loaded in init_worker
        self.model = None

        self.debug_save_dir = os.environ.get("DEBUG_IMAGE_SAVE_DIR", None)
        self.debug_success_count = 0
        self.debug_fail_count = 0
        self.reward_threshold = 0.6

    def init_worker(self):
        """Initialize the reward model from checkpoint."""
        reward_cfg = self.cfg.get("reward", {})
        model_cfg = reward_cfg.get("model", {})

        # Get debug save dir from config or environment variable
        self.debug_save_dir = model_cfg.get("debug_save_dir") or os.environ.get(
            "DEBUG_IMAGE_SAVE_DIR", None
        )

        # Setup debug image saving directories
        if self.debug_save_dir:
            os.makedirs(os.path.join(self.debug_save_dir, "success"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_save_dir, "fail"), exist_ok=True)
        checkpoint_path = model_cfg.get("checkpoint_path")

        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified for ImageRewardWorker")

        # Create model (will auto-load checkpoint if checkpoint_path is in model_cfg)
        self.model = ResNetRewardModel(model_cfg)

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards for images received from input channel.

        Expected input format via channel:
            dict with keys:
                - 'images': torch.Tensor of shape (B, C, H, W) or (B, H, W, C)
                - 'episode_ids': optional, for tracking

        Output format via channel:
            dict with keys:
                - 'rewards': torch.Tensor of shape (B,)
                - 'episode_ids': passed through if provided

        Args:
            input_channel: Channel to receive image batches from
            output_channel: Channel to send computed rewards to
        """
        with self.worker_timer():
            # Receive data from channel
            data = input_channel.get()

            # Try multiple image keys: reward_images (from StateWithRGBWrapper),
            # images, or main_images
            images = data.get("reward_images")
            if images is None:
                images = data.get("images")
            if images is None:
                images = data.get("main_images")
            if images is None:
                self.logger.warning(
                    "No images in input data (tried: reward_images, images, main_images)"
                )
                output_channel.put({"rewards": None}, async_op=True)
                return

            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images)

            images = images.to(self.device)
            self.model.eval()

            with torch.no_grad():
                outputs = self.model(images)
                probs = outputs["probabilities"]
                rewards = torch.where(
                    probs > self.reward_threshold,
                    probs,
                    torch.zeros_like(probs),
                )

            if self.debug_save_dir:
                original_images = data.get("main_images")
                if original_images is None:
                    original_images = data.get("images")
                if original_images is None:
                    original_images = data.get("reward_images")
                if original_images is not None:
                    if isinstance(original_images, np.ndarray):
                        original_images = torch.from_numpy(original_images)
                    original_images = original_images.to(self.device)
                    if original_images.dim() == 4 and original_images.shape[-1] in [
                        1,
                        3,
                        4,
                    ]:
                        original_images = original_images.permute(0, 3, 1, 2)
                    if original_images.dtype == torch.uint8:
                        original_images = original_images.float() / 255.0
                    self._save_debug_images(original_images, rewards, probs)

            output_data = {"rewards": rewards.cpu()}
            if "episode_ids" in data:
                output_data["episode_ids"] = data["episode_ids"]

            output_channel.put(output_data, async_op=True)

    def compute_reward_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Compute rewards for a batch of images directly (without channel).

        Args:
            images: Tensor of shape (B, C, H, W) or (B, H, W, C)

        Returns:
            rewards: Tensor of shape (B,)
        """
        images = self._preprocess_images(images)
        images = images.to(self.device)
        self.model.eval()

        with torch.no_grad():
            rewards = self.model.compute_reward({"images": images})
        return rewards

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images to (B, C, H, W) in [0, 1]."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        if images.dim() == 4 and images.shape[-1] in [1, 3, 4]:
            images = images.permute(0, 3, 1, 2)

        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.dtype != torch.float32:
            images = images.float()

        return images

    def _save_debug_images(
        self, images: torch.Tensor, rewards: torch.Tensor, probs: torch.Tensor = None
    ):
        """Save debug images based on ResNet classification results.

        Args:
            images: Preprocessed images (B, C, H, W) in [0, 1]
            rewards: Reward values from ResNet model (B,)
            probs: Raw probability outputs from ResNet (B,), used for filename
        """
        from PIL import Image

        images_np = images.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        probs_np = probs.cpu().numpy() if probs is not None else rewards_np

        for idx in range(len(rewards_np)):
            reward = rewards_np[idx]
            prob = probs_np[idx]
            img = images_np[idx]

            if img.shape[0] in [1, 3, 4]:
                img = img.transpose(1, 2, 0)
            img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            is_success = reward > self.reward_threshold
            if is_success:
                self.debug_success_count += 1
                save_dir = os.path.join(self.debug_save_dir, "success")
                filename = f"{self.debug_success_count:06d}_prob{prob:.4f}.png"
            else:
                self.debug_fail_count += 1
                save_dir = os.path.join(self.debug_save_dir, "fail")
                filename = f"{self.debug_fail_count:06d}_prob{prob:.4f}.png"

            os.makedirs(save_dir, exist_ok=True)
            try:
                pil_img = Image.fromarray(img)
                pil_img.save(os.path.join(save_dir, filename))
            except Exception:
                pass

    def run_inference_loop(self, input_channel: Channel, output_channel: Channel):
        """Run continuous inference loop (for use in RL training).

        This method runs until input_channel signals completion (returns None).

        Args:
            input_channel: Channel to receive image batches from
            output_channel: Channel to send computed rewards to
        """
        self.logger.info("Starting ImageRewardWorker inference loop")
        target_key = "0_train"

        while True:
            try:
                data = input_channel.get(key=target_key)
                if data is None:
                    self.logger.info("Received stop signal, ending inference loop")
                    break

                dones = data.get("dones")
                final_obs = data.get("final_obs")

                is_done_flag = False
                if dones is not None:
                    if isinstance(dones, torch.Tensor):
                        is_done_flag = dones.any().item()
                    else:
                        is_done_flag = bool(dones)

                # If Done=True is displayed but there's no final_obs, it means this is the starting frame after a Reset.
                # In this case, Reward should not be calculated, as it would trigger an Invariant check in RolloutWorker.
                if is_done_flag and final_obs is None:
                    output_data = data.copy()  # copy.deepcopy(data)
                    output_data["rewards"] = None
                    output_channel.put(output_data, key=target_key, async_op=True)
                    continue

                images = data.get("images")
                if images is None:
                    images = data.get("main_images")
                if images is None and "obs" in data and isinstance(data["obs"], dict):
                    images = data["obs"].get("main_images")

                if images is None:
                    self.logger.warning("Received data but no images found")
                    output_channel.put(data, key=target_key, async_op=True)
                    continue

                images = self._preprocess_images(images)
                images = images.to(self.device)
                self.model.eval()

                with torch.no_grad():
                    rewards = self.model.compute_reward({"images": images})

                # # print reward model output
                # r_mean = rewards.mean().item()
                # print(f"Model Inference | Reward Val: {r_mean:.4f}", flush=True)

                output_data = data.copy()  # copy.deepcopy(data)
                cpu_rewards = rewards.cpu()
                if cpu_rewards.dim() == 1:
                    cpu_rewards = cpu_rewards.unsqueeze(1)
                output_data["rewards"] = cpu_rewards

                output_channel.put(output_data, key=target_key, async_op=True)

            except Exception as e:
                self.logger.warning(f"Error in inference loop: {e}")
                continue

        self.logger.info("ImageRewardWorker inference loop ended")

    def save_checkpoint(self, save_path: str, global_step: int) -> None:
        """Save reward model checkpoint.

        Args:
            save_path: Directory to save the checkpoint
            global_step: Current global step for naming
        """
        if self.model is None:
            return

        os.makedirs(save_path, exist_ok=True)
        checkpoint_file = os.path.join(save_path, f"reward_model_step_{global_step}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "global_step": global_step,
            },
            checkpoint_file,
        )
