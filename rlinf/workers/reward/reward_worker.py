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


import os
from typing import Dict

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import EmbodiedRolloutResult, RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.models.reward_model.reward_classifier import BinaryRewardClassifier
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement


class RewardWorker(Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        
        # Determine if this is an embodied task
        self.is_embodied_task = self._is_embodied_task()
        
        if self.is_embodied_task:
            # For embodied tasks, batch size is based on env config
            if hasattr(cfg, "env") and hasattr(cfg.env, "train"):
                env_batch_size = cfg.env.train.num_group * cfg.env.train.group_size
            elif hasattr(cfg, "algorithm") and "num_group_envs" in cfg.algorithm:
                env_batch_size = cfg.algorithm.num_group_envs
            else:
                env_batch_size = cfg.get("data", {}).get("rollout_batch_size", 256)
                if env_batch_size == 256:
                    print(f"Warning: Could not find batch size config for embodied task, using default {env_batch_size}")
            
            self.total_batch_size_per_dp = (
                env_batch_size
                * cfg.algorithm.get("group_size", 1)
                // self._world_size
            )
            self.tokenizer = None  # Not needed for embodied tasks
            self.reward = None  # Will be set in init_worker if needed
            self.reward_model = None
        else:
            # For text-based reasoning tasks
            self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
            self.total_batch_size_per_dp = (
                self.cfg.data.rollout_batch_size
                * self.cfg.algorithm.get("group_size", 1)
                // self._world_size
            )
            self.reward = None  # Will be set in init_worker if needed
            self.reward_model = None

    def _is_embodied_task(self) -> bool:
        """Check if this is an embodied task based on config."""
        # Check for runner type
        if hasattr(self.cfg, "runner") and hasattr(self.cfg.runner, "task_type"):
            return self.cfg.runner.task_type == "embodied"
        
        # Check for env config (embodied tasks have env config)
        if hasattr(self.cfg, "env"):
            return True
        
        # Check if reward config specifies embodied reward model
        if hasattr(self.cfg, "reward"):
            reward_cfg = self.cfg.reward
            if reward_cfg.get("use_reward_model", False):
                reward_model_cfg = reward_cfg.get("reward_model", {})
                # If reward_model config has image_keys or image_size, it's likely embodied
                if "image_keys" in reward_model_cfg or "image_size" in reward_model_cfg:
                    return True
        
        return False

    def init_worker(self):
        # Set device first (needed before using self.device)
        self.device = torch.cuda.current_device()
        
        if self.is_embodied_task:
            # Embodied task initialization
            if self.cfg.reward.use_reward_model:
                # Get model configuration
                reward_cfg = self.cfg.reward.reward_model
                image_keys = reward_cfg.get("image_keys", self.cfg.actor.model.image_keys)
                image_size = reward_cfg.get("image_size", self.cfg.actor.model.image_size)
                hidden_dim = reward_cfg.get("hidden_dim", 256)
                num_spatial_blocks = reward_cfg.get("num_spatial_blocks", 8)
                pretrained_encoder_path = reward_cfg.get("pretrained_encoder_path", None)
                use_pretrain = reward_cfg.get("use_pretrain", True)
                freeze_encoder = reward_cfg.get("freeze_encoder", True)

                # Load checkpoint first to check configuration compatibility
                checkpoint_path = reward_cfg.get("checkpoint_path", None)
                checkpoint_state = None
                if checkpoint_path and os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    checkpoint_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                    
                    # Try to infer use_pretrain from checkpoint if possible
                    pooling_key = None
                    for key in checkpoint_state.keys():
                        if "pooling_layer.kernel" in key:
                            pooling_key = key
                            break
                    
                    if pooling_key is not None:
                        kernel_shape = checkpoint_state[pooling_key].shape
                        if kernel_shape[1] == 1 and kernel_shape[2] == 1:
                            inferred_use_pretrain = False
                            print(f"Inferred use_pretrain=False from checkpoint (pooling kernel shape: {kernel_shape})")
                        else:
                            inferred_use_pretrain = True
                            print(f"Inferred use_pretrain=True from checkpoint (pooling kernel shape: {kernel_shape})")
                        
                        if inferred_use_pretrain != use_pretrain:
                            print(f"Warning: use_pretrain mismatch! Config says {use_pretrain}, but checkpoint suggests {inferred_use_pretrain}. Using inferred value.")
                            use_pretrain = inferred_use_pretrain

                # Create reward model
                self.reward_model = BinaryRewardClassifier(
                    image_keys=image_keys,
                    image_size=image_size,
                    hidden_dim=hidden_dim,
                    num_spatial_blocks=num_spatial_blocks,
                    pretrained_encoder_path=pretrained_encoder_path,
                    use_pretrain=use_pretrain,
                    freeze_encoder=freeze_encoder,
                )

                # Load checkpoint if specified
                if checkpoint_path and os.path.exists(checkpoint_path):
                    if checkpoint_state is None:
                        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        checkpoint_state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                    
                    self.reward_model.load_state_dict(checkpoint_state, strict=True)
                    print(f"Loaded reward model from {checkpoint_path}")
                elif checkpoint_path:
                    print(f"Warning: Reward model checkpoint not found at {checkpoint_path}")

                # Move to device
                self.reward_model = self.reward_model.to(self.device)
                self.reward_model.eval()
            else:
                raise NotImplementedError(
                    "Only reward model is supported for embodiment tasks. Set reward.use_reward_model=True"
                )
        else:
            # Text-based reasoning task initialization
            if self.cfg.reward.use_reward_model:
                raise NotImplementedError("Reward model for text-based tasks is not implemented yet.")
            else:
                self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        """Get batch from channel. Only used for text-based reasoning tasks.
        
        Args:
            channel: Channel to read from.
            
        Returns:
            Tuple of (batch, rollout_result).
        """
        if self.is_embodied_task:
            raise NotImplementedError(
                "get_batch is not used for embodied tasks. "
                "Use compute_rewards directly instead."
            )
        
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        This method supports both text-based reasoning tasks (RolloutResult) 
        and embodied tasks (EmbodiedRolloutResult).

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result = input_channel.get()
            
            # Determine result type and count batch size accordingly
            if isinstance(rollout_result, EmbodiedRolloutResult):
                recv_batch_size += len(rollout_result.forward_inputs)
            elif isinstance(rollout_result, RolloutResult):
                recv_batch_size += rollout_result.num_sequence
            else:
                raise ValueError(f"Unexpected rollout result type: {type(rollout_result)}")
            
            with self.worker_timer():
                if isinstance(rollout_result, EmbodiedRolloutResult):
                    # Handle embodied task rewards
                    if rollout_result.rewards is None or len(rollout_result.rewards) == 0:
                        if self.cfg.reward.use_reward_model:
                            with self.device_lock:
                                rewards = self._compute_embodied_rewards_with_model(rollout_result)
                                rollout_result.rewards = rewards
                        else:
                            raise NotImplementedError(
                                "Rule-based rewards not implemented for embodiment tasks. "
                                "Set reward.use_reward_model=True"
                            )
                elif isinstance(rollout_result, RolloutResult):
                    # Handle text-based reasoning task rewards
                    if rollout_result.rewards is None:
                        if self.tokenizer is None:
                            raise ValueError(
                                "Tokenizer is required for text-based reasoning tasks. "
                                "This should not happen if _is_embodied_task() is working correctly."
                            )
                        
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

            output_channel.put(rollout_result)

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
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def compute_batch_rewards_with_model(self, batch: dict[str, torch.Tensor]):
        """Compute rewards for text-based tasks with reward model.
        
        Args:
            batch: Batch of text data for reward computation.
            
        Returns:
            Rewards tensor.
        """
        raise NotImplementedError("Reward model for text-based tasks is not implemented yet.")
    
    def _compute_embodied_rewards_with_model(
        self, rollout_result: EmbodiedRolloutResult
    ) -> list:
        """Compute rewards using the reward model for frame-based classification.

        Args:
            rollout_result: The rollout result containing observations.

        Returns:
            List of reward tensors, one per transition/step.
        """
        all_rewards = []

        # Process transitions (obs, next_obs pairs)
        if hasattr(rollout_result, 'transitions') and rollout_result.transitions:
            for obs, next_obs in rollout_result.transitions:
                # Use next_obs for frame-based reward (current frame prediction)
                images = self._extract_images_from_obs(next_obs)

                # Get reward prediction
                with torch.no_grad():
                    logits = self.reward_model(images, train=False)
                    probs = torch.sigmoid(logits)
                    reward_type = self.cfg.reward.reward_model.get("reward_type", "binary")
                    if reward_type == "binary":
                        rewards = (probs > 0.5).float()
                    else:  # continuous
                        rewards = probs

                    # Handle batch dimension
                    if rewards.dim() == 2 and rewards.shape[1] == 1:
                        rewards = rewards.squeeze(1)
                    all_rewards.append(rewards.cpu())

        # Process forward_inputs (current observations)
        for forward_input in rollout_result.forward_inputs:
            obs = self._extract_obs_from_forward_inputs(forward_input)
            images = self._extract_images_from_obs(obs)

            with torch.no_grad():
                logits = self.reward_model(images, train=False)
                probs = torch.sigmoid(logits)
                reward_type = self.cfg.reward.reward_model.get("reward_type", "binary")
                if reward_type == "binary":
                    rewards = (probs > 0.5).float()
                else:  # continuous
                    rewards = probs

                # Handle batch dimension
                if rewards.dim() == 2 and rewards.shape[1] == 1:
                    rewards = rewards.squeeze(1)
                all_rewards.append(rewards.cpu())

        # Concatenate all rewards
        if len(all_rewards) > 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            # Convert to list of tensors (one per step)
            return [all_rewards_tensor[i] for i in range(len(all_rewards_tensor))]
        else:
            return []
    
    def _extract_obs_from_forward_inputs(self, forward_input: Dict) -> Dict:
        """Extract observation dictionary from forward_inputs.

        Args:
            forward_input: Dictionary with keys like "obs/images/base_camera", "obs/states", etc.

        Returns:
            Dictionary with keys like "images/base_camera", "states", etc.
        """
        obs = {}
        for key, value in forward_input.items():
            if key.startswith("obs/"):
                new_key = key[len("obs/"):]
                obs[new_key] = value
        return obs

    def _extract_images_from_obs(self, obs: Dict) -> Dict[str, torch.Tensor]:
        """Extract and format images from observation dictionary.

        Args:
            obs: Observation dictionary with image keys.

        Returns:
            Dictionary of image tensors in [B, C, H, W] format.
        """
        images = {}
        image_keys = self.cfg.reward.reward_model.get(
            "image_keys", self.cfg.actor.model.image_keys
        )

        for key in image_keys:
            # Try different possible keys
            possible_keys = [
                f"images/{key}",
                key,
                f"obs/images/{key}",
            ]

            image_tensor = None
            for possible_key in possible_keys:
                if possible_key in obs:
                    image_tensor = obs[possible_key]
                    break

            if image_tensor is None:
                raise ValueError(
                    f"Image key {key} not found in observation. "
                    f"Available keys: {list(obs.keys())}"
                )

            # Ensure tensor format [B, C, H, W]
            if isinstance(image_tensor, torch.Tensor):
                if image_tensor.dim() == 3:
                    # Add batch dimension
                    image_tensor = image_tensor.unsqueeze(0)
                elif image_tensor.dim() == 4:
                    pass  # Already in correct format
                else:
                    raise ValueError(
                        f"Unexpected image tensor shape: {image_tensor.shape}"
                    )

                # Move to device if needed
                if image_tensor.device != self.device:
                    image_tensor = image_tensor.to(self.device)

                images[key] = image_tensor
            else:
                raise ValueError(f"Image {key} is not a torch.Tensor: {type(image_tensor)}")

        return images
