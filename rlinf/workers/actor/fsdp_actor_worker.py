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

import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.data.io_struct import RolloutResult
from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
    seq_mean_token_mean,
    seq_mean_token_sum,
)
from rlinf.workers.rollout.utils import RankMapper


class FSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_data_io_rank = True
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None

        if self.cfg.algorithm.loss_agg_func == "token-mean":
            self.loss_agg_func = masked_mean
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-sum":
            self.loss_agg_func = seq_mean_token_sum
        elif self.cfg.algorithm.loss_agg_func == "seq-mean-token-mean":
            self.loss_agg_func = seq_mean_token_mean
        else:
            raise NotImplementedError(
                f"algorithm.loss_agg_func={self.cfg.algorithm.loss_agg_func} is not supported!"
            )

    def init_worker(self) -> None:
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict

    def sync_model_to_rollout(self) -> None:
        if self.cfg.actor.get("enable_offload", False):
            self.offload_optimizer()

        if next(self.model.parameters()).is_cpu:
            self.load_param_and_grad(self.device, True)
        self.rollout_state_dict = self.get_model_state_dict()

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v)

            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()

    def compute_logprobs(self) -> None:
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def put_result(self, result: RolloutResult, channel: Channel) -> None:
        if channel.is_local:
            # Local channel, every process will put its own data locally
            # No need to broadcast
            channel.put(result)
        else:
            if self.is_data_io_rank:
                channel.put(result)

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits, responses, task_type=self.cfg.runner.task_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            self.put_result(rollout_result, output_channel)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        batch = self.compute_advantages_and_returns(batch)
        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        global_batches = get_iterator_k_split(
            batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for global_batch in global_batches:
                train_global_batch_size = global_batch["input_ids"].shape[0]

                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size=}"
                )

                self.gradient_accumulation = (
                    train_global_batch_size // self.cfg.actor.micro_batch_size
                )
                # split batch into micro_batches
                train_micro_batches = get_iterator_k_split(
                    global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                metrics = {}
                for idx, m_batch in enumerate(train_micro_batches):
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    for k, v in m_batch.items():
                        m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

                    multi_modal_inputs = {}
                    if "multi_modal_inputs" in m_batch.keys():
                        for key in m_batch["multi_modal_inputs"][0].keys():
                            multi_modal_inputs[key] = torch.cat(
                                [
                                    inputs[key]
                                    for inputs in m_batch["multi_modal_inputs"]
                                ],
                                dim=0,
                            ).cuda()

                    input_ids = m_batch["input_ids"]
                    attention_mask = m_batch["attention_mask"]
                    position_ids = m_batch["position_ids"]
                    prev_logprobs = m_batch["prev_logprobs"]
                    advantages = m_batch["advantages"]
                    ref_logprobs = None
                    if "ref_logprobs" in m_batch:
                        ref_logprobs = m_batch["ref_logprobs"]

                    loss_mask = m_batch["attention_mask"][:, -self.response_len :]
                    with self.amp_context:
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            **multi_modal_inputs,
                            use_cache=False,
                        )

                    logits = output.logits

                    logits.div_(self.cfg.algorithm.sampling_params.temperature)

                    responses = input_ids[:, -self.response_len :]
                    logits = logits[
                        :, -self.response_len - 1 : -1, :
                    ]  # (bsz, response_length, vocab_size)
                    logprobs = compute_logprobs_from_logits(
                        logits, responses, task_type=self.cfg.runner.task_type
                    )

                    clip_ratio = self.cfg.algorithm.ratio_clip_eps
                    clip_ratio_low = (
                        self.cfg.algorithm.clip_ratio_low
                        if self.cfg.algorithm.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.cfg.algorithm.clip_ratio_high
                        if self.cfg.algorithm.clip_ratio_high is not None
                        else clip_ratio
                    )
                    clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

                    if self.cfg.algorithm.get("importance_sampling_fix", False):
                        rollout_prev_logprobs = prev_logprobs
                        recompute_prev_logprobs = batch["recompute_prev_logprobs"]
                        advantages = advantages * torch.clamp(
                            (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                            min=self.cfg.algorithm.importance_sampling_clip,
                        )

                    loss, mbs_metrics_data = policy_loss(
                        loss_type=self.cfg.algorithm.loss_type,
                        loss_agg_func=self.loss_agg_func,
                        logprobs=logprobs,
                        old_logprobs=prev_logprobs,
                        advantages=advantages,
                        clip_ratio_low=clip_ratio_low,
                        clip_ratio_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_mask=loss_mask,
                        task_type=self.cfg.runner.task_type,
                    )

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if self.calculate_entropy:
                        entropy = output["entropy"][
                            :, -self.response_len - 1 : -1
                        ].contiguous()
                        entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                        if self.calculate_entropy_loss:
                            loss = (
                                loss - self.cfg.algorithm.entropy_bonus * entropy_loss
                            )

                    kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if self.kl_beta > 0 and ref_logprobs is not None:
                        kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                        kl_loss = self.loss_agg_func(kld, loss_mask)
                        loss = loss + kl_loss * self.kl_beta

                    # add to log
                    # scale loss for gradient accumulation and backprop
                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    mbs_metrics_data.update(
                        {
                            "final_loss": loss.detach(),
                            "entropy_loss": entropy_loss.detach(),
                            "kl_loss": kl_loss.detach(),
                        }
                    )

                    append_to_dict(metrics, mbs_metrics_data)

                grad_norm, lr_list = self.optimizer_step()

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["attention_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.cfg.runner.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self._world_size,), mesh_dim_names=["fsdp"]
        )
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.channel = self.connect_channel(cfg.actor.channel.name)
        
        # SAC-specific initialization
        self.is_sac = cfg.algorithm.loss_type == "embodied_sac"
        if self.is_sac:
            self.replay_buffer = None
            self.target_model = None
            self.target_model_initialized = False
            self.base_alpha = None
            self.demo_buffer = None
            self.alpha_optimizer = None
            self.qf_optimizer = None
            self.update_step = 0

    def init_worker(self):
        if self.is_sac:
            self.setup_model_and_optimizer(initialize_target=True)
            self.setup_sac_components()
            self.soft_update_target_model(tau=1.0)
        else:
            self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
            if self.is_sac:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        if self.cfg.actor.get("enable_offload", False):
            self.offload_optimizer()

        if next(self.model.parameters()).is_cpu:
            if self.cfg.actor.get("enable_offload", False):
                self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict()
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()

    async def recv_rollout_batch(self) -> None:
        """
        Receive rollout batch from rollout workers.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(
                await self.channel.get(
                    key=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            self.rollout_batch[key] = torch.cat(
                [recv_list[i][key] for i in range(split_num)], dim=1
            )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        
        # For SAC: add rollout batch to replay buffer
        if self.is_sac and hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            self.replay_buffer.add_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            rollout_batch[key] = new_value

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if self.rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if self.rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & self.rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def compute_advantages_and_returns(self):
        # SAC doesn't compute advantages/returns like PPO
        if self.is_sac:
            # Just compute basic rollout metrics without advantages/returns
            rollout_metrics = compute_rollout_metrics(self.rollout_batch)
            return rollout_metrics
        
        # PPO/other algorithms: compute advantages and returns
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs
            * stage_num
            * env_world_size
            // actor_world_size
        )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "num_group_envs": num_group_envs_for_train,
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        self.rollout_batch.update(
            {
                "loss_mask": kwargs["loss_mask"],
                "loss_mask_sum": kwargs["loss_mask_sum"],
            }
        )
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    # ========== SAC-specific methods ==========
    def setup_sac_components(self):
        """Initialize SAC-specific components (replay buffer, etc.)"""
        if not self.is_sac:
            return
        
        buffer_capacity = self.cfg.algorithm.get("replay_buffer_capacity", 100000)
        seed = self.cfg.actor.get("seed", 1234)
        self.replay_buffer = SACReplayBuffer(
            capacity=buffer_capacity,
            device=self.device,
            seed=seed
        )
        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator()
        self.critic_sample_generator.manual_seed(seed)

    def setup_model_and_optimizer(self, initialize_target=False):
        """Setup model and optimizer, with optional target network initialization for SAC"""
        # For SAC: build separate optimizers first, then call parent
        if self.is_sac:
            # Call parent to setup model
            super().setup_model_and_optimizer()
            # Build separate optimizers for actor and critic (Q-network)
            self.build_sac_optimizers()
            # Initialize target network
            if initialize_target:
                if not hasattr(self, 'target_model') or self.target_model is None:
                    # Create target model (copy of online model)
                    target_state_dict = self.get_model_state_dict()
                    # Note: target model should use same FSDP strategy
                    # For now, we'll keep it simple and just copy state dict when needed
                    self.target_model_initialized = False  # Will be set after first sync
        else:
            # Original behavior for PPO/other algorithms
            super().setup_model_and_optimizer()

    def build_sac_optimizers(self):
        """Build separate optimizers for actor and Q-network (critic) for SAC"""
        if not self.is_sac:
            return
        
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        params_actor = []
        params_critic = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "q_head" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)
        
        assert len(params_critic) > 0, "No Q-network parameters found!"
        
        # Create separate optimizers
        self.optimizer = torch.optim.Adam(
            [{"params": params_actor, "lr": self._cfg.optim.lr, "betas": betas}]
        )
        self.qf_optimizer = torch.optim.Adam(
            [{"params": params_critic, "lr": self._cfg.optim.value_lr, "betas": betas}]
        )
        
        # Initialize temperature parameter for automatic entropy tuning
        if self.cfg.algorithm.get("auto_entropy_tuning", False):
            target_entropy = self.cfg.algorithm.get(
                "target_entropy", -self.cfg.actor.model.action_dim
            )
            self.target_entropy = target_entropy
            self.alpha_type = "exp"  # or "softplus"
            if self.alpha_type == "exp":
                self.base_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            elif self.alpha_type == "softplus":
                self.base_alpha = torch.nn.Parameter(
                    np.log(np.exp(1)-1) * torch.ones(1, device=self.device), requires_grad=True
                )
            else:
                raise NotImplementedError(f"Alpha type {self.alpha_type} not supported")
            self.alpha_optimizer = torch.optim.Adam(
                [self.base_alpha], 
                lr=self.cfg.algorithm.get("alpha_lr", 3e-4)
            )

    def compute_alpha(self):
        """Compute current alpha (temperature) value"""
        if not self.is_sac or not self.cfg.algorithm.get("auto_entropy_tuning", False):
            return torch.tensor(self.cfg.algorithm.get("alpha", 0.2), device=self.device)
        
        if self.alpha_type == "exp":
            return self.base_alpha.exp()
        elif self.alpha_type == "softplus":
            return torch.nn.functional.softplus(self.base_alpha)
        else:
            raise NotImplementedError

    @property
    def alpha(self):
        """Get alpha value as float"""
        if not self.is_sac:
            return self.cfg.algorithm.get("alpha", 0.2)
        return self.compute_alpha().item()

    def soft_update_target_model(self, tau=None):
        """Soft update target model parameters (for SAC)"""
        if not self.is_sac:
            return
        
        if tau is None:
            tau = self.cfg.algorithm.get("tau", 0.005)
        
        # Initialize target model on first call
        if not self.target_model_initialized:
            # For FSDP, we'll sync state dict instead of maintaining separate model
            # Store target state dict instead
            if not hasattr(self, '_target_state_dict'):
                self._target_state_dict = {}
            online_state_dict = self.get_model_state_dict()
            for key, value in online_state_dict.items():
                self._target_state_dict[key] = value.clone()
            self.target_model_initialized = True
            return
        
        # Soft update: get online state dict and update target
        online_state_dict = self.get_model_state_dict()
        with torch.no_grad():
            for key in online_state_dict.keys():
                if key in self._target_state_dict:
                    self._target_state_dict[key].mul_(1.0 - tau)
                    self._target_state_dict[key].add_(online_state_dict[key] * tau)
    
    def get_target_model_state_dict(self):
        """Get target model state dict (for SAC)"""
        if not self.is_sac or not hasattr(self, '_target_state_dict'):
            return None
        return self._target_state_dict

    def run_sac_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            # self.device is already an int (from torch.cuda.current_device())
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
            # Note: qf_optimizer is a separate optimizer, not managed by FSDP
            # So it doesn't need to be loaded/offloaded separately (it's already on device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            if self._rank == 0:
                print(f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training")
            return {}

        self.model.train()
        metrics = {}
        
        # Number of gradient updates per training call
        num_updates = self.cfg.algorithm.get("num_updates_per_step", 1)
        batch_size = self.cfg.actor.global_batch_size
        
        for update_idx in range(num_updates):
            # Sample batch from replay buffer
            if self.demo_buffer is not None:
                if update_idx % 2 == 0:
                    batch = self.replay_buffer.sample(batch_size)
                else:
                    batch = self.demo_buffer.sample(batch_size)
            else:
                batch = self.replay_buffer.sample(batch_size)
            
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                elif isinstance(v, dict):
                    batch[k] = {
                        sub_k: sub_v.to(self.device) if isinstance(sub_v, torch.Tensor) else sub_v
                        for sub_k, sub_v in v.items()
                    }
            
            # Extract current and next observations from replay buffer
            # The keys in replay buffer come from forward_inputs:
            # - "obs/images/base_camera" or "obs/state" for observations
            # - "action" for actions
            # - "rewards" for rewards (from rollout_result_dict)
            curr_obs = {}
            next_obs = {}
            
            # Extract action and rewards
            # Action is stored as "action" in forward_inputs
            action = batch.get("action", None)
            # Rewards are stored separately in rollout_result_dict
            rewards = batch.get("rewards", None)
            
            # Extract observations from forward_inputs format ("obs/...")
            for key, value in batch.items():
                # Skip non-obs keys
                if key in ["action", "rewards", "dones", "prev_logprobs", "prev_values"]:
                    continue
                
                # Check for current obs (from forward_inputs: "obs/images/base_camera" or "obs/state")
                if key.startswith("obs/"):
                    # Remove "obs/" prefix to get the actual key (e.g., "images/base_camera" or "state")
                    obs_key = key[4:]  # Remove "obs/"
                    curr_obs[obs_key] = value
                    # For now, next_obs is same as curr_obs (will need proper transition construction later)
                    # In proper implementation, next_obs should come from next step's forward_inputs
                    next_obs[obs_key] = value.clone() if isinstance(value, torch.Tensor) else value
            
            # Validate that we have the required keys
            if not curr_obs:
                available_keys = list(batch.keys())
                raise ValueError(
                    f"No obs data found in replay buffer batch. "
                    f"Expected keys starting with 'obs/' (e.g., 'obs/images/base_camera', 'obs/state'). "
                    f"Available keys: {available_keys}"
                )
            
            # Ensure we have action and rewards
            if action is None:
                available_keys = list(batch.keys())
                raise ValueError(
                    f"No 'action' key found in replay buffer batch. "
                    f"Available keys: {available_keys}"
                )
            if rewards is None:
                available_keys = list(batch.keys())
                raise ValueError(
                    f"No 'rewards' key found in replay buffer batch. "
                    f"Available keys: {available_keys}"
                )
            

            # Compute target Q-values
            with torch.no_grad():
                # Sample next actions using online model
                pi_next, log_pi_next, shared_feature_next = self.model(
                    forward_type="sac_forward", obs=next_obs
                )
                next_state_log_pi = log_pi_next.sum(dim=-1, keepdim=True)
                
                # Compute Q-values using target model state dict
                # For FSDP, we use a simplified approach: compute Q with current model
                # and apply soft update periodically (target Q will stabilize over time)
                all_qf_next_target = self.model(
                    forward_type="sac_q_forward", 
                    obs=next_obs, actions=pi_next, shared_feature=shared_feature_next
                )
                
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0, all_qf_next_target.shape[0], self.critic_subsample_size, 
                        generator=self.critic_sample_generator, device=self.device
                    )
                    all_qf_next_target = all_qf_next_target[sample_idx]
                    
                min_qf_next_target, _ = torch.min(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    min_qf_next_target = min_qf_next_target - self.compute_alpha() * next_state_log_pi
                target_q_values = rewards + self.cfg.algorithm.gamma * min_qf_next_target
            
            # Update Q-network (critic)
            all_data_q_values = self.model(
                forward_type="sac_q_forward", obs=curr_obs, actions=action
            )
            critic_loss = F.mse_loss(all_data_q_values, target_q_values) * all_data_q_values.shape[0]
            self.qf_optimizer.zero_grad()
            critic_loss.backward()
            self.qf_optimizer.step()

            # Update actor (policy) and temperature
            actor_loss = torch.tensor(0.0, device=self.device)
            min_qf_pi = torch.tensor(0.0, device=self.device)
            if update_idx % self.critic_actor_ratio == 0:
                pi, log_pi, shared_feature = self.model(
                    forward_type="sac_forward", obs=curr_obs
                )
                log_pi = log_pi.sum(dim=-1, keepdim=True)
                all_qf_pi = self.model(
                    forward_type="sac_q_forward", 
                    obs=curr_obs, actions=pi, 
                    shared_feature=shared_feature, 
                    detach_encoder=True
                )
                min_qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
                actor_loss = ((self.compute_alpha() * log_pi) - min_qf_pi).mean()
                
                self.optimizer.zero_grad()
                actor_loss.backward()
                self.optimizer.step()
                
                # Update temperature parameter if using automatic entropy tuning
                if hasattr(self, 'base_alpha') and self.base_alpha is not None:
                    with torch.no_grad():
                        _, log_pi_new, _ = self.model(forward_type="sac_forward", obs=curr_obs)
                        log_pi_new = log_pi_new.sum(dim=-1, keepdim=True)

                    alpha_loss = (-self.compute_alpha() * (log_pi_new.mean() + self.target_entropy))
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    torch.distributed.all_reduce(self.base_alpha.grad, op=torch.distributed.ReduceOp.AVG)
                    self.alpha_optimizer.step()
            
            # Soft update target network
            if self.target_model_initialized and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0:
                self.soft_update_target_model()
                
            loss = actor_loss + critic_loss
            
            # Collect metrics
            metrics_data = {
                "sac/total_loss": loss.detach().item(),
                "sac/alpha": self.alpha, 
                "actor/lr": self.optimizer.param_groups[0]["lr"],
                "critic/lr": self.qf_optimizer.param_groups[0]["lr"], 
                "sac/actor_loss": actor_loss.detach().item(), 
                "sac/critic_loss": critic_loss.detach().item(), 
                "sac/qf_values": all_data_q_values.mean().detach().item(), 
                "sac/current_q": min_qf_pi.mean().detach().item(), 
                "replay_buffer/size": len(self.replay_buffer),
                "replay_buffer/utilization": len(self.replay_buffer) / self.replay_buffer.capacity
            }
            
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value
        
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict
    # ========== End SAC-specific methods ==========

    def run_training(self):
        # Use SAC training if configured
        if self.is_sac:
            return self.run_sac_training()
        
        # Original PPO training
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            for key, value in self.rollout_batch.items():
                if key in ["dones", "prev_values"]:
                    value = value[:-1]
                if "env_info" in key:
                    continue
                if value is None:
                    continue
                value = value.reshape(rollout_size, *value.shape[2:])
                self.rollout_batch[key] = value[shuffle_id]

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

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )
                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    for k, v in data.items():
                        data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = data["advantages"]
                    prev_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    if self.cfg.actor.model.model_name in ["openvla", "openvla_oft"]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            data=data,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if self.cfg.actor.model.model_name in ["gr00t"]:
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step):
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
