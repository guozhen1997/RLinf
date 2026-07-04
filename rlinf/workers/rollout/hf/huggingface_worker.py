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

import asyncio
import copy
import gc
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.scheduler.worker.routing import (
    build_send_key,
    decoupled_build_recv_plan,
    get_batch_size,
    get_group_world_size,
    merge_batches,
    split_batch,
    split_channel_message,
)
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.only_eval = cfg.runner.get("only_eval", False)
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None
        self._rlt_expert_model_config = None
        self.rlt_feature_model = None

        self.total_num_train_envs = (
            cfg.env.train.total_num_envs if self.enable_train else 0
        )
        self.total_num_eval_envs = (
            cfg.env.eval.total_num_envs if self.enable_eval else 0
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages

        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )

        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // self.model_cfg.num_action_chunks
            if self.enable_train
            else 0
        )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            self.n_eval_chunk_steps = (
                cfg.env.eval.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            assert weight_syncer_cfg is not None, (
                "rollout.weight_syncer config must be provided"
            )
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "rollout_results": [],
            }
        self.rollout_queue_size = self.cfg.rollout.get("rollout_queue_size", 0)

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.model_cfg)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is not None:
            self.rlt_feature_model = get_model(copy.deepcopy(rlt_feature_model_config))
            self.rlt_feature_model.eval()
            self.rlt_feature_model.requires_grad_(False)

        if self.cfg.rollout.get("expert_model", None):
            if self._use_rlt_maniskill_route():
                self._rlt_expert_model_config = self._build_rlt_expert_model_config()
            else:
                expert_model_config = copy.deepcopy(self.model_cfg)
                with open_dict(expert_model_config):
                    expert_model_config.precision = (
                        self.cfg.rollout.expert_model.precision
                    )
                    expert_model_config.model_path = (
                        self.cfg.rollout.expert_model.model_path
                    )
                self.expert_model = get_model(expert_model_config)

                if self.cfg.runner.get("expert_ckpt_path", None):
                    expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                    self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # sampling parameters for rollout
        sampling_params = self.cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": True
                if sampling_params.get("temperature_eval", -1) > 0
                else False,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.algorithm_cfg.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.algorithm_cfg.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: Any,
        *,
        tag: str,
        batch_size: int,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
    ):
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=output_channel,
            data=rollout_result,
            tag=tag,
            async_op=True,
            batch_size=batch_size,
            split_fn=split_fn,
        )

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
    ):
        """Receive routed batch shards and record their return routes.

        This method is used in env-decoupled mode. It builds a receive plan for the
        source worker group, receives shard messages from ``channel`` one by one, and
        stops when all planned items are received or ``timeout_time`` is reached.

        Each received channel item must be a dict with ``batch_index`` (route
        metadata) and ``batch`` (payload shard).

        The ``batch_index`` values are stored in ``self.batch_router[tag]`` so a
        later send call can split the response and send each shard back to the original
        source rank. The received payload shards are validated, then merged with
        ``merge_fn`` or the default ``merge_batches``.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size for each planned receive entry.
            merge_fn: Optional custom function for merging received shards.
            infer_batch_size_fn: Optional function used to infer shard batch size during
                validation.
            timeout_time: Maximum time in seconds to wait before finalizing partial
                results.
            recv_queue_size: Number of receive queue entries used when building the
                receive plan.

        Returns:

            A merged payload and its split sizes. If only one shard is received, the
            current implementation returns that shard directly.
        """
        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            if not received_items:
                assert False, "received_items is empty"

            # get the tag from the received_items
            _, _, _, tag = split_channel_message(received_items[0]["batch_index"])

            assert tag in self.batch_router, (
                f"{tag=} need to be already in the batch_router"
            )
            # Save the batch_index to the batch_router.
            list_received_items = []
            for item in received_items:
                batch_index = item["batch_index"]
                received_item = item["batch"]
                list_received_items.append(received_item)
                # Save the batch_index to the batch_router.
                self.batch_router[tag].append(batch_index)
            received_items = list_received_items
            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in received_items
            ]

            if merge_fn is not None:
                return merge_fn(received_items), split_sizes
            if len(received_items) == 1:
                return received_items[0]
            return merge_batches(received_items), split_sizes

        timeout_time = timeout_time + time.time()
        get_items = None
        max_item_num = len(plan.entries)
        get_item_num = 0
        received_items = []
        while get_item_num < max_item_num:
            # get the items
            if get_items is None:
                get_items = channel.get(
                    key=plan.entries[get_item_num].key, async_op=True
                )
            else:
                # Now, the worker is getting a item, sleep to wait
                await asyncio.sleep(0.0001)

            # handle the get_items finish
            if get_items.done():
                # save the data and init the get_items to get next data
                received_items.append(await get_items.async_wait())
                get_items = None
                get_item_num = get_item_num + 1

            # handle the timeout case
            if time.time() >= timeout_time:
                max_item_num = get_item_num
                if get_items is not None:
                    received_items.append(await get_items.async_wait())
                    get_items = None
                    get_item_num = get_item_num + 1

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Send split batch results back using recorded batch routes.

        This method is used after a previous receive call has populated
        ``self.batch_router[tag]`` with batch indices from incoming messages.
        The outgoing ``data`` is split according to ``split_sizes`` and each shard is
        sent to the rank encoded in the corresponding recorded batch index.

        Each outgoing channel item is a dict with ``batch_index`` (recorded batch
        index) and ``batch`` (payload shard).

        After all shards are queued, the recorded batch indices for ``tag`` are cleared
        to avoid reusing stale routes.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag whose recorded batch indices should be consumed.
            split_fn: Optional custom splitter. If omitted, ``split_batch`` is used.
            split_sizes: Batch sizes used to split ``data``. Must have the same length
                as ``self.batch_router[tag]``.

        Returns:

            AsyncRouteWork wrapping the async channel put operations.
        """
        from rlinf.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )

        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            # After enabling env_decoupled_mode, the data sending format is as follows:
            # {
            #     "batch_index": batch_index,
            #     "batch": batch,
            # }
            # The batch_index is the index of the batch in the data.
            # The batch is the data to send.
            # batch_index: {send_rank}_{batch_idx}_{mode}_{tag}
            # The send_rank is the rank of the worker that originally sent the data.
            # The batch_idx is the index of the batch in the data.
            # The tag is the tag of the data.
            senditem = {
                "batch_index": batch_index,
                "batch": payload,
            }
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            work = channel.put(
                item=senditem,
                key=key,
                async_op=True,
            )
            works.append(work)

        # clear the batch_router for the next send
        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    async def recv_env_output(
        self,
        input_channel: Channel,
        *,
        tag: str,
        batch_size: int,
    ):
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=input_channel,
            tag=tag,
            async_op=True,
            batch_size=batch_size,
            merge_fn=self._merge_obs_batches,
            infer_batch_size_fn=self._infer_env_batch_size,
        ).async_wait()

    def update_dagger_beta(self):
        if self.expert_model is None or not hasattr(self, "_dagger_sampling_params"):
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            loss_type = self.algorithm_cfg.get("loss_type", "actor")
            if loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.model_cfg.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.algorithm_cfg.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_rollout_actions(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        final_obs: dict[str, Any] | None = None,
        env_infos: dict[str, Any] | None = None,
        allow_expert: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.rlt_feature_model is not None:
            return self._predict_with_rlt_features(
                env_obs,
                final_obs,
                mode,
                env_infos=env_infos,
                allow_expert=allow_expert,
            )
        return self.predict(env_obs, mode=mode)

    def _rlt_schedule_cfg(self):
        return self.cfg.algorithm.get("rlt_schedule", {})

    def _rlt_schedule_value(self, key: str, default):
        schedule_cfg = self._rlt_schedule_cfg()
        if schedule_cfg is not None and key in schedule_cfg:
            return schedule_cfg.get(key, default)
        return self.cfg.algorithm.get(key, default)

    def _use_rlt_schedule(self) -> bool:
        if str(self.cfg.algorithm.get("loss_type", "")) != "rlt_ac":
            return False
        schedule_cfg = self._rlt_schedule_cfg()
        if schedule_cfg is not None and "enable" in schedule_cfg:
            return bool(schedule_cfg.get("enable", False))
        schedule_keys = {
            "warmup_post_collect_updates",
            "train_every_transitions",
            "train_every_episodes",
            "max_updates_per_train_step",
        }
        return any(key in self.cfg.algorithm for key in schedule_keys)

    def _use_rlt_maniskill_route(self) -> bool:
        train_env_cfg = self.cfg.env.get("train", None)
        return (
            str(self.cfg.algorithm.get("loss_type", "")) == "rlt_ac"
            and train_env_cfg is not None
            and str(train_env_cfg.get("env_type", "")) == "maniskill"
        )

    def _build_rlt_expert_model_config(self):
        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is None:
            raise ValueError(
                "ManiSkill RLT expert requires rollout.rlt_feature_model so the "
                "expert can reuse the same OpenPI/RL-token config."
            )
        expert_model_config = copy.deepcopy(rlt_feature_model_config)
        expert_overrides = OmegaConf.to_container(
            self.cfg.rollout.expert_model,
            resolve=True,
        )
        expert_overrides = {} if expert_overrides is None else dict(expert_overrides)
        return OmegaConf.merge(expert_model_config, expert_overrides)

    def _ensure_rlt_expert_model_loaded(self):
        if self.expert_model is not None:
            return self.expert_model
        if self._rlt_expert_model_config is None:
            raise RuntimeError(
                "ManiSkill RLT expert takeover was requested, but "
                "rollout.expert_model is not configured."
            )
        self.expert_model = get_model(copy.deepcopy(self._rlt_expert_model_config))
        self.expert_model.eval()
        self.expert_model.requires_grad_(False)
        return self.expert_model

    @staticmethod
    def _flatten_action_chunk(actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() <= 2:
            return actions
        return actions.reshape(actions.shape[0], -1)

    @staticmethod
    def _policy_info_bool(
        policy_info: dict[str, Any] | None,
        key: str,
        *,
        batch_size: int,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        if policy_info is None or key not in policy_info:
            return torch.full(
                (batch_size,), bool(default), dtype=torch.bool, device=device
            )
        value = torch.as_tensor(policy_info[key], device=device)
        if value.numel() == 1:
            return torch.full(
                (batch_size,),
                bool(value.reshape(-1)[0].item()),
                dtype=torch.bool,
                device=device,
            )
        return value.reshape(batch_size, -1).to(torch.bool).any(dim=1)

    @staticmethod
    def _policy_info_float(
        policy_info: dict[str, Any] | None,
        key: str,
        *,
        batch_size: int,
        device: torch.device,
        default: float,
    ) -> torch.Tensor:
        if policy_info is None or key not in policy_info:
            return torch.full(
                (batch_size,), float(default), dtype=torch.float32, device=device
            )
        value = torch.as_tensor(policy_info[key], device=device)
        if value.numel() == 1:
            return torch.full(
                (batch_size,),
                float(value.reshape(-1)[0].item()),
                dtype=torch.float32,
                device=device,
            )
        return value.reshape(batch_size, -1)[:, -1].to(torch.float32)

    def _rlt_ready_for_online(self) -> bool:
        if not self._use_rlt_schedule():
            return True
        return int(self.version) >= int(
            self._rlt_schedule_value("warmup_post_collect_updates", 0)
        )

    def _rlt_base_actions(
        self,
        ref_chunk: torch.Tensor,
        *,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ref_chunk.dim() == 2:
            ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
        return ref_chunk[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)

    def _rlt_expert_actions(
        self,
        env_obs: dict[str, Any],
        *,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        expert_model = self._ensure_rlt_expert_model_loaded()
        with torch.no_grad():
            expert_actions, _ = expert_model.predict_action_batch(
                env_obs=env_obs,
                mode="eval",
                compute_values=False,
            )
        if isinstance(expert_actions, np.ndarray):
            expert_actions = torch.from_numpy(expert_actions)
        if expert_actions.dim() == 2:
            expert_actions = expert_actions.reshape(
                expert_actions.shape[0], -1, action_dim
            )
        return expert_actions[:, :chunk_len, :action_dim].to(
            device=device,
            dtype=dtype,
        )

    def _route_rlt_maniskill_actions(
        self,
        *,
        env_obs: dict[str, Any],
        rlt_obs: dict[str, torch.Tensor],
        student_actions: torch.Tensor,
        result: dict[str, Any],
        mode: Literal["train", "eval"],
        env_infos: dict[str, Any] | None,
        allow_expert: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions = student_actions
        batch_size, chunk_len, action_dim = actions.shape
        policy_info = None
        if isinstance(env_infos, dict) and isinstance(
            env_infos.get("policy_info"), dict
        ):
            policy_info = env_infos["policy_info"]

        ready_for_online = self._rlt_ready_for_online()
        use_actor = env_obs.get("rlt_use_actor", None)
        if use_actor is None:
            fallback_critical_phase = torch.zeros(
                (batch_size,),
                dtype=torch.bool,
                device=actions.device,
            )
        else:
            fallback_critical_phase = torch.as_tensor(
                use_actor,
                device=actions.device,
            ).bool().reshape(batch_size, -1).any(dim=1)
        in_critical_phase = self._policy_info_bool(
            policy_info,
            "in_critical_phase",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        if policy_info is None or "in_critical_phase" not in policy_info:
            in_critical_phase = fallback_critical_phase
        requested_expert_takeover = self._policy_info_bool(
            policy_info,
            "expert_takeover",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        expert_takeover = (
            requested_expert_takeover
            & ready_for_online
            & bool(allow_expert)
            & (mode == "train")
        )
        if use_actor is None:
            actor_control = (
                torch.full(
                    (batch_size,),
                    bool(ready_for_online),
                    dtype=torch.bool,
                    device=actions.device,
                )
                & in_critical_phase
            )
        else:
            actor_control = torch.as_tensor(
                use_actor, device=actions.device
            ).bool().reshape(batch_size, -1).any(dim=1)
            if self._use_rlt_schedule():
                actor_control = actor_control & torch.full(
                    (batch_size,),
                    bool(ready_for_online),
                    dtype=torch.bool,
                    device=actions.device,
                )

        base_actions = self._rlt_base_actions(
            rlt_obs["ref_chunk"],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        routed_actions = torch.where(
            actor_control[:, None, None],
            actions,
            base_actions,
        ).contiguous()

        intervention_flags = torch.zeros(
            (batch_size, chunk_len),
            dtype=torch.bool,
            device=actions.device,
        )
        if expert_takeover.any():
            expert_actions = self._rlt_expert_actions(
                env_obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                device=actions.device,
                dtype=actions.dtype,
            )
            routed_actions = torch.where(
                expert_takeover[:, None, None],
                expert_actions,
                routed_actions,
            ).contiguous()
            intervention_flags[expert_takeover] = True

        forward_inputs = result["forward_inputs"]
        forward_inputs["action"] = self._flatten_action_chunk(routed_actions).detach()
        forward_inputs["intervention_flags"] = intervention_flags
        forward_inputs["student_control"] = actor_control[:, None]
        forward_inputs["intervention_requested"] = requested_expert_takeover[:, None]
        forward_inputs["in_critical_phase"] = in_critical_phase[:, None]
        record_transition = self._policy_info_bool(
            policy_info,
            "record_transition",
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        if policy_info is None or "record_transition" not in policy_info:
            record_transition = in_critical_phase
        forward_inputs["record_transition"] = record_transition[:, None]
        forward_inputs["intervention_phase"] = self._policy_info_float(
            policy_info,
            "intervention_phase",
            batch_size=batch_size,
            device=actions.device,
            default=0.0,
        )[:, None]
        forward_inputs["ready_for_online"] = torch.full(
            (batch_size, 1),
            bool(ready_for_online),
            dtype=torch.bool,
            device=actions.device,
        )
        result["expert_label_flag"] = bool(expert_takeover.any().item())
        return routed_actions, result

    def _predict_with_rlt_features(
        self,
        env_obs: dict[str, Any],
        final_obs: dict[str, Any] | None,
        mode: Literal["train", "eval"],
        *,
        env_infos: dict[str, Any] | None = None,
        allow_expert: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            rlt_obs = self.rlt_feature_model.extract_rlt_stage2_obs(env_obs)
            actions, result = self.hf_model.predict_action_batch(
                env_obs=rlt_obs,
                mode=mode,
                return_obs=True,
            )
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)

            if self._use_rlt_maniskill_route():
                actions, result = self._route_rlt_maniskill_actions(
                    env_obs=env_obs,
                    rlt_obs=rlt_obs,
                    student_actions=actions,
                    result=result,
                    mode=mode,
                    env_infos=env_infos,
                    allow_expert=allow_expert,
                )
            else:
                use_actor = env_obs.get("rlt_use_actor", None)
                if use_actor is None and self._use_rlt_schedule():
                    use_actor = torch.full(
                        (actions.shape[0],),
                        self._rlt_ready_for_online(),
                        dtype=torch.bool,
                        device=actions.device,
                    )
                if use_actor is not None:
                    use_actor = (
                        torch.as_tensor(use_actor, device=actions.device)
                        .bool()
                        .reshape(actions.shape[0], 1, 1)
                    )
                    ref_actions = result["forward_inputs"]["ref_chunk"].to(
                        device=actions.device, dtype=actions.dtype
                    )
                    actions = torch.where(
                        use_actor,
                        actions,
                        ref_actions[:, : actions.shape[1], : actions.shape[2]],
                    ).contiguous()
                    result["forward_inputs"]["action"] = actions.reshape(
                        actions.shape[0], -1
                    ).contiguous()

            transition_obs = rlt_obs
            if final_obs is not None:
                transition_obs = self.rlt_feature_model.extract_rlt_stage2_obs(
                    final_obs
                )
            for key in ("z_rl", "proprio", "ref_chunk"):
                result["forward_inputs"][f"rlt_transition_{key}"] = transition_obs[key]

        result["expert_label_flag"] = bool(result.get("expert_label_flag", False))
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self._predict_rollout_actions(
                final_obs,
                allow_expert=False,
            )
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(
                    input_channel=input_channel,
                    tag="train_rollout_results",
                    batch_size=self.train_batch_size,
                )
                actions, result = self._predict_rollout_actions(
                    env_output["obs"],
                    final_obs=env_output.get("final_obs", None),
                    env_infos=env_output.get("env_infos", None),
                )

                save_flags = result.get("forward_inputs", {}).get(
                    "intervention_flags", None
                )
                if save_flags is None and result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.model_cfg.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    ),
                )
                self.send_rollout_result(
                    output_channel=output_channel,
                    rollout_result=rollout_result,
                    tag="train_rollout_results",
                    batch_size=self.train_batch_size,
                    split_fn=self._split_rollout_result,
                )
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(
                input_channel=input_channel,
                tag="train_rollout_results",
                batch_size=self.train_batch_size,
            )
            actions, result = self._predict_rollout_actions(
                env_output["obs"],
                final_obs=env_output.get("final_obs", None),
                env_infos=env_output.get("env_infos", None),
                allow_expert=False,
            )

            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                forward_inputs=(
                    result["forward_inputs"]
                    if self.rlt_feature_model is not None
                    else {}
                ),
            )
            self.send_rollout_result(
                output_channel=output_channel,
                rollout_result=rollout_result,
                tag="train_rollout_results",
                batch_size=self.train_batch_size,
                split_fn=self._split_rollout_result,
            )

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        if self.env_decoupled_mode:
            while True:
                (
                    env_output,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.eval_batch_size,
                    merge_fn=self._merge_obs_batches,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
                actions, _ = self._predict_rollout_actions(
                    env_output["obs"],
                    mode="eval",
                    final_obs=env_output.get("final_obs", None),
                    env_infos=env_output.get("env_infos", None),
                    allow_expert=False,
                )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().contiguous()
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="rollout_results",
                    split_sizes=split_sizes,
                )
        else:
            for _ in tqdm(
                range(self.eval_rollout_epoch),
                desc="Evaluating Rollout Epochs",
                disable=(self._rank != 0),
            ):
                for _ in range(self.n_eval_chunk_steps):
                    for _ in range(self.num_pipeline_stages):
                        env_output = await self.recv_env_output(
                            input_channel=input_channel,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )
                        actions, _ = self._predict_rollout_actions(
                            env_output["obs"],
                            mode="eval",
                            final_obs=env_output.get("final_obs", None),
                            env_infos=env_output.get("env_infos", None),
                            allow_expert=False,
                        )
                        if isinstance(actions, torch.Tensor):
                            actions = actions.detach().cpu().contiguous()
                        self.send_rollout_result(
                            output_channel=output_channel,
                            rollout_result=actions,
                            tag="eval_rollout_results",
                            batch_size=self.eval_batch_size,
                        )

            if self.enable_offload:
                self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to("cpu")
        if self.expert_model is not None:
            self.expert_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to(self.device)
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        env_infos_list = [obs_batch.get("env_infos", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_env_infos = None
        if any(env_infos is not None for env_infos in env_infos_list):
            merged_env_infos = {}
            keys = set()
            for env_infos in env_infos_list:
                if isinstance(env_infos, dict):
                    keys.update(env_infos.keys())
            for key in keys:
                values = [
                    env_infos[key]
                    for env_infos in env_infos_list
                    if isinstance(env_infos, dict) and key in env_infos
                ]
                if not values:
                    continue
                first = values[0]
                if isinstance(first, dict):
                    merged_env_infos[key] = _merge_obs_dicts(values)
                elif isinstance(first, torch.Tensor):
                    merged_env_infos[key] = torch.cat(values, dim=0)
                else:
                    merged_env_infos[key] = values

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "env_infos": merged_env_infos,
        }

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
