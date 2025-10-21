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
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp import FSDPModule
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    apply_fsdp2_to_model,
    get_fsdp_full_state_dict,
    get_fsdp_state_ctx,
)
from rlinf.utils.utils import clear_memory


class FSDP2Strategy(FSDPStrategyBase):
    def __init__(self, cfg: DictConfig, world_size: int, rank: int, logger=None):
        super().__init__(cfg, world_size, rank, logger)

    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> FSDPModule:
        mixed_precision_config = self.cfg.fsdp_config.mixed_precision
        param_dtype = torch_dtype_from_precision(mixed_precision_config.param_dtype)
        reduce_dtype = torch_dtype_from_precision(mixed_precision_config.reduce_dtype)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=True,
        )

        offload_policy = (
            CPUOffloadPolicy(pin_memory=False)
            if self.cfg.fsdp_config.get("cpu_offload", False)
            else None
        )

        fsdp2_model = apply_fsdp2_to_model(
            module=model,
            config=self.cfg.fsdp_config,
            device_mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=self.cfg.fsdp_config.get(
                "reshard_after_forward", True
            ),
        )

        return fsdp2_model

    def save_checkpoint(
        self,
        model: FSDPModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        save_path: str,
    ) -> None:
        cuda_available = torch.cuda.is_available()
        if next(model.parameters()).is_cpu and cuda_available:
            self.get_model_state_dict(torch.cuda.current_device())
            self.get_optimizer_state_dict(torch.cuda.current_device())

        state_dict_cfg = ShardedStateDictConfig(
            offload_to_cpu=True if cuda_available else False
        )
        optim_cfg = ShardedOptimStateDictConfig(
            offload_to_cpu=True if cuda_available else False
        )
        with get_fsdp_state_ctx(
            model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            model_path = os.path.join(save_path, f"model_rank_{self.rank}.pt")
            optim_path = os.path.join(save_path, f"optim_rank_{self.rank}.pt")
            extra_path = os.path.join(
                save_path,
                f"extra_state_world_rank_{self.rank}.pt",
            )

            model_state_dict = model.state_dict()
            torch.save(model_state_dict, model_path)
            if self.rank == 0:
                self.logger.info(f"Saved model to {os.path.abspath(model_path)}")

            optimizer_state_dict = optimizer.state_dict()
            torch.save(optimizer_state_dict, optim_path)
            if self.rank == 0:
                self.logger.info(f"Saved optim to {os.path.abspath(optim_path)}")

            lr_scheduler_state_dict = (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            )
            extra_state_dict = {
                "lr_scheduler": lr_scheduler_state_dict,
                "rng": self.save_rng_state(),
            }
            torch.save(extra_state_dict, extra_path)
            if self.rank == 0:
                self.logger.info(f"Saved extra_state to {os.path.abspath(extra_path)}")

        torch.distributed.barrier()

    def load_checkpoint(
        self,
        model: FSDPModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        load_path: str,
    ) -> None:
        cuda_available = torch.cuda.is_available()
        if next(self.model.parameters()).is_cpu and cuda_available:
            self.get_model_state_dict(torch.cuda.current_device())
            self.get_optimizer_state_dict(torch.cuda.current_device())

        state_dict_cfg = ShardedStateDictConfig(
            offload_to_cpu=True if cuda_available else False
        )
        optim_cfg = ShardedOptimStateDictConfig(
            offload_to_cpu=True if cuda_available else False
        )

        with get_fsdp_state_ctx(
            model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg
        ):
            model_path = os.path.join(load_path, f"model_rank_{self.rank}.pt")
            model_state_dict = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(model_state_dict)
            if self.rank == 0:
                self.logger.info(f"Loaded model from {model_path}")

            optim_path = os.path.join(load_path, f"optim_rank_{self.rank}.pt")
            optimizer_state_dict = torch.load(optim_path, weights_only=False)
            self.optimizer.load_state_dict(optimizer_state_dict)
            if self.rank == 0:
                self.logger.info(f"Loaded optimizer from {optim_path}")

            extra_state_path = os.path.join(
                load_path,
                f"extra_state_rank_{self.rank}.pt",
            )
            extra_state_dict = torch.load(extra_state_path, weights_only=False)
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
                if self.rank == 0:
                    self.logger.info(f"Loaded rng from {extra_state_path}")

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                if self.rank == 0:
                    self.logger.info(f"Loaded lr_scheduler from {extra_state_path}")

        torch.distributed.barrier()

    def get_model_state_dict(self, model: FSDPModule) -> dict:
        state_dict = get_fsdp_full_state_dict(
            model, offload_to_cpu=True, rank0_only=False
        )
        return state_dict

    def get_optimizer_state_dict(self, optimizer: Optimizer) -> dict:
        raise NotImplementedError(
            "FSDP2Strategy does not support get_optimizer_state_dict yet."
        )

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: FSDPModule, device: torch.device, onload_grad: bool
    ) -> None:
        model.to(device=device)
        if onload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        clear_memory()

    @torch.no_grad()
    def offload_param_and_grad(self, model: FSDPModule, offload_grad: bool) -> None:
        model.to(device="cpu")

        if offload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.cpu()
        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device.type != "cpu":
                        st[k] = v.detach().to("cpu", non_blocking=True)
                        del v
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device != device:
                        st[k] = v.detach().to(device, non_blocking=True)
                        del v
        clear_memory()
