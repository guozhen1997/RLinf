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
from contextlib import nullcontext
from typing import ContextManager, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecision,
    StateDictType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp import FSDP
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    get_backward_prefetch_strategy,
    get_fsdp_wrap_policy,
    get_sharding_strategy,
    init_fn,
)
from rlinf.utils.utils import clear_memory


class FSDP1Strategy(FSDPStrategyBase):
    def __init__(self, cfg: DictConfig, world_size: int, rank: int, logger=None):
        super().__init__(cfg, world_size, rank, logger)

    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> FSDP:
        """
        Wrap the model with FSDP using the specified configuration,
        it will apply mixed precision, sharding strategy, and wrapping policy.

        Args:
            - model (nn.Module): The model to be wrapped.
            - device_mesh (DeviceMesh): The device mesh for distributed training.

        Returns:
            - FSDP: The wrapped FSDP model.
        """
        mixed_precision_config = self.cfg.fsdp_config.mixed_precision
        param_dtype = torch_dtype_from_precision(mixed_precision_config.param_dtype)
        reduce_dtype = torch_dtype_from_precision(mixed_precision_config.reduce_dtype)
        buffer_dtype = torch_dtype_from_precision(mixed_precision_config.buffer_dtype)
        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        sharding_strategy = get_sharding_strategy(
            self.cfg.fsdp_config.sharding_strategy
        )

        is_vla_model = (
            True
            if self.cfg.model.get("model_name", None) in ["openvla", "openvla_oft"]
            else False
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=model,
            config=None,
            is_lora=self.cfg.model.is_lora,
            is_vla_model=is_vla_model,
        )

        backward_prefetch = get_backward_prefetch_strategy(
            self.cfg.fsdp_config.backward_prefetch
        )

        fsdp_model = FSDP(
            module=model,
            param_init_fn=init_fn,
            auto_wrap_policy=auto_wrap_policy,
            device_id=int(os.environ["LOCAL_RANK"]),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=device_mesh,
            forward_prefetch=self.cfg.fsdp_config.forward_prefetch,
            backward_prefetch=backward_prefetch,
            limit_all_gathers=self.cfg.fsdp_config.limit_all_gathers,
            use_orig_params=self.cfg.fsdp_config.use_orig_params,
        )
        return fsdp_model

    def save_checkpoint(
        self,
        model: FSDP,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        save_path: str,
    ) -> None:
        """
        Save model, optimizer and lr_scheduler(if exists) state dicts to the specified path.
        Currently, saved state_dicts' filenames are hardcoded as 'model.pt', 'optimizer.pt', 'lr_scheduler.pt'.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - optimizer (Optimizer): The optimizer used for training.
            - lr_scheduler (Optional[LRScheduler]): The learning rate scheduler used for training.
            - save_path (str): The directory path to save the checkpoint files.
        """
        torch.distributed.barrier()
        model_state = self.get_model_state_dict(model=model)
        optim_state = self.get_optimizer_state_dict(model=model, optimizer=optimizer)

        if self.rank == 0:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model_state, os.path.join(save_path, "model.pt"))
            torch.save(optim_state, os.path.join(save_path, "optimizer.pt"))
            torch.save(
                lr_scheduler.state_dict(),
                os.path.join(save_path, "lr_scheduler.pt"),
            )
        torch.distributed.barrier()

    def load_checkpoint(
        self,
        model: FSDP,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        load_path: str,
    ) -> None:
        """
        Load model, optimizer and lr_scheduler(if exists) state dicts from the specified path.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - optimizer (Optimizer): The optimizer used for training.
            - save_path (str): The directory path to load the checkpoint files from.
            - lr_scheduler (Optional[LRScheduler]): The learning rate scheduler used for training.

        Raises:
            - FileNotFoundError: If the checkpoint files are not found in the specified path.
            - RuntimeError: If there is a mismatch in the state dict keys when loading the model
        """
        torch.distributed.barrier()

        model_path = os.path.join(load_path, "model.pt")
        optim_path = os.path.join(load_path, "optimizer.pt")
        if not (os.path.exists(model_path) and os.path.exists(optim_path)):
            raise FileNotFoundError(f"Missing checkpoint files in {load_path}")

        assert torch.cuda.is_available(), (
            "CUDA is not available for loading checkpoint."
        )
        local_device = torch.device(f"cuda:{self.rank}")
        model_state = torch.load(model_path, map_location=local_device)
        optim_full_state = torch.load(optim_path, map_location=local_device)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            missing, unexpected = model.load_state_dict(model_state, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"State dict mismatch. missing={missing}, unexpected={unexpected}"
                )

        sharded_osd = FSDP.shard_full_optim_state_dict(optim_full_state, model)
        optimizer.load_state_dict(sharded_osd)

        sched_path = os.path.join(load_path, "lr_scheduler.pt")
        if os.path.exists(sched_path):
            lr_scheduler.load_state_dict(torch.load(sched_path))
        else:
            raise FileNotFoundError(f"lr_scheduler.pt not found in {load_path}")
        torch.distributed.barrier()

    def get_model_state_dict(self, model: FSDP) -> Dict:
        """
        Get the full state dict of the FSDP wrapped model.

        Args:
            - model (FSDP): The FSDP wrapped model.

        Returns:
            Dict: The full state dict of the FSDP wrapped model.
        """
        with FSDP.state_dict_type(
            module=model, state_dict_type=StateDictType.FULL_STATE_DICT
        ):
            state_dict = model.state_dict()
        return state_dict

    def get_optimizer_state_dict(self, model: FSDP, optimizer: Optimizer) -> Dict:
        """
        Get the full state dict of the optimizer.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - optimizer (Optimizer): The optimizer used for training.

        Returns:
            Dict: The full state dict of the optimizer.
        """
        with FSDP.state_dict_type(
            module=model, state_dict_type=StateDictType.FULL_STATE_DICT
        ):
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
        return optimizer_state_dict

    @torch.no_grad()
    def offload_param_and_grad(self, model: FSDP, offload_grad: bool) -> None:
        """
        Offload model parameters and gradients to CPU.
        Args:
            - model (FSDP): The FSDP wrapped model.
            - offload_grad (bool): Whether to offload gradients or not.
        """
        for _, param in model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        "cpu", non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to("cpu", non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to("cpu", non_blocking=True)

            if param.data is not None:
                param.data = param.data.to("cpu", non_blocking=True)

            if offload_grad and param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        clear_memory()

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: FSDP, device: torch.device, onload_grad: bool
    ) -> None:
        """
        Load model parameters and gradients to the specified device.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - device (torch.device): The device to load the parameters and gradients to.
            - onload_grad (bool): Whether to load gradients or not.

        """
        for _, param in model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        device, non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to(device, non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to(device, non_blocking=True)

            if param.data is not None:
                param.data = param.data.to(device, non_blocking=True)

            if onload_grad and param.grad is not None:
                param.grad = param.grad.to(device, non_blocking=True)
        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        """
        Offload optimizer state to CPU.

        Args:
            - optimizer (Optimizer): The optimizer used for training.
        """
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        """
        Load optimizer state to the specified device.

        Args:
            - optimizer (Optimizer): The optimizer used for training.
            - device (torch.device): The device to load the optimizer state to.
        """
        if not optimizer.state:
            return
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)
        clear_memory()

    def optimizer_step(
        self,
        model: FSDP,
        optimizer: Optimizer,
        grad_scaler: GradScaler,
        lr_scheduler: LRScheduler,
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[float, float]:
        """
        Perform an optimizer step with gradient scaling.

        Args:
            - optimizer (Optimizer): The optimizer used for training.
            - grad_scaler (GradScaler): The gradient scaler for mixed precision training.

        Returns:
            - tuple[int, float, float]: A tuple containing grad_norm and lr.
        """

        assert self.cfg.optim.clip_grad is not None, (
            "Gradient clipping value must be set."
        )

        grad_scaler.unscale_(optimizer)
        grad_norm = model.clip_grad_norm_(self.cfg.optim.clip_grad)

        is_finite = torch.isfinite(torch.as_tensor(grad_norm))

        if not is_finite:
            self.logger.warning("[FSDP1] Grad norm is not finite, skip optimizer step.")
        else:
            grad_scaler.step(optimizer)
            lr_scheduler.step()

        grad_scaler.update()
        lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else 0.0

        grad_norm_value = (
            float(grad_norm.detach().item())
            if torch.is_tensor(grad_norm)
            else float(grad_norm)
        )

        return grad_norm_value, lr

    def before_micro_batch(
        self, model: FSDP, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Context manager for handling gradient synchronization during micro-batches for FSDP.
        it will disable gradient synchronization for non-last micro-batches to reduce all-reduce count.

        Args:
            - model (FSDP): The FSDP wrapped model.
            - is_last_micro_batch (bool): Whether the current micro-batch is the last one

        Returns:
            - ContextManager: The context manager for gradient synchronization.
        """
        return model.no_sync() if not is_last_micro_batch else nullcontext()
