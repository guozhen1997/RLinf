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

import random
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.distributed.device_mesh import DeviceMesh
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler


class FSDPStrategyBase(ABC):
    def __init__(self, cfg: DictConfig, world_size: int, rank: int, logger=None):
        self.cfg = cfg
        self._logger = logger
        self.world_size = world_size
        self.rank = rank

    @property
    def logger(self):
        if self._logger is None:
            import logging

            self._logger = logging.getLogger(__name__)
        return self._logger

    @abstractmethod
    def wrap_model(
        self, model: nn.Module, device_mesh: DeviceMesh
    ) -> Union[FSDP, FSDPModule]:
        """
        Wrap the model with FSDP or FSDPModule based on the strategy.

        Args:
            model (nn.Module): The model to be wrapped.

        Returns:
            Union[FSDP, FSDPModule]: The wrapped model.
        """
        raise NotImplementedError(
            "_wrap_model method must be implemented by subclasses."
        )

    def build_optimizer(self, model: Union[nn.Module, FSDPModule, FSDP]) -> Optimizer:
        """
        Build the optimizer based on the configuration, currently only support Adam optimizer.

        Args:
            model: The model to optimize, can be nn.Module, FSDPModule (used in FSDP2) or FSDP.

        Returns:
            Optimizer: The constructed optimizer.
        """
        betas = (self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2)

        params_actor = []
        params_critic = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "value_head" in name or "model.value_head" in name:
                    params_critic.append(param)
                else:
                    params_actor.append(param)

        if len(params_critic) > 0:
            return torch.optim.AdamW(
                [
                    {"params": params_actor, "lr": self.cfg.optim.lr, "betas": betas},
                    {
                        "params": params_critic,
                        "lr": self.cfg.optim.value_lr,
                        "betas": betas,
                    },
                ]
            )
        else:
            return torch.optim.AdamW(
                [
                    {
                        "params": params_actor,
                        "lr": self.cfg.optim.lr,
                        "betas": betas,
                    },
                ]
            )

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """
        Build the learning rate scheduler based on the configuration.
        Currently only support LambdaLR scheduler with various warmup styles.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.

        Returns:
            LRScheduler: The learning rate scheduler.
        """
        total_steps = self.cfg.optim.get("total_training_steps", 0)
        num_warmup_steps = int(self.cfg.optim.get("lr_warmup_steps", -1))
        warmup_style = self.cfg.optim.get("warmup_style", "constant")
        min_lr_ratio = self.cfg.optim.get("min_lr_ratio", 0.0)
        num_cycles = self.cfg.optim.get("num_cycles", 0.5)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = self.cfg.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        return get_lr_scheduler(
            warmup_style=warmup_style,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
            num_cycles=num_cycles,
        )

    def build_grad_scaler(self) -> GradScaler:
        """
        Build the gradient scaler for mixed precision training if model uses fp16.

        Returns:
            GradScaler: The gradient scaler.
        """
        use_fp16 = self.cfg.model.precision == "fp16"
        return GradScaler(enabled=use_fp16)

    @classmethod
    def create(
        cls, cfg: DictConfig, world_size: int, rank: int, logger=None
    ) -> "FSDPStrategyBase":
        """
        Factory method: create and return a concrete FSDP strategy instance based on cfg.

        Selection rules (case-insensitive):
        - fsdp / fsdp1 -> FSDP1Strategy (classic torch.distributed.fsdp)
        - fsdp2        -> FSDP2Strategy (fully_shard API)

        Args:
            cfg: DictConfig that must contain fsdp_config.strategy
            world_size: actor distributed world size
            rank: current process's distributed rank
            logger: optional logger, if none, a default logger will be created

        Returns:
            An instance of a subclass of FSDPStrategyBase.
        """
        assert hasattr(cfg, "fsdp_config"), (
            "fsdp_config is required for creating corresponding FSDP strategy"
        )
        strategy = str(cfg.fsdp_config.get("strategy", "fsdp2")).lower()

        if strategy in ("fsdp", "fsdp1"):
            from .fsdp1 import FSDP1Strategy

            return FSDP1Strategy(
                cfg=cfg, world_size=world_size, rank=rank, logger=logger
            )
        elif strategy == "fsdp2":
            from .fsdp2 import FSDP2Strategy

            return FSDP2Strategy(
                cfg=cfg, world_size=world_size, rank=rank, logger=logger
            )
        else:
            raise ValueError(
                f"Unknown FSDP strategy '{strategy}'. Expected one of: 'fsdp', 'fsdp1', 'fsdp2'."
            )

    def load_rng_state(self, rng_state: Dict) -> None:
        """
        Load the RNG state from the provided state dictionary.

        Args:
            rng_state (Dict): The RNG state dictionary containing states for 'cpu', 'cuda', 'rng_states', and 'amp'.
        """
        NEEDED_KEYS = ["cpu", "cuda", "rng_states", "amp"]
        assert set(NEEDED_KEYS).issubset(set(rng_state.keys())), (
            f"rng_state must contain the keys: {NEEDED_KEYS}"
        )

        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])

    def save_rng_state(self) -> Dict:
        """
        Save the current RNG state into a dictionary.

        Returns:
            Dict: The RNG state dictionary containing states for 'cpu', 'cuda', 'rng_states', and 'amp'.
        """
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        return rng_state

    @abstractmethod
    def save_checkpoint(
        self,
        model: Union[FSDP, FSDPModule],
        optimizer: Optimizer,
        checkpoint_path: str,
        rank: int,
    ) -> None:
        raise NotImplementedError(
            "save_checkpoint method must be implemented by subclasses."
        )

    @abstractmethod
    def load_checkpoint(
        self, model: Union[FSDP, FSDPModule], optimizer: Optimizer, checkpoint_path: str
    ) -> None:
        raise NotImplementedError(
            "load_checkpoint method must be implemented by subclasses."
        )

    @abstractmethod
    def get_model_state_dict(self, model: Union[FSDP, FSDPModule]) -> dict:
        raise NotImplementedError(
            "state_dict method must be implemented by subclasses."
        )

    @abstractmethod
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        raise NotImplementedError(
            "offload_optimizer method must be implemented by subclasses."
        )

    @abstractmethod
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        raise NotImplementedError(
            "onload_optimizer method must be implemented by subclasses."
        )

    @abstractmethod
    def offload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], offload_grad: bool
    ) -> None:
        raise NotImplementedError(
            "offload_param method must be implemented by subclasses."
        )

    @abstractmethod
    def onload_param_and_grad(
        self, model: Union[FSDP, FSDPModule], device: torch.device, onload_grad: bool
    ) -> None:
        raise NotImplementedError(
            "onload_param method must be implemented by subclasses."
        )
