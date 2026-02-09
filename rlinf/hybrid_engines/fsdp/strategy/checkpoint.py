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

from collections.abc import Iterable
from typing import Union

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rlinf.hybrid_engines.fsdp import FSDP, FSDPModule
from rlinf.hybrid_engines.fsdp.utils import FSDPVersion
from rlinf.utils.utils import get_rng_state, set_rng_state


class Checkpoint(Stateful):
    def __init__(
        self,
        model: Union[FSDP, FSDPModule],
        optimizers: Union[Optimizer, Iterable[Optimizer]],
        lr_schedulers: Union[LRScheduler, Iterable[LRScheduler]],
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
    ):
        self.model = model
        self.optimizers = (
            (optimizers,) if isinstance(optimizers, Optimizer) else tuple(optimizers)
        )
        self.lr_schedulers = (
            (lr_schedulers,)
            if isinstance(lr_schedulers, LRScheduler)
            else tuple(lr_schedulers)
        )
        self.opts = opts
        self.fsdp_version = fsdp_version

    def _uses_orig_params(self) -> bool:
        all_handles = getattr(self.model, "_all_handles", None)
        if not all_handles:
            return False
        return any(getattr(handle, "_use_orig_params", False) for handle in all_handles)

    def state_dict(self):
        use_raw_optim = self._uses_orig_params() and len(self.optimizers) > 1
        if use_raw_optim:
            model_sd = get_model_state_dict(model=self.model, options=self.opts)
            optim_sd = [optim.state_dict() for optim in self.optimizers]
        else:
            model_sd, optim_sd = get_state_dict(
                model=self.model, optimizers=self.optimizers, options=self.opts
            )
        lr_sched_sd = []
        for lr_sched in self.lr_schedulers:
            lr_sched_sd.append(lr_sched.state_dict())

        out = {
            "model": model_sd,
            "lr_schedulers": lr_sched_sd,
            "fsdp_version": self.fsdp_version.value,
        }
        if use_raw_optim:
            out["optimizers_raw"] = optim_sd
        else:
            out["optimizers"] = optim_sd
        out["rng"] = get_rng_state()
        return out

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: checkpoint version {ckpt_fsdp_version} != current version {self.fsdp_version}"
            )
        if "optimizers_raw" in state:
            set_model_state_dict(
                model=self.model,
                model_state_dict=state["model"],
                options=self.opts,
            )
            for optim, optim_sd in zip(self.optimizers, state["optimizers_raw"]):
                optim.load_state_dict(optim_sd)
        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizers,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizers"]
                if "optimizers" in state
                else state["optim"],
                options=self.opts,
            )
        if self.lr_schedulers is not None:
            if "lr_schedulers" in state:
                for lr_sched, lr_sched_sd in zip(
                    self.lr_schedulers, state["lr_schedulers"]
                ):
                    lr_sched.load_state_dict(lr_sched_sd)
            elif "lr_scheduler" in state:
                lr_sched_sd = [state["lr_scheduler"]]
                for lr_sched, lr_sched_sd in zip(self.lr_schedulers, lr_sched_sd):
                    lr_sched.load_state_dict(lr_sched_sd)
        if "rng" in state:
            set_rng_state(state["rng"])
