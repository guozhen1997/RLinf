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

from typing import Mapping, Sequence, Union

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
        optimizer: Union[Optimizer, Sequence[Optimizer], Mapping[str, Optimizer]],
        lr_scheduler: LRScheduler,
        opts: StateDictOptions,
        fsdp_version: FSDPVersion,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.opts = opts
        self.fsdp_version = fsdp_version

    @staticmethod
    def _is_multi_optimizer(
        optimizer: Union[Optimizer, Sequence[Optimizer], Mapping[str, Optimizer]],
    ) -> bool:
        return isinstance(optimizer, (Mapping, Sequence))

    def state_dict(self):
        if self._is_multi_optimizer(self.optimizer):
            model_sd = get_model_state_dict(model=self.model, options=self.opts)
            if isinstance(self.optimizer, Mapping):
                optim_sd = {
                    name: optim.state_dict() for name, optim in self.optimizer.items()
                }
            else:
                optim_sd = [optim.state_dict() for optim in self.optimizer]
            out = {
                "model": model_sd,
                "optim": optim_sd,
                "optim_format": "raw",
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "fsdp_version": self.fsdp_version.value,
            }
        else:
            model_sd, optim_sd = get_state_dict(
                model=self.model, optimizers=self.optimizer, options=self.opts
            )
            out = {
                "model": model_sd,
                "optim": optim_sd,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "fsdp_version": self.fsdp_version.value,
            }
        out["rng"] = get_rng_state()
        return out

    def load_state_dict(self, state):
        assert "fsdp_version" in state, "Checkpoint is missing FSDP version info."
        ckpt_fsdp_version = FSDPVersion(state["fsdp_version"])
        if ckpt_fsdp_version != self.fsdp_version:
            raise ValueError(
                f"FSDP version mismatch: checkpoint version {ckpt_fsdp_version} != current version {self.fsdp_version}"
            )
        if state.get("optim_format") == "raw":
            set_model_state_dict(
                model=self.model, model_state_dict=state["model"], options=self.opts
            )
            optim_state_dict = state["optim"]
            if isinstance(self.optimizer, Mapping):
                if isinstance(optim_state_dict, Mapping):
                    for name, optim in self.optimizer.items():
                        optim.load_state_dict(optim_state_dict[name])
                else:
                    names = state.get("optim_names") or list(self.optimizer.keys())
                    for name, payload in zip(names, optim_state_dict):
                        self.optimizer[name].load_state_dict(payload)
            elif isinstance(self.optimizer, Sequence):
                if isinstance(optim_state_dict, Mapping):
                    names = state.get("optim_names")
                    if names is None:
                        raise KeyError(
                            "Missing optimizer names for mapping-style checkpoint."
                        )
                    for optim, name in zip(self.optimizer, names):
                        optim.load_state_dict(optim_state_dict[name])
                else:
                    for optim, payload in zip(self.optimizer, optim_state_dict):
                        optim.load_state_dict(payload)
            else:
                if isinstance(optim_state_dict, Mapping):
                    first_key = next(iter(optim_state_dict))
                    optim_state_dict = optim_state_dict[first_key]
                self.optimizer.load_state_dict(optim_state_dict)
        else:
            optim_state_dict = state["optim"]
            optimizers = self.optimizer
            if isinstance(self.optimizer, Mapping):
                names = state.get("optim_names") or list(self.optimizer.keys())
                optimizers = [self.optimizer[name] for name in names]
                if isinstance(optim_state_dict, Mapping):
                    optim_state_dict = [optim_state_dict[name] for name in names]
            elif isinstance(self.optimizer, Sequence) and not isinstance(
                optim_state_dict, Sequence
            ):
                optim_state_dict = [optim_state_dict]
            set_state_dict(
                model=self.model,
                optimizers=optimizers,
                model_state_dict=state["model"],
                optim_state_dict=optim_state_dict,
                options=self.opts,
            )
        if self.lr_scheduler is not None and "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if "rng" in state:
            set_rng_state(state["rng"])
