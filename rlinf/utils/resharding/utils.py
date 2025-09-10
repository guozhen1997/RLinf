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

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Type

import torch
from megatron.core import parallel_state


def get_pp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return pp_reshard_fn_qwen2_5
    else:
        raise NotImplementedError(
            f"get_pp_reshard_fn for model_arch {model_arch} is not implemented"
        )


##############################
# tp reshard fn implementation
##############################


class TensorParallelReshardType(Enum):
    CONCAT = "concat"
    SUM = "sum"
    KEEP = "keep"


@dataclass
class TensorParallelReshardRule:
    pattern: re.Pattern
    action: TensorParallelReshardType
    dim: Optional[int] = None
    predicate: Optional[Callable[[str], bool]] = None


class BaseTensorParallelResharder:
    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        ep_group: Optional[torch.distributed.ProcessGroup] = None,
        strict: bool = True,
    ):
        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=tp_group)
        self.ep_group = ep_group
        if ep_group is not None:
            self.ep_world_size = torch.distributed.get_world_size(group=ep_group)
        # maybe here if ep > 1 should use ep_group to do things on mlp
        # if so, just pass ep_group and implement more rules for ep
        self.rules: List[TensorParallelReshardRule] = self.build_rules()
        self.strict = strict

    def build_rules(self) -> List[TensorParallelReshardRule]:
        raise NotImplementedError("Subclasses must implement build_rules method")

    def _match_rules(self, key: str) -> Optional[TensorParallelReshardRule]:
        for rule in self.rules:
            m = rule.pattern.match(key)
            if m and (rule.predicate is None or rule.predicate(key)):
                return rule
        return None

    def apply(self, model_state_dict: Dict) -> Dict:
        new_state_dict = {}
        for k, v in model_state_dict.items():
            rule = self._match_rules(k)
            if rule is None:
                if self.strict:
                    raise ValueError(
                        f"TpResharder set strict True but no matching rule for key: {k}"
                    )
                else:
                    new_state_dict[k] = v.clone()
                    continue
            if rule.action == TensorParallelReshardType.KEEP:
                new_state_dict[k] = v.clone()
            elif rule.action == TensorParallelReshardType.CONCAT:
                if rule.dim is None:
                    raise ValueError(
                        f"Dim must be specified for CONCAT action in key: {k}"
                    )
                new_state_dict[k] = self._gather_tp_group_tensor_and_reshard(
                    v, rule.dim, self.tp_world_size, self.tp_group
                )
            elif rule.action == TensorParallelReshardType.SUM:
                # may be strange but used in some cases
                gathered_tensors = [
                    torch.zeros_like(v) for _ in range(self.tp_world_size)
                ]
                torch.distributed.all_gather(gathered_tensors, v, group=self.tp_group)
                new_state_dict[k] = sum(gathered_tensors)
            else:
                raise ValueError(f"Unknown action {rule.action} for key: {k}")

        return new_state_dict

    @staticmethod
    def _gather_tp_group_tensor_and_reshard(tensor, dim, merge_factor, tp_group):
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(merge_factor)]

        torch.distributed.all_gather(gathered_tensors, tensor, group=tp_group)

        resharded_tensor = torch.cat(gathered_tensors, dim=dim)

        return resharded_tensor


class Qwen2_5_TP_Resharder(BaseTensorParallelResharder):
    def build_rules(self) -> List[TensorParallelReshardRule]:
        return NotImplementedError("Qwen2.5 TP reshard rules not implemented yet")


class Qwen2_5_VL_TP_Resharder(BaseTensorParallelResharder):
    def build_rules(self):
        return NotImplementedError("Qwen2.5 VL TP reshard rules not implemented yet")


_MG2HF_TP_RESHARDER_REGISTRY: Dict[str, Type[BaseTensorParallelResharder]] = {}


def register_mg2hf_tp_resharder(
    model_arch: str, cls: Type[BaseTensorParallelResharder]
):
    if model_arch in _MG2HF_TP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} already registered in mg2hf tp resharder registry"
        )
    _MG2HF_TP_RESHARDER_REGISTRY[model_arch] = cls
    return cls


register_mg2hf_tp_resharder("qwen2.5", Qwen2_5_TP_Resharder)
register_mg2hf_tp_resharder("qwen2.5-vl", Qwen2_5_VL_TP_Resharder)


def get_mg2hf_tp_resharder(
    model_arch: str, tp_group: torch.distributed.ProcessGroup, strict: bool = True
) -> BaseTensorParallelResharder:
    if model_arch not in _MG2HF_TP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} not registered in mg2hf tp resharder registry"
        )
    cls = _MG2HF_TP_RESHARDER_REGISTRY[model_arch]
    return cls(tp_group, strict)


##############################
# pp reshard fn implementation
##############################


def _gather_pp_group_tensor_and_reshard(
    model_state_dict, key, pp_src_idx, group, dtype
):
    tensor = model_state_dict.get(key)
    if tensor is not None:
        tensor_shape = [tensor.shape]
    else:
        tensor_shape = [None]

    torch.distributed.broadcast_object_list(tensor_shape, pp_src_idx, group=group)

    if tensor_shape[0] is None:
        return None
    if torch.distributed.get_rank() != pp_src_idx:
        tensor = torch.empty(tensor_shape[0], dtype=dtype).cuda()

    torch.distributed.broadcast(tensor.contiguous(), pp_src_idx, group=group)
    return tensor


def pp_reshard_fn_qwen2_5(model_state_dict, pp_group, dtype):
    pp_first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
    pp_last_rank = parallel_state.get_pipeline_model_parallel_last_rank()

    key = "decoder.final_layernorm.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "decoder.final_layernorm.bias"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "embedding.word_embeddings.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_first_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()

    key = "output_layer.weight"
    tensor = _gather_pp_group_tensor_and_reshard(
        model_state_dict, key, pp_last_rank, pp_group, dtype
    )
    if tensor is not None:
        model_state_dict[key] = tensor.clone()
    return model_state_dict
