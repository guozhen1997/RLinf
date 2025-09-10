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
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from megatron.core import parallel_state

#############################
# tp reshard  implementation
#############################


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
        merge_factor: int,
        strict: bool = True,
    ):
        self.merge_factor = merge_factor
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

    def apply(
        self, model_state_dict: Dict, sub_tp_group: torch.distributed.ProcessGroup
    ) -> Dict:
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
                    tensor=v,
                    dim=rule.dim,
                    merge_factor=self.merge_factor,
                    sub_tp_group=sub_tp_group,
                )
            elif rule.action == TensorParallelReshardType.SUM:
                # may be strange but used in some cases
                gathered_tensors = [
                    torch.zeros_like(v) for _ in range(self.merge_factor)
                ]
                torch.distributed.all_gather(gathered_tensors, v, group=sub_tp_group)
                new_state_dict[k] = sum(gathered_tensors)
            else:
                raise ValueError(f"Unknown action {rule.action} for key: {k}")

        return new_state_dict

    @staticmethod
    def _gather_tp_group_tensor_and_reshard(tensor, dim, merge_factor, sub_tp_group):
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(merge_factor)]

        torch.distributed.all_gather(gathered_tensors, tensor, group=sub_tp_group)

        resharded_tensor = torch.cat(gathered_tensors, dim=dim)

        return resharded_tensor


class Qwen2_5_TP_Resharder(BaseTensorParallelResharder):
    def build_rules(self) -> List[TensorParallelReshardRule]:
        LID = r"(?P<i>\d+)"
        WB = r"(?P<wb>weight|bias)"
        rules = [
            # embedding
            TensorParallelReshardRule(
                pattern=re.compile(r"embedding\.word_embeddings\.weight"),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # attn layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # attn o project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_proj\.{WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
                predicate=None,
            ),
            # attn qkv project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"decoder\.layers\.{LID}\.self_attention\.linear_qkv\.{WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
                predicate=None,
            ),
            # mlp fc1 project
            TensorParallelReshardRule(
                pattern=re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc1\.{WB}"),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
                predicate=None,
            ),
            # mlp fc2 project
            TensorParallelReshardRule(
                pattern=re.compile(rf"decoder\.layers\.{LID}\.mlp\.linear_fc2\.{WB}"),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
                predicate=None,
            ),
            # mlp final layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"decoder\.layers\.{LID}\.linear_fc1\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # final_layernorm
            TensorParallelReshardRule(
                pattern=re.compile(rf"decoder\.final_layernorm\.{WB}"),
                action=TensorParallelReshardType.KEEP,
            ),
            # output layer
            TensorParallelReshardRule(
                pattern=re.compile(r"output_layer\.weight"),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
                predicate=None,
            ),
        ]
        return rules


class Qwen2_5_VL_TP_Resharder(BaseTensorParallelResharder):
    LID = r"(?P<i>\d+)"
    WB = r"(?P<wb>weight|bias)"
    VISION_PREFIX = "vision_model"
    VISION_DECODER_LAYERS_PREFIX = f"{VISION_PREFIX}.decoder.layers"
    LLM_PREFIX = "language_model"
    LLM_DECODER_LAYERS_PREFIX = f"{LLM_PREFIX}.decoder.layers"

    def _build_vision_rules(self) -> List[TensorParallelReshardRule]:
        vision_rules = [
            # vision embedding
            TensorParallelReshardRule(
                pattern=re.compile(rf"{self.VISION_PREFIX}\.patch_embed\.proj\.weight"),
                action=TensorParallelReshardType.KEEP,
            ),
            # vision o proj weight
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_proj\.weight"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
            ),
            # vision o proj bias
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_proj\.bias"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # vision qkv proj weight/bias
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_qkv\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # vision attn layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_qkv\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # vision mlp fc1 project weight/bias
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.mlp\.linear_fc1\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # vision mlp fc2 project weight
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.mlp\.linear_fc2\.weight"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
            ),
            # vision mlp fc2 project bias
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.mlp\.linear_fc2\.bias"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # vision mlp final layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_DECODER_LAYERS_PREFIX}\.{self.LID}\.linear_fc1\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # vision final_layernorm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_PREFIX}\.decoder\.final_layernorm\.{self.WB}"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
        ]
        return vision_rules

    def _build_llm_rules(self) -> List[TensorParallelReshardRule]:
        llm_rules = [
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_PREFIX}\.embedding\.word_embeddings\.weight"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # llm attn layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_qkv\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # llm attn o project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_proj\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
            ),
            # llm attn qkv project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.self_attention\.linear_qkv\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # llm mlp fc1 project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.mlp\.linear_fc1\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            # llm mlp fc2 project
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.mlp\.linear_fc2\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
            ),
            # llm fc1 layer norm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_DECODER_LAYERS_PREFIX}\.{self.LID}\.linear_fc1\.layer_norm_weight"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
            # llm final_layernorm
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.LLM_PREFIX}\.decoder\.final_layernorm\.{self.WB}"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
        ]
        return llm_rules

    def _build_projection_rules(self) -> List[TensorParallelReshardRule]:
        projection_rules = [
            # projection linear_fc1 weight/bias
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_PREFIX}\.projection\.encoder\.linear_fc1\.{self.WB}"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=0,
            ),
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_PREFIX}\.projection\.encoder\.inear_fc2\.weight"
                ),
                action=TensorParallelReshardType.CONCAT,
                dim=1,
            ),
            TensorParallelReshardRule(
                pattern=re.compile(
                    rf"{self.VISION_PREFIX}\.projection\.encoder\.linear_fc2\.bias"
                ),
                action=TensorParallelReshardType.KEEP,
            ),
        ]
        return projection_rules

    def build_rules(self):
        rules = (
            self._build_vision_rules()
            + self._build_projection_rules()
            + self._build_llm_rules()
        )
        return rules


_TP_RESHARDER_REGISTRY: Dict[str, Type[BaseTensorParallelResharder]] = {}


def register_mg2hf_tp_resharder(
    model_arch: str, cls: Type[BaseTensorParallelResharder]
):
    if model_arch in _TP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} already registered in mg2hf tp resharder registry"
        )
    _TP_RESHARDER_REGISTRY[model_arch] = cls
    return cls


register_mg2hf_tp_resharder("qwen2.5", Qwen2_5_TP_Resharder)
register_mg2hf_tp_resharder("qwen2.5-vl", Qwen2_5_VL_TP_Resharder)


def get_tp_resharder(
    model_arch: str,
    merge_factor: int,
    strict: bool = True,
) -> BaseTensorParallelResharder:
    if model_arch not in _TP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} not registered in mg2hf tp resharder registry"
        )
    cls = _TP_RESHARDER_REGISTRY[model_arch]
    return cls(merge_factor=merge_factor, strict=strict)


##############################
# pp reshard implementation
##############################


class PipelineParallelReshardType(Enum):
    KEEP = "keep"
    BROADCAST_FROM_STAGE = "broadcast_from_stage"


@dataclass
class PipelineParallelReshardRule:
    pattern: re.Pattern
    action: PipelineParallelReshardType
    # "first", "last", "fixed index", "callable to get index"
    src: Union[str, int, Callable[[str], int], None] = None
    # used to filter stages need to receive
    dst_predicate: Optional[Callable[[str], bool]] = None
    # used to filter whether apply this rule
    predicate: Optional[Callable[[str], bool]] = None


class BasePipelineParallelResharder:
    def __init__(
        self,
        pp_group: torch.distributed.ProcessGroup,
        dtype: torch.dtype,
        strict: bool = True,
    ):
        self.pp_group = pp_group
        self.strict = strict
        self.dtype = dtype
        self.first_rank = parallel_state.get_pipeline_model_parallel_first_rank()
        self.last_rank = parallel_state.get_pipeline_model_parallel_last_rank()
        self.pp_rank = torch.distributed.get_rank(group=pp_group)
        self.pp_world_size = torch.distributed.get_world_size(group=pp_group)

        self.rules: List[PipelineParallelReshardRule] = self.build_rules()

    def build_rules(self) -> List[PipelineParallelReshardRule]:
        raise NotImplementedError("Subclasses must implement build_rules method")

    def _match_rules(self, key: str) -> Optional[PipelineParallelReshardRule]:
        for rule in self.rules:
            m = rule.pattern.match(key)
            if m and (rule.predicate is None or rule.predicate(key)):
                return rule
        return None

    @staticmethod
    def _gather_pp_group_tensor_and_reshard(
        model_state_dict: Dict,
        key: str,
        pp_src_rank: int,
        pp_group: torch.distributed.ProcessGroup,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        tensor = model_state_dict.get(key)
        if tensor is not None:
            tensor_shape = [tensor.shape]
        else:
            tensor_shape = [None]

        torch.distributed.broadcast_object_list(
            tensor_shape, src=pp_src_rank, group=pp_group
        )

        if tensor_shape[0] is None:
            return None
        if torch.distributed.get_rank() != pp_src_rank:
            tensor = torch.empty(tensor_shape[0], dtype=dtype).cuda()

        torch.distributed.broadcast(
            tensor.contiguous(), src=pp_src_rank, group=pp_group
        )
        return tensor

    def _resolve_src_rank(self, rule: PipelineParallelReshardRule, key: str) -> int:
        if rule.src == "first":
            return self.first_rank
        elif rule.src == "last":
            return self.last_rank
        elif isinstance(rule.src, int):
            if rule.src < 0 or rule.src >= self.pp_world_size:
                raise ValueError(f"Fixed src index {rule.src} out of range")
            return self.first_rank + rule.src
        elif callable(rule.src):
            idx = rule.src(key)
            if idx < 0 or idx >= self.pp_world_size:
                raise ValueError(f"Callable src index {idx} out of range")
            return self.first_rank + idx
        else:
            raise ValueError(f"Invalid src {rule.src} in rule for key: {key}")

    def apply(self, model_state_dict: Dict) -> Dict:
        for k in model_state_dict.keys():
            rule = self._match_rules(k)
            if rule is None:
                if self.strict:
                    raise ValueError(
                        f"PPResharder set strict True but no matching rule for key: {k}"
                    )
                else:
                    continue
            if rule.action == PipelineParallelReshardType.KEEP:
                continue
            elif rule.action == PipelineParallelReshardType.BROADCAST_FROM_STAGE:
                pp_src_rank = self._resolve_src_rank(rule, k)
                tensor = self._gather_pp_group_tensor_and_reshard(
                    model_state_dict, k, pp_src_rank, self.pp_group, self.dtype
                )
                if tensor is not None:
                    model_state_dict[k] = tensor.clone()

            else:
                raise ValueError(f"Unknown action {rule.action} for key: {k}")
        return model_state_dict


class Qwen2_5_PP_Resharder(BasePipelineParallelResharder):
    def build_rules(self) -> List[PipelineParallelReshardRule]:
        return [
            PipelineParallelReshardRule(
                pattern=re.compile(r"decoder\.final_layernorm\.weight"),
                action=PipelineParallelReshardType.BROADCAST_FROM_STAGE,
                src="last",
            ),
            PipelineParallelReshardRule(
                pattern=re.compile(r"decoder\.final_layernorm\.bias"),
                action=PipelineParallelReshardType.BROADCAST_FROM_STAGE,
                src="last",
            ),
            PipelineParallelReshardRule(
                pattern=re.compile(r"embedding\.word_embeddings\.weight"),
                action=PipelineParallelReshardType.BROADCAST_FROM_STAGE,
                src="first",
            ),
            PipelineParallelReshardRule(
                pattern=re.compile(r"output_layer\.weight"),
                action=PipelineParallelReshardType.BROADCAST_FROM_STAGE,
                src="last",
            ),
            # use strict = false to skip other params, just keep
        ]


class Qwen2_5_VL_PP_Resharder(BasePipelineParallelResharder):
    def build_rules(selfw) -> List[PipelineParallelReshardRule]:
        raise NotImplementedError


_PP_RESHARDER_REGISTRY: Dict[str, Type[BasePipelineParallelResharder]] = {}


def register_pp_resharder(model_arch: str, cls: Type[BasePipelineParallelResharder]):
    if model_arch in _PP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} already registered in pp resharder registry"
        )
    _PP_RESHARDER_REGISTRY[model_arch] = cls
    return cls


def get_pp_resharder(
    model_arch: str,
    pp_group: torch.distributed.ProcessGroup,
    strict: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> BasePipelineParallelResharder:
    if model_arch not in _PP_RESHARDER_REGISTRY:
        raise ValueError(
            f"Model arch {model_arch} not registered in pp resharder registry"
        )
    cls = _PP_RESHARDER_REGISTRY[model_arch]

    return cls(pp_group=pp_group, strict=strict, dtype=dtype)


register_pp_resharder("qwen2.5", Qwen2_5_PP_Resharder)
register_pp_resharder("qwen2.5-vl", Qwen2_5_VL_PP_Resharder)
