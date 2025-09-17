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


import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from megatron.core import parallel_state


def get_tp_reshard_fn(model_arch: str):
    if model_arch == "qwen2.5":
        return tp_reshard_fn_qwen2_5
    else:
        raise NotImplementedError(
            f"get_tp_reshard_fn for model_arch {model_arch} is not implemented"
        )


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


def _gather_tp_group_tensor_and_reshard(tensor, dim, merge_factor, tp_group):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(merge_factor)]

    torch.distributed.all_gather(gathered_tensors, tensor, group=tp_group)

    resharded_tensor = torch.cat(gathered_tensors, dim=dim)

    return resharded_tensor


def tp_reshard_fn_qwen2_5(model_state_dict, merge_factor, tp_group):
    for k, v in model_state_dict.items():
        if (
            "rotary_pos_emb.inv_freq" in k
            or "linear_qkv.layer_norm_weight" in k
            or "mlp.linear_fc1.layer_norm_weight" in k
            or "final_layernorm.weight" in k
        ):
            model_state_dict[k] = v.clone()
            continue

        dim = 0
        if "self_attention.linear_proj.weight" in k or "mlp.linear_fc2.weight" in k:
            dim = 1
        model_state_dict[k] = _gather_tp_group_tensor_and_reshard(
            v, dim, merge_factor, tp_group
        )
    return model_state_dict


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


################################
# hf resharder fn implementation
################################


@dataclass
class HFWeightReshardRule:
    """
    A rule for resharding weights in HuggingFace format.
    pattern: A regex pattern to match the parameter names.
    dim: The dimension to shard on. If None, the parameter is not sharded, just copied.
    """

    pattern: re.Pattern
    dim: Optional[int] = None


class BaseHFWeightResharder:
    """
    Base class for resharding weights in HuggingFace format.
    Subclasses should implement the build_rules method to define the resharding rules.
    """

    def __init__(self, rollout_tp_rank: int, reshard_tp_size: int, strict: bool = True):
        """
        Args:
            reshard_tp_size: The target tensor parallel size to reshard to.
            strict: Whether to raise an error if a parameter does not match any rule.
        """
        self.rules: List[HFWeightReshardRule] = self.build_rules()
        self.strict = strict
        self.reshard_tp_size = reshard_tp_size
        self.rollout_tp_rank = rollout_tp_rank

    def reshard(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        new_state_dict = {}
        for k, v in state_dict.items():
            matched = False
            for r in self.rules:
                if r.pattern.match(k):
                    matched = True
                    if r.dim is None:
                        new_state_dict[k] = v.clone()
                    else:
                        if v.ndim == 1:
                            shard_size = v.shape[0] // self.reshard_tp_size
                            start = self.rollout_tp_rank * shard_size
                            end = (self.rollout_tp_rank + 1) * shard_size
                            new_state_dict[k] = v[start:end].clone()
                        else:
                            shard_size = v.shape[r.dim] // self.reshard_tp_size
                            start = self.rollout_tp_rank * shard_size
                            end = (self.rollout_tp_rank + 1) * shard_size
                            new_state_dict[k] = v.narrow(
                                r.dim, start, shard_size
                            ).clone()
                    break
            if not matched and self.strict:
                raise ValueError(f"Parameter {k} does not match any resharding rule.")
            elif not matched:
                logging.warning(
                    f"Parameter {k} does not match any resharding rule, skipping."
                )
        return new_state_dict

    def build_rules(self) -> List[HFWeightReshardRule]:
        """
        Build the resharding rules for the model.
        Each rule contains a regex pattern to match the parameter names and the dimension to shard on.
        """
        raise NotImplementedError


class Qwen2_5HFRWeightResharder(BaseHFWeightResharder):
    def build_rules(self):
        rules = [
            # attention
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.self_attn\.q_proj\.(weight|bias)"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.self_attn\.k_proj\.(weight|bias)"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.self_attn\.v_proj\.(weight|bias)"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.self_attn\.o_proj\.(weight|bias)"),
                dim=1,
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.input_layernorm\.weight")
            ),
            # mlp
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.mlp\.gate_proj\.weight"), dim=0
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.mlp\.up_proj\.weight"), dim=0
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.mlp\.down_proj\.weight"), dim=1
            ),
            HFWeightReshardRule(
                re.compile(r"model\.layers\.\d+\.post_attention_layernorm\.weight")
            ),
            # embeddings
            HFWeightReshardRule(re.compile(r"model\.embed_tokens\.weight"), dim=0),
            HFWeightReshardRule(re.compile(r"lm_head\.weight"), dim=0),
            # norm
            HFWeightReshardRule(re.compile(r"model\.norm\.weight")),
        ]
        return rules


class Qwen2_5VLHFWeightResharder(BaseHFWeightResharder):
    def _build_llm_rules(self) -> List[HFWeightReshardRule]:
        llm_rules = [
            # attention
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.self_attn\.q_proj\.(weight|bias)"
                ),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.self_attn\.k_proj\.(weight|bias)"
                ),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.self_attn\.v_proj\.(weight|bias)"
                ),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.self_attn\.o_proj\.weight"
                ),
                dim=1,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.input_layernorm\.weight"
                )
            ),
            # mlp
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.mlp\.gate_proj\.weight"
                ),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"language_model\.layers\.\d+\.mlp\.up_proj\.weight"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.mlp\.down_proj\.weight"
                ),
                dim=1,
            ),
            HFWeightReshardRule(
                re.compile(
                    r"language_model\.layers\.\d+\.post_attention_layernorm\.weight"
                )
            ),
            # embedding
            HFWeightReshardRule(
                re.compile(r"language_model\.embed_tokens\.weight"), dim=0
            ),
            # norm
            HFWeightReshardRule(re.compile(r"language_model\.norm\.weight")),
        ]
        return llm_rules

    def _build_vision_rules(self) -> List[HFWeightReshardRule]:
        vision_rules = [
            # attention
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.attn\.qkv\.(weight|bias)"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.attn\.proj\.weight"), dim=1
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.attn\.proj\.bias")
            ),
            # mlp
            HFWeightReshardRule(
                re.compile(
                    r"visual\.blocks\.\d+\.mlp\.gate_proj\.(weight|bias)"
                ),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.mlp\.up_proj\.(weight|bias)"),
                dim=0,
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.mlp\.down_proj\.weight"), dim=1
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.blocks\.\d+\.mlp\.down_proj\.bias")
            ),
            # norm
            HFWeightReshardRule(
                re.compile(
                    r"visual\.blocks\.\d+\.norm1\.weight",
                )
            ),
            HFWeightReshardRule(
                re.compile(
                    r"visual\.blocks\.\d+\.norm2\.weight",
                )
            ),
            # vision embed
            HFWeightReshardRule(
                re.compile(r"visual\.patch_embed\.proj\.weight")
            ),
        ]
        return vision_rules

    def _build_projector_rules(self) -> List[HFWeightReshardRule]:
        projector_rules = [
            HFWeightReshardRule(re.compile(r"visual\.merger\.ln_q.weight")),
            HFWeightReshardRule(
                re.compile(r"visual\.merger\.mlp\.0\.(weight|bias)"), dim=0
            ),
            HFWeightReshardRule(
                re.compile(r"visual\.merger\.mlp\.2\.weight"), dim=1
            ),
            HFWeightReshardRule(re.compile(r"visual\.merger\.mlp\.2\.bias")),
        ]
        return projector_rules

    def build_rules(self) -> List[HFWeightReshardRule]:
        rules = []
        rules.extend(self._build_llm_rules())
        rules.extend(self._build_vision_rules())
        rules.extend(self._build_projector_rules())
        return rules


HFWeightResharderRegistry: Dict[str, BaseHFWeightResharder] = {}


def register_hf_resharder(model_arch: str, resharder: BaseHFWeightResharder):
    if model_arch in HFWeightResharderRegistry:
        raise ValueError(f"Model arch {model_arch} is already registered.")
    HFWeightResharderRegistry[model_arch] = resharder


def get_hf_resharder(model_arch: str) -> BaseHFWeightResharder:
    if model_arch not in HFWeightResharderRegistry:
        raise ValueError(f"Model arch {model_arch} is not registered.")
    return HFWeightResharderRegistry[model_arch]


register_hf_resharder("qwen2.5", Qwen2_5HFRWeightResharder)
register_hf_resharder("qwen2.5-vl", Qwen2_5VLHFWeightResharder)
