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


from typing import Dict, List

import torch
from megatron.core import parallel_state
from torch import nn

from rlinf.utils.convertor.utils import get_mg2hf_convertor

from .reshard_config import ReshardConfig, ReshardWeightFormat
from .utils import get_pp_resharder, get_tp_resharder


class MegatronCoreWeightReshard:
    def __init__(self, config: ReshardConfig):
        self.config = config

        assert (
            self.config.model_config.tensor_model_parallel_size
            >= self.config.reshard_tp_size
            and self.config.model_config.tensor_model_parallel_size
            % self.config.reshard_tp_size
            == 0
        ), (
            f"Invalid tensor model parallel size {self.config.model_config.tensor_model_parallel_size} "
            f"and reshard tp size {self.config.reshard_tp_size}. "
        )

        self.tp_subgroups = {}
        self.merge_factor = (
            self.config.model_config.tensor_model_parallel_size
            // self.config.reshard_tp_size
        )
        self._init_distribution_config()
        self._create_tp_subgroups()

        self._pp_resharder = (
            get_pp_resharder(model_arch=self.config.model_arch, pp_group=self.pp_group)
            if self.reshard_pp_model
            else None
        )
        self._tp_resharder = (
            get_tp_resharder(
                model_arch=self.config.model_arch, merge_factor=self.merge_factor
            )
            if self.reshard_tp_model
            else None
        )
        self._mg2hf_convertor = (
            get_mg2hf_convertor(
                model_arch=self.config.model_arch, config=self.config, strict=True
            )
            if self.convert_to_hf
            else None
        )

    def _init_distribution_config(self):
        self.world_size = torch.distributed.get_world_size()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.pp_group = parallel_state.get_pipeline_model_parallel_group()
        self.vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        if not self.vp_size:
            self.vp_size = 1

        self.convert_to_hf = (
            self.config.reshard_weights_format != ReshardWeightFormat.MCORE
        )
        self.reshard_pp_model = self.pp_size > 1 and (
            self.config.reshard_pp_size != self.pp_size
            or self.config.reshard_tp_size != self.tp_size
        )
        self.reshard_tp_model = True
        # NOTE (wyq): Always reshard TP model even when tp_size == reshard_tp_size.
        # When tp_size == reshard_tp_size, resharding is equivalent to copying.
        # The rollout engine may load incorrect weights if not copied before offloading.

    def _create_tp_subgroups(self):
        num_groups = self.world_size // self.merge_factor
        all_subgroups = [
            list(range(i * self.merge_factor, (i + 1) * self.merge_factor))
            for i in range(num_groups)
        ]

        for subgroup_ranks in all_subgroups:
            key = tuple(subgroup_ranks)
            if key not in self.tp_subgroups:
                self.tp_subgroups[key] = torch.distributed.new_group(
                    subgroup_ranks, backend="nccl"
                )

    def _get_tp_subgroup(self, rank: int) -> torch.distributed.ProcessGroup:
        """
        Retrieve an existing communication subgroup.
        """
        group_index = rank // self.merge_factor
        subgroup_ranks = list(
            range(
                group_index * self.merge_factor,
                (group_index + 1) * self.merge_factor,
            )
        )
        key = tuple(subgroup_ranks)
        if key in self.tp_subgroups:
            return self.tp_subgroups[key]

        raise ValueError(
            f"Subgroup {key} does not exist! Please call _create_tp_subgroups() to create this subgroup first."
        )

    def gather_and_reshard_model(self, model: List[nn.Module]) -> Dict:
        """
        Accumulate all vp model chunks together, and reshard model (i.e) gather all pp ranks
        if required and return the final model state dict
        """

        def _get_layer_index(split_key):
            for index, key in enumerate(split_key):
                if key == "layers":
                    return index + 1
            raise ValueError(f"Unknown layer name format: {split_key}")

        def rename_layer_num(param_name, layer_num):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            split_key[layer_index] = str(layer_num)
            return ".".join(split_key)

        def get_layer_num(param_name):
            split_key = param_name.split(".")
            layer_index = int(_get_layer_index(split_key))
            return int(split_key[layer_index])

        layers_per_pp = self.config.model_config.num_layers // self.pp_size
        layers_per_chunk = layers_per_pp // self.vp_size

        tl_params = {}
        model_level_params = {}
        if self.vp_size > 1:  # consolidate params across model chunks
            for idx, model_chunk in enumerate(model):
                for key, val in model_chunk.state_dict().items():
                    if "_extra_state" in key:
                        continue
                    if torch.is_tensor(val):
                        if "layers" in key:
                            key2 = rename_layer_num(
                                key,
                                get_layer_num(key)
                                + idx * self.pp_size * layers_per_chunk,
                            )
                            tl_params[key2] = val
                        else:
                            model_level_params[key] = val
        else:
            for key, val in model[0].state_dict().items():
                if "_extra_state" in key:
                    continue
                if torch.is_tensor(val):
                    if "decoder.layers" in key:
                        tl_params[key] = val
                    else:
                        model_level_params[key] = val

        if self.vp_size > 1 or self.reshard_pp_model:
            # gather layers across pp ranks
            gathered_params = {}
            for key, val in tl_params.items():
                weight_list = [torch.zeros_like(val) for _ in range(self.pp_size)]
                torch.distributed.all_gather(weight_list, val, group=self.pp_group)
                for idx in range(self.pp_size):
                    layer_num = get_layer_num(key) + idx * layers_per_chunk
                    key2 = rename_layer_num(key, layer_num)
                    if (
                        not self.reshard_pp_model
                    ):  # Save only layers of 1 single PP stage
                        layers_start = layers_per_pp * self.pp_rank
                        layers_end = layers_per_pp * (self.pp_rank + 1) - 1
                        if layer_num >= layers_start and layer_num <= layers_end:
                            key2 = rename_layer_num(key, layer_num % layers_per_pp)
                            gathered_params[key2] = weight_list[idx]
                    else:
                        gathered_params[key2] = weight_list[idx]
            tl_params = gathered_params

        model_state_dict = model_level_params
        model_state_dict.update(tl_params)

        if self.reshard_pp_model:
            model_state_dict = self._pp_resharder.apply(model_state_dict)

        if self.reshard_tp_model:
            rank = torch.distributed.get_rank()
            tp_subgroup = self._get_tp_subgroup(rank)
            model_state_dict = self._tp_resharder.apply(model_state_dict, tp_subgroup)

        if self.convert_to_hf:
            model_state_dict = self._mg2hf_convertor.convert(model_state_dict)

        return model_state_dict
