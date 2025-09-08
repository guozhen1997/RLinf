from typing import Dict

import torch


class FSDPWeightReshard:
    def __init__(self, reshard_tp_size):
        self.reshard_tp_size = reshard_tp_size

        self.rollout_tp_rank = torch.distributed.get_rank() % self.reshard_tp_size

    def reshard_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        reshard full state dict
        """
        local_state = {}

        for k, v in state_dict.items():
            if (
                "norm.weight" in k
                or "rotary_emb.inv_freq" in k
                or "input_layernorm.weight" in k
                or "post_attention_layernorm.weight" in k
            ):
                local_state[k] = v.clone()
                continue

            if any(
                x in k
                for x in [
                    "embed_tokens",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "gate_proj",
                    "up_proj",
                ]
            ):
                dim = 0
            elif any(x in k for x in ["o_proj", "down_proj", "lm_head"]):
                dim = 1
            else:
                local_state[k] = v.clone()
                continue

            if v.ndim == 1:
                shard_size = v.shape[0] // self.reshard_tp_size
                start = self.rollout_tp_rank * shard_size
                end = (self.rollout_tp_rank + 1) * shard_size
                local_state[k] = v[start:end].clone()
            else:
                shard_size = v.shape[dim] // self.reshard_tp_size
                start = self.rollout_tp_rank * shard_size
                end = (self.rollout_tp_rank + 1) * shard_size
                local_state[k] = v.narrow(dim, start, shard_size).clone()

        return local_state
