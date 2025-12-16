import torch
from typing import Union
from rlinf.utils.nested_dict_process import copy_dict_tensor


def init_real_next_obs(next_extracted_obs: Union[torch.Tensor, dict]):
    # Copy the next-extracted-obs
    if isinstance(next_extracted_obs, torch.Tensor):
        real_next_extracted_obs = next_extracted_obs.clone()
    elif isinstance(next_extracted_obs, dict):
        real_next_extracted_obs = copy_dict_tensor(
            next_extracted_obs
        )
    else:
        raise NotImplementedError
    return real_next_extracted_obs
