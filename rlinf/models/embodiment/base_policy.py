import torch.nn as nn

class BasePolicy(nn.Module):
    def preprocess_env_obs(self, env_obs):
        return env_obs
    
    def forward(
        self, forward_type="default_forward", **kwargs
    ):
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError
    
    def default_forward(self, **kwargs):
        raise NotImplementedError