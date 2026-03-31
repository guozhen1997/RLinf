# Copyright 2026 The RLinf Authors.
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
# dreamzero model configs
import os
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy
import sys
from pathlib import Path
import json
import torch

def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    dreamzero_root = dreamzero_root / "DreamZero"
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))
        
def get_model(cfg: DictConfig ,torch_dtype=None):
    """Load DreamZero policy from checkpoint.
    """
    _ensure_groot_importable()
    model_path = Path(cfg.get("model_path"))
    if not model_path.exists():
      raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    tokenizer_path = cfg.get("tokenizer_path","google/umt5-xxl")
    precision = cfg.get("precision", "bf16")

    from groot.vla.model.dreamzero.base_vla import VLA, VLAConfig
    from safetensors.torch import load_file

    config_path = model_path / "config.json"
    if not config_path.exists():
      raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
      config_dict = json.load(f)
    
    config = VLAConfig(**config_dict)
    # Disable defer_lora_injection for immediate loading
    if "config" in config.action_head_cfg and isinstance(config.action_head_cfg["config"], dict):
        config.action_head_cfg["config"]["defer_lora_injection"] = False
        config.action_head_cfg["config"]["skip_component_loading"] = True

    model = VLA(config)

    #  load safetensors (support index shard)
    state_dict = {}
    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_path / shard_file)))
    elif st.exists():
        state_dict.update(load_file(str(st)))
    else:
        raise FileNotFoundError(f"No safetensors weights under {model_path}")
    if any(".base_layer." in k for k in state_dict):
        state_dict = {k.replace(".base_layer.", "."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "post_initialize"):
        model.post_initialize()
    if precision == "bf16":
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device="cuda")

    from groot.vla.data.schema import DatasetMetadata
    from groot.vla.data.transform import ComposedModalityTransform
    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = exp_cfg_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)

    embodiment_tag = cfg.get("embodiment_tag", "libero_sim")
    metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
    eval_transform = instantiate(train_cfg.transforms[embodiment_tag])
    assert isinstance(eval_transform, ComposedModalityTransform), f"{eval_transform=}"
    eval_transform.set_metadata(metadata)
    eval_transform.eval()

    model = DreamZeroPolicy(
        model=model,
        eval_transform=eval_transform,
        train_cfg=train_cfg,
        cfg=cfg,
    )

    return model
