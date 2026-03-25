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
from omegaconf import DictConfig
from rlinf.models.embodiment.dreamzero.dreamzero_policy import DreamZeroPolicy

def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint.
    """
    model_path = cfg.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DreamZero model_path does not exist: {model_path}. "
            "Please provide a valid checkpoint directory."
        )

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
    print("tokenizer_path", tokenizer_path)
    max_seq_len = cfg.get("max_seq_len", 512)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = DreamZeroPolicy(
        model_path=model_path,
        device=device,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )

    return model
