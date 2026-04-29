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

"""Runtime replacements for DreamZero components (groot) used by RLinf."""

from rlinf.models.embodiment.dreamzero.patch.wan_causal_model_forward_train import (
    _forward_train,
)
from rlinf.models.embodiment.dreamzero.patch.wan_self_attention import (
    _process_clean_image_only,
    _process_noisy_action_blocks,
    _process_noisy_image_blocks,
    _process_state_blocks,
)
from rlinf.models.embodiment.dreamzero.patch.wan_video_vae import (
    WanVideoVAE,
    WanVideoVAE38,
    WanVideoVAEStateDictConverter,
)

__all__ = [
    "WanVideoVAE",
    "WanVideoVAE38",
    "WanVideoVAEStateDictConverter",
    "_forward_train",
    "_process_clean_image_only",
    "_process_noisy_action_blocks",
    "_process_noisy_image_blocks",
    "_process_state_blocks",
]
