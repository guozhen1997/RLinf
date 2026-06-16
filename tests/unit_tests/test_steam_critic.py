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

"""Unit tests for the STEAM critic model."""

from types import SimpleNamespace

import torch

from rlinf.models.embodiment.steam.modeling_critic import SteamCriticModel


def test_steam_gradient_checkpointing_is_idempotent():
    """enable()/disable() are idempotent and fan out to each backbone submodule."""

    class _CountingCheckpointModule:
        def __init__(self):
            self.enable_calls = 0
            self.disable_calls = 0

        def gradient_checkpointing_enable(self, **kwargs):
            del kwargs
            self.enable_calls += 1

        def gradient_checkpointing_disable(self):
            self.disable_calls += 1

    critic = object.__new__(SteamCriticModel)
    torch.nn.Module.__init__(critic)
    vision_encoder = _CountingCheckpointModule()
    language_model = _CountingCheckpointModule()
    critic.model = SimpleNamespace(
        vision_encoder=vision_encoder,
        language_model=language_model,
    )
    critic.gradient_checkpointing_enabled = False

    critic.gradient_checkpointing_enable()
    critic.gradient_checkpointing_enable()
    assert vision_encoder.enable_calls == 1
    assert language_model.enable_calls == 1

    critic.gradient_checkpointing_disable()
    critic.gradient_checkpointing_disable()
    assert vision_encoder.disable_calls == 1
    assert language_model.disable_calls == 1
