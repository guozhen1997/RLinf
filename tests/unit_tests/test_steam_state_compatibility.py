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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rlinf.data.datasets.steam import BinaryPairDataCollator, PairDataset
from rlinf.data.datasets.steam.pair_dataset import build_openpi_state_transform
from rlinf.models.embodiment.steam.modeling_critic import (
    SteamCriticModel,
)


class _FakeImageProcessor:
    def __call__(self, images, image_masks=None, return_tensors="pt", train=False):
        del return_tensors, train
        pixel_values = {}
        masks = {}
        for cam, frames in images.items():
            pixel_values[cam] = frames.permute(0, 3, 1, 2).to(dtype=torch.float32)
            pixel_values[cam] = pixel_values[cam] / 255.0
            if image_masks is None or cam not in image_masks:
                masks[cam] = torch.ones(frames.shape[0], dtype=torch.bool)
            else:
                masks[cam] = image_masks[cam].to(dtype=torch.bool)
        return {"pixel_values": pixel_values, "image_masks": masks}


class _FakeProcessor:
    image_processor = _FakeImageProcessor()

    def process_text(self, prompts, states=None, max_length=8, return_tensors="pt"):
        del prompts, states, return_tensors
        batch = 2
        return {
            "input_ids": torch.zeros(batch, max_length, dtype=torch.long),
            "attention_mask": torch.ones(batch, max_length, dtype=torch.bool),
        }


def test_compatibility_negative_weights_are_distance_weighted():
    distances = torch.tensor([0.0, 2.0, 5.0, 10.0])
    weights = SteamCriticModel._compatibility_negative_weights(
        distances,
        distance_scale=5.0,
        min_weight=0.2,
    )

    assert torch.all(weights[1:] >= weights[:-1])
    assert torch.isclose(weights[0], torch.tensor(0.2))
    assert torch.isclose(weights[2], torch.tensor(1.0))
    assert torch.isclose(weights[3], torch.tensor(1.0))


def test_steam_gradient_checkpointing_is_idempotent():
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


def test_compatibility_gate_matches_formula():
    progress = torch.tensor([-1.0, 0.0, 1.0])
    compat = torch.tensor([0.25, 0.5, 1.0])
    gated = SteamCriticModel._apply_compatibility_gate(
        progress,
        compat,
        gate_floor=0.1,
    )

    expected = 2.0 * (((progress + 1.0) / 2.0) * (0.1 + 0.9 * compat)) - 1.0
    assert torch.allclose(gated, expected)


def test_binary_pair_collator_emits_state_compatibility_schema():
    img0 = np.zeros((4, 4, 3), dtype=np.uint8)
    img1 = np.ones((4, 4, 3), dtype=np.uint8) * 255
    state0 = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    state1 = np.linspace(1.0, -1.0, 4, dtype=np.float32)

    examples = [
        {
            "image_t": {"cam": img0},
            "image_tk": {"cam": img1},
            "image_mask_t": {"cam": True},
            "image_mask_tk": {"cam": True},
            "prompt": "task",
            "state": state0,
            "state_tk": state1,
            "state_neg_t": np.stack([state1, state0]),
            "state_neg_tk": np.stack([state0, state1]),
            "state_neg_distance_t": np.asarray([1.0, 3.0], dtype=np.float32),
            "state_neg_distance_tk": np.asarray([2.0, 4.0], dtype=np.float32),
            "label": 1,
            "episode": 0,
            "frame_idx_t": 1,
            "frame_idx_tk": 3,
        },
        {
            "image_t": {"cam": img1},
            "image_tk": {"cam": img0},
            "image_mask_t": {"cam": True},
            "image_mask_tk": {"cam": True},
            "prompt": "task",
            "state": state1,
            "state_tk": state0,
            "state_neg_t": np.stack([state0, state1]),
            "state_neg_tk": np.stack([state1, state0]),
            "state_neg_distance_t": np.asarray([2.0, 5.0], dtype=np.float32),
            "state_neg_distance_tk": np.asarray([1.0, 6.0], dtype=np.float32),
            "label": 0,
            "episode": 0,
            "frame_idx_t": 3,
            "frame_idx_tk": 1,
        },
    ]

    batch = BinaryPairDataCollator(
        processor=_FakeProcessor(),
        max_length=8,
        train=False,
        num_bins=2,
    )(examples)
    obs = batch["observation"]

    assert obs["state_t"].shape == (2, 4)
    assert obs["state_tk"].shape == (2, 4)
    assert obs["state_neg_t"].shape == (2, 2, 4)
    assert obs["state_neg_tk"].shape == (2, 2, 4)
    assert obs["state_neg_distance_t"].shape == (2, 2)
    assert obs["state_neg_distance_tk"].shape == (2, 2)
    assert obs["episode"].tolist() == [0, 0]
    assert batch["labels"].tolist() == [1, 0]


def test_load_state_uses_state_only_transform_without_raw_sample():
    class _StateOnlySource:
        def __init__(self):
            self.state_calls = 0
            self.raw_calls = 0

        def get_state(self, episode, frame, state_key):
            self.state_calls += 1
            assert (episode, frame, state_key) == (3, 5, "state")
            return np.asarray([1.0, 2.0], dtype=np.float32)

        def get_raw_sample(self, episode, frame):
            self.raw_calls += 1
            raise AssertionError("state compatibility must not load raw image samples")

    source = _StateOnlySource()
    dataset = object.__new__(PairDataset)
    dataset._source = source
    dataset.state_key = "state"
    dataset.state_max_dim = 4
    dataset._state_transform = lambda state: np.asarray(state, dtype=np.float32) + 1.0

    state = PairDataset._load_state(dataset, 3, 5)

    assert np.allclose(state, np.asarray([2.0, 3.0, 0.0, 0.0], dtype=np.float32))
    assert source.state_calls == 1
    assert source.raw_calls == 0


def test_openpi_state_only_transform_matches_arx_state_path_without_images():
    openpi_transforms = pytest.importorskip("openpi.transforms")
    openpi_model = pytest.importorskip("openpi.models.model")

    from rlinf.data.datasets.steam.pair_dataset import _X2ROBOT_REPACK_KEYS
    from rlinf.models.embodiment.openpi.policies import arx_policy

    state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    sample = {
        "state": state.copy(),
        "actions": np.zeros((1, 14), dtype=np.float32),
        "left_wrist_view": image.copy(),
        "face_view": image.copy(),
        "right_wrist_view": image.copy(),
        "task": "fold towel",
    }
    full_transform = openpi_transforms.compose(
        [
            openpi_transforms.RepackTransform(_X2ROBOT_REPACK_KEYS),
            arx_policy.ArxInputs(
                mode="sm2sm",
                action_dim=28,
                model_type=openpi_model.ModelType.PI0,
            ),
            openpi_transforms.InjectDefaultPrompt(None),
            openpi_transforms.PadStatesAndActions(28),
        ]
    )
    expected = full_transform(sample)["state"]

    state_only_transform = build_openpi_state_transform(
        robot_type="x2robot_sm2sm",
        model_type="pi0",
        action_dim=28,
        default_prompt=None,
        norm_stats_dir=None,
        asset_id=None,
    )

    assert np.allclose(state_only_transform(state), expected)
