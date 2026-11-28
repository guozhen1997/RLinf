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

"""Unit and parity tests for RLinf's JAX-free OpenPI transforms, configs, and loader."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

from rlinf.data.datasets.openpi import torch_data_loader as rlinf_loader
from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT, get_openpi_config
from rlinf.models.embodiment.openpi.dataconfig.base import (
    AssetsConfig,
    DataConfig,
    ModelTransformFactory,
    TrainConfig,
    load_norm_stats,
)
from rlinf.models.embodiment.openpi.modules import model as rlinf_model
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.transforms import (
    core as rlinf_transforms,
    download as rlinf_download,
    image_tools as rlinf_image_tools,
    normalize as rlinf_normalize,
    tokenizer as rlinf_tokenizer,
)
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    build_openpi_transforms,
    split_short_memory_transforms,
)


def _sample_stats() -> dict[str, rlinf_normalize.NormStats]:
    return {
        "state": rlinf_normalize.NormStats(
            mean=np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float64),
            std=np.array([1.0, 0.5, 2.0, 0.25, 1.5, 0.8, 0.3], dtype=np.float64),
            q01=np.array([-1.0, -0.8, -2.0, -0.5, -1.2, -0.9, 0.0], dtype=np.float64),
            q99=np.array([1.2, 0.9, 2.5, 0.7, 1.4, 1.1, 1.0], dtype=np.float64),
        ),
        "actions": rlinf_normalize.NormStats(
            mean=np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.3, 0.5], dtype=np.float64),
            std=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.4], dtype=np.float64),
            q01=np.array([-0.5, -0.6, -0.7, -0.8, -0.9, -1.0, 0.0], dtype=np.float64),
            q99=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0], dtype=np.float64),
        ),
    }


def test_flatten_unflatten_and_repack_transform():
    nested = {
        "observation": {
            "images": {"top": np.zeros((4, 4, 3), dtype=np.uint8)},
            "state": np.arange(7, dtype=np.float32),
        },
        "action": np.ones((10, 7), dtype=np.float32),
    }
    flat = rlinf_transforms.flatten_dict(nested)
    assert set(flat.keys()) == {
        "observation/images/top",
        "observation/state",
        "action",
    }
    rebuilt = rlinf_transforms.unflatten_dict(flat)
    assert np.array_equal(rebuilt["observation"]["state"], nested["observation"]["state"])

    repack = rlinf_transforms.RepackTransform(
        {
            "images": {"cam_high": "observation/images/top"},
            "state": "observation/state",
            "actions": "action",
        }
    )
    out = repack(nested)
    assert np.array_equal(out["images"]["cam_high"], nested["observation"]["images"]["top"])
    assert np.array_equal(out["actions"], nested["action"])


def test_normalize_unnormalize_zscore_and_quantile():
    stats = _sample_stats()
    sample = {
        "state": np.array([0.2, 0.0, 0.5, 0.1, -0.3, 0.2, 0.8], dtype=np.float32),
        "actions": np.linspace(-0.4, 0.4, 35, dtype=np.float32).reshape(5, 7),
    }
    for use_quantiles in (False, True):
        norm = rlinf_transforms.Normalize(stats, use_quantiles=use_quantiles)
        unnorm = rlinf_transforms.Unnormalize(stats, use_quantiles=use_quantiles)
        encoded = norm(dict(sample))
        decoded = unnorm(dict(encoded))
        assert np.allclose(decoded["state"], sample["state"], atol=1e-5)
        assert np.allclose(decoded["actions"], sample["actions"], atol=1e-5)


def test_delta_and_absolute_actions_roundtrip():
    mask = rlinf_transforms.make_bool_mask(6, -1)
    assert mask == (True, True, True, True, True, True, False)
    state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5], dtype=np.float32)
    actions = np.ones((4, 7), dtype=np.float32) * 2.0
    data = {"state": state.copy(), "actions": actions.copy()}
    delta_out = rlinf_transforms.DeltaActions(mask)(data)
    assert np.allclose(delta_out["actions"][:, 6], 2.0)
    abs_out = rlinf_transforms.AbsoluteActions(mask)(delta_out)
    assert np.allclose(abs_out["actions"], actions)


def test_norm_stats_roundtrip_and_running_stats(tmp_path: pathlib.Path):
    rng = np.random.default_rng(0)
    running = rlinf_normalize.RunningStats()
    for _ in range(5):
        running.update(rng.normal(size=(32, 7)))
    stats = {"state": running.get_statistics()}
    rlinf_normalize.save(tmp_path / "demo_asset", stats)
    loaded = load_norm_stats(tmp_path, "demo_asset")
    assert np.allclose(loaded["state"].mean, stats["state"].mean)
    assert np.allclose(loaded["state"].std, stats["state"].std)
    assert np.allclose(loaded["state"].q01, stats["state"].q01)
    assert np.allclose(loaded["state"].q99, stats["state"].q99)


def test_fake_torch_data_loader_yields_observation_and_actions():
    config = TrainConfig(
        name="fake_test",
        model=Pi0Config(pi05=True, action_horizon=8, action_dim=14, max_token_len=32),
        batch_size=4,
        num_workers=0,
    )
    loader = rlinf_loader.create_data_loader(
        config, framework="pytorch", num_batches=2, skip_norm_stats=True
    )
    batches = list(loader)
    assert len(batches) == 2
    obs, actions = batches[0]
    assert isinstance(obs, rlinf_model.Observation)
    assert actions.shape == (4, 8, 14)
    assert obs.state.shape == (4, 14)
    assert obs.tokenized_prompt.shape == (4, 32)
    assert set(obs.images.keys()) == set(rlinf_model.IMAGE_KEYS)


def test_all_registered_openpi_configs_are_accessible():
    assert len(_CONFIGS_DICT) > 0
    for name in _CONFIGS_DICT:
        cfg = get_openpi_config(name)
        assert cfg.name == name
        assert isinstance(cfg.model, Pi0Config)
        assert cfg.model.model_type in (
            rlinf_model.ModelType.PI0,
            rlinf_model.ModelType.PI05,
        )


def test_build_openpi_transforms_and_short_memory_split(tmp_path: pathlib.Path):
    stats = _sample_stats()
    asset_dir = tmp_path / "physical-intelligence" / "libero"
    rlinf_normalize.save(asset_dir, stats)

    input_transforms, output_transforms = build_openpi_transforms(
        str(tmp_path), "pi05_libero"
    )
    frame_tf, sample_tf = split_short_memory_transforms(input_transforms)
    assert callable(frame_tf)
    assert callable(sample_tf)
    assert len(output_transforms) > 0


# ---------------------------------------------------------------------------
# Parity tests against upstream `openpi` (run when `openpi` is installed).
# ---------------------------------------------------------------------------


def test_parity_with_upstream_openpi_transforms_and_normalize(tmp_path: pathlib.Path):
    openpi_transforms = pytest.importorskip("openpi.transforms")
    openpi_normalize = pytest.importorskip("openpi.shared.normalize")
    openpi_image_tools = pytest.importorskip("openpi_client.image_tools")

    rng = np.random.default_rng(42)
    # 1. RunningStats + JSON serialization parity
    rs_new = rlinf_normalize.RunningStats()
    rs_old = openpi_normalize.RunningStats()
    for _ in range(4):
        batch = rng.normal(loc=0.5, scale=1.5, size=(64, 7))
        rs_new.update(batch)
        rs_old.update(batch)
    st_new = rs_new.get_statistics()
    st_old = rs_old.get_statistics()
    assert np.allclose(st_new.mean, st_old.mean, atol=0, rtol=0)
    assert np.allclose(st_new.std, st_old.std, atol=0, rtol=0)
    assert np.allclose(st_new.q01, st_old.q01, atol=0, rtol=0)
    assert np.allclose(st_new.q99, st_old.q99, atol=0, rtol=0)

    openpi_normalize.save(tmp_path / "old", {"state": st_old, "actions": st_old})
    loaded_by_new = rlinf_normalize.load(tmp_path / "old")
    rlinf_normalize.save(tmp_path / "new", loaded_by_new)
    loaded_by_old = openpi_normalize.load(tmp_path / "new")
    assert np.allclose(loaded_by_old["state"].mean, st_old.mean)
    assert np.allclose(loaded_by_old["state"].q99, st_old.q99)

    # 2. Image resize_with_pad parity (bit-exact)
    img = rng.integers(0, 256, size=(2, 180, 320, 3), dtype=np.uint8)
    assert np.array_equal(
        rlinf_image_tools.resize_with_pad(img, 224, 224),
        openpi_image_tools.resize_with_pad(img, 224, 224),
    )

    # 3. Normalize / Unnormalize parity (z-score and quantile)
    stats_new = {"state": st_new, "actions": st_new}
    stats_old = {"state": st_old, "actions": st_old}
    sample = {
        "state": rng.normal(size=(7,)).astype(np.float32),
        "actions": rng.normal(size=(10, 7)).astype(np.float32),
    }
    for use_q in (False, True):
        n_new = rlinf_transforms.Normalize(stats_new, use_quantiles=use_q)(dict(sample))
        n_old = openpi_transforms.Normalize(stats_old, use_quantiles=use_q)(dict(sample))
        assert np.allclose(n_new["state"], n_old["state"], atol=0, rtol=0)
        assert np.allclose(n_new["actions"], n_old["actions"], atol=0, rtol=0)

        u_new = rlinf_transforms.Unnormalize(stats_new, use_quantiles=use_q)(dict(n_new))
        u_old = openpi_transforms.Unnormalize(stats_old, use_quantiles=use_q)(dict(n_old))
        assert np.allclose(u_new["state"], u_old["state"], atol=0, rtol=0)
        assert np.allclose(u_new["actions"], u_old["actions"], atol=0, rtol=0)

    # 4. transform_dict + DeltaActions + AbsoluteActions + PadStatesAndActions parity
    tree = {"a": {"b": 1, "c": 2}, "d": 3}
    patterns = {"a/(.*)": r"x/\1", "d": None}
    assert rlinf_transforms.transform_dict(patterns, tree) == openpi_transforms.transform_dict(
        patterns, tree
    )

    mask = rlinf_transforms.make_bool_mask(6, -1)
    d_new = rlinf_transforms.DeltaActions(mask)(
        {"state": sample["state"].copy(), "actions": sample["actions"].copy()}
    )
    d_old = openpi_transforms.DeltaActions(mask)(
        {"state": sample["state"].copy(), "actions": sample["actions"].copy()}
    )
    assert np.array_equal(d_new["actions"], d_old["actions"])

    p_new = rlinf_transforms.PadStatesAndActions(32)(
        {"state": sample["state"].copy(), "actions": sample["actions"].copy()}
    )
    p_old = openpi_transforms.PadStatesAndActions(32)(
        {"state": sample["state"].copy(), "actions": sample["actions"].copy()}
    )
    assert np.array_equal(p_new["state"], p_old["state"])
    assert np.array_equal(p_new["actions"], p_old["actions"])


def test_parity_with_upstream_openpi_tokenizer():
    openpi_tokenizer = pytest.importorskip("openpi.models.tokenizer")

    tok_new_pi0 = rlinf_tokenizer.PaligemmaTokenizer(max_len=48)
    tok_old_pi0 = openpi_tokenizer.PaligemmaTokenizer(max_len=48)
    t_new, m_new = tok_new_pi0.tokenize("pick up the red_cube\nnow")
    t_old, m_old = tok_old_pi0.tokenize("pick up the red_cube\nnow")
    assert np.array_equal(t_new, t_old)
    assert np.array_equal(m_new, m_old)

    tok_new_pi05 = rlinf_tokenizer.PaligemmaTokenizer(max_len=200)
    tok_old_pi05 = openpi_tokenizer.PaligemmaTokenizer(max_len=200)
    state = np.linspace(-1.2, 1.2, 32, dtype=np.float32)
    t_new, m_new = tok_new_pi05.tokenize("insert peg into hole", state=state)
    t_old, m_old = tok_old_pi05.tokenize("insert peg into hole", state=state)
    assert np.array_equal(t_new, t_old)
    assert np.array_equal(m_new, m_old)
