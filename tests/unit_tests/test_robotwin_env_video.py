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
import os
import types

import numpy as np
import pytest

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv as robotwin_mod


@pytest.fixture
def dummy_env_cfg():
    class Cfg:
        def __init__(self):
            self.init_params = {"num_envs": 4}
            self.horizon = 1
            self.image_size = (64, 64)

    return Cfg()


def _build_minimal_self(cfg):
    self_obj = types.SimpleNamespace()
    self_obj.video_cfg = types.SimpleNamespace(
        video_base_dir="/tmp/robotwin_test_videos"
    )
    self_obj.seed = 42
    self_obj.video_cnt = 0
    self_obj.render_images = []
    self_obj.num_envs = cfg.init_params["num_envs"]
    return self_obj


def test_flush_video(monkeypatch, tmp_path, dummy_env_cfg):
    self_obj = _build_minimal_self(dummy_env_cfg)

    self_obj.video_cfg.video_base_dir = str(tmp_path)

    fake_frame = np.zeros((10, 20, 3), dtype=np.uint8)
    self_obj.render_images = [fake_frame for _ in range(3)]  # 三帧

    called = {"args": None, "kwargs": None, "times": 0}

    def fake_save_rollout_video(frames, output_dir, video_name, fps=30):
        called["times"] += 1
        called["args"] = (list(frames), output_dir, video_name)
        called["kwargs"] = {"fps": fps}
        os.makedirs(output_dir, exist_ok=True)

    monkeypatch.setattr(robotwin_mod, "save_rollout_video", fake_save_rollout_video)

    robotwin_mod.RoboTwin.flush_video(self_obj)

    assert called["times"] == 1
    frames, output_dir, video_name = called["args"]
    assert isinstance(frames, list) and len(frames) == 3
    assert frames[0].shape == fake_frame.shape
    assert output_dir.endswith(f"seed_{self_obj.seed}")
    assert video_name == "0"
    assert called["kwargs"]["fps"] == 30

    assert self_obj.video_cnt == 1
    assert self_obj.render_images == []

    self_obj.render_images = [fake_frame]
    robotwin_mod.RoboTwin.flush_video(self_obj, video_sub_dir="eval")
    assert called["times"] == 2
    frames2, output_dir2, video_name2 = called["args"]
    assert isinstance(frames2, list) and len(frames2) == 1
    assert output_dir2.endswith(f"seed_{self_obj.seed}/eval")
    assert video_name2 == "1"
    assert self_obj.video_cnt == 2
    assert self_obj.render_images == []


def test_add_new_frames(monkeypatch, dummy_env_cfg):
    self_obj = _build_minimal_self(dummy_env_cfg)

    H, W = 8, 12
    raw_obs = []
    for i in range(self_obj.num_envs):
        img = np.full((H, W, 3), fill_value=i, dtype=np.uint8)
        raw_obs.append({"agentview_image": img})

    plot_infos = {
        "scalar": 1.23,
        "per_env": np.array([10, 20, 30, 40], dtype=np.int32),
    }

    def fake_put_info_on_image(image, info):
        assert isinstance(image, np.ndarray)
        marked = image.copy()
        marked[0, 0, 0] = 255
        return marked

    def fake_tile_images(images, nrows):
        assert len(images) == self_obj.num_envs
        return np.concatenate(images, axis=1)

    monkeypatch.setattr(robotwin_mod, "put_info_on_image", fake_put_info_on_image)
    monkeypatch.setattr(robotwin_mod, "tile_images", fake_tile_images)

    robotwin_mod.RoboTwin.add_new_frames(self_obj, raw_obs, plot_infos)

    assert isinstance(self_obj.render_images, list)
    assert len(self_obj.render_images) == 1

    tiled = self_obj.render_images[0]
    assert tiled.shape == (H, W * self_obj.num_envs, 3)

    assert tiled[0, 0, 0] == 255
