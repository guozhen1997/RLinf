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

import numpy as np
import pytest
import torch

libero_env = pytest.importorskip("rlinf.envs.libero.libero_env")
LiberoEnv = libero_env.LiberoEnv

def _make_env(skip: bool):
    env = object.__new__(LiberoEnv)
    env.skip_intermediate_renders = skip
    env.num_envs = 1
    env._cached_camera_obs = [None]
    env.current_raw_obs = [{"cam": 0}]
    env.auto_reset = False
    env.ignore_terminations = True

    render_calls = []

    class _InnerEnv:
        def set_camera_rendering(self, enabled, id=None):
            render_calls.append(bool(enabled))

    env.env = _InnerEnv()

    def _step(action, auto_reset=False, _skip_obs_wrap=False):
        z = torch.zeros(env.num_envs)
        return {"obs": 0}, z, z.bool(), z.bool(), [{}]

    env.step = _step
    env._refresh_camera_cache = lambda raw: setattr(
        env, "_cached_camera_obs", [{"cam": 0}]
    )
    env._apply_cached_camera_obs = lambda raw: raw
    env._wrap_obs = lambda raw: {"obs": 0}
    return env, render_calls

@pytest.mark.parametrize("chunk_size", [4, 8])
def test_skip_render_skips_internediate_substeps(chunk_size):
    env, calls = _make_env(skip=True)
    env.chunk_step(np.zeros((1, chunk_size, 7), dtype=np.float32))
    assert calls.count(False) == chunk_size - 2, calls
    assert calls[-1] is True, calls

def test_no_skip_when_disabled():
    env, calls = _make_env(skip=False)
    env.chunk_step(np.zeros((1, 8, 7), dtype=np.float32))
    assert calls == []
