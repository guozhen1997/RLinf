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

"""Offline advantage-annotation helpers for the STEAM value critic.

Split into a CPU-only labelling/metadata layer and a torch inference layer so
the pure-pandas relabel tool can import the former without pulling in torch or
LeRobot:

* :mod:`~rlinf.data.process.steam.labelling` — signed-score → boolean-advantage
  labelling and quantile thresholds (pandas / numpy only).
* :mod:`~rlinf.data.process.steam.mixture_config` — ``meta/mixture_config.yaml``
  per-tag read/write (PyYAML only).
* :mod:`~rlinf.data.process.steam.inference` — ensemble inference loop, sharded
  dataloader, and per-frame record building (torch).
* :mod:`~rlinf.data.process.steam.pipeline` — the end-to-end
  ``compute_ensemble_advantages`` orchestration (torch).

Submodules are intentionally NOT eagerly imported here: ``inference`` /
``pipeline`` pull torch and the STEAM pair dataset, whereas ``labelling`` /
``mixture_config`` stay dependency-light. Import the specific submodule you need.
"""
