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

"""Hydra entry point: compute STEAM ensemble advantages for CFG-RL training.

Thin wrapper around
:func:`rlinf.data.process.steam.pipeline.compute_ensemble_advantages`. See the
STEAM pipeline docs for the output column schema, ``label_mode`` semantics, and
launch commands (single-GPU and ``torchrun``).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make the rlinf package importable regardless of the cwd the user launched from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Install the libav/libdav1d stderr filter before the heavy torch / torchcodec
# imports so the fd=2 redirect is in place before libav loads.
from rlinf.utils.logging import silence_libav_logs  # noqa: E402

silence_libav_logs()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from rlinf.data.process.steam.pipeline import (  # noqa: E402
    compute_ensemble_advantages,
)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_advantages_ensemble",
)
def main(cfg: DictConfig) -> None:
    compute_ensemble_advantages(cfg)


if __name__ == "__main__":
    main()
