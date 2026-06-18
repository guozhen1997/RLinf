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

"""Hydra entry point: compute per-frame returns for RECAP value-model SFT.

Thin wrapper around
:func:`rlinf.data.process.recap.compute_returns.compute_returns`. See the RECAP
pipeline docs for the output sidecar schema and launch command.
"""

import sys
from pathlib import Path

# Make the rlinf package importable regardless of the cwd the user launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from rlinf.data.process.recap.compute_returns import (  # noqa: E402
    compute_returns,
)


@hydra.main(version_base=None, config_path="config", config_name="compute_returns")
def main(cfg: DictConfig) -> None:
    compute_returns(cfg)


if __name__ == "__main__":
    main()
