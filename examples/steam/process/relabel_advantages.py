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

"""CLI: relabel the bool ``advantage`` column on an existing advantages parquet.

Pure CPU — no GPU inference. Thin wrapper around
:func:`rlinf.data.process.steam.relabel.relabel_advantages`. See the STEAM
pipeline docs for the threshold/quantile semantics and worked examples.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Make the rlinf package importable regardless of the cwd the user launched from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rlinf.data.process.steam.labelling import resolve_quantile_alias  # noqa: E402
from rlinf.data.process.steam.relabel import relabel_advantages  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel the bool advantage column on an existing advantages "
            "parquet using a new threshold or a cross-rollout quantile. "
            "Pure CPU; advantage_continuous is not recomputed."
        )
    )
    parser.add_argument(
        "--dataset_paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more LeRobot dataset roots (each containing a meta/ dir).",
    )
    parser.add_argument(
        "--source_tag",
        required=True,
        help="Tag of the existing advantages parquet to read as the baseline.",
    )
    parser.add_argument(
        "--new_tag",
        required=True,
        help="Output tag; written to meta/advantages_{new_tag}.parquet.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("threshold", "quantile"),
        help=(
            "Labelling mode. 'threshold' uses --positive_threshold as-is "
            "(sft frames stay True); 'quantile' labels the top "
            "--rollout_quantile fraction of rollout frames and, when "
            "--expert_quantile is set, the top --expert_quantile fraction of "
            "sft frames (scored over independent pools)."
        ),
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=None,
        help=(
            "Signed-score threshold in [-1, 1] (required when --mode=threshold)."
        ),
    )
    parser.add_argument(
        "--rollout_quantile",
        type=float,
        default=None,
        help=(
            "Top fraction of rollout frames to label True (required when "
            "--mode=quantile). Must be in (0, 1); e.g. 0.3 ⇒ top 30%%."
        ),
    )
    parser.add_argument(
        "--expert_quantile",
        type=float,
        default=None,
        help=(
            "Top fraction of sft (expert) frames to label True (optional, "
            "--mode=quantile only). Must be in (0, 1). Omit to label every "
            "sft frame True (historical default)."
        ),
    )
    parser.add_argument(
        "--positive_quantile",
        type=float,
        default=None,
        help="Deprecated alias for --rollout_quantile.",
    )
    parser.add_argument(
        "--dataset_types",
        nargs="+",
        default=None,
        choices=("sft", "rollout"),
        help=(
            "One entry per --dataset_paths, overriding mixture_config. If "
            "omitted, dataset_type is read from tags[source_tag].dataset_type."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    rollout_quantile = resolve_quantile_alias(
        args.rollout_quantile, args.positive_quantile
    )
    if (
        args.mode == "quantile"
        and args.positive_quantile is not None
        and args.rollout_quantile is None
    ):
        logger.warning(
            "--positive_quantile is a deprecated alias for --rollout_quantile; "
            "please use --rollout_quantile."
        )
    relabel_advantages(
        args.dataset_paths,
        source_tag=args.source_tag,
        new_tag=args.new_tag,
        mode=args.mode,
        positive_threshold=args.positive_threshold,
        rollout_quantile=rollout_quantile,
        expert_quantile=args.expert_quantile,
        dataset_types=args.dataset_types,
    )


if __name__ == "__main__":
    main()
