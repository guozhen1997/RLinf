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

"""Lightweight per-episode advantage visualization.

Given a trained value checkpoint and a dataset, select a small number of
episodes and — without materializing a full advantages parquet — compute
per-frame V(o_t) and advantages for ONLY those episodes, then render the
per-episode summary plot and video.

Reuses existing `compute_advantages_*.yaml` configs: override ``advantage.*``
and ``data.*`` as usual, and pass visualization parameters via ``+visualize.*``.

Usage:
    cd examples/recap/process
    python visualize_episodes_with_value.py \
        --config-name compute_advantages \
        advantage.value_checkpoint=/path/to/ckpt \
        data.train_data_paths.0.dataset_path=/path/to/dataset \
        +visualize.num_episodes=5 \
        +visualize.output_dir=./viz_out

Specific episodes instead of random sampling:
    ... +visualize.episodes="[0,7,12]"

Skip video rendering (plots only):
    ... +visualize.no_video=true
"""

import gc
import logging
import os
import sys
from functools import partial
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import matplotlib
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

matplotlib.use("Agg")

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent.parent.parent))
sys.path.insert(0, str(_THIS_DIR))

from compute_advantages import (  # noqa: E402
    ValueInferenceDataset,
    _parse_value_model_kwargs,
    advantage_collate_fn,
    load_lerobot_dataset,
    to_numpy,
    to_scalar,
)
from visualize_advantage_dataset import (  # noqa: E402
    create_episode_summary_plot,
    create_episode_video,
    detect_image_keys,
)

from rlinf.data.datasets.recap.utils import (  # noqa: E402
    load_return_stats_from_dataset,
)

logger = logging.getLogger(__name__)


def _select_episodes(total_episodes: int, viz_cfg: DictConfig) -> list[int]:
    """Pick episode indices from explicit list or random sample."""
    episodes = viz_cfg.get("episodes", None)
    if episodes is not None:
        ep_list = [int(e) for e in episodes]
        bad = [e for e in ep_list if e < 0 or e >= total_episodes]
        if bad:
            raise ValueError(
                f"Requested episodes {bad} out of range "
                f"(dataset has {total_episodes} episodes)."
            )
        return sorted(set(ep_list))

    num_episodes = int(viz_cfg.get("num_episodes", 5))
    seed = int(viz_cfg.get("seed", 42))
    if num_episodes <= 0 or num_episodes >= total_episodes:
        return list(range(total_episodes))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_episodes, num_episodes, replace=False).tolist())


def _episode_range(dataset, ep_idx: int) -> tuple[int, int]:
    ep_data = dataset.episode_data_index
    start = int(ep_data["from"][ep_idx].item())
    end = int(ep_data["to"][ep_idx].item())
    return start, end


@torch.no_grad()
def _compute_value_for_indices(
    value_model,
    dataset,
    tasks: dict,
    robot_type: str,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    returns_sidecar,
) -> dict[int, dict]:
    """Run V(o_t) on the given global indices only. Returns map idx -> {value, reward, true_return, episode_index, frame_index}."""
    cpu_prep_in_workers = num_workers > 0
    processor = getattr(value_model, "processor", None)
    adv_ds = ValueInferenceDataset(
        dataset,
        robot_type,
        tasks,
        input_transform=value_model._input_transform if cpu_prep_in_workers else None,
        prepare_observation_cpu=(
            partial(value_model.__class__._prepare_observation_cpu, processor=processor)
            if cpu_prep_in_workers
            else None
        ),
        returns_sidecar=returns_sidecar,
    )
    subset = torch.utils.data.Subset(adv_ds, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=advantage_collate_fn,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    out: dict[int, dict] = {}
    for obs_list, meta_list in tqdm(loader, desc="V(o_t) inference", unit="batch"):
        results = value_model.infer_batch(
            obs_list,
            batch_size=batch_size,
            pretransformed=cpu_prep_in_workers,
            already_cpu_prepared=cpu_prep_in_workers,
        )
        if len(results) != len(meta_list):
            raise RuntimeError(
                f"infer_batch returned {len(results)} values "
                f"for {len(meta_list)} inputs"
            )
        for r, m in zip(results, meta_list):
            out[int(m["global_idx"])] = {
                "value": float(r["value"]),
                "reward": float(m["reward"]),
                "true_return": float(m["true_return"]),
                "episode_index": int(m["episode_index"]),
                "frame_index": int(m["frame_index"]),
            }

    missing = [i for i in indices if i not in out]
    if missing:
        raise RuntimeError(
            f"Value inference missing {len(missing)} indices (e.g. {missing[:5]})."
        )
    return out


def _compute_advantages_for_episodes(
    frame_data: dict[int, dict],
    episode_ranges: dict[int, tuple[int, int]],
    gamma: float,
    action_horizon: int,
    discount_next_value: bool,
    global_return_min: float,
    global_return_max: float,
) -> pd.DataFrame:
    """Compute A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t) for selected episodes."""
    ret_range = global_return_max - global_return_min

    def normalize(x):
        if ret_range <= 0:
            return -0.5
        return (x - global_return_min) / ret_range - 1.0

    gamma_powers = np.array([gamma**i for i in range(action_horizon)], dtype=np.float64)

    rows = []
    for ep_idx, (ep_start, ep_end) in episode_ranges.items():
        for gidx in range(ep_start, ep_end):
            fd = frame_data[gidx]
            v_curr = fd["value"]
            true_return = fd["true_return"]
            next_gidx = gidx + action_horizon
            is_next_pad = next_gidx >= ep_end
            num_valid = min(action_horizon, ep_end - gidx)
            v_next = 0.0 if is_next_pad else frame_data[next_gidx]["value"]

            if abs(gamma - 1.0) < 1e-8:
                if is_next_pad:
                    reward_sum_raw = true_return
                else:
                    reward_sum_raw = true_return - frame_data[next_gidx]["true_return"]
            else:
                rewards = np.array(
                    [frame_data[gidx + k]["reward"] for k in range(num_valid)],
                    dtype=np.float64,
                )
                reward_sum_raw = float(np.sum(gamma_powers[:num_valid] * rewards))

            reward_sum = normalize(reward_sum_raw)
            gamma_k = gamma**num_valid if discount_next_value else 1.0
            advantage = reward_sum + gamma_k * v_next - v_curr

            rows.append(
                {
                    "episode_index": ep_idx,
                    "frame_index": fd["frame_index"],
                    "advantage_continuous": advantage,
                    "return": true_return,
                    "value_current": v_curr,
                    "value_next": v_next,
                    "reward_sum": reward_sum,
                    "reward_sum_raw": reward_sum_raw,
                    "num_valid_rewards": num_valid,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def _get_episode_viz_data(
    dataset,
    ep_idx: int,
    tasks: dict,
    image_keys: list[str],
    adv_df: pd.DataFrame,
) -> dict:
    """Gather images + advantage/value per frame for one episode (in-memory)."""
    start, end = _episode_range(dataset, ep_idx)
    indices = list(range(start, end))

    ep_adv = adv_df[adv_df["episode_index"] == ep_idx].sort_values("frame_index")
    adv_lookup = {int(r["frame_index"]): r for _, r in ep_adv.iterrows()}

    data = {
        "frames": [],
        "images": {k: [] for k in image_keys},
        "values": [],
        "advantages": [],
        "task": "",
        "episode_index": ep_idx,
    }

    for idx in tqdm(indices, desc=f"Episode {ep_idx}", leave=False):
        sample = dataset[idx]
        frame_idx = int(to_scalar(sample["frame_index"]))
        data["frames"].append(frame_idx)

        for key in image_keys:
            if key in sample:
                img = to_numpy(sample[key])
                if img.ndim == 4:
                    img = img[0]
                if img.dtype in (np.float32, np.float64):
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                data["images"][key].append(img)

        row = adv_lookup.get(frame_idx)
        if row is not None:
            data["advantages"].append(float(row["advantage_continuous"]))
            data["values"].append(float(row["value_current"]))
        else:
            data["advantages"].append(0.0)
            data["values"].append(0.0)

        if not data["task"]:
            if "task" in sample:
                data["task"] = str(to_scalar(sample["task"]))
            elif "task_index" in sample and tasks:
                t_idx = int(to_scalar(sample["task_index"]))
                data["task"] = tasks.get(t_idx, f"Task {t_idx}")

    return data


def _resolve_threshold(
    cfg: DictConfig,
    viz_cfg: DictConfig,
    ds_path: Path,
    adv_df: pd.DataFrame,
) -> float | None:
    """Pick threshold: explicit > mixture_config.yaml > local quantile of selected episodes."""
    explicit = viz_cfg.get("threshold", None)
    if explicit is not None:
        t = float(explicit)
        logger.info(f"Using explicit threshold: {t:.4f}")
        return t

    tag = cfg.advantage.get("tag", None)
    mixture_path = ds_path / "mixture_config.yaml"
    if mixture_path.exists():
        import yaml

        with open(mixture_path) as f:
            mix_cfg = yaml.safe_load(f) or {}
        if tag and "tags" in mix_cfg and tag in mix_cfg["tags"]:
            tc = mix_cfg["tags"][tag]
            if "unified_threshold" in tc:
                t = float(tc["unified_threshold"])
                logger.info(f"Using threshold from {mixture_path}[tags.{tag}]: {t:.4f}")
                return t
        if "unified_threshold" in mix_cfg:
            t = float(mix_cfg["unified_threshold"])
            logger.info(f"Using threshold from {mixture_path}: {t:.4f}")
            return t

    if len(adv_df) > 0:
        positive_quantile = cfg.advantage.get("positive_quantile", 0.3)
        t = float(
            np.percentile(
                adv_df["advantage_continuous"].values,
                (1 - positive_quantile) * 100,
            )
        )
        logger.warning(
            f"No stored threshold found; using {positive_quantile:.0%}-quantile "
            f"of selected episodes only: threshold={t:.4f} "
            "(local estimate, not dataset-wide)."
        )
        return t
    return None


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_advantages",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    device = cfg.advantage.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    cfg.advantage.device = device

    viz_cfg = cfg.get("visualize", OmegaConf.create({}))
    output_dir = Path(viz_cfg.get("output_dir", "./viz_episodes"))
    output_dir.mkdir(parents=True, exist_ok=True)
    no_video = bool(viz_cfg.get("no_video", False))
    fps = int(viz_cfg.get("fps", 10))
    ds_index = int(viz_cfg.get("dataset_index", 0))

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Output directory: {output_dir}")

    from rlinf.models.embodiment.value_model.modeling_critic import (
        ValueCriticModel,
    )

    value_model = ValueCriticModel.from_checkpoint(
        **_parse_value_model_kwargs(cfg), device=device
    )

    returns_tag = cfg.advantage.get("returns_tag", cfg.advantage.get("tag", None))
    ds_cfg = cfg.data.train_data_paths[ds_index]
    ds_path = Path(ds_cfg.dataset_path)
    logger.info(f"Dataset: {ds_path}")
    dataset, tasks, meta, returns_sidecar = load_lerobot_dataset(
        ds_path, returns_tag=returns_tag
    )

    global_return_min = cfg.data.get("return_min", None)
    global_return_max = cfg.data.get("return_max", None)
    if global_return_min is None or global_return_max is None:
        ds_min, ds_max = load_return_stats_from_dataset(ds_path)
        if global_return_min is None:
            global_return_min = ds_min
        if global_return_max is None:
            global_return_max = ds_max
    if global_return_min is None or global_return_max is None:
        raise ValueError(
            "Cannot determine return range. Set data.return_min/data.return_max "
            "or ensure meta/stats.json exists (run compute_returns.py first)."
        )
    logger.info(f"Return range: [{global_return_min}, {global_return_max}]")

    episode_indices = _select_episodes(meta.total_episodes, viz_cfg)
    preview = episode_indices[:20]
    logger.info(
        f"Rendering {len(episode_indices)}/{meta.total_episodes} episodes: "
        f"{preview}{'...' if len(episode_indices) > 20 else ''}"
    )

    episode_ranges = {ep: _episode_range(dataset, ep) for ep in episode_indices}
    frame_indices = sorted(
        {i for (s, e) in episode_ranges.values() for i in range(s, e)}
    )
    logger.info(f"Frames to infer: {len(frame_indices)}")

    robot_type = ds_cfg.get("robot_type", cfg.data.get("robot_type", "libero"))
    batch_size = int(cfg.advantage.get("batch_size", 64))
    num_workers = int(
        cfg.advantage.get(
            "num_dataloader_workers_per_gpu",
            cfg.advantage.get("num_dataloader_workers", 4),
        )
    )

    frame_data = _compute_value_for_indices(
        value_model,
        dataset,
        tasks,
        robot_type,
        frame_indices,
        batch_size,
        num_workers,
        returns_sidecar,
    )

    gamma = float(cfg.data.gamma)
    action_horizon = int(cfg.data.advantage_lookahead_step)
    discount_next_value = bool(cfg.advantage.get("discount_next_value", True))
    adv_df = _compute_advantages_for_episodes(
        frame_data,
        episode_ranges,
        gamma,
        action_horizon,
        discount_next_value,
        float(global_return_min),
        float(global_return_max),
    )
    logger.info(
        f"Computed advantages for {len(adv_df)} frames across "
        f"{len(episode_indices)} episodes."
    )
    if len(adv_df) > 0:
        a = adv_df["advantage_continuous"].values
        logger.info(
            f"  Advantage: mean={a.mean():.4f} std={a.std():.4f} "
            f"min={a.min():.4f} max={a.max():.4f}"
        )

    threshold = _resolve_threshold(cfg, viz_cfg, ds_path, adv_df)

    del value_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    first_ep_start = episode_ranges[episode_indices[0]][0]
    image_keys = detect_image_keys(dataset[first_ep_start])
    logger.info(f"Image keys: {image_keys}")
    if not image_keys:
        logger.warning(
            "No image keys detected — videos/plots will be empty. "
            "Check the dataset schema."
        )

    for ep in tqdm(episode_indices, desc="Rendering"):
        ep_data = _get_episode_viz_data(dataset, ep, tasks, image_keys, adv_df)
        summary_path = output_dir / f"episode_{ep:04d}_summary.png"
        create_episode_summary_plot(ep_data, summary_path, threshold=threshold)
        logger.info(f"  Wrote {summary_path.name}")
        if not no_video:
            video_path = output_dir / f"episode_{ep:04d}.mp4"
            create_episode_video(ep_data, video_path, threshold=threshold, fps=fps)
            logger.info(f"  Wrote {video_path.name}")

    slim_parquet = output_dir / "advantages_selected.parquet"
    adv_df.to_parquet(slim_parquet, index=False)
    logger.info(f"Saved slim advantages parquet: {slim_parquet}")
    logger.info(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    main()
