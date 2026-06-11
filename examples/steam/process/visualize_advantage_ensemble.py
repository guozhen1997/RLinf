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

"""Visualise the ensemble binary value model's per-frame advantage parquet.

Reads ``meta/advantages_{tag}.parquet`` produced by
``compute_advantages_ensemble.py`` and writes:

  Global plots (always written):
    1. ``distribution.png``                — aggregated signed-progress histogram
    2. ``members.png``                     — per-member signed-progress overlay
    3. ``uncertainty.png``                 — mean vs variance scatter
    4. ``positive_rate_per_episode.png``   — per-episode positive fraction
    5. ``timeline_episodes.png``           — 3x3 episode timeline grid

  Per-episode renders (when ``--num-episodes > 0``):
    6. ``episodes/episode_<NNNN>_summary.png`` — sampled-frame summary
       (5 thumbnails per camera + member curves + variance + cumulative
       progress) — same layout as
       ``logs/viz_binary/pnp_eval_ckpt8000_k8_ensemble4_10eps``.
    7. ``episodes/episode_<NNNN>.mp4``      — animated playback of every
       frame with a moving cursor on the score plot (omit with ``--no-video``).

This script does NOT load the value model and does NOT recompute anything
— all per-episode scores come from the parquet, frames come from the
LeRobot dataset on disk.

Usage:
    python visualize_advantage_ensemble.py \\
        --dataset /path/to/lerobot_dataset \\
        --tag fail150_k8_ensemble4_wco \\
        --output /path/to/visualization_dir \\
        --num-episodes 10
"""

import argparse
import json
import os
import sys
from pathlib import Path


def _silence_libav_logs() -> None:
    """Suppress the ``[libdav1d @ 0x..] libdav1d 0.9.2`` chatter that
    torchcodec (LeRobot's default video backend) emits every frame.

    The messages are written by libav* **directly to file-descriptor 2**
    — pyav's log level does not intercept them. We splice our own
    ``fd=2`` through a long-lived ``grep -v '\\[libdav1d'`` subprocess
    so every write to stderr is filtered line-by-line; all other
    stderr output (our own logs, tracebacks, etc.) still reaches the
    terminal.
    """
    import atexit
    import shutil
    import subprocess

    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    if not shutil.which("grep"):
        return
    saved_stderr_fd = os.dup(2)
    try:
        grep = subprocess.Popen(
            ["grep", "-v", "-E", "--line-buffered", r"libdav1d|libdav1d 0\.9"],
            stdin=subprocess.PIPE,
            stdout=saved_stderr_fd,
        )
    except Exception:
        os.close(saved_stderr_fd)
        return

    sys.stderr.flush()
    os.dup2(grep.stdin.fileno(), 2)

    def _restore_and_drain() -> None:
        # Restore fd=2 first so grep sees EOF on its stdin (otherwise the
        # pipe stays open via our fd=2 and ``grep.wait()`` deadlocks).
        try:
            os.dup2(saved_stderr_fd, 2)
        except OSError:
            pass
        try:
            os.close(saved_stderr_fd)
        except OSError:
            pass
        try:
            grep.stdin.close()
        except Exception:
            pass
        try:
            grep.wait(timeout=3)
        except Exception:
            grep.kill()

    atexit.register(_restore_and_drain)


_silence_libav_logs()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the rlinf package importable regardless of cwd, and let us pull in
# the per-episode rendering helpers from the existing binary-value
# visualization script (those helpers are pure plotting — no model load).
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent.parent))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_advantages_parquet(dataset_path: Path, tag: str) -> tuple[pd.DataFrame, Path]:
    parquet_path = dataset_path / "meta" / f"advantages_{tag}.parquet"
    if not parquet_path.exists():
        available = sorted(p.name for p in (dataset_path / "meta").glob("advantages*.parquet"))
        raise FileNotFoundError(
            f"Advantage parquet not found: {parquet_path}\n"
            f"Available: {available}"
        )
    df = pd.read_parquet(parquet_path)
    required_cols = {
        "episode_index",
        "frame_index",
        "advantage",
        "advantage_continuous",
        "p_progress_mean",
        "p_progress_min",
        "p_progress_variance",
        "member_values",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet {parquet_path} missing required columns: {sorted(missing)}"
        )
    return df, parquet_path


def load_threshold_from_mixture(dataset_path: Path, tag: str) -> float | None:
    """Read positive_threshold from meta/mixture_config.yaml if present."""
    cfg_path = dataset_path / "meta" / "mixture_config.yaml"
    if not cfg_path.exists():
        return None
    import yaml

    with open(cfg_path, "r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return None
    tag_entry = (loaded.get("tags") or {}).get(tag)
    if not isinstance(tag_entry, dict):
        return None
    th = tag_entry.get("positive_threshold")
    return float(th) if th is not None else None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _stack_member_values(df: pd.DataFrame) -> np.ndarray:
    """Return a [K, N] array of per-member signed-progress values in [-1, 1]."""
    rows = [np.asarray(row, dtype=np.float32) for row in df["member_values"].tolist()]
    if not rows:
        raise RuntimeError("Empty parquet — nothing to plot")
    K = rows[0].shape[0]
    if any(r.shape[0] != K for r in rows):
        raise RuntimeError(
            "member_values rows have inconsistent length — every row must have "
            "the same ensemble size"
        )
    return np.stack(rows, axis=1)  # [K, N]


def _data_driven_xrange(values: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    """Return ``(lo, hi)`` for histogram / axis bounds. Pads away from data
    limits so the extreme bins remain visible, and never clips corrected
    values that fall below ``-1``."""
    if values.size == 0:
        return -1.05, 1.05
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def plot_distribution(
    df: pd.DataFrame, out_path: Path, threshold: float | None, dataset_name: str, tag: str,
    *, tag_meta: dict | None = None,
) -> None:
    tag_meta = tag_meta or {}

    fig, ax = plt.subplots(figsize=(8, 5))
    adv = df["advantage_continuous"].to_numpy()
    xlo, xhi = _data_driven_xrange(adv)
    ax.hist(
        adv,
        bins=80,
        range=(xlo, xhi),
        color="steelblue",
        edgecolor="black",
        alpha=0.75,
        label="advantage_continuous",
    )
    ax.axvline(
        x=float(np.mean(adv)),
        color="green",
        linestyle="-",
        linewidth=1.5,
        label=f"mean = {np.mean(adv):.4f}",
    )
    if threshold is not None:
        n_pos = int((adv > threshold).sum())
        ax.axvline(
            x=threshold,
            color="orange",
            linestyle="--",
            linewidth=2.0,
            label=f"threshold = {threshold:.3f}  (positive: {n_pos}/{len(adv)})",
        )
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("advantage_continuous (aggregated signed progress)")
    ax.set_ylabel("count")
    ax.set_title(f"Advantage distribution — {dataset_name}\ntag = {tag}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_member_distributions(df: pd.DataFrame, out_path: Path, dataset_name: str, tag: str) -> None:
    members = _stack_member_values(df)  # [K, N]
    K = members.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 1.0, 80)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(K, 3)))
    for k in range(K):
        ax.hist(
            members[k],
            bins=bins,
            histtype="step",
            linewidth=1.7,
            color=colors[k],
            label=f"member {k}  (mean={members[k].mean():.3f})",
        )
    ax.set_xlabel("signed progress")
    ax.set_ylabel("count")
    ax.set_title(
        f"Per-member signed-progress distributions — {dataset_name}\n"
        f"tag = {tag} (K = {K})"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_uncertainty_scatter(
    df: pd.DataFrame, out_path: Path, threshold: float | None, dataset_name: str, tag: str
) -> None:
    mean = df["p_progress_mean"].to_numpy()
    var = df["p_progress_variance"].to_numpy()
    pos = df["advantage"].to_numpy().astype(bool)

    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(
        mean[~pos], var[~pos], s=4, alpha=0.5, color="tab:red", label="advantage = False"
    )
    ax_main.scatter(
        mean[pos], var[pos], s=4, alpha=0.5, color="tab:green", label="advantage = True"
    )
    if threshold is not None:
        ax_main.axvline(threshold, color="orange", linestyle="--", linewidth=1.4)
        ax_top.axvline(threshold, color="orange", linestyle="--", linewidth=1.4)
    ax_main.set_xlim(-1.0, 1.0)
    ax_main.set_ylim(0.0, max(float(var.max()) * 1.05, 1e-4))
    ax_main.set_xlabel("ensemble mean signed progress")
    ax_main.set_ylabel("ensemble variance of signed progress")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    ax_top.hist(mean, bins=80, range=(-1.0, 1.0), color="steelblue", alpha=0.75)
    ax_top.set_ylabel("count")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, alpha=0.3)

    ax_right.hist(
        var, bins=60, orientation="horizontal", color="indianred", alpha=0.75
    )
    ax_right.set_xlabel("count")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(True, alpha=0.3)

    fig.suptitle(
        f"Ensemble disagreement vs mean — {dataset_name}\ntag = {tag}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_positive_rate_per_episode(
    df: pd.DataFrame, out_path: Path, threshold: float | None, dataset_name: str, tag: str
) -> None:
    pos_per_ep = df.groupby("episode_index")["advantage"].mean().sort_index()
    counts_per_ep = df.groupby("episode_index").size()
    n_eps = len(pos_per_ep)
    fig, ax = plt.subplots(figsize=(max(8, n_eps * 0.18), 5))
    ax.bar(
        pos_per_ep.index,
        pos_per_ep.values,
        color=["tab:green" if v > 0.5 else "tab:red" for v in pos_per_ep.values],
        alpha=0.85,
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode_index")
    ax.set_ylabel("fraction of frames with advantage = True")
    ax.set_title(
        f"Positive rate per episode (threshold = "
        f"{threshold:.3f}) — {dataset_name}\ntag = {tag}"
        if threshold is not None
        else f"Positive rate per episode — {dataset_name}\ntag = {tag}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    # annotate frame count for short episodes (the rest get crowded)
    if n_eps <= 60:
        for ep, rate in zip(pos_per_ep.index, pos_per_ep.values):
            ax.text(
                ep,
                rate + 0.02,
                str(int(counts_per_ep.loc[ep])),
                ha="center",
                va="bottom",
                fontsize=7,
                color="dimgray",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _pick_representative_episodes(df: pd.DataFrame, n: int) -> list[int]:
    """Pick episodes that span the spectrum of average ensemble disagreement.

    Sorts episodes by mean variance and takes evenly-spaced quantile picks so
    the grid shows both confident and uncertain trajectories.
    """
    var_per_ep = (
        df.groupby("episode_index")["p_progress_variance"].mean().sort_values()
    )
    if len(var_per_ep) <= n:
        return [int(ep) for ep in var_per_ep.index.tolist()]
    quantile_idx = np.linspace(0, len(var_per_ep) - 1, n).round().astype(int)
    return [int(var_per_ep.index[i]) for i in quantile_idx]


def plot_episode_timelines(
    df: pd.DataFrame,
    out_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
    n_episodes: int = 9,
) -> None:
    selected = _pick_representative_episodes(df, n_episodes)
    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.0), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax_idx, ep in enumerate(selected):
        ax = axes.flat[ax_idx]
        ax.axis("on")
        sub = df[df["episode_index"] == ep].sort_values("frame_index")
        x = sub["frame_index"].to_numpy()
        members = np.stack(
            [np.asarray(r, dtype=np.float32) for r in sub["member_values"].tolist()],
            axis=1,
        )  # [K, T]
        member_min = members.min(axis=0)
        member_max = members.max(axis=0)
        mean = sub["p_progress_mean"].to_numpy()
        agg = sub["advantage_continuous"].to_numpy()
        ax.fill_between(
            x, member_min, member_max, color="steelblue", alpha=0.18, label="member min/max"
        )
        ax.plot(x, mean, color="steelblue", linewidth=1.2, label="member mean")
        ax.plot(
            x,
            agg,
            color="tab:purple",
            linewidth=1.5,
            linestyle="--",
            label="aggregated (wco)",
        )
        if threshold is not None:
            ax.axhline(threshold, color="orange", linestyle=":", linewidth=1.0)
        pos_rate = float(sub["advantage"].mean())
        ax.set_title(
            f"episode {ep}  (T={len(sub)}, pos_rate={pos_rate:.2f})", fontsize=10
        )
        ax.set_xlabel("frame_index")
        ax.set_ylabel("signed progress / advantage_continuous")
        # Data-driven y-range so out-of-range values are not clipped.
        y_lo, y_hi = _data_driven_xrange(
            np.concatenate([member_min, member_max, agg]), pad=0.02
        )
        # Never shrink below the native [-1, 1] envelope so uncorrected
        # parquets keep their familiar view.
        ax.set_ylim(min(y_lo, -1.02), max(y_hi, 1.02))
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"Episode timelines (sorted by mean ensemble variance) — {dataset_name}\n"
        f"tag = {tag}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------


def write_summary(
    df: pd.DataFrame,
    parquet_path: Path,
    threshold: float | None,
    dataset_name: str,
    tag: str,
    out_path: Path,
) -> None:
    members = _stack_member_values(df)  # [K, N]
    summary: dict = {
        "dataset": dataset_name,
        "tag": tag,
        "parquet": str(parquet_path),
        "positive_threshold": threshold,
        "ensemble_size": int(members.shape[0]),
        "total_anchors": int(len(df)),
        "num_episodes": int(df["episode_index"].nunique()),
        "num_positive": int(df["advantage"].sum()),
        "positive_rate": float(df["advantage"].mean()),
        "advantage_continuous": {
            "mean": float(df["advantage_continuous"].mean()),
            "std": float(df["advantage_continuous"].std()),
            "min": float(df["advantage_continuous"].min()),
            "max": float(df["advantage_continuous"].max()),
        },
        "p_progress_mean": {
            "mean": float(df["p_progress_mean"].mean()),
            "std": float(df["p_progress_mean"].std()),
        },
        "p_progress_variance": {
            "mean": float(df["p_progress_variance"].mean()),
            "max": float(df["p_progress_variance"].max()),
        },
        "per_member_mean_p_progress": [float(members[k].mean()) for k in range(members.shape[0])],
    }
    if "ensemble_signed_score" in df.columns:
        summary["ensemble_signed_score"] = {
            "mean": float(df["ensemble_signed_score"].mean()),
            "std": float(df["ensemble_signed_score"].std()),
            "min": float(df["ensemble_signed_score"].min()),
            "max": float(df["ensemble_signed_score"].max()),
        }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Per-episode rendering (sampled frames + plots + optional video)
# ---------------------------------------------------------------------------


def _build_progress_by_frame_from_df(df_for_episode: pd.DataFrame) -> dict:
    """Convert per-episode parquet rows into the score-bundle dict expected
    by ``_collect_episode_frames``.

    Field semantics:
      * ``value`` — decision score used for threshold highlight in the shared
        render path. Always equals ``advantage_continuous``.
      * ``raw_score`` — raw ``ensemble_signed_score`` when available; falls
        back to ``advantage_continuous`` for legacy parquets where the two
        are equal.

    Frames missing from the parquet (e.g. terminal frames that had no valid
    ``t+k`` pair pre-terminal-fill) are handled by the collector via fallback
    to the previous value.
    """
    has_ess = "ensemble_signed_score" in df_for_episode.columns

    out: dict[int, dict[str, object]] = {}
    for row in df_for_episode.itertuples(index=False):
        entry: dict[str, object] = {
            "value": float(row.advantage_continuous),
            "raw_score": float(
                getattr(row, "ensemble_signed_score", row.advantage_continuous)
            ) if has_ess else float(row.advantage_continuous),
            "value_mean": float(row.p_progress_mean),
            "value_min": float(row.p_progress_min),
            "value_variance": float(row.p_progress_variance),
            "member_values": [float(x) for x in row.member_values],
        }
        out[int(row.frame_index)] = entry
    return out


def _select_episode_indices(
    df: pd.DataFrame, n: int, strategy: str, seed: int, eligible: set[int]
) -> list[int]:
    """Pick ``n`` episodes from those present in ``df`` and ``eligible``.

    Strategies:
      * ``random``  — random sample, fixed seed.
      * ``variance`` — episodes with the highest mean ensemble variance
                       (most uncertain trajectories).
      * ``positive`` — episodes with the highest positive-rate.
      * ``negative`` — episodes with the lowest positive-rate.
    """
    pool = sorted(set(df["episode_index"].unique().tolist()) & eligible)
    if not pool:
        raise RuntimeError(
            "No episodes are both present in the parquet AND in the "
            "LeRobot dataset's episode index."
        )
    if n <= 0 or n >= len(pool):
        return pool

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(pool, n, replace=False).tolist())
    if strategy == "variance":
        var_per_ep = (
            df[df["episode_index"].isin(pool)]
            .groupby("episode_index")["p_progress_variance"]
            .mean()
            .sort_values(ascending=False)
        )
        return sorted(var_per_ep.head(n).index.astype(int).tolist())
    if strategy == "positive":
        pos_per_ep = (
            df[df["episode_index"].isin(pool)]
            .groupby("episode_index")["advantage"]
            .mean()
            .sort_values(ascending=False)
        )
        return sorted(pos_per_ep.head(n).index.astype(int).tolist())
    if strategy == "negative":
        pos_per_ep = (
            df[df["episode_index"].isin(pool)]
            .groupby("episode_index")["advantage"]
            .mean()
            .sort_values(ascending=True)
        )
        return sorted(pos_per_ep.head(n).index.astype(int).tolist())
    raise ValueError(
        f"Unknown --episode-strategy: {strategy!r}. "
        "Expected random / variance / positive / negative."
    )


def _open_lerobot_dataset(dataset_path: Path):
    """Open a LeRobot dataset, tolerant of both lerobot.common.datasets and
    lerobot.common.datasets layouts. Mirrors _LeRobotSource."""
    try:
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDataset,
        )
        from lerobot.common.datasets.utils import hf_transform_to_torch  # type: ignore
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import (  # type: ignore
            LeRobotDataset,
        )
        from lerobot.common.datasets.utils import hf_transform_to_torch  # type: ignore
    import io
    from PIL import Image as PILImage

    ds = LeRobotDataset(dataset_path.name, root=dataset_path, download_videos=False)
    ep_data_index = ds.episode_data_index
    ep_starts = [int(x) for x in ep_data_index["from"].tolist()]
    ep_ends = [int(x) for x in ep_data_index["to"].tolist()]

    def _decode(batch: dict) -> dict:
        for key in list(batch.keys()):
            vals = batch[key]
            if vals and isinstance(vals[0], dict) and "bytes" in vals[0]:
                batch[key] = [PILImage.open(io.BytesIO(v["bytes"])) for v in vals]
        return hf_transform_to_torch(batch)

    ds.hf_dataset.set_transform(_decode)

    tasks: dict[int, str] = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tasks[int(d.get("task_index", len(tasks)))] = str(d.get("task", ""))
    return ds, ep_starts, ep_ends, tasks


def render_per_episode(
    df: pd.DataFrame,
    dataset_path: Path,
    output_dir: Path,
    threshold: float,
    inference_mode: str,
    stride_k: int,
    num_episodes: int,
    strategy: str,
    seed: int,
    write_video: bool,
    fps: int,
    explicit_episodes: list[int] | None = None,
) -> list[Path]:
    """Render per-episode summary PNGs (and optionally MP4s) into output_dir.

    Reuses the rendering helpers from
    ``visualize_episodes_with_steam.py`` so the output layout
    matches ``logs/viz_binary/pnp_eval_ckpt8000_k8_ensemble4_10eps``.

    If ``explicit_episodes`` is non-empty it bypasses the strategy-based
    selection entirely — every listed episode present in both the parquet
    and the LeRobot dataset is rendered (missing ones are warned and
    skipped).
    """
    from visualize_episodes_with_steam import (
        _collect_episode_frames,
        _create_episode_summary_plot,
        _create_episode_video,
        detect_image_keys,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening LeRobot dataset {dataset_path}")
    lerobot_ds, ep_starts, ep_ends, tasks = _open_lerobot_dataset(dataset_path)
    eligible = set(range(len(ep_starts)))
    if explicit_episodes:
        requested = [int(e) for e in explicit_episodes]
        parquet_eps = set(df["episode_index"].astype(int).tolist())
        selected = [e for e in requested if e in eligible and e in parquet_eps]
        missing = sorted(set(requested) - set(selected))
        if missing:
            print(
                f"  warning: {len(missing)} episodes in --explicit-episodes are "
                f"not in both parquet and LeRobot dataset; skipping: {missing}"
            )
        if not selected:
            raise RuntimeError(
                "None of the explicit episodes are present in both parquet "
                f"and dataset. requested={requested}"
            )
        print(f"Selected episodes (explicit): {selected}")
    else:
        selected = _select_episode_indices(df, num_episodes, strategy, seed, eligible)
        print(f"Selected episodes ({strategy}): {selected}")

    image_keys = detect_image_keys(lerobot_ds[int(ep_starts[selected[0]])])
    print(f"Detected image keys: {image_keys}")
    if not image_keys:
        raise RuntimeError(
            "No image-like keys detected in the LeRobot sample — cannot render."
        )

    written: list[Path] = []
    from tqdm import tqdm as _tqdm

    for ep in _tqdm(selected, desc="rendering episodes"):
        ep_df = df[df["episode_index"] == int(ep)]
        if ep_df.empty:
            print(f"  ep {ep}: no parquet rows; skipping")
            continue
        progress_by_frame = _build_progress_by_frame_from_df(ep_df)
        ep_data = _collect_episode_frames(
            lerobot_ds=lerobot_ds,
            episode_index=int(ep),
            ep_starts=ep_starts,
            ep_ends=ep_ends,
            tasks=tasks,
            image_keys=image_keys,
            progress_by_frame=progress_by_frame,
        )

        summary_path = output_dir / f"episode_{int(ep):04d}_summary.png"
        _create_episode_summary_plot(
            ep_data,
            summary_path,
            decision_threshold=threshold,
            stride_k=stride_k,
            inference_mode=inference_mode,
        )
        written.append(summary_path)

        if write_video:
            video_path = output_dir / f"episode_{int(ep):04d}.mp4"
            _create_episode_video(
                ep_data,
                video_path,
                decision_threshold=threshold,
                stride_k=stride_k,
                inference_mode=inference_mode,
                fps=fps,
            )
            written.append(video_path)

    return written


def _read_mixture_meta(dataset_path: Path, tag: str) -> dict:
    """Pull tag-level metadata (inference_mode, ensemble_size, threshold)
    from meta/mixture_config.yaml."""
    cfg_path = dataset_path / "meta" / "mixture_config.yaml"
    if not cfg_path.exists():
        return {}
    import yaml

    with open(cfg_path) as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        return {}
    tag_entry = (loaded.get("tags") or {}).get(tag)
    return tag_entry if isinstance(tag_entry, dict) else {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="LeRobot dataset path (the dir that contains meta/advantages_{tag}.parquet).",
    )
    parser.add_argument("--tag", type=str, required=True, help="Advantage tag.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for the visualisation PNGs.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override positive_threshold. Defaults to the value recorded in "
            "meta/mixture_config.yaml under tags.<tag>.positive_threshold."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=9,
        help="Number of episodes in the timeline grid (default: 9, fits 3x3).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help=(
            "Number of episodes to render with sampled-frame summaries + "
            "videos (parallel to logs/viz_binary/...). Set 0 to skip."
        ),
    )
    parser.add_argument(
        "--episode-strategy",
        choices=("random", "variance", "positive", "negative"),
        default="random",
        help=(
            "How to pick the episodes to render. random (default) matches "
            "the reference 10-eps directory; variance picks the most "
            "uncertain trajectories."
        ),
    )
    parser.add_argument(
        "--episode-seed", type=int, default=42, help="RNG seed for random strategy."
    )
    parser.add_argument(
        "--explicit-episodes",
        type=str,
        default=None,
        help=(
            "Comma-separated explicit episode ids (e.g. '7,22,29,66'). When "
            "provided, --num-episodes / --episode-strategy / --episode-seed "
            "are ignored and every listed episode present in both parquet "
            "and dataset is rendered."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 generation (PNG summaries still produced).",
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS for episode videos.")
    parser.add_argument(
        "--stride-k",
        type=int,
        default=8,
        help=(
            "Pair stride k used at compute time. Drives the title/legend on "
            "per-episode plots — must match the value used by "
            "compute_advantages_ensemble.py."
        ),
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)

    df, parquet_path = load_advantages_parquet(args.dataset, args.tag)
    threshold = (
        args.threshold
        if args.threshold is not None
        else load_threshold_from_mixture(args.dataset, args.tag)
    )
    if threshold is None:
        raise ValueError(
            f"positive_threshold not provided and not found under "
            f"tags.{args.tag} in meta/mixture_config.yaml"
        )

    dataset_name = args.dataset.name
    print(f"Loaded {len(df)} anchors from {parquet_path}")
    print(f"Threshold = {threshold}")
    print(f"Output dir = {args.output}")

    tag_meta = _read_mixture_meta(args.dataset, args.tag)
    plot_distribution(
        df,
        args.output / "distribution.png",
        threshold,
        dataset_name,
        args.tag,
        tag_meta=tag_meta,
    )
    plot_member_distributions(
        df, args.output / "members.png", dataset_name, args.tag
    )
    plot_uncertainty_scatter(
        df, args.output / "uncertainty.png", threshold, dataset_name, args.tag
    )
    plot_positive_rate_per_episode(
        df,
        args.output / "positive_rate_per_episode.png",
        threshold,
        dataset_name,
        args.tag,
    )
    plot_episode_timelines(
        df,
        args.output / "timeline_episodes.png",
        threshold,
        dataset_name,
        args.tag,
        n_episodes=args.episodes,
    )
    write_summary(
        df, parquet_path, threshold, dataset_name, args.tag, args.output / "summary.json"
    )

    explicit_eps = None
    if args.explicit_episodes:
        try:
            explicit_eps = [
                int(x) for x in args.explicit_episodes.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise SystemExit(
                f"--explicit-episodes must be comma-separated integers, "
                f"got {args.explicit_episodes!r} ({exc})"
            )
        if not explicit_eps:
            raise SystemExit("--explicit-episodes resolved to an empty list")

    run_per_episode = args.num_episodes > 0 or bool(explicit_eps)
    if run_per_episode:
        meta = _read_mixture_meta(args.dataset, args.tag)
        inference_mode = str(meta.get("inference_mode", "wco"))
        episodes_dir = args.output / "episodes"
        render_per_episode(
            df=df,
            dataset_path=args.dataset,
            output_dir=episodes_dir,
            threshold=threshold,
            inference_mode=inference_mode,
            stride_k=int(args.stride_k),
            num_episodes=int(args.num_episodes),
            strategy=str(args.episode_strategy),
            seed=int(args.episode_seed),
            write_video=not args.no_video,
            fps=int(args.fps),
            explicit_episodes=explicit_eps,
        )

    print("Wrote:")
    for p in sorted(args.output.rglob("*")):
        if p.is_file():
            print(f"  {p}")


if __name__ == "__main__":
    main()
