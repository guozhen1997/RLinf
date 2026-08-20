#!/usr/bin/env python3
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

"""Convert a realworld-pni LeRobot v2.1 dump into a GR00T N1.7 dataset.

The source tree (RLinf ``LeRobotDatasetWriter`` / OpenPI realworld collection)
looks like this::

    realworld-pni/
      data/chunk-000/episode_XXXXXX.parquet   # images stored IN parquet
      meta/info.json, episodes.jsonl, tasks.jsonl
      norm_stats.json                         # OpenPI padded stats, unused here

GR00T's ``LeRobotEpisodeLoader`` instead needs::

    - videos as H.264 mp4 (not ``dtype: image`` parquet columns)
    - ``meta/modality.json`` mapping video / state / action / language
    - ``meta/stats.json`` keyed by parquet columns
    - standard column names ``observation.state`` and ``action``

This script writes a **new** dataset directory; the source is never modified.

Example (10-episode sample or the full ~250-episode dump)::

    python toolkits/lerobot/convert_realworld_pni_to_gr00t.py convert \\
        --src /path/to/realworld-pni \\
        --dst /path/to/realworld-pni-gr00t

    # Optional: merge new_embodiment keys into a GR00T-N1.7 processor copy
    python toolkits/lerobot/convert_realworld_pni_to_gr00t.py patch-processor \\
        --dataset /path/to/realworld-pni-gr00t \\
        --model-path /path/to/GR00T-N1.7 \\
        --output /path/to/GR00T-N1.7-realworld-pni

    python toolkits/lerobot/convert_realworld_pni_to_gr00t.py validate \\
        --dataset /path/to/realworld-pni-gr00t

Dependencies: ``pyarrow``, ``numpy``, ``pillow``. Video encoding prefers a
system ``ffmpeg`` with ``libx264``; ``imageio-ffmpeg`` or OpenCV are fallbacks.

State layout (19-D) matches ``realworld_env._wrap_obs``: keys concatenated in
**alphabetical** order after ``Quat2EulerWrapper``:

    gripper_position[1] | tcp_force[3] | tcp_pose xyz+euler[6]
    | tcp_torque[3] | tcp_vel[6]

Action layout (7-D): spacemouse xyz[3] + rpy[3] + gripper[1].

Cameras: ``image`` is the main view (``main_image_key: wrist_1``),
``extra_view_image`` is the third-person / exterior view.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

COLUMN_RENAME = {
    "state": "observation.state",
    "actions": "action",
}

# Source parquet image column -> GR00T video original_key (also the mp4 folder).
DEFAULT_IMAGE_MAP = {
    "image": "observation.images.image",
    "extra_view_image": "observation.images.extra_view_image",
}

# Slices into the 19-D flattened proprio (see module docstring).
STATE_LAYOUT: dict[str, tuple[int, int]] = {
    "gripper_position": (0, 1),
    "tcp_force": (1, 4),
    "tcp_pose": (4, 10),
    "tcp_torque": (10, 13),
    "tcp_vel": (13, 19),
}

ACTION_LAYOUT: dict[str, tuple[int, int]] = {
    "xyz": (0, 3),
    "rpy": (3, 6),
    "gripper": (6, 7),
}

PROCESSOR_REQUIRED = (
    "processor_config.json",
    "statistics.json",
    "embodiment_id.json",
)

STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "This script needs pyarrow. Install with: pip install pyarrow numpy pillow"
        ) from exc


def _episode_relpath(info: dict[str, Any], episode_index: int) -> str:
    chunk_size = int(info.get("chunks_size") or 1000)
    tmpl = str(info.get("data_path") or DATA_PATH_TEMPLATE)
    return tmpl.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _video_relpath(
    episode_index: int, video_key: str, chunk_size: int = 1000
) -> str:
    return VIDEO_PATH_TEMPLATE.format(
        episode_chunk=episode_index // chunk_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def discover_episode_indices(src: Path, info: dict[str, Any]) -> list[int]:
    """Return episode indices that actually have a parquet file on disk.

    ``meta/info.json`` / ``episodes.jsonl`` may list the full 260-episode
    corpus while this checkout only contains a 10-episode sample.
    """
    found: list[int] = []
    claimed = int(info.get("total_episodes") or 0)
    for episode_index in range(max(claimed, 0)):
        if (src / _episode_relpath(info, episode_index)).is_file():
            found.append(episode_index)

    if found:
        return found

    # Fallback: glob, in case total_episodes is 0 / wrong.
    for path in sorted((src / "data").glob("chunk-*/episode_*.parquet")):
        stem = path.stem  # episode_000012
        try:
            found.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(found))


def _is_image_struct(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "bytes" in value
        and "path" in value
    )


def decode_image_value(value: Any, dataset_root: Path) -> np.ndarray:
    """Decode one LeRobot image cell to uint8 RGB ``(H, W, 3)``."""
    from PIL import Image

    raw_bytes: Optional[bytes] = None
    file_path: Optional[Path] = None

    if _is_image_struct(value):
        raw = value.get("bytes")
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        if isinstance(raw, (bytes, bytearray)) and raw:
            raw_bytes = bytes(raw)
        path_str = value.get("path")
        if path_str:
            candidate = Path(path_str)
            file_path = (
                candidate if candidate.is_absolute() else dataset_root / candidate
            )
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(value)
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 3:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.transpose(array, (1, 2, 0))
            return np.ascontiguousarray(array)
    elif isinstance(value, list):
        array = np.asarray(value)
        if array.ndim == 3:
            return decode_image_value(array, dataset_root)

    if raw_bytes:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return np.array(image.convert("RGB"))
    if file_path is not None and file_path.is_file():
        with Image.open(file_path) as image:
            return np.array(image.convert("RGB"))
    raise ValueError(f"Cannot decode image cell of type {type(value)!r}")


def find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_h264_video(frames: list[np.ndarray], output_path: Path, fps: int) -> str:
    """Write RGB frames to an H.264 (preferred) or MPEG-4 mp4. Returns codec used."""
    if not frames:
        raise ValueError(f"No frames to encode for {output_path}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    # yuv420p needs even dimensions.
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        raise ValueError(f"Frame size too small: {frames[0].shape}")

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        try:
            for frame in frames:
                rgb = np.ascontiguousarray(frame[:height, :width], dtype=np.uint8)
                proc.stdin.write(rgb.tobytes())
        finally:
            proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        code = proc.wait()
        if code == 0:
            return "h264"
        raise RuntimeError(
            f"ffmpeg failed for {output_path} (exit {code}): "
            f"{stderr.decode('utf-8', errors='replace')[-2000:]}"
        )

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "No ffmpeg on PATH and OpenCV is not installed. "
            "Install ffmpeg (recommended) or `pip install opencv-python-headless`."
        ) from exc

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open {output_path}")
    for frame in frames:
        bgr = cv2.cvtColor(frame[:height, :width], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(
        f"warning: encoded {output_path.name} with OpenCV mp4v (mpeg4). "
        "GR00T torchcodec prefers H.264; re-encode with ffmpeg if decode fails.",
        file=sys.stderr,
    )
    return "mpeg4"


def _feature_image_keys(info: dict[str, Any]) -> list[str]:
    keys = []
    for name, spec in (info.get("features") or {}).items():
        if isinstance(spec, dict) and spec.get("dtype") == "image":
            keys.append(name)
    return keys


def _build_modality_json(
    image_map: dict[str, str],
    state_key: str = "observation.state",
    action_key: str = "action",
) -> dict[str, Any]:
    video = {}
    for original_key in image_map.values():
        short = original_key.split("observation.images.", 1)[-1]
        video[short] = {"original_key": original_key}
    return {
        "state": {
            name: {"original_key": state_key, "start": start, "end": end}
            for name, (start, end) in STATE_LAYOUT.items()
        },
        "action": {
            name: {"original_key": action_key, "start": start, "end": end}
            for name, (start, end) in ACTION_LAYOUT.items()
        },
        "video": video,
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }


def _absolute_action_configs(n: int) -> list[dict[str, Any]]:
    return [
        {
            "rep": "ABSOLUTE",
            "type": "NON_EEF",
            "format": "DEFAULT",
            "state_key": None,
        }
        for _ in range(n)
    ]


def _build_processor_modality_config(
    image_map: dict[str, str], action_horizon: int
) -> dict[str, Any]:
    video_keys = [
        original_key.split("observation.images.", 1)[-1]
        for original_key in image_map.values()
    ]
    state_keys = list(STATE_LAYOUT.keys())
    action_keys = list(ACTION_LAYOUT.keys())
    return {
        "video": {
            "delta_indices": [0],
            "modality_keys": video_keys,
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": None,
        },
        "state": {
            "delta_indices": [0],
            "modality_keys": state_keys,
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": None,
        },
        "action": {
            "delta_indices": list(range(int(action_horizon))),
            "modality_keys": action_keys,
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": _absolute_action_configs(len(action_keys)),
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.action.task_description"],
            "sin_cos_embedding_keys": None,
            "mean_std_embedding_keys": None,
            "action_configs": None,
        },
    }


def _compute_column_stats(values: list[np.ndarray]) -> dict[str, list[float]]:
    data = np.vstack([np.asarray(item, dtype=np.float32).reshape(1, -1) for item in values])
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "q01": np.quantile(data, 0.01, axis=0).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).tolist(),
    }


def _slice_stats(
    column_stats: dict[str, list[float]], start: int, end: int
) -> dict[str, list[float]]:
    return {name: column_stats[name][start:end] for name in STAT_NAMES}


def _nested_processor_stats(
    state_stats: dict[str, list[float]],
    action_stats: dict[str, list[float]],
    embodiment_tag: str,
) -> dict[str, Any]:
    return {
        embodiment_tag: {
            "state": {
                name: _slice_stats(state_stats, start, end)
                for name, (start, end) in STATE_LAYOUT.items()
            },
            "action": {
                name: _slice_stats(action_stats, start, end)
                for name, (start, end) in ACTION_LAYOUT.items()
            },
        }
    }


def _build_info_json(
    src_info: dict[str, Any],
    *,
    n_episodes: int,
    n_frames: int,
    image_map: dict[str, str],
    video_codec: str,
    max_episode_index: int,
) -> dict[str, Any]:
    fps = int(src_info.get("fps") or 30)
    features = dict(src_info.get("features") or {})
    for image_col in list(features):
        spec = features[image_col]
        if isinstance(spec, dict) and spec.get("dtype") == "image":
            features.pop(image_col)

    if "state" in features:
        features["observation.state"] = features.pop("state")
        features["observation.state"]["names"] = ["observation.state"]
    if "actions" in features:
        features["action"] = features.pop("actions")
        features["action"]["names"] = ["action"]

    src_features = src_info.get("features") or {}
    for source_col, original_key in image_map.items():
        src_spec = src_features.get(source_col) or {}
        shape = list(src_spec.get("shape") or [256, 256, 3])
        height, width = int(shape[0]), int(shape[1])
        features[original_key] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": video_codec,
                "video.pix_fmt": "yuv420p",
                "video.fps": fps,
                "video.channels": int(shape[2]) if len(shape) > 2 else 3,
                "has_audio": False,
            },
        }

    n_videos = n_episodes * len(image_map)
    chunk_size = int(src_info.get("chunks_size") or 1000)
    return {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "panda"),
        "total_episodes": n_episodes,
        "total_frames": n_frames,
        "total_tasks": int(src_info.get("total_tasks") or 1),
        "total_videos": n_videos,
        "total_chunks": max(1, (max_episode_index // chunk_size) + 1),
        "chunks_size": chunk_size,
        "fps": fps,
        "splits": {"train": f"0:{max_episode_index + 1}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }


def convert_one_episode(
    *,
    src: Path,
    dst: Path,
    info: dict[str, Any],
    episode_index: int,
    image_map: dict[str, str],
    fps: int,
    overwrite: bool,
    keep_images: bool,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_parquet = src / _episode_relpath(info, episode_index)
    dst_parquet = dst / DATA_PATH_TEMPLATE.format(
        episode_chunk=episode_index // int(info.get("chunks_size") or 1000),
        episode_index=episode_index,
    )
    chunk_size = int(info.get("chunks_size") or 1000)

    table = pq.read_table(src_parquet)
    n_rows = table.num_rows
    available = set(table.column_names)
    missing_images = [col for col in image_map if col not in available]
    if missing_images:
        raise KeyError(
            f"episode {episode_index}: parquet missing image columns {missing_images}; "
            f"have {sorted(available)}"
        )

    codec_used = "h264"
    for source_col, original_key in image_map.items():
        video_path = dst / _video_relpath(episode_index, original_key, chunk_size)
        if video_path.exists() and not overwrite:
            continue
        frames = [
            decode_image_value(cell, src)
            for cell in table.column(source_col).to_pylist()
        ]
        if len(frames) != n_rows:
            raise RuntimeError(
                f"episode {episode_index} camera {source_col}: "
                f"{len(frames)} frames vs {n_rows} rows"
            )
        codec_used = encode_h264_video(frames, video_path, fps)

    if not dst_parquet.exists() or overwrite:
        keep_names: list[str] = []
        arrays = []
        drop = set() if keep_images else set(image_map)
        for name in table.column_names:
            if name in drop:
                continue
            keep_names.append(COLUMN_RENAME.get(name, name))
            arrays.append(table.column(name))
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, names=keep_names), dst_parquet)

    return {
        "episode_index": episode_index,
        "n_rows": n_rows,
        "codec": codec_used,
        "src": str(src_parquet),
    }


def _write_processor_overlay(
    dst: Path,
    *,
    image_map: dict[str, str],
    action_horizon: int,
    embodiment_tag: str,
    state_stats: dict[str, list[float]],
    action_stats: dict[str, list[float]],
) -> None:
    overlay_dir = dst / "meta" / "gr00t_processor_overlay"
    _dump_json(
        overlay_dir / "new_embodiment_modality_config.json",
        _build_processor_modality_config(image_map, action_horizon),
    )
    _dump_json(
        overlay_dir / "statistics.json",
        _nested_processor_stats(state_stats, action_stats, embodiment_tag),
    )


def cmd_convert(args: argparse.Namespace) -> None:
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    if src == dst:
        raise SystemExit("--dst must be a different directory from --src")

    info = _load_json(src / "meta" / "info.json")
    episodes_meta = {
        int(row["episode_index"]): row
        for row in _load_jsonl(src / "meta" / "episodes.jsonl")
        if "episode_index" in row
    }
    tasks = _load_jsonl(src / "meta" / "tasks.jsonl")
    image_map = dict(DEFAULT_IMAGE_MAP)
    if args.image_map:
        image_map = {}
        for item in args.image_map:
            if "=" not in item:
                raise SystemExit(f"--image-map entries must be src=dst, got {item!r}")
            src_col, dst_key = item.split("=", 1)
            image_map[src_col.strip()] = dst_key.strip()

    all_present = discover_episode_indices(src, info)
    present = list(all_present)
    if args.max_episodes is not None:
        present = present[: int(args.max_episodes)]
    if not present:
        raise SystemExit(
            f"No episode parquet files under {src / 'data'}. "
            "Copy the full dump (data/chunk-*/episode_*.parquet) first."
        )

    claimed = int(info.get("total_episodes") or 0)
    print(
        f"Source {src}: meta claims {claimed} episodes, "
        f"found {len(all_present)} parquet files, "
        f"converting {len(present)}."
    )
    if claimed and len(all_present) < claimed:
        print(
            "note: this looks like a subset checkout. Re-run the same command "
            "on the full ~250-episode dump when it is available; episode_index "
            "values are preserved so you can merge later.",
            file=sys.stderr,
        )

    if args.dry_run:
        for episode_index in present:
            print(f"  would convert episode {episode_index:06d}")
        return

    _require_pyarrow()
    import pyarrow.parquet as pq

    dst.mkdir(parents=True, exist_ok=True)
    fps = int(info.get("fps") or 30)
    jobs = max(1, int(args.jobs))
    results: list[dict[str, Any]] = []

    def _work(episode_index: int) -> dict[str, Any]:
        return convert_one_episode(
            src=src,
            dst=dst,
            info=info,
            episode_index=episode_index,
            image_map=image_map,
            fps=fps,
            overwrite=bool(args.overwrite),
            keep_images=bool(args.keep_images),
        )

    if jobs == 1:
        for episode_index in present:
            print(f"converting episode {episode_index:06d} ...")
            results.append(_work(episode_index))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_work, idx): idx for idx in present}
            for future in as_completed(futures):
                episode_index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed converting episode {episode_index}: {exc}"
                    ) from exc
                print(
                    f"converted episode {episode_index:06d} "
                    f"({result['n_rows']} frames)"
                )
                results.append(result)

    results.sort(key=lambda row: int(row["episode_index"]))
    n_frames = sum(int(row["n_rows"]) for row in results)
    codecs = {row["codec"] for row in results}
    video_codec = "h264" if codecs == {"h264"} else sorted(codecs)[0]

    # Stats from converted parquet (renamed columns, no images).
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    for row in results:
        parquet_path = dst / DATA_PATH_TEMPLATE.format(
            episode_chunk=int(row["episode_index"])
            // int(info.get("chunks_size") or 1000),
            episode_index=int(row["episode_index"]),
        )
        table = pq.read_table(
            parquet_path, columns=["observation.state", "action"]
        )
        for cell in table.column("observation.state").to_pylist():
            state_values.append(np.asarray(cell, dtype=np.float32).reshape(-1))
        for cell in table.column("action").to_pylist():
            action_values.append(np.asarray(cell, dtype=np.float32).reshape(-1))

    unique_state = {int(v.shape[0]) for v in state_values}
    unique_action = {int(v.shape[0]) for v in action_values}
    if unique_state != {19}:
        print(
            f"warning: expected 19-D state, got shapes {sorted(unique_state)}",
            file=sys.stderr,
        )
    if unique_action != {7}:
        print(
            f"warning: expected 7-D action, got shapes {sorted(unique_action)}",
            file=sys.stderr,
        )

    state_stats = _compute_column_stats(state_values)
    action_stats = _compute_column_stats(action_values)
    stats = {
        "observation.state": state_stats,
        "action": action_stats,
    }

    converted_meta = []
    for row in results:
        episode_index = int(row["episode_index"])
        meta = dict(episodes_meta.get(episode_index, {}))
        meta["episode_index"] = episode_index
        meta["length"] = int(row["n_rows"])
        meta.setdefault(
            "tasks",
            ["Pick up the block and insert it into the hole"],
        )
        converted_meta.append(meta)

    max_episode_index = max(int(row["episode_index"]) for row in results)
    dst_info = _build_info_json(
        info,
        n_episodes=len(results),
        n_frames=n_frames,
        image_map=image_map,
        video_codec=video_codec,
        max_episode_index=max_episode_index,
    )

    meta_dir = dst / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _dump_json(meta_dir / "info.json", dst_info)
    _dump_json(meta_dir / "modality.json", _build_modality_json(image_map))
    _dump_json(meta_dir / "stats.json", stats)
    _dump_jsonl(meta_dir / "episodes.jsonl", converted_meta)
    if tasks:
        _dump_jsonl(meta_dir / "tasks.jsonl", tasks)
    else:
        _dump_jsonl(
            meta_dir / "tasks.jsonl",
            [
                {
                    "task_index": 0,
                    "task": "Pick up the block and insert it into the hole",
                }
            ],
        )
    _write_processor_overlay(
        dst,
        image_map=image_map,
        action_horizon=int(args.action_horizon),
        embodiment_tag=str(args.embodiment_tag),
        state_stats=state_stats,
        action_stats=action_stats,
    )
    print(
        f"Wrote GR00T dataset: {dst}\n"
        f"  episodes={len(results)} frames={n_frames} codec={video_codec}\n"
        f"  overlay={dst / 'meta' / 'gr00t_processor_overlay'}"
    )


def find_processor_dir(model_path: Path) -> Path:
    for candidate in (model_path / "processor", model_path):
        if candidate.is_dir() and all(
            (candidate / name).is_file() for name in PROCESSOR_REQUIRED
        ):
            return candidate
    raise SystemExit(
        f"No GR00T processor files under {model_path}. "
        f"Expected {', '.join(PROCESSOR_REQUIRED)}."
    )


def cmd_patch_processor(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    overlay_dir = dataset / "meta" / "gr00t_processor_overlay"
    modality_overlay = overlay_dir / "new_embodiment_modality_config.json"
    stats_overlay_path = overlay_dir / "statistics.json"
    if not modality_overlay.is_file() or not stats_overlay_path.is_file():
        raise SystemExit(
            f"Missing processor overlay under {overlay_dir}. "
            "Run the convert subcommand first."
        )

    src_processor = find_processor_dir(Path(args.model_path).expanduser().resolve())
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"--output already exists: {output} (pass --overwrite)")

    if output.resolve() != src_processor.resolve():
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(src_processor, output)
    processor_dir = output

    processor_cfg_path = processor_dir / "processor_config.json"
    processor_cfg = _load_json(processor_cfg_path)
    kwargs = processor_cfg.setdefault("processor_kwargs", processor_cfg)
    modality_configs = kwargs.setdefault("modality_configs", {})
    embodiment_tag = str(args.embodiment_tag)
    modality_configs[embodiment_tag] = _load_json(modality_overlay)

    # Cartesian 7-D actions are already task-space deltas; do not apply a
    # second relative-action transform on top unless the user opts out.
    if not args.keep_relative_action:
        kwargs["use_relative_action"] = False

    stats_path = processor_dir / "statistics.json"
    stats = _load_json(stats_path)
    overlay_stats = _load_json(stats_overlay_path)
    stats[embodiment_tag] = overlay_stats[embodiment_tag]

    _dump_json(processor_cfg_path, processor_cfg)
    _dump_json(stats_path, stats)
    print(
        f"Patched processor at {processor_dir}\n"
        f"  added modality_configs[{embodiment_tag!r}] and statistics[{embodiment_tag!r}]\n"
        f"  use this directory as actor.model.model_path (or its parent checkpoint)."
    )


def cmd_validate(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).expanduser().resolve()
    info = _load_json(dataset / "meta" / "info.json")
    modality = _load_json(dataset / "meta" / "modality.json")
    stats = _load_json(dataset / "meta" / "stats.json")
    episodes = _load_jsonl(dataset / "meta" / "episodes.jsonl")
    errors: list[str] = []

    if "video_path" not in info:
        errors.append("meta/info.json missing video_path")
    if "video" not in modality:
        errors.append("meta/modality.json missing video")
    for col in ("observation.state", "action"):
        if col not in stats:
            errors.append(f"meta/stats.json missing {col}")
        else:
            for name in STAT_NAMES:
                if name not in stats[col]:
                    errors.append(f"meta/stats.json {col} missing {name}")

    video_keys = [
        spec.get("original_key", f"observation.images.{short}")
        for short, spec in (modality.get("video") or {}).items()
    ]
    chunk_size = int(info.get("chunks_size") or 1000)
    for row in episodes:
        episode_index = int(row["episode_index"])
        parquet = dataset / DATA_PATH_TEMPLATE.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
        )
        if not parquet.is_file():
            errors.append(f"missing parquet for episode {episode_index}: {parquet}")
            continue
        for video_key in video_keys:
            video = dataset / _video_relpath(episode_index, video_key, chunk_size)
            if not video.is_file():
                errors.append(f"missing video for episode {episode_index}: {video}")

    overlay = dataset / "meta" / "gr00t_processor_overlay"
    if not (overlay / "statistics.json").is_file():
        errors.append("missing meta/gr00t_processor_overlay/statistics.json")

    if errors:
        print("VALIDATION FAILED:")
        for item in errors:
            print(f"  - {item}")
        raise SystemExit(1)

    print(
        f"OK: {dataset}\n"
        f"  episodes={len(episodes)} "
        f"frames={info.get('total_frames')} "
        f"videos={info.get('total_videos')}\n"
        f"  cameras={list((modality.get('video') or {}).keys())}\n"
        f"  state_groups={list((modality.get('state') or {}).keys())}\n"
        f"  action_groups={list((modality.get('action') or {}).keys())}"
    )


def _add_shared_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image-map",
        nargs="*",
        default=None,
        help=(
            "Override camera mapping as src_column=observation.images.<name>. "
            f"Default: {DEFAULT_IMAGE_MAP}"
        ),
    )
    parser.add_argument(
        "--embodiment-tag",
        default="new_embodiment",
        help="GR00T embodiment tag used in the processor overlay.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="Processor action chunk length; must match actor.model.num_action_chunks.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert realworld-pni (LeRobot v2.1 image-parquet) to a GR00T N1.7 "
            "LeRobot dataset with mp4 videos, modality.json and stats.json."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Write a GR00T-ready dataset directory.")
    convert.add_argument("--src", required=True, help="Source realworld-pni root.")
    convert.add_argument("--dst", required=True, help="Output GR00T dataset root.")
    convert.add_argument("--max-episodes", type=int, default=None)
    convert.add_argument("--jobs", type=int, default=4)
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep original image columns in parquet (debug; larger files).",
    )
    convert.add_argument("--dry-run", action="store_true")
    _add_shared_dataset_args(convert)
    convert.set_defaults(func=cmd_convert)

    patch = sub.add_parser(
        "patch-processor",
        help="Copy a GR00T processor dir and inject new_embodiment configs/stats.",
    )
    patch.add_argument("--dataset", required=True, help="Converted GR00T dataset root.")
    patch.add_argument(
        "--model-path",
        required=True,
        help="GR00T-N1.7 checkpoint or its processor/ subdirectory.",
    )
    patch.add_argument(
        "--output",
        required=True,
        help="Directory to write the patched processor files into.",
    )
    patch.add_argument("--overwrite", action="store_true")
    patch.add_argument(
        "--keep-relative-action",
        action="store_true",
        help="Leave processor use_relative_action unchanged (default: set it False).",
    )
    patch.add_argument("--embodiment-tag", default="new_embodiment")
    patch.set_defaults(func=cmd_patch_processor)

    validate = sub.add_parser("validate", help="Check a converted dataset tree.")
    validate.add_argument("--dataset", required=True)
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
