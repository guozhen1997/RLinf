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

"""DreamZero SFT data utilities for LeRobot datasets.

DreamZeroLeRobotDataset samples raw LeRobot modalities and DreamZeroCollator
delegates embodiment-specific processing to the DreamZero data transform.

"""

import bisect
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.utils.logging import get_logger

logger = get_logger()


def _load_task_texts(meta_dir: Path) -> dict[int, str]:
    """Build task_index -> instruction string mapping from tasks.jsonl or tasks.parquet."""
    import pandas as pd

    task_map: dict[int, str] = {}

    tasks_jsonl = meta_dir / "tasks.jsonl"
    if tasks_jsonl.exists():
        with open(tasks_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                task_id = int(entry.get("task_index", 0))
                task_text = str(entry.get("task", ""))
                task_map[task_id] = task_text
        if task_map:
            return task_map

    task_path = meta_dir / "tasks.parquet"
    if not task_path.exists():
        return {}

    tasks_df = pd.read_parquet(task_path)
    if list(tasks_df.columns) == ["task_index"] and tasks_df.index.dtype.kind in (
        "U",
        "O",
        "S",
    ):
        for text, row in tasks_df.iterrows():
            task_map[int(row["task_index"])] = str(text)
        return task_map

    text_col = None
    for candidate in ("task", "task_text", "language", "instruction", "prompt"):
        if candidate in tasks_df.columns:
            text_col = candidate
            break
    if text_col is None:
        cols = [c for c in tasks_df.columns if c != "task_index"]
        text_col = cols[0] if cols else None

    for _, row in tasks_df.iterrows():
        task_id = int(row.get("task_index", 0))
        if text_col is None:
            task_text = ""
        else:
            value = row.get(text_col, "")
            task_text = "" if value is None else str(value)
        task_map[task_id] = task_text
    return task_map


def _probe_video_container_fps(video_path: Path) -> float | None:
    """Read average FPS from the video container (PyAV), not meta/info.json.

    Many RoboMIND / converted trees ship ``fps`` in info.json that does not match the
    muxed stream (e.g. meta 14 vs H.264 30). Using meta for ``index / fps`` decode times
    then violates lerobot's PTS tolerance against torchvision/pyav.
    """
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(video_path), mode="r") as container:
            streams = container.streams.video
            if not streams:
                return None
            st = streams[0]

            def _as_positive_fps(rate: Any) -> float | None:
                if rate is None:
                    return None
                try:
                    f = float(rate)
                except (TypeError, ValueError, ZeroDivisionError):
                    return None
                if 0.5 < f < 480.0:
                    return f
                return None

            for attr in ("average_frame_rate", "guessed_frame_rate", "average_rate"):
                f = _as_positive_fps(getattr(st, attr, None))
                if f is not None:
                    return f

            cc = getattr(st, "codec_context", None)
            if cc is not None:
                f = _as_positive_fps(getattr(cc, "framerate", None))
                if f is not None:
                    return f

            nb = int(getattr(st, "frames", 0) or 0)
            if nb > 0 and st.duration is not None and st.time_base is not None:
                try:
                    dur_s = float(st.duration * st.time_base)
                    if dur_s > 1e-3:
                        f = float(nb) / dur_s
                        if 0.5 < f < 480.0:
                            return f
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
    except OSError:
        return None
    except Exception:
        logger.debug("PyAV fps probe failed for %s", video_path, exc_info=True)
        return None
    return None



def _droid_default_state_action_slices() -> tuple[slice, slice, slice, slice]:
    """Slice ranges into convert_droid-style concatenated vectors.

    state: cartesian(6) + gripper(1) + joint(7)
    action: cartesian(6) + cartesian_vel(6) + gripper(1) + gripper_vel(1) + joint(7) + joint_vel(7)
    """
    st_joint = slice(7, 14)
    st_grip = slice(6, 7)
    ac_joint = slice(14, 21)
    ac_grip = slice(12, 13)
    return st_joint, st_grip, ac_joint, ac_grip


def _load_modality_json(meta_dir: Path) -> dict[str, Any]:
    modality_path = meta_dir / "modality.json"
    if not modality_path.exists():
        return {}
    with open(modality_path, encoding="utf-8") as f:
        return json.load(f)


def _feature_component_spans(names: Any, feature_dim: int) -> dict[str, slice]:
    spans: dict[str, slice] = {}
    if not names:
        return spans

    if isinstance(names, dict):
        cursor = 0
        for key, values in names.items():
            width = len(_flatten_leaves(values))
            if width <= 0:
                continue
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        return spans

    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        component_widths = {
            "cartesian_position": 6,
            "cartesian_velocity": 6,
            "gripper_position": 1,
            "gripper_velocity": 1,
            "joint_position": 7,
            "joint_velocity": 7,
            "state": feature_dim,
            "actions": feature_dim,
        }
        cursor = 0
        for key in names:
            width = int(component_widths.get(key, 1))
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        return spans

    cursor = 0
    for entry in names if isinstance(names, list) else []:
        if isinstance(entry, str):
            key, width = entry, 1
        elif isinstance(entry, dict):
            key = str(entry.get("name", ""))
            shape = entry.get("shape")
            width = int(np.prod(shape)) if shape is not None else int(entry.get("dim", 1))
        else:
            continue
        if key:
            spans[key] = slice(cursor, cursor + width)
        cursor += width
    return spans


def _infer_modality_json_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Best-effort modality metadata for LeRobot trees without meta/modality.json."""
    modality: dict[str, Any] = {"video": {}, "state": {}, "action": {}, "annotation": {}}

    for source_key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        if feature.get("dtype") == "video" or source_key.startswith("observation.images."):
            short = source_key.split("observation.images.", 1)[-1]
            modality["video"][short] = {"original_key": source_key}
        elif source_key in ("image", "wrist_image"):
            modality["video"][source_key] = {"original_key": source_key}
        elif source_key.startswith("annotation."):
            short = source_key.split("annotation.", 1)[-1]
            modality["annotation"][short] = {"original_key": source_key}

    for modality_name, source_candidates in (
        ("state", ("observation.state", "state")),
        ("action", ("action", "actions")),
    ):
        source_key = next((key for key in source_candidates if key in features), None)
        if source_key is None:
            continue
        feature = features.get(source_key) or {}
        feature_dim = int((feature.get("shape") or [0])[0] or 0)
        for key, span in _feature_component_spans(feature.get("names"), feature_dim).items():
            modality[modality_name][key] = {
                "original_key": source_key,
                "start": int(span.start or 0),
                "end": None if span.stop is None else int(span.stop),
            }

    return {key: value for key, value in modality.items() if value}


def _safe_lang_text(value: Any, task_map: dict[int, str]) -> str:
    """Decode language field into a non-empty string when possible."""
    raw = value
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return ""
        raw = raw[0]
        if hasattr(raw, "item"):
            raw = raw.item()
    if isinstance(raw, (int, np.integer)) and task_map:
        return str(task_map.get(int(raw), "")).strip()
    if raw is None:
        return ""
    return str(raw).strip()


def _discover_local_lerobot_episode_indices(
    root: Path, info: dict, allowed_episode_indices: set[int] | None = None
) -> list[int]:
    """Episode indices that have both parquet and all video files on disk.

    LeRobot 0.3.x otherwise checks ``range(total_episodes)`` from info.json; any missing
    file triggers Hub download (``get_safe_version``), which breaks offline machines even
    when ``data/`` already contains a subset of episodes.
    """
    root = root.resolve()
    data_root = root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"LeRobot dataset missing data/ directory: {data_root}."
            "Offline loading requires local data, videos, and meta to be aligned."
        )
    ep_re = re.compile(r"^episode_(\d+)\.parquet$")
    present: set[int] = set()
    for p in data_root.rglob("episode_*.parquet"):
        m = ep_re.match(p.name)
        if m:
            present.add(int(m.group(1)))
    if not present:
        raise FileNotFoundError(
            f"No episode_*.parquet found in {data_root}."
            "Please confirm data_path matches disk directory (e.g. data/chunk-000/episode_000000.parquet)."
        )
    chunks_size = int(info.get("chunks_size") or 1000)
    data_tmpl = info.get("data_path")
    video_tmpl = info.get("video_path")
    if not data_tmpl:
        raise ValueError("meta/info.json missing data_path")
    feats = info.get("features") or {}
    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    complete: list[int] = []
    for ep_idx in sorted(present):
        ep_chunk = ep_idx // chunks_size
        rel_p = Path(data_tmpl.format(episode_chunk=ep_chunk, episode_index=ep_idx))
        if not (root / rel_p).is_file():
            continue
        if video_tmpl and video_keys:
            if not all(
                (
                    root
                    / Path(
                        video_tmpl.format(
                            episode_chunk=ep_chunk,
                            video_key=vk,
                            episode_index=ep_idx,
                        )
                    )
                ).is_file()
                for vk in video_keys
            ):
                continue
        complete.append(ep_idx)
    if allowed_episode_indices is not None:
        complete = [e for e in complete if e in allowed_episode_indices]
    if not complete:
        raise FileNotFoundError(
            f"Found parquet in {root}/data/, but no episode that satisfies "
            f"data_path and video_path (both in meta/episodes.jsonl) in info.json."
            f"({len(present)} parquet files on disk). Please fill in the corresponding videos/ or check if the paths match meta."
        )
    return complete


def _load_dreamzero_transform(model_path: str, embodiment_tag: str, tokenizer_path: str):
    """Instantiate the same DreamZero data transform used by the model."""
    from groot.vla.data.schema import DatasetMetadata
    from groot.vla.data.transform import ComposedModalityTransform
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    exp_cfg_dir = Path(str(model_path)).expanduser() / "experiment_cfg"
    with open(exp_cfg_dir / "metadata.json", "r") as f:
        metadatas = json.load(f)
    if embodiment_tag not in metadatas:
        raise KeyError(
            f"embodiment_tag={embodiment_tag!r} not found in {exp_cfg_dir / 'metadata.json'}"
        )

    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    if "transforms" not in train_cfg or embodiment_tag not in train_cfg.transforms:
        raise KeyError(
            f"transforms.{embodiment_tag} not found in {exp_cfg_dir / 'conf.yaml'}"
        )
    transform_cfg = train_cfg.transforms[embodiment_tag]
    dream_transform_cfg = transform_cfg.transforms[-1]
    dream_transform_cfg._target_ = (
        "rlinf.models.embodiment.dreamzero.patch.dreamzero_cotrain.DreamTransform"
    )
    dream_transform_cfg.tokenizer_path = tokenizer_path
    data_transform = instantiate(transform_cfg)
    assert isinstance(data_transform, ComposedModalityTransform), f"{data_transform=}"
    data_transform.set_metadata(DatasetMetadata.model_validate(metadatas[embodiment_tag]))
    return data_transform


def _collect_transform_keys(data_transform: Any) -> tuple[list[str], list[str], list[str]]:
    """Read expected video/state/action keys from the transform chain."""
    video_keys: list[str] = []
    state_keys: list[str] = []
    action_keys: list[str] = []
    for transform in getattr(data_transform, "transforms", []):
        video_keys.extend(getattr(transform, "video_concat_order", []) or [])
        state_keys.extend(getattr(transform, "state_concat_order", []) or [])
        action_keys.extend(getattr(transform, "action_concat_order", []) or [])
    return video_keys, state_keys, action_keys


def _load_dreamzero_language_keys(model_path: str, embodiment_tag: str) -> list[str]:
    """Read language modality keys from DreamZero's dataset config."""
    from omegaconf import OmegaConf

    cfg_path = Path(str(model_path)).expanduser() / "experiment_cfg" / "conf.yaml"
    train_cfg = OmegaConf.load(cfg_path)

    candidate_configs = []
    for root_key in ("train_dataset", "dataset"):
        root = train_cfg.get(root_key)
        if root is not None:
            candidate_configs.append(root)
            all_modality_configs = root.get("all_modality_configs")
            if all_modality_configs is not None:
                candidate_configs.append(all_modality_configs)
    if train_cfg.get("all_modality_configs") is not None:
        candidate_configs.append(train_cfg.all_modality_configs)

    direct_key = f"modality_config_{embodiment_tag}"
    if train_cfg.get(direct_key) is not None:
        candidate_configs.append({embodiment_tag: train_cfg.get(direct_key)})

    for config in candidate_configs:
        if config is None or embodiment_tag not in config:
            continue
        language_cfg = config[embodiment_tag].get("language")
        if language_cfg is None:
            continue
        keys = language_cfg.get("modality_keys")
        if keys:
            return [str(k) for k in keys]

    # DreamTransform only recognizes keys containing "annotation" as language.
    # Use a single fallback instead of fabricating every known language key.
    return ["annotation.language.action_text"]


def _flatten_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_flatten_leaves(v))
        return out
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def _infer_named_component_slices(
    names: Any, feature_dim: int, wanted: list[str]
) -> dict[str, slice] | None:
    """Infer component slices from LeRobot feature names for generic DreamZero transforms."""
    if not wanted:
        return {}
    if not names:
        return None

    if isinstance(names, dict):
        cursor = 0
        spans: dict[str, slice] = {}
        for key, values in names.items():
            width = len(_flatten_leaves(values))
            if width <= 0:
                continue
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        if all(k in spans for k in wanted):
            return {k: spans[k] for k in wanted}
        if len(wanted) == 1 and wanted[0] in ("state", "actions"):
            return {wanted[0]: slice(0, feature_dim)}
        return None

    if isinstance(names, list) and all(isinstance(x, str) for x in names):
        plan = {
            "cartesian_position": 6,
            "cartesian_velocity": 6,
            "gripper_position": 1,
            "gripper_velocity": 1,
            "joint_position": 7,
            "joint_velocity": 7,
            "state": feature_dim,
            "actions": feature_dim,
        }
        cursor = 0
        spans: dict[str, slice] = {}
        for key in names:
            width = int(plan.get(key, 1))
            spans[str(key)] = slice(cursor, cursor + width)
            cursor += width
        if all(k in spans for k in wanted):
            return {k: spans[k] for k in wanted}

    cursor = 0
    spans = {}
    for entry in names if isinstance(names, list) else []:
        if isinstance(entry, str):
            key, width = entry, 1
        elif isinstance(entry, dict):
            key = str(entry.get("name", ""))
            shape = entry.get("shape")
            width = int(np.prod(shape)) if shape is not None else int(entry.get("dim", 1))
        else:
            continue
        spans[key] = slice(cursor, cursor + width)
        cursor += width
    if all(k in spans for k in wanted):
        return {k: spans[k] for k in wanted}

    return None


class DreamZeroLeRobotDataset(Dataset):
    """Generic LeRobot-backed DreamZero SFT dataset.

    This class only samples raw modalities from LeRobot. Embodiment-specific
    normalization, video layout, padding, tokenization, and embodiment id mapping
    are handled by DreamZeroCollator via the DreamZero transform.
    """

    def __init__(
        self,
        data_path: str | list[str],
        video_keys: list[str],
        state_keys: list[str],
        action_keys: list[str],
        language_keys: list[str],
        lazy_load: bool,
        num_video_frames: int,
        state_horizon: int,
        action_horizon: int,
        num_chunks: int,
        relative_action: bool = False,
        relative_action_keys: list[str] | None = None,
        pq_cache_max_episodes: int = 128,
        video_tolerance_s: float = 0.1,
    ):
        if isinstance(data_path, (list, tuple)):
            if len(data_path) == 0:
                raise ValueError("DreamZeroLeRobotDataset requires at least one data path.")
            data_path = data_path[0]
        self.data_path = str(data_path)
        self.lazy_load = bool(lazy_load)
        self.num_video_frames = int(num_video_frames)
        self.state_horizon = int(state_horizon)
        self.action_horizon = int(action_horizon)
        self.num_chunks = int(num_chunks)
        if self.num_video_frames <= 0:
            raise ValueError(f"num_video_frames must be positive, got {num_video_frames!r}")
        if self.state_horizon <= 0:
            raise ValueError(f"state_horizon must be positive, got {state_horizon!r}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon!r}")
        if self.num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks!r}")
        self.relative_action = bool(relative_action)
        self.relative_action_keys = list(relative_action_keys or [])

        self.video_keys = list(video_keys)
        self.state_keys = list(state_keys)
        self.action_keys = list(action_keys)
        self.language_keys = list(language_keys)
        if not self.video_keys or not self.state_keys or not self.action_keys:
            raise ValueError(
                "DreamZeroLeRobotDataset requires video/state/action modality keys; "
                f"got video={self.video_keys}, state={self.state_keys}, action={self.action_keys}"
            )
        if not self.language_keys:
            raise ValueError("DreamZeroLeRobotDataset requires at least one language key.")

        self._root = Path(self.data_path).resolve()
        self._meta_dir = self._root / "meta"
        with open(self._meta_dir / "info.json") as f:
            self._info = json.load(f)
        self._fps = float(self._info.get("fps", 10))
        self._version = str(self._info.get("codebase_version", "v3.0"))
        self._tasks = _load_task_texts(self._meta_dir)
        self._features = self._info.get("features") or {}
        self._modality_meta = _load_modality_json(self._meta_dir)
        if not self._modality_meta:
            self._modality_meta = _infer_modality_json_from_features(self._features)
            logger.info(
                "meta/modality.json not found under %s; inferred modality mapping from info.json features.",
                self._meta_dir,
            )
        self._source_video_key = self._build_source_video_key_map()
        self._state_components = self._build_component_sources("state", self.state_keys)
        self._action_components = self._build_component_sources(
            "action", self.action_keys
        )
        self._language_sources = self._build_language_sources()
        self._vector_source_keys = sorted(
            {
                source
                for source, _ in [
                    *self._state_components.values(),
                    *self._action_components.values(),
                ]
            }
        )

        self._video_offsets = np.asarray(range(self.num_video_frames), dtype=np.int64)
        self._state_offsets = np.asarray(
            [
                chunk_idx * self.action_horizon + state_idx
                for chunk_idx in range(self.num_chunks)
                for state_idx in range(self.state_horizon)
            ],
            dtype=np.int64,
        )
        self._action_offsets = np.asarray(
            [
                chunk_idx * self.action_horizon + action_idx
                for chunk_idx in range(self.num_chunks)
                for action_idx in range(self.action_horizon)
            ],
            dtype=np.int64,
        )

        self._use_image_parquet_tree = self._uses_image_parquet_storage()
        self._use_lazy_video_tree = bool(
            self.lazy_load
            and not self._use_image_parquet_tree
            and self._info.get("video_path")
            and self._root.exists()
        )
        if self._use_lazy_video_tree:
            self._init_lazy_map_style(pq_cache_max_episodes, video_tolerance_s)
        else:
            self._init_lerobot_or_v2_parquet()

    @staticmethod
    def _short_modality_key(key: str) -> str:
        return key.split(".", 1)[1] if "." in key else key

    def _uses_image_parquet_storage(self) -> bool:
        """Detect LeRobot trees where image frames live in parquet, not mp4 files."""
        source_keys = set(self._source_video_key.values())
        if not source_keys:
            return False
        source_features = [self._features.get(key) or {} for key in source_keys]
        has_video_feature = any(feature.get("dtype") == "video" for feature in source_features)
        if has_video_feature:
            return False
        all_image_features = all(feature.get("dtype") == "image" for feature in source_features)
        if not all_image_features:
            return False
        return (
            self._version.startswith("v2")
            or int(self._info.get("total_videos") or 0) == 0
            or not self._info.get("video_path")
        )

    def _modality_entry(self, modality: str, key: str) -> dict[str, Any] | None:
        entries = self._modality_meta.get(modality)
        if not isinstance(entries, dict):
            return None
        return entries.get(self._short_modality_key(key))

    def _build_source_video_key_map(self) -> dict[str, str]:
        image_features = [
            k for k in self._features if k.startswith("observation.images.")
        ]
        mapping: dict[str, str] = {}
        for transform_key in self.video_keys:
            short = self._short_modality_key(transform_key)
            entry = self._modality_entry("video", transform_key)
            if entry is not None and entry.get("original_key"):
                mapping[transform_key] = str(entry["original_key"])
                continue
            canonical = f"observation.images.{short}"
            if canonical in self._features:
                mapping[transform_key] = canonical
                continue
            # LeRobot v2 LIBERO stores image columns without the observation.images prefix.
            if short in self._features or short in ("image", "wrist_image"):
                mapping[transform_key] = short
                continue
            match = [k for k in image_features if k.endswith(f".{short}")]
            if match:
                mapping[transform_key] = match[0]
        if len(mapping) != len(self.video_keys):
            missing = [k for k in self.video_keys if k not in mapping]
            raise KeyError(
                f"Could not map transform video keys {missing} to LeRobot image features "
                f"under {self.data_path}; available={sorted(image_features)}"
            )
        return mapping

    def _default_vector_source_key(self, modality: str) -> str:
        candidates = (
            ("observation.state", "state")
            if modality == "state"
            else ("action", "actions")
        )
        for candidate in candidates:
            if candidate in self._features:
                return candidate
        return candidates[0]

    def _build_component_sources(
        self, modality: str, transform_keys: list[str]
    ) -> dict[str, tuple[str, slice]]:
        sources: dict[str, tuple[str, slice]] = {}
        missing_keys: list[str] = []
        for key in transform_keys:
            entry = self._modality_entry(modality, key)
            if entry is None:
                missing_keys.append(key)
                continue
            source = str(entry.get("original_key") or self._default_vector_source_key(modality))
            start = int(entry.get("start", 0))
            end = entry.get("end")
            sources[key] = (source, slice(start, None if end is None else int(end)))
        if not missing_keys:
            return sources

        missing_short_keys = [self._short_modality_key(k) for k in missing_keys]
        if modality == "state":
            feature = self._features.get("observation.state") or self._features.get(
                "state"
            ) or {}
        else:
            feature = self._features.get("action") or self._features.get("actions") or {}
        dim = int((feature.get("shape") or [0])[0] or 0)
        inferred = _infer_named_component_slices(
            feature.get("names"), dim, missing_short_keys
        )
        if inferred is None and len(missing_short_keys) == 1:
            inferred = {missing_short_keys[0]: slice(0, dim or None)}
        if inferred is None:
            # Backward-compatible DROID fallback when meta names are missing.
            if set(missing_short_keys).issubset({"joint_position", "gripper_position"}):
                st_j, st_g, ac_j, ac_g = _droid_default_state_action_slices()
                inferred = (
                    {"joint_position": st_j, "gripper_position": st_g}
                    if modality == "state"
                    else {"joint_position": ac_j, "gripper_position": ac_g}
                )
            else:
                raise ValueError(
                    f"Cannot infer {modality} component slices for keys={transform_keys} "
                    f"from feature names={feature.get('names')!r} dim={dim}"
                )
        source = self._default_vector_source_key(modality)
        for key in missing_keys:
            sources[key] = (source, inferred[self._short_modality_key(key)])
        return sources

    def _build_language_sources(self) -> dict[str, str]:
        sources: dict[str, str] = {}
        annotations = self._modality_meta.get("annotation")
        for key in self.language_keys:
            subkey = self._short_modality_key(key)
            entry = annotations.get(subkey) if isinstance(annotations, dict) else None
            source = entry.get("original_key") if entry else None
            sources[key] = str(source or f"annotation.{subkey}")
        return sources

    def _init_lazy_map_style(
        self, pq_cache_max_episodes: int, video_tolerance_s: float
    ) -> None:
        if not self._root.exists():
            raise FileNotFoundError(f"DreamZero data_path must be local: {self.data_path}")
        self._chunks_size = int(self._info.get("chunks_size") or 1000)
        self._data_tmpl = str(self._info.get("data_path") or "")
        self._video_tmpl = str(self._info.get("video_path") or "")
        if not self._data_tmpl:
            raise ValueError("meta/info.json missing data_path")

        meta_episode_indices: set[int] = set()
        episode_lengths: dict[int, int] = {}
        with open(self._meta_dir / "episodes.jsonl") as epf:
            for line in epf:
                if not line.strip():
                    continue
                obj = json.loads(line)
                ep_idx = int(obj.get("episode_index", 0))
                meta_episode_indices.add(ep_idx)
                for k in ("episode_length", "length", "num_frames", "num_steps"):
                    if obj.get(k) is not None:
                        episode_lengths[ep_idx] = int(obj[k])
                        break

        self._episodes = _discover_local_lerobot_episode_indices(
            self._root, self._info, allowed_episode_indices=meta_episode_indices
        )
        self._episode_lengths = [
            episode_lengths.get(ep, self._infer_episode_length_from_parquet(ep))
            for ep in self._episodes
        ]
        self._episode_starts = [0]
        total = 0
        for n in self._episode_lengths:
            total += int(n)
            self._episode_starts.append(total)
        self._total_frames = int(total)
        self._pq_cache: "OrderedDict[int, Any]" = OrderedDict()
        self._pq_cache_max_episodes = max(1, int(pq_cache_max_episodes))
        self._video_decode_fps_cache: "OrderedDict[str, float]" = OrderedDict()
        self._video_decode_fps_cache_max = 512
        self._video_backend = "pyav"
        self._video_tolerance_s = float(video_tolerance_s)
        if self._video_tolerance_s <= 0:
            raise ValueError(f"video_tolerance_s must be positive, got {video_tolerance_s!r}")

    def _init_lerobot_or_v2_parquet(self) -> None:
        if self._use_image_parquet_tree:
            self._init_v2_image_parquet()
            return
        import lerobot.datasets.lerobot_dataset as lerobot_dataset

        delta_timestamps = {
            self._source_video_key[k]: [t / self._fps for t in self._video_offsets]
            for k in self.video_keys
        }
        state_sources = {source for source, _ in self._state_components.values()}
        action_sources = {source for source, _ in self._action_components.values()}
        for source in state_sources:
            delta_timestamps[source] = [t / self._fps for t in self._state_offsets]
        for source in action_sources:
            delta_timestamps[source] = [t / self._fps for t in self._action_offsets]
        self.dataset = lerobot_dataset.LeRobotDataset(
            self.data_path,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )
        self._use_v2_image_parquet = False

    def _init_v2_image_parquet(self) -> None:
        import pyarrow.parquet as pq

        data_root = self._root / "data"
        episodes_path = self._meta_dir / "episodes.jsonl"
        self._episodes_meta = []
        with open(episodes_path) as f:
            for line in f:
                if line.strip():
                    self._episodes_meta.append(json.loads(line))
        self._ep_frames = []
        self._ep_parquet_paths = []
        for ep in self._episodes_meta:
            ep_idx = int(ep["episode_index"])
            pq_path = data_root / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
            table = pq.read_table(pq_path)
            self._ep_frames.append(len(table))
            self._ep_parquet_paths.append(pq_path)
        self._cumulative = np.cumsum(self._ep_frames)
        self._total_frames = int(self._cumulative[-1])
        self._pq_cache = {}
        self._use_v2_image_parquet = True

    def _read_v2_episode(self, ep_pos: int):
        if ep_pos not in self._pq_cache:
            import pyarrow.parquet as pq

            self._pq_cache[ep_pos] = pq.read_table(str(self._ep_parquet_paths[ep_pos]))
            if len(self._pq_cache) > 50:
                del self._pq_cache[next(iter(self._pq_cache))]
        return self._pq_cache[ep_pos]

    def _decode_v2_image(self, cell) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        raw = cell.as_py()
        if isinstance(raw, dict):
            raw = raw.get("bytes", raw)
        if isinstance(raw, bytes):
            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        return np.asarray(raw)

    def _get_v2_image_sample(self, idx: int) -> dict[str, Any]:
        ep_pos = int(np.searchsorted(self._cumulative, idx, side="right"))
        start = int(self._cumulative[ep_pos - 1]) if ep_pos > 0 else 0
        frame_in_ep = int(idx) - start
        table = self._read_v2_episode(ep_pos)
        n = len(table)

        def clamp(offset: int) -> int:
            return min(max(frame_in_ep + int(offset), 0), n - 1)

        sample: dict[str, Any] = {
            "episode_index": int(self._episodes_meta[ep_pos].get("episode_index", ep_pos)),
            "frame_index": int(frame_in_ep),
        }
        for transform_key, source_key in self._source_video_key.items():
            sample[transform_key] = np.stack(
                [
                    self._decode_v2_image(table.column(source_key)[clamp(o)])
                    for o in self._video_offsets
                ],
                axis=0,
            )
        state_rows = [clamp(o) for o in self._state_offsets]
        action_rows = [clamp(o) for o in self._action_offsets]
        state_sources = {source for source, _ in self._state_components.values()}
        action_sources = {source for source, _ in self._action_components.values()}
        for source in state_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in state_rows], dtype=np.float32
            )
        for source in action_sources:
            if source not in table.column_names:
                continue
            sample[source] = np.asarray(
                [table.column(source)[r].as_py() for r in action_rows], dtype=np.float32
            )
        for key, source in self._language_sources.items():
            if source in table.column_names:
                sample[key] = table.column(source)[frame_in_ep].as_py()
        if "task" in table.column_names:
            sample["task"] = table.column("task")[frame_in_ep].as_py()
        elif "task_index" in table.column_names:
            sample["task_index"] = table.column("task_index")[frame_in_ep].as_py()
        return sample

    def _infer_episode_length_from_parquet(self, episode_index: int) -> int:
        import pyarrow.parquet as pq

        return int(pq.read_metadata(str(self._get_parquet_path(episode_index))).num_rows)

    def _get_parquet_path(self, episode_index: int) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._data_tmpl.format(
                episode_chunk=ep_chunk, episode_index=int(episode_index)
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Parquet file not found for episode {episode_index}: {p}")
        return p

    def _get_video_path(self, episode_index: int, video_key: str) -> Path:
        ep_chunk = int(episode_index) // self._chunks_size
        rel = Path(
            self._video_tmpl.format(
                episode_chunk=ep_chunk,
                video_key=video_key,
                episode_index=int(episode_index),
            )
        )
        p = (self._root / rel).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"Video file not found for episode {episode_index} key {video_key}: {p}"
            )
        return p

    def _decode_fps_for_video_file(self, video_path: Path) -> float:
        key = str(video_path.resolve())
        if key in self._video_decode_fps_cache:
            fps = self._video_decode_fps_cache.pop(key)
            self._video_decode_fps_cache[key] = fps
            return fps
        fps = float(_probe_video_container_fps(video_path) or self._fps)
        self._video_decode_fps_cache[key] = fps
        if len(self._video_decode_fps_cache) > self._video_decode_fps_cache_max:
            self._video_decode_fps_cache.popitem(last=False)
        return fps

    def _get_episode_table(self, episode_index: int):
        episode_index = int(episode_index)
        if episode_index in self._pq_cache:
            tbl = self._pq_cache.pop(episode_index)
            self._pq_cache[episode_index] = tbl
            return tbl
        import pyarrow.parquet as pq

        p = self._get_parquet_path(episode_index)
        schema = set(pq.read_schema(str(p)).names)
        cols = [
            c
            for c in (
                "observation",
                *self._vector_source_keys,
                "task",
                "task_index",
                *self._language_sources.values(),
            )
            if c in schema
        ]
        tbl = pq.read_table(str(p), columns=list(dict.fromkeys(cols)))
        self._pq_cache[episode_index] = tbl
        if len(self._pq_cache) > self._pq_cache_max_episodes:
            self._pq_cache.popitem(last=False)
        return tbl

    @staticmethod
    def _clip_indices(indices: np.ndarray, length: int) -> np.ndarray:
        return np.clip(indices.astype(np.int64), 0, max(0, int(length) - 1))

    @staticmethod
    def _video_to_thwc_uint8(frames: Any) -> np.ndarray:
        """Match Groot VideoTransform: numpy (T, H, W, C) uint8.

        LeRobot ``decode_video_frames`` returns float32 (T, C, H, W) in [0, 1].
        Parquet/PIL paths usually already yield uint8 (T, H, W, C).
        """
        if torch.is_tensor(frames):
            arr = frames.detach().cpu().numpy()
        else:
            arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        elif arr.ndim == 5:
            # (B, T, C, H, W) -> (B, T, H, W, C)
            if arr.shape[2] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 1, 3, 4, 2))
        elif arr.ndim == 4:
            if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32, copy=False)
            max_v = float(arr_f.max()) if arr_f.size else 0.0
            if max_v <= 1.0 + 1e-3:
                arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr_f, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _col_exists(table, name: str) -> bool:
        return hasattr(table, "column_names") and name in table.column_names

    @staticmethod
    def _read_list_column(table, name: str, indices: np.ndarray) -> np.ndarray:
        col = table.column(name)
        return np.asarray([col[int(i)].as_py() for i in indices.tolist()], dtype=np.float32)

    @staticmethod
    def _read_struct_list_field(
        table, struct_col: str, field: str, indices: np.ndarray
    ) -> np.ndarray:
        col = table.column(struct_col)
        arr = col.chunk(0) if hasattr(col, "num_chunks") and col.num_chunks > 0 else col
        field_arr = arr.field(field)
        return np.asarray([field_arr[int(i)].as_py() for i in indices.tolist()], dtype=np.float32)

    def _get_lazy_sample(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(f"Index {idx} out of range for dataset of len {self._total_frames}")
        ep_pos = bisect.bisect_right(self._episode_starts, int(idx)) - 1
        ep_pos = max(0, min(ep_pos, len(self._episodes) - 1))
        frame_in_ep = int(idx) - int(self._episode_starts[ep_pos])
        episode_index = int(self._episodes[ep_pos])
        ep_len = int(self._episode_lengths[ep_pos])
        table = self._get_episode_table(episode_index)
        video_idx = self._clip_indices(frame_in_ep + self._video_offsets, ep_len)
        state_idx = self._clip_indices(frame_in_ep + self._state_offsets, ep_len)
        action_idx = self._clip_indices(frame_in_ep + self._action_offsets, ep_len)

        from lerobot.datasets.video_utils import decode_video_frames

        sample: dict[str, Any] = {
            "episode_index": episode_index,
            "frame_index": frame_in_ep,
        }
        for transform_key, source_key in self._source_video_key.items():
            video_path = self._get_video_path(episode_index, source_key)
            fps = self._decode_fps_for_video_file(video_path)
            sample[transform_key] = decode_video_frames(
                video_path,
                [float(int(i)) / fps for i in video_idx.tolist()],
                tolerance_s=self._video_tolerance_s,
                backend=self._video_backend,
            )

        for key in (
            "task",
            "task_index",
        ):
            if self._col_exists(table, key):
                sample[key] = table.column(key)[int(frame_in_ep)].as_py()
        for key, source in self._language_sources.items():
            if self._col_exists(table, source):
                sample[key] = table.column(source)[int(frame_in_ep)].as_py()

        for source, _ in self._state_components.values():
            if source in sample:
                continue
            if self._col_exists(table, source):
                sample[source] = self._read_list_column(table, source, state_idx)
            elif source == "observation.state" and self._col_exists(table, "observation"):
                sample[source] = self._read_struct_list_field(
                    table, "observation", "state", state_idx
                )
            else:
                raise KeyError(f"episode parquet missing state source column {source!r}")
        for source, _ in self._action_components.values():
            if source in sample:
                continue
            if not self._col_exists(table, source):
                raise KeyError(f"episode parquet missing action source column {source!r}")
            sample[source] = self._read_list_column(table, source, action_idx)
        return sample

    def __len__(self) -> int:
        if self._use_lazy_video_tree or getattr(self, "_use_v2_image_parquet", False):
            return int(self._total_frames)
        return len(self.dataset)

    def _resolve_task_text(self, sample: dict[str, Any]) -> str:
        task_text = sample.get("task")
        if task_text is None:
            task_text = self._tasks.get(int(sample.get("task_index", 0)), "")
        candidates = []
        for key in self.language_keys:
            if key in sample:
                text = _safe_lang_text(sample[key], self._tasks)
                if text:
                    candidates.append(text)
        if candidates:
            return str(np.random.choice(candidates))
        return str(task_text or "")

    def _put_components(
        self,
        out: dict[str, Any],
        sample: dict[str, Any],
        components: dict[str, tuple[str, slice]],
        *,
        is_action: bool,
        state_ref: np.ndarray | None = None,
    ) -> None:
        for key, (source, sl) in components.items():
            raw = np.asarray(sample[source], dtype=np.float32)
            if raw.ndim == 1:
                raw = raw[None, :]
            value = raw[:, sl].astype(np.float32)
            if (
                is_action
                and self.relative_action
                and self._short_modality_key(key) in self.relative_action_keys
                and state_ref is not None
            ):
                value = value - state_ref[0:1, : value.shape[-1]]
            out[key] = value

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._use_lazy_video_tree:
            sample = self._get_lazy_sample(idx)
        elif getattr(self, "_use_v2_image_parquet", False):
            sample = self._get_v2_image_sample(idx)
        else:
            sample = self.dataset[idx]

        out: dict[str, Any] = {}
        for transform_key, source_key in self._source_video_key.items():
            if transform_key in sample:
                raw_frames = sample[transform_key]
            else:
                raw_frames = sample[source_key]
            out[transform_key] = self._video_to_thwc_uint8(raw_frames)

        self._put_components(out, sample, self._state_components, is_action=False)
        first_state_component = None
        if self.state_keys:
            first_state_component = out[self.state_keys[0]]
        self._put_components(
            out,
            sample,
            self._action_components,
            is_action=True,
            state_ref=first_state_component,
        )

        fallback_text = self._resolve_task_text(sample)
        wrote_language = False
        for key in self.language_keys:
            source = self._language_sources.get(key, key)
            value = sample.get(key, sample.get(source, ""))
            text = _safe_lang_text(value, self._tasks) if value != "" else ""
            if text:
                out[key] = text
                wrote_language = True
        if not wrote_language:
            out[self.language_keys[0]] = fallback_text
        return out



class DreamZeroCollator:
    """Batch raw LeRobot samples through the configured DreamZero transform."""

    def __init__(
        self,
        data_transform: Any,
    ):
        self.data_transform = data_transform

    @staticmethod
    def _stack_features(features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        keys = sorted({k for feature in features for k in feature.keys()})
        for key in keys:
            values = [feature[key] for feature in features]
            if isinstance(values[0], str):
                # dm-tree treats Python lists as nested structures; keep string batches
                # as an ndarray leaf so DreamTransform.apply_batch can index by sample.
                batch[key] = np.asarray([str(v) for v in values], dtype=object)
            else:
                batch[key] = np.stack(values, axis=0)
        return batch

    @staticmethod
    def _as_tensor_batch(batch: Any) -> dict[str, Any]:
        if not isinstance(batch, dict) and hasattr(batch, "__getstate__"):
            batch = batch.__getstate__()
        out: dict[str, Any] = {}
        for key, value in dict(batch).items():
            if torch.is_tensor(value):
                out[key] = value
            elif isinstance(value, np.ndarray):
                out[key] = torch.as_tensor(value)
            else:
                out[key] = value
        return out

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        raw_batch = self._stack_features(features)
        transformed = self.data_transform(raw_batch)
        return self._as_tensor_batch(transformed)


def build_dreamzero_sft_dataloader(
    cfg,
    world_size: int,
    rank: int,
    data_paths: str,
    eval_dataset: bool = False,
):
    """Build DreamZero SFT dataloader -- callable from FSDPVlaSftWorker.

    Uses DistributedSampler to shard data across GPUs:
      - Each of the 8 GPUs sees 1/8 of the dataset per epoch
      - micro_batch_size samples are returned per iteration per GPU
      - Global effective batch size = micro_batch_size * world_size * grad_accum_steps
    """
    model_cfg = cfg.actor.model
    tokenizer_path = model_cfg.get("tokenizer_path", "google/umt5-xxl")

    embodiment_tag = str(model_cfg.get("embodiment_tag", "libero_sim")).lower()
    model_path = str(model_cfg.get("model_path", ""))
    data_transform = _load_dreamzero_transform(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        tokenizer_path=tokenizer_path,
    )
    if eval_dataset:
        data_transform.eval()
    else:
        data_transform.train()
    video_keys, state_keys, action_keys = _collect_transform_keys(data_transform)
    language_keys = _load_dreamzero_language_keys(model_path, embodiment_tag)

    action_horizon = int(model_cfg.action_horizon)
    state_horizon = int(model_cfg.state_horizon)
    num_chunks = int(model_cfg.num_chunks)

    dataset = DreamZeroLeRobotDataset(
        data_path=data_paths,
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys,
        lazy_load=cfg.data.get("lazy_load", True),
        num_video_frames=int(model_cfg.num_video_frames),
        state_horizon=state_horizon,
        action_horizon=action_horizon,
        num_chunks=num_chunks,
        relative_action=bool(model_cfg.get("relative_action", False)),
        relative_action_keys=list(model_cfg.get("relative_action_keys", [])),
        pq_cache_max_episodes=cfg.data.get("parquet_cache_size", 128),
        video_tolerance_s=cfg.data.get("video_tolerance_s", 0.1),
    )
    logger.info(
        "DreamZero LeRobot dataset: embodiment=%s video_keys=%s state_keys=%s action_keys=%s language_keys=%s",
        embodiment_tag,
        dataset.video_keys,
        dataset.state_keys,
        dataset.action_keys,
        dataset.language_keys,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    num_workers = int(cfg.data.get("num_workers", 4))
    prefetch_factor = int(cfg.data.get("prefetch_factor", 4))
    data_loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.micro_batch_size,  # samples per GPU per step
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=num_workers,
        pin_memory=True,  # faster CPU->GPU transfer
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=DreamZeroCollator(data_transform=data_transform),
    )
    return data_loader, {"num_samples": len(dataset)}
