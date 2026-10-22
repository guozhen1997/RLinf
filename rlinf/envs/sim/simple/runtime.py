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

"""Isaac/HSSD process patches used by the SIMPLE runtime."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Any

from filelock import FileLock


def install_data_path_lock() -> None:
    """Guard the upstream resolver before importing SIMPLE runtime modules.

    Hugging Face locks only the download. SIMPLE subsequently extracts and
    deletes the archive, so callers must share a lock for the whole operation.
    Even an existing target must be checked under the lock: another worker may
    have created that directory without finishing extraction yet.

    This changes only the SIMPLE resolver in the current worker process. All
    RLinf SIMPLE workers sharing its data directory use the same lock file.
    """
    import simple.utils as simple_utils

    original = simple_utils.resolve_data_path
    if getattr(original, "_rlinf_data_path_lock", False):
        return

    lock_path = str(
        Path(simple_utils.get_data_dir()) / ".cache" / "rlinf" / "resolve-data.lock"
    )

    def keep_log(record: logging.LogRecord) -> bool:
        # Suppress only this lock's DEBUG chatter, not other locks or warnings.
        return not (
            record.levelno == logging.DEBUG
            and isinstance(record.args, tuple)
            and lock_path in record.args
        )

    logging.getLogger("filelock").addFilter(keep_log)

    @wraps(original)
    def resolve_data_path(
        rel_path: str | None = None,
        create_if_not_exist: bool = False,
        auto_download: bool = False,
    ) -> str:
        if not rel_path:
            return original(rel_path, create_if_not_exist, auto_download)

        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        # Separate from HF's download lock to avoid recursively acquiring it.
        # Never unlink this file: waiters must keep locking the same inode.
        with FileLock(lock_path).acquire(poll_interval=0.5):
            resolved = original(rel_path, create_if_not_exist, auto_download)
            if not Path(resolved).exists():
                raise FileNotFoundError(
                    f"SIMPLE asset preparation did not create {resolved}."
                )
            return resolved

    resolve_data_path._rlinf_data_path_lock = True
    simple_utils.resolve_data_path = resolve_data_path


def install_articulation_reuse_guard(native_env: Any) -> None:
    """Keep SIMPLE from rebuilding an unchanged Isaac articulation on reset."""
    isaac = native_env.isaac
    original_add = isaac.add_articulated_object
    asset_keys: dict[str, tuple[str, str]] = {}

    def add_articulated_object_once(obj_name: str, obj_info: Any) -> None:
        asset_name = str(obj_info.asset.name)
        asset_key = (str(obj_info.asset.uid), str(obj_info.asset.usd_path))
        if asset_name in isaac.articulated_objects:
            if asset_keys.get(asset_name) != asset_key:
                raise RuntimeError(
                    f"Cannot replace SIMPLE articulated object {asset_name!r} "
                    "without rebuilding its Isaac physics views."
                )
            return
        original_add(obj_name, obj_info)
        asset_keys[asset_name] = asset_key

    isaac.add_articulated_object = add_articulated_object_once


def resolve_hssd_scene_dir(state: dict[str, Any]) -> Path | None:
    """Resolve the HSSD scene directory used by relative USD references."""
    scene = state.get("dr_state_dict", {}).get("scene", {})
    if not str(scene.get("uid", "")).startswith("hssd:"):
        return None

    data_dir = scene.get("data_dir")
    if not isinstance(data_dir, str):
        raise ValueError("SIMPLE HSSD reset state is missing scene.data_dir.")

    from simple.utils import resolve_data_path

    scene_dir = Path(resolve_data_path(data_dir, auto_download=True))
    props_dir = scene_dir / "props"
    if not props_dir.is_dir():
        raise FileNotFoundError(
            f"SIMPLE HSSD material assets are missing under {props_dir}."
        )
    return scene_dir


def anchor_hssd_material_references(scene_dir: Path) -> Any:
    """Anchor HSSD material references to their containing scene layer."""
    from pxr import Sdf

    scene_path = scene_dir / f"{scene_dir.name}.usd"
    layer = Sdf.Layer.FindOrOpen(str(scene_path))
    if layer is None:
        raise FileNotFoundError(f"Cannot open SIMPLE HSSD scene layer {scene_path}.")
    props_dir = scene_dir / "props"
    material_files = {
        path.relative_to(props_dir).as_posix().casefold(): path.relative_to(
            props_dir
        ).as_posix()
        for path in props_dir.rglob("*")
        if path.is_file()
    }
    material_refs = [
        asset_path
        for asset_path in layer.GetCompositionAssetDependencies()
        if "/props/" in asset_path
    ]
    if not material_refs:
        raise ValueError(f"SIMPLE HSSD scene has no material references: {scene_path}.")
    for asset_path in material_refs:
        requested_path = asset_path.rsplit("/props/", 1)[1]
        actual_path = material_files.get(requested_path.casefold())
        if actual_path is None:
            raise FileNotFoundError(
                f"SIMPLE HSSD material {requested_path} is missing under {props_dir}."
            )
        anchored_path = f"./props/{actual_path}"
        if not layer.UpdateCompositionAssetDependency(asset_path, anchored_path):
            raise RuntimeError(
                f"Cannot update SIMPLE HSSD material reference {asset_path}."
            )
    return layer
