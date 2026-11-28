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

"""Lightweight local/remote asset resolver matching ``openpi.shared.download``."""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import time
import urllib.parse

_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


def _get_mtime(year: int, month: int, day: int) -> float:
    date = datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc)
    return time.mktime(date.timetuple())


_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(
        2025, 2, 17
    ),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def get_cache_dir() -> pathlib.Path:
    cache_dir = (
        pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR))
        .expanduser()
        .resolve()
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(
    url: str, *, force_download: bool = False, **kwargs
) -> pathlib.Path:
    """Resolve a local path or download a remote URI into the OpenPI cache dir."""
    parsed = urllib.parse.urlparse(url)

    # Short-circuit local paths without touching fsspec or cache directories.
    if parsed.scheme == "":
        path = pathlib.Path(url).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()
    local_path = (cache_dir / parsed.netloc / parsed.path.strip("/")).resolve()

    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        import filelock
    except ImportError as exc:
        raise FileNotFoundError(
            f"Asset {url!r} was not found in cache ({local_path}) and `filelock` "
            "is not installed to download remote URIs. Run "
            "`bash requirements/embodied/download_assets.sh --assets openpi` "
            "or install `filelock` and `fsspec[gcs]`."
        ) from exc

    try:
        lock_path = local_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(lock_path):
            _ensure_permissions(lock_path)
            if invalidate_cache and local_path.exists():
                logger.info("Removing expired cached entry: %s", local_path)
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()
            elif local_path.exists():
                return local_path

            logger.info("Downloading %s to %s", url, local_path)
            scratch_path = local_path.with_suffix(".partial")
            _download_fsspec(url, scratch_path, **kwargs)
            shutil.move(scratch_path, local_path)
            _ensure_permissions(local_path)
    except PermissionError as exc:
        raise PermissionError(
            f"Local file permission error while downloading {url}. "
            f"Please retry after removing cached data: `rm -rf {local_path}*`"
        ) from exc

    return local_path


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    try:
        import fsspec
    except ImportError as exc:
        raise ImportError(
            f"Downloading remote asset {url!r} requires `fsspec` (and `gcsfs` "
            "for gs:// URIs). Alternatively, pre-download OpenPI assets with "
            "`bash requirements/embodied/download_assets.sh --assets openpi`."
        ) from exc

    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    is_dir = info["type"] == "directory" or (
        info["size"] == 0 and info["name"].endswith("/")
    )
    total_size = fs.du(url) if is_dir else info["size"]

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    if tqdm is None:
        fs.get(url, str(local_path), recursive=is_dir)
        return

    with tqdm(
        total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fs.get, url, str(local_path), recursive=is_dir)
            while not future.done():
                current_size = sum(
                    f.stat().st_size
                    for f in [*local_path.rglob("*"), local_path]
                    if f.is_file()
                )
                pbar.update(max(0, current_size - pbar.n))
                time.sleep(1)
            future.result()
            pbar.update(max(0, total_size - pbar.n))


def _set_permission(path: pathlib.Path, target_permission: int) -> None:
    try:
        if path.stat().st_mode & target_permission == target_permission:
            return
        path.chmod(target_permission)
    except PermissionError:
        pass


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    cache_dir = get_cache_dir()
    try:
        relative_path = path.relative_to(cache_dir)
    except ValueError:
        return

    moving_path = cache_dir
    for part in relative_path.parts:
        candidate = moving_path / part
        if candidate.is_dir():
            _set_folder_permission(candidate)
        moving_path = candidate

    file_rw = (
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IRGRP
        | stat.S_IWGRP
        | stat.S_IROTH
        | stat.S_IWOTH
    )

    def _set_file_permission(file_path: pathlib.Path) -> None:
        if file_path.stat().st_mode & 0o100:
            _set_permission(
                file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )
        else:
            _set_permission(file_path, file_rw)

    if path.is_file():
        _set_file_permission(path)
        return

    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            _set_file_permission(root_path / file)
        for directory in dirs:
            _set_folder_permission(root_path / directory)


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    assert local_path.exists(), f"File not found at {local_path}"
    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            return local_path.stat().st_mtime <= expire_time
    return False
