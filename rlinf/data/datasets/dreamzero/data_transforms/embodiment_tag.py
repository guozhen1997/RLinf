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

from groot.vla.data.schema.embodiment_tags import EmbodimentTag as DefaultTags

_PATCH_PATH = (
    "rlinf.data.datasets.dreamzero.data_transforms.embodiment_tag.EmbodimentTag"
)
_PATCH_TARGETS = (
    "groot.vla.data.schema.embodiment_tags.EmbodimentTag",
    "groot.vla.data.schema.EmbodimentTag",
)


class EmbodimentTag(DefaultTags):
    FRANKA_PNP = "franka_pnp"


def add_embodiment_tag_patches(patcher) -> None:
    for target in _PATCH_TARGETS:
        if target not in patcher._mappings_dict:
            patcher.add_patch(target, _PATCH_PATH)


def ensure_groot_embodiment_tag_patched() -> None:
    from groot.vla.data.schema import embodiment_tags as groot_tags

    if groot_tags.EmbodimentTag is EmbodimentTag:
        return
    from rlinf.utils.patcher import Patcher

    add_embodiment_tag_patches(Patcher)
    Patcher.apply()
