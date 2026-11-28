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

"""Pure Python/NumPy transforms matching ``openpi.transforms`` (JAX/Flax-free)."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

import numpy as np

from rlinf.models.embodiment.openpi.transforms import image_tools
from rlinf.models.embodiment.openpi.transforms import normalize as _normalize
from rlinf.models.embodiment.openpi.transforms import tokenizer as _tokenizer

DataDict: TypeAlias = dict[str, Any]
NormStats: TypeAlias = _normalize.NormStats

T = TypeVar("T")
S = TypeVar("S")


def _tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    """Recursively apply ``fn`` to leaves of nested mappings/lists/tuples."""
    if isinstance(tree, Mapping):
        return type(tree)({k: _tree_map(fn, v) for k, v in tree.items()})
    if isinstance(tree, tuple):
        if hasattr(tree, "_fields"):
            return type(tree)(*(_tree_map(fn, v) for v in tree))
        return tuple(_tree_map(fn, v) for v in tree)
    if isinstance(tree, list):
        return [_tree_map(fn, v) for v in tree]
    return fn(tree)


@runtime_checkable
class DataTransformFn(Protocol):
    """A per-sample transform over an unbatched nested dict."""

    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to a single unbatched sample."""


@dataclasses.dataclass(frozen=True)
class Group:
    """A pair of input and output transform sequences."""

    inputs: Sequence[DataTransformFn] = ()
    outputs: Sequence[DataTransformFn] = ()

    def push(
        self,
        *,
        inputs: Sequence[DataTransformFn] = (),
        outputs: Sequence[DataTransformFn] = (),
    ) -> Group:
        """Append ``inputs`` to the end and prepend ``outputs`` to the start."""
        return Group(
            inputs=(*self.inputs, *inputs),
            outputs=(*outputs, *self.outputs),
        )


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """Apply a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repack an input dictionary into a target nested dictionary structure."""

    structure: Any

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return _tree_map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: Mapping[str, Any] | None
    use_quantiles: bool = False
    strict: bool = False

    def __post_init__(self) -> None:
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: Mapping[str, Any] | None
    use_quantiles: bool = False

    def __post_init__(self) -> None:
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x: np.ndarray, stats: NormStats) -> np.ndarray:
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [
                    (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01,
                    x[..., dim:],
                ],
                axis=-1,
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {
            k: image_tools.resize_with_pad(v, self.height, self.width)
            for k, v in data["image"].items()
        }
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repack absolute actions into delta action space."""

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repack delta actions into absolute action space."""

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(
            prompt, state, actions
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(
            tokens.astype(np.int32), self.action_horizon, self.action_dim
        )
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extract a prompt from the current LeRobot dataset task."""

    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pad states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(
                data["actions"], self.model_action_dim, axis=-1
            )
        return data


def flatten_dict(tree: Mapping[str, Any], *, sep: str = "/") -> dict[str, Any]:
    """Flatten a nested mapping using ``sep`` as separator (matches flax.traverse_util)."""
    flat: dict[str, Any] = {}

    def _recurse(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            full_key = f"{prefix}{sep}{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                if len(value) == 0:
                    continue
                _recurse(value, full_key)
            else:
                flat[full_key] = value

    _recurse(tree, "")
    return flat


def unflatten_dict(tree: Mapping[str, Any], *, sep: str = "/") -> dict[str, Any]:
    """Unflatten a flat mapping using ``sep`` as separator (matches flax.traverse_util)."""
    nested: dict[str, Any] = {}
    for flat_key, value in tree.items():
        parts = flat_key.split(sep)
        cursor = nested
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            elif not isinstance(cursor[part], dict):
                raise ValueError(
                    f"Cannot unflatten key {flat_key!r}: {part!r} is already a leaf."
                )
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return nested


def transform_dict(
    patterns: Mapping[str, str | None], tree: Mapping[str, Any]
) -> dict[str, Any]:
    """Transform a nested dictionary structure using fullmatch regex patterns."""
    data = flatten_dict(tree)
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output: dict[str, Any] = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: Mapping[str, Any],
    selector: Mapping[str, Any],
    fn: Callable[[Any, Any], Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    flat_tree = flatten_dict(tree)
    flat_selector = flatten_dict(selector)

    if strict:
        for k in flat_selector:
            if k not in flat_tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict(
        {
            k: fn(v, flat_selector[k]) if k in flat_selector else v
            for k, v in flat_tree.items()
        }
    )


def pad_to_dim(
    x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0
) -> np.ndarray:
    """Pad an array to ``target_dim`` along ``axis`` with ``value``."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Create a tuple of booleans: positive ints append True, negative append False."""
    result: list[bool] = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: Mapping[str, Any]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                "quantile stats must be provided if use_quantile_norm is True. "
                f"Key {k} is missing q01 or q99."
            )
