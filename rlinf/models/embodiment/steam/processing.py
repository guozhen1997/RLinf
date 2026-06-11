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

"""Image and text processors for the STEAM value model.

Standalone module — does NOT import anything from
``rlinf.models.embodiment.value_model.processing``.

Key differences from the value_model processor:
    * Native vision-encoder image resolution (not fixed 224x224 + interpolate)
    * Outputs raw [0, 1] BCHW float images (not [-1, 1])
      so the downstream SteamBackbone can apply its own SigLIP-style
      mean/std normalization in ``_preprocess_images``.
"""

import logging
import os
import string
from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BatchFeature, PreTrainedTokenizerBase
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType

logger = logging.getLogger(__name__)


def resize_with_pad(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize an image to target size without distortion by padding with black.

    For float images the [0, 1] range is preserved (padding value 0.0).
    For uint8 images padding value is 0.

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input.
    """
    added_batch_dim = False

    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
            added_batch_dim = True
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)
            added_batch_dim = True

    batch_size, channels, cur_height, cur_width = images.shape

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        # The processor outputs [0, 1] floats; clamp accordingly.
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
        if added_batch_dim:
            padded_images = padded_images.squeeze(0)

    return padded_images


# Camera view names. Chosen so pre-existing LeRobot datasets (and the
# openpi policies that repack to these keys) plug in without
# modification. The time axis — frame_t vs frame_{t+k} —
# is handled outside the processor by the pair collator, which runs
# ``process_images`` once per frame and stacks the results along a new
# ``num_frames`` dimension before the backbone sees them.
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

# Default SigLIP vision-encoder resolution (siglip-so400m-patch14-384).
IMAGE_RESOLUTION = (384, 384)


def resolve_image_size(image_processor: Any) -> tuple[int, int]:
    """Resolve ``(height, width)`` from a HuggingFace image processor."""
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return IMAGE_RESOLUTION


def resolve_vision_image_size(
    vision_repo_id: str,
    revision: Optional[str] = None,
) -> tuple[int, int]:
    """Load a vision processor and return its native image size."""
    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(
        vision_repo_id,
        revision=revision,
        use_fast=True,
    )
    return resolve_image_size(image_processor)


def normalize_image_to_steam_format(
    img: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Convert any image format to BCHW float in [0, 1] range.

    The backbone expects raw [0, 1] floats; the model itself applies SigLIP
    mean/std normalization internally. This helper is the public counterpart
    to ``ValueProcessor.normalize_image_to_model_format`` but with target
    range [0, 1] instead of [-1, 1].
    """
    if device is not None:
        img = img.to(device)

    if img.dim() == 3:
        is_chw = img.shape[0] == 3
    elif img.dim() == 4:
        is_chw = img.shape[1] == 3
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {img.dim()}D")

    if img.dim() == 3:
        img = img[None, ...]

    img = img.float()

    if not is_chw:
        img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW

    if img.max() > 1.0:
        img = img / 255.0
    img = img.clamp(0.0, 1.0)

    if dtype is not None:
        img = img.to(dtype)
    return img


class SteamImageProcessor(ImageProcessingMixin):
    """STEAM image processor.

    Resizes raw multi-camera input to the configured native resolution and
    outputs BCHW [0, 1] float tensors. The downstream SteamBackbone handles
    SigLIP normalization internally.
    """

    model_input_names: ClassVar[list[str]] = ["pixel_values", "image_masks"]

    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_RESOLUTION,
        do_resize: bool = True,
        do_augment: bool = True,
        image_keys: Sequence[str] = IMAGE_KEYS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.do_resize = do_resize
        self.do_augment = do_augment
        self.image_keys = image_keys

    def apply_augmentations(
        self, image: torch.Tensor, is_wrist_camera: bool = False
    ) -> torch.Tensor:
        """Apply OpenPI-style augmentations.

        Input/output range: [0, 1] BHWC float (no [-1, 1] conversions —
        the processor keeps [0, 1] all the way through).
        """
        if not is_wrist_camera:
            height, width = image.shape[1:3]

            crop_height = int(height * 0.95)
            crop_width = int(width * 0.95)

            max_h = height - crop_height
            max_w = width - crop_width
            if max_h > 0 and max_w > 0:
                start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                image = image[
                    :,
                    start_h : start_h + crop_height,
                    start_w : start_w + crop_width,
                    :,
                ]

            image = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            angle = torch.rand(1, device=image.device) * 10 - 5
            if torch.abs(angle) > 0.1:
                angle_rad = angle * torch.pi / 180.0
                cos_a = torch.cos(angle_rad)
                sin_a = torch.sin(angle_rad)

                grid_x = torch.linspace(-1, 1, width, device=image.device)
                grid_y = torch.linspace(-1, 1, height, device=image.device)

                grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
                grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                grid_x_rot = grid_x * cos_a - grid_y * sin_a
                grid_y_rot = grid_x * sin_a + grid_y * cos_a
                grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                image = F.grid_sample(
                    image.permute(0, 3, 1, 2),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                ).permute(0, 2, 3, 1)

        # Color augmentations (apply to all cameras)
        brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6
        image = image * brightness_factor

        contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8
        mean = image.mean(dim=[1, 2, 3], keepdim=True)
        image = (image - mean) * contrast_factor + mean

        saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0
        gray = image.mean(dim=-1, keepdim=True)
        image = gray + (image - gray) * saturation_factor

        image = torch.clamp(image, 0.0, 1.0)
        return image

    def process_images(
        self,
        images_dict: dict[str, torch.Tensor],
        image_masks_dict: Optional[dict[str, torch.Tensor]] = None,
        train: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Process a batch of images at the configured native resolution.

        Output:
            (processed_images_dict, processed_masks_dict)
            Images are in BCHW format, [0, 1] float.
        """
        out_images = {}
        out_masks = {}

        batch_size = None
        template_device = None
        for key in images_dict:
            if images_dict[key] is not None:
                batch_size = images_dict[key].shape[0]
                template_device = images_dict[key].device
                break

        for key in self.image_keys:
            image = images_dict.get(key)

            # Missing keys get placeholder zero images with mask=False.
            if image is None:
                if batch_size is not None:
                    h, w = self.image_size
                    placeholder = torch.zeros(
                        batch_size, 3, h, w, device=template_device
                    )
                    out_images[key] = placeholder
                    out_masks[key] = torch.zeros(
                        batch_size, dtype=torch.bool, device=template_device
                    )
                continue

            is_wrist = "wrist" in key

            is_bchw = image.shape[1] == 3
            if is_bchw:
                image = image.permute(0, 2, 3, 1)  # BCHW -> BHWC

            if self.do_resize and tuple(image.shape[1:3]) != tuple(self.image_size):
                image = resize_with_pad(image, self.image_size[1], self.image_size[0])
                if image.dim() == 3:
                    image = image.unsqueeze(0)

            # Normalize to [0, 1] (SteamBackbone handles SigLIP mean/std internally)
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0
            image = image.clamp(0.0, 1.0)

            if train and self.do_augment:
                image = self.apply_augmentations(image, is_wrist_camera=is_wrist)

            image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW

            out_images[key] = image

            if image_masks_dict is not None and key in image_masks_dict:
                out_masks[key] = image_masks_dict[key]
            else:
                bsize = image.shape[0]
                out_masks[key] = torch.ones(
                    bsize, dtype=torch.bool, device=image.device
                )

        return out_images, out_masks

    def __call__(
        self,
        images: dict[str, torch.Tensor],
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_augment: Optional[bool] = None,
        train: bool = False,
        **kwargs,
    ) -> BatchFeature:
        apply_augmentations = train and (
            do_augment if do_augment is not None else self.do_augment
        )

        output_images, output_masks = self.process_images(
            images, image_masks, train=apply_augmentations
        )

        return {"pixel_values": output_images, "image_masks": output_masks}


class SteamProcessor(ProcessorMixin):
    """STEAM value model processor.

    Standalone — does not inherit from or import any class in
    ``rlinf.models.embodiment.value_model.processing``.

    Text templates:

        - With state (``include_state_in_prompt=True`` and state is not None):
          ``Task: {prompt}, State: {b0 b1 ... bN}\\nValue: ``
          where each ``bi`` is the integer bucket index of the corresponding
          state dim, obtained by clipping to ``[-1, 1]`` and applying
          ``np.digitize`` over ``state_discretization_bins`` uniform buckets.
        - Without state: ``Task: {prompt}\\nValue: ``
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SteamImageProcessor"
    tokenizer_class = "AutoTokenizer"
    _tokenize_log_count = 0

    def __init__(
        self,
        image_processor: Optional[SteamImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 200,
        tokenizer_name_or_path: Optional[str] = None,
        image_keys: Optional[tuple] = None,
        do_augment: bool = True,
        include_state_in_prompt: bool = True,
        max_state_dim: int = 32,
        state_discretization_bins: int = 256,
        **kwargs,
    ):
        if image_processor is None:
            image_processor = (
                SteamImageProcessor(image_keys=image_keys, do_augment=do_augment)
                if image_keys
                else SteamImageProcessor(do_augment=do_augment)
            )

        if tokenizer is None:
            tokenizer_path = tokenizer_name_or_path or os.environ.get(
                "VLA_TOKENIZER_PATH"
            )
            if not tokenizer_path or not os.path.exists(tokenizer_path):
                raise ValueError(
                    f"No tokenizer found. Provide tokenizer_name_or_path, "
                    f"set VLA_TOKENIZER_PATH env var, or place tokenizer files "
                    f"in the project pretrained_models directory. "
                    f"Tried: {tokenizer_path!r}"
                )
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, add_bos_token=True, local_files_only=True
            )

        self.image_processor = image_processor
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_token_len = max_token_len
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.include_state_in_prompt = include_state_in_prompt
        self.max_state_dim = max_state_dim
        self.state_discretization_bins = state_discretization_bins
        # Required for save_pretrained compatibility.
        self.chat_template = None
        self.audio_tokenizer = None

    def _clean_text(self, text: str) -> str:
        return text.lower().strip().replace("_", " ").replace("\n", " ")

    def _strip_trailing_punctuation(self, text: str) -> str:
        if text and text[-1] in string.punctuation and text[-1] not in "\"'":
            return text[:-1]
        return text

    def _build_prefix_text(
        self, prompt: str, state: Optional[np.ndarray] = None
    ) -> str:
        """Build the prefix text given a task prompt and optional state.

        When ``self.include_state_in_prompt`` is True and ``state`` is provided,
        the state is padded/truncated to ``self.max_state_dim``, clipped to
        ``[-1, 1]``, discretized via ``np.digitize`` into
        ``self.state_discretization_bins`` uniform buckets, and serialized as
        a space-separated string:

            ``Task: {cleaned}, State: {b0 b1 ... bN}\\nValue: ``

        Otherwise (``include_state_in_prompt=False`` or ``state is None``):

            ``Task: {cleaned}\\nValue: ``
        """
        cleaned = self._strip_trailing_punctuation(self._clean_text(prompt))
        if not self.include_state_in_prompt or state is None:
            return f"Task: {cleaned}\nValue: "

        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        target_dim = int(self.max_state_dim)
        if state_arr.shape[0] < target_dim:
            state_arr = np.pad(
                state_arr,
                (0, target_dim - state_arr.shape[0]),
                constant_values=0.0,
            )
        elif state_arr.shape[0] > target_dim:
            state_arr = state_arr[:target_dim]

        state_arr = np.clip(state_arr, -1.0, 1.0)
        bins = int(self.state_discretization_bins)
        # Interior edges only → np.digitize returns indices in [0, bins-1].
        edges = np.linspace(-1.0, 1.0, bins + 1, dtype=np.float32)[1:-1]
        bucket_indices = np.digitize(state_arr, edges)
        state_str = " ".join(str(int(b)) for b in bucket_indices)
        return f"Task: {cleaned}, State: {state_str}\nValue: "

    def _tokenize_single(
        self,
        prompt: str,
        state: Optional[np.ndarray] = None,
        max_length: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_length is None:
            max_length = self.max_token_len

        prefix_text = self._build_prefix_text(prompt, state)
        tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)

        seq_len = len(tokens)
        if seq_len < max_length:
            pad = max_length - seq_len
            mask = [True] * seq_len + [False] * pad
            tokens = tokens + [0] * pad
        else:
            if seq_len > max_length:
                logger.warning(
                    "Token length (%d) exceeds max (%d), truncating.",
                    seq_len,
                    max_length,
                )
            tokens = tokens[:max_length]
            mask = [True] * max_length

        worker_info = torch.utils.data.get_worker_info()
        is_worker_0 = worker_info is None or worker_info.id == 0
        if (
            is_worker_0
            and int(os.environ.get("RANK", 0)) == 0
            and SteamProcessor._tokenize_log_count < 2
        ):
            SteamProcessor._tokenize_log_count += 1
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            logger.info(
                "[Tokenization Example #%d] prompt=%r (state=%s) → %r  "
                "(raw_len=%d, pad_to=%d)",
                self._tokenize_log_count,
                prompt,
                "provided" if state is not None else "none",
                decoded,
                seq_len,
                max_length,
            )

        return np.asarray(tokens), np.asarray(mask)

    def process_text(
        self,
        prompts: list[str],
        states: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of prompts with optional per-sample state.

        Args:
            prompts: List of task prompts.
            states: Optional batched state — either ``np.ndarray``/``torch.Tensor``
                of shape ``[B, D]`` or a list of length ``B`` whose elements are
                1-D arrays. When provided (and ``include_state_in_prompt=True``),
                state is threaded into each prompt via ``_build_prefix_text``.
            max_length: Padding/truncation length. Defaults to ``self.max_token_len``.
            return_tensors: ``"pt"`` returns ``torch.Tensor``; otherwise ``np.ndarray``.

        Extra ``**kwargs`` are accepted and ignored for forward-compat with the
        sibling ``ValueProcessor`` signature.
        """
        del kwargs  # accepted for signature parity with ValueProcessor
        if max_length is None:
            max_length = self.max_token_len

        if states is not None:
            if isinstance(states, torch.Tensor):
                states = states.detach().cpu().numpy()
            if isinstance(states, np.ndarray):
                if states.ndim == 1:
                    states = states[None, :]
                if states.shape[0] != len(prompts):
                    raise ValueError(
                        f"states batch size ({states.shape[0]}) does not match "
                        f"prompts batch size ({len(prompts)})"
                    )
            elif isinstance(states, (list, tuple)):
                if len(states) != len(prompts):
                    raise ValueError(
                        f"states length ({len(states)}) does not match "
                        f"prompts length ({len(prompts)})"
                    )
            else:
                raise TypeError(
                    f"states must be ndarray/Tensor/list, got {type(states)}"
                )

        batch_tokens = []
        batch_masks = []
        for i, prompt in enumerate(prompts):
            state_i = states[i] if states is not None else None
            tokens, mask = self._tokenize_single(
                prompt=prompt, state=state_i, max_length=max_length
            )
            batch_tokens.append(tokens)
            batch_masks.append(mask)

        result = {
            "input_ids": np.stack(batch_tokens),
            "attention_mask": np.stack(batch_masks),
        }

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Union[dict[str, torch.Tensor], list[torch.Tensor], torch.Tensor] = None,
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        state: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        return_tensors: Optional[str] = "pt",
        train: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if text is None and images is None:
            raise ValueError("You must provide either text or images")

        result_data = {}

        if text is not None:
            is_batched = isinstance(text, list)
            texts = text if is_batched else [text]
            # When a single prompt is passed alongside a single state, lift the
            # state to a batch of 1 so the per-sample path below still works.
            batched_state = state
            if state is not None and not is_batched:
                if isinstance(state, torch.Tensor):
                    batched_state = state.detach().cpu().numpy()
                    if batched_state.ndim == 1:
                        batched_state = batched_state[None, :]
                elif isinstance(state, np.ndarray):
                    if state.ndim == 1:
                        batched_state = state[None, :]
                elif isinstance(state, (list, tuple)):
                    # Already iterable; wrap in outer list.
                    batched_state = [state]

            processed = self.process_text(
                prompts=texts,
                states=batched_state,
                return_tensors=return_tensors,
            )
            result_data.update(processed)

            if not is_batched:
                for key in result_data:
                    if result_data[key].dim() > 0:
                        result_data[key] = result_data[key][0]

        if images is not None:
            image_inputs = self.image_processor(
                images,
                image_masks=image_masks,
                return_tensors=return_tensors,
                train=train,
            )
            result_data.update(image_inputs)

        return BatchFeature(data=result_data, tensor_type=return_tensors)

    def decode(self, token_ids: Union[list[int], torch.Tensor], **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = [t for t in token_ids if t != 0]
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(
        self, token_ids_batch: Union[list[list[int]], torch.Tensor], **kwargs
    ) -> list[str]:
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        return [self.decode(tokens, **kwargs) for tokens in token_ids_batch]

    @property
    def model_input_names(self):
        return [
            "pixel_values",
            "image_masks",
            "input_ids",
            "attention_mask",
        ]


__all__ = [
    "SteamImageProcessor",
    "SteamProcessor",
    "normalize_image_to_steam_format",
    "resolve_image_size",
    "resolve_vision_image_size",
    "resize_with_pad",
    "IMAGE_KEYS",
    "IMAGE_RESOLUTION",
]
