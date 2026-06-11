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

"""Backbone for the STEAM binary pair classifier.

This module owns the neural architecture for deciding whether a pair
``(frame_t, frame_{t+k})`` drawn with an optional proprio state came from
a forward success demo (**progress**) or a rewound demo (**regress**).

The stack is a SigLIP vision encoder + Gemma3 language backbone + an MLP
head, with a forward path designed *for pair classification* rather than
scalar value regression:

* The per-frame image tokens are **concatenated in order** rather than
  mean-pooled across cameras. Concatenation preserves the frame ordering,
  which is the signal the classifier relies on (mean-pooling would make
  ``(t, t+k)`` and ``(t+k, t)`` indistinguishable).
* The head is a **2-way classifier** over ``{regress, progress}``.
  A single softmax layer on two logits expresses "which class does this
  pair belong to?" more faithfully than the single-logit sigmoid formulation,
  even though they are mathematically equivalent. Probabilities come out
  as ``[p_regress, p_progress]`` at inference time.
* There are no ``bin_centers``, ``v_min/v_max``, or scalar-return
  utilities — those are categorical-value concepts and do not apply here.

``SteamBackbone.forward`` returns the raw logits ``[B, 2]``; the
critic wrapper in :mod:`modeling_critic` owns the loss, metrics and
CriticOutput packaging.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from .configuration import SteamConfig


try:
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModel,
        AutoModelForCausalLM,
    )

    _transformers_available = True
except ImportError:  # pragma: no cover
    AutoConfig = None  # type: ignore[assignment]
    AutoImageProcessor = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    _transformers_available = False


NUM_CLASSES = 2
CLASS_REGRESS = 0
CLASS_PROGRESS = 1


# ---------------------------------------------------------------------------
# Backbone loader helpers
# ---------------------------------------------------------------------------


def _resolve_load_dtype(dtype_name: str) -> torch.dtype:
    """Map a dtype string to a torch dtype, degrading bf16 to fp32 on CPU-only hosts."""
    if dtype_name == "bfloat16":
        requested = torch.bfloat16
    elif dtype_name == "float16":
        requested = torch.float16
    else:
        requested = torch.float32
    if requested == torch.bfloat16 and not torch.cuda.is_available():
        return torch.float32
    return requested


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def _maybe_enable_gradient_checkpointing(module: nn.Module) -> None:
    if hasattr(module, "gradient_checkpointing_enable"):
        module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    elif hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = True


def _extract_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            f"Cannot infer hidden size for model type {type(model)}: missing .config"
        )
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return int(config.text_config.hidden_size)
    raise ValueError(f"Cannot infer hidden size for model config type {type(config)}")


def _module_parameter_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return the effective forward dtype for ``module`` when possible."""
    mixed_precision = getattr(module, "mixed_precision", None)
    if mixed_precision is not None and mixed_precision.param_dtype is not None:
        return mixed_precision.param_dtype

    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype

    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.dtype

    return fallback


def _extract_vision_feature_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            f"Cannot infer vision feature size for model type {type(model)}: "
            "missing .config"
        )
    if hasattr(config, "projection_dim"):
        return int(config.projection_dim)
    if hasattr(config, "vision_config") and hasattr(
        config.vision_config, "projection_dim"
    ):
        return int(config.vision_config.projection_dim)
    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(config, "vision_config") and hasattr(
        config.vision_config, "hidden_size"
    ):
        return int(config.vision_config.hidden_size)
    raise ValueError(
        f"Cannot infer vision feature size for model config type {type(config)}"
    )


def _resolve_image_size(image_processor: Any) -> tuple[int, int]:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        if "shortest_edge" in size:
            edge = int(size["shortest_edge"])
            return edge, edge
    if isinstance(size, int):
        return int(size), int(size)
    return 384, 384


def _resolve_norm_stats(
    image_processor: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mean_raw = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
    std_raw = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
    if len(mean_raw) != 3 or len(std_raw) != 3:
        raise ValueError(
            f"Expected RGB normalization stats of len=3, got mean={mean_raw} "
            f"std={std_raw}"
        )
    mean = (float(mean_raw[0]), float(mean_raw[1]), float(mean_raw[2]))
    std = (float(std_raw[0]), float(std_raw[1]), float(std_raw[2]))
    if any(v <= 0 for v in std):
        raise ValueError(f"Invalid image std values: {std}")
    return mean, std


def _load_language_model(
    repo_id: str,
    revision: str | None,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the text backbone from ``repo_id`` (causal LM if available)."""
    if AutoConfig is None or AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError(
            "transformers is not installed. Please install the embodied stack."
        )
    model_config = AutoConfig.from_pretrained(repo_id, revision=revision)
    architectures = getattr(model_config, "architectures", None) or []
    prefer_causal_lm = any(
        isinstance(arch, str) and arch.endswith("ForCausalLM") for arch in architectures
    )
    if prefer_causal_lm:
        lm_with_head = AutoModelForCausalLM.from_pretrained(
            repo_id, revision=revision, torch_dtype=dtype
        )
        if not hasattr(lm_with_head, "model"):
            raise RuntimeError(
                f"AutoModelForCausalLM loaded from '{repo_id}' does not expose "
                "`.model` text backbone."
            )
        return lm_with_head.model
    language_model = AutoModel.from_pretrained(
        repo_id, revision=revision, torch_dtype=dtype
    )
    if not isinstance(language_model, nn.Module):
        raise TypeError(
            f"AutoModel loaded from '{repo_id}' returned unexpected type: "
            f"{type(language_model)}"
        )
    return language_model


# ---------------------------------------------------------------------------
# STEAM backbone
# ---------------------------------------------------------------------------


class SteamBackbone(nn.Module):
    """SigLIP + Gemma3 backbone for STEAM pair classification.

    The forward path takes an observation dict whose ``images`` slot carries
    exactly :attr:`SteamConfig.num_frames_per_pair` camera entries —
    one per pair-frame in the order ``(frame_t, frame_{t+k})`` — plus a
    tokenised prompt. Each frame is run independently through the vision
    encoder, the per-frame image tokens are concatenated (preserving
    ordering), the language feature is concatenated alongside, and the
    resulting vector feeds a 2-way classifier head.
    """

    def __init__(self, cfg: "SteamConfig") -> None:
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError(
                "transformers is not installed. Please install the embodied stack."
            )

        self.cfg = cfg
        self.model_dtype = _resolve_load_dtype(cfg.dtype)

        self.vision_encoder = AutoModel.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            torch_dtype=self.model_dtype,
        )
        self.language_model = _load_language_model(
            repo_id=cfg.language_repo_id,
            revision=cfg.language_revision,
            dtype=self.model_dtype,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            cfg.vision_repo_id,
            revision=cfg.vision_revision,
            use_fast=True,
        )
        h, w = _resolve_image_size(image_processor)
        mean, std = _resolve_norm_stats(image_processor)
        self.image_resolution = (h, w)
        self.register_buffer(
            "image_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        vision_feat_dim = _extract_vision_feature_size(self.vision_encoder)
        lang_feat_dim = _extract_hidden_size(self.language_model)

        # Per-modality projectors — shared across frames on the image side.
        self.image_projector = nn.Sequential(
            nn.Linear(vision_feat_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(lang_feat_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        fused_dim = cfg.fusion_hidden_dim * (cfg.num_frames_per_pair + 1)
        self.fusion_norm = nn.LayerNorm(fused_dim)
        # Head width is num_bins: 2 for the legacy binary mode (output is
        # [regress_logit, progress_logit]) or num_bins > 2 for multi-bin
        # classification over signed-stride bins. The critic applies softmax
        # and cross-entropy with label smoothing on top in either case.
        head_out_dim = int(getattr(cfg, "num_bins", NUM_CLASSES))
        self.value_head = nn.Sequential(
            nn.Linear(fused_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden_dim, head_out_dim),
        )
        if getattr(cfg, "use_state_compatibility", False):
            self.state_projector = nn.Sequential(
                nn.LayerNorm(int(cfg.max_state_dim)),
                nn.Linear(int(cfg.max_state_dim), cfg.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
                nn.GELU(),
            )
            compat_dim = cfg.fusion_hidden_dim * 3
            self.compatibility_head = nn.Sequential(
                nn.LayerNorm(compat_dim),
                nn.Linear(compat_dim, cfg.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden_dim, 1),
            )

        if cfg.use_gradient_checkpointing:
            _maybe_enable_gradient_checkpointing(self.language_model)
            _maybe_enable_gradient_checkpointing(self.vision_encoder)
        if cfg.freeze_language_model:
            _freeze_module(self.language_model)
        if cfg.freeze_vision_encoder:
            _freeze_module(self.vision_encoder)

    # ------------------------------------------------------------------
    # Image preprocessing — ranges and resolution
    # ------------------------------------------------------------------

    def _apply_siglip_normalisation(self, flat_images: Tensor) -> Tensor:
        """SigLIP mean/std normalisation on a flat ``[N, C, H, W]`` tensor.

        The collator's processor has already resized to native resolution
        and converted to ``[0, 1]`` float, so this step only needs to
        subtract ``image_mean`` and divide by ``image_std``. A safety
        rescale/resize handles degenerate inputs (e.g. uint8 tensors fed
        directly through ``infer()``).
        """
        if flat_images.dtype == torch.uint8:
            flat_images = flat_images.to(dtype=torch.float32) / 255.0
        else:
            flat_images = flat_images.to(dtype=torch.float32)
            if bool(flat_images.max() > 1.0) or bool(flat_images.min() < 0.0):
                flat_images = (flat_images / 255.0).clamp(0.0, 1.0)

        if flat_images.shape[-2:] != self.image_resolution:
            flat_images = F.interpolate(
                flat_images,
                size=self.image_resolution,
                mode="bilinear",
                align_corners=False,
            )

        mean = self.image_mean.to(
            device=flat_images.device, dtype=flat_images.dtype
        ).view(1, 3, 1, 1)
        std = self.image_std.to(
            device=flat_images.device, dtype=flat_images.dtype
        ).view(1, 3, 1, 1)
        return (flat_images - mean) / std

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _encode_vision(self, flat_images: Tensor) -> Tensor:
        """Run the vision encoder on ``[N, C, H, W]`` and return pooled features."""
        if hasattr(self.vision_encoder, "get_image_features"):
            return self.vision_encoder.get_image_features(pixel_values=flat_images)
        vision_outputs = self.vision_encoder(pixel_values=flat_images, return_dict=True)
        pooler = getattr(vision_outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        if hasattr(vision_outputs, "last_hidden_state"):
            return vision_outputs.last_hidden_state.mean(dim=1)
        raise ValueError(
            "Unsupported vision encoder output — expected pooler_output or "
            "last_hidden_state"
        )

    def _encode_prompt(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean-pool the language model's last hidden state under the mask."""
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Language model output does not contain last_hidden_state")
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    # ------------------------------------------------------------------
    # Fusion + forward
    # ------------------------------------------------------------------

    def _fuse(
        self,
        per_frame_image_features: Tensor,
        language_feature: Tensor,
    ) -> Tensor:
        """Concat ``[frame_0, frame_1, ..., frame_{N-1}, lang]`` → LayerNorm.

        Args:
            per_frame_image_features: ``[B, num_frames, D]`` — one feature
                per frame (already camera-pooled + projected).
            language_feature: ``[B, D]`` — already through the projector.

        Returns:
            ``[B, D * (num_frames + 1)]``.
        """
        bsize = per_frame_image_features.shape[0]
        # Per-frame concat preserves frame ordering ``(t, t+k)``.
        frames_concat = per_frame_image_features.reshape(bsize, -1)
        fused = torch.cat([frames_concat, language_feature], dim=-1)
        fusion_dtype = _module_parameter_dtype(self.fusion_norm, fused.dtype)
        return self.fusion_norm(fused.to(dtype=fusion_dtype))

    def _compute_projected_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """End-to-end encoder + fusion plus per-frame features.

        Args:
            input_ids: ``[B, T]`` token ids for the (state-aware) prompt.
            attention_mask: ``[B, T]`` prompt mask.
            images: ``[B, num_cameras, num_frames, C, H, W]`` — per-camera
                per-frame pair images in ``[0, 1]`` BCHW, as emitted by
                :class:`BinaryPairDataCollator`. Camera order is sorted by
                key; frame order is ``(t, t+k)``.
            image_attention_mask: ``[B, num_cameras, num_frames]`` bool —
                zeros mask missing camera/frame slots (e.g. an episode that
                only provides the base view).

        Returns:
            Tuple ``(fused, per_frame_image_features, language_feature)`` where:
                * ``fused`` has shape
                  ``[B, fusion_hidden_dim * (num_frames + 1)]``.
                * ``per_frame_image_features`` has shape
                  ``[B, num_frames, fusion_hidden_dim]``.
                * ``language_feature`` has shape ``[B, fusion_hidden_dim]``.
        """
        self._check_shapes(input_ids, attention_mask, images, image_attention_mask)

        bsize, num_cameras, num_frames = images.shape[:3]
        image_attention_mask = image_attention_mask.to(
            dtype=torch.bool, device=images.device
        )
        if not torch.all(image_attention_mask.any(dim=(1, 2))):
            raise ValueError(
                "Each sample must have at least one valid (camera, frame) slot"
            )

        language_mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
        if not torch.all(language_mask.any(dim=1)):
            raise ValueError("Each sample must have at least one valid language token")

        # Flatten (batch, camera, frame) to run the vision encoder once per
        # (sample, camera, frame) triple.
        flat_count = bsize * num_cameras * num_frames
        flat_images = images.reshape(flat_count, *images.shape[3:])
        flat_images = self._apply_siglip_normalisation(flat_images).to(
            dtype=self.model_dtype
        )

        image_ctx = torch.no_grad() if self.cfg.freeze_vision_encoder else nullcontext()
        with image_ctx:
            vision_feats = self._encode_vision(flat_images)

        lang_ctx = torch.no_grad() if self.cfg.freeze_language_model else nullcontext()
        with lang_ctx:
            lang_feat_raw = self._encode_prompt(
                input_ids=input_ids, attention_mask=language_mask.long()
            )

        # Under plain eager, projector weights often stay fp32; under FSDP
        # mixed precision they may be bf16/fp16. Always cast features to the
        # receiving module's dtype so Linear/LayerNorm matmuls stay consistent.
        image_projector_dtype = _module_parameter_dtype(
            self.image_projector,
            vision_feats.dtype,
        )
        projected = self.image_projector(vision_feats.to(dtype=image_projector_dtype))
        projected = projected.view(
            bsize, num_cameras, num_frames, self.cfg.fusion_hidden_dim
        )

        # Mean-pool across cameras *per frame*, respecting the per-slot
        # mask — missing camera slots contribute nothing and don't dilute
        # the average.
        cam_mask_float = image_attention_mask.to(dtype=projected.dtype).unsqueeze(-1)
        projected_masked = projected * cam_mask_float  # [B, Nc, Nf, D]
        cam_counts = cam_mask_float.sum(dim=1).clamp_min(1.0)  # [B, Nf, 1]
        per_frame_features = projected_masked.sum(dim=1) / cam_counts  # [B, Nf, D]

        language_projector_dtype = _module_parameter_dtype(
            self.language_projector,
            lang_feat_raw.dtype,
        )
        lang = self.language_projector(lang_feat_raw.to(dtype=language_projector_dtype))
        return self._fuse(per_frame_features, lang), per_frame_features, lang

    def _compute_features(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        """End-to-end encoder + fusion.

        Args:
            input_ids: ``[B, T]`` token ids for the (state-aware) prompt.
            attention_mask: ``[B, T]`` prompt mask.
            images: ``[B, num_cameras, num_frames, C, H, W]`` — per-camera
                per-frame pair images in ``[0, 1]`` BCHW, as emitted by
                :class:`BinaryPairDataCollator`. Camera order is sorted by
                key; frame order is ``(t, t+k)``.
            image_attention_mask: ``[B, num_cameras, num_frames]`` bool —
                zeros mask missing camera/frame slots.

        Returns:
            Fused feature of shape ``[B, fusion_hidden_dim * (num_frames + 1)]``.
        """
        fused, _per_frame, _lang = self._compute_projected_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )
        return fused

    def compute_compatibility_logits(
        self,
        per_frame_image_features: Tensor,
        language_feature: Tensor,
        states: Tensor,
    ) -> Tensor:
        """Score whether each normalized state matches its image/task frame.

        Args:
            per_frame_image_features: ``[B, num_frames, D]`` from
                :meth:`_compute_projected_features`.
            language_feature: ``[B, D]`` from :meth:`_compute_projected_features`.
            states: ``[B, num_frames, max_state_dim]`` normalized states in
                the same frame order as ``per_frame_image_features``.

        Returns:
            Raw BCE logits of shape ``[B, num_frames]``.
        """
        if not hasattr(self, "state_projector") or not hasattr(
            self, "compatibility_head"
        ):
            raise RuntimeError(
                "State compatibility head is not initialized. Set "
                "use_state_compatibility=true in SteamConfig."
            )
        if states.ndim != 3:
            raise ValueError(
                f"states must have shape [B, num_frames, D], got {tuple(states.shape)}"
            )
        if states.shape[:2] != per_frame_image_features.shape[:2]:
            raise ValueError(
                "states and per_frame_image_features must share batch/frame dims: "
                f"states={tuple(states.shape)}, "
                f"features={tuple(per_frame_image_features.shape)}"
            )
        if language_feature.shape[0] != states.shape[0]:
            raise ValueError(
                "language_feature batch dim must match states; got "
                f"{tuple(language_feature.shape)} vs {tuple(states.shape)}"
            )

        state_dtype = _module_parameter_dtype(self.state_projector, states.dtype)
        state_features = self.state_projector(states.to(dtype=state_dtype))
        compat_dtype = state_features.dtype
        per_frame = per_frame_image_features.to(dtype=compat_dtype)
        language_expanded = (
            language_feature.to(dtype=compat_dtype)
            .unsqueeze(1)
            .expand(-1, states.shape[1], -1)
        )
        compat_input = torch.cat(
            [per_frame, language_expanded, state_features],
            dim=-1,
        )
        head_dtype = _module_parameter_dtype(
            self.compatibility_head, compat_input.dtype
        )
        return self.compatibility_head(compat_input.to(dtype=head_dtype)).squeeze(-1)

    def _check_shapes(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                f"'input_ids' must have shape [B, T], got {tuple(input_ids.shape)}"
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "Batch/length mismatch between input_ids and attention_mask"
            )
        if images.ndim != 6:
            raise ValueError(
                "'images' must have shape [B, num_cameras, num_frames, C, H, W], "
                f"got {tuple(images.shape)}"
            )
        if (
            image_attention_mask.ndim != 3
            or image_attention_mask.shape[:3] != images.shape[:3]
        ):
            raise ValueError(
                "Batch/camera/frame mismatch between images and image_attention_mask"
            )
        expected_frames = self.cfg.num_frames_per_pair
        if images.shape[2] != expected_frames:
            raise ValueError(
                f"num_frames_per_pair={expected_frames} but got {images.shape[2]} "
                "frame slots"
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        images: Tensor,
        image_attention_mask: Tensor,
    ) -> Tensor:
        """Return raw 2-class logits ``[B, 2]``."""
        features = self._compute_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_attention_mask,
        )
        return self.value_head(features)


__all__ = [
    "CLASS_PROGRESS",
    "CLASS_REGRESS",
    "NUM_CLASSES",
    "SteamBackbone",
]
