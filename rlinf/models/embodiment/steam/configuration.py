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

"""STEAM value model configuration.

Shares the SigLIP + Gemma3 backbone knobs and state-in-prompt fields with
the sibling embodied value models, but:

* the categorical-value fields (``num_bins`` / ``bin_min`` / ``bin_max``) are
  replaced by ``label_smoothing`` because the head is a single-logit binary
  classifier, and
* ``action_dim`` / ``action_horizon`` are dropped because a ``(frame_t,
  frame_{t+k})`` pair model doesn't consume actions.

A new ``num_frames_per_pair`` knob (default 2) controls the image-token
fanout in ``modeling_steam.SteamBackbone``.
"""

from typing import Optional

from transformers import PretrainedConfig

VALID_STEAM_INFERENCE_MODES = ("mo", "wco", "uwo")
VALID_STEAM_TARGET_MODES = ("rewind", "positive_only")
VALID_COMPATIBILITY_NEGATIVE_MODES = (
    "none",
    "same_episode_distance_weighted",
    "perturb",
    "same_episode_distance_weighted_plus_perturb",
)
VALID_COMPATIBILITY_GATE_FRAMES = ("t", "tk", "mean", "min")


def normalize_steam_inference_mode(mode: str) -> str:
    """Normalise an ensemble inference mode string."""
    mode_norm = str(mode).strip().lower()
    if mode_norm not in VALID_STEAM_INFERENCE_MODES:
        raise ValueError(
            f"inference_mode must be one of {VALID_STEAM_INFERENCE_MODES}, "
            f"got {mode!r}"
        )
    return mode_norm


def normalize_steam_target_mode(mode: str) -> str:
    """Normalise the training target layout string."""
    mode_norm = str(mode).strip().lower()
    if mode_norm in ("positive-only", "positive", "norewind", "no_rewind"):
        mode_norm = "positive_only"
    if mode_norm not in VALID_STEAM_TARGET_MODES:
        raise ValueError(
            f"target_mode must be one of {VALID_STEAM_TARGET_MODES}, "
            f"got {mode!r}"
        )
    return mode_norm


def validate_steam_ensemble_settings(
    *,
    ensemble_size: int,
    inference_mode: str,
    uwo_lambda: float,
    micro_batch_size: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> tuple[int, str, float]:
    """Validate binary-value ensemble settings shared by training and inference.

    ``micro_batch_size`` / ``global_batch_size`` are accepted for call-site
    parity but are intentionally not constrained: per-member sequential
    training treats both as the per-member batch (each member fetches its
    own micro batches independently), so divisibility by ``ensemble_size``
    is no longer required.
    """
    del micro_batch_size, global_batch_size  # accepted for backward compat

    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")

    inference_mode = normalize_steam_inference_mode(inference_mode)
    uwo_lambda = float(uwo_lambda)
    if uwo_lambda < 0.0:
        raise ValueError("uwo_lambda must be >= 0")

    return ensemble_size, inference_mode, uwo_lambda


class SteamConfig(PretrainedConfig):
    """Configuration for the :class:`SteamCriticModel`.

    Uses a SigLIP vision encoder (default ``google/siglip-so400m-patch14-384``,
    native 384x384) and a Gemma3-270m language backbone, fused via
    per-frame-concat + 2-layer MLP, with a scalar binary logit head.
    """

    model_type = "steam"

    def __init__(
        self,
        # Backbones
        vision_repo_id: str = "",
        language_repo_id: str = "",
        vision_revision: Optional[str] = None,
        language_revision: Optional[str] = None,
        # Fusion + binary / multi-bin head
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.05,
        num_frames_per_pair: int = 2,
        # num_bins == 2 → legacy binary mode (fixed k, labels are long
        # bin indices in {0, 1}: 0 = regress, 1 = progress).
        # num_bins  > 2 → multi-bin mode: pair_dataset samples i ∈ [1, K],
        # signed stride in [-K, K] \ {0} is discretized into num_bins
        # contiguous bins. Must be even so the sign split lands exactly at
        # num_bins // 2.
        num_bins: int = 2,
        target_mode: str = "rewind",
        stride_k: Optional[int] = None,
        ensemble_size: int = 1,
        inference_mode: str = "mo",
        uwo_lambda: float = 1.0,
        ensemble_head_seed_base: Optional[int] = None,
        # Runtime
        dtype: str = "bfloat16",
        precision: Optional[str] = None,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = True,
        use_gradient_checkpointing: bool = False,
        # Interface compat for SteamProcessor (state-in-prompt)
        max_token_len: int = 200,
        include_state_in_prompt: bool = True,
        max_state_dim: int = 32,
        state_discretization_bins: int = 256,
        # Optional normalized state-image compatibility branch. Disabled by
        # default so legacy configs/checkpoints preserve the original
        # pair-classification behavior.
        use_state_compatibility: bool = False,
        compatibility_loss_weight: float = 0.2,
        compatibility_negative_mode: str = "same_episode_distance_weighted_plus_perturb",
        compatibility_distance_scale: Optional[int] = None,
        compatibility_same_episode_negative_max_distance: Optional[int] = None,
        compatibility_negative_min_weight: float = 0.1,
        compatibility_num_same_episode_negatives: int = 1,
        compatibility_num_perturb_negatives: int = 1,
        compatibility_perturb_std: float = 0.03,
        compatibility_perturb_max: float = 0.12,
        compatibility_gate_value: bool = True,
        compatibility_gate_frame: str = "tk",
        compatibility_gate_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_repo_id = vision_repo_id
        self.language_repo_id = language_repo_id
        self.vision_revision = vision_revision
        self.language_revision = language_revision

        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.num_frames_per_pair = num_frames_per_pair
        self.num_bins = int(num_bins)
        self.target_mode = normalize_steam_target_mode(target_mode)
        self.stride_k = None if stride_k is None else int(stride_k)
        self.ensemble_size = int(ensemble_size)
        self.inference_mode = normalize_steam_inference_mode(inference_mode)
        self.uwo_lambda = float(uwo_lambda)
        self.ensemble_head_seed_base = (
            None if ensemble_head_seed_base is None else int(ensemble_head_seed_base)
        )

        self.dtype = precision if precision is not None else dtype
        self.precision = self.dtype
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_language_model = freeze_language_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.max_token_len = max_token_len
        self.include_state_in_prompt = include_state_in_prompt
        self.max_state_dim = max_state_dim
        self.state_discretization_bins = state_discretization_bins
        self.use_state_compatibility = bool(use_state_compatibility)
        self.compatibility_loss_weight = float(compatibility_loss_weight)
        self.compatibility_negative_mode = str(compatibility_negative_mode).lower()
        self.compatibility_distance_scale = (
            None
            if compatibility_distance_scale is None
            else int(compatibility_distance_scale)
        )
        self.compatibility_same_episode_negative_max_distance = (
            None
            if compatibility_same_episode_negative_max_distance is None
            else int(compatibility_same_episode_negative_max_distance)
        )
        self.compatibility_negative_min_weight = float(
            compatibility_negative_min_weight
        )
        self.compatibility_num_same_episode_negatives = int(
            compatibility_num_same_episode_negatives
        )
        self.compatibility_num_perturb_negatives = int(
            compatibility_num_perturb_negatives
        )
        self.compatibility_perturb_std = float(compatibility_perturb_std)
        self.compatibility_perturb_max = float(compatibility_perturb_max)
        self.compatibility_gate_value = bool(compatibility_gate_value)
        self.compatibility_gate_frame = str(compatibility_gate_frame).lower()
        self.compatibility_gate_floor = float(compatibility_gate_floor)

        self._validate()

    def _validate(self) -> None:
        if not self.vision_repo_id:
            raise ValueError(
                "SteamConfig.vision_repo_id must be a non-empty path or "
                "HF repo id"
            )
        if not self.language_repo_id:
            raise ValueError(
                "SteamConfig.language_repo_id must be a non-empty path or "
                "HF repo id"
            )
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.num_frames_per_pair < 1:
            raise ValueError("num_frames_per_pair must be >= 1")
        # num_bins must be even so the progressive / regressive split lands
        # exactly at num_bins // 2. num_bins == 2 selects the legacy binary
        # mode; num_bins > 2 activates multi-bin. An odd num_bins would leave
        # a center bin straddling signed-stride == 0, which we don't sample.
        if self.num_bins < 2 or self.num_bins % 2 != 0:
            raise ValueError(f"num_bins must be >= 2 and even, got {self.num_bins}")
        if self.target_mode == "positive_only":
            if self.stride_k is None:
                raise ValueError(
                    "positive_only target_mode requires stride_k so predicted "
                    "values can be normalized consistently."
                )
            if self.stride_k < 1:
                raise ValueError(f"stride_k must be >= 1, got {self.stride_k}")
            if self.num_bins > self.stride_k or self.stride_k % self.num_bins != 0:
                raise ValueError(
                    "positive_only target_mode requires "
                    "1 <= num_bins <= stride_k and stride_k % num_bins == 0; "
                    f"got stride_k={self.stride_k}, num_bins={self.num_bins}."
                )
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if self.uwo_lambda < 0.0:
            raise ValueError("uwo_lambda must be >= 0")
        if self.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError(
                f"dtype must be one of bfloat16/float32/float16, got {self.dtype}"
            )
        if self.max_token_len <= 0:
            raise ValueError("max_token_len must be > 0")
        if self.max_state_dim <= 0:
            raise ValueError("max_state_dim must be > 0")
        if self.state_discretization_bins < 2:
            raise ValueError("state_discretization_bins must be >= 2")
        if self.compatibility_loss_weight < 0.0:
            raise ValueError("compatibility_loss_weight must be >= 0")
        if self.compatibility_negative_mode not in VALID_COMPATIBILITY_NEGATIVE_MODES:
            raise ValueError(
                "compatibility_negative_mode must be one of "
                f"{VALID_COMPATIBILITY_NEGATIVE_MODES}, got "
                f"{self.compatibility_negative_mode!r}"
            )
        if (
            self.compatibility_distance_scale is not None
            and self.compatibility_distance_scale <= 0
        ):
            raise ValueError("compatibility_distance_scale must be null or > 0")
        if (
            self.compatibility_same_episode_negative_max_distance is not None
            and self.compatibility_same_episode_negative_max_distance <= 0
        ):
            raise ValueError(
                "compatibility_same_episode_negative_max_distance must be null or > 0"
            )
        if not 0.0 <= self.compatibility_negative_min_weight <= 1.0:
            raise ValueError("compatibility_negative_min_weight must be in [0, 1]")
        if self.compatibility_num_same_episode_negatives < 0:
            raise ValueError("compatibility_num_same_episode_negatives must be >= 0")
        if self.compatibility_num_perturb_negatives < 0:
            raise ValueError("compatibility_num_perturb_negatives must be >= 0")
        if self.compatibility_perturb_std < 0.0:
            raise ValueError("compatibility_perturb_std must be >= 0")
        if self.compatibility_perturb_max < 0.0:
            raise ValueError("compatibility_perturb_max must be >= 0")
        if self.compatibility_gate_frame not in VALID_COMPATIBILITY_GATE_FRAMES:
            raise ValueError(
                "compatibility_gate_frame must be one of "
                f"{VALID_COMPATIBILITY_GATE_FRAMES}, got "
                f"{self.compatibility_gate_frame!r}"
            )
        if not 0.0 <= self.compatibility_gate_floor <= 1.0:
            raise ValueError("compatibility_gate_floor must be in [0, 1]")

    def to_diff_dict(self) -> dict:
        """Return a full config dict without instantiating an empty default config.

        ``PretrainedConfig.to_diff_dict`` creates ``self.__class__()`` to compute
        a diff against default values. That does not work here because
        ``SteamConfig`` intentionally requires non-empty backbone ids.
        Returning the full config keeps HuggingFace save/load helpers working
        for checkpoint metadata.
        """
        return self.to_dict()
