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

"""Ensemble wrapper for the STEAM value critic.

This module owns every ensemble-only concept in the stack:

* :class:`EnsembleCriticOutput` — extends :class:`CriticOutput` with
  member-wise and aggregate prediction statistics that only make sense
  under a deep ensemble.
* :class:`EnsembleSteamCriticModel` — the
  :class:`SteamCriticModel` members are the ones that actually
  produce logits; this class is purely responsible for cloning /
  re-seeding members, per-member training orchestration (via
  ``forward(..., member_idx=int)``), and aggregating member predictions
  into a single inference output under the configured
  ``inference_mode``.

The single-model path in :mod:`modeling_critic` intentionally knows
nothing about ensembles — no ``member_predicted_values`` fake-stats
padding, no ``hasattr(model, "members")`` duck typing. That separation
is the whole reason this module exists.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .configuration import SteamConfig
from .modeling_critic import CriticOutput, SteamCriticModel

logger = logging.getLogger(__name__)


@dataclass
class EnsembleCriticOutput(CriticOutput):
    """CriticOutput extended with ensemble aggregate stats.

    All four fields are only populated by
    :meth:`EnsembleSteamCriticModel.predict` (the inference path).
    The per-member training path returns a plain :class:`CriticOutput`
    with these fields absent — there's no cross-member aggregate to
    report when each member is trained on its own independent random
    micro-batch.
    """

    member_predicted_values: Optional[torch.FloatTensor] = None
    # Per-member softmax distribution and raw logits, shape [E, B, num_bins].
    # Only populated on the inference path (predict / forward with
    # labels is None). Consumers (rich visualization, advantage-pipeline
    # entropy stats) rely on these; the single-model forward does not
    # have an ensemble axis so :class:`CriticOutput` intentionally omits
    # them.
    member_probs: Optional[torch.FloatTensor] = None
    member_logits: Optional[torch.FloatTensor] = None
    member_progress_values: Optional[torch.FloatTensor] = None
    member_compatibility_logits: Optional[torch.FloatTensor] = None
    member_compatibility_probs: Optional[torch.FloatTensor] = None
    member_final_values: Optional[torch.FloatTensor] = None
    prediction_mean: Optional[torch.FloatTensor] = None
    prediction_min: Optional[torch.FloatTensor] = None
    prediction_variance: Optional[torch.FloatTensor] = None


def clone_ensemble_members(
    base_member: nn.Module,
    ensemble_size: int,
) -> list[nn.Module]:
    """Clone a base member ``ensemble_size`` times."""
    if ensemble_size < 1:
        raise ValueError(f"ensemble_size must be >= 1, got {ensemble_size}")
    return [base_member] + [
        copy.deepcopy(base_member) for _ in range(ensemble_size - 1)
    ]


def _reinitialize_module_parameters(module: nn.Module, seed: int) -> None:
    """Reset all resettable submodules under ``module`` with a fixed seed."""
    cuda_devices = sorted(
        {
            int(parameter.device.index)
            for parameter in module.parameters()
            if parameter.is_cuda and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        for submodule in module.modules():
            if hasattr(submodule, "reset_parameters"):
                submodule.reset_parameters()


def reinitialize_member_value_heads(
    members: list[nn.Module],
    head_seed_base: int,
) -> None:
    """Reinitialize each member's trainable heads with a distinct seed."""
    for member_idx, member in enumerate(members):
        backbone = getattr(member, "model", None)
        value_head = getattr(backbone, "value_head", None)
        if value_head is None:
            raise AttributeError(
                f"Ensemble member {member_idx} does not expose model.value_head"
            )
        _reinitialize_module_parameters(value_head, int(head_seed_base) + member_idx)
        compatibility_head = getattr(backbone, "compatibility_head", None)
        if compatibility_head is not None:
            _reinitialize_module_parameters(
                compatibility_head,
                int(head_seed_base) + 10_000 + member_idx,
            )


def build_ensemble_members(
    base_member: nn.Module,
    ensemble_size: int,
    head_seed_base: int,
) -> list[nn.Module]:
    """Clone a base member and reinitialize only the value heads."""
    members = clone_ensemble_members(base_member, ensemble_size)
    reinitialize_member_value_heads(members, head_seed_base)
    return members


class EnsembleSteamCriticModel(nn.Module):
    """Deep-ensemble wrapper for :class:`SteamCriticModel`.

    Supports both the legacy binary (num_bins == 2) and the multi-bin
    (num_bins > 2) head shapes. The per-member call returns logits /
    probs of shape ``[B, num_bins]`` in either mode, and the aggregator
    below reduces across the ensemble axis according to
    ``config.inference_mode``.

    Aggregation contract. ``predicted_values`` carries the single-model
    ``_predicted_signed_value`` output — a bin-weighted, ``half``-normalized
    expectation in ``[-1, 1]`` (see
    :meth:`SteamCriticModel._predicted_signed_value`). The ensemble
    preserves that scale across all three modes:

        * ``mo``  — ``aggregated_probs`` is the per-bin member mean, and
          ``aggregated`` is the signed-value of that mean distribution.
          By linearity it equals ``prediction_mean``, so the equality
          ``signed_value(aggregated_probs) == predicted_values`` holds
          by construction.
        * ``wco`` — gather the worst member's logits / probs per batch
          item (worst = lowest signed value, i.e. most regressive). The
          gathered distribution is a real single-member distribution and
          its signed-value equals ``prediction_min``, so the equality
          above still holds.
        * ``uwo`` — apply the mean-minus-``λ``·variance penalty directly
          in the ``[-1, 1]`` signed-value space (no logit/sigmoid
          round-trip) and clamp to ``[-1, 1]``. ``aggregated_logits /
          probs`` fall back to the member mean for reporting only — they
          are **not** the distribution that produced ``aggregated`` in
          UWO mode, so the equality above does **not** hold in UWO.
          Downstream code in this repo consumes ``predicted_values``
          (plus the ensemble stats), so the UWO-specific divergence is
          safe.
    """

    def __init__(
        self,
        config: SteamConfig,
        members: list[SteamCriticModel],
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("EnsembleSteamCriticModel requires at least one member")

        self.config = config
        self.members = nn.ModuleList(members)
        self.gradient_checkpointing_enabled = False

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return self.members[0]._no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return self.members[0]._no_split_names

    def gradient_checkpointing_enable(self) -> None:
        if self.gradient_checkpointing_enabled:
            return
        self.gradient_checkpointing_enabled = True
        for member in self.members:
            member.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if not self.gradient_checkpointing_enabled:
            return
        self.gradient_checkpointing_enabled = False
        for member in self.members:
            member.gradient_checkpointing_disable()

    def attach_runtime_assets(self, processor, device) -> None:
        """Attach inference-time runtime assets to self AND every member.

        Overrides :meth:`SteamCriticModel.attach_runtime_assets`:
        the per-member ``_prepare_observation*`` delegations in this class
        call into ``members[0]``. Every member needs the same processor and
        device target for collator-prepared pair observations.
        """
        self.processor = processor
        self._device = device
        for member in self.members:
            member.attach_runtime_assets(processor, device)

    @staticmethod
    def _gather_member_batch_values(
        member_tensor: Tensor, member_indices: Tensor
    ) -> Tensor:
        """Gather one member prediction per batch item from ``[E, B, ...]`` tensors."""
        batch_indices = torch.arange(
            member_tensor.shape[1], device=member_tensor.device
        )
        return member_tensor[member_indices, batch_indices]

    def _aggregate_member_predictions(
        self,
        member_logits: Tensor,
        member_probs: Tensor,
        member_predicted_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        prediction_mean = member_predicted_values.mean(dim=0)
        prediction_min, worst_member_indices = member_predicted_values.min(dim=0)
        prediction_variance = member_predicted_values.var(dim=0, unbiased=False)

        if self.config.inference_mode == "mo":
            aggregated_probs = member_probs.mean(dim=0)  # [B, num_bins]
            # By linearity of expectation, the signed-value of the
            # mean-of-probs equals the mean of member signed-values. Use
            # the pre-computed ``prediction_mean`` so ``predicted_values``
            # stays bit-identical with ``prediction_mean`` in mo mode.
            aggregated = prediction_mean
            # Reporting-only: the mean-of-logits does not recover
            # ``aggregated`` via softmax, but the mean-of-probs does.
            # Keep the member-logit mean so downstream code that
            # inspects the logits sees a sensible per-bin signal.
            aggregated_logits = member_logits.mean(dim=0)
        elif self.config.inference_mode == "wco":
            aggregated_logits = self._gather_member_batch_values(
                member_logits,
                worst_member_indices,
            )
            aggregated_probs = self._gather_member_batch_values(
                member_probs,
                worst_member_indices,
            )
            # Worst member = lowest signed value = most regressive. The
            # gathered distribution's signed-value equals prediction_min
            # by construction (worst_member_indices come from argmin on
            # member_predicted_values), so reuse it directly.
            aggregated = prediction_min
        elif self.config.inference_mode == "uwo":
            # UWO in signed-value space: ``mean - λ · variance`` applied
            # directly on the ``[-1, 1]`` score, with a final clamp to
            # keep ``predicted_values`` in range. No logit/sigmoid
            # round-trip — the signed score already encodes direction
            # and strength, so penalizing disagreement on this scale is
            # the native form. Note: ``uwo_lambda`` is now a coefficient
            # on variance in ``[-1, 1]`` space rather than log-odds
            # space, so its tuned magnitude will differ from the old
            # logit-based formulation.
            aggregated_margin = prediction_mean - (
                self.config.uwo_lambda * prediction_variance
            )
            aggregated = aggregated_margin.clamp(min=-1.0, max=1.0).to(
                dtype=member_predicted_values.dtype
            )
            # Reporting-only: member means do NOT match ``aggregated``
            # under UWO. Documented on the class docstring; downstream
            # consumers only read ``predicted_values`` plus the
            # ensemble stats, so this divergence is safe here.
            aggregated_probs = member_probs.mean(dim=0)
            aggregated_logits = member_logits.mean(dim=0)
        else:
            raise ValueError(
                f"Unsupported inference_mode: {self.config.inference_mode}"
            )

        return (
            aggregated,
            aggregated_logits,
            aggregated_probs,
            prediction_mean,
            prediction_min,
            prediction_variance,
        )

    def forward(
        self,
        observation,
        labels=None,
        *,
        member_idx: Optional[int] = None,
        **kwargs,
    ) -> CriticOutput:
        """Training / inference dispatch.

        * ``labels is None`` → inference; delegates to :meth:`predict`,
          which returns a full :class:`EnsembleCriticOutput` with member
          and aggregate stats.
        * ``labels is not None`` → training; ``member_idx`` **must** be an
          int. The caller (the FSDP SFT worker) drives an outer loop
          over members and feeds each member its OWN fresh micro batch
          from the dataloader, so each member's training trajectory
          sees independent random data — the bagging-style randomness
          that makes ensemble prediction variance a meaningful epistemic
          uncertainty signal. There is intentionally no
          "split one batch across members" path: it wouldn't give
          independent random data (members would see disjoint slices of
          the same micro batch) and it isn't exercised by any caller in
          this repo.
        """
        if labels is None:
            return self.predict(observation)

        if member_idx is None:
            raise ValueError(
                "EnsembleSteamCriticModel.forward requires member_idx "
                "to be an int during training. The ensemble worker must drive "
                "a per-member outer loop and feed each member its own micro "
                "batch; there is no parallel batch-slicing path."
            )

        member = self.members[int(member_idx)]
        member_output = member(observation=observation, labels=labels, **kwargs)
        if member_output.loss is None:
            raise RuntimeError(
                f"Ensemble member {int(member_idx)} returned no loss during training"
            )
        return member_output

    @torch.no_grad()
    def predict(self, observation) -> EnsembleCriticOutput:
        member_outputs = [member.predict(observation) for member in self.members]
        member_logits = torch.stack([output.logits for output in member_outputs], dim=0)
        member_probs = torch.stack([output.probs for output in member_outputs], dim=0)
        member_predicted_values = torch.stack(
            [output.predicted_values for output in member_outputs],
            dim=0,
        )
        member_progress_values = None
        if all(output.progress_values is not None for output in member_outputs):
            member_progress_values = torch.stack(
                [output.progress_values for output in member_outputs],
                dim=0,
            )
        member_compatibility_logits = None
        if all(output.compatibility_logits is not None for output in member_outputs):
            member_compatibility_logits = torch.stack(
                [output.compatibility_logits for output in member_outputs],
                dim=0,
            )
        member_compatibility_probs = None
        if all(output.compatibility_probs is not None for output in member_outputs):
            member_compatibility_probs = torch.stack(
                [output.compatibility_probs for output in member_outputs],
                dim=0,
            )
        member_final_values = None
        if all(output.final_values is not None for output in member_outputs):
            member_final_values = torch.stack(
                [output.final_values for output in member_outputs],
                dim=0,
            )
        (
            aggregated,
            aggregated_logits,
            aggregated_probs,
            prediction_mean,
            prediction_min,
            prediction_variance,
        ) = self._aggregate_member_predictions(
            member_logits,
            member_probs,
            member_predicted_values,
        )

        return EnsembleCriticOutput(
            predicted_values=aggregated,
            logits=aggregated_logits,
            probs=aggregated_probs,
            atoms=None,
            hidden_states=None,
            progress_values=member_progress_values.mean(dim=0)
            if member_progress_values is not None
            else None,
            compatibility_logits=member_compatibility_logits.mean(dim=0)
            if member_compatibility_logits is not None
            else None,
            compatibility_probs=member_compatibility_probs.mean(dim=0)
            if member_compatibility_probs is not None
            else None,
            final_values=aggregated if member_final_values is not None else None,
            member_predicted_values=member_predicted_values,
            # member_{logits,probs} let downstream tools (rich viz,
            # advantage parquet) compute per-member entropy / expected
            # stride without re-running inference. Shape [E, B, num_bins].
            member_probs=member_probs,
            member_logits=member_logits,
            member_progress_values=member_progress_values,
            member_compatibility_logits=member_compatibility_logits,
            member_compatibility_probs=member_compatibility_probs,
            member_final_values=member_final_values,
            prediction_mean=prediction_mean,
            prediction_min=prediction_min,
            prediction_variance=prediction_variance,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        return self.predict(observation).predicted_values

    @staticmethod
    def _prepare_observation_cpu(inputs: dict, processor) -> dict:
        return SteamCriticModel._prepare_observation_cpu(inputs, processor)

    def _prepare_observation(self, inputs: dict) -> dict:
        return self.members[0]._prepare_observation(inputs)

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        return self.members[0]._prepare_observation_batch(inputs_list)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        del obs
        raise RuntimeError(
            "EnsembleSteamCriticModel is a pair-classification model and does "
            "not accept single-frame raw observations. Use BinaryPairDataCollator "
            "to build a pair observation, then call predict(observation)."
        )

    @torch.no_grad()
    def infer_batch(
        self,
        obs_list: list[dict],
        *,
        batch_size: int = 64,
        pretransformed: bool = False,
        already_cpu_prepared: bool = False,
    ) -> list[dict]:
        del obs_list, batch_size, pretransformed, already_cpu_prepared
        raise RuntimeError(
            "EnsembleSteamCriticModel is a pair-classification model and does "
            "not accept raw observation batches. Use BinaryPairDataCollator "
            "to build pair observations, then call predict(observation)."
        )

    @classmethod
    def from_checkpoint(cls, *args, **kwargs):
        from .modeling_critic import SteamCriticModel

        return SteamCriticModel.from_checkpoint(*args, **kwargs)


__all__ = [
    "EnsembleSteamCriticModel",
    "EnsembleCriticOutput",
    "build_ensemble_members",
    "clone_ensemble_members",
    "reinitialize_member_value_heads",
]
