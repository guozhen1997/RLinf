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

"""RoboTTT's DiT: GR00T N1.7's action-head DiT with TTT layers along time.

The upstream ``AlternateVLDiT`` attends *within* a single timestep: its
self-attention sees the state/action tokens of one observation and its
cross-attention sees the vision-language tokens of that same observation. This
subclass inserts a TTT layer after every block so information also flows
*across* timesteps, through fast weights rather than through attention over a
growing history.

Parameter names are inherited unchanged from ``AlternateVLDiT`` so a pretrained
GR00T N1.7 checkpoint still loads; the TTT parameters are new keys that
``from_pretrained`` reports as missing and leaves at their initialization.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from gr00t.model.modules.dit import AlternateVLDiT, _sdpa_context
from torch import nn

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    TTTConfig,
    TTTContext,
    TTTTimeMixer,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def resolve_ttt_config(
    ttt_config: Any, hidden_size: int, num_heads: int
) -> TTTConfig:
    """Build a :class:`TTTConfig`, defaulting its width to the host model's."""
    defaults = {"hidden_size": hidden_size, "num_heads": num_heads}
    if isinstance(ttt_config, TTTConfig):
        return ttt_config
    if ttt_config is None:
        return TTTConfig(**defaults)
    overrides = {
        key: value for key, value in dict(ttt_config).items() if value is not None
    }
    defaults.update(overrides)
    return TTTConfig(**defaults)


def _nonempty_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Keep at least one unmasked key per row so SDPA softmax stays defined.

    AlternateVLDiT splits VL tokens into image vs text. A row whose subset is
    empty (all False) makes ``scaled_dot_product_attention`` return NaN, which
    then poisons TTT fast weights and the rest of the DiT.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    empty = ~mask.any(dim=-1)
    if not bool(empty.any()):
        return mask
    filled = mask.clone()
    filled[empty, 0] = True
    return filled


def _attention_residual(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    encoder_attention_mask: Optional[torch.Tensor],
    temb: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run a ``BasicTransformerBlock`` through the attention residual only.

    RoboTTT Eq. (3) adds the gated TTT output to the *attention* output, then
    the feed-forward continues. The upstream block fuses Attn+FFN, so this
    helper (and :func:`_feed_forward`) split it without changing Isaac-GR00T.
    """
    if block.norm_type == "ada_norm":
        norm_hidden_states = block.norm1(hidden_states, temb)
    else:
        norm_hidden_states = block.norm1(hidden_states)
    if block.pos_embed is not None:
        norm_hidden_states = block.pos_embed(norm_hidden_states)
    attn_mask = (
        _nonempty_bool_mask(encoder_attention_mask)
        if encoder_hidden_states is not None and encoder_attention_mask is not None
        else None
    )
    with _sdpa_context():
        attn_output = block.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attn_mask,
        )
    if block.final_dropout:
        attn_output = block.final_dropout(attn_output)
    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    return hidden_states


def _feed_forward(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """The feed-forward half of ``BasicTransformerBlock.forward``."""
    norm_hidden_states = block.norm3(hidden_states)
    ff_output = block.ff(norm_hidden_states)
    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    return hidden_states


def install_ttt_into_block_forward(
    block: nn.Module,
    mixer: TTTTimeMixer,
    layer_id: int,
    context_holder: Any,
) -> None:
    """Put Attn → TTT → FFN inside ``block.forward``.

    FSDP wraps each ``BasicTransformerBlock`` and only all-gathers its
    parameters when that module's ``forward`` runs. The TTT layer and gate are
    therefore registered as children of the block (not of the mixer) so they
    unshard together with AdaLN / attention.
    """
    gate_snapshot = None
    installed_gate = getattr(block, "ttt_gate", None)
    if installed_gate is not None:
        if installed_gate.device.type == "meta":
            gate_snapshot = torch.full(
                tuple(installed_gate.shape),
                float(mixer.config.gate_init),
                dtype=torch.float32,
                device="cpu",
            )
        else:
            gate_snapshot = installed_gate.detach().cpu().float().clone()
        block._ttt_gate_snapshot = gate_snapshot

    def _restore_gate() -> None:
        gate = getattr(block, "ttt_gate", None)
        snap = getattr(block, "_ttt_gate_snapshot", None)
        if (
            gate is None
            or snap is None
            or getattr(block, "_ttt_gate_restored", False)
        ):
            return
        if gate.device.type == "meta" or tuple(gate.shape) != tuple(snap.shape):
            return
        if gate._is_view():
            # FSDP all-gathered view: inplace copy_ breaks autograd.
            block._ttt_gate_restored = True
            return
        current = gate.detach().float()
        abs_max = float(current.abs().amax()) if current.numel() else 0.0
        block._ttt_gate_restored = True
        if bool(torch.isfinite(current).all()) and abs_max <= 10.0:
            return
        logger.warning(
            "TTT gate corrupted (abs_max=%s); restoring construction init.",
            abs_max,
        )
        with torch.no_grad():
            gate.copy_(snap.to(device=gate.device, dtype=gate.dtype))

    def forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        del attention_mask
        _restore_gate()
        hidden_states = _attention_residual(
            block,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            temb=temb,
        )
        hidden_states = mixer.apply_layer(
            layer_id,
            hidden_states,
            getattr(context_holder, "_ttt_context", None),
            layer=getattr(block, "ttt_layer", None),
            gate=getattr(block, "ttt_gate", None),
        )
        return _feed_forward(block, hidden_states)

    block.forward = forward


class TTTAlternateVLDiT(AlternateVLDiT):
    """``AlternateVLDiT`` with a gated TTT layer after each attention residual."""

    def __init__(self, *args, ttt_config: Any = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.ttt_config = resolve_ttt_config(
            ttt_config, self.inner_dim, self.config.num_attention_heads
        )
        self.ttt = TTTTimeMixer(self.ttt_config, self.config.num_layers)
        self._ttt_context: Optional[TTTContext] = None
        layers = list(self.ttt.layers)
        gates = list(self.ttt.gates)
        # The mixer keeps ``num_layers`` for TTT state layout, but the actual
        # parameters must live on the transformer block FSDP wraps.
        self.ttt.layers = nn.ModuleList()
        self.ttt.gates = nn.ParameterList()
        for idx, block in enumerate(self.transformer_blocks):
            block.add_module("ttt_layer", layers[idx])
            block.register_parameter("ttt_gate", gates[idx])
            install_ttt_into_block_forward(block, self.ttt, idx, self)

    def restore_uninitialized_params_(self) -> None:
        """Fix HF-uninitialized TTT weights before FSDP wrap."""
        for block in self.transformer_blocks:
            layer = getattr(block, "ttt_layer", None)
            if layer is not None and hasattr(layer, "restore_uninitialized_params_"):
                layer.restore_uninitialized_params_()
            gate = getattr(block, "ttt_gate", None)
            snap = getattr(block, "_ttt_gate_snapshot", None)
            if not isinstance(gate, nn.Parameter) or not torch.is_tensor(snap):
                continue
            if gate.device.type == "meta" or gate._is_view():
                continue
            current = gate.detach().float()
            abs_max = float(current.abs().amax()) if current.numel() else 0.0
            block._ttt_gate_restored = True
            if bool(torch.isfinite(current).all()) and abs_max <= 10.0:
                continue
            logger.warning(
                "TTT gate corrupted (abs_max=%s); restoring construction init.",
                abs_max,
            )
            with torch.no_grad():
                gate.copy_(snap.to(device=gate.device, dtype=gate.dtype))

    def unfreeze_ttt(self) -> None:
        """Mark TTT layers/gates trainable after they were moved onto blocks."""
        for block in self.transformer_blocks:
            if hasattr(block, "ttt_layer"):
                block.ttt_layer.requires_grad_(True)
            gate = getattr(block, "ttt_gate", None)
            if isinstance(gate, torch.nn.Parameter):
                gate.requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        ttt_context: Optional[TTTContext] = None,
    ):
        """Same contract as ``AlternateVLDiT.forward`` plus a TTT context.

        With ``ttt_context=None`` the TTT layers still run, but over a single
        timestep starting from ``W_0``: that is the context-length-one
        behaviour, which keeps upstream call sites working unchanged.

        Computation per layer matches the paper: attention (within a timestep)
        → gated TTT (across time) added to the attention residual → FFN.

        TTT is installed *inside* each ``BasicTransformerBlock.forward`` so
        FSDP unshards the block — including the TTT layer registered on it —
        before AdaLN / attention / TTT see the weights.
        """
        self._ttt_context = ttt_context
        try:
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                return_all_hidden_states=return_all_hidden_states,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        finally:
            self._ttt_context = None
