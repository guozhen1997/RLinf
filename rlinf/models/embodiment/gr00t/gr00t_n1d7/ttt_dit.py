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
    with _sdpa_context():
        attn_output = block.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=(
                encoder_attention_mask if encoder_hidden_states is not None else None
            ),
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


class TTTAlternateVLDiT(AlternateVLDiT):
    """``AlternateVLDiT`` with a gated TTT layer after each attention residual."""

    def __init__(self, *args, ttt_config: Any = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.ttt_config = resolve_ttt_config(
            ttt_config, self.inner_dim, self.config.num_attention_heads
        )
        self.ttt = TTTTimeMixer(self.ttt_config, self.config.num_layers)

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
        """
        del encoder_attention_mask
        assert image_mask is not None, "Image mask is required"

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask

        all_hidden_states = [hidden_states]
        assert self.config.interleave_self_attention, (
            "Interleave self attention must be enabled"
        )

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = _attention_residual(
                    block,
                    hidden_states,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_encoder_attention_mask = non_image_attention_mask
                else:
                    curr_encoder_attention_mask = image_attention_mask
                hidden_states = _attention_residual(
                    block,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_encoder_attention_mask,
                    temb=temb,
                )

            hidden_states = self.ttt.apply_layer(idx, hidden_states, ttt_context)
            hidden_states = _feed_forward(block, hidden_states)
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(nn.functional.silu(temb)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)
