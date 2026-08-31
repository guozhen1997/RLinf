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

"""Test-Time Training (TTT) layers used by RoboTTT.

A TTT layer is a sequence model whose recurrent state is a set of *fast
weights*: parameters of a small network :math:`f_W` that are updated by
gradient descent during both training and inference. At step :math:`t` the
layer first updates the fast weights so that :math:`f_W(K_t)` reconstructs
:math:`V_t`, then reads out :math:`O_t = f_{W_t}(Q_t)`.

The inner loop follows the mini-batch dual form of Sun et al. 2024: tokens
inside a mini-batch all differentiate the fast weights held at the start of
the mini-batch, while each token still reads out from the prefix-accumulated
update. This keeps the inner loop a handful of batched matmuls instead of a
per-token Python loop.

For RoboTTT the sequence fed to these layers is a robot trajectory flattened
as ``[step_1 tokens, step_2 tokens, ...]``, so the mini-batch size defaults to
the number of tokens per timestep: fast weights then advance exactly once per
environment step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class TTTConfig:
    """Configuration for a stack of :class:`TTTLayer`."""

    hidden_size: int = 1536
    num_heads: int = 32
    fast_model: Literal["mlp", "linear"] = "mlp"
    fast_mlp_ratio: int = 4
    base_lr: float = 0.1
    # ``None`` resolves to the number of tokens per timestep, i.e. one fast
    # weight update per environment step.
    mini_batch_size: Optional[int] = None
    max_mini_batch_size: int = 256
    use_rope: bool = True
    rope_theta: float = 10000.0
    fast_weight_init_std: float = 0.02
    # tanh gate keeping the TTT contribution near zero at the start of training
    # so a pretrained backbone is reproduced exactly.
    gate_init: float = 0.001
    # Minimum precision of the inner loop. The inner loop is a gradient step
    # and is sensitive to bf16 rounding, so it runs at least in fp32; a higher
    # precision input dtype is preserved rather than downcast.
    min_fast_weight_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        if self.fast_model not in ("mlp", "linear"):
            raise ValueError(f"Unknown fast model type: {self.fast_model}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def _seq_offset_to_tensor(
    seq_offset: int | torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Broadcast a scalar or per-row RoPE offset to ``[B]``."""
    if torch.is_tensor(seq_offset):
        offset = seq_offset.to(device=device, dtype=torch.long).reshape(-1)
        if offset.numel() == 1 and batch_size > 1:
            return offset.expand(batch_size).clone()
        if offset.numel() != batch_size:
            raise ValueError(
                f"seq_offset length {offset.numel()} does not match batch {batch_size}"
            )
        return offset
    return torch.full((batch_size,), int(seq_offset), dtype=torch.long, device=device)


@dataclass
class TTTState:
    """Recurrent state of a TTT layer: the fast weights plus a RoPE offset.

    ``seq_offset`` is stored per batch row so an env that resets mid-batch can
    restart RoPE at 0 while its neighbors keep their episode positions.
    """

    params: tuple[torch.Tensor, ...]
    seq_offset: int | torch.Tensor = 0

    def detach(self) -> "TTTState":
        """Cut the autograd graph while keeping the state values.

        This is the truncation point of truncated backpropagation through time:
        fast weights keep flowing forward, their gradients do not flow back.
        """
        return replace(self, params=tuple(p.detach() for p in self.params))

    def to(self, *args, **kwargs) -> "TTTState":
        offset = self.seq_offset
        if torch.is_tensor(offset):
            offset_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in {"device", "non_blocking"}
            }
            if offset_kwargs:
                offset = offset.to(*args, **offset_kwargs)
        return replace(
            self,
            params=tuple(p.to(*args, **kwargs) for p in self.params),
            seq_offset=offset,
        )

    def index_select(self, index: torch.Tensor) -> "TTTState":
        """Reorder / subset the batch dimension of the state."""
        offset = self.seq_offset
        if torch.is_tensor(offset):
            offset = offset.index_select(0, index.to(device=offset.device))
        return replace(
            self,
            params=tuple(p.index_select(0, index) for p in self.params),
            seq_offset=offset,
        )


def _ln_fwd(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


def _ln_fused_l2_bwd(
    x: torch.Tensor,
    l2_target: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient of ``0.5 * ||LN(x) - target||^2`` w.r.t. ``x``.

    The layer norm is folded into the inner loss (rather than applied after it)
    so the fast weights are optimized against a scale-free reconstruction
    target, which keeps the inner gradient well conditioned.
    """
    dim = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    grad_output = gamma * x_hat + beta - l2_target
    grad_x_hat = grad_output * gamma
    return (
        grad_x_hat * dim
        - grad_x_hat.sum(dim=-1, keepdim=True)
        - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
    ) / (std * dim)


def _gelu_bwd(x: torch.Tensor) -> torch.Tensor:
    """Derivative of the tanh approximation of GeLU."""
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * (
        x * (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
        + (1 + tanh_out)
    )


class _RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to the TTT query/key projections."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}.")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        offset: int | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)
        if torch.is_tensor(offset) and offset.ndim > 0:
            # Per-row offsets: [B, seq_len, head_dim]
            steps = torch.arange(seq_len, device=device, dtype=dtype)
            pos = offset.to(device=device, dtype=dtype).reshape(-1, 1) + steps
            freqs = pos.unsqueeze(-1) * inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        start = int(offset.item()) if torch.is_tensor(offset) else int(offset)
        pos = torch.arange(start, start + seq_len, device=device, dtype=dtype)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [B, num_heads, seq_len, head_dim]
    # cos/sin: [seq_len, head_dim] or [B, seq_len, head_dim]
    if cos.ndim == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    return x * cos + _rotate_half(x) * sin


def _batched_fast_weight(param: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Independent ``[B, ...]`` copy of ``W_0`` that is safe after FSDP unshard.

    ``unsqueeze().expand().clone()`` on an FSDP all-gathered view whose storage
    was just ``copy_``'d raises ViewBackward. ``repeat`` allocates new storage.
    """
    return param.repeat((batch_size,) + (1,) * param.ndim)


class TTTLayer(nn.Module):
    """Base class for TTT layers; subclasses define the fast model."""

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        dim = config.hidden_size

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Per-token, per-head inner learning rate on top of the constant base
        # learning rate. Initialized so sigmoid(.) starts near 0.5.
        self.ttt_lr_proj = nn.Linear(dim, config.num_heads, bias=True)
        self.token_idx_bias = nn.Parameter(torch.zeros(config.max_mini_batch_size))

        self.ttt_norm_weight = nn.Parameter(
            torch.ones(config.num_heads, self.head_dim)
        )
        self.ttt_norm_bias = nn.Parameter(
            torch.zeros(config.num_heads, self.head_dim)
        )
        self.post_norm = nn.LayerNorm(dim, eps=1e-6)

        self.rotary = (
            _RotaryEmbedding(self.head_dim, config.rope_theta)
            if config.use_rope
            else None
        )

        self._init_fast_weights()
        self._reset_projections()
        # CPU copy of construction init. HuggingFace ``from_pretrained``
        # (``low_cpu_mem_usage``) builds the module on the meta device, so we
        # must not ``.cpu()`` parameter storage here. Missing checkpoint keys
        # later land as uninitialized CUDA (~1e24–1e38 / NaN) and are restored
        # in-place on the first real forward after FSDP all-gather.
        self._init_snapshot = {
            name: self._snapshot_tensor(name, param)
            for name, param in self.named_parameters()
        }
        self._sane_checked = False
        self.fast_weight_resets = 0

    # ---------------------------------------------------------------- fast model

    def _init_fast_weights(self) -> None:
        raise NotImplementedError

    def _mini_batch_step(
        self,
        params: Sequence[torch.Tensor],
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        eta: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run one mini-batch of the inner loop.

        Args:
            params: fast weights entering the mini-batch, each ``[B, H, ...]``.
            xq, xk, xv: ``[B, H, M, head_dim]`` projections of the mini-batch.
            eta: ``[B, H, M, M]`` inner learning rates, ``eta[i, j]`` scaling
                the contribution of key ``j`` to the readout of query ``i``.
            ln_weight, ln_bias: ``[H, 1, head_dim]`` inner layer-norm affine.

        Returns:
            The fast weights leaving the mini-batch and the ``[B, H, M,
            head_dim]`` readout.
        """
        raise NotImplementedError

    def _initial_params(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _reset_projections(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(proj.weight, std=0.02)
        # The read-out projection starts at zero so a freshly initialized TTT
        # layer is an exact no-op even before the tanh gate is applied.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.normal_(self.ttt_lr_proj.weight, std=0.02)
        nn.init.zeros_(self.ttt_lr_proj.bias)

    def _synthesized_init(self, name: str, shape: torch.Size) -> torch.Tensor:
        """CPU init that matches construction, used when params live on meta."""
        cpu = torch.device("cpu")
        if (
            name.endswith("o_proj.weight")
            or name.endswith("ttt_lr_proj.bias")
            or name.endswith("post_norm.bias")
            or name in {"token_idx_bias", "ttt_norm_bias", "b1_init", "b2_init"}
        ):
            return torch.zeros(shape, dtype=torch.float32, device=cpu)
        if name.endswith("post_norm.weight") or name == "ttt_norm_weight":
            return torch.ones(shape, dtype=torch.float32, device=cpu)
        if name in {"w1_init", "w2_init"}:
            return torch.randn(shape, dtype=torch.float32, device=cpu) * (
                self.config.fast_weight_init_std
            )
        return torch.randn(shape, dtype=torch.float32, device=cpu) * 0.02

    def _snapshot_tensor(self, name: str, param: torch.Tensor) -> torch.Tensor:
        if param.device.type == "meta":
            return self._synthesized_init(name, param.shape)
        return param.detach().cpu().float().clone()

    def restore_uninitialized_params_(self) -> None:
        """Copy construction init over HF-uninitialized CUDA storage.

        Must run *before* FSDP wrap. Inplace ``copy_`` on FSDP all-gathered
        views is rejected by autograd (the subsequent ``unsqueeze`` of
        ``w1_init`` in ``_initial_params`` raises ViewBackward).
        """
        if self._sane_checked:
            return
        snapshot = getattr(self, "_init_snapshot", None)
        if not snapshot:
            self._sane_checked = True
            return

        named = dict(self.named_parameters())
        bad: list[tuple[str, float]] = []
        for name, snap in snapshot.items():
            param = named.get(name)
            if param is None or param.device.type == "meta":
                return
            if param._is_view():
                # FSDP all-gathered orig params are views of the flat tensor.
                # Inplace copy_ here makes _initial_params' batch expand illegal.
                self._sane_checked = True
                return
            if tuple(param.shape) != tuple(snap.shape):
                return
            current = param.detach().float()
            abs_max = float(current.abs().amax()) if current.numel() else 0.0
            if not bool(torch.isfinite(current).all()) or abs_max > 10.0:
                bad.append((name, abs_max))

        self._sane_checked = True
        if not bad:
            return
        logger.warning(
            "TTT layer weights corrupted %s; restoring construction init.",
            bad[:8],
        )
        with torch.no_grad():
            for name, snap in snapshot.items():
                param = named[name]
                param.copy_(snap.to(device=param.device, dtype=param.dtype))

    def _ensure_sane_weights(self) -> None:
        """Restore only when the Parameter is a real tensor, not an FSDP view."""
        self.restore_uninitialized_params_()

    # -------------------------------------------------------------------- state

    def init_state(self, batch_size: int) -> TTTState:
        """Build the learned initial state ``W_0`` for a batch."""
        device = self.w1_init.device
        return TTTState(
            params=self._initial_params(batch_size),
            seq_offset=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def reset_state(
        self,
        state: Optional[TTTState],
        reset_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> TTTState:
        """Restore ``W_0`` (and RoPE offset 0) for rows flagged in ``reset_mask``."""
        if state is None:
            return self.init_state(batch_size)
        device = state.params[0].device
        offset = _seq_offset_to_tensor(state.seq_offset, batch_size, device)
        if reset_mask is None or not bool(reset_mask.any()):
            return replace(state, seq_offset=offset)

        init_params = self._initial_params(batch_size)
        mask = reset_mask.reshape(batch_size).to(device=device)
        params = []
        for current, initial in zip(state.params, init_params):
            view = mask.reshape((-1,) + (1,) * (current.ndim - 1))
            params.append(torch.where(view, initial.to(current.dtype), current))
        offset = torch.where(mask, torch.zeros_like(offset), offset)
        return replace(state, params=tuple(params), seq_offset=offset)

    # ------------------------------------------------------------------ forward

    def _resolve_mini_batch_size(
        self, seq_len: int, tokens_per_step: Optional[int]
    ) -> int:
        """Pick a mini-batch size that divides the sequence exactly.

        Preference order: the configured value, then the per-timestep token
        count (which always divides a flattened trajectory), then 1.
        """
        candidates = [self.config.mini_batch_size, tokens_per_step, 1]
        for candidate in candidates:
            if candidate is None or candidate <= 0:
                continue
            if candidate > self.config.max_mini_batch_size:
                continue
            if seq_len % candidate == 0:
                return candidate
        return 1

    def _inner_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        return torch.promote_types(input_dtype, self.config.min_fast_weight_dtype)

    def _compute_eta(
        self,
        hidden_states: torch.Tensor,
        num_mini_batches: int,
        mini_batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        # [B, L, H] -> [B, H, num_mb, 1, M]: the inner lr of each *key* token.
        ttt_lr = torch.sigmoid(self.ttt_lr_proj(hidden_states)).to(dtype)
        ttt_lr = ttt_lr.permute(0, 2, 1).reshape(
            batch_size, self.num_heads, num_mini_batches, 1, mini_batch_size
        )
        ttt_lr = self.config.base_lr * ttt_lr / self.head_dim

        # [M, 1]: 1/(i+1) averaging over the prefix, plus a learned bias.
        token_idx = 1.0 / torch.arange(
            1, mini_batch_size + 1, device=hidden_states.device, dtype=dtype
        )
        token_idx = token_idx + self.token_idx_bias[:mini_batch_size].to(dtype)
        token_idx = token_idx.clamp_min(0.0).reshape(1, 1, 1, mini_batch_size, 1)

        return token_idx * ttt_lr

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[TTTState] = None,
        reset_mask: Optional[torch.Tensor] = None,
        tokens_per_step: Optional[int] = None,
    ) -> tuple[torch.Tensor, TTTState]:
        """Update the fast weights along ``hidden_states`` and read them out.

        Args:
            hidden_states: ``[B, L, D]`` sequence, time-major flattened as
                ``[step_1 tokens, ..., step_T tokens]``.
            state: fast weights entering the sequence; ``None`` uses ``W_0``.
            reset_mask: ``[B]`` boolean, resetting those entries to ``W_0``
                before consuming the sequence (episode boundaries at rollout).
            tokens_per_step: tokens per timestep, used to align the inner-loop
                mini-batch with environment steps.

        Returns:
            The ``[B, L, D]`` output and the fast weights leaving the sequence.
        """
        batch_size, seq_len, _ = hidden_states.shape
        input_dtype = hidden_states.dtype
        device = hidden_states.device
        self._ensure_sane_weights()

        state = self.reset_state(state, reset_mask, batch_size)
        inner_dtype = self._inner_dtype(input_dtype)
        params = tuple(p.to(device=device, dtype=inner_dtype) for p in state.params)

        xq = self._project(self.q_proj, hidden_states, inner_dtype)
        xk = self._project(self.k_proj, hidden_states, inner_dtype)
        xv = self._project(self.v_proj, hidden_states, inner_dtype)

        if self.rotary is not None:
            cos, sin = self.rotary(seq_len, state.seq_offset, device, inner_dtype)
            xq = _apply_rope(xq, cos, sin)
            xk = _apply_rope(xk, cos, sin)

        # The fast model reconstructs the *residual* V - K, which makes the
        # identity map the trivial solution and stabilizes the inner loop.
        xv = xv - xk

        mini_batch_size = self._resolve_mini_batch_size(seq_len, tokens_per_step)
        num_mini_batches = seq_len // mini_batch_size
        eta = self._compute_eta(
            hidden_states, num_mini_batches, mini_batch_size, inner_dtype
        )

        def to_mini_batches(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(
                batch_size,
                self.num_heads,
                num_mini_batches,
                mini_batch_size,
                self.head_dim,
            )

        xq_mb = to_mini_batches(xq)
        xk_mb = to_mini_batches(xk)
        xv_mb = to_mini_batches(xv)

        ln_weight = self.ttt_norm_weight.reshape(
            1, self.num_heads, 1, self.head_dim
        ).to(inner_dtype)
        ln_bias = self.ttt_norm_bias.reshape(
            1, self.num_heads, 1, self.head_dim
        ).to(inner_dtype)

        outputs = []
        for index in range(num_mini_batches):
            params, out = self._mini_batch_step(
                params,
                xq_mb[:, :, index],
                xk_mb[:, :, index],
                xv_mb[:, :, index],
                eta[:, :, index],
                ln_weight,
                ln_bias,
            )
            outputs.append(out)

        if not all(torch.isfinite(p).all() for p in params):
            self.fast_weight_resets += 1
            logger.warning(
                "TTT inner loop produced non-finite fast weights; "
                "resetting to W_0 so the rest of this sequence does not "
                "carry a poisoned recurrent state."
            )
            params = tuple(
                p.to(device=device, dtype=inner_dtype)
                for p in self._initial_params(batch_size)
            )

        # [B, H, L, head_dim] -> [B, L, D]
        readout = torch.cat(outputs, dim=2)
        readout = (
            readout.transpose(1, 2)
            .reshape(batch_size, seq_len, self.config.hidden_size)
            .to(input_dtype)
        )
        # Zero o_proj is a no-op only if readout is finite; 0 * NaN is NaN.
        readout = torch.where(
            torch.isfinite(readout), readout, torch.zeros_like(readout)
        )
        output = self.o_proj(self.post_norm(readout))
        output = torch.where(
            torch.isfinite(output), output, torch.zeros_like(output)
        )

        offset = _seq_offset_to_tensor(
            state.seq_offset, batch_size, device=device
        )
        new_state = TTTState(params=params, seq_offset=offset + seq_len)
        return output, new_state

    def _project(
        self, proj: nn.Linear, hidden_states: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        projected = proj(hidden_states)
        return (
            projected.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .to(dtype)
        )


class TTTLinear(TTTLayer):
    """TTT layer whose fast model is a single linear layer."""

    def _init_fast_weights(self) -> None:
        std = self.config.fast_weight_init_std
        self.w1_init = nn.Parameter(
            torch.normal(
                0.0, std, size=(self.num_heads, self.head_dim, self.head_dim)
            )
        )
        self.b1_init = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def _initial_params(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            _batched_fast_weight(param, batch_size)
            for param in (self.w1_init, self.b1_init)
        )

    def _mini_batch_step(self, params, xq, xk, xv, eta, ln_weight, ln_bias):
        w1, b1 = params
        eta_tril = torch.tril(eta)

        z1 = xk @ w1 + b1
        grad_z1 = _ln_fused_l2_bwd(z1, xv, ln_weight, ln_bias)

        # Readout: every query sees the fast weights updated by the gradients of
        # the strictly preceding keys in this mini-batch (causal prefix).
        attn = torch.tril(xq @ xk.transpose(-2, -1))
        b1_bar = b1 - eta_tril @ grad_z1
        z1_bar = xq @ w1 - (eta * attn) @ grad_z1 + b1_bar

        last_eta = eta[..., -1, :, None]
        w1_next = w1 - (last_eta * xk).transpose(-2, -1) @ grad_z1
        b1_next = b1 - (last_eta * grad_z1).sum(dim=-2, keepdim=True)

        out = xq + _ln_fwd(z1_bar, ln_weight, ln_bias)
        return (w1_next, b1_next), out


class TTTMLP(TTTLayer):
    """TTT layer whose fast model is a two-layer MLP with GeLU."""

    def _init_fast_weights(self) -> None:
        std = self.config.fast_weight_init_std
        inner_dim = self.head_dim * self.config.fast_mlp_ratio
        self.w1_init = nn.Parameter(
            torch.normal(0.0, std, size=(self.num_heads, self.head_dim, inner_dim))
        )
        self.b1_init = nn.Parameter(torch.zeros(self.num_heads, 1, inner_dim))
        self.w2_init = nn.Parameter(
            torch.normal(0.0, std, size=(self.num_heads, inner_dim, self.head_dim))
        )
        self.b2_init = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def _initial_params(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            _batched_fast_weight(param, batch_size)
            for param in (self.w1_init, self.b1_init, self.w2_init, self.b2_init)
        )

    def _mini_batch_step(self, params, xq, xk, xv, eta, ln_weight, ln_bias):
        w1, b1, w2, b2 = params
        eta_tril = torch.tril(eta)

        z1 = xk @ w1 + b1
        x2 = F.gelu(z1, approximate="tanh")
        z2 = x2 @ w2 + b2

        grad_z2 = _ln_fused_l2_bwd(z2, xv, ln_weight, ln_bias)
        grad_z1 = (grad_z2 @ w2.transpose(-2, -1)) * _gelu_bwd(z1)

        attn1 = torch.tril(xq @ xk.transpose(-2, -1))
        b1_bar = b1 - eta_tril @ grad_z1
        z1_bar = xq @ w1 - (eta * attn1) @ grad_z1 + b1_bar
        x2_bar = F.gelu(z1_bar, approximate="tanh")

        attn2 = torch.tril(x2_bar @ x2.transpose(-2, -1))
        b2_bar = b2 - eta_tril @ grad_z2
        z2_bar = x2_bar @ w2 - (eta * attn2) @ grad_z2 + b2_bar

        last_eta = eta[..., -1, :, None]
        w1_next = w1 - (last_eta * xk).transpose(-2, -1) @ grad_z1
        b1_next = b1 - (last_eta * grad_z1).sum(dim=-2, keepdim=True)
        w2_next = w2 - (last_eta * x2).transpose(-2, -1) @ grad_z2
        b2_next = b2 - (last_eta * grad_z2).sum(dim=-2, keepdim=True)

        out = xq + _ln_fwd(z2_bar, ln_weight, ln_bias)
        return (w1_next, b1_next, w2_next, b2_next), out


def build_ttt_layer(config: TTTConfig) -> TTTLayer:
    """Instantiate the TTT layer selected by ``config.fast_model``."""
    if config.fast_model == "linear":
        return TTTLinear(config)
    return TTTMLP(config)


@dataclass
class TTTContext:
    """Mutable carrier for the TTT recurrent state across a model call.

    The DiT's ``forward`` signature is fixed by upstream call sites, so the fast
    weights travel in this object rather than in the return value: the model
    reads ``states`` on entry and overwrites it on exit.

    Attributes:
        num_timesteps: ``T``, how many timesteps the flattened batch holds.
        states: per-layer fast weights, ``None`` before the first call.
        reset_mask: ``[B]`` boolean marking trajectories that must restart from
            the learned initialization ``W_0`` (episode boundaries at rollout).
        token_mask: ``[tokens_per_step]`` boolean selecting which tokens are
            routed through TTT; ``None`` routes all of them.
        detach_states: cut the graph on the way out, which turns a sequence of
            calls into truncated backpropagation through time.
    """

    num_timesteps: int = 1
    states: Optional[list[Optional[TTTState]]] = None
    reset_mask: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    detach_states: bool = False

    def branch(self) -> "TTTContext":
        """A copy that reads the same states but discards its own writes.

        Used for the intermediate iterations of a denoising loop: each
        iteration must read the fast weights entering the timestep, but only
        one of them may advance them, otherwise the state would step once per
        denoising iteration instead of once per timestep.
        """
        return TTTContext(
            num_timesteps=self.num_timesteps,
            states=None if self.states is None else list(self.states),
            reset_mask=self.reset_mask,
            token_mask=self.token_mask,
            detach_states=self.detach_states,
        )

    def detached(self) -> "TTTContext":
        """A copy whose states no longer carry gradient history."""
        states = (
            None
            if self.states is None
            else [None if s is None else s.detach() for s in self.states]
        )
        return TTTContext(
            num_timesteps=self.num_timesteps,
            states=states,
            reset_mask=None,
            token_mask=self.token_mask,
            detach_states=self.detach_states,
        )


class TTTTimeMixer(nn.Module):
    """A stack of gated TTT layers operating across the time axis.

    Hosts one TTT layer (and one tanh gate) per transformer layer of the model
    it augments. The model calls :meth:`apply_layer` after each of its own
    layers; this class owns the folding between the model's per-timestep batch
    layout and the time-major layout the TTT inner loop needs.

    Layout: the model is fed ``[B * T, tokens_per_step, D]`` in **batch-major**
    order, i.e. row ``b * T + t``. Folding to ``[B, T * tokens_per_step, D]``
    therefore yields each trajectory's tokens in temporal order.
    """

    def __init__(self, config: TTTConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = int(num_layers)
        self.layers = nn.ModuleList(
            [build_ttt_layer(config) for _ in range(self.num_layers)]
        )
        # Per-channel tanh gate, initialized near zero so the TTT branch starts
        # as a no-op and a pretrained backbone is reproduced exactly.
        self.gates = nn.ParameterList(
            [
                nn.Parameter(torch.full((config.hidden_size,), config.gate_init))
                for _ in range(self.num_layers)
            ]
        )

    def __len__(self) -> int:
        return self.num_layers

    def apply_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        context: Optional[TTTContext],
        layer: Optional[nn.Module] = None,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fold to time-major, run one TTT layer, add the gated result back.

        Args:
            layer_index: which TTT layer to run.
            hidden_states: ``[B * T, tokens_per_step, D]``, batch-major.
            context: recurrent state carrier; ``None`` means a single timestep
                starting from ``W_0``.

        Returns:
            A tensor shaped like ``hidden_states``.
        """
        num_timesteps = 1 if context is None else max(1, context.num_timesteps)
        flat_batch, tokens_per_step, dim = hidden_states.shape
        if flat_batch % num_timesteps != 0:
            raise ValueError(
                f"Batch {flat_batch} is not divisible by num_timesteps "
                f"{num_timesteps}; the trajectory batch must be flattened as "
                f"[B, T] in batch-major order."
            )
        batch_size = flat_batch // num_timesteps

        ttt_layer = self.layers[layer_index] if layer is None else layer
        ttt_gate = self.gates[layer_index] if gate is None else gate

        per_step = hidden_states.reshape(
            batch_size, num_timesteps, tokens_per_step, dim
        )
        token_mask = None if context is None else context.token_mask
        if token_mask is None:
            selected = per_step
        else:
            token_mask = token_mask.to(hidden_states.device)
            selected = per_step[:, :, token_mask]
        num_selected = selected.shape[2]

        state = None
        if context is not None and context.states is not None:
            state = context.states[layer_index]

        ttt_output, new_state = ttt_layer(
            selected.reshape(batch_size, num_timesteps * num_selected, dim),
            state=state,
            reset_mask=None if context is None else context.reset_mask,
            tokens_per_step=num_selected,
        )

        if context is not None:
            if context.states is None:
                context.states = [None] * self.num_layers
            context.states[layer_index] = (
                new_state.detach() if context.detach_states else new_state
            )

        gate_value = torch.tanh(ttt_gate).to(ttt_output.dtype)
        gate_value = torch.where(
            torch.isfinite(gate_value), gate_value, torch.zeros_like(gate_value)
        )
        ttt_output = torch.where(
            torch.isfinite(ttt_output), ttt_output, torch.zeros_like(ttt_output)
        )
        gated = (gate_value * ttt_output).reshape(
            batch_size, num_timesteps, num_selected, dim
        )

        if token_mask is None:
            return (per_step + gated).reshape(flat_batch, tokens_per_step, dim)

        residual = torch.zeros_like(per_step)
        residual[:, :, token_mask] = gated
        return (per_step + residual).reshape(flat_batch, tokens_per_step, dim)
