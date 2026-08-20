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

import pytest
import torch

from rlinf.models.embodiment.gr00t.gr00t_n1d7.ttt import (
    TTTConfig,
    TTTContext,
    TTTLinear,
    TTTMLP,
    TTTTimeMixer,
    build_ttt_layer,
)

TOKENS_PER_STEP = 4


def _make_layer(fast_model: str = "mlp") -> torch.nn.Module:
    torch.manual_seed(0)
    config = TTTConfig(
        hidden_size=16,
        num_heads=2,
        fast_model=fast_model,
        fast_mlp_ratio=2,
        mini_batch_size=None,
        max_mini_batch_size=8,
    )
    layer = build_ttt_layer(config)
    # o_proj is zero-initialized so a fresh layer is a no-op; give it real
    # weights so the tests actually exercise the read-out path.
    torch.nn.init.normal_(layer.o_proj.weight, std=0.1)
    return layer.double()


def _inputs(batch_size: int, num_steps: int, hidden_size: int = 16) -> torch.Tensor:
    return torch.randn(
        batch_size, num_steps * TOKENS_PER_STEP, hidden_size, dtype=torch.float64
    )


@pytest.mark.parametrize("fast_model", ["mlp", "linear"])
def test_builder_selects_fast_model(fast_model):
    layer = build_ttt_layer(TTTConfig(hidden_size=16, num_heads=2, fast_model=fast_model))
    expected = TTTMLP if fast_model == "mlp" else TTTLinear
    assert isinstance(layer, expected)


@pytest.mark.parametrize("fast_model", ["mlp", "linear"])
def test_zero_initialized_readout_is_a_noop(fast_model):
    """A freshly built layer must not perturb a pretrained backbone."""
    config = TTTConfig(hidden_size=16, num_heads=2, fast_model=fast_model)
    layer = build_ttt_layer(config).double()
    output, _ = layer(_inputs(2, 3), tokens_per_step=TOKENS_PER_STEP)
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("fast_model", ["mlp", "linear"])
def test_state_carry_matches_single_pass(fast_model):
    """Splitting a sequence and carrying the state reproduces one long pass.

    This is the property truncated BPTT relies on: segment boundaries change
    where gradients stop, never the forward values.
    """
    layer = _make_layer(fast_model)
    hidden = _inputs(2, 4)
    split = 2 * TOKENS_PER_STEP

    full_output, full_state = layer(hidden, tokens_per_step=TOKENS_PER_STEP)

    first_output, mid_state = layer(
        hidden[:, :split], tokens_per_step=TOKENS_PER_STEP
    )
    second_output, final_state = layer(
        hidden[:, split:], state=mid_state, tokens_per_step=TOKENS_PER_STEP
    )

    chunked_output = torch.cat([first_output, second_output], dim=1)
    torch.testing.assert_close(full_output, chunked_output)
    for expected, actual in zip(full_state.params, final_state.params):
        torch.testing.assert_close(expected, actual)
    torch.testing.assert_close(
        final_state.seq_offset.to(dtype=torch.long),
        full_state.seq_offset.to(dtype=torch.long),
    )


@pytest.mark.parametrize("fast_model", ["mlp", "linear"])
def test_readout_is_causal_across_timesteps(fast_model):
    """Perturbing a later timestep must not change earlier read-outs."""
    layer = _make_layer(fast_model)
    hidden = _inputs(1, 4)
    perturbed = hidden.clone()
    perturbed[:, 2 * TOKENS_PER_STEP :] += 5.0

    base, _ = layer(hidden, tokens_per_step=TOKENS_PER_STEP)
    changed, _ = layer(perturbed, tokens_per_step=TOKENS_PER_STEP)

    boundary = 2 * TOKENS_PER_STEP
    torch.testing.assert_close(base[:, :boundary], changed[:, :boundary])
    assert not torch.allclose(base[:, boundary:], changed[:, boundary:])


def test_detach_cuts_gradient_but_keeps_values():
    layer = _make_layer("mlp")
    hidden = _inputs(1, 4)
    split = 2 * TOKENS_PER_STEP

    first_half = hidden[:, :split].clone().requires_grad_(True)
    second_half = hidden[:, split:].clone().requires_grad_(True)

    _, mid_state = layer(first_half, tokens_per_step=TOKENS_PER_STEP)
    detached_state = mid_state.detach()
    for original, detached in zip(mid_state.params, detached_state.params):
        torch.testing.assert_close(original, detached)
        assert not detached.requires_grad

    output, _ = layer(
        second_half, state=detached_state, tokens_per_step=TOKENS_PER_STEP
    )
    output.sum().backward()

    assert first_half.grad is None
    assert second_half.grad is not None
    assert torch.isfinite(second_half.grad).all()


def test_gradient_flows_through_carried_state_without_detach():
    layer = _make_layer("mlp")
    hidden = _inputs(1, 4)
    split = 2 * TOKENS_PER_STEP

    first_half = hidden[:, :split].clone().requires_grad_(True)
    second_half = hidden[:, split:].clone().requires_grad_(True)

    _, mid_state = layer(first_half, tokens_per_step=TOKENS_PER_STEP)
    output, _ = layer(second_half, state=mid_state, tokens_per_step=TOKENS_PER_STEP)
    output.sum().backward()

    assert first_half.grad is not None
    assert first_half.grad.abs().sum() > 0


def test_reset_mask_restores_initial_fast_weights():
    layer = _make_layer("mlp")
    hidden = _inputs(2, 3)

    _, state = layer(hidden, tokens_per_step=TOKENS_PER_STEP)
    reset_mask = torch.tensor([True, False])
    reset_state = layer.reset_state(state, reset_mask, batch_size=2)

    initial = layer.init_state(2)
    for restored, fresh, carried in zip(
        reset_state.params, initial.params, state.params
    ):
        torch.testing.assert_close(restored[0], fresh[0].to(restored.dtype))
        torch.testing.assert_close(restored[1], carried[1])

    torch.testing.assert_close(
        reset_state.seq_offset[0], torch.zeros((), dtype=reset_state.seq_offset.dtype)
    )
    torch.testing.assert_close(reset_state.seq_offset[1], state.seq_offset[1])


def test_learned_initial_state_receives_gradient():
    """``W_0`` is meta-learned, so it must be reachable from the outer loss."""
    layer = _make_layer("mlp")
    output, _ = layer(_inputs(1, 2), tokens_per_step=TOKENS_PER_STEP)
    output.sum().backward()

    assert layer.w1_init.grad is not None
    assert layer.w1_init.grad.abs().sum() > 0


def test_mini_batch_size_falls_back_to_tokens_per_step():
    config = TTTConfig(hidden_size=16, num_heads=2, mini_batch_size=5)
    layer = build_ttt_layer(config)
    # 5 does not divide 3 * 4 tokens, so the per-step count is used instead.
    assert layer._resolve_mini_batch_size(3 * TOKENS_PER_STEP, TOKENS_PER_STEP) == 4
    assert layer._resolve_mini_batch_size(3 * 5, TOKENS_PER_STEP) == 5


# --------------------------------------------------------------- TTTTimeMixer

MIXER_TOKENS = 5
MIXER_DIM = 16


def _make_mixer(num_layers: int = 2, *, active: bool = True) -> TTTTimeMixer:
    torch.manual_seed(0)
    config = TTTConfig(
        hidden_size=MIXER_DIM,
        num_heads=2,
        fast_mlp_ratio=2,
        max_mini_batch_size=16,
    )
    mixer = TTTTimeMixer(config, num_layers).double()
    if active:
        # Undo the deliberate zero/near-zero initialization so the tests see a
        # TTT branch that actually contributes.
        for layer in mixer.layers:
            torch.nn.init.normal_(layer.o_proj.weight, std=0.1)
        for gate in mixer.gates:
            torch.nn.init.constant_(gate, 0.5)
    return mixer


def _flat_batch(batch_size: int, num_steps: int) -> torch.Tensor:
    return torch.randn(
        batch_size * num_steps, MIXER_TOKENS, MIXER_DIM, dtype=torch.float64
    )


def test_mixer_is_a_noop_at_initialization():
    mixer = _make_mixer(active=False)
    hidden = _flat_batch(2, 3)
    output = mixer.apply_layer(0, hidden, TTTContext(num_timesteps=3))
    torch.testing.assert_close(output, hidden)


def test_mixer_keeps_trajectories_independent():
    """Batch-major folding must not let one trajectory read another's history."""
    mixer = _make_mixer()
    num_steps = 3
    hidden = _flat_batch(2, num_steps)
    perturbed = hidden.clone()
    perturbed[num_steps:] += 3.0  # rows of trajectory b=1 only

    base = mixer.apply_layer(0, hidden, TTTContext(num_timesteps=num_steps))
    changed = mixer.apply_layer(0, perturbed, TTTContext(num_timesteps=num_steps))

    torch.testing.assert_close(base[:num_steps], changed[:num_steps])
    assert not torch.allclose(base[num_steps:], changed[num_steps:])


def test_mixer_whole_sequence_matches_step_by_step():
    """Training over T steps at once must match stepping through a rollout."""
    mixer = _make_mixer()
    batch_size, num_steps = 2, 4
    hidden = _flat_batch(batch_size, num_steps)

    whole = mixer.apply_layer(0, hidden, TTTContext(num_timesteps=num_steps))

    per_step = hidden.reshape(batch_size, num_steps, MIXER_TOKENS, MIXER_DIM)
    context = TTTContext(num_timesteps=1)
    stepped = []
    for step in range(num_steps):
        stepped.append(mixer.apply_layer(0, per_step[:, step], context))

    stepped = torch.stack(stepped, dim=1).reshape_as(hidden)
    torch.testing.assert_close(whole, stepped)


def test_mixer_token_mask_restricts_the_ttt_branch():
    """Ablations select which tokens TTT reads and writes."""
    mixer = _make_mixer()
    num_steps = 3
    hidden = _flat_batch(2, num_steps)
    token_mask = torch.zeros(MIXER_TOKENS, dtype=torch.bool)
    token_mask[:2] = True

    output = mixer.apply_layer(
        0, hidden, TTTContext(num_timesteps=num_steps, token_mask=token_mask)
    )

    torch.testing.assert_close(output[:, 2:], hidden[:, 2:])
    assert not torch.allclose(output[:, :2], hidden[:, :2])


def test_mixer_writes_state_back_into_the_context():
    mixer = _make_mixer(num_layers=2)
    context = TTTContext(num_timesteps=2)
    hidden = _flat_batch(1, 2)

    mixer.apply_layer(0, hidden, context)
    assert context.states is not None
    assert context.states[0] is not None
    assert context.states[1] is None

    mixer.apply_layer(1, hidden, context)
    assert context.states[1] is not None


def test_mixer_rejects_batch_not_divisible_by_timesteps():
    mixer = _make_mixer()
    hidden = _flat_batch(1, 3)
    with pytest.raises(ValueError, match="not divisible by num_timesteps"):
        mixer.apply_layer(0, hidden, TTTContext(num_timesteps=2))


def test_mixer_detach_states_truncates_gradient():
    mixer = _make_mixer()
    context = TTTContext(num_timesteps=1, detach_states=True)
    first = _flat_batch(1, 1).requires_grad_(True)
    second = _flat_batch(1, 1).requires_grad_(True)

    mixer.apply_layer(0, first, context)
    output = mixer.apply_layer(0, second, context)
    output.sum().backward()

    assert first.grad is None
    assert second.grad is not None


def test_context_branch_does_not_write_back():
    mixer = _make_mixer()
    original = TTTContext(num_timesteps=1)
    mixer.apply_layer(0, _flat_batch(1, 1), original)
    snapshot = original.states[0].params[0].clone()

    branch = original.branch()
    mixer.apply_layer(0, _flat_batch(1, 1) + 1.0, branch)
    torch.testing.assert_close(original.states[0].params[0], snapshot)
    assert not torch.allclose(branch.states[0].params[0], snapshot)


def test_mixer_updates_num_timesteps_when_windows_change_length():
    """TBPTT's last segment can be shorter than the rest."""
    mixer = _make_mixer()
    context = TTTContext(num_timesteps=3)
    mixer.apply_layer(0, _flat_batch(1, 3), context)
    offset_after_first = context.states[0].seq_offset.clone()

    context.num_timesteps = 2
    mixer.apply_layer(0, _flat_batch(1, 2), context)

    torch.testing.assert_close(
        context.states[0].seq_offset,
        offset_after_first + 2 * MIXER_TOKENS,
    )
