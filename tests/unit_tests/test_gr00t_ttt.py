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
    TTTMLP,
    TTTConfig,
    TTTContext,
    TTTLinear,
    TTTTimeMixer,
    _batched_fast_weight,
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


def test_apply_layer_accepts_external_layer_and_gate():
    mixer = _make_mixer()
    hidden = _flat_batch(2, 3)
    via_index = mixer.apply_layer(0, hidden.clone(), TTTContext(num_timesteps=3))
    via_args = mixer.apply_layer(
        0,
        hidden.clone(),
        TTTContext(num_timesteps=3),
        layer=mixer.layers[0],
        gate=mixer.gates[0],
    )
    torch.testing.assert_close(via_index, via_args)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 inner loop needs CUDA")
def test_ttt_mlp_stays_finite_in_bf16_at_dit_width():
    """GR00T N1.7 DiT is 1536-wide with 16 registers + state + action tokens."""
    torch.manual_seed(0)
    config = TTTConfig(
        hidden_size=1536,
        num_heads=32,
        fast_model="mlp",
        max_mini_batch_size=64,
    )
    layer = build_ttt_layer(config).cuda().to(dtype=torch.bfloat16)
    tokens_per_step = 57
    hidden = torch.randn(
        2, 16 * tokens_per_step, 1536, device="cuda", dtype=torch.bfloat16
    )
    output, state = layer(hidden, tokens_per_step=tokens_per_step)
    assert torch.isfinite(output.float()).all()
    for param in state.params:
        assert torch.isfinite(param.float()).all()


def test_ttt_layer_snapshots_meta_parameters_on_cpu():
    config = TTTConfig(hidden_size=16, num_heads=2, fast_model="mlp")
    layer = build_ttt_layer(config)
    meta = torch.empty(16, 16, device="meta")
    q_snap = layer._snapshot_tensor("q_proj.weight", meta)
    o_snap = layer._snapshot_tensor("o_proj.weight", meta)
    assert q_snap.device.type == "cpu"
    assert torch.isfinite(q_snap).all()
    assert float(q_snap.abs().amax()) < 1.0
    torch.testing.assert_close(o_snap, torch.zeros_like(o_snap))


def test_ttt_layer_constructs_under_meta_device():
    config = TTTConfig(hidden_size=16, num_heads=2, fast_model="mlp")
    with torch.device("meta"):
        layer = build_ttt_layer(config)
    assert layer._init_snapshot
    for name, snap in layer._init_snapshot.items():
        assert snap.device.type == "cpu", name
        assert torch.isfinite(snap).all(), name


def test_ttt_layer_restores_corrupted_q_proj():
    torch.manual_seed(0)
    config = TTTConfig(hidden_size=16, num_heads=2, fast_model="mlp")
    layer = build_ttt_layer(config)
    with torch.no_grad():
        layer.q_proj.weight.fill_(1.0e20)
        layer.w1_init.fill_(1.0e20)
        layer.o_proj.weight.fill_(1.0e20)
    hidden = torch.randn(2, 8, 16)
    output, _ = layer(hidden, tokens_per_step=4)
    assert torch.isfinite(output).all()
    assert float(layer.q_proj.weight.detach().abs().amax()) < 1.0
    assert float(layer.w1_init.detach().abs().amax()) < 1.0
    assert float(layer.o_proj.weight.detach().abs().amax()) < 1e-5


def test_batched_fast_weight_keeps_grad_to_w0():
    param = torch.randn(2, 3, 4, requires_grad=True)
    out = _batched_fast_weight(param, 5)
    assert out.shape == (5, 2, 3, 4)
    out.sum().backward()
    assert param.grad is not None


def test_ttt_layer_zeros_nan_readout_when_restore_is_skipped():
    """Inner-loop NaNs must not leak through zero o_proj (0 * NaN is NaN)."""
    torch.manual_seed(0)
    config = TTTConfig(hidden_size=16, num_heads=2, fast_model="mlp")
    layer = build_ttt_layer(config)
    layer._sane_checked = True
    with torch.no_grad():
        layer.w1_init.fill_(float("nan"))
    hidden = torch.randn(2, 8, 16)
    output, _ = layer(hidden, tokens_per_step=4)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_ttt_layer_counts_fast_weight_reset():
    """Non-finite inner-loop state is reset to W_0 and counted for logging."""
    layer = _make_layer("mlp")
    original = layer._mini_batch_step

    def exploding(*args, **kwargs):
        params, out = original(*args, **kwargs)
        nan_params = tuple(torch.full_like(p, float("nan")) for p in params)
        return nan_params, out

    layer._mini_batch_step = exploding
    output, state = layer(_inputs(1, 2), tokens_per_step=TOKENS_PER_STEP)
    assert torch.isfinite(output).all()
    for param in state.params:
        assert torch.isfinite(param).all()
    assert layer.fast_weight_resets == 1


# --------------------------------------------------------------- RoboTTT SFT


def _make_sample_time_head(noise_s: float = 0.999):
    pytest.importorskip("gr00t")
    from torch.distributions import Beta

    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        FlowMatchingActionHeadForRLActionPrediction,
    )

    head = FlowMatchingActionHeadForRLActionPrediction.__new__(
        FlowMatchingActionHeadForRLActionPrediction
    )
    head.beta_dist = Beta(
        torch.tensor(1.5, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )
    head.config = type("Cfg", (), {"noise_s": noise_s})()
    return head


def test_sample_time_matches_groot_n1d7_schedule():
    """RoboTTT Eq. (5) / GR00T N1.7: tau = s * (1 - Beta(1.5, 1))."""
    head = _make_sample_time_head(noise_s=0.999)
    torch.manual_seed(0)
    expected = (1.0 - head.beta_dist.sample((16,))) * 0.999
    torch.manual_seed(0)
    tau = head.sample_time(16, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(tau, expected)
    assert float(tau.max()) <= 0.999 + 1e-6
    assert float(tau.min()) >= 0.0


def test_register_tokens_keep_loaded_checkpoint_values():
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        RoboTTTRegisterTokens,
    )

    torch.manual_seed(1)
    bank = RoboTTTRegisterTokens(4, 8)
    trained = torch.randn(4, 8)
    with torch.no_grad():
        bank.tokens.copy_(trained)

    out = bank(torch.zeros(2, 1, 8))
    torch.testing.assert_close(bank.tokens.detach(), trained)
    torch.testing.assert_close(out[0], trained.to(out.dtype))
    torch.testing.assert_close(out[1], trained.to(out.dtype))


def test_register_tokens_restore_uninitialized_storage():
    pytest.importorskip("gr00t")
    from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
        RoboTTTRegisterTokens,
    )

    bank = RoboTTTRegisterTokens(4, 8)
    with torch.no_grad():
        bank.tokens.fill_(1.0e20)
    bank(torch.zeros(2, 1, 8))
    assert torch.isfinite(bank.tokens).all()
    assert float(bank.tokens.detach().abs().amax()) < 1.0
