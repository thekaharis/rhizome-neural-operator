import pytest
import torch
from torch import nn

from ebm21cm.model.recurrent import SpectralConv2d
from ebm21cm.model.rhizome import RhizomeOperator2d, StateDependentIntegral2d


def _dense_quadrature(interaction, state, forcing):
    """Independent all-pairs Fourier-series reference, not FFT convolution."""
    _, width, h, w = state.shape
    y, x = torch.meshgrid(torch.arange(h, dtype=state.dtype) / h,
                          torch.arange(w, dtype=state.dtype) / w, indexing="ij")
    positions = torch.stack([y.flatten(), x.flatten()], -1)
    delta = positions[:, None] - positions[None, :]
    spectral = interaction.kernel
    # (out_channel, in_channel, receiving_site, sending_site)
    kernel = spectral.dc.T[:, :, None, None].expand(width, width, h * w, h * w).clone()

    def pair(weight, ky, kx):
        phase = torch.exp(2j * torch.pi * (ky * delta[..., 0] + kx * delta[..., 1]))
        # The omitted conjugate frequency contributes the conjugate term.
        return 2 * (weight.T[:, :, None, None] * phase).real

    for ky in range(1, spectral.modes):
        kernel = kernel + pair(spectral.axis[:, :, ky - 1], ky, 0)
    for kx in range(1, spectral.modes):
        for ky in range(spectral.modes):
            kernel = kernel + pair(spectral.positive[:, :, ky, kx - 1], ky, kx)
        for index, ky in enumerate(range(-(spectral.modes - 1), 0)):
            kernel = kernel + pair(spectral.negative[:, :, index, kx - 1], ky, kx)
    source, receiver, value = interaction.factors(state, forcing)
    message = torch.einsum("ocij,bcj->boi", kernel, (source * value).flatten(2)) / (h * w)
    return receiver * message.reshape_as(state)


@pytest.mark.parametrize("shape", [(6, 8), (7, 9)])
def test_fft_interaction_matches_dense_integral_and_gradients(shape):
    torch.manual_seed(11)
    interaction = StateDependentIntegral2d(2, modes=3).double()
    state = torch.randn(2, 2, *shape, dtype=torch.float64, requires_grad=True)
    forcing = torch.randn_like(state, requires_grad=True)
    actual = interaction(state, forcing)
    expected = _dense_quadrature(interaction, state, forcing)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
    probe = torch.randn_like(actual)
    variables = (state, forcing, *interaction.parameters())
    actual_grad = torch.autograd.grad((actual * probe).sum(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad((expected * probe).sum(), variables)
    for a, e in zip(actual_grad, expected_grad):
        assert torch.allclose(a, e, atol=1e-7, rtol=1e-6)


def test_source_and_receiver_factors_change_with_state_and_conditioning():
    torch.manual_seed(1)
    interaction = StateDependentIntegral2d(2, modes=2)
    state = torch.randn(1, 2, 8, 8)
    forcing = torch.randn_like(state)
    original = interaction.factors(state, forcing)
    for changed in (interaction.factors(state + 1, forcing),
                    interaction.factors(state, forcing + 1)):
        assert not torch.allclose(original[0], changed[0])
        assert not torch.allclose(original[1], changed[1])
    assert all(torch.all((g > 0) & (g < 2)) for g in original[:2])


def test_interaction_is_nonlocal_and_source_receiver_gates_control_it():
    interaction = StateDependentIntegral2d(1, modes=1)
    with torch.no_grad():
        for p in interaction.parameters():
            p.zero_()
        interaction.kernel.dc.fill_(1)
        interaction.value.weight[0, 0, 0, 0] = 1
    state = torch.zeros(1, 1, 8, 10)
    state[0, 0, 0, 0] = 1
    forcing = torch.zeros_like(state)
    assert torch.allclose(interaction(state, forcing), torch.full_like(state, 1 / 80))
    with torch.no_grad():
        interaction.gates.bias[0] = -100  # close transmission at the sources
    assert interaction(state, forcing).abs().max() < 1e-20
    with torch.no_grad():
        interaction.gates.bias[0] = 0
        interaction.gates.bias[1] = -100  # independently close reception
    assert interaction(state, forcing).abs().max() < 1e-20


def test_model_has_only_one_spatial_interaction_path():
    model = RhizomeOperator2d(2, width=4, modes=2, n_steps=3)
    assert sum(isinstance(m, SpectralConv2d) for m in model.modules()) == 1
    assert all(m.kernel_size == (1, 1) for m in model.modules() if isinstance(m, nn.Conv2d))
    # If the integral kernel is zero, no information can travel between sites.
    with torch.no_grad():
        for p in model.cells[0].interaction.kernel.parameters():
            p.zero_()
        x = torch.zeros(1, 2, 8, 10)
        original = model(x)
        x[:, :, 0, 0] = 1
        changed = model(x)
    assert torch.equal(original[:, :, 3:, 3:], changed[:, :, 3:, 3:])


def test_rhizome_rollout_shares_weights_and_backpropagates():
    torch.manual_seed(7)
    model = RhizomeOperator2d(2, scalar_dim=3, width=4, modes=2, n_steps=3)
    cond, scalars = torch.randn(2, 2, 8, 10), torch.randn(2, 3)
    out, states = model(cond, scalars, return_states=True)
    assert out.shape == (2, 1, 8, 10)
    assert len(model.cells) == 1 and len(states) == 4
    forcing = model.lift(cond) + model.scalar_embed(scalars)[:, :, None, None]
    for before, after in zip(states[:-1], states[1:]):
        assert torch.equal(model.cells[0](before, forcing), after)
    out.square().mean().backward()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    other = RhizomeOperator2d(2, scalar_dim=3, width=4, modes=2, n_steps=20)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in other.parameters())
    with torch.no_grad():
        _, longer = model(cond, scalars, n_steps=20, return_states=True)
    assert all(h.abs().max() <= 1.000001 for h in longer)


def test_rhizome_arbitrary_periodic_roll_equivariance():
    model = RhizomeOperator2d(2, scalar_dim=1, width=4, modes=2, n_steps=3).double()
    cond, scalars = torch.randn(2, 2, 9, 12).double(), torch.randn(2, 1).double()
    roll = lambda x: torch.roll(x, shifts=(3, -5), dims=(-2, -1))
    with torch.no_grad():
        assert torch.allclose(model(roll(cond), scalars), roll(model(cond, scalars)), atol=1e-10, rtol=1e-9)


def test_constant_field_is_resolution_consistent():
    # Same physical field/domain at two quadrature resolutions: no extra 1/N
    # should shrink the interaction when there are more cells.
    model = RhizomeOperator2d(2, scalar_dim=1, width=4, modes=2, n_steps=3).double()
    value = torch.tensor([0.2, -0.7], dtype=torch.float64)[None, :, None, None]
    scalars = torch.ones(1, 1, dtype=torch.float64)
    with torch.no_grad():
        coarse = model(value.expand(1, 2, 8, 12), scalars)
        fine = model(value.expand(1, 2, 16, 24), scalars)
    assert torch.allclose(coarse, fine[:, :, ::2, ::2], atol=1e-10, rtol=1e-9)


def test_untied_integral_updates_are_all_executed():
    model = RhizomeOperator2d(1, width=4, modes=2, n_steps=3, untied=True)
    x = torch.randn(2, 1, 8, 8)
    model(x).square().mean().backward()
    assert len(model.cells) == 3
    for cell in model.cells:
        assert cell.interaction.gates.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="untied rollout"):
        model(x, n_steps=4)


def test_integral_input_validation():
    interaction = StateDependentIntegral2d(2, modes=2)
    with pytest.raises(ValueError, match="matching"):
        interaction(torch.zeros(1, 2, 8, 8), torch.zeros(1, 1, 8, 8))
    model = RhizomeOperator2d(1, scalar_dim=2, width=4, modes=2)
    with pytest.raises(ValueError, match="scalars"):
        model(torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError, match="2\\*modes"):
        model(torch.zeros(1, 1, 3, 8), torch.zeros(1, 2))
