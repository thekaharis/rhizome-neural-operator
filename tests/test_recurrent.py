import pytest
import torch

from ebm21cm.model.recurrent import RecurrentFNO2d, SpectralConv2d


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(1)
    return (torch.randn(2, 3, 12, 16, generator=generator),
            torch.randn(2, 2, generator=generator))


def small_model(**kwargs):
    return RecurrentFNO2d(3, scalar_dim=2, width=4, modes=2, n_steps=3, **kwargs)


def test_shapes_and_bounded_rollout(inputs):
    model = small_model()
    cond, scalars = inputs
    output, states = model(cond, scalars, n_steps=20, return_states=True)
    assert output.shape == (2, 1, 12, 16) and len(states) == 21
    assert all(h.shape == (2, 4, 12, 16) and h.abs().max() <= 1.000001 for h in states)
    assert torch.isfinite(output).all()


def test_gradients_reach_both_communication_paths(inputs):
    torch.manual_seed(2)
    model = small_model()
    cond, scalars = inputs
    model(cond, scalars).square().mean().backward()
    cell = model.cells[0]
    for parameter in (cell.local.weight, cell.spectral.positive, cell.spectral.negative,
                      cell.spectral.dc, cell.spectral.axis,
                      model.lift.weight, model.scalar_embed.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_recurrent_parameters_do_not_grow_with_steps():
    a = RecurrentFNO2d(2, width=4, modes=2, n_steps=1)
    b = RecurrentFNO2d(2, width=4, modes=2, n_steps=9)
    assert len(a.cells) == len(b.cells) == 1
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    untied = RecurrentFNO2d(2, width=4, modes=2, n_steps=9, untied=True)
    assert len(untied.cells) == 9
    assert len({id(cell.pointwise.weight) for cell in untied.cells}) == 9
    assert sum(p.numel() for p in untied.parameters()) > sum(p.numel() for p in b.parameters())


def test_shared_cell_is_reapplied_synchronously(inputs):
    model = small_model()
    cond, scalars = inputs
    _, states = model(cond, scalars, return_states=True)
    forcing = model.lift(cond) + model.scalar_embed(scalars)[:, :, None, None]
    assert torch.equal(states[0], forcing.tanh())
    for before, after in zip(states[:-1], states[1:]):
        assert torch.equal(model.cells[0](before, forcing), after)


def test_arbitrary_periodic_roll_equivariance(inputs):
    torch.manual_seed(3)
    model = small_model().double().eval()
    cond, scalars = (a.double() for a in inputs)
    roll = lambda x: torch.roll(x, (3, -5), (-2, -1))
    with torch.no_grad():
        assert torch.allclose(model(roll(cond), scalars), roll(model(cond, scalars)), rtol=1e-8, atol=1e-9)


def test_spectral_dc_mode_communicates_globally():
    spectral = SpectralConv2d(1, 1, modes=2)
    with torch.no_grad():
        for p in spectral.parameters():
            p.zero_()
        spectral.dc.fill_(1)
    impulse = torch.zeros(1, 1, 8, 10)
    impulse[0, 0, 2, 3] = 1
    assert torch.allclose(spectral(impulse), torch.full_like(impulse, 1 / 80))


def test_self_conjugate_axis_preserves_complex_gain():
    spectral = SpectralConv2d(1, 1, modes=3).double()
    with torch.no_grad():
        for p in spectral.parameters():
            p.zero_()
        spectral.axis[0, 0, 0] = 2 + 1j
    phase = 2 * torch.pi * torch.arange(12, dtype=torch.float64) / 12
    x = phase.cos()[None, None, :, None].expand(1, 1, 12, 16)
    expected = (2 * phase.cos() - phase.sin())[None, None, :, None].expand_as(x)
    assert torch.allclose(spectral(x), expected, atol=1e-10, rtol=1e-10)


def test_spectral_parameters_have_no_dead_real_or_imaginary_components():
    torch.manual_seed(9)
    spectral = SpectralConv2d(2, 3, modes=3)
    x = torch.randn(2, 2, 10, 12)
    target = torch.randn(2, 3, 10, 12)
    (spectral(x) * target).sum().backward()
    for p in spectral.parameters():
        assert (p.grad.real.abs() > 0).all()
        if p.is_complex():
            assert (p.grad.imag.abs() > 0).all()


def test_single_mode_is_supported():
    spectral = SpectralConv2d(2, 3, modes=1)
    x = torch.randn(2, 2, 5, 7)
    output = spectral(x)
    assert torch.allclose(output, output[:, :, :1, :1].expand_as(output), atol=1e-7)
    output.square().mean().backward()
    assert spectral.dc.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(8, 8), (11, 15), (20, 12)])
def test_different_grid_resolutions(shape):
    model = RecurrentFNO2d(1, out_channels=2, width=4, modes=3, n_steps=2)
    x = torch.randn(1, 1, *shape)
    assert model(x).shape == (1, 2, *shape)


def test_too_small_grid_rejected():
    model = RecurrentFNO2d(1, width=4, modes=3)
    with pytest.raises(ValueError, match="2\\*modes"):
        model(torch.randn(1, 1, 5, 8))


def test_untied_rollout_limits(inputs):
    model = small_model(untied=True)
    cond, scalars = inputs
    assert model(cond, scalars, n_steps=1).shape == (2, 1, 12, 16)
    with pytest.raises(ValueError, match="untied rollout"):
        model(cond, scalars, n_steps=4)


def test_every_untied_block_is_executed(inputs):
    model = small_model(untied=True)
    cond, scalars = inputs
    model(cond, scalars).square().mean().backward()
    for cell in model.cells:
        assert cell.pointwise.weight.grad is not None
        assert torch.isfinite(cell.pointwise.weight.grad).all()
        assert cell.pointwise.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"modes": 0}, {"n_steps": 0}, {"n_steps": 1.5},
    {"scalar_dim": -1}, {"step_size": 0}, {"step_size": 1.01}, {"step_size": float("nan")},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RecurrentFNO2d(1, **kwargs)


def test_invalid_inputs(inputs):
    model = small_model()
    cond, scalars = inputs
    with pytest.raises(ValueError, match="scalars"):
        model(cond)
    with pytest.raises(ValueError, match="scalars"):
        model(cond, scalars[:, :1])
    with pytest.raises(ValueError, match="conditioning"):
        model(cond[:, :1], scalars)
    with pytest.raises(ValueError, match="n_steps"):
        model(cond, scalars, n_steps=0)
    with pytest.raises(ValueError, match="float32 or float64"):
        model(cond.to(torch.int32), scalars)


@pytest.mark.parametrize("use_local,use_spectral", [(False, True), (True, False), (False, False)])
def test_ablations_have_no_unused_path_parameters(inputs, use_local, use_spectral):
    model = small_model(use_local=use_local, use_spectral=use_spectral)
    cond, scalars = inputs
    assert model(cond, scalars).shape == (2, 1, 12, 16)
    cell = model.cells[0]
    assert (cell.local is not None) == use_local
    assert (cell.spectral is not None) == use_spectral
