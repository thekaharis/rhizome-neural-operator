import pytest
import torch
from torch import nn

from ebm21cm.model.recurrent import SpectralConv2d
from ebm21cm.model.rhizome import LocalConv2d, RhizomeOperator2d, StateDependentIntegral2d, disc_offsets


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


@pytest.mark.parametrize("radius,count", [(1, 5), (2 ** 0.5, 9), (2, 13), (5 ** 0.5, 21), (3, 29), (4, 49)])
def test_disc_offsets_count_lattice_points(radius, count):
    assert len(disc_offsets(radius)) == count


def test_local_kernel_single_offset_is_periodic_shift():
    conv = LocalConv2d(1, 2.5).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, disc_offsets(2.5).tolist().index([1, -2])] = 1
    x = torch.randn(1, 1, 7, 9, dtype=torch.float64)
    # Cross-correlation on the torus: y(r) = x(r + d).
    assert torch.allclose(conv(x), torch.roll(x, (-1, 2), dims=(2, 3)))


@pytest.mark.parametrize("rank", [None, 3])
def test_local_model_influence_stays_inside_steps_times_radius(rank):
    torch.manual_seed(0)
    model = RhizomeOperator2d(4, width=6, modes=2, n_steps=2, radius=1.5, rank=rank).double().eval()
    x = torch.randn(1, 4, 12, 13, dtype=torch.float64)
    bumped = x.clone()
    bumped[0, :, 0, 5] += 1.0
    changed = (model(bumped) - model(x)).abs()[0, 0] > 0
    i, j = torch.meshgrid(torch.arange(12), torch.arange(13), indexing="ij")
    di, dj = torch.minimum(i, 12 - i), torch.minimum((j - 5).abs(), 13 - (j - 5).abs())
    dist = (di.square() + dj.square()).double().sqrt()
    assert changed[dist <= 1.5].all() and not changed[dist > 2 * 1.5 + 1e-9].any()
    assert changed[dist > 1.5].any()  # the second update does extend the reach
    model(x).square().mean().backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())


def test_local_model_commutes_with_periodic_rolls():
    torch.manual_seed(0)
    model = RhizomeOperator2d(4, width=6, modes=2, n_steps=2, radius=2).double().eval()
    x = torch.randn(1, 4, 9, 10, dtype=torch.float64)
    assert torch.allclose(model(torch.roll(x, (3, 5), dims=(2, 3))), torch.roll(model(x), (3, 5), dims=(2, 3)))


def test_local_kernel_matches_dense_integral():
    torch.manual_seed(0)
    conv = LocalConv2d(3, 2).double()
    x = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    expected = torch.zeros_like(x)
    for (dy, dx), w in zip(disc_offsets(2).tolist(), conv.weight.unbind(-1)):
        expected += torch.einsum("oi,bihw->bohw", w, torch.roll(x, (-dy, -dx), dims=(2, 3)))
    assert torch.allclose(conv(x), expected)


def test_native_cyl_matches_fno_cylindrical_sums_and_identity():
    import sys

    import numpy as np

    from ebm21cm.native_cyl import FNO_ROOT, NativeCyl
    sys.path.insert(0, FNO_ROOT)
    from viz.power_spectrum_evaluation import PowerSpectrumAccumulator, SpectrumConfig, ZChunk

    rng = np.random.default_rng(0)
    nc = NativeCyl([10.0], n_cells=16, kpar_bins="log")
    truth, pred = rng.normal(size=(140, 140, 16)), rng.normal(size=(140, 140, 16))
    # fno-21cm's own chunk sums, then the k_par log binning applied to them.
    k = 2 * np.pi * np.fft.rfftfreq(16, d=200 / 140)
    chunk = ZChunk(10.0, 0, 16, 200 / 140, k[1:], np.concatenate([[k[1] / 2], k[1:] + k[1] / 2]), np.hanning(16))
    acc = PowerSpectrumAccumulator(SpectrumConfig(), nc.binner, chunks=[chunk])
    ref = [x.T @ nc.kpar_matrix[1:] for x in acc._cylindrical_sums(truth, pred, chunk)]
    ours = nc._sums(truth, pred)
    assert all(np.allclose(a.T, b) for a, b in zip(ref, ours))
    # A perfect prediction on a whole cone: ratio and coherence are exactly 1.
    z, field = np.linspace(9, 11, 40), rng.normal(size=(140, 140, 40))
    nc.add("xhi", 1, z, field, field)
    pt, pp, cross, ok = nc.records["xhi"][0][0]
    assert ok and np.allclose(pp, pt) and np.allclose(cross, pt)


def test_native_cyl_linear_kpar_rows_are_the_los_frequencies():
    import numpy as np

    from ebm21cm.native_cyl import NativeCyl

    nc = NativeCyl([10.0], n_cells=16)
    k = 2 * np.pi * np.fft.rfftfreq(16, d=200 / 140)
    assert nc.kpar_matrix.shape == (9, 8) and np.allclose(0.5 * (nc.kpar_edges[1:] + nc.kpar_edges[:-1]), k[1:])
    pt, pp, cross = nc._sums(*np.random.default_rng(1).normal(size=(2, 140, 140, 16)))
    assert pt.shape == (8, 15) and np.all(pt > 0)


def test_periodic_upsample_is_exact_for_constants_and_wraps():
    from ebm21cm.model.rhizome import periodic_upsample

    x = torch.randn(1, 2, 5, 7, dtype=torch.float64)
    assert torch.allclose(periodic_upsample(torch.ones(1, 1, 5, 7), 4), torch.ones(1, 1, 20, 28))
    up = periodic_upsample(x, 4)
    assert up.shape == (1, 2, 20, 28)
    # A roll of the coarse grid is a roll by factor cells of the fine grid.
    assert torch.allclose(periodic_upsample(torch.roll(x, (1, 2), (2, 3)), 4), torch.roll(up, (4, 8), (2, 3)))


@pytest.mark.parametrize("fine_radius,coarse_radius", [(None, None), (None, 2), (1.5, None), (1.5, 2)])
def test_unet_shapes_roll_equivariance_and_gradients(fine_radius, coarse_radius):
    from ebm21cm.model.rhizome import RhizomeUNet2d, build_model

    torch.manual_seed(0)
    cfg = dict(in_channels=3, out_channels=2, scalar_dim=4, width=6, modes=3, n_steps=2, radius=fine_radius,
               factor=4, coarse_modes=2, coarse_steps=2, coarse_radius=coarse_radius)
    model = build_model({"kind": "unet", **cfg}).double().eval()
    assert isinstance(model, RhizomeUNet2d)
    cond, scal = torch.randn(2, 3, 24, 20, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    out = model(cond, scal)
    assert out.shape == (2, 2, 24, 20)
    rolled = model(torch.roll(cond, (8, 4), (2, 3)), scal)
    assert torch.allclose(rolled, torch.roll(out, (8, 4), (2, 3)))
    model.train()
    model(cond, scal).square().mean().backward()
    assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 22, 20, dtype=torch.float64), scal[:1])


@pytest.mark.parametrize("mode", ["full", "frozen", "none", "source", "receiver", "static"])
def test_gate_modes(mode):
    torch.manual_seed(0)
    model = RhizomeOperator2d(3, 2, 4, width=6, modes=3, n_steps=3, gate_mode=mode).double()
    cond, scal = torch.randn(2, 3, 12, 10, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    with torch.no_grad():
        _, states = model(cond, scal, return_states=True)
        forcing = model.lift(cond) + model.scalar_embed(scal)[:, :, None, None]
        inter = model.cells[0].interaction
        gates = [inter.factors(s, forcing)[:2] for s in states[:-1]]
    first_src, first_rcv = gates[0]
    for src, rcv in gates:
        assert ((src > 0) & (src < 2)).all() and ((rcv > 0) & (rcv < 2)).all()
        if mode in ("frozen", "none", "static"):  # independent of the evolving state
            assert torch.allclose(src, first_src) and torch.allclose(rcv, first_rcv)
        if mode in ("none", "receiver"):
            assert torch.equal(src, torch.ones_like(src))
        if mode in ("none", "source"):
            assert torch.equal(rcv, torch.ones_like(rcv))
    if mode in ("full", "source", "receiver"):  # state-dependent: gates change over the rollout
        assert not torch.allclose(gates[0][0] * gates[0][1], gates[-1][0] * gates[-1][1])
    model.train()
    model(cond, scal).square().mean().backward()
    assert all(q.grad is not None for q in model.parameters() if q.requires_grad)


def test_unet_rollout_lengths_can_be_overridden():
    from ebm21cm.model.rhizome import RhizomeUNet2d

    torch.manual_seed(0)
    model = RhizomeUNet2d(3, 2, 4, width=6, modes=3, n_steps=2, factor=4, coarse_modes=2, coarse_steps=2).double()
    cond, scal = torch.randn(1, 3, 16, 16, dtype=torch.float64), torch.randn(1, 4, dtype=torch.float64)
    out, states = model(cond, scal, n_steps=4, coarse_steps=1, return_states=True)
    assert len(states) == (1 + 1) + (1 + 4)  # coarse h0 + 1 update, fine h0 + 4 updates
    assert torch.allclose(model(cond, scal), model(cond, scal, n_steps=2, coarse_steps=2))
    assert not torch.allclose(model(cond, scal), out)


def _copy_dense_into_grouped(dense, grouped):
    """Embed a dense SpectralConv2d's weights as group 0 of a GroupedSpectralConv2d (others zero)."""
    k = dense.in_channels
    with torch.no_grad():
        for name in ("dc", "axis", "positive", "negative"):
            target = getattr(grouped, name)
            target.zero_()
            target[0, :k, :k] = getattr(dense, name)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_grouped_spectral_conv_is_block_diagonal_dense(groups):
    from ebm21cm.model.rhizome import GroupedSpectralConv2d

    torch.manual_seed(2)
    channels, modes = 4, 3
    grouped = GroupedSpectralConv2d(channels, modes, groups).double()
    dense = SpectralConv2d(channels, channels, modes).double()
    k = channels // groups
    with torch.no_grad():  # dense weights = block-diagonal embedding of the grouped ones
        for name in ("dc", "axis", "positive", "negative"):
            d, g = getattr(dense, name), getattr(grouped, name)
            d.zero_()
            for i in range(groups):
                d[i * k:(i + 1) * k, i * k:(i + 1) * k] = g[i].to(d.dtype)
    x = torch.randn(2, channels, 8, 9, dtype=torch.float64)
    assert torch.allclose(grouped(x), dense(x), atol=1e-12)


def test_single_dense_pathway_reproduces_state_dependent_integral():
    from ebm21cm.model.rhizome import MultiPathIntegral2d

    torch.manual_seed(3)
    old = StateDependentIntegral2d(3, modes=3).double()
    new = MultiPathIntegral2d(3, modes=3, pathways=1, groups=1).double()
    new.gates.load_state_dict(old.gates.state_dict())
    new.value.load_state_dict(old.value.state_dict())
    _copy_dense_into_grouped(old.kernel, new.kernel)
    state, forcing = torch.randn(2, 3, 8, 8, dtype=torch.float64), torch.randn(2, 3, 8, 8, dtype=torch.float64)
    assert torch.allclose(new(state, forcing), old(state, forcing), atol=1e-12)


def test_pathways_are_independent_concatenated_and_shareable():
    from ebm21cm.model.rhizome import MultiPathIntegral2d

    torch.manual_seed(4)
    width, modes = 4, 3
    state, forcing = torch.randn(2, width, 8, 8, dtype=torch.float64), torch.randn(2, width, 8, 8, dtype=torch.float64)
    free = MultiPathIntegral2d(width, modes, pathways=2, groups=width).double()
    out = free(state, forcing)
    assert out.shape == (2, 2 * width, 8, 8)
    # Each pathway is its own depthwise interaction: zeroing pathway 1's kernel leaves pathway 0 untouched.
    with torch.no_grad():
        for name in ("dc", "axis", "positive", "negative"):
            getattr(free.kernel, name)[width:] = 0
    cut = free(state, forcing)
    assert torch.allclose(cut[:, :width], out[:, :width]) and cut[:, width:].abs().max() < 1e-12
    # Shared kernels equal independent pathways with copied kernels.
    shared = MultiPathIntegral2d(width, modes, pathways=2, groups=width, share="kernels").double()
    tied = MultiPathIntegral2d(width, modes, pathways=2, groups=width).double()
    tied.gates.load_state_dict(shared.gates.state_dict())
    tied.value.load_state_dict(shared.value.state_dict())
    with torch.no_grad():
        for name in ("dc", "axis", "positive", "negative"):
            getattr(tied.kernel, name).copy_(getattr(shared.kernel, name).repeat(2, *[1] * (getattr(shared.kernel, name).ndim - 1)))
    assert torch.allclose(shared(state, forcing), tied(state, forcing), atol=1e-12)
    # Shared gates are repeated over pathways.
    gated = MultiPathIntegral2d(width, modes, pathways=2, groups=width, share="gates").double()
    source, receiver, _ = gated.factors(state, forcing)
    assert torch.equal(source[:, :width], source[:, width:]) and torch.equal(receiver[:, :width], receiver[:, width:])


@pytest.mark.parametrize("pathways,groups,share", [(1, 4, ""), (2, 4, ""), (2, 1, ""), (2, 2, "gates"), (2, 4, "kernels")])
def test_multipath_model_rolls_and_backpropagates(pathways, groups, share):
    torch.manual_seed(5)
    model = RhizomeOperator2d(3, 2, 4, width=4, modes=3, n_steps=2,
                              pathways=pathways, kernel_groups=groups, share=share).double()
    cond, scal = torch.randn(2, 3, 12, 10, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    out = model(cond, scal)
    roll = lambda x: torch.roll(x, shifts=(3, -5), dims=(-2, -1))
    assert torch.allclose(model(roll(cond), scal), roll(out), atol=1e-10)
    out.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_multipath_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="single|spectral"):
        RhizomeOperator2d(3, width=4, modes=3, radius=1, pathways=2)
    with pytest.raises(ValueError, match="divisible"):
        RhizomeOperator2d(3, width=4, modes=3, kernel_groups=3)
    with pytest.raises(ValueError, match="share"):
        RhizomeOperator2d(3, width=4, modes=3, pathways=2, share="values")


def test_kernel_overlap_detects_collapsed_filters():
    from ebm21cm.model.rhizome import GroupedSpectralConv2d, kernel_overlap

    torch.manual_seed(6)
    conv = GroupedSpectralConv2d(6, 4, groups=6)
    _, rank = kernel_overlap(conv)
    assert rank > 4  # random filters are nearly orthogonal
    with torch.no_grad():
        for name in ("dc", "axis", "positive", "negative"):
            p = getattr(conv, name)
            p.copy_(p[:1].expand_as(p) * torch.arange(1, 7).view(-1, *[1] * (p.ndim - 1)))
    gram, rank = kernel_overlap(conv)
    assert torch.allclose(gram, torch.ones_like(gram), atol=1e-5) and rank == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("shape,radius", [((12, 12), 2), ((11, 14), 2.5), ((16, 13), 3)])
def test_disc_fft_conv_matches_direct_periodic_sum_and_local_conv(shape, radius):
    from ebm21cm.model.rhizome import DiscFFTConv2d

    torch.manual_seed(7)
    channels = 3
    disc = DiscFFTConv2d(channels, radius).double()
    x = torch.randn(2, channels, *shape, dtype=torch.float64, requires_grad=True)
    expected = sum(disc.weight[:, k, None, None] * torch.roll(x, (-int(dy), -int(dx)), dims=(2, 3))
                   for k, (dy, dx) in enumerate(disc_offsets(radius)))
    actual = disc(x)
    assert torch.allclose(actual, expected, atol=1e-12)
    probe = torch.randn_like(actual)
    for a, e in zip(torch.autograd.grad((actual * probe).sum(), (x, disc.weight), retain_graph=True),
                    torch.autograd.grad((expected * probe).sum(), (x, disc.weight))):
        assert torch.allclose(a, e, atol=1e-10)
    # The same operator as a depthwise LocalConv2d (rank = channels, identity projections).
    local = LocalConv2d(channels, radius, rank=channels).double()
    with torch.no_grad():
        local.proj_in.copy_(torch.eye(channels, dtype=torch.float64)[:, :, None, None])
        local.proj_out.copy_(torch.eye(channels, dtype=torch.float64)[:, :, None, None])
        local.weight.copy_(disc.weight[:, None])
    assert torch.allclose(local(x), actual, atol=1e-12)


def test_disc_pathway_model_influence_stays_inside_steps_times_radius():
    torch.manual_seed(0)
    model = RhizomeOperator2d(4, width=6, modes=2, n_steps=2, disc_radius=1.5, disc_pathways=1).double().eval()
    x = torch.randn(1, 4, 12, 13, dtype=torch.float64)
    bumped = x.clone()
    bumped[0, :, 0, 5] += 1.0
    changed = (model(bumped) - model(x)).abs()[0, 0] > 1e-12  # FFT round-off elsewhere
    i, j = torch.meshgrid(torch.arange(12), torch.arange(13), indexing="ij")
    di, dj = torch.minimum(i, 12 - i), torch.minimum((j - 5).abs(), 13 - (j - 5).abs())
    dist = (di.square() + dj.square()).double().sqrt()
    assert changed[dist <= 1.5].all() and not changed[dist > 2 * 1.5 + 1e-9].any()
    assert changed[dist > 1.5].any()


def test_hybrid_spectral_and_disc_pathways():
    from ebm21cm.model.rhizome import DiscFFTConv2d, GroupedSpectralConv2d, MultiPathIntegral2d

    torch.manual_seed(8)
    width = 4
    hybrid = MultiPathIntegral2d(width, 3, pathways=2, groups=width, disc_radius=2, disc_pathways=1).double()
    assert isinstance(hybrid.kernel, GroupedSpectralConv2d) and hybrid.kernel.channels == width
    assert isinstance(hybrid.disc, DiscFFTConv2d) and hybrid.disc.channels == width
    state, forcing = torch.randn(2, width, 10, 10, dtype=torch.float64), torch.randn(2, width, 10, 10, dtype=torch.float64)
    out = hybrid(state, forcing)
    with torch.no_grad():
        hybrid.disc.weight.zero_()
    cut = hybrid(state, forcing)  # pathway 0 (spectral) untouched, pathway 1 (disc) silenced
    assert torch.allclose(cut[:, :width], out[:, :width]) and cut[:, width:].abs().max() < 1e-12
    model = RhizomeOperator2d(3, 2, 4, width=4, modes=3, n_steps=2, pathways=2, kernel_groups=4,
                              disc_radius=2, disc_pathways=1).double()
    cond, scal = torch.randn(2, 3, 12, 10, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)
    out = model(cond, scal)
    roll = lambda x: torch.roll(x, shifts=(3, -5), dims=(-2, -1))
    assert torch.allclose(model(roll(cond), scal), roll(out), atol=1e-10)
    out.square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_disc_pathway_validation():
    from ebm21cm.model.rhizome import MultiPathIntegral2d

    with pytest.raises(ValueError, match="disc_pathways"):
        MultiPathIntegral2d(4, 3, pathways=1, disc_radius=2, disc_pathways=2)
    with pytest.raises(ValueError, match="disc_pathways"):
        MultiPathIntegral2d(4, 3, pathways=2, disc_pathways=1)
    with pytest.raises(ValueError, match="same kernel type"):
        MultiPathIntegral2d(4, 3, pathways=2, share="kernels", disc_radius=2, disc_pathways=1)
    with pytest.raises(ValueError, match="too small"):
        RhizomeOperator2d(3, width=4, modes=2, disc_radius=5, disc_pathways=1)(torch.randn(1, 3, 8, 8))
