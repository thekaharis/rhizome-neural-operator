"""Reproduce the 3-D Rhizome review without modifying model parameters on disk.

Run from the repository root:
    python scripts/rhizome3d_audit.py --out docs/rhizome3d_audit_results.json

Small CPU probes compare the current FFT with an independent all-pairs Fourier
series, including gradients. Findings describe current behavior, not training
accuracy or CUDA validation. No external dataset or FNO checkout is required.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ebm21cm.model.rhizome3d import RhizomeOperator3d, SpectralConv3d


def coefficients(conv, dtype=torch.complex128):
    """Materialize channel matrices; never evaluate a spatial FFT here."""
    if conv.rank is None:
        return conv.weight.to(dtype)
    return torch.einsum("ir,rxyz,ro->ioxyz", conv.proj_in.to(dtype),
                        conv.weight.to(dtype), conv.proj_out.to(dtype))


def dense_series(conv, x):
    """Independent real Fourier-series quadrature on physical source cells.

    At kz=0, taking the real sum implements the Hermitian projection performed
    by irfftn. Positive kz terms have an omitted conjugate and count twice.
    The denominator is the *padded* voxel count used by the current model.
    """
    nx, ny, nz = x.shape[-3:]
    length = nz + conv.los_pad
    axes = (torch.arange(nx, dtype=x.dtype) / nx,
            torch.arange(ny, dtype=x.dtype) / ny,
            torch.arange(nz, dtype=x.dtype) / length)
    pos = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    delta = pos[:, None] - pos[None, :]
    mx, my, mz = conv.modes
    fx = [*range(mx), *range(-(mx - 1), 0)]
    fy = [*range(my), *range(-(my - 1), 0)]
    weight = coefficients(conv)
    kernel = x.new_zeros(conv.channels, conv.channels, len(pos), len(pos))
    for ix, kx in enumerate(fx):
        for iy, ky in enumerate(fy):
            for kz in range(mz):
                phase = torch.exp(2j * torch.pi * (
                    kx * delta[..., 0] + ky * delta[..., 1] + kz * delta[..., 2]))
                term = (weight[:, :, ix, iy, kz].T[:, :, None, None] * phase).real
                kernel = kernel + (1 if kz == 0 else 2) * term
    message = torch.einsum("ocij,bcj->boi", kernel, x.flatten(2)) / (nx * ny * length)
    return message.reshape_as(x)


def reference_checks():
    results = []
    for shape in ((4, 6, 5), (5, 7, 5)):
        for pad in (0, 4, 5):
            for rank in (None, 2):
                torch.manual_seed(17)
                conv = SpectralConv3d(2, (2, 2, 2), los_pad=pad, rank=rank).double()
                x = torch.randn(1, 2, *shape, dtype=torch.float64, requires_grad=True)
                actual, expected = conv(x), dense_series(conv, x)
                probe = torch.randn_like(actual)
                variables = (x, *conv.parameters())
                ga = torch.autograd.grad((actual * probe).sum(), variables, retain_graph=True)
                ge = torch.autograd.grad((expected * probe).sum(), variables)
                output_error = float((actual - expected).abs().max().detach())
                gradient_error = max(float((a - e).abs().max()) for a, e in zip(ga, ge))
                assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-9)
                assert all(torch.allclose(a, e, atol=2e-6, rtol=2e-5) for a, e in zip(ga, ge))
                results.append({"shape": shape, "pad": pad, "rank": rank,
                                "max_output_error": output_error,
                                "max_gradient_error": gradient_error})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fno-root", type=Path, help="optional read-only external API signature inspection")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    result = {"torch": str(torch.__version__), "device": "cpu", "source_sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
        ("ebm21cm/model/rhizome3d.py", "ebm21cm/train_rhizome3d.py", "tests/test_rhizome3d.py")}}
    result["dense_reference"] = reference_checks()

    # Axis-by-axis Nyquist requirements hold: 6>=2*3, 10>=2*4, 8>=2*2.
    conv = SpectralConv3d(1, (3, 4, 2))
    try:
        conv(torch.ones(1, 1, 6, 10, 8))
        result["valid_anisotropic_grid"] = {"accepted": True}
    except ValueError as exc:
        result["valid_anisotropic_grid"] = {"accepted": False, "error": str(exc)}

    dc_results = []
    for nz, pad in ((8, 0), (8, 4), (16, 4), (16, 8)):
        conv = SpectralConv3d(1, (1, 1, 1), los_pad=pad).double()
        with torch.no_grad():
            conv.weight.fill_(1)
        y = conv(torch.ones(1, 1, 4, 4, nz, dtype=torch.float64))
        expected = nz / (nz + pad)
        assert torch.allclose(y, torch.full_like(y, expected), atol=1e-12)
        dc_results.append({"Z": nz, "pad": pad, "actual": float(y.mean().detach()),
                           "padded_measure_prediction": expected})
    result["dc_normalization"] = dc_results

    conv = SpectralConv3d(2, (2, 2, 2), los_pad=2).double()
    # Use complex128 parameters here so cancellation is not obscured by
    # rounding independent perturbations into complex64 storage.
    conv.weight = torch.nn.Parameter(conv.weight.to(torch.complex128))
    x = torch.randn(1, 2, 4, 6, 3, dtype=torch.float64)
    before = conv(x).detach()
    with torch.no_grad():
        conv.weight[0, 0, 0, 0, 0] += 10j
    dc_change = float((conv(x).detach() - before).abs().max())
    before = conv(x).detach()
    with torch.no_grad():
        change = 1 + 2j
        conv.weight[0, 0, 1, 0, 0] += change
        conv.weight[0, 0, 2, 0, 0] -= change.conjugate()
    pair_change = float((conv(x).detach() - before).abs().max())
    result["hermitian_null_directions"] = {
        "parameter_dtype": str(conv.weight.dtype),
        "imaginary_dc_output_change": dc_change,
        "anti_hermitian_pair_output_change": pair_change}

    low = SpectralConv3d(4, (2, 2, 2), rank=1).double()
    dc = coefficients(low)[:, :, 0, 0, 0]
    result["factorized_effective_dc_rank"] = {
        "configured_rank": 1, "complex_matrix_rank": int(torch.linalg.matrix_rank(dc)),
        "real_dc_matrix_rank": int(torch.linalg.matrix_rank(dc.real))}

    # Default circle: signed distances +160 and -160 share residue 160.
    # Both can reach supervised core receivers, even with halo=32.
    pad = 64
    nx, ny, nz = 4, 4, 256
    conv = SpectralConv3d(1, (1, 1, 2), los_pad=pad).double()
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[..., 1] = 1 + 1j
    first, last = torch.zeros(1, 1, nx, ny, nz, dtype=torch.float64), torch.zeros(1, 1, nx, ny, nz, dtype=torch.float64)
    first[..., 0], last[..., 255] = 1, 1
    positive, negative = conv(first)[..., 160], conv(last)[..., 95]
    result["default_los_collision"] = {
        "Z": nz, "pad": pad, "halo": 32, "padded_length": nz + pad,
        "receiver_source_pairs": [[160, 0], [95, 255]], "signed_offsets": [160, -160],
        "modular_offsets": [160 % (nz + pad), (-160) % (nz + pad)],
        "positive_offset_response": float(positive.mean().detach()),
        "negative_offset_response": float(negative.mean().detach()),
        "max_response_difference": float((positive - negative).abs().max().detach()),
        "full_domain_minimum_pad": nz - 1,
        "single_update_core_receiver_minimum_pad": nz - 2 * 32 - 1}

    # Existing translation test passes even with insufficient padding.
    shift = []
    for pad in (0, 4, 9):
        conv = SpectralConv3d(2, (3, 3, 4), los_pad=pad).double()
        x = torch.zeros(1, 2, 8, 8, 10, dtype=torch.float64)
        x[..., -1] = torch.randn(1, 2, 8, 8, dtype=torch.float64)
        shifted = torch.zeros_like(x)
        shifted[..., -2] = x[..., -1]
        shift.append({"pad": pad, "existing_shift_assertion_passes": bool(torch.allclose(
            conv(shifted)[..., :-1], conv(x)[..., 1:], atol=1e-10))})
    result["existing_boundary_test_sensitivity"] = shift

    model = RhizomeOperator3d(2, width=3, modes=(2, 2, 2), n_steps=2,
                             los_pad=2, amp=False, checkpoint=False).double()
    out = model(torch.randn(1, 2, 4, 6, 3, dtype=torch.float64), logits=True)
    result["double_model_output_dtype"] = str(out.dtype)

    count = lambda mx, my, mz, c, r: (
        2 * c * c * (2 * mx - 1) * (2 * my - 1) * mz if r is None else
        2 * (r * (2 * mx - 1) * (2 * my - 1) * mz + 2 * c * r))
    result["default_kernel_real_parameters"] = {"dense": count(24, 24, 16, 48, None),
                                               "rank48": count(24, 24, 16, 48, 48)}
    if args.fno_root:
        root = args.fno_root.resolve()
        window_path = root / "dataset/los_windows.py"
        runner_path = root / "fno_multifield.py"
        window_ast = ast.parse(window_path.read_text())
        cls = next(n for n in window_ast.body if isinstance(n, ast.ClassDef) and n.name == "LOSWindowDataset")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        params = [a.arg for a in (*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs)]
        runner_ast = ast.parse(runner_path.read_text())
        bindings = set()
        for n in runner_ast.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                bindings.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                bindings.update(a.asname or a.name.split(".")[0] for a in n.names)
        result["external_api_static_inspection"] = {
            "root": str(root), "constructor_parameters": params,
            "constructor_accepts_augment": "augment" in params or init.args.kwarg is not None,
            "validation_subset_bound_by_definition_or_import": "validation_subset" in bindings,
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (window_path, runner_path)},
            "scope": "AST signatures only; remote checkout and end-to-end execution not checked"}
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
