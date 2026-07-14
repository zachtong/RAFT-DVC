"""Equivalence tests: CorrBlockOnTheFly vs standard CorrBlock.

The on-the-fly variant must produce output that is numerically equivalent
to the standard all-pairs CorrBlock (within fp32 / fp16 tolerance), so it
can be swapped in transparently in the RAFT-DVC forward pass.

NOTE (2026-07-12): the OTF path still implements the LEGACY sampler
convention (W<->D axis swap, see tests/test_corr_sampler.py), so the
standard reference here is constructed with legacy_wd_swap=True.  Once the
OTF/CUDA paths are ported to the fixed convention (corr_sampler_version=2),
drop these flags.

Tests use small spatial sizes (H=8, C=16, B=2) so they run quickly and
fit on any GPU (or CPU).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.core.corr import CorrBlock
from src.core.corr_otf import CorrBlockOnTheFly


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_fmaps(
    batch: int = 2,
    channels: int = 16,
    size: int = 8,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic pair of feature maps for testing."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    fmap1 = torch.randn(batch, channels, size, size, size, generator=g, dtype=torch.float32)
    fmap2 = torch.randn(batch, channels, size, size, size, generator=g, dtype=torch.float32)
    return fmap1.to(device=DEVICE, dtype=dtype), fmap2.to(device=DEVICE, dtype=dtype)


def _make_coords(
    batch: int = 2,
    size: int = 8,
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
    max_flow: float = 1.0,
) -> torch.Tensor:
    """Random sub-voxel coordinates close to the identity grid.

    Returns shape (B, 3, H, W, D) where each voxel's coords are its own
    index plus a small random perturbation (matches what RAFT's GRU emits
    during early training).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    base_y = torch.arange(size, dtype=torch.float32).view(1, 1, size, 1, 1)
    base_x = torch.arange(size, dtype=torch.float32).view(1, 1, 1, size, 1)
    base_z = torch.arange(size, dtype=torch.float32).view(1, 1, 1, 1, size)
    base = torch.cat([
        base_y.expand(batch, 1, size, size, size),
        base_x.expand(batch, 1, size, size, size),
        base_z.expand(batch, 1, size, size, size),
    ], dim=1)
    noise = (torch.rand(batch, 3, size, size, size, generator=g) - 0.5) * 2 * max_flow
    coords = base + noise
    return coords.to(device=DEVICE, dtype=dtype)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_output_shape() -> None:
    """OTF must produce the same output shape as the standard CorrBlock."""
    fmap1, fmap2 = _make_fmaps(batch=2, channels=16, size=8)
    coords = _make_coords(batch=2, size=8)

    standard = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, legacy_wd_swap=True)
    otf = CorrBlockOnTheFly(fmap1, fmap2, num_levels=2, radius=4)

    out_std = standard(coords)
    out_otf = otf(coords)

    assert out_otf.shape == out_std.shape, (
        f"Shape mismatch: OTF={out_otf.shape} vs standard={out_std.shape}"
    )


def test_init_does_not_materialize_full_corr() -> None:
    """OTF must not store the O(H^6) correlation pyramid at init.

    Standard CorrBlock allocates ~B*H^6 floats; OTF should allocate ~B*C*H^3
    (much less). We measure peak GPU memory before/after `__init__`.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA to measure GPU memory")

    B, C, H = 2, 16, 16
    fmap1, fmap2 = _make_fmaps(batch=B, channels=C, size=H)

    # Standard CorrBlock memory footprint
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_std = torch.cuda.memory_allocated()
    _ = CorrBlock(fmap1, fmap2, num_levels=2, radius=4)
    after_std = torch.cuda.max_memory_allocated()
    std_alloc = after_std - before_std

    # OTF memory footprint
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_otf = torch.cuda.memory_allocated()
    _ = CorrBlockOnTheFly(fmap1, fmap2, num_levels=2, radius=4)
    after_otf = torch.cuda.max_memory_allocated()
    otf_alloc = after_otf - before_otf

    # OTF should use noticeably less memory at __init__.
    # For H=16, C=16, B=2: standard ~= B*H^6*4 = 134 MB; OTF ~= B*C*H^3*4*1.3 = 0.5 MB.
    # Allow generous slack: OTF must be < 1/5 of standard.
    assert otf_alloc < std_alloc / 5, (
        f"OTF allocated {otf_alloc/1e6:.1f} MB vs standard {std_alloc/1e6:.1f} MB "
        f"at __init__; expected OTF < standard/5"
    )


@pytest.mark.parametrize("num_levels", [1, 2, 4])
@pytest.mark.parametrize("radius", [3, 4])
def test_lookup_equivalence_fp32(num_levels: int, radius: int) -> None:
    """OTF and standard CorrBlock must produce numerically equivalent output (fp32)."""
    fmap1, fmap2 = _make_fmaps(batch=2, channels=16, size=8)
    coords = _make_coords(batch=2, size=8)

    standard = CorrBlock(fmap1, fmap2, num_levels=num_levels, radius=radius, legacy_wd_swap=True)
    otf = CorrBlockOnTheFly(fmap1, fmap2, num_levels=num_levels, radius=radius)

    out_std = standard(coords)
    out_otf = otf(coords)

    max_diff = (out_otf - out_std).abs().max().item()
    assert max_diff < 1e-5, (
        f"fp32 lookup diverges: max abs diff = {max_diff:.3e} (tolerance 1e-5) "
        f"[num_levels={num_levels}, radius={radius}]"
    )


def test_lookup_equivalence_fp16_autocast() -> None:
    """Under autocast (fp16), the two implementations take slightly different
    numerical paths and diverge by ~fp16 noise floor (typically 2-5e-3).

    Standard CorrBlock: computes the all-pairs corr in fp16 via matmul,
    then samples the fp16 corr volume — every step in fp16.

    OTF: pools fmap2 in fp32 (avg_pool is not in autocast's whitelist),
    samples fmap2 in fp32 via grid_sample (also not autocast'd), and only
    the final einsum runs in fp16. The trilinear interpolation is therefore
    more precise in OTF than in standard — they are NOT bit-identical under
    autocast, but the divergence is well below fp16 weight-noise scale and
    has no measurable training impact.
    """
    if not torch.cuda.is_available():
        pytest.skip("autocast equivalence test requires CUDA")

    fmap1, fmap2 = _make_fmaps(batch=2, channels=16, size=8)
    coords = _make_coords(batch=2, size=8)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        standard = CorrBlock(fmap1, fmap2, num_levels=2, radius=4, legacy_wd_swap=True)
        otf = CorrBlockOnTheFly(fmap1, fmap2, num_levels=2, radius=4)
        out_std = standard(coords).float()
        out_otf = otf(coords).float()

    max_diff = (out_otf - out_std).abs().max().item()
    assert max_diff < 5e-3, (
        f"fp16 lookup diverges beyond fp16 noise floor: "
        f"max abs diff = {max_diff:.3e} (tolerance 5e-3)"
    )


def test_gradient_equivalence() -> None:
    """Gradients w.r.t. fmap1, fmap2 must match between the two implementations."""
    fmap1_std, fmap2_std = _make_fmaps(batch=2, channels=16, size=8)
    fmap1_otf = fmap1_std.detach().clone()
    fmap2_otf = fmap2_std.detach().clone()
    fmap1_std.requires_grad_(True)
    fmap2_std.requires_grad_(True)
    fmap1_otf.requires_grad_(True)
    fmap2_otf.requires_grad_(True)

    coords = _make_coords(batch=2, size=8)

    out_std = CorrBlock(fmap1_std, fmap2_std, num_levels=2, radius=4, legacy_wd_swap=True)(coords)
    out_otf = CorrBlockOnTheFly(
        fmap1_otf, fmap2_otf, num_levels=2, radius=4,
        chunk_size=27, use_checkpoint=False,
    )(coords)

    # Use a deterministic scalar loss so grads are comparable
    loss_std = out_std.sum()
    loss_otf = out_otf.sum()
    loss_std.backward()
    loss_otf.backward()

    grad_diff_fmap1 = (fmap1_std.grad - fmap1_otf.grad).abs().max().item()
    grad_diff_fmap2 = (fmap2_std.grad - fmap2_otf.grad).abs().max().item()

    assert grad_diff_fmap1 < 1e-4, f"fmap1 grad diff = {grad_diff_fmap1:.3e}"
    assert grad_diff_fmap2 < 1e-4, f"fmap2 grad diff = {grad_diff_fmap2:.3e}"


@pytest.mark.parametrize("chunk_size", [1, 9, 27, 81])
def test_chunk_size_invariance(chunk_size: int) -> None:
    """Output must be invariant to chunk_size (it only affects memory/speed)."""
    fmap1, fmap2 = _make_fmaps(batch=2, channels=16, size=8)
    coords = _make_coords(batch=2, size=8)

    out_ref = CorrBlockOnTheFly(
        fmap1, fmap2, num_levels=2, radius=4, chunk_size=27,
    )(coords)
    out_test = CorrBlockOnTheFly(
        fmap1, fmap2, num_levels=2, radius=4, chunk_size=chunk_size,
    )(coords)

    max_diff = (out_ref - out_test).abs().max().item()
    assert max_diff < 1e-6, (
        f"chunk_size={chunk_size} produces different output (max diff {max_diff:.3e})"
    )


def test_pyramid_levels_match() -> None:
    """Multi-level pyramid output must match across implementations."""
    fmap1, fmap2 = _make_fmaps(batch=2, channels=16, size=16)  # size=16 to support L=4
    coords = _make_coords(batch=2, size=16)

    for L in [1, 2, 3, 4]:
        out_std = CorrBlock(fmap1, fmap2, num_levels=L, radius=4, legacy_wd_swap=True)(coords)
        out_otf = CorrBlockOnTheFly(fmap1, fmap2, num_levels=L, radius=4)(coords)
        max_diff = (out_otf - out_std).abs().max().item()
        assert max_diff < 1e-5, f"L={L} diverges: max diff = {max_diff:.3e}"
        # Channel count should be L * (2r+1)^3
        expected_channels = L * (2 * 4 + 1) ** 3
        assert out_otf.shape[1] == expected_channels
