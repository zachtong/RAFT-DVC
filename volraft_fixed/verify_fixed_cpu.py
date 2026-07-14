"""CPU unit checks for the FIXED VolRAFT sampler and model plumbing.

Run:  python volraft_fixed/verify_fixed_cpu.py

Checks:
  1. Impulse test on the patched ``bilinear_sampler`` with a NON-CUBIC
     volume: querying exactly at a spike location returns 1.0 (and the
     upstream buggy sampler, reproduced inline as a negative control,
     does NOT).
  2. Correlation-chain consistency: with fmap2 = fmap1 rolled by +1 voxel
     along one axis, the CorrBlock lookup at coords0 + true_flow puts the
     correlation maximum at the CENTER of the (2r+1)^3 neighborhood for
     interior voxels -- this validates coords_grid (unflipped) +
     CorrBlock + fixed sampler end-to-end and justifies NOT restoring
     the 2-D-RAFT dimension flip.
  3. Tiny forward passes return finite flows of the right shape:
     16^3 (corr_levels=2 -- a 16^3 input only supports 2 pyramid levels
     since the 1/8 feature map is 2^3) and 64^3 with the real training
     config (corr_levels=4, iters=12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from volraft_fixed.model.utils.utils import bilinear_sampler, coords_grid  # noqa: E402
from volraft_fixed.model.volcorr import CorrBlock                          # noqa: E402
from volraft_fixed.model.volraft import ModuleVolRAFT                      # noqa: E402
from volraft_fixed.train_volraft_fixed import VolRAFTFixed                 # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
_failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(name)


def buggy_bilinear_sampler_upstream(img, coords):
    """Verbatim reproduction of the upstream (buggy) sampler -- negative control."""
    _, _, H, W, D = img.shape
    xgrid, ygrid, zgrid = coords.split([1, 1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    zgrid = 2 * zgrid / (D - 1) - 1
    grid = torch.cat([xgrid, ygrid, zgrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)


def test_impulse() -> None:
    torch.manual_seed(0)
    H, W, D = 5, 7, 9          # deliberately non-cubic
    h0, w0, d0 = 2, 5, 7
    vol = torch.zeros(1, 1, H, W, D)
    vol[0, 0, h0, w0, d0] = 1.0

    coords = torch.tensor([[[[[float(h0), float(w0), float(d0)]]]]])  # (1,1,1,1,3)

    out_fixed = bilinear_sampler(vol, coords)
    val_fixed = out_fixed.flatten()[0].item()
    check("impulse: fixed sampler returns 1.0 at spike (non-cubic vol)",
          abs(val_fixed - 1.0) < 1e-6, f"got {val_fixed:.6f}")

    out_buggy = buggy_bilinear_sampler_upstream(vol, coords)
    val_buggy = out_buggy.flatten()[0].item()
    check("impulse: upstream buggy sampler does NOT return 1.0 (negative control)",
          abs(val_buggy - 1.0) > 1e-3, f"got {val_buggy:.6f}")

    # A second spike/query pair, off-axis, to rule out coincidences
    vol2 = torch.zeros(1, 1, H, W, D)
    vol2[0, 0, 4, 1, 3] = 1.0
    coords2 = torch.tensor([[[[[4.0, 1.0, 3.0]]]]])
    val2 = bilinear_sampler(vol2, coords2).flatten()[0].item()
    check("impulse: fixed sampler, second location", abs(val2 - 1.0) < 1e-6,
          f"got {val2:.6f}")


def test_corr_chain() -> None:
    """fmap2 = fmap1 rolled +1 along one axis -> corr max at neighborhood center."""
    torch.manual_seed(1)
    # C=64 so the self-correlation ||f||^2/sqrt(C) ~ sqrt(C) dominates random
    # cross terms ~ N(0, 1) at every voxel (C=16 leaves occasional near-ties).
    B, C, h, w, d = 1, 64, 5, 6, 7   # non-cubic feature map
    fmap1 = torch.randn(B, C, h, w, d)

    r = 1
    n = 2 * r + 1
    center = (n ** 3 - 1) // 2  # flattened center index (i=j=k=r)

    for axis, name in ((2, "h"), (3, "w"), (4, "d")):
        fmap2 = torch.roll(fmap1, shifts=1, dims=axis)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=1, radius=r)

        coords0 = coords_grid(B, h, w, d, device=fmap1.device)
        flow = torch.zeros_like(coords0)
        flow[:, axis - 2] = 1.0          # true displacement: +1 along `axis`
        out = corr_fn(coords0 + flow)    # (B, n^3, h, w, d)

        # interior voxels only (avoid the roll wrap plane and border clamping)
        interior = out[:, :, 1:h - 1, 1:w - 1, 1:d - 1]
        argmax = interior.argmax(dim=1)
        frac = (argmax == center).float().mean().item()
        check(f"corr chain: max at center for +1 roll along {name}",
              frac == 1.0, f"center-hit fraction {frac:.2f}")

        # negative control: query with the flow on a WRONG channel
        wrong = torch.zeros_like(coords0)
        wrong[:, (axis - 2 + 1) % 3] = 1.0
        out_wrong = corr_fn(coords0 + wrong)
        interior_w = out_wrong[:, :, 1:h - 1, 1:w - 1, 1:d - 1]
        frac_w = (interior_w.argmax(dim=1) == center).float().mean().item()
        check(f"corr chain: wrong-channel flow does NOT center the max ({name})",
              frac_w < 0.5, f"center-hit fraction {frac_w:.2f}")


def _tiny_forward(size: int, corr_levels: int, iters: int) -> None:
    torch.manual_seed(2)
    if corr_levels == 4 and size >= 64:
        model = VolRAFTFixed(mixed_precision=False)
        fwd = lambda a, b: model(a, b, iters=iters)  # noqa: E731
    else:
        args = argparse.Namespace(
            corr_levels=corr_levels, corr_radius=3, num_channels=1,
            small=True, dropout=0.0, iters=iters, mixed_precision=False,
            alternate_corr=False, should_normalize=True,
            flow_shape=(1, 3, size, size, size))
        module = ModuleVolRAFT(args)
        module.eval()
        fwd = lambda a, b: module(a, b, iters=iters)  # noqa: E731

    v0 = torch.rand(1, 1, size, size, size)
    v1 = torch.rand(1, 1, size, size, size)
    with torch.no_grad():
        preds = fwd(v0, v1)

    ok_list = isinstance(preds, list) and len(preds) == iters
    check(f"forward {size}^3 (levels={corr_levels}, iters={iters}): "
          f"list of {iters} predictions", ok_list,
          f"type={type(preds).__name__}, len={len(preds) if isinstance(preds, list) else 'n/a'}")
    final = preds[-1]
    ok_shape = tuple(final.shape) == (1, 3, size, size, size)
    check(f"forward {size}^3: final shape (1, 3, {size}, {size}, {size})",
          ok_shape, f"got {tuple(final.shape)}")
    ok_finite = bool(torch.isfinite(final).all())
    check(f"forward {size}^3: all finite", ok_finite,
          f"|flow| max = {final.abs().max().item():.3f}")


def main() -> None:
    print("=== volraft_fixed CPU verification ===\n")
    test_impulse()
    print()
    test_corr_chain()
    print()
    # 16^3: feature map is 2^3 at 1/8 -- only 2 corr pyramid levels fit.
    _tiny_forward(size=16, corr_levels=2, iters=3)
    print()
    # Real training configuration.
    _tiny_forward(size=64, corr_levels=4, iters=12)

    print()
    if _failures:
        print(f"RESULT: {len(_failures)} check(s) FAILED: {_failures}")
        sys.exit(1)
    print("RESULT: all checks passed")


if __name__ == "__main__":
    main()
