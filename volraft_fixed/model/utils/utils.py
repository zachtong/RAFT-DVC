"""VolRAFT sampler utilities -- FIXED version (volraft_fixed).

Copied from VolRAFT ``models/VolRAFT/utils/utils.py`` (wong2024, CVPR-W) with
two deliberate changes relative to the released upstream code.  Everything
else in the ``volraft_fixed.model`` package is a verbatim copy of upstream.

CHANGE 1 -- ``bilinear_sampler``: fixed H<->D axis mishandling.
    Upstream fed a (B, C, H, W, D) volume straight into ``F.grid_sample``
    together with a grid whose channels were normalized as (W, H, D).
    ``F.grid_sample`` for 5-D inputs expects the input laid out as
    (B, C, D_in, H_in, W_in) and grid channels ordered (x, y, z) indexing
    (W_in, H_in, D_in).  Passing (B, C, H, W, D) unpermuted makes grid
    channel 0 (normalized by W-1) index the D axis, channel 1 (normalized
    by H-1) index the W axis, etc. -- a silent axis scramble that only
    cancels out for cubic volumes with symmetric content.  The fix mirrors
    our reference implementation
    ``src/core/corr.py::bilinear_sampler_3d`` (legacy_wd_swap=False):
    permute the volume to (B, C, D, H, W), reorder grid channels from the
    (h, w, d) query order to (w, h, d), permute grid dims to (D', H', W'),
    and permute the output back to (B, C, H', W', D').

CHANGE 2 -- ``coords_grid``: the commented-out 2-D-RAFT dimension flip is
    NOT restored; instead the no-flip choice is now load-bearing and
    documented.  2-D RAFT flips (row, col) -> (x, y) because its sampler
    expects (x, y)-ordered channels.  The FIXED 3-D sampler above expects
    (h, w, d)-ordered channels -- exactly what the unflipped
    ``torch.meshgrid(ht, wd, dp, indexing='ij')`` stack produces -- so the
    pipeline is self-consistent WITHOUT a flip.  Restoring the flip would
    (a) feed (d, w, h)-ordered coords into a sampler expecting (h, w, d)
    and (b) reverse the flow-channel <-> tensor-axis correspondence that
    the ground-truth flow (channel c = displacement along spatial axis c)
    and ``upflow8`` (which rescales channel c by the axis-c zoom factor)
    both rely on.  This matches our reference ``coords_grid_3d`` in
    ``src/core/corr.py``, which also stacks (h, w, d) without a flip.

Unused 2-D leftovers from upstream (``InputPadder``,
``forward_interpolate``) are dropped; ``upflow8`` is kept byte-identical.

Consistency of the whole chain (coords_grid -> CorrBlock -> bilinear_sampler)
is verified by ``volraft_fixed/verify_fixed_cpu.py``.
"""

import torch
import torch.nn.functional as F
import numpy as np


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """Sample a 3-D volume at pixel coordinates (FIXED axis handling).

    Args:
        img:    (B, C, H, W, D) volume.
        coords: (B, H', W', D', 3) query coordinates in pixels, channels
                ordered (h, w, d) -- i.e. channel c indexes spatial axis c
                of ``img`` (matching ``coords_grid`` below).
        mask:   if True, additionally return an in-bounds mask.

    Returns:
        (B, C, H', W', D') sampled values; out-of-range queries -> 0.
    """
    B, C, H, W, D = img.shape

    hgrid, wgrid, dgrid = coords.split([1, 1, 1], dim=-1)
    hgrid = 2 * hgrid / (H - 1) - 1
    wgrid = 2 * wgrid / (W - 1) - 1
    dgrid = 2 * dgrid / (D - 1) - 1

    # grid_sample 5-D convention: input (B, C, D_in, H_in, W_in), grid
    # channels (x, y, z) index (W_in, H_in, D_in).  With img permuted to
    # (B, C, D, H, W) the correct channel order is (w, h, d).
    grid = torch.cat([wgrid, hgrid, dgrid], dim=-1)     # (B, H', W', D', 3)
    grid = grid.permute(0, 3, 1, 2, 4)                  # (B, D', H', W', 3)

    img_p = img.permute(0, 1, 4, 2, 3)                  # (B, C, D, H, W)

    out = F.grid_sample(img_p, grid, mode=mode, align_corners=True)

    # (B, C, D', H', W') -> (B, C, H', W', D').  ``.contiguous()`` because
    # the verbatim upstream volcorr.py calls ``.view`` on the result, which
    # requires a contiguous tensor (the upstream sampler returned
    # grid_sample output directly, which was already contiguous).
    out = out.permute(0, 1, 3, 4, 2).contiguous()

    if mask:
        valid = (hgrid > -1) & (wgrid > -1) & (dgrid > -1) & \
                (hgrid < 1) & (wgrid < 1) & (dgrid < 1)
        return out, valid.float()

    return out


def coords_grid(batch, ht, wd, dp, device):
    """Pixel coordinate grid, channels ordered (h, w, d).

    NOTE: intentionally NOT flipped (see CHANGE 2 in the module docstring).
    Channel c of the resulting (B, 3, H, W, D) grid -- and therefore channel
    c of the flow ``coords1 - coords0`` -- is the coordinate/displacement
    along spatial axis c, matching the GT flow convention and upflow8.
    """
    coords = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        torch.arange(dp, device=device),
        indexing='ij')

    coords = torch.stack(coords, dim=0).float()

    return coords[None].repeat(batch, 1, 1, 1, 1)


def upflow8(flow, mode='trilinear', desired_flow_shape=None):
    if desired_flow_shape is None:
        new_size = (8 * flow.shape[2],
                    8 * flow.shape[3],
                    8 * flow.shape[4])
    else:
        new_size = desired_flow_shape[-3:]

    _, _, ht_old, wd_old, dp_old = flow.shape
    _, _, ht_new, wd_new, dp_new = desired_flow_shape

    resized_flow = 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
    resized_flow[:, 0, :, :, :] *= float(ht_new) / float(ht_old)
    resized_flow[:, 1, :, :, :] *= float(wd_new) / float(wd_old)
    resized_flow[:, 2, :, :, :] *= float(dp_new) / float(dp_old)

    return resized_flow
