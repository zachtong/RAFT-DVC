"""On-the-fly 3D correlation block for memory-efficient RAFT-DVC.

Drop-in replacement for :class:`CorrBlock` that computes correlation values
during each lookup call rather than precomputing the full 6D all-pairs
tensor. Reduces stored memory from O(B * H^6) to O(B * C * H^3), enabling
architecture comparison at feature map sizes 32^3+ on consumer GPUs.

See ``docs/plans/2026-05-14-on-the-fly-corr.md`` for the design rationale.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

from .corr import bilinear_sampler_3d


class CorrBlockOnTheFly:
    """Memory-efficient 3D correlation block (on-the-fly lookup).

    Functionally equivalent to :class:`CorrBlock` but does not precompute
    the full all-pairs correlation volume.  Correlation values are computed
    inside each :meth:`__call__` by sampling fmap2 in a local neighborhood
    around the current flow estimate and dot-producting with fmap1.

    Args:
        fmap1: Source feature map, shape (B, C, H, W, D).
        fmap2: Target feature map, same shape as fmap1.
        num_levels: Number of pyramid levels.  Default 4 matches CorrBlock.
        radius: Lookup radius in feature voxels.  Default 4.
        chunk_size: Number of neighbors to process per chunk.  Trades peak
            forward memory for kernel-launch overhead.  Default 27.  Set
            smaller for larger feature maps.
        use_checkpoint: If True (default), wraps the chunk computation in
            ``torch.utils.checkpoint`` so the large sampled-features tensor
            is freed after forward and recomputed during backward.  Adds
            ~30-50% wall-clock overhead but is essential to keep peak
            memory bounded during training.  Disable for unit tests or
            inference where memory is not a concern.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        chunk_size: int = 27,
        use_checkpoint: bool = True,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must have matching shapes; got "
                f"{tuple(fmap1.shape)} vs {tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(
                f"Expected 5D feature maps (B, C, H, W, D); got {fmap1.ndim}D"
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")

        self.num_levels = num_levels
        self.radius = radius
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

        # Source feature map kept at original resolution
        self.fmap1 = fmap1

        # Target feature pyramid via average pooling (matches CorrBlock's
        # implicit pyramid built on the post-correlation tensor).  By
        # linearity of the dot product, pooling fmap2 before the dot is
        # equivalent to pooling the correlation tensor itself.
        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

        # Per-channel normalization constant (same as standard CorrBlock)
        self._norm_const = math.sqrt(fmap1.shape[1])

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Look up correlation values at the given flow coordinates.

        Args:
            coords: Current flow estimate, shape (B, 3, H, W, D).  Channel
                axis holds (y, x, z) components in voxel units.

        Returns:
            Correlation features, shape (B, num_levels * (2r+1)^3, H, W, D).
            Output dtype is float32 to match :class:`CorrBlock`.
        """
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(
                f"coords must be (B, 3, H, W, D); got {tuple(coords.shape)}"
            )

        B, _, H, W, D = coords.shape
        r = self.radius
        n_neighbors = (2 * r + 1) ** 3

        # (B, H, W, D, 3) for broadcasting with neighbor offsets
        coords_perm = coords.permute(0, 2, 3, 4, 1)

        # Neighbor offset grid.  Order must match CorrBlock's meshgrid:
        # delta[i, j, k] = (i - r, j - r, k - r) under indexing='ij'.
        dy = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        dx = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        dz = torch.arange(-r, r + 1, device=coords.device, dtype=coords.dtype)
        delta = torch.stack(
            torch.meshgrid(dy, dx, dz, indexing="ij"),
            dim=-1,
        )  # (2r+1, 2r+1, 2r+1, 3)
        delta_flat = delta.reshape(n_neighbors, 3)

        out_pyramid: List[torch.Tensor] = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            centroid_lvl = coords_perm / (2 ** level)
            corr_level = self._lookup_one_level(
                fmap2_i, centroid_lvl, delta_flat, B, H, W, D,
            )
            out_pyramid.append(corr_level)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 4, 1, 2, 3).contiguous().float()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_one_level(
        self,
        fmap2_i: torch.Tensor,
        centroid_lvl: torch.Tensor,
        delta_flat: torch.Tensor,
        B: int, H: int, W: int, D: int,
    ) -> torch.Tensor:
        """Compute correlation values at one pyramid level.

        Optionally checkpointed at the *whole-level* granularity: backward
        re-runs the chunk loop once rather than re-running each individual
        chunk, which is dramatically cheaper in Python+kernel-launch
        overhead while still bounding peak memory to one chunk's
        intermediates during the forward pass.

        Returns shape (B, H, W, D, (2r+1)^3), normalized by sqrt(C).
        """
        if (
            self.use_checkpoint
            and self.fmap1.requires_grad
            and torch.is_grad_enabled()
        ):
            return torch_checkpoint.checkpoint(
                self._lookup_one_level_body,
                fmap2_i, centroid_lvl, delta_flat, B, H, W, D,
                use_reentrant=False,
            )
        return self._lookup_one_level_body(
            fmap2_i, centroid_lvl, delta_flat, B, H, W, D,
        )

    def _lookup_one_level_body(
        self,
        fmap2_i: torch.Tensor,
        centroid_lvl: torch.Tensor,
        delta_flat: torch.Tensor,
        B: int, H: int, W: int, D: int,
    ) -> torch.Tensor:
        """Pure chunk-loop body for one pyramid level (not checkpointed)."""
        n_neighbors = delta_flat.shape[0]
        corr_level = torch.empty(
            B, H, W, D, n_neighbors,
            device=self.fmap1.device,
            dtype=self.fmap1.dtype,
        )

        for chunk_start in range(0, n_neighbors, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_neighbors)
            chunk_delta = delta_flat[chunk_start:chunk_end]
            corr_chunk = _compute_chunk_body(
                self.fmap1, fmap2_i, centroid_lvl, chunk_delta,
            )
            corr_level[..., chunk_start:chunk_end] = corr_chunk

        return corr_level / self._norm_const


def _compute_chunk_body(
    fmap1: torch.Tensor,
    fmap2_i: torch.Tensor,
    centroid_lvl: torch.Tensor,
    chunk_delta: torch.Tensor,
) -> torch.Tensor:
    """Pure chunk computation: sample fmap2 at neighbors, dot with fmap1.

    Args:
        fmap1: (B, C, H, W, D).
        fmap2_i: (B, C, H_i, W_i, D_i) — pyramid level i.
        centroid_lvl: (B, H, W, D, 3) — flow coords scaled to level i.
        chunk_delta: (K, 3) — neighbor offsets in voxel units at level i.

    Returns:
        (B, H, W, D, K) un-normalized correlation values.
    """
    B, _, H, W, D = fmap1.shape
    K = chunk_delta.shape[0]

    # target_coords (B, H, W, D, K, 3) via broadcasting
    target_coords = (
        centroid_lvl.unsqueeze(-2)
        + chunk_delta.view(1, 1, 1, 1, K, 3)
    )

    # Flatten K into the D dimension so bilinear_sampler_3d's standard
    # (B, H', W', D', 3) interface accepts it.  grid_sample treats spatial
    # dims as independent query positions, so this is purely a shape trick.
    # legacy_wd_swap=True: the whole OTF path was validated for numerical
    # equivalence against the pre-2026-07-12 standard CorrBlock, which had a
    # W<->D axis swap in its sampler.  Until this module (and the CUDA
    # kernels) are ported to the fixed convention, OTF may ONLY be used with
    # corr_sampler_version=1 checkpoints (enforced in RAFTDVC.forward).
    target_flat = target_coords.reshape(B, H, W, D * K, 3)
    sampled = bilinear_sampler_3d(
        fmap2_i, target_flat, legacy_wd_swap=True)  # (B, C, H, W, D*K)
    sampled = sampled.reshape(B, fmap1.shape[1], H, W, D, K)

    return torch.einsum("bchwd,bchwdk->bhwdk", fmap1, sampled)
