"""
Correlation Volume for RAFT-DVC

Computes 3D correlation volumes between feature maps for optical flow estimation.
Based on VolRAFT (CVPR 2024) with improvements for clarity and memory efficiency.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def bilinear_sampler_3d(
    vol: torch.Tensor,
    coords: torch.Tensor,
    legacy_wd_swap: bool = False
) -> torch.Tensor:
    """
    Sample from 3D volume using trilinear interpolation.

    Args:
        vol: Input volume. Shape: (B, C, H, W, D)
        coords: Sampling coordinates in (h, w, d) channel order,
            matching coords_grid_3d. Shape: (B, H', W', D', 3)
        legacy_wd_swap: Reproduce the historical behaviour that swapped the
            W and D axes of the sampled volume (querying (h, w, d) returned
            vol[h, d, w]). Checkpoints trained before 2026-07-12 are
            co-adapted to that behaviour and MUST set this to True
            (handled automatically via RAFTDVCConfig.corr_sampler_version).

    Returns:
        Sampled values. Shape: (B, C, H', W', D')
    """
    B, C, H, W, D = vol.shape

    # Normalize pixel coordinates to [-1, 1] for grid_sample
    coords_normalized = coords.clone()
    coords_normalized[..., 0] = 2.0 * coords_normalized[..., 0] / (H - 1) - 1.0  # h
    coords_normalized[..., 1] = 2.0 * coords_normalized[..., 1] / (W - 1) - 1.0  # w
    coords_normalized[..., 2] = 2.0 * coords_normalized[..., 2] / (D - 1) - 1.0  # d

    # grid_sample 5-D convention: input (B, C, D_in, H_in, W_in), grid channels
    # (x, y, z) index (W_in, H_in, D_in).  With vol permuted to (B, C, D, H, W)
    # the correct channel order is (w, h, d) -> [1, 0, 2].
    if legacy_wd_swap:
        grid = coords_normalized[..., [2, 0, 1]]  # historical W<->D swap
    else:
        grid = coords_normalized[..., [1, 0, 2]]

    vol_permuted = vol.permute(0, 1, 4, 2, 3)  # (B, C, D, H, W)
    grid_permuted = grid.permute(0, 3, 1, 2, 4)  # (B, D', H', W', 3)

    output = F.grid_sample(
        vol_permuted,
        grid_permuted,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    # Permute back: (B, C, D, H, W) -> (B, C, H, W, D)
    output = output.permute(0, 1, 3, 4, 2)

    return output


def coords_grid_3d(
    batch: int, 
    ht: int, 
    wd: int, 
    dp: int, 
    device: torch.device
) -> torch.Tensor:
    """
    Create a coordinate grid for 3D volume.
    
    Args:
        batch: Batch size
        ht: Height
        wd: Width
        dp: Depth
        device: Target device
    
    Returns:
        Coordinate grid. Shape: (B, 3, H, W, D)
    """
    coords_h = torch.arange(ht, device=device, dtype=torch.float32)
    coords_w = torch.arange(wd, device=device, dtype=torch.float32)
    coords_d = torch.arange(dp, device=device, dtype=torch.float32)
    
    coords = torch.meshgrid(coords_h, coords_w, coords_d, indexing='ij')
    coords = torch.stack(coords, dim=0)  # (3, H, W, D)
    coords = coords.unsqueeze(0).repeat(batch, 1, 1, 1, 1)  # (B, 3, H, W, D)
    
    return coords


class CorrBlock:
    """
    3D Correlation Block for RAFT-DVC.
    
    Computes multi-scale correlation volumes between two feature maps
    and provides lookup functionality for iterative updates.
    
    Args:
        fmap1: Feature map from first volume. Shape: (B, C, H, W, D)
        fmap2: Feature map from second volume. Shape: (B, C, H, W, D)
        num_levels: Number of pyramid levels
        radius: Lookup radius for correlation
    """
    
    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        legacy_wd_swap: bool = False
    ):
        self.num_levels = num_levels
        self.radius = radius
        self.legacy_wd_swap = legacy_wd_swap
        self.corr_pyramid: List[torch.Tensor] = []
        
        # Compute all-pairs correlation
        corr = self._compute_correlation(fmap1, fmap2)
        
        batch, h1, w1, d1, dim, h2, w2, d2 = corr.shape
        corr = corr.reshape(batch * h1 * w1 * d1, dim, h2, w2, d2)
        
        # Build correlation pyramid
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool3d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
    
    @staticmethod
    def _compute_correlation(
        fmap1: torch.Tensor, 
        fmap2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute all-pairs correlation between two feature maps.
        
        Args:
            fmap1, fmap2: Feature maps. Shape: (B, C, H, W, D)
        
        Returns:
            Correlation volume. Shape: (B, H, W, D, 1, H, W, D)
        """
        batch, dim, ht, wd, dp = fmap1.shape

        fmap1_flat = fmap1.reshape(batch, dim, ht * wd * dp)
        fmap2_flat = fmap2.reshape(batch, dim, ht * wd * dp)

        # Dot product correlation
        corr = torch.matmul(fmap1_flat.transpose(1, 2), fmap2_flat)
        corr = corr.reshape(batch, ht, wd, dp, 1, ht, wd, dp)
        
        # Normalize by feature dimension
        corr = corr / torch.sqrt(torch.tensor(dim, dtype=torch.float32, device=corr.device))
        
        return corr
    
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Lookup correlation values at given coordinates.
        
        Args:
            coords: Current flow coordinates. Shape: (B, 3, H, W, D)
        
        Returns:
            Correlation features. Shape: (B, num_levels * (2*radius+1)^3, H, W, D)
        """
        r = self.radius
        coords = coords.permute(0, 2, 3, 4, 1)  # (B, H, W, D, 3)
        batch, h1, w1, d1, _ = coords.shape
        
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            
            # Create sampling grid
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dz = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(
                torch.meshgrid(dy, dx, dz, indexing='ij'), 
                dim=-1
            )
            
            # Compute sampling coordinates at this pyramid level
            centroid_lvl = coords.reshape(batch * h1 * w1 * d1, 1, 1, 1, 3) / 2**i
            delta_lvl = delta.reshape(1, 2*r+1, 2*r+1, 2*r+1, 3)
            coords_lvl = centroid_lvl + delta_lvl
            
            # Sample from correlation volume
            corr_sampled = bilinear_sampler_3d(
                corr, coords_lvl, legacy_wd_swap=self.legacy_wd_swap)
            corr_sampled = corr_sampled.reshape(batch, h1, w1, d1, -1)
            out_pyramid.append(corr_sampled)
        
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 4, 1, 2, 3).contiguous().float()


def upflow_3d(
    flow: torch.Tensor, 
    target_shape: Optional[Tuple[int, int, int]] = None,
    scale_factor: int = 8
) -> torch.Tensor:
    """
    Upsample 3D flow field.
    
    Args:
        flow: Input flow. Shape: (B, 3, H, W, D)
        target_shape: Target (H, W, D) shape. If None, uses scale_factor.
        scale_factor: Upsampling factor if target_shape is None.
    
    Returns:
        Upsampled flow. Shape: (B, 3, H', W', D')
    """
    if target_shape is not None:
        new_size = target_shape
    else:
        _, _, h, w, d = flow.shape
        new_size = (h * scale_factor, w * scale_factor, d * scale_factor)
    
    # Trilinear interpolation for flow upsampling
    flow_up = F.interpolate(
        flow, 
        size=new_size, 
        mode='trilinear', 
        align_corners=True
    )
    
    # Scale flow values proportionally
    if target_shape is not None:
        scale_h = new_size[0] / flow.shape[2]
        scale_w = new_size[1] / flow.shape[3]
        scale_d = new_size[2] / flow.shape[4]
    else:
        scale_h = scale_w = scale_d = scale_factor
    
    flow_up[:, 0] *= scale_h
    flow_up[:, 1] *= scale_w
    flow_up[:, 2] *= scale_d
    
    return flow_up
