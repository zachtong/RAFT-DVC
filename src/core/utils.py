"""
Utility functions for RAFT-DVC core module.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_batch_norm(model: torch.nn.Module):
    """Freeze all batch normalization layers in the model."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()


def warp_volume_3d(
    volume: torch.Tensor, 
    flow: torch.Tensor
) -> torch.Tensor:
    """
    Warp a 3D volume using a displacement field.
    
    Args:
        volume: Input volume. Shape: (B, C, H, W, D)
        flow: Displacement field. Shape: (B, 3, H, W, D)
               Where flow[:, 0] is displacement in H (y),
               flow[:, 1] is displacement in W (x),
               flow[:, 2] is displacement in D (z).
    
    Returns:
        Warped volume. Shape: (B, C, H, W, D)
    """
    B, C, H, W, D = volume.shape
    
    # Create base grid
    grid_h = torch.linspace(-1, 1, H, device=volume.device)
    grid_w = torch.linspace(-1, 1, W, device=volume.device)
    grid_d = torch.linspace(-1, 1, D, device=volume.device)
    
    grid = torch.stack(
        torch.meshgrid(grid_h, grid_w, grid_d, indexing='ij'), 
        dim=-1
    )  # (H, W, D, 3)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B, H, W, D, 3)
    
    # Normalize flow to [-1, 1] range
    flow_normalized = flow.clone()
    flow_normalized[:, 0] = 2.0 * flow[:, 0] / (H - 1)  # y direction
    flow_normalized[:, 1] = 2.0 * flow[:, 1] / (W - 1)  # x direction
    flow_normalized[:, 2] = 2.0 * flow[:, 2] / (D - 1)  # z direction
    
    # Add flow to grid
    flow_permuted = flow_normalized.permute(0, 2, 3, 4, 1)  # (B, H, W, D, 3)
    grid = grid + flow_permuted
    
    # Rearrange for grid_sample (expects z, y, x order)
    # grid_sample for 5D: grid is (B, D_out, H_out, W_out, 3) with (x, y, z) coords
    # Our grid is (B, H, W, D, 3) in (y, x, z) order
    grid_reordered = grid[..., [2, 0, 1]]  # Reorder to (z, y, x)
    grid_reordered = grid_reordered.permute(0, 3, 1, 2, 4)  # (B, D, H, W, 3)
    
    # Warp volume
    volume_permuted = volume.permute(0, 1, 4, 2, 3)  # (B, C, D, H, W)
    warped = F.grid_sample(
        volume_permuted, 
        grid_reordered, 
        mode='bilinear', 
        padding_mode='zeros',
        align_corners=True
    )
    
    # Permute back
    warped = warped.permute(0, 1, 3, 4, 2)  # (B, C, H, W, D)
    
    return warped


def compute_flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute the magnitude of a 3D flow field.
    
    Args:
        flow: Flow field. Shape: (B, 3, H, W, D)
    
    Returns:
        Flow magnitude. Shape: (B, 1, H, W, D)
    """
    return torch.sqrt(torch.sum(flow ** 2, dim=1, keepdim=True))


def flow_to_color_3d(
    flow: torch.Tensor, 
    max_flow: Optional[float] = None,
    slice_axis: int = 0,
    slice_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Convert 3D flow to RGB color visualization for a 2D slice.
    
    Uses HSV color wheel encoding where:
    - Hue represents flow direction
    - Saturation represents flow magnitude
    
    Args:
        flow: Flow field. Shape: (B, 3, H, W, D)
        max_flow: Maximum flow value for normalization. If None, uses data max.
        slice_axis: Axis to slice (0=H, 1=W, 2=D)
        slice_idx: Index of slice. If None, uses middle slice.
    
    Returns:
        RGB image. Shape: (B, 3, H', W') where H', W' are remaining dims after slicing.
    """
    B = flow.shape[0]
    
    # Get slice
    if slice_idx is None:
        slice_idx = flow.shape[slice_axis + 2] // 2
    
    if slice_axis == 0:
        flow_slice = flow[:, :, slice_idx, :, :]  # (B, 3, W, D)
        flow_2d = flow_slice[:, [1, 2]]  # Use x, z components
    elif slice_axis == 1:
        flow_slice = flow[:, :, :, slice_idx, :]  # (B, 3, H, D)
        flow_2d = flow_slice[:, [0, 2]]  # Use y, z components
    else:  # slice_axis == 2
        flow_slice = flow[:, :, :, :, slice_idx]  # (B, 3, H, W)
        flow_2d = flow_slice[:, [0, 1]]  # Use y, x components
    
    # Compute magnitude
    magnitude = torch.sqrt(torch.sum(flow_2d ** 2, dim=1))  # (B, H', W')
    
    if max_flow is None:
        max_flow = magnitude.max().item()
    
    if max_flow > 0:
        magnitude = magnitude / max_flow
    
    # Compute angle
    angle = torch.atan2(flow_2d[:, 1], flow_2d[:, 0])  # (B, H', W')
    angle = (angle / (2 * np.pi) + 0.5) % 1.0  # Normalize to [0, 1]
    
    # HSV to RGB
    # H = angle, S = magnitude, V = 1
    hsv = torch.stack([angle, magnitude, torch.ones_like(magnitude)], dim=1)  # (B, 3, H', W')
    
    # Convert HSV to RGB
    h = hsv[:, 0] * 6.0
    s = hsv[:, 1]
    v = hsv[:, 2]
    
    i = torch.floor(h).long()
    f = h - i.float()
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    i = i % 6
    
    rgb = torch.zeros(B, 3, *hsv.shape[2:], device=flow.device)
    
    mask = (i == 0).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([v, t, p], dim=1), rgb)
    mask = (i == 1).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([q, v, p], dim=1), rgb)
    mask = (i == 2).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([p, v, t], dim=1), rgb)
    mask = (i == 3).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([p, q, v], dim=1), rgb)
    mask = (i == 4).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([t, p, v], dim=1), rgb)
    mask = (i == 5).unsqueeze(1)
    rgb = torch.where(mask, torch.stack([v, p, q], dim=1), rgb)
    
    return rgb
