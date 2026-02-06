"""
Memory estimation formulas for RAFT-DVC inference planning.

All functions are pure (no side effects, no file I/O, no GPU access).
They can be unit-tested with known expected values.

Usage:
    from src.utils.memory import estimate_6d_correlation_bytes, estimate_tile_count
"""

import math
from typing import Tuple


def estimate_6d_correlation_bytes(tile_size: int, dtype_bytes: int = 4) -> int:
    """
    Estimate memory for the 6D all-pairs correlation volume.

    For a tile of size S, the correlation volume is S^6 elements.
    This is the dominant memory consumer in RAFT-DVC inference.

    Args:
        tile_size: Side length of the cubic tile (e.g. 32).
        dtype_bytes: Bytes per element (4 for float32).

    Returns:
        Memory in bytes.
    """
    raise NotImplementedError


def estimate_pyramid_bytes(
    tile_size: int,
    num_levels: int,
    dtype_bytes: int = 4,
) -> int:
    """
    Estimate total memory for the correlation pyramid (all levels combined).

    Level 0 is the full correlation volume (tile^6).
    Level k pools the last 3 dims by 2^k.

    Args:
        tile_size: Side length of the cubic tile.
        num_levels: Number of pyramid levels (e.g. 2 or 4).
        dtype_bytes: Bytes per element.

    Returns:
        Total memory in bytes across all pyramid levels.
    """
    raise NotImplementedError


def estimate_tile_peak_memory_bytes(
    tile_size: int,
    num_levels: int,
    feature_dim: int = 128,
    hidden_dim: int = 96,
    context_dim: int = 64,
    dtype_bytes: int = 4,
) -> int:
    """
    Estimate peak GPU memory for processing one tile.

    Includes: correlation pyramid, feature maps, hidden state,
    context features, flow field, and a safety margin for CUDA overhead.

    Args:
        tile_size: Cubic tile side length.
        num_levels: Pyramid levels.
        feature_dim: Feature encoder output channels.
        hidden_dim: GRU hidden state channels.
        context_dim: Context encoder output channels.
        dtype_bytes: Bytes per element.

    Returns:
        Estimated peak memory in bytes.
    """
    raise NotImplementedError


def estimate_tile_count(
    volume_shape: Tuple[int, int, int],
    tile_size: int,
    overlap: int,
) -> int:
    """
    Estimate total number of tiles for a given volume.

    Args:
        volume_shape: (D, H, W) of the processed volume.
        tile_size: Cubic tile side length.
        overlap: Overlap in voxels between adjacent tiles.

    Returns:
        Total number of tiles.
    """
    raise NotImplementedError


def estimate_max_displacement(
    pyramid_levels: int,
    radius: int,
    downsample_factor: int = 1,
) -> float:
    """
    Estimate maximum displacement (in original-resolution voxels) that a
    model configuration can capture.

    Formula: radius * 2^(pyramid_levels - 1) * downsample_factor

    Args:
        pyramid_levels: Number of correlation pyramid levels.
        radius: Correlation lookup radius.
        downsample_factor: User pre-downsampling factor.

    Returns:
        Max displacement in original-resolution voxels.
    """
    raise NotImplementedError


def estimate_precision(downsample_factor: int) -> float:
    """
    Rough estimate of sub-voxel precision (in original-resolution voxels).

    Args:
        downsample_factor: User pre-downsampling factor.

    Returns:
        Estimated precision (smaller is better).
    """
    raise NotImplementedError


def estimate_inference_seconds(
    tile_count: int,
    seconds_per_tile: float = 0.1,
    num_stages: int = 1,
) -> float:
    """
    Rough wall-clock time estimate for inference.

    Args:
        tile_count: Total tiles to process.
        seconds_per_tile: Average time per tile (hardware-dependent).
        num_stages: Number of inference stages.

    Returns:
        Estimated total seconds.
    """
    raise NotImplementedError
