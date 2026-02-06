"""
Volume I/O utilities for RAFT-DVC.

Centralizes all file format handling so that other modules
never import tifffile / h5py / nibabel directly.

Supported formats: .npy, .npz, .tif/.tiff, .h5/.hdf5, .nii/.nii.gz
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def load_volume(
    path: str,
    dtype: str = "float32",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load a 3D volume from disk.

    Args:
        path: File path (any supported format).
        dtype: Target numpy dtype string.
        device: Target torch device.

    Returns:
        Volume tensor of shape (1, 1, D, H, W).
    """
    raise NotImplementedError


def save_volume(
    path: str,
    volume: torch.Tensor,
) -> None:
    """
    Save a 3D volume to disk.

    Args:
        path: Output file path (format inferred from extension).
        volume: Tensor of shape (1, 1, D, H, W) or (D, H, W).
    """
    raise NotImplementedError


def save_flow(
    path: str,
    flow: torch.Tensor,
) -> None:
    """
    Save a 3D flow field to disk.

    Args:
        path: Output file path.
        flow: Tensor of shape (1, 3, D, H, W) or (3, D, H, W).
    """
    raise NotImplementedError


def load_flow(
    path: str,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load a 3D flow field from disk.

    Args:
        path: File path.
        device: Target torch device.

    Returns:
        Flow tensor of shape (1, 3, D, H, W).
    """
    raise NotImplementedError


def read_volume_shape(path: str) -> tuple:
    """
    Read volume shape from file header WITHOUT loading the full volume.

    This is used by VolumeAnalyzer to estimate memory/time before
    the user commits to loading the data.

    Args:
        path: File path.

    Returns:
        (D, H, W) shape tuple.
    """
    raise NotImplementedError
