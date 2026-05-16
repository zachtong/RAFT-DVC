"""
Dataset classes for RAFT-DVC

Provides PyTorch Dataset classes for loading 3D volume pairs
for training and inference.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, Any, List, Callable
from pathlib import Path
import tifffile


class VolumeDataset(Dataset):
    """
    Dataset for loading individual 3D volumes.
    
    Supports multiple file formats:
    - NumPy (.npy, .npz)
    - TIFF stacks (.tif, .tiff)
    - HDF5 (.h5, .hdf5)
    
    Args:
        root_dir: Directory containing volume files
        pattern: Glob pattern for file selection (e.g., '*.npy')
        transform: Optional transform to apply to volumes
        cache: Whether to cache loaded volumes in memory
    """
    
    def __init__(
        self,
        root_dir: str,
        pattern: str = '*.npy',
        transform: Optional[Callable] = None,
        cache: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {}
        
        # Find all matching files
        self.files = sorted(list(self.root_dir.glob(pattern)))
        
        if len(self.files) == 0:
            raise ValueError(f"No files found in {root_dir} matching pattern {pattern}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def _load_volume(self, path: Path) -> np.ndarray:
        """Load volume from file based on extension."""
        ext = path.suffix.lower()
        
        if ext == '.npy':
            return np.load(str(path))
        elif ext == '.npz':
            data = np.load(str(path))
            # Assume first key is the volume
            key = list(data.keys())[0]
            return data[key]
        elif ext in ['.tif', '.tiff']:
            return tifffile.imread(str(path))
        elif ext in ['.h5', '.hdf5']:
            import h5py  # lazy import: HDF5 is optional
            with h5py.File(str(path), 'r') as f:
                # Assume first dataset is the volume
                key = list(f.keys())[0]
                return f[key][:]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.cache and idx in self._cache:
            volume = self._cache[idx]
        else:
            volume = self._load_volume(self.files[idx])
            if self.cache:
                self._cache[idx] = volume
        
        # Convert to tensor
        volume = torch.from_numpy(volume.astype(np.float32))
        
        # Add channel dimension if needed
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        
        if self.transform is not None:
            volume = self.transform(volume)
        
        return volume


class VolumePairDataset(Dataset):
    """
    Dataset for loading 3D volume pairs with ground truth flow.
    
    Expected directory structure:
    root_dir/
        vol0/           # Reference volumes
            sample_001.npy
            sample_002.npy
            ...
        vol1/           # Deformed volumes
            sample_001.npy
            sample_002.npy
            ...
        flow/           # Ground truth flow (optional)
            sample_001.npy
            sample_002.npy
            ...
    
    Args:
        root_dir: Root directory containing vol0, vol1, and optionally flow subdirs
        transform: Optional transform to apply to volume pairs
        augment: Whether to apply data augmentation
        patch_size: If specified, extract random patches of this size
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
        patch_size: Optional[Tuple[int, int, int]] = None,
        has_flow: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.patch_size = patch_size
        self.has_flow = has_flow
        
        # Find volume pairs
        self.vol0_dir = self.root_dir / 'vol0'
        self.vol1_dir = self.root_dir / 'vol1'
        self.flow_dir = self.root_dir / 'flow' if has_flow else None

        # Auto-detect mask directory (for cutout datasets)
        self.mask_dir = self.root_dir / 'mask'
        self.has_mask = self.mask_dir.exists()
        
        if not self.vol0_dir.exists():
            raise ValueError(f"vol0 directory not found: {self.vol0_dir}")
        if not self.vol1_dir.exists():
            raise ValueError(f"vol1 directory not found: {self.vol1_dir}")
        
        # Get file list (assumes matching filenames)
        self.files = sorted([f.name for f in self.vol0_dir.glob('*.npy')])
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {self.vol0_dir}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def _load_volume(self, path: Path) -> np.ndarray:
        """Load volume from NumPy file."""
        return np.load(str(path))
    
    def _random_crop(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> Tuple:
        """Extract random patch from volumes and flow."""
        h, w, d = vol0.shape[-3:]
        ph, pw, pd = self.patch_size
        
        # Random start positions
        sh = np.random.randint(0, h - ph + 1)
        sw = np.random.randint(0, w - pw + 1)
        sd = np.random.randint(0, d - pd + 1)
        
        vol0_patch = vol0[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]
        vol1_patch = vol1[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]
        
        if flow is not None:
            flow_patch = flow[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]
            return vol0_patch, vol1_patch, flow_patch
        
        return vol0_patch, vol1_patch, None
    
    def _augment(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> Tuple:
        """Apply random data augmentation."""
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                vol0 = np.flip(vol0, axis=-3+axis)
                vol1 = np.flip(vol1, axis=-3+axis)
                if flow is not None:
                    flow = np.flip(flow, axis=-3+axis)
                    # Flip flow component
                    flow[axis] = -flow[axis]
        
        # Random 90-degree rotation in xy plane
        k = np.random.randint(0, 4)
        if k > 0:
            vol0 = np.rot90(vol0, k, axes=(-3, -2))
            vol1 = np.rot90(vol1, k, axes=(-3, -2))
            if flow is not None:
                flow = np.rot90(flow, k, axes=(-3, -2))
                # Rotate flow vectors
                if k == 1:  # 90 deg
                    flow[[0, 1]] = flow[[1, 0]]
                    flow[0] = -flow[0]
                elif k == 2:  # 180 deg
                    flow[[0, 1]] = -flow[[0, 1]]
                elif k == 3:  # 270 deg
                    flow[[0, 1]] = flow[[1, 0]]
                    flow[1] = -flow[1]
        
        return (
            np.ascontiguousarray(vol0),
            np.ascontiguousarray(vol1),
            np.ascontiguousarray(flow) if flow is not None else None
        )
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.files[idx]
        
        # Load volumes
        vol0 = self._load_volume(self.vol0_dir / filename)
        vol1 = self._load_volume(self.vol1_dir / filename)
        
        # Load flow if available
        flow = None
        if self.has_flow and self.flow_dir is not None:
            flow_path = self.flow_dir / filename
            if flow_path.exists():
                flow = self._load_volume(flow_path)
        
        # Random crop if patch_size specified
        if self.patch_size is not None:
            vol0, vol1, flow = self._random_crop(vol0, vol1, flow)
        
        # Data augmentation
        if self.augment:
            vol0, vol1, flow = self._augment(vol0, vol1, flow)
        
        # Convert to tensors
        vol0 = torch.from_numpy(vol0.astype(np.float32))
        vol1 = torch.from_numpy(vol1.astype(np.float32))
        
        # Add channel dimension if needed
        if vol0.ndim == 3:
            vol0 = vol0.unsqueeze(0)
            vol1 = vol1.unsqueeze(0)
        
        result = {
            'vol0': vol0,
            'vol1': vol1,
            'filename': filename
        }
        
        if flow is not None:
            flow = torch.from_numpy(flow.astype(np.float32))
            result['flow'] = flow

        # Load mask if available (cutout datasets)
        if self.has_mask:
            mask_path = self.mask_dir / filename
            if mask_path.exists():
                mask = self._load_volume(mask_path)
                result['mask'] = torch.from_numpy(mask.astype(np.float32))

        if self.transform is not None:
            result = self.transform(result)

        return result


class InferenceDataset(Dataset):
    """
    Dataset for inference on volume pairs.
    
    Simple dataset that loads pairs of volumes for inference.
    Does not expect ground truth flow.
    
    Args:
        vol0_paths: List of paths to reference volumes
        vol1_paths: List of paths to deformed volumes
    """
    
    def __init__(
        self,
        vol0_paths: List[str],
        vol1_paths: List[str]
    ):
        assert len(vol0_paths) == len(vol1_paths), \
            "Number of vol0 and vol1 files must match"
        
        self.vol0_paths = [Path(p) for p in vol0_paths]
        self.vol1_paths = [Path(p) for p in vol1_paths]
    
    def __len__(self) -> int:
        return len(self.vol0_paths)
    
    def _load_volume(self, path: Path) -> np.ndarray:
        """Load volume from file based on extension."""
        ext = path.suffix.lower()
        
        if ext == '.npy':
            return np.load(str(path))
        elif ext in ['.tif', '.tiff']:
            return tifffile.imread(str(path))
        elif ext in ['.h5', '.hdf5']:
            import h5py  # lazy import: HDF5 is optional
            with h5py.File(str(path), 'r') as f:
                key = list(f.keys())[0]
                return f[key][:]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol0 = self._load_volume(self.vol0_paths[idx])
        vol1 = self._load_volume(self.vol1_paths[idx])
        
        vol0 = torch.from_numpy(vol0.astype(np.float32))
        vol1 = torch.from_numpy(vol1.astype(np.float32))
        
        if vol0.ndim == 3:
            vol0 = vol0.unsqueeze(0)
            vol1 = vol1.unsqueeze(0)
        
        return {
            'vol0': vol0,
            'vol1': vol1,
            'filename': self.vol0_paths[idx].stem
        }


class VolRAFTDataset(Dataset):
    """
    Dataset for loading VolRAFT format data.

    Expected directory structure:
    root_dir/
        samples/
            sample_0000000/
                v0.npy      - Reference volume (1, D, H, W)
                v1.npy      - Deformed volume (1, D, H, W)
                flow.npy    - Ground truth flow (3, D, H, W)
                vm.npy      - Middle volume (optional)
                meta.json   - Sample metadata
            sample_0000001/
                ...

    This format is used by VolRAFT synthetic datasets where each sample
    is stored in its own directory.

    Args:
        root_dir: Root directory containing 'samples' subdirectory
        transform: Optional transform to apply to volume pairs
        augment: Whether to apply data augmentation
        patch_size: If specified, extract random patches of this size
        has_flow: Whether ground truth flow is available
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        augment: bool = False,
        patch_size: Optional[Tuple[int, int, int]] = None,
        has_flow: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.augment = augment
        self.patch_size = patch_size
        self.has_flow = has_flow

        # Find all sample directories
        samples_dir = self.root_dir / 'samples'
        if not samples_dir.exists():
            # If 'samples' subdir doesn't exist, assume root_dir IS the samples dir
            samples_dir = self.root_dir

        self.sample_dirs = sorted(list(samples_dir.glob('sample_*')))

        if len(self.sample_dirs) == 0:
            raise ValueError(f"No sample directories found in {samples_dir}")

        print(f"Found {len(self.sample_dirs)} samples in VolRAFT format")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_volume(self, path: Path) -> np.ndarray:
        """Load volume from NumPy file."""
        return np.load(str(path))

    def _random_crop(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> Tuple:
        """Extract random patch from volumes and flow."""
        h, w, d = vol0.shape[-3:]
        ph, pw, pd = self.patch_size

        # Random start positions
        sh = np.random.randint(0, h - ph + 1)
        sw = np.random.randint(0, w - pw + 1)
        sd = np.random.randint(0, d - pd + 1)

        vol0_patch = vol0[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]
        vol1_patch = vol1[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]

        if flow is not None:
            flow_patch = flow[..., sh:sh+ph, sw:sw+pw, sd:sd+pd]
            return vol0_patch, vol1_patch, flow_patch

        return vol0_patch, vol1_patch, None

    def _augment(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: Optional[np.ndarray] = None
    ) -> Tuple:
        """Apply random data augmentation."""
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                vol0 = np.flip(vol0, axis=-3+axis)
                vol1 = np.flip(vol1, axis=-3+axis)
                if flow is not None:
                    flow = np.flip(flow, axis=-3+axis)
                    # Flip flow component
                    flow[axis] = -flow[axis]

        # Random 90-degree rotation in xy plane
        k = np.random.randint(0, 4)
        if k > 0:
            vol0 = np.rot90(vol0, k, axes=(-3, -2))
            vol1 = np.rot90(vol1, k, axes=(-3, -2))
            if flow is not None:
                flow = np.rot90(flow, k, axes=(-3, -2))
                # Rotate flow vectors
                if k == 1:  # 90 deg
                    flow[[0, 1]] = flow[[1, 0]]
                    flow[0] = -flow[0]
                elif k == 2:  # 180 deg
                    flow[[0, 1]] = -flow[[0, 1]]
                elif k == 3:  # 270 deg
                    flow[[0, 1]] = flow[[1, 0]]
                    flow[1] = -flow[1]

        return (
            np.ascontiguousarray(vol0),
            np.ascontiguousarray(vol1),
            np.ascontiguousarray(flow) if flow is not None else None
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]

        # Load volumes (shape: (1, D, H, W))
        vol0 = self._load_volume(sample_dir / 'v0.npy')
        vol1 = self._load_volume(sample_dir / 'v1.npy')

        # Load flow if available (shape: (3, D, H, W))
        flow = None
        if self.has_flow:
            flow_path = sample_dir / 'flow.npy'
            if flow_path.exists():
                flow = self._load_volume(flow_path)

        # Load metadata if available
        meta = None
        meta_path = sample_dir / 'meta.json'
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)

        # Random crop if patch_size specified
        if self.patch_size is not None:
            vol0, vol1, flow = self._random_crop(vol0, vol1, flow)

        # Data augmentation
        if self.augment:
            vol0, vol1, flow = self._augment(vol0, vol1, flow)

        # Convert to tensors
        vol0 = torch.from_numpy(vol0.astype(np.float32))
        vol1 = torch.from_numpy(vol1.astype(np.float32))

        # VolRAFT format already has channel dimension (1, D, H, W)
        # No need to add it

        result = {
            'vol0': vol0,
            'vol1': vol1,
            'filename': sample_dir.name
        }

        if flow is not None:
            flow = torch.from_numpy(flow.astype(np.float32))
            result['flow'] = flow

        if meta is not None:
            result['meta'] = meta

        if self.transform is not None:
            result = self.transform(result)

        return result


class Phase1NPZDataset(Dataset):
    """
    Dataset for the Phase-1 synthetic NPZ format.

    Each sample is a single compressed NumPy file with keys:
        I1   : (D, H, W) float32 -- reference volume (noisy)
        I2   : (D, H, W) float32 -- warped volume  (noisy, independent noise)
        flow : (3, D, H, W) float32 -- ground-truth displacement (U, V, W)

    Plus metadata scalars (deform_type, fm_target, input_max_disp, radius,
    density, size, seed, num_beads).

    Expected directory layout::

        root_dir/
            sample_00000.npz
            sample_00001.npz
            ...

    Typically `root_dir` points at `data_phase1/<config>/<split>/`
    (e.g. `data_phase1/r2_sparse_size32/train/`).

    The dataset maps the stored keys to the interface the Trainer expects:
        I1 -> vol0  (with channel dim added)
        I2 -> vol1  (with channel dim added)
        flow -> flow

    Args:
        root_dir: Directory containing `sample_*.npz` files.
        augment: Whether to apply random flips + 90-deg rotations.
        patch_size: Optional (D, H, W) random crop size (None = full volume).
        include_metadata: Whether to include the per-sample metadata dict.
    """

    # Keys expected in every NPZ sample
    _VOLUME_KEYS = ('I1', 'I2', 'flow')

    def __init__(
        self,
        root_dir: str,
        augment: bool = False,
        patch_size: Optional[Tuple[int, int, int]] = None,
        include_metadata: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.patch_size = patch_size
        self.include_metadata = include_metadata

        if not self.root_dir.exists():
            raise ValueError(f"Phase-1 dataset root not found: {self.root_dir}")

        self.files = sorted(self.root_dir.glob('sample_*.npz'))
        if len(self.files) == 0:
            raise ValueError(
                f"No `sample_*.npz` files found in {self.root_dir}"
            )

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _augment_pair(
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Random flips + 90-deg rotations. Flow vectors transformed accordingly.

        vol0 / vol1: (D, H, W)  -- NO channel dim yet
        flow      : (3, D, H, W) with components (U, V, W) = (z, y, x)
        """
        # Axis mapping for flow components: axis 0 (D) <-> flow[0],
        # axis 1 (H) <-> flow[1], axis 2 (W) <-> flow[2]
        for axis in range(3):
            if np.random.random() > 0.5:
                vol0 = np.flip(vol0, axis=axis)
                vol1 = np.flip(vol1, axis=axis)
                flow = np.flip(flow, axis=axis + 1)  # +1: skip channel dim
                flow[axis] = -flow[axis]

        # Random 90-deg rotation in the H-W plane (axes=(1,2) of vols,
        # axes=(2,3) of flow which has a leading channel dim).
        k = int(np.random.randint(0, 4))
        if k > 0:
            vol0 = np.rot90(vol0, k, axes=(1, 2))
            vol1 = np.rot90(vol1, k, axes=(1, 2))
            flow = np.rot90(flow, k, axes=(2, 3))
            # Rotate the in-plane (H, W) = flow[1], flow[2] components
            for _ in range(k):
                # 90-deg CCW in (H, W): (fy, fx) -> (-fx, fy)
                fy, fx = flow[1].copy(), flow[2].copy()
                flow[1] = -fx
                flow[2] = fy

        return (
            np.ascontiguousarray(vol0),
            np.ascontiguousarray(vol1),
            np.ascontiguousarray(flow),
        )

    def _random_crop(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray,
        flow: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract matching random patch from vol0/vol1/flow."""
        d, h, w = vol0.shape
        pd, ph, pw = self.patch_size
        sd = np.random.randint(0, d - pd + 1)
        sh = np.random.randint(0, h - ph + 1)
        sw = np.random.randint(0, w - pw + 1)
        return (
            vol0[sd:sd + pd, sh:sh + ph, sw:sw + pw],
            vol1[sd:sd + pd, sh:sh + ph, sw:sw + pw],
            flow[:, sd:sd + pd, sh:sh + ph, sw:sw + pw],
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        # `allow_pickle=True` is required because metadata is stored as 0-d
        # object arrays (e.g. `deform_type` is a string).
        with np.load(str(path), allow_pickle=True) as data:
            I1 = data['I1']
            I2 = data['I2']
            flow = data['flow']
            if self.include_metadata and 'metadata' in data.files:
                # Stored as a 0-d object array wrapping either a dict or a
                # JSON string (the generator uses json.dumps).
                meta_obj = data['metadata'].item()
                if isinstance(meta_obj, str):
                    import json
                    meta = json.loads(meta_obj)
                elif isinstance(meta_obj, dict):
                    meta = dict(meta_obj)
                else:
                    meta = None
            else:
                meta = None

        # Optional random crop
        if self.patch_size is not None:
            I1, I2, flow = self._random_crop(I1, I2, flow)

        # Augmentation
        if self.augment:
            I1, I2, flow = self._augment_pair(I1, I2, flow)

        # To tensor + add channel dim to volumes
        vol0 = torch.from_numpy(I1.astype(np.float32)).unsqueeze(0)
        vol1 = torch.from_numpy(I2.astype(np.float32)).unsqueeze(0)
        flow_t = torch.from_numpy(flow.astype(np.float32))

        result: Dict[str, Any] = {
            'vol0': vol0,
            'vol1': vol1,
            'flow': flow_t,
            'filename': path.name,
        }
        if meta is not None:
            result['meta'] = meta
        return result
