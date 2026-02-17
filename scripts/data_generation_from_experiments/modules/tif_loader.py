"""
TIF file loader with support for large files.
"""

import numpy as np
from pathlib import Path
import tifffile
from typing import List, Dict, Tuple


class TifLoader:
    """Load and manage TIF/TIFF files."""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with 'input' section
        """
        self.config = config
        self.raw_data_dir = Path(config['input']['raw_data_dir'])
        self.file_pattern = config['input']['file_pattern']
        self.min_volume_size = config['input'].get('min_volume_size', [128, 128, 128])

        # Find all TIF files
        self.tif_files = self._discover_tif_files()

        # Cache for file metadata
        self.file_metadata = {}

    def _discover_tif_files(self) -> List[Path]:
        """
        Discover all TIF files in the raw data directory.

        Returns:
            List of Path objects to TIF files
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {self.raw_data_dir}"
            )

        # Find files matching pattern
        tif_files = list(self.raw_data_dir.glob(self.file_pattern))

        # Also try alternative pattern if specified
        if 'file_pattern_alternative' in self.config['input']:
            alt_pattern = self.config['input']['file_pattern_alternative']
            tif_files.extend(self.raw_data_dir.glob(alt_pattern))

        # Remove duplicates and sort
        tif_files = sorted(set(tif_files))

        if len(tif_files) == 0:
            raise ValueError(
                f"No TIF files found in {self.raw_data_dir} "
                f"matching pattern '{self.file_pattern}'"
            )

        print(f"Found {len(tif_files)} TIF files in {self.raw_data_dir}")

        return tif_files

    def get_file_shape(self, file_path: Path) -> Tuple[int, int, int]:
        """
        Get the shape of a TIF file without loading it into memory.

        Args:
            file_path: Path to TIF file

        Returns:
            Shape tuple (D, H, W)
        """
        if file_path in self.file_metadata:
            return self.file_metadata[file_path]['shape']

        with tifffile.TiffFile(file_path) as tif:
            n_pages = len(tif.pages)
            page_shape = tif.pages[0].shape  # (H, W) of a single page

            # Handle single-page case
            if len(page_shape) == 2:
                H, W = page_shape
                shape = (n_pages, H, W)
            elif len(page_shape) == 3:
                # Some TIFFs store channels in pages: (C, H, W)
                shape = (n_pages,) + page_shape
            else:
                raise ValueError(
                    f"Unexpected page dimensions: {page_shape} in {file_path.name}"
                )

            self.file_metadata[file_path] = {'shape': shape}

            return shape

    def load_file(self, file_path: Path) -> np.ndarray:
        """
        Load a TIF file into memory.
        Bypasses OME metadata to read actual page data directly.

        Args:
            file_path: Path to TIF file

        Returns:
            3D numpy array (D, H, W)
        """
        with tifffile.TiffFile(file_path) as tif:
            n_pages = len(tif.pages)
            page_shape = tif.pages[0].shape

            if n_pages == 1:
                # Single page - just read it
                volume = tif.pages[0].asarray()
                if volume.ndim == 2:
                    volume = volume[np.newaxis, ...]
            else:
                # Multiple pages - stack them (bypasses OME metadata issues)
                volume = np.array([page.asarray() for page in tif.pages])

        # Ensure 3D
        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D image after loading, got {volume.ndim}D from {file_path.name}"
            )

        # Convert to float32
        if volume.dtype != np.float32:
            if np.issubdtype(volume.dtype, np.integer):
                info = np.iinfo(volume.dtype)
                volume = volume.astype(np.float32) / info.max
            else:
                volume = volume.astype(np.float32)

        return volume

    def filter_valid_files(self) -> List[Path]:
        """
        Filter TIF files that are large enough to extract volumes.

        Returns:
            List of valid file paths
        """
        valid_files = []
        skipped_files = []

        min_d, min_h, min_w = self.min_volume_size

        for file_path in self.tif_files:
            shape = self.get_file_shape(file_path)
            d, h, w = shape

            if d >= min_d and h >= min_h and w >= min_w:
                valid_files.append(file_path)
            else:
                skipped_files.append((file_path.name, shape))

        # Report skipped files
        if skipped_files:
            print(f"\nSkipped {len(skipped_files)} files (too small):")
            for name, shape in skipped_files[:5]:  # Show first 5
                print(f"  - {name}: {shape}")
            if len(skipped_files) > 5:
                print(f"  ... and {len(skipped_files) - 5} more")

        print(f"\nValid files: {len(valid_files)}/{len(self.tif_files)}")

        if len(valid_files) == 0:
            raise ValueError(
                f"No files large enough to extract {self.min_volume_size} volumes"
            )

        return valid_files

    def get_summary(self) -> Dict:
        """
        Get summary of discovered TIF files.

        Returns:
            Dictionary with file statistics
        """
        valid_files = self.filter_valid_files()

        shapes = [self.get_file_shape(f) for f in valid_files]
        total_voxels = sum(np.prod(s) for s in shapes)

        return {
            'num_total_files': len(self.tif_files),
            'num_valid_files': len(valid_files),
            'valid_file_names': [f.name for f in valid_files],
            'shapes': shapes,
            'total_voxels': total_voxels,
            'min_shape': tuple(np.min(shapes, axis=0)),
            'max_shape': tuple(np.max(shapes, axis=0)),
            'mean_shape': tuple(np.mean(shapes, axis=0).astype(int))
        }
