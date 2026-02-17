"""
Volume extraction from large TIF files.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class VolumeExtractor:
    """Extract fixed-size volumes from large TIF files."""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with 'extraction' section
        """
        self.config = config
        self.volume_size = tuple(config['extraction']['volume_size'])
        self.strategy = config['extraction']['strategy']
        self.total_volumes = config['extraction'].get('total_volumes', 100)
        self.margin = config['extraction'].get('margin', 10)

        # Quality filters
        self.min_mean_intensity = config['extraction'].get('min_mean_intensity', 0.05)
        self.min_std_intensity = config['extraction'].get('min_std_intensity', 0.02)

        # Random seed
        self.rng = np.random.RandomState(config.get('random_seed', 42))

    def extract_volumes(
        self,
        tif_volumes: List[np.ndarray],
        tif_file_names: List[str]
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Extract multiple volumes from TIF files.

        Args:
            tif_volumes: List of loaded TIF volumes
            tif_file_names: List of corresponding file names

        Returns:
            extracted_volumes: List of extracted volumes
            metadata: List of extraction metadata (source file, position, etc.)
        """
        if self.strategy == 'random':
            return self._extract_random(tif_volumes, tif_file_names)
        else:
            raise ValueError(f"Unknown extraction strategy: {self.strategy}")

    def _extract_random(
        self,
        tif_volumes: List[np.ndarray],
        tif_file_names: List[str]
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Randomly extract volumes from TIF files.

        Strategy: Randomly sample positions from all valid regions across all files.

        Args:
            tif_volumes: List of loaded TIF volumes
            tif_file_names: List of file names

        Returns:
            extracted_volumes: List of extracted volumes
            metadata: List of metadata dicts
        """
        extracted_volumes = []
        metadata_list = []

        # Calculate how many valid positions exist in each file
        valid_positions_per_file = []
        for vol in tif_volumes:
            valid_pos = self._count_valid_positions(vol.shape)
            valid_positions_per_file.append(valid_pos)

        total_valid_positions = sum(valid_positions_per_file)

        if total_valid_positions == 0:
            raise ValueError("No valid extraction positions found in any file")

        print(f"\nExtracting {self.total_volumes} volumes randomly...")
        print(f"Total valid positions: {total_valid_positions}")

        # Extract volumes with retry logic for quality filtering
        max_attempts = self.total_volumes * 10  # Allow multiple retries
        attempts = 0

        while len(extracted_volumes) < self.total_volumes and attempts < max_attempts:
            attempts += 1

            # Randomly select a file (weighted by number of valid positions)
            file_idx = self.rng.choice(
                len(tif_volumes),
                p=np.array(valid_positions_per_file) / total_valid_positions
            )

            vol = tif_volumes[file_idx]
            file_name = tif_file_names[file_idx]

            # Get a random valid position
            position = self._get_random_position(vol.shape)

            # Extract volume
            extracted = self._extract_at_position(vol, position)

            # Quality check
            if self._passes_quality_check(extracted):
                extracted_volumes.append(extracted)

                metadata_list.append({
                    'source_file': file_name,
                    'position': position,
                    'file_index': file_idx,
                    'mean_intensity': float(np.mean(extracted)),
                    'std_intensity': float(np.std(extracted))
                })

        if len(extracted_volumes) < self.total_volumes:
            print(f"Warning: Only extracted {len(extracted_volumes)}/{self.total_volumes} "
                  f"volumes after {attempts} attempts (quality filtering)")

        print(f"✓ Extracted {len(extracted_volumes)} volumes")

        return extracted_volumes, metadata_list

    def _count_valid_positions(self, file_shape: Tuple[int, int, int]) -> int:
        """
        Count how many valid extraction positions exist in a file.

        Args:
            file_shape: Shape of the TIF file (D, H, W)

        Returns:
            Number of valid positions
        """
        d, h, w = file_shape
        vol_d, vol_h, vol_w = self.volume_size

        # Count positions ignoring margin (margin is adapted at extraction time)
        valid_d = max(0, d - vol_d + 1)
        valid_h = max(0, h - vol_h + 1)
        valid_w = max(0, w - vol_w + 1)

        return valid_d * valid_h * valid_w

    def _get_random_position(
        self,
        file_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """
        Get a random valid extraction position.

        Args:
            file_shape: Shape of the file (D, H, W)

        Returns:
            Position tuple (d_start, h_start, w_start)
        """
        d, h, w = file_shape
        vol_d, vol_h, vol_w = self.volume_size

        # Adaptive margin: reduce if file dimension can't accommodate full margin
        eff_margin_d = min(self.margin, max(0, (d - vol_d) // 2))
        eff_margin_h = min(self.margin, max(0, (h - vol_h) // 2))
        eff_margin_w = min(self.margin, max(0, (w - vol_w) // 2))

        # Valid ranges with adaptive margin
        d_low, d_high = eff_margin_d, max(eff_margin_d, d - vol_d - eff_margin_d)
        h_low, h_high = eff_margin_h, max(eff_margin_h, h - vol_h - eff_margin_h)
        w_low, w_high = eff_margin_w, max(eff_margin_w, w - vol_w - eff_margin_w)

        d_start = self.rng.randint(d_low, d_high + 1)
        h_start = self.rng.randint(h_low, h_high + 1)
        w_start = self.rng.randint(w_low, w_high + 1)

        return (d_start, h_start, w_start)

    def _extract_at_position(
        self,
        volume: np.ndarray,
        position: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Extract a volume at the specified position.

        Args:
            volume: Full TIF volume
            position: Starting position (d, h, w)

        Returns:
            Extracted volume
        """
        d_start, h_start, w_start = position
        vol_d, vol_h, vol_w = self.volume_size

        extracted = volume[
            d_start:d_start + vol_d,
            h_start:h_start + vol_h,
            w_start:w_start + vol_w
        ].copy()

        return extracted

    def _passes_quality_check(self, volume: np.ndarray) -> bool:
        """
        Check if extracted volume meets quality criteria.

        Args:
            volume: Extracted volume

        Returns:
            True if volume passes quality checks
        """
        # Check mean intensity (not too dark)
        mean_intensity = np.mean(volume)
        if mean_intensity < self.min_mean_intensity:
            return False

        # Check std intensity (not too uniform)
        std_intensity = np.std(volume)
        if std_intensity < self.min_std_intensity:
            return False

        # Check for NaN or Inf
        if not np.all(np.isfinite(volume)):
            return False

        return True
