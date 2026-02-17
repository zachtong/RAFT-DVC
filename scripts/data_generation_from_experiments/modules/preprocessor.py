"""
Preprocessing for extracted volumes.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List


class Preprocessor:
    """Preprocess extracted volumes."""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dict with 'preprocessing' section
        """
        self.config = config['preprocessing']
        self.normalize = self.config.get('normalize', True)
        self.clip_percentile = self.config.get('clip_percentile', [1, 99])
        self.denoise = self.config.get('denoise', False)
        self.denoise_sigma = self.config.get('denoise_sigma', 0.5)

    def process(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a volume.

        Args:
            volume: Input volume

        Returns:
            Processed volume
        """
        volume = volume.copy()

        # 1. Clip extreme values
        if self.clip_percentile:
            low, high = self.clip_percentile
            vmin = np.percentile(volume, low)
            vmax = np.percentile(volume, high)
            volume = np.clip(volume, vmin, vmax)

        # 2. Normalize to [0, 1]
        if self.normalize:
            vmin = np.min(volume)
            vmax = np.max(volume)
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)
            else:
                volume = np.zeros_like(volume)

        # 3. Optional denoising
        if self.denoise:
            volume = gaussian_filter(volume, sigma=self.denoise_sigma)

        return volume

    def process_batch(self, volumes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of volumes.

        Args:
            volumes: List of input volumes

        Returns:
            List of processed volumes
        """
        return [self.process(vol) for vol in volumes]
