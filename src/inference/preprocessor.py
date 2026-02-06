"""
Preprocessor: downsample and normalize volumes before inference.

Records all metadata needed by Postprocessor to reverse the operations.

Design:
    - Stateless: each call is independent.
    - Output includes PreprocessMeta, a passive data object that flows
      forward through the pipeline to the Postprocessor.
"""

from dataclasses import dataclass
from typing import Tuple, Dict

import torch


@dataclass
class PreprocessMeta:
    """Everything the Postprocessor needs to undo the preprocessing."""

    original_shape: Tuple[int, int, int]     # (D, H, W) before downsample
    downsample_factor: int
    processed_shape: Tuple[int, int, int]    # (D', H', W') after downsample
    normalization_params: Dict[str, float]   # min, max for denormalization


class Preprocessor:
    """
    Downsample and normalize volumes.

    Usage:
        prep = Preprocessor()
        vol0_p, vol1_p, meta = prep.process_pair(vol0, vol1, downsample_factor=2)
    """

    def process(
        self,
        volume: torch.Tensor,
        downsample_factor: int = 1,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, PreprocessMeta]:
        """
        Process a single volume.

        Args:
            volume: (1, 1, D, H, W)
            downsample_factor: integer >= 1
            normalize: whether to normalize to [0, 1]

        Returns:
            (processed_volume, meta)
        """
        raise NotImplementedError

    def process_pair(
        self,
        vol0: torch.Tensor,
        vol1: torch.Tensor,
        downsample_factor: int = 1,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, PreprocessMeta]:
        """
        Process a volume pair with shared normalization.

        Returns:
            (vol0_processed, vol1_processed, meta)
        """
        raise NotImplementedError
