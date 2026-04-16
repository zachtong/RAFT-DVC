"""
VolumeAnalyzer: estimate memory, time, and recommend configurations.

This module is the user's first contact point.  It reads volume metadata
(shape, dtype) and produces a human-readable report with multiple
strategy recommendations, so the user can make an informed choice about
downsample factor and inference strategy.

Design:
    - Stateless: no volume data is loaded, only shape/dtype.
    - Pure: all estimation formulas live in src/utils/memory.py.
    - The output (AnalysisReport) is a plain dataclass, easy to
      serialize to JSON, print to console, or display in a GUI.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import torch


# ---------------------------------------------------------------------------
# Data classes (output contract)
# ---------------------------------------------------------------------------

@dataclass
class StrategyRecommendation:
    """One candidate inference configuration."""

    downsample_factor: int
    strategy_name: str            # "fast", "balanced", "accurate"
    tile_count: int
    peak_vram_bytes: int
    estimated_seconds: float
    max_displacement_voxels: float
    precision_estimate: float
    feasible: bool                # True if peak VRAM <= available VRAM
    notes: str = ""


@dataclass
class AnalysisReport:
    """Immutable report returned by the analyzer."""

    volume_shape: Tuple[int, int, int]
    dtype: str
    volume_memory_bytes: int
    available_gpu_memory_bytes: int
    recommendations: List[StrategyRecommendation] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for CLI output."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class VolumeAnalyzer:
    """
    Stateless analyzer.  Does NOT modify or store volumes.

    Usage:
        analyzer = VolumeAnalyzer()
        report = analyzer.analyze(
            volume_shape=(800, 800, 800),
            gpu_memory_bytes=12 * 1024**3,
        )
        print(report.summary())
    """

    def analyze(
        self,
        volume_shape: Tuple[int, int, int],
        dtype: str = "float32",
        tile_size: int = 32,
        overlap: int = 8,
        gpu_memory_bytes: Optional[int] = None,
        available_models: Optional[List[Any]] = None,
    ) -> AnalysisReport:
        """
        Produce an AnalysisReport with recommendations for multiple
        downsample factors and strategies.

        Args:
            volume_shape: (D, H, W) of the raw input volume.
            dtype: Data type string (e.g. "float32").
            tile_size: Tile side length for memory estimation.
            overlap: Tile overlap in voxels.
            gpu_memory_bytes: Available GPU VRAM.  Auto-detected if None.
            available_models: List of ModelCard objects (from ModelRegistry).

        Returns:
            AnalysisReport with sorted recommendations.
        """
        raise NotImplementedError
