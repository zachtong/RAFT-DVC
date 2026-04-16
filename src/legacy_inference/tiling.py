"""
TilingEngine & StitchingEngine: split volumes into tiles and merge results.

Design:
    - TilingEngine is stateless: same shape always produces the same tile list.
    - StitchingEngine is stateful: accumulates tile results and produces
      the final merged flow field on finalize().
    - Both share configuration (tile_size, overlap, weight_type) to
      guarantee consistency.  Construct both from the same config values.

Weight functions:
    - "cosine": 3D cosine (Tukey) window -- smooth falloff at edges.
    - "gaussian": 3D Gaussian centered in the tile.
    - "uniform": flat weight=1 everywhere (no blending).
    Developers can add new weight functions by extending
    _create_weight_volume().
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TileInfo:
    """Describes one tile's position in the full volume."""

    index: int                                # sequential tile id
    slices: Tuple[slice, slice, slice]        # (z_slice, y_slice, x_slice)
    padding: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0))
    # ((z_pad_before, z_pad_after), (y_...), (x_...))
    # Non-zero when tile extends beyond volume boundary


# ---------------------------------------------------------------------------
# TilingEngine
# ---------------------------------------------------------------------------

class TilingEngine:
    """
    Generate tile coordinates for a volume.

    Usage:
        tiler = TilingEngine(tile_size=32, overlap=8)
        tiles = tiler.generate_tiles(volume_shape=(500, 500, 500))
        for ti in tiles:
            patch = tiler.extract_tile(volume, ti)
    """

    def __init__(
        self,
        tile_size: int = 32,
        overlap: int = 8,
    ):
        self.tile_size = tile_size
        self.overlap = overlap

    def generate_tiles(
        self,
        volume_shape: Tuple[int, int, int],
    ) -> List[TileInfo]:
        """
        Deterministic tile generation.

        Args:
            volume_shape: (D, H, W)

        Returns:
            Ordered list of TileInfo.
        """
        raise NotImplementedError

    def extract_tile(
        self,
        volume: torch.Tensor,
        tile_info: TileInfo,
    ) -> torch.Tensor:
        """
        Extract one tile from the full volume, with zero-padding
        if the tile extends beyond the boundary.

        Args:
            volume: (B, C, D, H, W)
            tile_info: TileInfo from generate_tiles()

        Returns:
            Tile tensor of shape (B, C, tile_size, tile_size, tile_size)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StitchingEngine
# ---------------------------------------------------------------------------

class StitchingEngine:
    """
    Merge tile-level flow predictions into a full-volume flow field.

    Usage:
        stitcher = StitchingEngine(volume_shape, tile_size=32, overlap=8)
        for tile_info in tiles:
            flow_tile = model(...)
            stitcher.add_tile(flow_tile, tile_info)
        flow_full = stitcher.finalize()
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        tile_size: int = 32,
        overlap: int = 8,
        weight_type: str = "cosine",
    ):
        self.volume_shape = volume_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.weight_type = weight_type

        # Internal accumulators (allocated on first add_tile)
        self._flow_accum: Optional[torch.Tensor] = None
        self._weight_accum: Optional[torch.Tensor] = None

    def add_tile(
        self,
        flow_tile: torch.Tensor,
        tile_info: TileInfo,
    ) -> None:
        """
        Accumulate one tile's flow prediction with weighting.

        Args:
            flow_tile: (1, 3, td, th, tw) -- may include padding region
            tile_info: Corresponding TileInfo
        """
        raise NotImplementedError

    def finalize(self) -> torch.Tensor:
        """
        Normalize accumulated flow by accumulated weights.

        Returns:
            (1, 3, D, H, W) -- full-volume flow field
        """
        raise NotImplementedError

    def _create_weight_volume(self) -> torch.Tensor:
        """
        Create a weight volume for blending.

        Returns:
            (1, 1, tile_size, tile_size, tile_size) weight tensor
        """
        raise NotImplementedError
