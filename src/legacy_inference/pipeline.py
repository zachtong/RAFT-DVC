"""
InferencePipeline: multi-stage inference orchestrator.

This is the integration point that composes all inference modules:
    VolumeAnalyzer -> Preprocessor -> TilingEngine -> Model -> StitchingEngine -> Postprocessor

Design:
    - Stateless orchestrator: does not store volumes or results.
    - Each stage is configured independently (model, downsample, tile_size).
    - Adding a third stage requires zero code changes -- just add another
      StageConfig to the PipelineConfig.
    - progress_callback allows both CLI progress bars and GUI indicators.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import torch
import torch.nn.functional as F

from .model_registry import ModelRegistry
from .tiling import TilingEngine, StitchingEngine
from .preprocessor import PreprocessMeta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """Configuration for one inference stage."""

    model_name: str                    # key into ModelRegistry
    internal_downsample: int = 1       # additional downsample within this stage
    tile_size: int = 32
    overlap: int = 8
    iters: int = 12
    weight_type: str = "cosine"


@dataclass
class PipelineConfig:
    """
    Full inference configuration.

    Can be loaded from YAML or constructed in Python.
    """

    stages: List[StageConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load from a YAML strategy file (e.g. configs/inference/accurate.yaml)."""
        raise NotImplementedError

    @classmethod
    def single_stage(
        cls,
        model_name: str = "fine_p2_r4",
        tile_size: int = 32,
        overlap: int = 8,
    ) -> "PipelineConfig":
        """Convenience: create a single-stage config."""
        return cls(stages=[StageConfig(
            model_name=model_name,
            tile_size=tile_size,
            overlap=overlap,
        )])

    @classmethod
    def dual_stage(
        cls,
        coarse_model: str = "coarse_p4_r4",
        fine_model: str = "fine_p2_r4",
        coarse_downsample: int = 4,
        tile_size: int = 32,
        overlap: int = 8,
    ) -> "PipelineConfig":
        """Convenience: create a coarse+fine two-stage config."""
        return cls(stages=[
            StageConfig(
                model_name=coarse_model,
                internal_downsample=coarse_downsample,
                tile_size=tile_size,
                overlap=overlap,
            ),
            StageConfig(
                model_name=fine_model,
                internal_downsample=1,
                tile_size=tile_size,
                overlap=overlap,
            ),
        ])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """
    Multi-stage inference orchestrator.

    Usage:
        registry = ModelRegistry("./checkpoints")
        registry.scan()

        pipeline = InferencePipeline(registry, device=torch.device("cuda"))
        config = PipelineConfig.dual_stage()
        flow = pipeline.run(vol0, vol1, config)
    """

    def __init__(
        self,
        registry: ModelRegistry,
        device: torch.device = torch.device("cuda"),
    ):
        self.registry = registry
        self.device = device

    def run(
        self,
        vol0: torch.Tensor,
        vol1: torch.Tensor,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Execute multi-stage inference.

        Args:
            vol0: Reference volume (1, 1, D, H, W) -- preprocessed.
            vol1: Deformed volume (1, 1, D, H, W) -- preprocessed.
            config: PipelineConfig with stage definitions.
            progress_callback: Optional function(stage_idx, tile_idx, total_tiles).

        Returns:
            flow: (1, 3, D, H, W) displacement field at input resolution.
        """
        raise NotImplementedError

    def _run_stage(
        self,
        vol0: torch.Tensor,
        vol1: torch.Tensor,
        stage: StageConfig,
        init_flow: Optional[torch.Tensor],
        stage_idx: int,
        progress_callback: Optional[Callable],
    ) -> torch.Tensor:
        """
        Run a single inference stage with tiling.

        Args:
            vol0, vol1: Volumes for this stage (may be internally downsampled).
            stage: Stage configuration.
            init_flow: Initial flow from previous stage (or None).
            stage_idx: Index for progress reporting.
            progress_callback: Optional progress function.

        Returns:
            flow: (1, 3, D_stage, H_stage, W_stage) at stage resolution.
        """
        raise NotImplementedError
