# Inference engine for RAFT-DVC
#
# Public API:
#   VolumeAnalyzer  -- estimate memory/time, recommend configurations
#   Preprocessor    -- downsample and normalize volumes
#   Postprocessor   -- restore flow to original resolution
#   TilingEngine    -- split volume into overlapping tiles
#   StitchingEngine -- merge tile results with weighted blending
#   ModelRegistry   -- discover and load trained models
#   InferencePipeline -- orchestrate multi-stage inference

from .analyzer import VolumeAnalyzer, AnalysisReport, StrategyRecommendation
from .preprocessor import Preprocessor, PreprocessMeta
from .postprocessor import Postprocessor
from .tiling import TilingEngine, StitchingEngine, TileInfo
from .model_registry import ModelRegistry, ModelCard
from .pipeline import InferencePipeline, PipelineConfig, StageConfig

__all__ = [
    "VolumeAnalyzer",
    "AnalysisReport",
    "StrategyRecommendation",
    "Preprocessor",
    "PreprocessMeta",
    "Postprocessor",
    "TilingEngine",
    "StitchingEngine",
    "TileInfo",
    "ModelRegistry",
    "ModelCard",
    "InferencePipeline",
    "PipelineConfig",
    "StageConfig",
]
