# RAFT-DVC Architecture & Project Plan

## Overview

RAFT-DVC is a modular, extensible deep learning framework for 3D Digital Volume
Correlation. It adapts the RAFT optical flow architecture to volumetric data with
DVC-specific optimizations inspired by RAFT-DIC (Pan & Liu, 2024).

**Core design principles:**

1. **Decoupled modules** -- each module has a clear boundary, well-defined input/output
   contract, and can be developed/tested independently.
2. **Extensibility** -- developers can add new models (different pyramid levels, search
   radii, encoder architectures) without modifying existing code.
3. **User-facing flexibility** -- end users choose their own downsample factor, strategy,
   and quality/speed tradeoff at inference time.

---

## 1. Inference Pipeline (High-Level Flow)

```
User Input: vol1, vol2 (any size, e.g. 1000^3)
        |
        v
+------------------+
|  VolumeAnalyzer  |  --> Reports: shape, dtype, memory estimates,
+------------------+      recommended downsample options, estimated
        |                 time per strategy
        v
   User Decision: downsample_factor, strategy
        |
        v
+------------------+
|   Preprocessor   |  --> Downsample, normalize, record metadata
+------------------+      for later upsampling
        |
        v
+------------------+
|   TilingEngine   |  --> Split volume into overlapping tiles
+------------------+      yield (tile_vol1, tile_vol2, tile_coords)
        |
        v
+------------------+
|   ModelRegistry  |  --> Select model(s) by name or by strategy
+------------------+
        |
        v
+---------------------+
|  InferencePipeline  |  --> Orchestrate multi-stage inference
|                     |      Stage 1: coarse model on all tiles
|                     |      Stage 2: fine model on all tiles
+---------------------+      (init_flow from Stage 1)
        |
        v
+--------------------+
|  StitchingEngine   |  --> Merge tile results with weighted blending
+--------------------+
        |
        v
+------------------+
|  Postprocessor   |  --> Upsample flow to original resolution,
+------------------+      scale displacement values
        |
        v
Output: flow field at original resolution (3, D_orig, H_orig, W_orig)
```

---

## 2. Directory Structure (Target)

```
RAFT-DVC/
├── src/
│   ├── __init__.py
│   │
│   ├── core/                        # [EXISTING] Network architecture
│   │   ├── __init__.py
│   │   ├── raft_dvc.py              # RAFTDVC model + RAFTDVCConfig
│   │   ├── extractor.py             # BasicEncoder, ContextEncoder
│   │   ├── corr.py                  # CorrBlock, coords_grid_3d, upflow_3d
│   │   ├── update.py                # ConvGRU3D, BasicUpdateBlock
│   │   └── utils.py                 # warp_volume_3d, flow helpers
│   │
│   ├── inference/                   # [NEW] Inference engine (this plan)
│   │   ├── __init__.py
│   │   ├── analyzer.py              # VolumeAnalyzer
│   │   ├── preprocessor.py          # Preprocessor
│   │   ├── tiling.py                # TilingEngine, StitchingEngine
│   │   ├── model_registry.py        # ModelRegistry
│   │   ├── pipeline.py              # InferencePipeline (orchestrator)
│   │   └── postprocessor.py         # Postprocessor
│   │
│   ├── data/                        # [EXISTING] Dataset & synthetic generation
│   │   ├── __init__.py
│   │   ├── dataset.py               # VolumeDataset, VolumePairDataset
│   │   └── synthetic.py             # SyntheticFlowGenerator
│   │
│   ├── training/                    # [EXISTING] Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py               # Trainer
│   │   └── loss.py                  # SequenceLoss, SmoothLoss, etc.
│   │
│   └── utils/                       # [NEW] Shared utilities
│       ├── __init__.py
│       ├── io.py                    # Volume I/O (npy, tiff, h5, nifti)
│       └── memory.py                # Memory estimation formulas
│
├── configs/
│   ├── training/
│   │   └── default.yaml             # [EXISTING]
│   ├── models/                      # [NEW] Per-model configs
│   │   ├── coarse_p4_r4.yaml        # pyramid=4, radius=4
│   │   ├── fine_p2_r4.yaml          # pyramid=2, radius=4
│   │   └── fine_p2_r6.yaml          # pyramid=2, radius=6
│   └── inference/                   # [NEW] Inference strategy configs
│       ├── auto.yaml
│       ├── fast.yaml
│       └── accurate.yaml
│
├── scripts/
│   ├── train.py                     # [EXISTING] Training entry point
│   ├── train_model_suite.py         # [NEW] Train all models in the suite
│   └── infer.py                     # [REWRITE] New inference entry point
│
├── checkpoints/                     # Trained model weights
│   ├── coarse_p4_r4.pth
│   ├── fine_p2_r4.pth
│   └── fine_p2_r6.pth
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_tiling.py
│   ├── test_pipeline.py
│   └── test_stitching.py
│
└── (other existing dirs: logs/, data/, docs/, gui/, notebooks/, VolRAFT/)
```

---

## 3. Module Specifications

### 3.1 VolumeAnalyzer (`src/inference/analyzer.py`)

**Responsibility:** Read volume metadata, estimate memory/time, present options to user.

```python
# --- Interface ---

@dataclass
class AnalysisReport:
    """Immutable report returned by the analyzer."""
    volume_shape: Tuple[int, int, int]       # (D, H, W)
    dtype: str                                # e.g. "float32"
    volume_memory_bytes: int                  # raw volume size
    available_gpu_memory_bytes: int           # detected free VRAM
    recommendations: List[StrategyRecommendation]

@dataclass
class StrategyRecommendation:
    """One candidate configuration."""
    downsample_factor: int                    # 1, 2, 4, 8
    strategy_name: str                        # "fast", "balanced", "accurate"
    tile_count: int                           # total tiles to process
    peak_vram_bytes: int                      # estimated peak VRAM per tile
    estimated_seconds: float                  # rough wall-clock estimate
    max_displacement_voxels: float            # displacement upper bound
    precision_estimate: float                 # expected sub-voxel precision
    notes: str                                # human-readable explanation


class VolumeAnalyzer:
    """
    Stateless analyzer.  Does NOT modify or store volumes.
    All estimation formulas live in src/utils/memory.py so they can
    be unit-tested independently.
    """

    def analyze(
        self,
        volume_shape: Tuple[int, int, int],
        dtype: str = "float32",
        tile_size: int = 32,
        available_models: List[ModelCard] = ...,
        gpu_memory_bytes: Optional[int] = None,   # auto-detect if None
    ) -> AnalysisReport:
        """
        Pure function: shape in, report out.
        No side effects, no file I/O.
        """
        ...
```

**Key formulas (in `src/utils/memory.py`):**

```python
def estimate_6d_correlation_bytes(tile_size: int) -> int:
    """6D correlation volume: tile^6 * 4 bytes (float32)."""
    return tile_size ** 6 * 4

def estimate_tile_count(
    volume_shape: Tuple[int, int, int],
    tile_size: int,
    overlap: int,
) -> int:
    stride = tile_size - overlap
    return math.prod(
        math.ceil(s / stride) for s in volume_shape
    )

def estimate_max_displacement(
    pyramid_levels: int,
    radius: int,
    downsample_factor: int,
) -> float:
    """Max displacement in original-resolution voxels."""
    max_in_feature_space = radius * (2 ** (pyramid_levels - 1))
    return max_in_feature_space * downsample_factor

def estimate_precision(downsample_factor: int) -> float:
    """Rough sub-voxel precision estimate."""
    return 0.01 * downsample_factor
```

**Design notes:**

- `VolumeAnalyzer` never loads the full volume into memory.
  It only needs the shape (from file header / metadata).
- All formulas are in a separate `memory.py` so they can be tested with
  known expected values.
- The report is a plain dataclass -- easy to serialize to JSON, print to
  console, or display in a future GUI.

---

### 3.2 Preprocessor (`src/inference/preprocessor.py`)

**Responsibility:** Downsample and normalize volumes.  Record metadata needed
for Postprocessor to reverse the operation.

```python
# --- Interface ---

@dataclass
class PreprocessMeta:
    """Everything the Postprocessor needs to undo the preprocessing."""
    original_shape: Tuple[int, int, int]
    downsample_factor: int
    processed_shape: Tuple[int, int, int]
    normalization_params: Dict[str, float]   # min, max, mean, std ...


class Preprocessor:

    def process(
        self,
        volume: torch.Tensor,            # (1, 1, D, H, W)
        downsample_factor: int = 1,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, PreprocessMeta]:
        """
        Returns:
            processed_volume: (1, 1, D', H', W')
            meta: everything needed to reverse
        """
        ...

    def process_pair(
        self,
        vol0: torch.Tensor,
        vol1: torch.Tensor,
        downsample_factor: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, PreprocessMeta]:
        ...
```

**Design notes:**

- Downsample uses `F.avg_pool3d` (anti-aliased) or `F.interpolate`.
- Normalization uses shared min/max across both volumes (consistent with
  existing `RAFTDVC._normalize_volumes`).
- `PreprocessMeta` is a passive data object that flows forward to
  `Postprocessor` -- no hidden state.

---

### 3.3 TilingEngine & StitchingEngine (`src/inference/tiling.py`)

**Responsibility:** Split volumes into overlapping tiles; merge tile-level
results back into a single volume.

```python
# --- Interface ---

@dataclass
class TileInfo:
    """Describes one tile's position in the full volume."""
    index: int                             # sequential tile id
    slices: Tuple[slice, slice, slice]     # (z_slice, y_slice, x_slice)
    padding: Tuple[int, ...]              # if tile extends beyond boundary


class TilingEngine:
    """
    Generates tile coordinates.  Does NOT hold volume data.
    """

    def __init__(
        self,
        tile_size: int = 32,
        overlap: int = 8,
        weight_type: str = "cosine",       # "cosine", "gaussian", "uniform"
    ):
        ...

    def generate_tiles(
        self,
        volume_shape: Tuple[int, int, int],
    ) -> List[TileInfo]:
        """
        Deterministic.  Same shape always produces the same tile list.
        """
        ...

    def extract_tile(
        self,
        volume: torch.Tensor,             # (B, C, D, H, W)
        tile_info: TileInfo,
    ) -> torch.Tensor:
        """Extract one tile from the full volume."""
        ...


class StitchingEngine:
    """
    Merges tile-level flow predictions into a full-volume flow field.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        tile_size: int = 32,
        overlap: int = 8,
        weight_type: str = "cosine",
    ):
        ...

    def add_tile(
        self,
        flow_tile: torch.Tensor,          # (1, 3, td, th, tw)
        tile_info: TileInfo,
    ) -> None:
        """Accumulate one tile's result."""
        ...

    def finalize(self) -> torch.Tensor:
        """
        Returns the blended full-volume flow field.
        Shape: (1, 3, D, H, W)
        """
        ...
```

**Design notes:**

- `TilingEngine` is stateless; `StitchingEngine` is stateful (accumulates).
- Weight functions (cosine, Gaussian, uniform) are interchangeable via
  `weight_type`.  Future developers can add new weight functions without
  touching existing code.
- Boundary handling: tiles at volume edges are zero-padded, and the stitcher
  knows to ignore the padded region.
- **TilingEngine and StitchingEngine share the same configuration** (tile_size,
  overlap, weight_type) to guarantee consistency.  The recommended pattern is
  to construct both from the same config object.

---

### 3.4 ModelRegistry (`src/inference/model_registry.py`)

**Responsibility:** Manage a collection of trained models.  Load by name,
list available models, validate compatibility.

```python
# --- Interface ---

@dataclass
class ModelCard:
    """
    Metadata about one trained model.  Stored alongside the .pth file
    as a YAML sidecar (e.g. fine_p2_r4.yaml).
    """
    name: str                              # "fine_p2_r4"
    pyramid_levels: int                    # 2
    radius: int                            # 4
    feature_dim: int                       # 128
    hidden_dim: int                        # 96
    iters: int                             # 12
    checkpoint_path: str                   # "checkpoints/fine_p2_r4.pth"
    description: str                       # human-readable
    max_displacement_formula: str          # "radius * 2^(levels-1)"
    training_date: Optional[str] = None
    training_dataset: Optional[str] = None
    metrics: Optional[Dict] = None         # validation AEE, etc.


class ModelRegistry:
    """
    Discovers and loads models.  Thread-safe for future GUI use.
    """

    def __init__(self, model_dir: str = "./checkpoints"):
        ...

    def scan(self) -> None:
        """
        Scan model_dir for .pth files with companion .yaml cards.
        Populates internal catalog.
        """
        ...

    def list_models(self) -> List[ModelCard]:
        """Return all discovered ModelCards."""
        ...

    def get_model(
        self,
        name: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[RAFTDVC, ModelCard]:
        """
        Load a model by name.  Caches loaded models to avoid
        redundant disk reads.
        """
        ...

    def register_model(
        self,
        checkpoint_path: str,
        card: ModelCard,
    ) -> None:
        """
        Register a new model (e.g. after training).
        Writes the YAML sidecar file.
        """
        ...
```

**Design notes:**

- **Convention over configuration:** each model is a pair of files
  (`checkpoints/foo.pth` + `configs/models/foo.yaml`).  The registry
  auto-discovers them.
- The registry is **the only place** that calls `RAFTDVC.load_checkpoint`.
  All other modules receive an already-loaded `nn.Module`.
- Future developers add a new model by:
  1. Training with any pyramid_levels/radius combination.
  2. Saving `.pth` + writing a `.yaml` card.
  3. Done.  No code changes.

---

### 3.5 InferencePipeline (`src/inference/pipeline.py`)

**Responsibility:** Orchestrate the full inference flow.  This is the only
module that knows about stages (coarse -> fine).

```python
# --- Interface ---

@dataclass
class StageConfig:
    """Configuration for one inference stage."""
    model_name: str                        # key into ModelRegistry
    internal_downsample: int = 1           # additional downsample within this stage
    tile_size: int = 32
    overlap: int = 8
    iters: int = 12
    weight_type: str = "cosine"


@dataclass
class PipelineConfig:
    """
    Full inference configuration.
    Loaded from YAML or constructed programmatically.
    """
    stages: List[StageConfig]
    # Example "accurate":
    #   stages:
    #     - model_name: coarse_p4_r4
    #       internal_downsample: 4
    #       tile_size: 32
    #     - model_name: fine_p2_r4
    #       internal_downsample: 1
    #       tile_size: 32


class InferencePipeline:
    """
    Stateless orchestrator.  Composes all other modules.
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
        vol0: torch.Tensor,               # (1, 1, D, H, W)  preprocessed
        vol1: torch.Tensor,
        config: PipelineConfig,
        progress_callback: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Execute multi-stage inference.

        Returns:
            flow: (1, 3, D, H, W) at the resolution of the input volumes
        """
        flow = None  # no initial flow

        for stage_idx, stage in enumerate(config.stages):
            # --- optional internal downsample for this stage ---
            if stage.internal_downsample > 1:
                vol0_stage = F.avg_pool3d(vol0, stage.internal_downsample)
                vol1_stage = F.avg_pool3d(vol1, stage.internal_downsample)
            else:
                vol0_stage = vol0
                vol1_stage = vol1

            stage_shape = vol0_stage.shape[-3:]

            # --- prepare init flow for this stage ---
            if flow is not None:
                flow_init = F.interpolate(
                    flow, size=feature_size(stage_shape), ...
                )
                # scale values by resolution ratio
            else:
                flow_init = None

            # --- load model ---
            model, card = self.registry.get_model(stage.model_name, self.device)
            model.eval()

            # --- tile, infer, stitch ---
            tiler = TilingEngine(stage.tile_size, stage.overlap)
            stitcher = StitchingEngine(stage_shape, stage.tile_size, stage.overlap)

            tiles = tiler.generate_tiles(stage_shape)
            for tile_info in tiles:
                tile_v0 = tiler.extract_tile(vol0_stage, tile_info)
                tile_v1 = tiler.extract_tile(vol1_stage, tile_info)
                tile_init = ...  # extract init flow for this tile

                with torch.no_grad():
                    _, flow_tile = model(
                        tile_v0.to(self.device),
                        tile_v1.to(self.device),
                        iters=stage.iters,
                        flow_init=tile_init,
                        test_mode=True,
                    )

                stitcher.add_tile(flow_tile.cpu(), tile_info)

                if progress_callback:
                    progress_callback(stage_idx, tile_info.index, len(tiles))

            flow = stitcher.finalize()

            # --- upsample if this stage was internally downsampled ---
            if stage.internal_downsample > 1:
                flow = F.interpolate(flow, size=vol0.shape[-3:], ...)
                flow *= stage.internal_downsample

        return flow
```

**Design notes:**

- `InferencePipeline` is **the integration point**.  It composes
  `TilingEngine`, `StitchingEngine`, `ModelRegistry`, and the RAFT-DVC model.
- Each stage is independent: different model, different downsample, different
  tile size.  Adding a third stage requires zero code changes -- just add
  another `StageConfig` to the YAML.
- `progress_callback` allows both CLI progress bars and GUI progress
  indicators without the pipeline knowing about either.
- The pipeline does NOT handle user interaction, file I/O, or visualization.
  Those are the caller's responsibility.

---

### 3.6 Postprocessor (`src/inference/postprocessor.py`)

**Responsibility:** Map the flow field from processed resolution back to
original resolution.

```python
# --- Interface ---

class Postprocessor:

    def restore(
        self,
        flow: torch.Tensor,               # (1, 3, D', H', W')
        meta: PreprocessMeta,
    ) -> torch.Tensor:
        """
        Upsample and rescale flow to original volume resolution.

        Returns:
            flow_original: (1, 3, D_orig, H_orig, W_orig)
        """
        ...
```

**Design notes:**

- This module reverses what `Preprocessor` did.
- Uses the `PreprocessMeta` produced earlier -- no hidden coupling.
- Flow values are scaled by `downsample_factor` (a 2-voxel displacement at
  1/2 resolution = 4-voxel displacement at original resolution).

---

## 4. Configuration Schema

### 4.1 Model Config (`configs/models/fine_p2_r4.yaml`)

```yaml
# Model identity
name: fine_p2_r4
description: >
  Fine-stage model with 2-level correlation pyramid and radius 4.
  Optimized for sub-voxel precision in DVC measurements.
  Recommended for the refinement stage.

# Architecture parameters (must match training)
architecture:
  input_channels: 1
  feature_dim: 128
  hidden_dim: 96
  context_dim: 64
  encoder_norm: instance
  corr_levels: 2          # <-- 2-level pyramid
  corr_radius: 4          # <-- search radius
  use_sep_conv: true
  iters: 12

# Checkpoint
checkpoint: checkpoints/fine_p2_r4.pth

# Metadata (informational, not used by code)
metadata:
  training_date: null
  training_dataset: null
  training_epochs: 120
  validation_aee: null
  max_displacement: "radius * 2^(levels-1) = 4 * 2 = 8 voxels"
  recommended_role: fine   # "coarse" or "fine"
```

### 4.2 Inference Strategy Config (`configs/inference/accurate.yaml`)

```yaml
name: accurate
description: >
  Two-stage strategy: coarse localization + fine refinement.
  Best precision, slower speed.

stages:
  - name: coarse
    model: coarse_p4_r4
    internal_downsample: 4
    tile_size: 32
    overlap: 8
    iters: 12
    weight_type: cosine

  - name: fine
    model: fine_p2_r4
    internal_downsample: 1
    tile_size: 32
    overlap: 8
    iters: 12
    weight_type: cosine
```

### 4.3 Example: Adding a New Model

A developer trains a model with `pyramid_levels=3, radius=6`:

1. Train:
   ```bash
   python scripts/train.py --corr_levels 3 --corr_radius 6 \
       --output checkpoints/medium_p3_r6.pth
   ```

2. Write `configs/models/medium_p3_r6.yaml`:
   ```yaml
   name: medium_p3_r6
   architecture:
     corr_levels: 3
     corr_radius: 6
     # ... (copy other fields from any existing model yaml)
   checkpoint: checkpoints/medium_p3_r6.pth
   ```

3. Use in a strategy:
   ```yaml
   # configs/inference/custom.yaml
   stages:
     - model: medium_p3_r6
       internal_downsample: 2
       tile_size: 32
   ```

**No Python code changes required.**

---

## 5. Model Suite & Training Plan

### 5.1 Recommended Model Suite

| Name | Pyramid | Radius | Role | Max Disp (feature space) | Priority |
|------|---------|--------|------|--------------------------|----------|
| `coarse_p4_r4` | 4 | 4 | Coarse localization | +/-32 voxels | HIGH |
| `fine_p2_r4` | 2 | 4 | Standard refinement | +/-8 voxels | HIGH |
| `fine_p2_r6` | 2 | 6 | Large-disp refinement | +/-12 voxels | MEDIUM |

### 5.2 Training Script (`scripts/train_model_suite.py`)

```python
"""
Train all models in the suite.

Usage:
    python scripts/train_model_suite.py --data_dir ./data/train
    python scripts/train_model_suite.py --only fine_p2_r4
"""

SUITE = {
    'coarse_p4_r4': RAFTDVCConfig(corr_levels=4, corr_radius=4),
    'fine_p2_r4':   RAFTDVCConfig(corr_levels=2, corr_radius=4),
    'fine_p2_r6':   RAFTDVCConfig(corr_levels=2, corr_radius=6),
}

# All models share:
# - The same training dataset
# - The same loss function (SequenceLoss with gamma adjusted per model)
# - The same training schedule (AdamW, OneCycleLR)
#
# Differences:
# - Architecture config (pyramid levels, radius)
# - Batch size may vary (larger radius → more VRAM → smaller batch)
```

### 5.3 GRU Input Dimension

**Critical:** the GRU input dimension depends on `pyramid_levels` and
`radius`.  Models with different (levels, radius) have **different
architectures** and cannot share weights for the update block.

```python
gru_corr_input = pyramid_levels * (2 * radius + 1) ** 3

# Examples:
# coarse_p4_r4: 4 * 9^3  = 2916
# fine_p2_r4:   2 * 9^3  = 1458
# fine_p2_r6:   2 * 13^3 = 4394
```

The feature encoder and context encoder have identical architecture across
all models, so weight transfer / pretraining is possible for those
sub-modules.

---

## 6. Memory Budget (Reference Table)

All values assume `float32`, `batch_size=1`, `tile_size=32`.

| Component | Formula | 32^3 tile |
|-----------|---------|-----------|
| 6D Correlation Volume | `tile^6 * 4` | **4.29 GB** |
| Correlation Pyramid L1 | `tile^3 * (tile/2)^3 * 4` | 0.54 GB |
| Feature maps (x2) | `2 * C * tile^3 * 4` (C=128) | 33.6 MB |
| Hidden state | `hidden * tile^3 * 4` (hidden=96) | 12.6 MB |
| Context features | `ctx * tile^3 * 4` (ctx=64) | 8.4 MB |
| Flow field | `3 * tile^3 * 4` | 0.4 MB |
| **Total (2 pyramid)** | | **~4.9 GB** |
| **Total (4 pyramid)** | | **~5.0 GB** |

**Note:** These are theoretical minimums for tensor storage.  Actual PyTorch
VRAM usage includes CUDA context (~300-500 MB), intermediate activations,
and memory fragmentation.  Budget ~6 GB per tile on a 12 GB GPU.

---

## 7. User-Facing Downsample Decision Table

Displayed by `VolumeAnalyzer` for a 1000^3 volume, tile=32, overlap=8:

| Downsample | Processed Size | Tile Count | Est. Time | Max Disp (orig) | Precision |
|------------|---------------|------------|-----------|-----------------|-----------|
| 1x | 1000^3 | ~74,000 | ~2 hr | +/-8 vx | 0.01 vx |
| 2x | 500^3 | ~9,300 | ~15 min | +/-16 vx | 0.02 vx |
| 4x | 250^3 | ~1,200 | ~2 min | +/-32 vx | 0.04 vx |
| 8x | 125^3 | ~150 | ~15 sec | +/-64 vx | 0.08 vx |

(These are for the fine stage alone; a coarse+fine strategy has additional
overhead from Stage 1, typically <5% of total time.)

---

## 8. Module Dependency Graph

```
src/utils/memory.py          (standalone, no imports from src/)
src/utils/io.py              (standalone)
      |
      v
src/core/*                   (existing, standalone)
      |
      v
src/inference/analyzer.py    --> src/utils/memory
src/inference/preprocessor.py  --> (torch only)
src/inference/tiling.py      --> (torch only)
src/inference/model_registry.py --> src/core/raft_dvc (for loading)
src/inference/postprocessor.py  --> (torch only)
      |
      v
src/inference/pipeline.py    --> all of the above
      |
      v
scripts/infer.py             --> src/inference/pipeline
                                  src/utils/io
```

**Key rule:** no circular dependencies.  Each module only imports from modules
above it in this graph.

---

## 9. Extension Points

### 9.1 New Model Architectures

```
Developer action:
  1. Modify src/core/ to support new architecture variant
  2. Train model, save checkpoint
  3. Write YAML model card
  4. (Optional) Write new inference strategy YAML

Files touched: src/core/*, configs/models/*, configs/inference/*
Files NOT touched: src/inference/* (pipeline is architecture-agnostic)
```

### 9.2 New Weight Functions for Stitching

```
Developer action:
  1. Add function to src/inference/tiling.py
  2. Register name in weight_type enum

Files touched: src/inference/tiling.py only
```

### 9.3 New Preprocessing Methods

```
Developer action:
  1. Add method to Preprocessor (e.g. anisotropic downsampling)
  2. Update PreprocessMeta to carry new metadata
  3. Update Postprocessor to handle new metadata

Files touched: src/inference/preprocessor.py, src/inference/postprocessor.py
```

### 9.4 GUI Integration

```
The InferencePipeline accepts a progress_callback.
GUI code (in gui/) calls pipeline.run(..., progress_callback=gui_update).
No changes to src/inference/*.
```

### 9.5 New File Formats

```
Developer action:
  1. Add reader/writer to src/utils/io.py

Files touched: src/utils/io.py only
```

---

## 10. Implementation Priority

### Phase 1 -- Core Inference (current focus)

- [ ] `src/utils/memory.py` -- memory estimation formulas
- [ ] `src/utils/io.py` -- volume I/O helpers
- [ ] `src/inference/analyzer.py` -- VolumeAnalyzer
- [ ] `src/inference/preprocessor.py` -- Preprocessor
- [ ] `src/inference/tiling.py` -- TilingEngine + StitchingEngine
- [ ] `src/inference/model_registry.py` -- ModelRegistry
- [ ] `src/inference/postprocessor.py` -- Postprocessor
- [ ] `src/inference/pipeline.py` -- InferencePipeline
- [ ] `scripts/infer.py` -- rewrite CLI entry point
- [ ] `configs/models/*.yaml` -- model card templates
- [ ] `configs/inference/*.yaml` -- strategy templates

### Phase 2 -- Training Suite

- [ ] `scripts/train_model_suite.py` -- train coarse + fine models
- [ ] Validate dual-model strategy on synthetic data
- [ ] Ablation: tile_size, overlap, pyramid levels, radius

### Phase 3 -- Polish & Open-Source

- [ ] Unit tests for every module
- [ ] CLI documentation
- [ ] GUI integration via progress_callback
- [ ] PyPI packaging

---

## 11. Interface Contracts (Summary)

| Module | Input | Output | Stateful? |
|--------|-------|--------|-----------|
| `VolumeAnalyzer` | shape, dtype | `AnalysisReport` | No |
| `Preprocessor` | volume, downsample_factor | volume, `PreprocessMeta` | No |
| `TilingEngine` | volume_shape | `List[TileInfo]` | No |
| `StitchingEngine` | flow tiles + TileInfo | merged flow | Yes (accumulator) |
| `ModelRegistry` | model_name | (model, ModelCard) | Yes (cache) |
| `InferencePipeline` | vol0, vol1, PipelineConfig | flow | No |
| `Postprocessor` | flow, PreprocessMeta | flow at original res | No |

---

## 12. Example End-to-End Usage

### CLI

```bash
# Step 1: Analyze and get recommendations
python scripts/infer.py analyze \
    --vol0 data/ref.nii.gz \
    --vol1 data/def.nii.gz

# Output:
# Volume shape: 800 x 800 x 800
# Recommended configurations:
#   [1] downsample=1, strategy=accurate  | ~1.5 hr  | precision 0.01 vx
#   [2] downsample=2, strategy=accurate  | ~12 min  | precision 0.02 vx
#   [3] downsample=4, strategy=fast      | ~1 min   | precision 0.04 vx

# Step 2: Run with chosen configuration
python scripts/infer.py run \
    --vol0 data/ref.nii.gz \
    --vol1 data/def.nii.gz \
    --downsample 2 \
    --strategy accurate \
    --output results/flow.npy
```

### Python API

```python
from src.inference import (
    VolumeAnalyzer, Preprocessor, Postprocessor,
    InferencePipeline, ModelRegistry,
)
from src.utils.io import load_volume

# Load
vol0 = load_volume("data/ref.nii.gz")
vol1 = load_volume("data/def.nii.gz")

# Analyze
analyzer = VolumeAnalyzer()
report = analyzer.analyze(vol0.shape[-3:])
for rec in report.recommendations:
    print(rec)

# Preprocess
preprocessor = Preprocessor()
vol0_p, vol1_p, meta = preprocessor.process_pair(vol0, vol1, downsample_factor=2)

# Infer
registry = ModelRegistry("./checkpoints")
registry.scan()

pipeline = InferencePipeline(registry, device=torch.device("cuda"))
config = PipelineConfig.from_yaml("configs/inference/accurate.yaml")
flow = pipeline.run(vol0_p, vol1_p, config)

# Postprocess
postprocessor = Postprocessor()
flow_original = postprocessor.restore(flow, meta)

# Save
save_flow("results/flow.npy", flow_original)
```

---

*Document version: 0.1 (draft)*
*Last updated: 2026-01-28*
