# Experimental Dataset Generation from TIF Images

This document describes how to generate datasets from experimental TIF images with known ground truth deformations.

## Overview

The experimental data generation pipeline:
1. **Loads** real 3D TIF images from experiments
2. **Extracts** multiple 128³ volumes from large images
3. **Applies** controlled deformations (affine, B-spline, etc.)
4. **Generates** ground truth flow fields
5. **Saves** in standard format compatible with existing inference pipeline

## Directory Structure

```
data_tif/                                  # Raw experimental TIF files
└── JinYang_confocal_beads_indentation/
    ├── image_001.tif
    ├── scan_xyz.tif
    └── ...

configs/data_generation_from_experiments/  # Generation configs
└── JinYang_confocal_beads_indentation.yaml

data/                                      # Generated datasets (same level as synthetic)
└── JinYang_confocal_beads_indentation/
    ├── train/
    │   ├── vol0/
    │   ├── vol1/
    │   ├── flow/
    │   ├── metadata/
    │   └── visualizations/
    ├── val/
    └── test/
```

## Quick Start

### 1. Prepare Raw TIF Files

Place your experimental TIF files in `data_tif/experiment_name/`:

```bash
mkdir -p data_tif/JinYang_confocal_beads_indentation
# Copy your TIF files there
```

**Requirements:**
- File format: `.tif` or `.tiff`
- Dimensionality: 2D or 3D
- Minimum size: 128×128×128 voxels
- File names: Any naming convention (no strict ordering required)

### 2. Create Configuration File

Copy and modify the example config:

```bash
cp configs/data_generation_from_experiments/JinYang_confocal_beads_indentation.yaml \
   configs/data_generation_from_experiments/your_experiment.yaml
```

Key parameters to adjust:

```yaml
dataset_name: "your_experiment_name"

input:
  raw_data_dir: "data_tif/your_experiment_name"

extraction:
  total_volumes: 60              # How many 128³ volumes to extract

splits:
  train:
    num_samples: 0               # Usually 0 for testing only
  val:
    num_samples: 0
  test:
    num_samples: 50              # Number of test samples

deformations:
  affine:
    enabled: true
    probability: 0.5
    translation_range: [-5, 5]   # Adjust based on expected motion
    rotation_range: [-10, 10]

  bspline:
    enabled: true
    probability: 0.5
    max_displacement: 6.0        # Adjust based on expected deformation
```

### 3. Generate Dataset

Use `run.bat`:

```
run.bat
→ 3. Dataset Generation
→ 2. Generate Dataset from Experimental TIF Images
→ Select your config
→ Confirm and start
```

Or via command line:

```bash
python scripts/data_generation_from_experiments/generate_from_tif.py --config configs/data_generation_from_experiments/your_experiment.yaml
```

### 4. Use Generated Dataset

The generated dataset is automatically compatible with inference:

```
run.bat
→ 2. Inference / Testing
→ Select dataset: your_experiment_name
→ Select checkpoint and run
```

## Configuration Parameters

### Input Settings

```yaml
input:
  raw_data_dir: "data_tif/experiment_name"
  file_pattern: "*.tif"                    # Glob pattern for TIF files
  file_pattern_alternative: "*.tiff"       # Alternative extension
  min_volume_size: [128, 128, 128]         # Skip smaller files
```

### Extraction Strategy

```yaml
extraction:
  volume_size: [128, 128, 128]
  strategy: "random"                       # Random extraction across all files
  total_volumes: 60                        # Total volumes to extract
  margin: 10                               # Distance from edges (voxels)

  # Quality filtering
  min_mean_intensity: 0.05                 # Skip too-dark regions
  min_std_intensity: 0.02                  # Skip too-uniform regions
```

**Strategy options:**
- `random`: Randomly sample positions across all valid files (recommended)

### Split Allocation

```yaml
splits:
  train:
    num_samples: 0                         # Number of training samples
  val:
    num_samples: 0                         # Number of validation samples
  test:
    num_samples: 50                        # Number of test samples
```

Total samples must not exceed `extraction.total_volumes`.

### Deformation Types

#### Affine Transformations
```yaml
affine:
  enabled: true
  probability: 0.5                         # 50% of samples
  translation_range: [-5, 5]               # voxels
  rotation_range: [-10, 10]                # degrees
  scale_range: [0.95, 1.05]                # ratio
  shear_range: [-0.03, 0.03]               # ratio
```

#### B-spline Deformations
```yaml
bspline:
  enabled: true
  probability: 0.5
  control_point_spacing: 16                # Coarser = smoother
  max_displacement: 6.0                    # Maximum displacement (voxels)
```

#### Localized Deformations
```yaml
localized:
  enabled: false                           # Usually disabled (too strong)
  probability: 0.2
  num_centers: [1, 3]                      # Number of deformation centers
  radius_range: [15, 25]                   # Radius of deformation (voxels)
  displacement_range: [5, 10]              # Displacement magnitude
```

### Preprocessing

```yaml
preprocessing:
  normalize: true                          # Normalize to [0, 1]
  clip_percentile: [1, 99]                 # Clip extreme values
  denoise: false                           # Optional Gaussian denoising
  denoise_sigma: 0.5                       # Sigma for Gaussian filter
```

### Output Settings

```yaml
output:
  base_dir: "data/experiment_name"         # Output directory
  save_metadata: true                      # Save JSON metadata files
  save_visualizations: true                # Save PNG visualizations
  num_visualize: 5                         # Visualize first N samples per split
```

## Output Structure

The generated dataset follows the exact same structure as synthetic datasets:

```
data/experiment_name/
├── train/
│   ├── vol0/
│   │   ├── sample_0000.npy              # Original volumes
│   │   └── ...
│   ├── vol1/
│   │   ├── sample_0000.npy              # Deformed volumes
│   │   └── ...
│   ├── flow/
│   │   ├── sample_0000.npy              # Ground truth flow (3, D, H, W)
│   │   └── ...
│   ├── metadata/
│   │   ├── sample_0000.json             # Per-sample metadata
│   │   └── ...
│   └── visualizations/
│       ├── sample_0000.png              # Visual inspection
│       └── ...
├── val/                                  # Same structure
├── test/                                 # Same structure
├── generation_summary.yaml               # Generation summary
└── generation_config.yaml                # Config used for generation
```

### Metadata Format

Each `sample_XXXX.json` contains:

```json
{
  "sample_idx": 0,
  "split": "test",
  "source": "experimental",
  "extraction": {
    "source_file": "image_001.tif",
    "position": [100, 200, 50],
    "mean_intensity": 0.35,
    "std_intensity": 0.12
  },
  "deform_type": "affine",
  "deform_params": {
    "translation": [2.3, -1.5, 0.8],
    "rotation_deg": [5.2, -3.1, 1.4],
    "scale": [1.02, 0.98, 1.01],
    ...
  },
  "flow_stats": {
    "magnitude_mean": 5.85,
    "magnitude_std": 1.86,
    "magnitude_max": 11.97,
    ...
  },
  "volume_stats": {
    "vol0_mean": 0.42,
    "vol0_std": 0.18,
    ...
  }
}
```

## Tips and Best Practices

### 1. TIF File Preparation

- **Size**: Larger files allow more volume extraction
- **Quality**: High SNR data works better
- **Preprocessing**: Consider denoising very noisy data
- **Format**: 16-bit or 32-bit TIFF recommended

### 2. Extraction Strategy

- **total_volumes**: Extract more than needed, quality filtering may reject some
- **margin**: Increase if seeing edge artifacts after warping (try 15-20)
- **Quality filters**: Adjust based on your data's intensity characteristics

### 3. Deformation Parameters

- **Start small**: Begin with small deformations (translation < 5, rotation < 10)
- **Test first**: Generate a few samples to check if deformations look realistic
- **Match physics**: Use deformation types that match your experimental conditions
  - Affine: Rigid motion, uniform compression/tension
  - B-spline: Elastic deformations, local strain

### 4. Split Allocation

For testing model generalization:
- Use `test` split only (train=0, val=0, test=50)
- Models trained on synthetic data are tested on real data

For fine-tuning on real data:
- Allocate some to train/val (e.g., train=40, val=5, test=5)

### 5. Troubleshooting

**Issue: "No valid extraction positions"**
- Files too small or margin too large
- Reduce margin or use larger TIF files

**Issue: "Only extracted X/Y volumes after quality filtering"**
- Images too dark or too uniform
- Adjust `min_mean_intensity` and `min_std_intensity`
- Check TIF file quality

**Issue: "Raw data directory not found"**
- Check `raw_data_dir` path in config
- Ensure TIF files are in correct location
- Use absolute or correct relative paths

## Examples

### Example 1: Testing Model on Real Data

```yaml
# configs/data_generation_from_experiments/test_real_indentation.yaml
dataset_name: "real_indentation_test"

input:
  raw_data_dir: "data_tif/real_indentation"

extraction:
  total_volumes: 60

splits:
  test:
    num_samples: 50              # Only test samples

deformations:
  affine:
    enabled: true
    probability: 0.6
    translation_range: [-3, 3]   # Small deformations
    rotation_range: [-5, 5]
  bspline:
    enabled: true
    probability: 0.4
    max_displacement: 4.0
```

### Example 2: Fine-tuning Dataset

```yaml
# configs/data_generation_from_experiments/finetune_real_data.yaml
dataset_name: "real_data_finetune"

input:
  raw_data_dir: "data_tif/real_data"

extraction:
  total_volumes: 200             # Extract more for training

splits:
  train:
    num_samples: 150             # Training samples
  val:
    num_samples: 25              # Validation
  test:
    num_samples: 25              # Testing

deformations:
  # Use full range of deformations
  affine:
    enabled: true
    probability: 0.4
  bspline:
    enabled: true
    probability: 0.4
  localized:
    enabled: true
    probability: 0.2             # Include challenging cases
```

## Implementation Details

### Modules

- **TifLoader**: Load TIF files, handle various formats
- **VolumeExtractor**: Extract 128³ volumes with quality filtering
- **Preprocessor**: Normalize, clip, optional denoising
- **DatasetBuilder**: Apply deformations, compute flow fields, save data

### Deformation Application

For real data, deformations are applied directly to intensity volumes:
1. Vol0 (original) is extracted from TIF
2. Deformation field is generated (same as synthetic data)
3. Vol1 is created by warping Vol0 using the flow field
4. Ground truth flow is the generated deformation field

This differs from synthetic data where beads are warped and re-rendered.

### Quality Control

Same QualityChecker as synthetic data:
- Flow field statistics
- Displacement magnitude checks
- NaN/Inf detection
- Intensity range validation

## Future Enhancements

Potential improvements:
- Support for 4D (time-series) TIF files
- Adaptive quality thresholding
- Multi-scale extraction (64³, 128³, 256³)
- Domain adaptation preprocessing
- Uncertainty quantification during generation
