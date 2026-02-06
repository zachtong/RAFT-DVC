# RAFT-DVC: Deep Learning for Digital Volume Correlation

A comprehensive PyTorch implementation of 3D Digital Volume Correlation using RAFT optical flow architecture, with advanced synthetic data generation and training pipelines.

**Based on**: [VolRAFT (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Wong_VolRAFT_Volumetric_Optical_Flow_Network_for_Digital_Volume_Correlation_of_CVPRW_2024_paper.html)

**Authors**: Zach Tong, Prof. Jin Yang, Lehu Bu

## ✨ Features

- **3D RAFT Architecture**: Volumetric optical flow estimation with iterative refinement
- **Synthetic Data Generation**: Comprehensive confocal microscopy simulation system
  - Multiple deformation types (affine, B-spline, localized, combined)
  - Realistic particle rendering with configurable properties
  - Flexible noise models and imaging simulation
- **Flexible Training Pipeline**:
  - Multiple model configurations (varying correlation radius, pyramid levels)
  - CyclicLR scheduler matching VolRAFT
  - Automatic Mixed Precision (AMP) support
  - TensorBoard logging with visualization
- **Robust Inference System**:
  - Sliding window for large volumes
  - Batch processing and single-sample testing
  - Model registry for easy checkpoint management
- **Unified Interface**: Windows batch script (`run.bat`) for all operations

## 🖥️ Hardware Requirements

**Developed and tested on**: NVIDIA RTX 5090 (24GB VRAM)

**Minimum requirements**:
- CUDA-capable GPU with at least 16GB VRAM for training 128³ volumes
- 32GB system RAM recommended
- For 32³ volumes: 8GB VRAM sufficient

## 📦 Installation

### Prerequisites

- Python 3.9+
- CUDA 12.1+ (for RTX 5090 / Ada Lovelace GPUs)
- Conda (recommended) or pip

### Setup

```bash
# Create conda environment
conda create -n raft-dvc python=3.10
conda activate raft-dvc

# Install PyTorch with CUDA 12.1 (for RTX 5090)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## 📁 Project Structure

```
RAFT-DVC/
├── src/                          # Source code
│   ├── core/                     # Core RAFT-DVC network
│   │   ├── raft_dvc.py           # Main model
│   │   ├── extractor.py          # Feature extraction
│   │   ├── corr.py               # Correlation volume
│   │   └── update.py             # GRU update operator
│   ├── training/                 # Training pipeline
│   │   ├── trainer.py            # Training loop
│   │   ├── dataset.py            # Data loading
│   │   └── loss.py               # Loss functions
│   ├── inference/                # Inference engine
│   │   ├── pipeline.py           # Inference pipeline
│   │   ├── model_registry.py    # Model management
│   │   └── tiling.py             # Sliding window
│   └── utils/                    # Utilities
│       ├── io.py                 # File I/O
│       └── memory.py             # Memory management
│
├── scripts/                      # Executable scripts
│   ├── data_generation/          # Synthetic data generation
│   │   ├── generate_confocal_dataset.py
│   │   └── modules/              # Generation modules
│   │       ├── beads.py          # Particle rendering
│   │       ├── deformation.py    # Deformation fields
│   │       └── imaging.py        # Imaging simulation
│   ├── train_confocal.py         # Training script
│   └── test_confocal.py          # Testing script
│
├── configs/                      # Configuration files
│   ├── training/                 # Training configs
│   │   └── confocal_*.yaml       # Various configurations
│   ├── models/                   # Model architecture configs
│   │   └── raft_dvc_*.yaml       # Model variants
│   ├── data_generation/          # Data generation configs
│   │   └── confocal_*.yaml       # Dataset configurations
│   └── inference/                # Inference configs
│
├── tools/                        # Utility tools
│   ├── visualize_confocal.py     # Visualization tools
│   └── test_cuda.py              # CUDA setup verification
│
├── docs/                         # Documentation
│   └── (See documentation below)
│
├── QUICK_START_CN.md             # Quick start (Chinese)
├── run.bat                       # Unified entry point (Windows)
├── inference_test.py             # Inference testing script
└── preview_v3_dataset.py         # Dataset preview tool
```

**Note**: `data/`, `outputs/`, and model checkpoints are excluded from the repository due to size.

## 🚀 Quick Start

### Option 1: Unified Interface (Windows - Recommended)

```bash
run.bat
```

This provides an interactive menu for:
1. **Training**: Train models with different configurations
2. **Inference**: Test on validation/test sets (single or batch)
3. **Dataset Generation**: Create synthetic confocal datasets
4. **Visualization**: Preview datasets and results

### Option 2: Manual Command Line

#### 1. Generate Synthetic Dataset

```bash
python scripts/data_generation/generate_confocal_dataset.py \
    --config configs/data_generation/confocal_particles_128.yaml
```

#### 2. Train Model

```bash
python scripts/train_confocal.py \
    --config configs/training/confocal_128_1_8_p4_r4.yaml \
    --model-config configs/models/raft_dvc_1_8_p4_r4.yaml
```

#### 3. Inference

```bash
# Single sample
python inference_test.py \
    --checkpoint outputs/training/confocal_128_v1_1_8_p4_r4/checkpoint_best.pth \
    --split val --sample 50

# Batch processing (all samples)
python inference_test.py \
    --checkpoint outputs/training/confocal_128_v1_1_8_p4_r4/checkpoint_best.pth \
    --split test --num-vis 10
```

## 📊 Dataset Configurations

Multiple dataset variants are provided:

- **confocal_128**: Original configuration (larger particles, 400-1500/volume)
- **confocal_128_v2**: Smaller particles (0.8-3.0 voxels), wider density (200-2000)
- **confocal_128_v3**: Same as v2 but uses v1 dataset for comparison
- **confocal_32**: Smaller volumes for fast prototyping

See [configs/data_generation/](configs/data_generation/) for details.

## 🎯 Model Configurations

Multiple model architectures available:

- **1_8_p4_r4**: Hidden dim=128, 4 pyramid levels, correlation radius=4 (recommended)
- **1_4_p3_r4**: Hidden dim=128, 3 pyramid levels, correlation radius=4 (faster)
- **1_2_p2_r4**: Hidden dim=128, 2 pyramid levels, correlation radius=4 (lightweight)

See [configs/models/](configs/models/) for architecture details.

## 📚 Documentation

- **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**: Detailed training instructions
- **[docs/DATA_GENERATION.md](docs/DATA_GENERATION.md)**: Synthetic data generation guide
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System architecture overview
- **[QUICK_START_CN.md](QUICK_START_CN.md)**: 中文快速开始指南

## 🔧 Development

### Testing CUDA Setup

```bash
python tools/test_cuda.py
```

### Visualizing Data

```bash
python tools/visualize_confocal.py --data-dir data/synthetic_confocal_128/train
```

### Preview Dataset Configuration

```bash
python preview_v3_dataset.py --config configs/data_generation/confocal_particles_32_v3.yaml
```

## 📖 Citation

If you use this code in your research, please cite the original VolRAFT paper:

```bibtex
@inproceedings{wong2024volraft,
  title={VolRAFT: Volumetric Optical Flow Network for Digital Volume Correlation},
  author={Wong, Chun Yin and Schanz, Daniel and Schr\"oder, Andreas and Geisler, Reinhard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
  pages={3621--3630}
}
```

## 🙏 Acknowledgments

This implementation builds upon:
- [VolRAFT](https://github.com/hereon-mbs/VolRAFT) - Original implementation
- [RAFT](https://github.com/princeton-vl/RAFT) - 2D optical flow foundation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: Please replace `[Your Name]` in the LICENSE file with your actual name before publishing.
