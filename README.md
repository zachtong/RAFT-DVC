# RAFT-DVC: Deep Learning for Digital Volume Correlation

A PyTorch-based implementation for 3D Digital Volume Correlation using RAFT optical flow network, based on the VolRAFT paper (CVPR 2024).

## Features

- 3D RAFT network for volumetric optical flow estimation
- Training pipeline with synthetic data generation
- Inference with sliding window for large volumes
- GUI application for easy DVC computation (coming soon)

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Conda or Mamba (recommended)

### Setup

```bash
# Create conda environment
conda create -n raft-dvc python=3.10
conda activate raft-dvc

# Install PyTorch with CUDA
conda install pytorch==2.1.0 torchvision==0.16.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
RAFT-DVC/
├── src/                    # Source code
│   ├── core/               # Core RAFT-DVC network
│   ├── data/               # Data loading and processing
│   ├── training/           # Training pipeline
│   ├── inference/          # Inference engine
│   └── utils/              # Utilities
├── gui/                    # GUI application
├── configs/                # Configuration files
├── checkpoints/            # Model checkpoints
├── scripts/                # Training/inference scripts
└── tests/                  # Unit tests
```

## Usage

### Training

```bash
python scripts/train.py --config configs/training/default.yaml
```

### Inference

```bash
python scripts/infer.py --input <volume_pairs> --checkpoint <model.pth> --output <results>
```

## References

- [VolRAFT Paper (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Wong_VolRAFT_Volumetric_Optical_Flow_Network_for_Digital_Volume_Correlation_of_CVPRW_2024_paper.html)
- [VolRAFT GitHub](https://github.com/hereon-mbs/VolRAFT)
- [RAFT: Recurrent All-Pairs Field Transforms](https://github.com/princeton-vl/RAFT)

## License

MIT License
