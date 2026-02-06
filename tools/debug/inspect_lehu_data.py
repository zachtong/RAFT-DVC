"""
Inspect Lehu's dataset structure and format.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def inspect_dataset(data_root):
    """Inspect dataset structure and sample data."""

    data_root = Path(data_root)
    print("=" * 70)
    print("Lehu's Dataset Inspection")
    print("=" * 70)
    print()

    # Check dataset index
    index_file = data_root / 'dataset_index.json'
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        print(f"Dataset Index: {len(index_data)} samples")
        print(f"First sample metadata:")
        print(json.dumps(index_data[0], indent=2))
        print()

    # Check samples directory
    samples_dir = data_root / 'samples'
    if not samples_dir.exists():
        print(f"ERROR: samples directory not found at {samples_dir}")
        return

    sample_dirs = sorted(list(samples_dir.glob('sample_*')))
    print(f"Number of samples: {len(sample_dirs)}")
    print()

    if len(sample_dirs) == 0:
        print("No samples found!")
        return

    # Inspect first sample
    first_sample = sample_dirs[0]
    print(f"Inspecting first sample: {first_sample.name}")
    print("-" * 70)

    # Check files
    files = {
        'v0': first_sample / 'v0.npy',
        'v1': first_sample / 'v1.npy',
        'flow': first_sample / 'flow.npy',
        'vm': first_sample / 'vm.npy',
        'meta': first_sample / 'meta.json'
    }

    for name, path in files.items():
        if path.exists():
            if name == 'meta':
                with open(path, 'r') as f:
                    meta = json.load(f)
                print(f"  {name}: {json.dumps(meta, indent=4)}")
            else:
                data = np.load(path)
                print(f"  {name}: shape={data.shape}, dtype={data.dtype}, "
                      f"min={data.min():.4f}, max={data.max():.4f}")
        else:
            print(f"  {name}: NOT FOUND")

    print()
    print("=" * 70)
    print("Data Format Summary")
    print("=" * 70)
    print()
    print("Directory structure:")
    print("  samples/")
    print("    sample_0000000/")
    print("      v0.npy        - Shape: (1, 64, 64, 64) - Reference volume")
    print("      v1.npy        - Shape: (1, 64, 64, 64) - Deformed volume")
    print("      flow.npy      - Shape: (3, 64, 64, 64) - Ground truth flow (X, Y, Z)")
    print("      vm.npy        - Shape: (1, 64, 64, 64) - Middle/mask volume (optional)")
    print("      meta.json     - Sample metadata")
    print()
    print("Data characteristics:")
    print("  - Volume size: 64x64x64 voxels")
    print("  - Data type: float32")
    print("  - Format: VolRAFT synthetic dataset format")
    print()

    # Compare with current format
    print("=" * 70)
    print("Comparison with Current Format")
    print("=" * 70)
    print()
    print("Current format (synthetic_confocal):")
    print("  root_dir/")
    print("    vol0/")
    print("      000000.npy")
    print("      000001.npy")
    print("    vol1/")
    print("      000000.npy")
    print("    flow/")
    print("      000000.npy")
    print()
    print("Lehu's format (VolRAFT):")
    print("  samples/")
    print("    sample_0000000/")
    print("      v0.npy, v1.npy, flow.npy")
    print()
    print("Key differences:")
    print("  1. Organization: Per-sample directories vs. separated by type")
    print("  2. Volume size: 64^3 (Lehu) vs. 128^3 (current)")
    print("  3. File naming: v0/v1/flow vs. vol0/vol1/flow")
    print()

    return sample_dirs, files


def visualize_sample(data_root, sample_idx=0):
    """Visualize a sample from the dataset."""
    data_root = Path(data_root)
    samples_dir = data_root / 'samples'

    sample_dirs = sorted(list(samples_dir.glob('sample_*')))
    if sample_idx >= len(sample_dirs):
        print(f"Sample {sample_idx} not found!")
        return

    sample_dir = sample_dirs[sample_idx]
    print(f"\nVisualizing {sample_dir.name}...")

    # Load data
    v0 = np.load(sample_dir / 'v0.npy')[0]  # Remove channel dim
    v1 = np.load(sample_dir / 'v1.npy')[0]
    flow = np.load(sample_dir / 'flow.npy')

    # Get middle slices
    mid = v0.shape[0] // 2

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Volumes and flow magnitude
    axes[0, 0].imshow(v0[mid], cmap='gray')
    axes[0, 0].set_title('V0 (Reference)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(v1[mid], cmap='gray')
    axes[0, 1].set_title('V1 (Deformed)')
    axes[0, 1].axis('off')

    flow_mag = np.sqrt(np.sum(flow**2, axis=0))
    im = axes[0, 2].imshow(flow_mag[mid], cmap='viridis')
    axes[0, 2].set_title(f'Flow Magnitude (max={flow_mag.max():.2f})')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Row 2: Flow components
    for i, comp in enumerate(['X', 'Y', 'Z']):
        im = axes[1, i].imshow(flow[i, mid], cmap='RdBu_r', vmin=-5, vmax=5)
        axes[1, i].set_title(f'Flow {comp}')
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()

    output_path = Path('lehu_sample_inspection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {output_path.absolute()}")


if __name__ == '__main__':
    import sys

    data_root = "data/Lehu's_data/VolRAFT_Synthetic_Dataset"

    if len(sys.argv) > 1:
        data_root = sys.argv[1]

    # Inspect dataset
    sample_dirs, files = inspect_dataset(data_root)

    # Visualize first sample
    if len(sample_dirs) > 0:
        visualize_sample(data_root, sample_idx=0)
