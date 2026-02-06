"""
Visualization tool for synthetic confocal dataset.

Usage:
    python tools/visualize_confocal.py --sample 0
    python tools/visualize_confocal.py --random 5
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_sample(dataset_dir, split, sample_idx):
    """
    Load a sample from the dataset.

    Args:
        dataset_dir: Path to dataset directory
        split: 'train', 'val', or 'test'
        sample_idx: Sample index

    Returns:
        sample_data: dict with vol0, vol1, flow, metadata
    """
    split_dir = Path(dataset_dir) / split

    vol0 = np.load(split_dir / 'vol0' / f'sample_{sample_idx:04d}.npy')
    vol1 = np.load(split_dir / 'vol1' / f'sample_{sample_idx:04d}.npy')
    flow = np.load(split_dir / 'flow' / f'sample_{sample_idx:04d}.npy')

    # Load metadata if exists
    meta_file = split_dir / 'metadata' / f'sample_{sample_idx:04d}.json'
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return {
        'vol0': vol0,
        'vol1': vol1,
        'flow': flow,
        'metadata': metadata
    }


def visualize_sample_detailed(sample_data, output_path, sample_idx):
    """
    Create detailed visualization of a sample.

    Args:
        sample_data: dict with vol0, vol1, flow, metadata
        output_path: Path to save figure
        sample_idx: Sample index
    """
    vol0 = sample_data['vol0']
    vol1 = sample_data['vol1']
    flow = sample_data['flow']
    metadata = sample_data['metadata']

    # Create figure
    fig = plt.figure(figsize=(24, 14))

    D, H, W = vol0.shape
    z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

    # Compute flow magnitude
    flow_mag = np.sqrt(np.sum(flow**2, axis=0))

    # Row 1: Volume slices (multiple planes)
    planes_z = [D//4, D//2, 3*D//4]

    for i, z in enumerate(planes_z):
        # Vol0
        ax = plt.subplot(4, 6, i+1)
        ax.imshow(vol0[z], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Vol0 XY (z={z})')
        ax.axis('off')

        # Vol1
        ax = plt.subplot(4, 6, i+7)
        ax.imshow(vol1[z], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Vol1 XY (z={z})')
        ax.axis('off')

    # Orthogonal views
    # XZ view
    ax = plt.subplot(4, 6, 4)
    ax.imshow(vol0[:, y_mid, :], cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Vol0 XZ (y={y_mid})')
    ax.axis('off')

    ax = plt.subplot(4, 6, 10)
    ax.imshow(vol1[:, y_mid, :], cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Vol1 XZ (y={y_mid})')
    ax.axis('off')

    # YZ view
    ax = plt.subplot(4, 6, 5)
    ax.imshow(vol0[:, :, x_mid], cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Vol0 YZ (x={x_mid})')
    ax.axis('off')

    ax = plt.subplot(4, 6, 11)
    ax.imshow(vol1[:, :, x_mid], cmap='gray', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Vol1 YZ (x={x_mid})')
    ax.axis('off')

    # Difference image
    ax = plt.subplot(4, 6, 6)
    diff = np.abs(vol1[z_mid] - vol0[z_mid])
    im = ax.imshow(diff, cmap='hot')
    ax.set_title('Abs Difference (mid)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(4, 6, 12)
    diff_xz = np.abs(vol1[:, y_mid, :] - vol0[:, y_mid, :])
    im = ax.imshow(diff_xz, cmap='hot', aspect='auto')
    ax.set_title('Abs Difference XZ')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: Flow magnitude
    for i, z in enumerate(planes_z):
        ax = plt.subplot(4, 6, i+13)
        im = ax.imshow(flow_mag[z], cmap='jet')
        ax.set_title(f'Flow Mag (z={z})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Flow magnitude orthogonal
    ax = plt.subplot(4, 6, 16)
    im = ax.imshow(flow_mag[:, y_mid, :], cmap='jet', aspect='auto')
    ax.set_title(f'Flow Mag XZ')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(4, 6, 17)
    im = ax.imshow(flow_mag[:, :, x_mid], cmap='jet', aspect='auto')
    ax.set_title(f'Flow Mag YZ')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 4: Flow components
    vmax_flow = max(np.abs(flow).max(), 1.0)

    ax = plt.subplot(4, 6, 19)
    im = ax.imshow(flow[0, z_mid], cmap='RdBu_r', vmin=-vmax_flow, vmax=vmax_flow)
    ax.set_title('Flow Z Component')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(4, 6, 20)
    im = ax.imshow(flow[1, z_mid], cmap='RdBu_r', vmin=-vmax_flow, vmax=vmax_flow)
    ax.set_title('Flow Y Component')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = plt.subplot(4, 6, 21)
    im = ax.imshow(flow[2, z_mid], cmap='RdBu_r', vmin=-vmax_flow, vmax=vmax_flow)
    ax.set_title('Flow X Component')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Statistics panel
    ax = plt.subplot(4, 6, 18)
    ax.axis('off')

    stats_text = f"Sample {sample_idx}\n\n"

    if 'deform_type' in metadata:
        stats_text += f"Deformation: {metadata['deform_type']}\n\n"

    if 'flow_stats' in metadata:
        fs = metadata['flow_stats']
        stats_text += f"Flow Statistics:\n"
        stats_text += f"  Mean: {fs['magnitude_mean']:.2f} voxels\n"
        stats_text += f"  Std:  {fs['magnitude_std']:.2f} voxels\n"
        stats_text += f"  Max:  {fs['magnitude_max']:.2f} voxels\n"
        stats_text += f"  Min:  {fs['magnitude_min']:.2f} voxels\n\n"

    if 'bead_stats' in metadata:
        bs = metadata['bead_stats']
        stats_text += f"Bead Statistics:\n"
        stats_text += f"  Count: {bs['num_beads']}\n"
        stats_text += f"  Radius: {bs['radius_mean']:.2f}±{bs['radius_std']:.2f}\n"
        stats_text += f"  Intensity: {bs['intensity_mean']:.2f}±{bs['intensity_std']:.2f}\n"
        stats_text += f"  Vol. Fraction: {bs['volume_fraction']:.3%}\n\n"

    if 'warp_stats' in metadata:
        ws = metadata['warp_stats']
        stats_text += f"Warping:\n"
        stats_text += f"  Original: {ws['original_count']} beads\n"
        stats_text += f"  Visible: {ws['warped_count']} ({ws['visible_ratio']:.1%})\n"
        stats_text += f"  Displacement: {ws['displacement_mean']:.2f}±{ws['displacement_std']:.2f}\n\n"

    if 'imaging_vol0' in metadata:
        im0 = metadata['imaging_vol0']
        stats_text += f"Imaging:\n"
        stats_text += f"  PSF σ: {im0['psf_sigma']:.2f} voxels\n"
        stats_text += f"  SNR: {im0['snr_db']:.1f} dB\n"

    ax.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top',
           family='monospace', transform=ax.transAxes)

    # Histograms
    ax = plt.subplot(4, 6, 22)
    ax.hist(vol0.flatten(), bins=50, alpha=0.5, label='Vol0', density=True, color='blue')
    ax.hist(vol1.flatten(), bins=50, alpha=0.5, label='Vol1', density=True, color='red')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.set_title('Intensity Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = plt.subplot(4, 6, 23)
    ax.hist(flow_mag.flatten(), bins=50, color='purple', alpha=0.7)
    ax.set_xlabel('Displacement (voxels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Flow Magnitude Distribution')
    ax.grid(alpha=0.3)

    ax = plt.subplot(4, 6, 24)
    components = ['Z', 'Y', 'X']
    colors = ['red', 'green', 'blue']
    for i, (comp, color) in enumerate(zip(components, colors)):
        ax.hist(flow[i].flatten(), bins=50, alpha=0.5, label=comp,
               density=True, color=color)
    ax.set_xlabel('Displacement (voxels)')
    ax.set_ylabel('Density')
    ax.set_title('Flow Components')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Synthetic Confocal Dataset - Sample {sample_idx}',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize synthetic confocal dataset samples'
    )
    parser.add_argument('--dataset', type=str,
                       default='data/synthetic_confocal',
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to visualize')
    parser.add_argument('--sample', type=int, default=None,
                       help='Specific sample index to visualize')
    parser.add_argument('--random', type=int, default=None,
                       help='Visualize N random samples')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Determine which samples to visualize
    if args.sample is not None:
        sample_indices = [args.sample]
    elif args.random is not None:
        # Find all samples
        split_dir = Path(args.dataset) / args.split / 'vol0'
        all_samples = sorted(split_dir.glob('sample_*.npy'))
        max_idx = len(all_samples) - 1

        # Random sample
        sample_indices = np.random.choice(max_idx + 1, min(args.random, max_idx + 1),
                                         replace=False)
    else:
        # Default: visualize first 5
        sample_indices = list(range(5))

    print(f"Visualizing {len(sample_indices)} samples from {args.split} set...")

    for idx in sample_indices:
        print(f"\nProcessing sample {idx}...")

        # Load sample
        sample_data = load_sample(args.dataset, args.split, idx)

        # Visualize
        output_path = output_dir / f'{args.split}_sample_{idx:04d}.png'
        visualize_sample_detailed(sample_data, output_path, idx)

    print(f"\n✓ Done! Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
