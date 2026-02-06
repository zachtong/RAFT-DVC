"""
Visualize a generated sample: volumes, flow field, and statistics.

Usage:
    python check_sample.py --sample 0  # Check sample_0000
    python check_sample.py --sample 5 --split val  # Check val sample_0005
"""

import numpy as np
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def load_sample(data_dir, split, sample_idx):
    """Load a sample from separate vol0, vol1, flow files."""
    sample_name = f"sample_{sample_idx:04d}"
    split_dir = Path(data_dir) / split

    # Load volumes and flow
    vol0_path = split_dir / 'vol0' / f'{sample_name}.npy'
    vol1_path = split_dir / 'vol1' / f'{sample_name}.npy'
    flow_path = split_dir / 'flow' / f'{sample_name}.npy'
    metadata_path = split_dir / 'metadata' / f'{sample_name}.npy'

    # Check if files exist
    if not vol0_path.exists():
        raise FileNotFoundError(f"vol0 not found: {vol0_path}")
    if not vol1_path.exists():
        raise FileNotFoundError(f"vol1 not found: {vol1_path}")
    if not flow_path.exists():
        raise FileNotFoundError(f"flow not found: {flow_path}")

    # Load data
    data = {
        'vol0': np.load(vol0_path),
        'vol1': np.load(vol1_path),
        'flow': np.load(flow_path)
    }

    # Load metadata if available
    if metadata_path.exists():
        data['metadata'] = np.load(metadata_path, allow_pickle=True).item()

    return data


def print_statistics(data, sample_name):
    """Print detailed statistics about the sample."""
    vol0 = data['vol0']
    vol1 = data['vol1']
    flow = data['flow']

    # Compute flow magnitude
    flow_mag = np.sqrt(np.sum(flow**2, axis=0))

    print("=" * 70)
    print(f"Sample: {sample_name}")
    print("=" * 70)

    # Volume statistics
    print("\n[Volume Statistics]")
    print(f"  Shape: {vol0.shape}")
    print(f"  Vol0 intensity: [{vol0.min():.3f}, {vol0.max():.3f}], mean={vol0.mean():.3f}")
    print(f"  Vol1 intensity: [{vol1.min():.3f}, {vol1.max():.3f}], mean={vol1.mean():.3f}")

    # Flow statistics
    print("\n[Flow Field Statistics]")
    print(f"  Flow shape: {flow.shape}")
    print(f"  Flow components (X/Y/Z):")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        print(f"    {axis}: [{flow[i].min():7.2f}, {flow[i].max():7.2f}] voxels, "
              f"mean={flow[i].mean():6.2f}, std={flow[i].std():5.2f}")

    print(f"\n  Flow Magnitude:")
    print(f"    Mean:   {flow_mag.mean():6.2f} voxels")
    print(f"    Median: {np.median(flow_mag):6.2f} voxels")
    print(f"    Std:    {flow_mag.std():6.2f} voxels")
    print(f"    Max:    {flow_mag.max():6.2f} voxels")
    print(f"    Min:    {flow_mag.min():6.2f} voxels")

    # Percentiles
    p25, p50, p75, p95, p99 = np.percentile(flow_mag, [25, 50, 75, 95, 99])
    print(f"\n  Flow Magnitude Percentiles:")
    print(f"    25th: {p25:6.2f} voxels")
    print(f"    50th: {p50:6.2f} voxels")
    print(f"    75th: {p75:6.2f} voxels")
    print(f"    95th: {p95:6.2f} voxels")
    print(f"    99th: {p99:6.2f} voxels")

    # Metadata if available
    if 'metadata' in data:
        print("\n[Metadata]")
        metadata = data['metadata']
        if 'deformation_type' in metadata:
            print(f"  Deformation type: {metadata['deformation_type']}")
        if 'num_beads' in metadata:
            print(f"  Number of beads: {metadata['num_beads']}")

    print("=" * 70)

    # Quality assessment
    print("\n[Quality Assessment]")
    if flow_mag.mean() < 2:
        print("  ⚠ WARNING: Mean flow is very small (<2 voxels) - deformation may be too weak")
    elif flow_mag.mean() > 7:
        print("  ⚠ WARNING: Mean flow is large (>7 voxels) - may be hard to learn")
    else:
        print("  ✓ Flow magnitude looks reasonable (2-7 voxels mean)")

    if flow_mag.max() > 20:
        print("  ⚠ WARNING: Maximum flow is very large (>20 voxels)")
    elif flow_mag.max() > 15:
        print("  ⚠ Note: Maximum flow is somewhat large (>15 voxels)")
    else:
        print("  ✓ Maximum flow is reasonable (<15 voxels)")

    print()


def visualize_sample(data, output_path, sample_name):
    """Create comprehensive visualization."""
    if not HAS_MATPLOTLIB:
        print("Cannot create visualization: matplotlib not installed")
        return

    vol0 = data['vol0']
    vol1 = data['vol1']
    flow = data['flow']

    # Compute flow magnitude
    flow_mag = np.sqrt(np.sum(flow**2, axis=0))

    # Get central slices
    D, H, W = vol0.shape
    z_mid = D // 2
    y_mid = H // 2
    x_mid = W // 2

    # Create figure
    fig = plt.figure(figsize=(20, 16))

    # Row 1: Volumes (Z-slice)
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(vol0[z_mid], cmap='gray')
    ax1.set_title('Vol0 (Reference)\nZ-slice', fontsize=10)
    ax1.axis('off')

    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(vol1[z_mid], cmap='gray')
    ax2.set_title('Vol1 (Deformed)\nZ-slice', fontsize=10)
    ax2.axis('off')

    ax3 = plt.subplot(4, 4, 3)
    diff = np.abs(vol1[z_mid] - vol0[z_mid])
    ax3.imshow(diff, cmap='hot')
    ax3.set_title('Absolute Difference\nZ-slice', fontsize=10)
    ax3.axis('off')

    ax4 = plt.subplot(4, 4, 4)
    ax4.imshow(flow_mag[z_mid], cmap='jet')
    ax4.set_title('Flow Magnitude\nZ-slice', fontsize=10)
    ax4.axis('off')
    plt.colorbar(ax4.images[0], ax=ax4, fraction=0.046)

    # Row 2: Y-slice
    ax5 = plt.subplot(4, 4, 5)
    ax5.imshow(vol0[:, y_mid, :], cmap='gray')
    ax5.set_title('Vol0\nY-slice', fontsize=10)
    ax5.axis('off')

    ax6 = plt.subplot(4, 4, 6)
    ax6.imshow(vol1[:, y_mid, :], cmap='gray')
    ax6.set_title('Vol1\nY-slice', fontsize=10)
    ax6.axis('off')

    ax7 = plt.subplot(4, 4, 7)
    ax7.imshow(np.abs(vol1[:, y_mid, :] - vol0[:, y_mid, :]), cmap='hot')
    ax7.set_title('Absolute Difference\nY-slice', fontsize=10)
    ax7.axis('off')

    ax8 = plt.subplot(4, 4, 8)
    ax8.imshow(flow_mag[:, y_mid, :], cmap='jet')
    ax8.set_title('Flow Magnitude\nY-slice', fontsize=10)
    ax8.axis('off')
    plt.colorbar(ax8.images[0], ax=ax8, fraction=0.046)

    # Row 3: Flow components (Z-slice)
    vmax = max(np.abs(flow).max(), 0.1)

    ax9 = plt.subplot(4, 4, 9)
    im9 = ax9.imshow(flow[0, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax9.set_title('Flow Z-component\n(depth)', fontsize=10)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    ax10 = plt.subplot(4, 4, 10)
    im10 = ax10.imshow(flow[1, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax10.set_title('Flow Y-component\n(vertical)', fontsize=10)
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046)

    ax11 = plt.subplot(4, 4, 11)
    im11 = ax11.imshow(flow[2, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax11.set_title('Flow X-component\n(horizontal)', fontsize=10)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)

    # Flow magnitude histogram
    ax12 = plt.subplot(4, 4, 12)
    ax12.hist(flow_mag.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax12.axvline(flow_mag.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {flow_mag.mean():.2f}')
    ax12.axvline(np.median(flow_mag), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(flow_mag):.2f}')
    ax12.set_xlabel('Flow Magnitude (voxels)', fontsize=10)
    ax12.set_ylabel('Frequency', fontsize=10)
    ax12.set_title('Flow Magnitude Distribution', fontsize=10)
    ax12.legend(fontsize=8)
    ax12.grid(alpha=0.3)

    # Row 4: Statistics and flow vector field
    ax13 = plt.subplot(4, 4, 13)
    # Downsample for quiver plot
    step = 8
    Y, X = np.meshgrid(
        np.arange(0, H, step),
        np.arange(0, W, step),
        indexing='ij'
    )
    U = flow[2, z_mid, ::step, ::step]  # X component
    V = flow[1, z_mid, ::step, ::step]  # Y component
    M = flow_mag[z_mid, ::step, ::step]

    ax13.imshow(vol0[z_mid], cmap='gray', alpha=0.5)
    Q = ax13.quiver(X, Y, U, V, M, cmap='jet', scale=50, width=0.003)
    ax13.set_title('Flow Vectors (2D projection)\nZ-slice', fontsize=10)
    ax13.axis('off')
    plt.colorbar(Q, ax=ax13, fraction=0.046)

    # Statistics text
    ax14 = plt.subplot(4, 4, 14)
    ax14.axis('off')
    stats_text = f"""
    STATISTICS

    Volume Shape: {vol0.shape}

    Flow Magnitude:
      Mean:   {flow_mag.mean():6.2f} voxels
      Median: {np.median(flow_mag):6.2f} voxels
      Std:    {flow_mag.std():6.2f} voxels
      Max:    {flow_mag.max():6.2f} voxels
      Min:    {flow_mag.min():6.2f} voxels

    Flow Components:
      Z: [{flow[0].min():6.2f}, {flow[0].max():6.2f}]
      Y: [{flow[1].min():6.2f}, {flow[1].max():6.2f}]
      X: [{flow[2].min():6.2f}, {flow[2].max():6.2f}]

    Intensity:
      Vol0: [{vol0.min():.3f}, {vol0.max():.3f}]
      Vol1: [{vol1.min():.3f}, {vol1.max():.3f}]
    """
    ax14.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
              verticalalignment='center')

    # 3D flow visualization (simplified)
    ax15 = plt.subplot(4, 4, 15)
    # Show maximum intensity projection
    mip = vol0.max(axis=0)
    ax15.imshow(mip, cmap='gray')
    ax15.set_title('Vol0 MIP\n(Max Intensity Projection)', fontsize=10)
    ax15.axis('off')

    ax16 = plt.subplot(4, 4, 16)
    flow_mip = flow_mag.max(axis=0)
    im16 = ax16.imshow(flow_mip, cmap='jet')
    ax16.set_title('Flow Magnitude MIP', fontsize=10)
    ax16.axis('off')
    plt.colorbar(im16, ax=ax16, fraction=0.046)

    plt.suptitle(f'{sample_name} - Comprehensive Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Check and visualize a generated sample'
    )
    parser.add_argument(
        '--sample', type=int, default=0,
        help='Sample index (e.g., 0 for sample_0000)'
    )
    parser.add_argument(
        '--split', type=str, default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/synthetic_confocal',
        help='Data directory'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization (only print statistics)'
    )
    args = parser.parse_args()

    # Check if sample exists
    sample_name = f"sample_{args.sample:04d}"
    split_dir = Path(args.data_dir) / args.split
    vol0_path = split_dir / 'vol0' / f"{sample_name}.npy"

    if not vol0_path.exists():
        print(f"Error: Sample not found at {split_dir}")
        print(f"Looking for: {vol0_path}")
        print("\nMake sure the sample has been generated.")
        return

    # Load sample
    print(f"Loading: {sample_name} from {args.split} split")
    try:
        data = load_sample(args.data_dir, args.split, args.sample)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print statistics
    print_statistics(data, sample_name)

    # Create visualization
    if not args.no_viz:
        output_path = (
            Path(args.data_dir) /
            f"{sample_name}_{args.split}_visualization.png"
        )
        visualize_sample(data, output_path, f"{sample_name} ({args.split})")


if __name__ == '__main__':
    main()
