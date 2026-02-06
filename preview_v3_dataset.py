"""
Temporary script to preview v3 dataset configuration.
Generates a small batch of samples and creates a comprehensive visualization.
"""
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.data_generation.generate_confocal_dataset import DatasetGenerator

def create_preview_config(base_config_path, temp_dir, num_samples=12):
    """Create a temporary config with fewer samples for preview."""
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Modify config for preview
    config['dataset']['name'] = 'confocal_32_v3_preview'
    config['dataset']['output_dir'] = str(temp_dir)
    config['dataset']['num_samples'] = {
        'train': num_samples,
        'val': 0,
        'test': 0
    }
    config['generation']['visualize_samples'] = False  # We'll do custom viz

    # Save temporary config
    temp_config_path = temp_dir / 'preview_config.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    return temp_config_path, config


def visualize_multi_samples(data_dir, output_path, num_samples=10):
    """Create comprehensive visualization of multiple samples."""
    train_dir = Path(data_dir) / 'train'
    metadata_dir = train_dir / 'metadata'

    # Load samples
    samples = []
    for i in range(num_samples):
        vol0_path = train_dir / 'vol0' / f'sample_{i:04d}.npy'
        vol1_path = train_dir / 'vol1' / f'sample_{i:04d}.npy'
        flow_path = train_dir / 'flow' / f'sample_{i:04d}.npy'
        meta_path = metadata_dir / f'sample_{i:04d}.json'

        if not all(p.exists() for p in [vol0_path, vol1_path, flow_path, meta_path]):
            print(f"Warning: Sample {i} incomplete, skipping")
            continue

        vol0 = np.load(vol0_path)
        vol1 = np.load(vol1_path)
        flow = np.load(flow_path)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        samples.append({
            'vol0': vol0,
            'vol1': vol1,
            'flow': flow,
            'metadata': metadata
        })

    if len(samples) == 0:
        print("Error: No valid samples found")
        return

    print(f"\nLoaded {len(samples)} samples for visualization")

    # Create figure with gridspec
    fig = plt.figure(figsize=(20, 3 * len(samples)))
    gs = GridSpec(len(samples), 5, figure=fig, hspace=0.3, wspace=0.3)

    for idx, sample in enumerate(samples):
        vol0 = sample['vol0']
        vol1 = sample['vol1']
        flow = sample['flow']
        meta = sample['metadata']

        # Get middle slice
        d_mid = vol0.shape[0] // 2

        # Extract metadata info
        deform_type = meta['deform_type']
        num_beads = meta['bead_stats']['num_beads']
        flow_mag_mean = meta['flow_stats']['magnitude_mean']
        flow_mag_max = meta['flow_stats']['magnitude_max']

        # Vol0 (reference)
        ax0 = fig.add_subplot(gs[idx, 0])
        im0 = ax0.imshow(vol0[d_mid], cmap='gray', vmin=0, vmax=1)
        ax0.set_title(f'Sample {idx}\nVol0 ({num_beads} beads)',
                     fontsize=9, fontweight='bold')
        ax0.axis('off')
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Vol1 (deformed)
        ax1 = fig.add_subplot(gs[idx, 1])
        im1 = ax1.imshow(vol1[d_mid], cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Vol1\n{deform_type.upper()}',
                     fontsize=9, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Flow U (X component)
        ax2 = fig.add_subplot(gs[idx, 2])
        u = flow[0, d_mid]
        vmax_u = max(abs(u.min()), abs(u.max()), 0.1)
        im2 = ax2.imshow(u, cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u)
        ax2.set_title(f'Flow U\n[{u.min():.2f}, {u.max():.2f}]',
                     fontsize=9, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Flow V (Y component)
        ax3 = fig.add_subplot(gs[idx, 3])
        v = flow[1, d_mid]
        vmax_v = max(abs(v.min()), abs(v.max()), 0.1)
        im3 = ax3.imshow(v, cmap='RdBu_r', vmin=-vmax_v, vmax=vmax_v)
        ax3.set_title(f'Flow V\n[{v.min():.2f}, {v.max():.2f}]',
                     fontsize=9, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Flow W (Z component)
        ax4 = fig.add_subplot(gs[idx, 4])
        w = flow[2, d_mid]
        vmax_w = max(abs(w.min()), abs(w.max()), 0.1)
        im4 = ax4.imshow(w, cmap='RdBu_r', vmin=-vmax_w, vmax=vmax_w)
        ax4.set_title(f'Flow W\n[{w.min():.2f}, {w.max():.2f}]',
                     fontsize=9, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # Add row label with key info
        fig.text(0.01, 1 - (idx + 0.5) / len(samples),
                f'#{idx}\nBeads: {num_beads}\nType: {deform_type}\nMag: {flow_mag_mean:.2f}±{flow_mag_max:.2f}',
                va='center', fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('V3 Dataset Preview: Vol0, Vol1, Flow (U, V, W)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def print_summary_stats(data_dir):
    """Print summary statistics of generated samples."""
    train_dir = Path(data_dir) / 'train'
    metadata_dir = train_dir / 'metadata'

    if not metadata_dir.exists():
        print("Error: Metadata directory not found")
        return

    # Collect statistics
    bead_counts = []
    deform_types = []
    flow_mags = []

    for meta_file in sorted(metadata_dir.glob('sample_*.json')):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        bead_counts.append(meta['bead_stats']['num_beads'])
        deform_types.append(meta['deform_type'])
        flow_mags.append(meta['flow_stats']['magnitude_mean'])

    print("\n" + "="*70)
    print("V3 Dataset Preview Summary")
    print("="*70)
    print(f"\nTotal samples: {len(bead_counts)}")

    print(f"\nParticle count range:")
    print(f"  Min: {min(bead_counts)}")
    print(f"  Max: {max(bead_counts)}")
    print(f"  Mean: {np.mean(bead_counts):.1f} ± {np.std(bead_counts):.1f}")

    print(f"\nDeformation types:")
    from collections import Counter
    type_counts = Counter(deform_types)
    for dtype, count in sorted(type_counts.items()):
        percentage = 100 * count / len(deform_types)
        print(f"  {dtype:10s}: {count:2d} ({percentage:5.1f}%)")

    print(f"\nFlow magnitude (mean per sample):")
    print(f"  Min: {min(flow_mags):.3f}")
    print(f"  Max: {max(flow_mags):.3f}")
    print(f"  Mean: {np.mean(flow_mags):.3f} ± {np.std(flow_mags):.3f}")
    print("="*70 + "\n")


def main():
    print("="*70)
    print("V3 Dataset Preview Generator")
    print("="*70)

    # Paths
    base_config_path = project_root / 'configs' / 'data_generation' / 'confocal_particles_32_v3.yaml'
    temp_dir = project_root / 'temp_v3_preview'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Create preview config
    print(f"\n1. Creating preview config with 12 samples...")
    temp_config_path, config = create_preview_config(base_config_path, temp_dir, num_samples=12)
    print(f"   ✓ Config saved to: {temp_config_path}")

    # Generate data
    print(f"\n2. Generating preview data...")
    print(f"   Output directory: {temp_dir}")
    print(f"   This will take a few minutes...\n")

    generator = DatasetGenerator(str(temp_config_path))
    generator.generate_split('train')

    print(f"\n   ✓ Data generation complete!")

    # Print summary statistics
    print_summary_stats(temp_dir)

    # Create visualization
    print(f"\n3. Creating multi-sample visualization...")
    viz_output = temp_dir / 'v3_preview_summary.png'
    visualize_multi_samples(temp_dir, viz_output, num_samples=10)

    print("\n" + "="*70)
    print("Preview Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  Data directory: {temp_dir}")
    print(f"  Visualization:  {viz_output}")
    print(f"\nNext steps:")
    print(f"  1. Open the visualization to check if v3 meets your expectations")
    print(f"  2. If satisfied, generate the full dataset using run.bat")
    print(f"  3. The temp directory can be safely deleted")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
