"""
Generate a single sample to preview what the data looks like.

Usage:
    python generate_single_sample.py
"""

import sys
from pathlib import Path
import yaml
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import generation modules
from scripts.data_generation.modules.deformation import DeformationGenerator, compute_flow_statistics
from scripts.data_generation.modules.beads import BeadGenerator, BeadRenderer, compute_bead_statistics
from scripts.data_generation.modules.warping import ForwardWarper
from scripts.data_generation.modules.imaging import ImagingSimulator

# Optional matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualization")


def generate_and_visualize():
    """Generate a single sample and create visualization."""

    # Load config
    config_path = project_root / 'configs' / 'data_generation' / 'confocal_particles.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*70)
    print("Generating Single Sample Preview")
    print("="*70)

    volume_shape = tuple(config['dataset']['volume_shape'])

    # Initialize generators
    print("\n[1/6] Initializing generators...")
    deform_gen = DeformationGenerator(volume_shape, config['deformation'])
    bead_gen = BeadGenerator(volume_shape, config['particles'])
    renderer = BeadRenderer(volume_shape)
    warper = ForwardWarper(volume_shape)
    imaging = ImagingSimulator(config['imaging'])

    # Generate deformation
    print("[2/6] Generating deformation field...")
    flow, deform_meta = deform_gen.generate()
    flow_stats = compute_flow_statistics(flow)

    print(f"  Deformation type: {deform_meta['type']}")
    print(f"  Flow magnitude: mean={flow_stats['magnitude_mean']:.2f}, "
          f"max={flow_stats['magnitude_max']:.2f} voxels")

    # Generate beads
    print("[3/6] Generating beads...")
    beads_vol0 = bead_gen.generate_beads()
    print(f"  Number of beads: {len(beads_vol0['positions'])}")

    # Render vol0
    print("[4/6] Rendering vol0...")
    vol0_clean = renderer.render(beads_vol0)

    # Warp beads
    print("[5/6] Warping beads and rendering vol1...")
    beads_vol1, num_visible = warper.warp_beads(beads_vol0, flow)
    vol1_clean = renderer.render(beads_vol1)
    print(f"  Visible beads after warping: {num_visible}/{len(beads_vol0['positions'])}")

    # Apply imaging effects
    print("[6/6] Applying imaging effects (PSF, noise)...")
    vol0, _ = imaging.apply_imaging_effects(vol0_clean, apply_photobleaching=False)
    vol1, _ = imaging.apply_imaging_effects(vol1_clean, apply_photobleaching=True)

    print("\n" + "="*70)
    print("Sample Generation Complete!")
    print("="*70)

    # Print statistics
    print("\n[Statistics]")
    print(f"  Volume shape: {vol0.shape}")
    print(f"  Vol0 intensity: [{vol0.min():.3f}, {vol0.max():.3f}], mean={vol0.mean():.3f}")
    print(f"  Vol1 intensity: [{vol1.min():.3f}, {vol1.max():.3f}], mean={vol1.mean():.3f}")
    print(f"  Flow magnitude: mean={flow_stats['magnitude_mean']:.2f} voxels, "
          f"max={flow_stats['magnitude_max']:.2f} voxels")

    # Save data
    output_dir = project_root / 'data' / 'preview_sample'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Saving to {output_dir}]")
    np.save(output_dir / 'vol0.npy', vol0)
    np.save(output_dir / 'vol1.npy', vol1)
    np.save(output_dir / 'flow.npy', flow)
    print("  ✓ Saved: vol0.npy, vol1.npy, flow.npy")

    # Create visualization
    if HAS_MATPLOTLIB:
        print("\n[Creating Visualization]")
        create_visualization(vol0, vol1, flow, output_dir / 'preview.png', deform_meta['type'])
    else:
        print("\n[Skipping Visualization - matplotlib not available]")
        print("  Install matplotlib to see visualizations: pip install matplotlib")

    print("\n" + "="*70)
    print("✓ All done!")
    print(f"  Data saved to: {output_dir}")
    if HAS_MATPLOTLIB:
        print(f"  Visualization: {output_dir / 'preview.png'}")
    print("="*70)


def create_visualization(vol0, vol1, flow, output_path, deform_type):
    """Create comprehensive visualization."""

    # Compute flow magnitude
    flow_mag = np.sqrt(np.sum(flow**2, axis=0))

    # Get central slices
    D, H, W = vol0.shape
    z_mid = D // 2
    y_mid = H // 2

    # Create figure
    fig = plt.figure(figsize=(20, 12))

    # Row 1: Z-slice volumes
    ax1 = plt.subplot(3, 5, 1)
    ax1.imshow(vol0[z_mid], cmap='gray')
    ax1.set_title('Vol0 (Reference)\nZ-slice', fontsize=10)
    ax1.axis('off')

    ax2 = plt.subplot(3, 5, 2)
    ax2.imshow(vol1[z_mid], cmap='gray')
    ax2.set_title('Vol1 (Deformed)\nZ-slice', fontsize=10)
    ax2.axis('off')

    ax3 = plt.subplot(3, 5, 3)
    diff = np.abs(vol1[z_mid] - vol0[z_mid])
    ax3.imshow(diff, cmap='hot')
    ax3.set_title('Absolute Difference\nZ-slice', fontsize=10)
    ax3.axis('off')

    ax4 = plt.subplot(3, 5, 4)
    im4 = ax4.imshow(flow_mag[z_mid], cmap='jet')
    ax4.set_title('Flow Magnitude\nZ-slice', fontsize=10)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)

    # Histogram
    ax5 = plt.subplot(3, 5, 5)
    ax5.hist(flow_mag.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax5.axvline(flow_mag.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {flow_mag.mean():.2f}')
    ax5.set_xlabel('Flow Magnitude (voxels)', fontsize=9)
    ax5.set_ylabel('Frequency', fontsize=9)
    ax5.set_title('Flow Distribution', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # Row 2: Y-slice
    ax6 = plt.subplot(3, 5, 6)
    ax6.imshow(vol0[:, y_mid, :], cmap='gray')
    ax6.set_title('Vol0\nY-slice', fontsize=10)
    ax6.axis('off')

    ax7 = plt.subplot(3, 5, 7)
    ax7.imshow(vol1[:, y_mid, :], cmap='gray')
    ax7.set_title('Vol1\nY-slice', fontsize=10)
    ax7.axis('off')

    ax8 = plt.subplot(3, 5, 8)
    ax8.imshow(np.abs(vol1[:, y_mid, :] - vol0[:, y_mid, :]), cmap='hot')
    ax8.set_title('Absolute Difference\nY-slice', fontsize=10)
    ax8.axis('off')

    ax9 = plt.subplot(3, 5, 9)
    im9 = ax9.imshow(flow_mag[:, y_mid, :], cmap='jet')
    ax9.set_title('Flow Magnitude\nY-slice', fontsize=10)
    ax9.axis('off')
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    # Statistics text
    ax10 = plt.subplot(3, 5, 10)
    ax10.axis('off')
    stats_text = f"""
    STATISTICS

    Deformation: {deform_type}

    Flow Magnitude:
      Mean:  {flow_mag.mean():6.2f} voxels
      Std:   {flow_mag.std():6.2f} voxels
      Max:   {flow_mag.max():6.2f} voxels

    Intensity:
      Vol0: [{vol0.min():.2f}, {vol0.max():.2f}]
      Vol1: [{vol1.min():.2f}, {vol1.max():.2f}]

    Quality:
    """

    # Add quality assessment
    if flow_mag.mean() < 2:
        stats_text += "  ⚠ Flow too small\n"
    elif flow_mag.mean() > 7:
        stats_text += "  ⚠ Flow too large\n"
    else:
        stats_text += "  ✓ Flow reasonable\n"

    if flow_mag.max() > 20:
        stats_text += "  ⚠ Max flow very large"
    elif flow_mag.max() > 15:
        stats_text += "  ⚠ Max flow large"
    else:
        stats_text += "  ✓ Max flow OK"

    ax10.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
              verticalalignment='center')

    # Row 3: Flow components
    vmax = max(np.abs(flow).max(), 0.1)

    ax11 = plt.subplot(3, 5, 11)
    im11 = ax11.imshow(flow[0, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax11.set_title('Flow Z-component', fontsize=10)
    ax11.axis('off')
    plt.colorbar(im11, ax=ax11, fraction=0.046)

    ax12 = plt.subplot(3, 5, 12)
    im12 = ax12.imshow(flow[1, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax12.set_title('Flow Y-component', fontsize=10)
    ax12.axis('off')
    plt.colorbar(im12, ax=ax12, fraction=0.046)

    ax13 = plt.subplot(3, 5, 13)
    im13 = ax13.imshow(flow[2, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax13.set_title('Flow X-component', fontsize=10)
    ax13.axis('off')
    plt.colorbar(im13, ax=ax13, fraction=0.046)

    # MIP
    ax14 = plt.subplot(3, 5, 14)
    mip0 = vol0.max(axis=0)
    ax14.imshow(mip0, cmap='gray')
    ax14.set_title('Vol0 MIP\n(Max Projection)', fontsize=10)
    ax14.axis('off')

    ax15 = plt.subplot(3, 5, 15)
    flow_mip = flow_mag.max(axis=0)
    im15 = ax15.imshow(flow_mip, cmap='jet')
    ax15.set_title('Flow Magnitude MIP', fontsize=10)
    ax15.axis('off')
    plt.colorbar(im15, ax=ax15, fraction=0.046)

    plt.suptitle(f'Preview Sample - Deformation: {deform_type}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Visualization saved to: {output_path}")


if __name__ == '__main__':
    generate_and_visualize()
