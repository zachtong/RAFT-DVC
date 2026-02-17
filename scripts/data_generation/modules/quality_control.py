"""
Quality control tools for generated synthetic data.
"""

import numpy as np
from pathlib import Path

# Optional matplotlib import for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Optional PyVista import for 3D rendering
try:
    import pyvista  # noqa: F401
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


class QualityChecker:
    """Quality control for generated samples."""

    def __init__(self, config):
        """
        Args:
            config: Configuration dict for quality control
        """
        self.config = config

    def check_sample(self, sample_data):
        """
        Check if a generated sample meets quality criteria.

        Args:
            sample_data: dict with vol0, vol1, flow, metadata

        Returns:
            passed: bool - True if sample passes all checks
            issues: list of str - list of quality issues found
        """
        issues = []

        # 1. Check visible beads ratio
        if 'warp_stats' in sample_data['metadata']:
            visible_ratio = sample_data['metadata']['warp_stats']['visible_ratio']
            min_ratio = self.config['min_visible_beads_ratio']

            if visible_ratio < min_ratio:
                issues.append(
                    f"Too few visible beads after warping: "
                    f"{visible_ratio:.2%} < {min_ratio:.2%}"
                )

        # 2. Check displacement magnitude
        flow = sample_data['flow']
        flow_mag = np.sqrt(np.sum(flow**2, axis=0))
        max_disp = np.max(flow_mag)
        volume_size = np.min(flow.shape[1:])  # Smallest dimension

        max_ratio = self.config['max_displacement_ratio']
        if max_disp > max_ratio * volume_size:
            issues.append(
                f"Excessive displacement: {max_disp:.1f} voxels "
                f"(> {max_ratio:.1%} of volume size)"
            )

        # 3. Check for NaN or Inf
        for key in ['vol0', 'vol1', 'flow']:
            if np.any(~np.isfinite(sample_data[key])):
                issues.append(f"Found NaN or Inf in {key}")

        # 4. Check intensity range
        for key in ['vol0', 'vol1']:
            vol = sample_data[key]
            if np.min(vol) < 0 or np.max(vol) > 1.5:
                issues.append(
                    f"{key} intensity out of range: "
                    f"[{np.min(vol):.3f}, {np.max(vol):.3f}]"
                )

        # 5. Check if volumes are not all zeros
        for key in ['vol0', 'vol1']:
            if np.max(sample_data[key]) < 0.01:
                issues.append(f"{key} is nearly empty (max intensity < 0.01)")

        passed = len(issues) == 0
        return passed, issues

    def visualize_sample(self, sample_data, output_path, sample_idx):
        """
        Create visualization for quality inspection.

        Args:
            sample_data: dict with vol0, vol1, flow, metadata
            output_path: Path to save visualization
            sample_idx: Sample index (for title)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping visualization")
            return

        vol0 = sample_data['vol0']
        vol1 = sample_data['vol1']
        flow = sample_data['flow']

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # Central slices
        D, H, W = vol0.shape
        z_mid, y_mid, x_mid = D // 2, H // 2, W // 2

        # Row 1: Volume slices
        # Vol0 XY
        ax1 = plt.subplot(3, 4, 1)
        ax1.imshow(vol0[z_mid], cmap='gray')
        ax1.set_title(f'Vol0 - XY (z={z_mid})')
        ax1.axis('off')

        # Vol0 XZ
        ax2 = plt.subplot(3, 4, 2)
        ax2.imshow(vol0[:, y_mid, :], cmap='gray', aspect='auto')
        ax2.set_title(f'Vol0 - XZ (y={y_mid})')
        ax2.axis('off')

        # Vol1 XY
        ax3 = plt.subplot(3, 4, 3)
        ax3.imshow(vol1[z_mid], cmap='gray')
        ax3.set_title(f'Vol1 - XY (z={z_mid})')
        ax3.axis('off')

        # Vol1 XZ
        ax4 = plt.subplot(3, 4, 4)
        ax4.imshow(vol1[:, y_mid, :], cmap='gray', aspect='auto')
        ax4.set_title(f'Vol1 - XZ (y={y_mid})')
        ax4.axis('off')

        # Row 2: Flow magnitude and components
        flow_mag = np.sqrt(np.sum(flow**2, axis=0))

        # Flow magnitude XY
        ax5 = plt.subplot(3, 4, 5)
        im5 = ax5.imshow(flow_mag[z_mid], cmap='jet')
        ax5.set_title(f'Flow Magnitude - XY')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046)

        # Flow magnitude XZ
        ax6 = plt.subplot(3, 4, 6)
        im6 = ax6.imshow(flow_mag[:, y_mid, :], cmap='jet', aspect='auto')
        ax6.set_title(f'Flow Magnitude - XZ')
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)

        # Flow X component
        ax7 = plt.subplot(3, 4, 7)
        im7 = ax7.imshow(flow[2, z_mid], cmap='RdBu_r', vmin=-10, vmax=10)
        ax7.set_title('Flow X Component')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)

        # Flow Y component
        ax8 = plt.subplot(3, 4, 8)
        im8 = ax8.imshow(flow[1, z_mid], cmap='RdBu_r', vmin=-10, vmax=10)
        ax8.set_title('Flow Y Component')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)

        # Row 3: Histograms and statistics
        # Intensity histogram
        ax9 = plt.subplot(3, 4, 9)
        ax9.hist(vol0.flatten(), bins=50, alpha=0.5, label='Vol0', density=True)
        ax9.hist(vol1.flatten(), bins=50, alpha=0.5, label='Vol1', density=True)
        ax9.set_xlabel('Intensity')
        ax9.set_ylabel('Density')
        ax9.set_title('Intensity Distribution')
        ax9.legend()
        ax9.grid(alpha=0.3)

        # Flow magnitude histogram
        ax10 = plt.subplot(3, 4, 10)
        ax10.hist(flow_mag.flatten(), bins=50, color='blue', alpha=0.7)
        ax10.set_xlabel('Displacement (voxels)')
        ax10.set_ylabel('Frequency')
        ax10.set_title('Flow Magnitude Distribution')
        ax10.grid(alpha=0.3)

        # Statistics text
        ax11 = plt.subplot(3, 4, 11)
        ax11.axis('off')

        stats_text = f"Sample {sample_idx}\n\n"
        stats_text += f"Deformation Type: {sample_data['metadata'].get('deform_type', 'N/A')}\n\n"

        if 'flow_stats' in sample_data['metadata']:
            fs = sample_data['metadata']['flow_stats']
            stats_text += f"Flow Statistics:\n"
            stats_text += f"  Mean: {fs['magnitude_mean']:.2f} voxels\n"
            stats_text += f"  Std:  {fs['magnitude_std']:.2f} voxels\n"
            stats_text += f"  Max:  {fs['magnitude_max']:.2f} voxels\n\n"

        if 'bead_stats' in sample_data['metadata']:
            bs = sample_data['metadata']['bead_stats']
            stats_text += f"Bead Statistics:\n"
            stats_text += f"  Count: {bs['num_beads']}\n"
            stats_text += f"  Radius: {bs['radius_mean']:.2f}±{bs['radius_std']:.2f}\n"
            stats_text += f"  Vol. Frac: {bs['volume_fraction']:.3%}\n\n"

        if 'warp_stats' in sample_data['metadata']:
            ws = sample_data['metadata']['warp_stats']
            stats_text += f"Warping:\n"
            stats_text += f"  Visible: {ws['visible_ratio']:.1%}\n"

        ax11.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                 family='monospace')

        # Quality check result
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')

        passed, issues = self.check_sample(sample_data)

        if passed:
            result_text = "✓ PASSED\n\n"
            result_color = 'green'
        else:
            result_text = "✗ FAILED\n\n"
            result_color = 'red'
            result_text += "Issues:\n"
            for issue in issues:
                result_text += f"• {issue}\n"

        ax12.text(0.1, 0.5, result_text, fontsize=10, verticalalignment='center',
                 color=result_color, family='monospace')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_3d_sample(self, sample_data, output_path, sample_idx):
        """
        Create 3D volume rendering for quality inspection using PyVista.

        Renders vol0 and vol1 side-by-side as 3D volumes, reusing
        the existing render_side_by_side_pyvista function.

        Args:
            sample_data: dict with vol0, vol1, flow, metadata
            output_path: Path to save 3D rendering image
            sample_idx: Sample index (for logging)
        """
        if not HAS_PYVISTA:
            print("Warning: PyVista not available, skipping 3D visualization")
            return

        try:
            # Import the existing rendering function
            import sys
            project_root = Path(__file__).parent.parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from src.visualization.side_by_side_render import render_side_by_side_pyvista

            vol0 = sample_data['vol0']
            vol1 = sample_data['vol1']

            render_side_by_side_pyvista(
                volume=vol0,            # placeholder base layer
                scalar_left=vol0,
                scalar_right=vol1,
                save_path=str(output_path),
                volume_opacity=0,       # hide base layer, show scalars only
                left_cmap='gray',
                right_cmap='gray',
                left_opacity=1.0,
                left_opacity_mode='linear',
                right_opacity=1.0,
                right_opacity_mode='linear',
                left_title='Vol0 (Reference)',
                right_title='Vol1 (Deformed)',
                left_unit='a.u.',
                right_unit='a.u.',
                background_color='white',
                show_axes=True,
                show_scalar_bar=True,
                scalar_bar_vertical=True,
                show_bounds=True,
                bounds_color='gray',
                bounds_width=1.0,
                camera_elevation=30.0,
                camera_azimuth=45.0,
                zoom=0.7,
                window_size=(2560, 1080),
            )
            print(f"  3D rendering saved: {output_path}")

        except Exception as e:
            print(f"Warning: 3D rendering failed for sample {sample_idx}: {e}")


def analyze_dataset(dataset_path, num_samples=None):
    """
    Analyze statistics across the entire dataset.

    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to analyze (None = all)

    Returns:
        stats: dict with dataset-level statistics
    """
    from pathlib import Path
    import json

    dataset_path = Path(dataset_path)

    # Collect statistics from all samples
    flow_mags = []
    bead_counts = []
    visible_ratios = []
    snr_values = []

    # Iterate through samples
    train_dir = dataset_path / 'train'
    flow_files = sorted((train_dir / 'flow').glob('*.npy'))

    if num_samples is not None:
        flow_files = flow_files[:num_samples]

    for flow_file in flow_files:
        sample_idx = flow_file.stem

        # Load flow
        flow = np.load(flow_file)
        flow_mag = np.sqrt(np.sum(flow**2, axis=0))
        flow_mags.append(np.mean(flow_mag))

        # Load metadata if exists
        meta_file = train_dir / 'metadata' / f'{sample_idx}.json'
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            if 'bead_stats' in meta:
                bead_counts.append(meta['bead_stats']['num_beads'])

            if 'warp_stats' in meta:
                visible_ratios.append(meta['warp_stats']['visible_ratio'])

            if 'imaging_vol0' in meta:
                snr_values.append(meta['imaging_vol0']['snr_db'])

    # Compute aggregate statistics
    stats = {
        'num_samples_analyzed': len(flow_files),
        'flow_magnitude': {
            'mean': float(np.mean(flow_mags)),
            'std': float(np.std(flow_mags)),
            'min': float(np.min(flow_mags)),
            'max': float(np.max(flow_mags))
        }
    }

    if bead_counts:
        stats['bead_count'] = {
            'mean': float(np.mean(bead_counts)),
            'std': float(np.std(bead_counts)),
            'min': int(np.min(bead_counts)),
            'max': int(np.max(bead_counts))
        }

    if visible_ratios:
        stats['visible_ratio'] = {
            'mean': float(np.mean(visible_ratios)),
            'std': float(np.std(visible_ratios)),
            'min': float(np.min(visible_ratios)),
            'max': float(np.max(visible_ratios))
        }

    if snr_values:
        stats['snr_db'] = {
            'mean': float(np.mean(snr_values)),
            'std': float(np.std(snr_values)),
            'min': float(np.min(snr_values)),
            'max': float(np.max(snr_values))
        }

    return stats
