"""
Test RAFT-DVC model on Lehu's VolRAFT format dataset.

Usage:
    python test_lehu_data.py --sample 0
    python test_lehu_data.py --all
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core import RAFTDVC, RAFTDVCConfig
from src.data import VolRAFTDataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = RAFTDVCConfig(
            feature_dim=config['model']['feature_dim'],
            context_dim=config['model']['context_dim'],
            hidden_dim=config['model']['hidden_dim'],
            corr_levels=config['model']['corr_levels'],
            corr_radius=config['model']['corr_radius']
        )
    else:
        # Default config if not in checkpoint
        model_config = RAFTDVCConfig(
            feature_dim=128,
            context_dim=64,
            hidden_dim=96,
            corr_levels=4,
            corr_radius=4
        )

    # Create and load model
    model = RAFTDVC(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best validation EPE: {checkpoint.get('best_epe', 'N/A'):.4f}")

    return model


def compute_metrics(pred_flow, gt_flow):
    """Compute various error metrics."""
    # End-Point Error (L2 distance)
    epe = torch.sqrt(torch.sum((pred_flow - gt_flow)**2, dim=1))

    # L1 error
    l1_error = torch.sum(torch.abs(pred_flow - gt_flow), dim=1)

    # Component-wise errors
    errors_x = torch.abs(pred_flow[:, 0] - gt_flow[:, 0])
    errors_y = torch.abs(pred_flow[:, 1] - gt_flow[:, 1])
    errors_z = torch.abs(pred_flow[:, 2] - gt_flow[:, 2])

    metrics = {
        'EPE_mean': epe.mean().item(),
        'EPE_median': epe.median().item(),
        'EPE_std': epe.std().item(),
        'L1_mean': l1_error.mean().item(),
        'Error_X': errors_x.mean().item(),
        'Error_Y': errors_y.mean().item(),
        'Error_Z': errors_z.mean().item(),
        '1px_accuracy': (epe < 1.0).float().mean().item() * 100,
        '2px_accuracy': (epe < 2.0).float().mean().item() * 100,
        '3px_accuracy': (epe < 3.0).float().mean().item() * 100,
    }

    return metrics, epe


@torch.no_grad()
def test_single_sample(model, vol0, vol1, flow_gt, device='cuda', iters=20):
    """Test model on a single sample."""
    # Move to device
    vol0 = vol0.to(device)
    vol1 = vol1.to(device)
    flow_gt = flow_gt.to(device)

    # Run inference
    _, flow_pred = model(vol0, vol1, iters=iters, test_mode=True)

    # Compute metrics
    metrics, epe_map = compute_metrics(flow_pred, flow_gt)

    return flow_pred, metrics, epe_map


def visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path, meta=None):
    """Visualize a single sample prediction."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    # Move to CPU and remove batch dimension
    vol0 = vol0.cpu().numpy()[0, 0]  # [D, H, W]
    vol1 = vol1.cpu().numpy()[0, 0]
    flow_gt = flow_gt.cpu().numpy()[0]  # [3, D, H, W]
    flow_pred = flow_pred.cpu().numpy()[0]
    epe_map = epe_map.cpu().numpy()[0]  # [D, H, W]

    # Get middle slices
    d_mid = vol0.shape[0] // 2

    # Create figure
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Row 1: Input volumes and flow magnitudes
    axes[0, 0].imshow(vol0[d_mid], cmap='gray')
    axes[0, 0].set_title('Volume 0 (Z-slice)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vol1[d_mid], cmap='gray')
    axes[0, 1].set_title('Volume 1 (Z-slice)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Flow magnitude
    flow_mag_gt = np.sqrt(np.sum(flow_gt**2, axis=0))
    flow_mag_pred = np.sqrt(np.sum(flow_pred**2, axis=0))
    mag_max = max(flow_mag_gt.max(), flow_mag_pred.max())

    im_mag_gt = axes[0, 2].imshow(flow_mag_gt[d_mid], cmap='viridis', vmin=0, vmax=mag_max)
    axes[0, 2].set_title(f'GT Flow Magnitude\n(max={flow_mag_gt.max():.2f})',
                         fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    plt.colorbar(im_mag_gt, ax=axes[0, 2], fraction=0.046)

    im_mag_pred = axes[0, 3].imshow(flow_mag_pred[d_mid], cmap='viridis', vmin=0, vmax=mag_max)
    axes[0, 3].set_title(f'Pred Flow Magnitude\n(max={flow_mag_pred.max():.2f})',
                         fontsize=12, fontweight='bold', color='blue')
    axes[0, 3].axis('off')
    plt.colorbar(im_mag_pred, ax=axes[0, 3], fraction=0.046)

    im_epe = axes[0, 4].imshow(epe_map[d_mid], cmap='hot', vmin=0, vmax=5)
    axes[0, 4].set_title(f'EPE Error Map\n(mean={epe_map.mean():.3f})',
                        fontsize=12, fontweight='bold', color='red')
    axes[0, 4].axis('off')
    plt.colorbar(im_epe, ax=axes[0, 4], fraction=0.046)

    # Row 2: Flow X and Y components
    vmin, vmax = -8, 8

    im_x_gt = axes[1, 0].imshow(flow_gt[0, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('GT Flow X', fontsize=12, fontweight='bold', color='green')
    axes[1, 0].axis('off')
    plt.colorbar(im_x_gt, ax=axes[1, 0], fraction=0.046)

    im_x_pred = axes[1, 1].imshow(flow_pred[0, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Pred Flow X', fontsize=12, fontweight='bold', color='blue')
    axes[1, 1].axis('off')
    plt.colorbar(im_x_pred, ax=axes[1, 1], fraction=0.046)

    im_y_gt = axes[1, 2].imshow(flow_gt[1, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('GT Flow Y', fontsize=12, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    plt.colorbar(im_y_gt, ax=axes[1, 2], fraction=0.046)

    im_y_pred = axes[1, 3].imshow(flow_pred[1, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1, 3].set_title('Pred Flow Y', fontsize=12, fontweight='bold', color='blue')
    axes[1, 3].axis('off')
    plt.colorbar(im_y_pred, ax=axes[1, 3], fraction=0.046)

    error_x = np.abs(flow_pred[0] - flow_gt[0])
    im_err_x = axes[1, 4].imshow(error_x[d_mid], cmap='hot', vmin=0, vmax=3)
    axes[1, 4].set_title(f'Error X\n(mean={error_x.mean():.3f})',
                        fontsize=12, fontweight='bold', color='red')
    axes[1, 4].axis('off')
    plt.colorbar(im_err_x, ax=axes[1, 4], fraction=0.046)

    # Row 3: Flow Z component and error analysis
    im_z_gt = axes[2, 0].imshow(flow_gt[2, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2, 0].set_title('GT Flow Z', fontsize=12, fontweight='bold', color='green')
    axes[2, 0].axis('off')
    plt.colorbar(im_z_gt, ax=axes[2, 0], fraction=0.046)

    im_z_pred = axes[2, 1].imshow(flow_pred[2, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2, 1].set_title('Pred Flow Z', fontsize=12, fontweight='bold', color='blue')
    axes[2, 1].axis('off')
    plt.colorbar(im_z_pred, ax=axes[2, 1], fraction=0.046)

    error_y = np.abs(flow_pred[1] - flow_gt[1])
    im_err_y = axes[2, 2].imshow(error_y[d_mid], cmap='hot', vmin=0, vmax=3)
    axes[2, 2].set_title(f'Error Y\n(mean={error_y.mean():.3f})',
                        fontsize=12, fontweight='bold', color='red')
    axes[2, 2].axis('off')
    plt.colorbar(im_err_y, ax=axes[2, 2], fraction=0.046)

    error_z = np.abs(flow_pred[2] - flow_gt[2])
    im_err_z = axes[2, 3].imshow(error_z[d_mid], cmap='hot', vmin=0, vmax=3)
    axes[2, 3].set_title(f'Error Z\n(mean={error_z.mean():.3f})',
                        fontsize=12, fontweight='bold', color='red')
    axes[2, 3].axis('off')
    plt.colorbar(im_err_z, ax=axes[2, 3], fraction=0.046)

    # EPE histogram
    axes[2, 4].hist(epe_map.flatten(), bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[2, 4].axvline(epe_map.mean(), color='r', linestyle='--', linewidth=2,
                      label=f'Mean: {epe_map.mean():.3f}')
    axes[2, 4].axvline(np.median(epe_map), color='g', linestyle='--', linewidth=2,
                      label=f'Median: {np.median(epe_map):.3f}')
    axes[2, 4].set_xlabel('EPE (voxels)', fontsize=11)
    axes[2, 4].set_ylabel('Frequency', fontsize=11)
    axes[2, 4].set_title('EPE Distribution', fontsize=12, fontweight='bold')
    axes[2, 4].legend(fontsize=10)
    axes[2, 4].grid(True, alpha=0.3)

    # Add title with metrics
    metrics_text = (f'EPE: {epe_map.mean():.3f} ± {epe_map.std():.3f} voxels  |  '
                   f'Median: {np.median(epe_map):.3f}  |  '
                   f'<1px: {(epe_map < 1.0).mean()*100:.1f}%  |  '
                   f'<2px: {(epe_map < 2.0).mean()*100:.1f}%  |  '
                   f'<3px: {(epe_map < 3.0).mean()*100:.1f}%')

    title = f'Ground Truth vs Prediction Comparison (Lehu\'s Data)\n{metrics_text}'
    if meta:
        title = f"{title}\nCase: {meta.get('case_id', 'N/A')}, Max Disp: {meta.get('max_disp_case', 0):.2f}"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test RAFT-DVC on Lehu\'s VolRAFT data')
    parser.add_argument('--data_dir', type=str,
                       default='data/Lehu\'s_data/VolRAFT_Synthetic_Dataset',
                       help='Path to Lehu\'s data directory')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/training/confocal_baseline/checkpoint_best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--sample', type=int, default=None,
                       help='Test single sample by index (None = test all)')
    parser.add_argument('--all', action='store_true',
                       help='Test all samples')
    parser.add_argument('--iters', type=int, default=20,
                       help='Number of refinement iterations')
    parser.add_argument('--output_dir', type=str, default='outputs/lehu_inference',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load dataset
    print(f"\nLoading Lehu's dataset from {args.data_dir}...")
    dataset = VolRAFTDataset(
        root_dir=Path(args.data_dir),
        augment=False,
        patch_size=None,
        has_flow=True
    )
    print(f"Found {len(dataset)} samples")

    # Test single sample or all samples
    if args.sample is not None or not args.all:
        # Test single sample
        sample_idx = args.sample if args.sample is not None else 0
        print(f"\n{'='*70}")
        print(f"Testing sample {sample_idx}")
        print('='*70)

        sample = dataset[sample_idx]
        vol0 = sample['vol0'].unsqueeze(0)
        vol1 = sample['vol1'].unsqueeze(0)
        flow_gt = sample['flow'].unsqueeze(0)
        meta = sample.get('meta', None)

        # Run inference
        flow_pred, metrics, epe_map = test_single_sample(
            model, vol0, vol1, flow_gt, device, args.iters
        )

        # Print metrics
        print(f"\nMetrics for sample {sample_idx}:")
        print(f"  EPE: {metrics['EPE_mean']:.4f} ± {metrics['EPE_std']:.4f} voxels")
        print(f"  Median EPE: {metrics['EPE_median']:.4f} voxels")
        print(f"  L1 Error: {metrics['L1_mean']:.4f} voxels")
        print(f"  Component errors - X: {metrics['Error_X']:.4f}, "
              f"Y: {metrics['Error_Y']:.4f}, Z: {metrics['Error_Z']:.4f}")
        print(f"  <1px accuracy: {metrics['1px_accuracy']:.2f}%")
        print(f"  <2px accuracy: {metrics['2px_accuracy']:.2f}%")
        print(f"  <3px accuracy: {metrics['3px_accuracy']:.2f}%")

        if meta:
            print(f"\nSample metadata:")
            print(f"  Case ID: {meta.get('case_id', 'N/A')}")
            print(f"  Max displacement: {meta.get('max_disp_case', 0):.4f}")
            print(f"  N beads: {meta.get('n_beads_case', 'N/A')}")

        # Visualize
        output_path = output_dir / f'sample_{sample_idx:04d}_prediction.png'
        visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path, meta)

    else:
        # Test all samples
        print(f"\n{'='*70}")
        print(f"Testing all {len(dataset)} samples")
        print('='*70)

        all_metrics = []

        for i in tqdm(range(len(dataset)), desc="Testing"):
            sample = dataset[i]
            vol0 = sample['vol0'].unsqueeze(0)
            vol1 = sample['vol1'].unsqueeze(0)
            flow_gt = sample['flow'].unsqueeze(0)

            # Run inference
            flow_pred, metrics, epe_map = test_single_sample(
                model, vol0, vol1, flow_gt, device, args.iters
            )

            all_metrics.append(metrics)

            # Visualize first few samples
            if i < 3:
                output_path = output_dir / f'sample_{i:04d}_prediction.png'
                meta = sample.get('meta', None)
                visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path, meta)

        # Compute average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

        # Print average metrics
        print(f"\n{'='*70}")
        print(f"Average metrics on Lehu's dataset:")
        print('='*70)
        print(f"  EPE: {avg_metrics['EPE_mean']:.4f} ± {avg_metrics['EPE_mean_std']:.4f} voxels")
        print(f"  Median EPE: {avg_metrics['EPE_median']:.4f} voxels")
        print(f"  L1 Error: {avg_metrics['L1_mean']:.4f} voxels")
        print(f"  Component errors - X: {avg_metrics['Error_X']:.4f}, "
              f"Y: {avg_metrics['Error_Y']:.4f}, Z: {avg_metrics['Error_Z']:.4f}")
        print(f"  <1px accuracy: {avg_metrics['1px_accuracy']:.2f}%")
        print(f"  <2px accuracy: {avg_metrics['2px_accuracy']:.2f}%")
        print(f"  <3px accuracy: {avg_metrics['3px_accuracy']:.2f}%")

        # Save results
        results_file = output_dir / 'lehu_test_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.data_dir}\n")
            f.write(f"Iterations: {args.iters}\n")
            f.write(f"Number of samples: {len(dataset)}\n")
            f.write(f"\nAverage Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")

        print(f"\n✓ Results saved to: {results_file}")


if __name__ == '__main__':
    main()
