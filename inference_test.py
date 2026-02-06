"""
Quick inference script for testing trained RAFT-DVC model.

Usage:
    python inference_test.py --checkpoint outputs/training/confocal_baseline/checkpoint_best.pth
    python inference_test.py --checkpoint outputs/training/confocal_baseline/checkpoint_best.pth --sample 5
"""

# Fix OpenMP library conflict (must be set before importing numpy/torch)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import RAFTDVC, RAFTDVCConfig
from src.data import VolumePairDataset


def load_model(checkpoint_path, device='cuda', model_config_path=None):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # If model config file is provided, use it (overrides checkpoint config)
    if model_config_path is not None:
        print(f"Using model config from: {model_config_path}")
        import yaml
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_cfg_file = yaml.safe_load(f)

        model_arch = model_cfg_file['architecture']
        model_config = RAFTDVCConfig(
            encoder_type=model_arch.get('encoder_type', '1/8'),
            feature_dim=model_arch.get('feature_dim', 128),
            context_dim=model_arch.get('context_dim', 64),
            hidden_dim=model_arch.get('hidden_dim', 96),
            corr_levels=model_arch.get('corr_levels', 4),
            corr_radius=model_arch.get('corr_radius', 4),
            encoder_norm=model_arch.get('encoder_norm', 'instance'),
            encoder_dropout=model_arch.get('encoder_dropout', 0.0),
            context_norm=model_arch.get('context_norm', 'none'),
            context_dropout=model_arch.get('context_dropout', 0.0),
            use_sep_conv=model_arch.get('use_sep_conv', True),
            iters=model_arch.get('iters', 12)
        )
    # Priority 1: Check for new format model_config (saved by updated trainer)
    elif 'model_config' in checkpoint:
        print("Loading model config from checkpoint (new format)")
        config = checkpoint['model_config']
        model_config = RAFTDVCConfig(
            encoder_type=config.get('encoder_type', '1/8'),
            feature_dim=config.get('feature_dim', 128),
            context_dim=config.get('context_dim', 64),
            hidden_dim=config.get('hidden_dim', 96),
            corr_levels=config.get('corr_levels', 4),
            corr_radius=config.get('corr_radius', 4),
            encoder_norm=config.get('encoder_norm', 'instance'),
            encoder_dropout=config.get('encoder_dropout', 0.0),
            context_norm=config.get('context_norm', 'none'),
            context_dropout=config.get('context_dropout', 0.0),
            use_sep_conv=config.get('use_sep_conv', True),
            iters=config.get('iters', 12)
        )
    # Priority 2: Legacy format - Get config from checkpoint or use defaults
    elif 'config' in checkpoint:
        config = checkpoint['config']

        # Check if config is new format (flat dict) or old format (nested dict)
        if 'model' in config:
            # Old format: config['model'] contains model parameters
            model_cfg = config['model']
            model_config = RAFTDVCConfig(
                encoder_type=model_cfg.get('encoder_type', '1/8'),
                feature_dim=model_cfg.get('feature_dim', 128),
                context_dim=model_cfg.get('context_dim', 64),
                hidden_dim=model_cfg.get('hidden_dim', 96),
                corr_levels=model_cfg.get('corr_levels', 4),
                corr_radius=model_cfg.get('corr_radius', 4)
            )
        else:
            # New format: config directly contains all model parameters
            model_config = RAFTDVCConfig(
                encoder_type=config.get('encoder_type', '1/8'),
                feature_dim=config.get('feature_dim', 128),
                context_dim=config.get('context_dim', 64),
                hidden_dim=config.get('hidden_dim', 96),
                corr_levels=config.get('corr_levels', 4),
                corr_radius=config.get('corr_radius', 4),
                encoder_norm=config.get('encoder_norm', 'instance'),
                encoder_dropout=config.get('encoder_dropout', 0.0),
                context_norm=config.get('context_norm', 'none'),
                context_dropout=config.get('context_dropout', 0.0),
                use_sep_conv=config.get('use_sep_conv', True),
                iters=config.get('iters', 12)
            )
    else:
        # Default config if not in checkpoint
        print("Warning: No config found in checkpoint, using defaults")
        model_config = RAFTDVCConfig(
            encoder_type='1/8',
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

    print(f"✓ Model loaded successfully")
    print(f"  Encoder type: {model_config.encoder_type}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")

    return model


def compute_metrics(pred_flow, gt_flow):
    """Compute various error metrics."""
    # End-Point Error (L2 distance)
    epe = torch.sqrt(torch.sum((pred_flow - gt_flow)**2, dim=1))

    # L1 error (used in training)
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


@torch.no_grad()
def test_dataset(model, dataset, device='cuda', iters=20, max_samples=None):
    """Test model on entire dataset."""
    print(f"\nTesting on {len(dataset)} samples...")

    all_metrics = []

    # Limit samples if specified
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    for i in tqdm(range(num_samples), desc="Testing"):
        sample = dataset[i]

        # Add batch dimension
        vol0 = sample['vol0'].unsqueeze(0)
        vol1 = sample['vol1'].unsqueeze(0)
        flow_gt = sample['flow'].unsqueeze(0)

        # Run inference
        _, metrics, _ = test_single_sample(
            model, vol0, vol1, flow_gt, device, iters
        )

        all_metrics.append(metrics)

    # Compute average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        avg_metrics[f'{key}_std'] = np.std(values)

    return avg_metrics


def visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path):
    """Visualize a single sample prediction with GT vs Pred comparison."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print(f"\nGenerating visualization...")
    except ImportError as e:
        print(f"Warning: matplotlib not available, skipping visualization: {e}")
        return
    except Exception as e:
        print(f"Error importing matplotlib: {e}")
        return

    try:
        # Move to CPU and remove batch dimension
        vol0 = vol0.cpu().numpy()[0, 0]  # [D, H, W]
        vol1 = vol1.cpu().numpy()[0, 0]
        flow_gt = flow_gt.cpu().numpy()[0]  # [3, D, H, W]
        flow_pred = flow_pred.cpu().numpy()[0]
        epe_map = epe_map.cpu().numpy()[0]  # [D, H, W]

        # Get middle slices
        d_mid = vol0.shape[0] // 2

        # Create figure with 3 rows and 5 columns for comprehensive comparison
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # ========== Row 1: Input volumes and flow magnitudes ==========
        # Volume 0
        axes[0, 0].imshow(vol0[d_mid], cmap='gray')
        axes[0, 0].set_title('Volume 0 (Z-slice)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        # Volume 1
        axes[0, 1].imshow(vol1[d_mid], cmap='gray')
        axes[0, 1].set_title('Volume 1 (Z-slice)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        # Flow magnitude - GT
        flow_mag_gt = np.sqrt(np.sum(flow_gt**2, axis=0))
        flow_mag_pred = np.sqrt(np.sum(flow_pred**2, axis=0))
        mag_max = max(flow_mag_gt.max(), flow_mag_pred.max())

        im_mag_gt = axes[0, 2].imshow(flow_mag_gt[d_mid], cmap='viridis', vmin=0, vmax=mag_max)
        axes[0, 2].set_title(f'GT Flow Magnitude\n(max={flow_mag_gt.max():.2f})',
                             fontsize=12, fontweight='bold', color='green')
        axes[0, 2].axis('off')
        plt.colorbar(im_mag_gt, ax=axes[0, 2], fraction=0.046)

        # Flow magnitude - Pred
        im_mag_pred = axes[0, 3].imshow(flow_mag_pred[d_mid], cmap='viridis', vmin=0, vmax=mag_max)
        axes[0, 3].set_title(f'Pred Flow Magnitude\n(max={flow_mag_pred.max():.2f})',
                             fontsize=12, fontweight='bold', color='blue')
        axes[0, 3].axis('off')
        plt.colorbar(im_mag_pred, ax=axes[0, 3], fraction=0.046)

        # EPE Map (adaptive range)
        epe_max = max(epe_map.max(), 0.1)  # Avoid zero range
        im_epe = axes[0, 4].imshow(epe_map[d_mid], cmap='hot', vmin=0, vmax=epe_max)
        axes[0, 4].set_title(f'EPE Error Map\n(mean={epe_map.mean():.3f}, max={epe_max:.2f})',
                            fontsize=12, fontweight='bold', color='red')
        axes[0, 4].axis('off')
        plt.colorbar(im_epe, ax=axes[0, 4], fraction=0.046)

        # ========== Row 2: Flow X and Y components comparison ==========
        # Adaptive range for flow components (based on actual data range)
        flow_min = min(flow_gt.min(), flow_pred.min())
        flow_max = max(flow_gt.max(), flow_pred.max())
        flow_absmax = max(abs(flow_min), abs(flow_max))
        vmin, vmax = -flow_absmax, flow_absmax

        # Flow X - GT
        im_x_gt = axes[1, 0].imshow(flow_gt[0, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'GT Flow X\n(range=[{flow_gt[0].min():.2f}, {flow_gt[0].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[1, 0].axis('off')
        plt.colorbar(im_x_gt, ax=axes[1, 0], fraction=0.046)

        # Flow X - Pred
        im_x_pred = axes[1, 1].imshow(flow_pred[0, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'Pred Flow X\n(range=[{flow_pred[0].min():.2f}, {flow_pred[0].max():.2f}])',
                            fontsize=12, fontweight='bold', color='blue')
        axes[1, 1].axis('off')
        plt.colorbar(im_x_pred, ax=axes[1, 1], fraction=0.046)

        # Flow Y - GT
        im_y_gt = axes[1, 2].imshow(flow_gt[1, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 2].set_title(f'GT Flow Y\n(range=[{flow_gt[1].min():.2f}, {flow_gt[1].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[1, 2].axis('off')
        plt.colorbar(im_y_gt, ax=axes[1, 2], fraction=0.046)

        # Flow Y - Pred
        im_y_pred = axes[1, 3].imshow(flow_pred[1, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1, 3].set_title(f'Pred Flow Y\n(range=[{flow_pred[1].min():.2f}, {flow_pred[1].max():.2f}])',
                            fontsize=12, fontweight='bold', color='blue')
        axes[1, 3].axis('off')
        plt.colorbar(im_y_pred, ax=axes[1, 3], fraction=0.046)

        # Error X component (adaptive range)
        error_x = np.abs(flow_pred[0] - flow_gt[0])
        error_x_max = max(error_x.max(), 0.1)  # Avoid zero range
        im_err_x = axes[1, 4].imshow(error_x[d_mid], cmap='hot', vmin=0, vmax=error_x_max)
        axes[1, 4].set_title(f'Error X\n(mean={error_x.mean():.3f}, max={error_x_max:.2f})',
                            fontsize=12, fontweight='bold', color='red')
        axes[1, 4].axis('off')
        plt.colorbar(im_err_x, ax=axes[1, 4], fraction=0.046)

        # ========== Row 3: Flow Z component and error analysis ==========
        # Flow Z - GT
        im_z_gt = axes[2, 0].imshow(flow_gt[2, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2, 0].set_title(f'GT Flow Z\n(range=[{flow_gt[2].min():.2f}, {flow_gt[2].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[2, 0].axis('off')
        plt.colorbar(im_z_gt, ax=axes[2, 0], fraction=0.046)

        # Flow Z - Pred
        im_z_pred = axes[2, 1].imshow(flow_pred[2, d_mid], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2, 1].set_title(f'Pred Flow Z\n(range=[{flow_pred[2].min():.2f}, {flow_pred[2].max():.2f}])',
                            fontsize=12, fontweight='bold', color='blue')
        axes[2, 1].axis('off')
        plt.colorbar(im_z_pred, ax=axes[2, 1], fraction=0.046)

        # Error Y component (adaptive range)
        error_y = np.abs(flow_pred[1] - flow_gt[1])
        error_y_max = max(error_y.max(), 0.1)  # Avoid zero range
        im_err_y = axes[2, 2].imshow(error_y[d_mid], cmap='hot', vmin=0, vmax=error_y_max)
        axes[2, 2].set_title(f'Error Y\n(mean={error_y.mean():.3f}, max={error_y_max:.2f})',
                            fontsize=12, fontweight='bold', color='red')
        axes[2, 2].axis('off')
        plt.colorbar(im_err_y, ax=axes[2, 2], fraction=0.046)

        # Error Z component (adaptive range)
        error_z = np.abs(flow_pred[2] - flow_gt[2])
        error_z_max = max(error_z.max(), 0.1)  # Avoid zero range
        im_err_z = axes[2, 3].imshow(error_z[d_mid], cmap='hot', vmin=0, vmax=error_z_max)
        axes[2, 3].set_title(f'Error Z\n(mean={error_z.mean():.3f}, max={error_z_max:.2f})',
                            fontsize=12, fontweight='bold', color='red')
        axes[2, 3].axis('off')
        plt.colorbar(im_err_z, ax=axes[2, 3], fraction=0.046)

        # EPE histogram with statistics
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

        # Add overall title with metrics summary
        metrics_text = (f'EPE: {epe_map.mean():.3f} ± {epe_map.std():.3f} voxels  |  '
                       f'Median: {np.median(epe_map):.3f}  |  '
                       f'<1px: {(epe_map < 1.0).mean()*100:.1f}%  |  '
                       f'<2px: {(epe_map < 2.0).mean()*100:.1f}%  |  '
                       f'<3px: {(epe_map < 3.0).mean()*100:.1f}%')
        fig.suptitle(f'Ground Truth vs Prediction Comparison\n{metrics_text}',
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Visualization saved to: {output_path}")
        print(f"  Full path: {output_path.absolute()}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Test RAFT-DVC model')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/training/confocal_baseline/checkpoint_best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Optional: Path to model config YAML to override checkpoint config')
    parser.add_argument('--data_dir', type=str,
                       default='data/synthetic_confocal',
                       help='Data directory')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to test on')
    parser.add_argument('--sample', type=int, default=None,
                       help='Test single sample by index (None = test all)')
    parser.add_argument('--iters', type=int, default=20,
                       help='Number of refinement iterations')
    parser.add_argument('--output_dir', type=str, default='outputs/inference',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_vis', type=int, default=3,
                       help='Number of samples to visualize in batch mode (-1 = all samples, default=3)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, device, args.model_config)

    # Load dataset
    print(f"\nLoading {args.split} dataset from {args.data_dir}...")
    dataset = VolumePairDataset(
        root_dir=Path(args.data_dir) / args.split,
        augment=False,
        patch_size=None,
        has_flow=True
    )
    print(f"Found {len(dataset)} samples")

    # Test single sample or entire dataset
    if args.sample is not None:
        # Test single sample
        print(f"\n{'='*70}")
        print(f"Testing sample {args.sample}")
        print('='*70)

        sample = dataset[args.sample]
        vol0 = sample['vol0'].unsqueeze(0)
        vol1 = sample['vol1'].unsqueeze(0)
        flow_gt = sample['flow'].unsqueeze(0)

        # Run inference
        flow_pred, metrics, epe_map = test_single_sample(
            model, vol0, vol1, flow_gt, device, args.iters
        )

        # Print metrics
        print(f"\nMetrics for sample {args.sample}:")
        print(f"  EPE: {metrics['EPE_mean']:.4f} ± {metrics['EPE_std']:.4f} voxels")
        print(f"  Median EPE: {metrics['EPE_median']:.4f} voxels")
        print(f"  L1 Error: {metrics['L1_mean']:.4f} voxels")
        print(f"  Component errors - X: {metrics['Error_X']:.4f}, "
              f"Y: {metrics['Error_Y']:.4f}, Z: {metrics['Error_Z']:.4f}")
        print(f"  <1px accuracy: {metrics['1px_accuracy']:.2f}%")
        print(f"  <2px accuracy: {metrics['2px_accuracy']:.2f}%")
        print(f"  <3px accuracy: {metrics['3px_accuracy']:.2f}%")

        # Visualize
        output_path = output_dir / f'sample_{args.sample:04d}_prediction.png'
        visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path)

    else:
        # Test entire dataset
        print(f"\n{'='*70}")
        print(f"Testing entire {args.split} dataset")
        print('='*70)

        avg_metrics = test_dataset(model, dataset, device, args.iters)

        # Print average metrics
        print(f"\n{'='*70}")
        print(f"Average metrics on {args.split} set:")
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
        results_file = output_dir / f'{args.split}_results.txt'
        with open(results_file, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {args.data_dir}/{args.split}\n")
            f.write(f"Iterations: {args.iters}\n")
            f.write(f"\nAverage Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")

        print(f"\n✓ Results saved to: {results_file}")

        # Generate visualizations for samples
        if args.num_vis == -1:
            num_to_visualize = len(dataset)
            print(f"\nGenerating visualizations for ALL {num_to_visualize} samples...")
            print("(This may take a while...)")
        else:
            num_to_visualize = min(args.num_vis, len(dataset))
            print(f"\nGenerating visualizations for {num_to_visualize} samples...")

        for i in tqdm(range(num_to_visualize), desc="Creating visualizations"):
            sample = dataset[i]
            vol0 = sample['vol0'].unsqueeze(0)
            vol1 = sample['vol1'].unsqueeze(0)
            flow_gt = sample['flow'].unsqueeze(0)

            flow_pred, _, epe_map = test_single_sample(
                model, vol0, vol1, flow_gt, device, args.iters
            )

            output_path = output_dir / f'sample_{i:04d}_prediction.png'
            visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path)

        print(f"\n✓ Generated {num_to_visualize} visualizations in: {output_dir}")


if __name__ == '__main__':
    main()
