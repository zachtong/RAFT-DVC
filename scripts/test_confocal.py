"""
Test/Evaluation script for RAFT-DVC on synthetic confocal dataset.

Usage:
    python scripts/test_confocal.py --checkpoint results/confocal_baseline/checkpoint_best.pth
    python scripts/test_confocal.py --checkpoint results/confocal_baseline/checkpoint_best.pth --save_predictions
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.core import RAFTDVC
from src.data import VolumePairDataset
from torch.utils.data import DataLoader


def compute_metrics(flow_pred, flow_gt):
    """
    Compute various metrics for flow prediction.

    Args:
        flow_pred: Predicted flow (3, D, H, W) or (B, 3, D, H, W)
        flow_gt: Ground truth flow

    Returns:
        metrics: Dict with EPE, outlier percentages, etc.
    """
    # Compute EPE (End-Point Error)
    epe = torch.sqrt(torch.sum((flow_pred - flow_gt)**2, dim=-4))  # Assume channel dim is -4

    metrics = {
        'EPE': epe.mean().item(),
        'EPE_std': epe.std().item(),
        'EPE_median': epe.median().item(),
        '1px': (epe < 1).float().mean().item() * 100,  # Percentage
        '3px': (epe < 3).float().mean().item() * 100,
        '5px': (epe < 5).float().mean().item() * 100,
    }

    return metrics, epe


def visualize_prediction(vol0, vol1, flow_pred, flow_gt, epe, output_path, sample_idx):
    """
    Create visualization comparing prediction and ground truth.

    Args:
        vol0: Reference volume (1, D, H, W) or (D, H, W)
        vol1: Deformed volume
        flow_pred: Predicted flow (3, D, H, W)
        flow_gt: Ground truth flow (3, D, H, W)
        epe: End-point error map (D, H, W)
        output_path: Path to save figure
        sample_idx: Sample index for title
    """
    # Remove batch dimension if present
    if vol0.ndim == 4:
        vol0 = vol0[0]
        vol1 = vol1[0]

    if vol0.ndim == 4:  # Has channel dimension
        vol0 = vol0[0]
        vol1 = vol1[0]

    # Convert to numpy and move to CPU
    vol0 = vol0.cpu().numpy() if torch.is_tensor(vol0) else vol0
    vol1 = vol1.cpu().numpy() if torch.is_tensor(vol1) else vol1
    flow_pred = flow_pred.cpu().numpy() if torch.is_tensor(flow_pred) else flow_pred
    flow_gt = flow_gt.cpu().numpy() if torch.is_tensor(flow_gt) else flow_gt
    epe = epe.cpu().numpy() if torch.is_tensor(epe) else epe

    # Compute flow magnitudes
    flow_pred_mag = np.sqrt(np.sum(flow_pred**2, axis=0))
    flow_gt_mag = np.sqrt(np.sum(flow_gt**2, axis=0))

    # Central slices
    D, H, W = vol0.shape
    z_mid = D // 2

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Volumes
    axes[0, 0].imshow(vol0[z_mid], cmap='gray')
    axes[0, 0].set_title('Vol0 (Reference)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vol1[z_mid], cmap='gray')
    axes[0, 1].set_title('Vol1 (Deformed)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.abs(vol1[z_mid] - vol0[z_mid]), cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')

    axes[0, 3].axis('off')

    # Row 2: Flow magnitude
    im1 = axes[1, 0].imshow(flow_gt_mag[z_mid], cmap='jet')
    axes[1, 0].set_title('GT Flow Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(flow_pred_mag[z_mid], cmap='jet')
    axes[1, 1].set_title('Predicted Flow Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    im3 = axes[1, 2].imshow(epe[z_mid], cmap='hot')
    axes[1, 2].set_title('EPE (Error Map)')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    # Histogram of EPE
    axes[1, 3].hist(epe.flatten(), bins=50, color='red', alpha=0.7)
    axes[1, 3].set_xlabel('EPE (voxels)')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].set_title('EPE Distribution')
    axes[1, 3].grid(alpha=0.3)

    # Row 3: Flow components
    vmax = max(np.abs(flow_gt).max(), np.abs(flow_pred).max())

    im4 = axes[2, 0].imshow(flow_gt[0, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, 0].set_title('GT Flow Z')
    axes[2, 0].axis('off')
    plt.colorbar(im4, ax=axes[2, 0], fraction=0.046)

    im5 = axes[2, 1].imshow(flow_pred[0, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, 1].set_title('Pred Flow Z')
    axes[2, 1].axis('off')
    plt.colorbar(im5, ax=axes[2, 1], fraction=0.046)

    im6 = axes[2, 2].imshow(flow_gt[1, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, 2].set_title('GT Flow Y')
    axes[2, 2].axis('off')
    plt.colorbar(im6, ax=axes[2, 2], fraction=0.046)

    im7 = axes[2, 3].imshow(flow_pred[1, z_mid], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2, 3].set_title('Pred Flow Y')
    axes[2, 3].axis('off')
    plt.colorbar(im7, ax=axes[2, 3], fraction=0.046)

    plt.suptitle(f'Sample {sample_idx} - EPE: {epe.mean():.3f} voxels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def test_model(model, test_loader, device, args):
    """
    Test model on test set.

    Args:
        model: RAFT-DVC model
        test_loader: DataLoader for test set
        device: torch device
        args: Command line arguments

    Returns:
        results: Dict with aggregate metrics
    """
    model.eval()

    all_metrics = []
    sample_idx = 0

    # Create output directory for visualizations
    if args.save_visualizations:
        vis_dir = Path(args.output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory for predictions
    if args.save_predictions:
        pred_dir = Path(args.output_dir) / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing on {len(test_loader)} samples...")

    for batch in tqdm(test_loader, desc="Testing"):
        vol0 = batch['vol0'].to(device)
        vol1 = batch['vol1'].to(device)
        flow_gt = batch['flow'].to(device)
        filename = batch['filename'][0]  # Batch size 1

        # Run inference
        _, flow_pred = model(vol0, vol1, iters=args.iters, test_mode=True)

        # Compute metrics
        metrics, epe = compute_metrics(flow_pred, flow_gt)
        all_metrics.append(metrics)

        # Save visualization
        if args.save_visualizations and sample_idx < args.num_visualize:
            vis_path = vis_dir / f'{filename}.png'
            visualize_prediction(
                vol0[0], vol1[0], flow_pred[0], flow_gt[0], epe[0],
                vis_path, filename
            )

        # Save prediction
        if args.save_predictions:
            pred_path = pred_dir / f'{filename}.npy'
            np.save(pred_path, flow_pred[0].cpu().numpy())

        sample_idx += 1

    # Aggregate metrics
    results = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Test RAFT-DVC on confocal dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default='data/synthetic_confocal/test',
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as checkpoint dir)')
    parser.add_argument('--iters', type=int, default=24,
                       help='Number of refinement iterations')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predicted flow fields')
    parser.add_argument('--save_visualizations', action='store_true', default=True,
                       help='Save visualization images')
    parser.add_argument('--num_visualize', type=int, default=10,
                       help='Number of samples to visualize')
    args = parser.parse_args()

    # Output directory
    if args.output_dir is None:
        args.output_dir = Path(args.checkpoint).parent / 'test_results'
    else:
        args.output_dir = Path(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model
    model = RAFTDVC()  # Will use default config or config from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load test dataset
    print(f"Loading test data from: {args.test_dir}")
    test_dataset = VolumePairDataset(
        root_dir=args.test_dir,
        augment=False,
        patch_size=None,
        has_flow=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")

    # Run testing
    results = test_model(model, test_loader, device, args)

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    for metric, values in results.items():
        print(f"{metric:10s}: {values['mean']:8.4f} ± {values['std']:.4f} "
              f"(min: {values['min']:.4f}, max: {values['max']:.4f})")

    # Save results to JSON
    results_path = args.output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    if args.save_visualizations:
        print(f"✓ Visualizations saved to: {args.output_dir / 'visualizations'}")

    if args.save_predictions:
        print(f"✓ Predictions saved to: {args.output_dir / 'predictions'}")


if __name__ == '__main__':
    main()
