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
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import RAFTDVC, RAFTDVCConfig
from src.data import VolumePairDataset
from src.visualization import (
    compute_feature_density,
    auto_render_uncertainty_volume,
)
from viz_helpers import visualize_uncertainty_density_comparison


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

        config_dict = dict(model_cfg_file['architecture'])
        # Merge training-specific fields from checkpoint (e.g. uncertainty_mode)
        if 'model_config' in checkpoint:
            for key in ['uncertainty_mode', 'predict_uncertainty']:
                if key in checkpoint['model_config'] and key not in config_dict:
                    config_dict[key] = checkpoint['model_config'][key]
        model_config = RAFTDVCConfig.from_dict(config_dict)
    # Priority 1: Check for new format model_config (saved by updated trainer)
    elif 'model_config' in checkpoint:
        print("Loading model config from checkpoint (new format)")
        model_config = RAFTDVCConfig.from_dict(checkpoint['model_config'])
    # Priority 2: Legacy format - Get config from checkpoint or use defaults
    elif 'config' in checkpoint:
        config = checkpoint['config']

        # Check if config is new format (flat dict) or old format (nested dict)
        if 'model' in config:
            # Old format: config['model'] contains model parameters
            model_config = RAFTDVCConfig.from_dict(config['model'])
        else:
            model_config = RAFTDVCConfig.from_dict(config)
    else:
        # Default config if not in checkpoint
        print("Warning: No config found in checkpoint, using defaults")
        model_config = RAFTDVCConfig()

    # Create and load model
    model = RAFTDVC(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Encoder type: {model_config.encoder_type}")
    if model_config.predict_uncertainty:
        print(f"  Uncertainty: enabled")
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
    """Test model on a single sample.

    Returns:
        flow_pred, metrics, epe_map, uncertainty_map (or None)
    """
    # Move to device
    vol0 = vol0.to(device)
    vol1 = vol1.to(device)
    flow_gt = flow_gt.to(device)

    # Run inference (test_mode returns 2-tuple or 3-tuple with uncertainty)
    output = model(vol0, vol1, iters=iters, test_mode=True)
    flow_pred = output[1]

    # Extract uncertainty if present
    uncertainty_map = None
    alpha_map = None
    if len(output) == 3:
        unc_raw = output[2]
        if unc_raw.shape[1] == 4:
            # MoL mode: channels 0-2 = log_b2, channel 3 = logit_alpha
            uncertainty_map = torch.exp(unc_raw[:, :3])      # b2
            alpha_map = torch.sigmoid(unc_raw[:, 3:4])       # alpha (confidence)
        else:
            # NLL mode: 3 channels = log_b
            uncertainty_map = torch.exp(unc_raw)              # b = exp(log_b)

    # Compute metrics
    metrics, epe_map = compute_metrics(flow_pred, flow_gt)

    return flow_pred, metrics, epe_map, uncertainty_map, alpha_map


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
        _, metrics, _, _, _ = test_single_sample(
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
        # Per-component adaptive range (GT and Pred share range within each component)
        comp_labels = ['X', 'Y', 'Z']
        comp_vmax = []
        for c in range(3):
            c_absmax = max(
                abs(flow_gt[c].min()), abs(flow_gt[c].max()),
                abs(flow_pred[c].min()), abs(flow_pred[c].max()),
                0.01  # avoid zero range
            )
            comp_vmax.append(c_absmax)

        # Flow X - GT
        im_x_gt = axes[1, 0].imshow(flow_gt[0, d_mid], cmap='RdBu_r',
                                     vmin=-comp_vmax[0], vmax=comp_vmax[0])
        axes[1, 0].set_title(f'GT Flow X\n(range=[{flow_gt[0].min():.2f}, {flow_gt[0].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[1, 0].axis('off')
        plt.colorbar(im_x_gt, ax=axes[1, 0], fraction=0.046)

        # Flow X - Pred
        im_x_pred = axes[1, 1].imshow(flow_pred[0, d_mid], cmap='RdBu_r',
                                       vmin=-comp_vmax[0], vmax=comp_vmax[0])
        axes[1, 1].set_title(f'Pred Flow X\n(range=[{flow_pred[0].min():.2f}, {flow_pred[0].max():.2f}])',
                            fontsize=12, fontweight='bold', color='blue')
        axes[1, 1].axis('off')
        plt.colorbar(im_x_pred, ax=axes[1, 1], fraction=0.046)

        # Flow Y - GT
        im_y_gt = axes[1, 2].imshow(flow_gt[1, d_mid], cmap='RdBu_r',
                                     vmin=-comp_vmax[1], vmax=comp_vmax[1])
        axes[1, 2].set_title(f'GT Flow Y\n(range=[{flow_gt[1].min():.2f}, {flow_gt[1].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[1, 2].axis('off')
        plt.colorbar(im_y_gt, ax=axes[1, 2], fraction=0.046)

        # Flow Y - Pred
        im_y_pred = axes[1, 3].imshow(flow_pred[1, d_mid], cmap='RdBu_r',
                                       vmin=-comp_vmax[1], vmax=comp_vmax[1])
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
        im_z_gt = axes[2, 0].imshow(flow_gt[2, d_mid], cmap='RdBu_r',
                                     vmin=-comp_vmax[2], vmax=comp_vmax[2])
        axes[2, 0].set_title(f'GT Flow Z\n(range=[{flow_gt[2].min():.2f}, {flow_gt[2].max():.2f}])',
                            fontsize=12, fontweight='bold', color='green')
        axes[2, 0].axis('off')
        plt.colorbar(im_z_gt, ax=axes[2, 0], fraction=0.046)

        # Flow Z - Pred
        im_z_pred = axes[2, 1].imshow(flow_pred[2, d_mid], cmap='RdBu_r',
                                       vmin=-comp_vmax[2], vmax=comp_vmax[2])
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


def visualize_uncertainty(vol0, flow_pred, flow_gt, epe_map, uncertainty_map,
                          output_path, edge_crop=0.15):
    """Visualize uncertainty map alongside error for models with UncertaintyHead.

    Generates a separate figure (does not modify the flow visualization).
    Layout: 4x4 grid
      Row 1: Input vol0, EPE error, Uncertainty magnitude, EPE-vs-b_epe scatter
      Row 2: Uncertainty X, Uncertainty Y, Uncertainty Z, Anisotropy map
      Row 3: Error X, Error Y, Error Z, Principal direction map
      Row 4: Scatter b_x vs |err_x|, Scatter b_y vs |err_y|, Scatter b_z vs |err_z|,
             Anisotropy histogram

    The heatmaps show the INTERIOR region only (edges cropped by edge_crop fraction)
    to avoid the boundary-high-uncertainty effect dominating the colorbar.

    Args:
        edge_crop: fraction of each dimension to crop from each side (default 0.15 = 15%)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return

    try:
        # To numpy, remove batch dim
        v0 = vol0.cpu().numpy()[0, 0]           # (D, H, W)
        epe = epe_map.cpu().numpy()[0]           # (D, H, W)
        unc = uncertainty_map.cpu().numpy()[0]   # (3, D, H, W)
        fp = flow_pred.cpu().numpy()[0]          # (3, D, H, W)
        fg = flow_gt.cpu().numpy()[0]            # (3, D, H, W)

        comp_names = ['X', 'Y', 'Z']

        # --- Compute crop boundaries ---
        D, H, W = v0.shape
        cd = max(int(D * edge_crop), 1)
        ch = max(int(H * edge_crop), 1)
        cw = max(int(W * edge_crop), 1)
        ds, de = cd, D - cd
        hs, he = ch, H - ch
        ws, we = cw, W - cw
        d_mid = D // 2

        def crop_slice(arr_3d):
            return arr_3d[d_mid, hs:he, ws:we]

        def crop_comp(arr_4d, c):
            return arr_4d[c, d_mid, hs:he, ws:we]

        def crop_3d(arr_3d):
            return arr_3d[ds:de, hs:he, ws:we]

        def crop_4d_comp(arr_4d, c):
            return arr_4d[c, ds:de, hs:he, ws:we]

        # --- Spearman correlation ---
        def spearman(a, b):
            a_f, b_f = a.flatten(), b.flatten()
            ra = np.empty_like(a_f)
            rb = np.empty_like(b_f)
            ra[np.argsort(a_f)] = np.arange(len(a_f))
            rb[np.argsort(b_f)] = np.arange(len(b_f))
            return np.corrcoef(ra, rb)[0, 1]

        # --- Derived quantities ---
        # b_epe: uncertainty in EPE-compatible scale
        b_epe = np.sqrt((unc ** 2).sum(axis=0))  # (D, H, W)
        # Anisotropy: max(b)/min(b) per voxel (1=isotropic)
        b_max = unc.max(axis=0)
        b_min = np.clip(unc.min(axis=0), 1e-8, None)
        aniso = b_max / b_min  # (D, H, W)
        # Principal direction: argmax over XYZ
        principal = unc.argmax(axis=0)  # (D, H, W), values 0/1/2

        # Correlations (interior 3D block)
        epe_int = crop_3d(epe)
        b_epe_int = crop_3d(b_epe)
        rho_full = spearman(epe, b_epe)
        rho_interior = spearman(epe_int, b_epe_int)

        # Per-component correlations (interior)
        rho_comp = []
        for c in range(3):
            err_c_int = np.abs(crop_4d_comp(fp, c) - crop_4d_comp(fg, c))
            unc_c_int = crop_4d_comp(unc, c)
            rho_comp.append(spearman(err_c_int, unc_c_int))

        fig, axes = plt.subplots(4, 4, figsize=(24, 22))

        # ================================================================
        # Row 1: Overview
        # ================================================================
        # Input (cropped)
        axes[0, 0].imshow(crop_slice(v0), cmap='gray')
        axes[0, 0].set_title('Input vol0 (interior)', fontsize=12,
                             fontweight='bold')
        axes[0, 0].axis('off')

        # EPE error (cropped)
        epe_crop = crop_slice(epe)
        epe_vmax = max(np.percentile(epe_crop, 99), 1e-6)
        im_epe = axes[0, 1].imshow(epe_crop, cmap='hot', vmin=0,
                                    vmax=epe_vmax)
        axes[0, 1].set_title(
            f'EPE Error (mean={epe_crop.mean():.3f})',
            fontsize=12, fontweight='bold', color='red')
        axes[0, 1].axis('off')
        plt.colorbar(im_epe, ax=axes[0, 1], fraction=0.046)

        # b_epe magnitude (cropped) — same scale as EPE
        b_epe_crop = crop_slice(b_epe)
        b_epe_vmax = max(np.percentile(b_epe_crop, 99), 1e-6)
        im_bepe = axes[0, 2].imshow(b_epe_crop, cmap='hot', vmin=0,
                                     vmax=b_epe_vmax)
        axes[0, 2].set_title(
            f'b_epe = \u221a(\u03a3b\u00b2) (mean={b_epe_crop.mean():.3f})',
            fontsize=12, fontweight='bold', color='purple')
        axes[0, 2].axis('off')
        plt.colorbar(im_bepe, ax=axes[0, 2], fraction=0.046)

        # Scatter: EPE vs b_epe (same scale, with y=x diagonal)
        rng = np.random.default_rng(42)
        epe_sc = epe_int.flatten()
        bepe_sc = b_epe_int.flatten()
        n_pts = len(epe_sc)
        if n_pts > 5000:
            idx = rng.choice(n_pts, 5000, replace=False)
            epe_sc = epe_sc[idx]
            bepe_sc = bepe_sc[idx]
        axes[0, 3].scatter(bepe_sc, epe_sc, s=1, alpha=0.3, c='steelblue')
        lim = max(np.percentile(epe_sc, 99), np.percentile(bepe_sc, 99)) * 1.1
        axes[0, 3].plot([0, lim], [0, lim], 'r--', lw=1.5, label='y = x')
        axes[0, 3].set_xlim(0, lim)
        axes[0, 3].set_ylim(0, lim)
        axes[0, 3].set_aspect('equal')
        axes[0, 3].set_xlabel('b_epe', fontsize=11)
        axes[0, 3].set_ylabel('EPE', fontsize=11)
        axes[0, 3].set_title(
            f'EPE vs b_epe (interior)\n'
            f'\u03c1_full={rho_full:.3f}  \u03c1_int={rho_interior:.3f}',
            fontsize=11, fontweight='bold')
        axes[0, 3].legend(fontsize=9)
        axes[0, 3].grid(True, alpha=0.3)

        # ================================================================
        # Row 2: Per-component uncertainty + Anisotropy map
        # ================================================================
        for c in range(3):
            uc = crop_comp(unc, c)
            uc_vmax = max(np.percentile(uc, 99), 1e-6)
            im = axes[1, c].imshow(uc, cmap='hot', vmin=0, vmax=uc_vmax)
            axes[1, c].set_title(
                f'Uncertainty {comp_names[c]} '
                f'(mean={uc.mean():.3f})',
                fontsize=11, fontweight='bold', color='purple')
            axes[1, c].axis('off')
            plt.colorbar(im, ax=axes[1, c], fraction=0.046)

        # Anisotropy map: max(b)/min(b)
        aniso_crop = crop_slice(aniso)
        aniso_vmax = max(np.percentile(aniso_crop, 99), 1.01)
        im_aniso = axes[1, 3].imshow(aniso_crop, cmap='viridis',
                                      vmin=1.0, vmax=aniso_vmax)
        axes[1, 3].set_title(
            f'Anisotropy max(b)/min(b)\n'
            f'(mean={aniso_crop.mean():.2f}, 1=isotropic)',
            fontsize=11, fontweight='bold', color='darkgreen')
        axes[1, 3].axis('off')
        plt.colorbar(im_aniso, ax=axes[1, 3], fraction=0.046)

        # ================================================================
        # Row 3: Per-component error + Principal direction map
        # ================================================================
        for c in range(3):
            err_c = np.abs(crop_comp(fp, c) - crop_comp(fg, c))
            err_c_vmax = max(np.percentile(err_c, 99), 1e-6)
            im = axes[2, c].imshow(err_c, cmap='hot', vmin=0,
                                    vmax=err_c_vmax)
            axes[2, c].set_title(
                f'Error {comp_names[c]} '
                f'(mean={err_c.mean():.3f})',
                fontsize=11, fontweight='bold', color='red')
            axes[2, c].axis('off')
            plt.colorbar(im, ax=axes[2, c], fraction=0.046)

        # Principal direction map (RGB: R=X, G=Y, B=Z)
        principal_crop = crop_slice(principal)
        # Create RGB image
        dir_colors = np.array([[0.9, 0.2, 0.2],    # X = red
                               [0.2, 0.8, 0.2],    # Y = green
                               [0.2, 0.4, 0.9]])   # Z = blue
        rgb = dir_colors[principal_crop]  # (H', W', 3)
        axes[2, 3].imshow(rgb)
        # Count percentages
        total_px = principal_crop.size
        pct = [(principal_crop == c).sum() / total_px * 100 for c in range(3)]
        axes[2, 3].set_title(
            f'Principal Direction\n'
            f'X={pct[0]:.0f}% Y={pct[1]:.0f}% Z={pct[2]:.0f}%',
            fontsize=11, fontweight='bold', color='darkgreen')
        axes[2, 3].axis('off')

        # ================================================================
        # Row 4: Per-component scatter (b_c vs |err_c|) + Anisotropy hist
        # ================================================================
        comp_colors = ['#d62728', '#2ca02c', '#1f77b4']  # R, G, B
        for c in range(3):
            err_c = np.abs(crop_4d_comp(fp, c) - crop_4d_comp(fg, c)).flatten()
            unc_c = crop_4d_comp(unc, c).flatten()
            n_c = len(err_c)
            if n_c > 5000:
                idx_c = rng.choice(n_c, 5000, replace=False)
                err_c = err_c[idx_c]
                unc_c = unc_c[idx_c]
            axes[3, c].scatter(unc_c, err_c, s=1, alpha=0.3,
                               c=comp_colors[c])
            lim_c = max(np.percentile(err_c, 99),
                        np.percentile(unc_c, 99)) * 1.1
            axes[3, c].plot([0, lim_c], [0, lim_c], 'k--', lw=1.5,
                            label='y = x')
            axes[3, c].set_xlim(0, lim_c)
            axes[3, c].set_ylim(0, lim_c)
            axes[3, c].set_aspect('equal')
            axes[3, c].set_xlabel(f'b_{comp_names[c]}', fontsize=11)
            axes[3, c].set_ylabel(f'|err_{comp_names[c]}|', fontsize=11)
            axes[3, c].set_title(
                f'{comp_names[c]}: b vs |err|  '
                f'\u03c1={rho_comp[c]:.3f}',
                fontsize=11, fontweight='bold')
            axes[3, c].legend(fontsize=9)
            axes[3, c].grid(True, alpha=0.3)

        # Anisotropy histogram
        aniso_flat = crop_3d(aniso).flatten()
        axes[3, 3].hist(aniso_flat, bins=50, alpha=0.7, edgecolor='black',
                        color='teal')
        axes[3, 3].axvline(np.median(aniso_flat), color='r', ls='--', lw=2,
                           label=f'Median: {np.median(aniso_flat):.2f}')
        axes[3, 3].axvline(1.0, color='gray', ls=':', lw=1.5,
                           label='Isotropic (1.0)')
        axes[3, 3].set_xlabel('Anisotropy ratio', fontsize=11)
        axes[3, 3].set_ylabel('Frequency', fontsize=11)
        axes[3, 3].set_title(
            f'Anisotropy Distribution\n'
            f'(mean={aniso_flat.mean():.2f})',
            fontsize=11, fontweight='bold')
        axes[3, 3].legend(fontsize=9)
        axes[3, 3].grid(True, alpha=0.3)

        fig.suptitle(
            f'Uncertainty Analysis (edges cropped {edge_crop:.0%})  |  '
            f'Spearman \u03c1(EPE, b_epe): '
            f'full={rho_full:.3f}, int={rho_interior:.3f}  |  '
            f'\u03c1 per-comp: X={rho_comp[0]:.3f} Y={rho_comp[1]:.3f} '
            f'Z={rho_comp[2]:.3f}  |  z={d_mid}',
            fontsize=13, fontweight='bold', y=0.998)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  \u2713 Uncertainty visualization saved to: {output_path}")

        # === Intentionally removed - 3D rendering now optional ===
        # See --enable_3d_rendering flag and render_3d_volume_comparison() function

    except Exception as e:
        print(f"  Warning: uncertainty visualization failed: {e}")
        import traceback
        traceback.print_exc()


def load_visualization_config(config_path):
    """Load visualization configuration from YAML file.

    Args:
        config_path: Path to visualization config YAML file

    Returns:
        Dictionary with all visualization parameters
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Flatten nested structure for easy parameter access
    viz_params = {}

    # Camera settings
    viz_params['camera_elevation'] = config['camera']['elevation']
    viz_params['camera_azimuth'] = config['camera']['azimuth']
    viz_params['zoom'] = config['camera']['zoom']

    # Uncertainty settings
    viz_params['unc_opacity'] = config['uncertainty']['opacity']
    viz_params['unc_opacity_mode'] = config['uncertainty']['opacity_mode']
    viz_params['unc_cmap'] = config['uncertainty']['colormap']
    viz_params['unc_unit'] = config['uncertainty']['unit']

    # Density settings
    viz_params['density_opacity'] = config['density']['opacity']
    viz_params['density_opacity_mode'] = config['density']['opacity_mode']
    viz_params['density_cmap'] = config['density']['colormap']
    viz_params['density_unit'] = config['density']['unit']
    viz_params['density_threshold'] = config['density']['threshold']
    viz_params['density_sigma'] = config['density']['sigma']

    # Input volume settings
    viz_params['volume_opacity'] = config['input_volume']['opacity']
    viz_params['volume_unit'] = config['input_volume']['unit']

    # Rendering settings
    viz_params['window_size'] = tuple(config['rendering']['window_size'])
    viz_params['background_color'] = config['rendering']['background_color']

    # Display settings
    viz_params['show_scalar_bar'] = config['display']['show_scalar_bar']
    viz_params['show_volume_scalar_bar'] = config['display']['show_volume_scalar_bar']
    viz_params['scalar_bar_font_size'] = config['display']['scalar_bar_font_size']
    viz_params['scalar_bar_font_family'] = config['display']['scalar_bar_font_family']
    viz_params['scalar_bar_outline'] = config['display']['scalar_bar_outline']
    viz_params['show_axes'] = config['display']['show_axes']
    viz_params['show_bounds'] = config['display']['show_bounds']
    viz_params['bounds_color'] = config['display']['bounds_color']
    viz_params['bounds_width'] = config['display']['bounds_width']

    # Slice settings
    viz_params['show_slice'] = config['slice']['show_slice']
    viz_params['slice_z'] = config['slice']['slice_z']
    viz_params['slice_frame_color'] = config['slice']['frame_color']
    viz_params['slice_frame_width'] = config['slice']['frame_width']

    # Error volume settings (optional - may not exist in all configs)
    if 'error' in config:
        viz_params['error_enabled'] = config['error'].get('enabled', False)
        viz_params['error_opacity'] = config['error'].get('opacity', 1.0)
        viz_params['error_opacity_mode'] = config['error'].get('opacity_mode', 'adaptive')
        viz_params['error_cmap'] = config['error'].get('colormap', 'hot')
        viz_params['error_unit'] = config['error'].get('unit', 'vx')
        viz_params['error_metric'] = config['error'].get('metric', 'epe')
    else:
        viz_params['error_enabled'] = False

    return viz_params


def compute_error_volume(flow_pred, flow_gt, metric='epe'):
    """Compute error volume between predicted and ground truth flow.

    Args:
        flow_pred: (3, D, H, W) predicted flow numpy array
        flow_gt: (3, D, H, W) ground truth flow numpy array
        metric: 'epe' for endpoint error, 'magnitude' for magnitude difference

    Returns:
        error: (D, H, W) error volume
    """
    if metric == 'epe':
        # Endpoint error: ||u_pred - u_gt||_2
        diff = flow_pred - flow_gt
        error = np.sqrt(np.sum(diff ** 2, axis=0))
    elif metric == 'magnitude':
        # Magnitude difference: | ||u_pred||_2 - ||u_gt||_2 |
        mag_pred = np.sqrt(np.sum(flow_pred ** 2, axis=0))
        mag_gt = np.sqrt(np.sum(flow_gt ** 2, axis=0))
        error = np.abs(mag_pred - mag_gt)
    else:
        raise ValueError(f"Unknown error metric: {metric}")

    return error


def render_3d_volume_comparison(vol0, uncertainty_map, output_path,
                                 camera_elevation=25, camera_azimuth=135, zoom=0.4,
                                 unc_opacity=1.0, unc_opacity_mode='adaptive',
                                 unc_cmap='Reds',
                                 density_opacity=0.7, density_opacity_mode='linear',
                                 density_cmap='Greens',
                                 density_threshold=0.1, density_sigma=3.0,
                                 background_color='white', show_scalar_bar=True,
                                 scalar_bar_font_size=55, scalar_bar_font_family='arial',
                                 scalar_bar_outline=False, unc_unit='vx', density_unit='',
                                 show_bounds=True, bounds_color='black', bounds_width=5,
                                 show_volume_scalar_bar=True, volume_opacity=0.3,
                                 volume_unit='', show_axes=False,
                                 show_slice=False, slice_z=None,
                                 slice_frame_color='darkblue', slice_frame_width=3.0,
                                 window_size=(3840, 2120),
                                 mask=None):
    """Generate side-by-side 3D volume rendering of uncertainty vs density.

    Args:
        vol0: (B, C, D, H, W) input volume tensor
        uncertainty_map: (B, 3, D, H, W) uncertainty tensor (b = exp(log_b))
        output_path: output file path
        ... (same visualization parameters as demo_side_by_side.py)
    """
    try:
        # Convert to numpy
        v0 = vol0.cpu().numpy()[0, 0]           # (D, H, W)
        unc = uncertainty_map.cpu().numpy()[0]  # (3, D, H, W)

        # Average uncertainty over components for visualization
        unc_avg = unc.mean(axis=0)  # (D, H, W)

        # Compute feature density
        try:
            density = compute_feature_density(
                v0, threshold=density_threshold, sigma=density_sigma
            )
            # Apply mask to density (for cutout datasets)
            if mask is not None:
                density = density * mask
        except Exception as e:
            print(f"  Warning: Failed to compute feature density: {e}")
            density = None

        # Use viz_helpers for high-quality side-by-side rendering
        visualize_uncertainty_density_comparison(
            volume=v0,
            uncertainty=unc_avg,
            output_path=str(output_path),
            density=density,
            backend='auto',
            # Camera settings
            camera_elevation=camera_elevation,
            camera_azimuth=camera_azimuth,
            zoom=zoom,
            # Uncertainty (left panel) settings
            unc_opacity=unc_opacity,
            unc_opacity_mode=unc_opacity_mode,
            unc_cmap=unc_cmap,
            # Density (right panel) settings
            density_opacity=density_opacity,
            density_opacity_mode=density_opacity_mode,
            density_cmap=density_cmap,
            # Common settings
            volume_opacity=volume_opacity,
            background_color=background_color,
            show_axes=show_axes,
            show_scalar_bar=show_scalar_bar,
            show_volume_scalar_bar=show_volume_scalar_bar,
            scalar_bar_vertical=True,
            scalar_bar_font_size=scalar_bar_font_size,
            scalar_bar_font_family=scalar_bar_font_family,
            scalar_bar_outline=scalar_bar_outline,
            # Units
            left_unit=unc_unit,
            right_unit=density_unit,
            volume_unit=volume_unit,
            # Bounding box
            show_bounds=show_bounds,
            bounds_color=bounds_color,
            bounds_width=bounds_width,
            # Slice visualization
            show_slice=show_slice,
            slice_z=slice_z,
            slice_frame_color=slice_frame_color,
            slice_frame_width=slice_frame_width,
            # Resolution
            window_size=window_size,
        )

        print(f"  ✓ 3D volume comparison saved to: {output_path}")

    except ImportError as e:
        print(f"  Note: Install PyVista for 3D rendering: pip install pyvista")
        print(f"        Error: {e}")
    except Exception as e:
        print(f"  Warning: 3D volume rendering failed: {e}")
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
    # 3D volume rendering options
    parser.add_argument('--enable_3d_rendering', action='store_true',
                       help='Enable high-quality 3D volume rendering for uncertainty (requires model with uncertainty head)')
    parser.add_argument('--viz-config', type=str, default=None,
                       help='Path to visualization config YAML (overrides individual render_* args if provided)')
    parser.add_argument('--render_camera_elevation', type=float, default=25,
                       help='Camera elevation angle for 3D rendering (default: 25)')
    parser.add_argument('--render_camera_azimuth', type=float, default=135,
                       help='Camera azimuth angle for 3D rendering (default: 135)')
    parser.add_argument('--render_zoom', type=float, default=0.4,
                       help='Zoom factor for 3D rendering (default: 0.4)')
    parser.add_argument('--render_resolution', type=int, nargs=2, default=[3840, 2120],
                       help='3D rendering resolution (width height, default: 3840 2120)')
    parser.add_argument('--render_show_slice', action='store_true',
                       help='Show 2D slices below 3D volumes in rendering')
    parser.add_argument('--render_slice_z', type=int, default=None,
                       help='Z-index for slice plane (default: middle)')
    args = parser.parse_args()

    # Load visualization config if provided
    viz_params = None
    if args.viz_config is not None:
        print(f"Loading visualization config from: {args.viz_config}")
        viz_params = load_visualization_config(args.viz_config)

    # Create output directory (handle Windows MAX_PATH 260 char limit)
    output_dir = Path(args.output_dir).resolve()
    abs_len = len(str(output_dir))
    # Reserve 30 chars for filenames like 'sample_0000_3d_comparison.png'
    max_dir_len = 230
    if sys.platform == 'win32' and abs_len > max_dir_len:
        excess = abs_len - max_dir_len
        parts = list(output_dir.parts)
        longest_idx = max(range(len(parts)), key=lambda i: len(parts[i]))
        original = parts[longest_idx]
        truncated_len = max(len(original) - excess, 20)
        parts[longest_idx] = original[:truncated_len]
        output_dir = Path(*parts)
        print(f"Warning: Output path truncated for Windows MAX_PATH limit")
        print(f"  {output_dir}")
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
        mask_np = sample['mask'].numpy() if 'mask' in sample else None

        # Run inference
        flow_pred, metrics, epe_map, uncertainty_map, alpha_map = test_single_sample(
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
        if alpha_map is not None:
            print(f"  MoL alpha (confidence): {alpha_map.mean().item():.4f}")

        # Visualize flow
        output_path = output_dir / f'sample_{args.sample:04d}_prediction.png'
        visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path)

        # Visualize uncertainty (if model has UncertaintyHead)
        if uncertainty_map is not None:
            unc_path = output_dir / f'sample_{args.sample:04d}_uncertainty.png'
            visualize_uncertainty(
                vol0, flow_pred, flow_gt, epe_map, uncertainty_map, unc_path
            )

            # Generate 3D volume comparison if requested
            if args.enable_3d_rendering:
                render_3d_path = output_dir / f'sample_{args.sample:04d}_3d_comparison.png'

                # Check if error volume should be rendered
                error_enabled = viz_params.get('error_enabled', False) if viz_params else False
                has_gt_flow = flow_gt is not None

                if error_enabled and has_gt_flow:
                    # Render 3-panel: uncertainty, density, error
                    from viz_helpers import visualize_uncertainty_density_error_comparison

                    # Compute error volume
                    error_metric = viz_params.get('error_metric', 'epe') if viz_params else 'epe'
                    error_volume = compute_error_volume(
                        flow_pred[0].cpu().numpy(),  # (3, D, H, W)
                        flow_gt[0].cpu().numpy(),    # (3, D, H, W)
                        metric=error_metric
                    )

                    # Extract numpy arrays
                    vol0_np = vol0[0, 0].cpu().numpy()  # (D, H, W)
                    unc_np = uncertainty_map[0].mean(0).cpu().numpy()  # (D, H, W)

                    if viz_params is not None:
                        visualize_uncertainty_density_error_comparison(
                            vol0_np, unc_np, error_volume, render_3d_path,
                            mask=mask_np,
                            unc_opacity=viz_params.get('unc_opacity', 1.0),
                            unc_opacity_mode=viz_params.get('unc_opacity_mode', 'adaptive'),
                            unc_cmap=viz_params.get('unc_cmap', 'Reds'),
                            unc_unit=viz_params.get('unc_unit', 'vx'),
                            density_opacity=viz_params.get('density_opacity', 0.7),
                            density_opacity_mode=viz_params.get('density_opacity_mode', 'linear'),
                            density_cmap=viz_params.get('density_cmap', 'Greens'),
                            density_unit=viz_params.get('density_unit', ''),
                            density_threshold=viz_params.get('density_threshold', 0.1),
                            density_sigma=viz_params.get('density_sigma', 3.0),
                            error_opacity=viz_params.get('error_opacity', 1.0),
                            error_opacity_mode=viz_params.get('error_opacity_mode', 'adaptive'),
                            error_cmap=viz_params.get('error_cmap', 'hot'),
                            error_unit=viz_params.get('error_unit', 'vx'),
                            camera_elevation=viz_params.get('camera_elevation', 25),
                            camera_azimuth=viz_params.get('camera_azimuth', 135),
                            zoom=viz_params.get('zoom', 0.4),
                            volume_opacity=viz_params.get('volume_opacity', 0.3),
                            volume_unit=viz_params.get('volume_unit', ''),
                            background_color=viz_params.get('background_color', 'white'),
                            show_scalar_bar=viz_params.get('show_scalar_bar', True),
                            show_volume_scalar_bar=viz_params.get('show_volume_scalar_bar', True),
                            scalar_bar_font_size=viz_params.get('scalar_bar_font_size', 55),
                            scalar_bar_font_family=viz_params.get('scalar_bar_font_family', 'arial'),
                            scalar_bar_outline=viz_params.get('scalar_bar_outline', False),
                            show_axes=viz_params.get('show_axes', False),
                            show_bounds=viz_params.get('show_bounds', True),
                            bounds_color=viz_params.get('bounds_color', 'black'),
                            bounds_width=viz_params.get('bounds_width', 5),
                            show_slice=viz_params.get('show_slice', False),
                            slice_z=viz_params.get('slice_z', None),
                            slice_frame_color=viz_params.get('slice_frame_color', 'darkblue'),
                            slice_frame_width=viz_params.get('slice_frame_width', 3.0),
                            window_size=viz_params.get('window_size', (5760, 2120))
                        )
                    else:
                        visualize_uncertainty_density_error_comparison(
                            vol0_np, unc_np, error_volume, render_3d_path,
                            mask=mask_np,
                            camera_elevation=args.render_camera_elevation,
                            camera_azimuth=args.render_camera_azimuth,
                            zoom=args.render_zoom,
                            show_slice=args.render_show_slice,
                            slice_z=args.render_slice_z,
                            window_size=tuple(args.render_resolution)
                        )
                else:
                    # Render 2-panel: uncertainty, density (original behavior)
                    if viz_params is not None:
                        # Filter out error-related params not accepted by 2-panel renderer
                        viz_2panel = {k: v for k, v in viz_params.items()
                                      if not k.startswith('error_')}
                        render_3d_volume_comparison(vol0, uncertainty_map, render_3d_path, mask=mask_np, **viz_2panel)
                    else:
                        render_3d_volume_comparison(
                            vol0, uncertainty_map, render_3d_path,
                            mask=mask_np,
                            camera_elevation=args.render_camera_elevation,
                            camera_azimuth=args.render_camera_azimuth,
                            zoom=args.render_zoom,
                            show_slice=args.render_show_slice,
                            slice_z=args.render_slice_z,
                            window_size=tuple(args.render_resolution)
                        )

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
            mask_np = sample['mask'].numpy() if 'mask' in sample else None

            flow_pred, _, epe_map, uncertainty_map, _ = test_single_sample(
                model, vol0, vol1, flow_gt, device, args.iters
            )

            output_path = output_dir / f'sample_{i:04d}_prediction.png'
            visualize_sample(vol0, vol1, flow_gt, flow_pred, epe_map, output_path)

            if uncertainty_map is not None:
                unc_path = output_dir / f'sample_{i:04d}_uncertainty.png'
                visualize_uncertainty(
                    vol0, flow_pred, flow_gt, epe_map, uncertainty_map,
                    unc_path
                )

                # Generate 3D volume comparison if requested
                if args.enable_3d_rendering:
                    render_3d_path = output_dir / f'sample_{i:04d}_3d_comparison.png'

                    # Check if error volume should be rendered
                    error_enabled = viz_params.get('error_enabled', False) if viz_params else False
                    has_gt_flow = flow_gt is not None

                    if error_enabled and has_gt_flow:
                        # Render 3-panel: uncertainty, density, error
                        from viz_helpers import visualize_uncertainty_density_error_comparison

                        # Compute error volume
                        error_metric = viz_params.get('error_metric', 'epe') if viz_params else 'epe'
                        error_volume = compute_error_volume(
                            flow_pred[0].cpu().numpy(),  # (3, D, H, W)
                            flow_gt[0].cpu().numpy(),    # (3, D, H, W)
                            metric=error_metric
                        )

                        # Extract numpy arrays
                        vol0_np = vol0[0, 0].cpu().numpy()  # (D, H, W)
                        unc_np = uncertainty_map[0].mean(0).cpu().numpy()  # (D, H, W)

                        if viz_params is not None:
                            visualize_uncertainty_density_error_comparison(
                                vol0_np, unc_np, error_volume, render_3d_path,
                                mask=mask_np,
                                unc_opacity=viz_params.get('unc_opacity', 1.0),
                                unc_opacity_mode=viz_params.get('unc_opacity_mode', 'adaptive'),
                                unc_cmap=viz_params.get('unc_cmap', 'Reds'),
                                unc_unit=viz_params.get('unc_unit', 'vx'),
                                density_opacity=viz_params.get('density_opacity', 0.7),
                                density_opacity_mode=viz_params.get('density_opacity_mode', 'linear'),
                                density_cmap=viz_params.get('density_cmap', 'Greens'),
                                density_unit=viz_params.get('density_unit', ''),
                                density_threshold=viz_params.get('density_threshold', 0.1),
                                density_sigma=viz_params.get('density_sigma', 3.0),
                                error_opacity=viz_params.get('error_opacity', 1.0),
                                error_opacity_mode=viz_params.get('error_opacity_mode', 'adaptive'),
                                error_cmap=viz_params.get('error_cmap', 'hot'),
                                error_unit=viz_params.get('error_unit', 'vx'),
                                camera_elevation=viz_params.get('camera_elevation', 25),
                                camera_azimuth=viz_params.get('camera_azimuth', 135),
                                zoom=viz_params.get('zoom', 0.4),
                                volume_opacity=viz_params.get('volume_opacity', 0.3),
                                volume_unit=viz_params.get('volume_unit', ''),
                                background_color=viz_params.get('background_color', 'white'),
                                show_scalar_bar=viz_params.get('show_scalar_bar', True),
                                show_volume_scalar_bar=viz_params.get('show_volume_scalar_bar', True),
                                scalar_bar_font_size=viz_params.get('scalar_bar_font_size', 55),
                                scalar_bar_font_family=viz_params.get('scalar_bar_font_family', 'arial'),
                                scalar_bar_outline=viz_params.get('scalar_bar_outline', False),
                                show_axes=viz_params.get('show_axes', False),
                                show_bounds=viz_params.get('show_bounds', True),
                                bounds_color=viz_params.get('bounds_color', 'black'),
                                bounds_width=viz_params.get('bounds_width', 5),
                                show_slice=viz_params.get('show_slice', False),
                                slice_z=viz_params.get('slice_z', None),
                                slice_frame_color=viz_params.get('slice_frame_color', 'darkblue'),
                                slice_frame_width=viz_params.get('slice_frame_width', 3.0),
                                window_size=viz_params.get('window_size', (5760, 2120))
                            )
                        else:
                            visualize_uncertainty_density_error_comparison(
                                vol0_np, unc_np, error_volume, render_3d_path,
                                mask=mask_np,
                                camera_elevation=args.render_camera_elevation,
                                camera_azimuth=args.render_camera_azimuth,
                                zoom=args.render_zoom,
                                show_slice=args.render_show_slice,
                                slice_z=args.render_slice_z,
                                window_size=tuple(args.render_resolution)
                            )
                    else:
                        # Render 2-panel: uncertainty, density (original behavior)
                        if viz_params is not None:
                            render_3d_volume_comparison(vol0, uncertainty_map, render_3d_path, mask=mask_np, **viz_params)
                        else:
                            render_3d_volume_comparison(
                                vol0, uncertainty_map, render_3d_path,
                                mask=mask_np,
                                camera_elevation=args.render_camera_elevation,
                                camera_azimuth=args.render_camera_azimuth,
                                zoom=args.render_zoom,
                                show_slice=args.render_show_slice,
                                slice_z=args.render_slice_z,
                                window_size=tuple(args.render_resolution)
                            )

        print(f"\n✓ Generated {num_to_visualize} visualizations in: {output_dir}")


if __name__ == '__main__':
    main()
