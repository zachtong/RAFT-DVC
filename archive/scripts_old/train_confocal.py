"""
Training script for RAFT-DVC on synthetic confocal dataset.

Model and training configurations are completely decoupled.
Users must specify both model architecture and training settings.

Usage:
    # Train with specific model and training configs
    python scripts/train_confocal.py \
        --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \
        --training-config configs/training/baseline_300ep.yaml

    # Resume training
    python scripts/train_confocal.py \
        --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \
        --training-config configs/training/baseline_300ep.yaml \
        --resume outputs/training/experiment/checkpoint_epoch_100.pth

    # Use defaults (1/8 model, baseline training)
    python scripts/train_confocal.py
"""

import os
import sys
from pathlib import Path
import argparse
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.core import RAFTDVC, RAFTDVCConfig
from src.data import VolumePairDataset
from src.training import Trainer
from src.training.augmentations import CutoutAugmentation3D, GaussianBlur3D
from src.visualization import (
    compute_feature_density,
    auto_render_uncertainty_volume,
    create_uncertainty_comparison_figure,
)


def setup_strategies(config, device):
    """
    Initialize training strategies from config.

    Each strategy is toggled via the 'strategies' section in the training config.
    When no 'strategies' section exists, returns empty dict (baseline behavior).

    Args:
        config: Training configuration dict
        device: Torch device

    Returns:
        Dict of initialized strategy objects/parameters
    """
    strategies_cfg = config.get('strategies', {})
    strategies = {}

    if not strategies_cfg:
        return strategies

    # --- Input Augmentation Strategies ---

    # Cutout
    cutout_cfg = strategies_cfg.get('cutout', {})
    if cutout_cfg.get('enabled', False):
        cutout = CutoutAugmentation3D(
            prob=cutout_cfg.get('prob', 0.7),
            keep_min=cutout_cfg.get('keep_min', 0.60),
            keep_max=cutout_cfg.get('keep_max', 0.85),
            region_size_min=cutout_cfg.get('region_size_min', 0.2),
            region_size_max=cutout_cfg.get('region_size_max', 0.5),
        )
        cutout.train()
        strategies['cutout'] = cutout
        print(f"  [Strategy] Cutout: {cutout}")

    # Gaussian Blur
    blur_cfg = strategies_cfg.get('gaussian_blur', {})
    if blur_cfg.get('enabled', False):
        blur = GaussianBlur3D(
            sigma=blur_cfg.get('sigma', 1.0),
        ).to(device)
        strategies['gaussian_blur'] = blur
        print(f"  [Strategy] Gaussian Blur: {blur}")

    # --- Loss Modification Strategies ---
    # (Placeholder for Phase 2-4, will be added incrementally)

    # Laplacian Smooth
    smooth_cfg = strategies_cfg.get('laplacian_smooth', {})
    if smooth_cfg.get('enabled', False):
        from src.training.loss import LaplacianSmoothLoss
        strides = smooth_cfg.get('strides', [1, 2, 4])
        strategies['laplacian_smooth'] = LaplacianSmoothLoss(strides=strides).to(device)
        strategies['lambda_smooth'] = smooth_cfg.get('weight', 0.1)
        print(f"  [Strategy] Laplacian Smooth: weight={strategies['lambda_smooth']}, strides={strides}")

    # Masked Loss
    masked_cfg = strategies_cfg.get('masked_loss', {})
    if masked_cfg.get('enabled', False):
        # Will be implemented in Phase 3
        from src.training.loss import MaskedSequenceLoss
        strategies['masked_loss'] = MaskedSequenceLoss(
            gamma=config['training']['gamma'],
            threshold=masked_cfg.get('feature_threshold', 0.1),
            sigma=masked_cfg.get('density_sigma', 3.0),
            min_weight=masked_cfg.get('min_weight', 0.2),
        ).to(device)
        print(f"  [Strategy] Masked Loss: threshold={masked_cfg.get('feature_threshold', 0.1)}")

    # Uncertainty (NLL or MoL)
    uncert_cfg = strategies_cfg.get('uncertainty', {})
    if uncert_cfg.get('enabled', False):
        loss_type = uncert_cfg.get('loss_type', 'nll')
        strategies['uncertainty'] = True
        strategies['loss_type'] = loss_type

        if loss_type == 'mol':
            from src.training.loss import MoLSequenceLoss
            mol_b1 = uncert_cfg.get('mol_b1', 0.5)
            use_softplus = uncert_cfg.get('use_softplus', False)
            strategies['nll_loss'] = MoLSequenceLoss(
                gamma=config['training']['gamma'],
                b1=mol_b1,
                use_softplus=use_softplus,
            ).to(device)
            print(f"  [Strategy] MoL Uncertainty: b1={mol_b1}, softplus={use_softplus}")
        else:
            from src.training.loss import NLLSequenceLoss
            import math
            b_min = uncert_cfg.get('b_min', None)
            log_b_min = math.log(b_min) if b_min is not None else None
            strategies['nll_loss'] = NLLSequenceLoss(
                gamma=config['training']['gamma'],
                log_b_min=log_b_min,
            ).to(device)
            floor_str = f", b_min={b_min} (log_b_min={log_b_min:.2f})" if b_min else ""
            print(f"  [Strategy] NLL Uncertainty: enabled{floor_str}")

    return strategies


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model_config(model_config_path):
    """Load model configuration from file."""
    with open(model_config_path, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)
    return model_cfg['architecture']


def create_model(model_arch, uncertainty_mode="none"):
    """
    Create RAFT-DVC model from model architecture config.

    Args:
        model_arch: Model architecture configuration dict
        uncertainty_mode: ``"none"`` | ``"nll"`` | ``"mol"``

    Returns:
        RAFTDVC model instance
    """
    # Create model config object
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
        iters=model_arch.get('iters', 12),
        uncertainty_mode=uncertainty_mode,
    )

    print(f"\n{'='*70}")
    print(f"Model Architecture:")
    print(f"  Encoder Type: {model_config.encoder_type}")
    print(f"  Feature Dim: {model_config.feature_dim}")
    print(f"  Context Dim: {model_config.context_dim}")
    print(f"  Hidden Dim: {model_config.hidden_dim}")
    print(f"  Corr Levels: {model_config.corr_levels}")
    print(f"  Corr Radius: {model_config.corr_radius}")
    if uncertainty_mode != "none":
        print(f"  Uncertainty: {uncertainty_mode.upper()}")
    print(f"{'='*70}")

    model = RAFTDVC(model_config)
    return model


def create_dataloaders(config):
    """Create training and validation dataloaders."""

    # Training dataset
    train_dataset = VolumePairDataset(
        root_dir=config['data']['train_dir'],
        augment=config['data']['augment'],
        patch_size=config['data']['patch_size'],
        has_flow=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    # Validation dataset
    val_dataset = VolumePairDataset(
        root_dir=config['data']['val_dir'],
        augment=False,  # No augmentation for validation
        patch_size=None,  # Use full volumes for validation
        has_flow=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Batch size 1 for validation
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader


def compute_epe(flow_pred, flow_gt):
    """Compute End-Point Error."""
    epe = torch.sqrt(torch.sum((flow_pred - flow_gt)**2, dim=1))
    return epe.mean().item()


def sequence_loss(flow_predictions, flow_gt, gamma=0.9):
    """
    Compute weighted sequence loss for RAFT.

    Args:
        flow_predictions: List of flow predictions at each iteration
        flow_gt: Ground truth flow
        gamma: Weight decay factor for earlier predictions
    """
    n_predictions = len(flow_predictions)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_predictions[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss.mean()

    return flow_loss


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch, writer, device, strategies=None):
    """Train for one epoch.

    Args:
        strategies: Dict of active training strategies (from setup_strategies).
                   None or empty dict = baseline behavior.
    """
    import time as _time

    model.train()
    if strategies is None:
        strategies = {}

    total_loss = 0
    total_epe = 0
    total_smooth_loss = 0

    # Diagnose first 3 iterations of epoch 0
    _diagnose = (epoch == 0)
    _DIAG_ITERS = 3

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, batch in enumerate(pbar):
        _do_diag = _diagnose and i < _DIAG_ITERS

        if _do_diag:
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()

        vol0 = batch['vol0'].to(device)
        vol1 = batch['vol1'].to(device)
        flow_gt = batch['flow'].to(device)

        # === Strategy: Input Preprocessing (Gaussian Blur) ===
        if 'gaussian_blur' in strategies:
            vol0 = strategies['gaussian_blur'](vol0)
            vol1 = strategies['gaussian_blur'](vol1)

        # === Strategy: Input Augmentation (Cutout) ===
        if 'cutout' in strategies:
            vol0, vol1 = strategies['cutout'](vol0, vol1)

        if _do_diag:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()

        optimizer.zero_grad()

        # === Forward pass ===
        if config['training']['use_amp']:
            with torch.amp.autocast('cuda'):
                model_output = model(vol0, vol1, iters=model.config.iters)

                if _do_diag:
                    torch.cuda.synchronize()
                    _t2 = _time.perf_counter()

                # Handle uncertainty output format (Phase 4)
                if 'uncertainty' in strategies:
                    flow_predictions, uncertainty_predictions = model_output
                else:
                    flow_predictions = model_output

                # === Loss computation ===
                if 'uncertainty' in strategies:
                    loss = strategies['nll_loss'](
                        flow_predictions, uncertainty_predictions, flow_gt
                    )
                elif 'masked_loss' in strategies:
                    loss = strategies['masked_loss'](
                        flow_predictions, flow_gt, vol0
                    )
                else:
                    loss = sequence_loss(
                        flow_predictions, flow_gt,
                        gamma=config['training']['gamma']
                    )

                if _do_diag:
                    torch.cuda.synchronize()
                    _t3 = _time.perf_counter()

                # === Strategy: Smoothness regularization (additive) ===
                smooth_loss_val = 0.0
                if 'laplacian_smooth' in strategies:
                    smooth_loss = strategies['laplacian_smooth'](flow_predictions[-1])
                    loss = loss + strategies['lambda_smooth'] * smooth_loss
                    smooth_loss_val = smooth_loss.item()
        else:
            model_output = model(vol0, vol1, iters=model.config.iters)

            if _do_diag:
                torch.cuda.synchronize()
                _t2 = _time.perf_counter()

            if 'uncertainty' in strategies:
                flow_predictions, uncertainty_predictions = model_output
            else:
                flow_predictions = model_output

            if 'uncertainty' in strategies:
                loss = strategies['nll_loss'](
                    flow_predictions, uncertainty_predictions, flow_gt
                )
            elif 'masked_loss' in strategies:
                loss = strategies['masked_loss'](
                    flow_predictions, flow_gt, vol0
                )
            else:
                loss = sequence_loss(
                    flow_predictions, flow_gt,
                    gamma=config['training']['gamma']
                )

            if _do_diag:
                torch.cuda.synchronize()
                _t3 = _time.perf_counter()

            smooth_loss_val = 0.0
            if 'laplacian_smooth' in strategies:
                smooth_loss = strategies['laplacian_smooth'](flow_predictions[-1])
                loss = loss + strategies['lambda_smooth'] * smooth_loss
                smooth_loss_val = smooth_loss.item()

        # Backward pass
        if config['training']['use_amp']:
            scaler.scale(loss).backward()

            if _do_diag:
                torch.cuda.synchronize()
                _t4 = _time.perf_counter()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if _do_diag:
                torch.cuda.synchronize()
                _t4 = _time.perf_counter()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            optimizer.step()

        if _do_diag:
            torch.cuda.synchronize()
            _t5 = _time.perf_counter()
            _mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n  [DIAG iter {i}] Data: {_t1-_t0:.3f}s | "
                  f"Forward: {_t2-_t1:.3f}s | Loss: {_t3-_t2:.3f}s | "
                  f"Backward: {_t4-_t3:.3f}s | OptStep: {_t5-_t4:.3f}s | "
                  f"TOTAL: {_t5-_t0:.3f}s | PeakMem: {_mem:.2f}GB")

        # Step scheduler per batch (CyclicLR steps per iteration)
        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            epe = compute_epe(flow_predictions[-1], flow_gt)

        total_loss += loss.item()
        total_epe += epe
        total_smooth_loss += smooth_loss_val

        # Update progress bar
        postfix = {'loss': loss.item(), 'EPE': epe}
        if smooth_loss_val > 0:
            postfix['smooth'] = smooth_loss_val
        pbar.set_postfix(postfix)

        # Log to tensorboard
        if writer is not None and i % config['output']['log_freq'] == 0:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/EPE', epe, global_step)

            # Strategy-specific logging
            if smooth_loss_val > 0:
                writer.add_scalar('train/smooth_loss', smooth_loss_val, global_step)
            if 'uncertainty' in strategies:
                with torch.no_grad():
                    unc_last = uncertainty_predictions[-1]
                    if strategies.get('loss_type') == 'mol':
                        # MoL: 4 channels — log_b2 (3ch) + logit_alpha (1ch)
                        mean_log_b2 = unc_last[:, :3].mean().item()
                        mean_alpha = torch.sigmoid(unc_last[:, 3:4]).mean().item()
                        writer.add_scalar('train/mean_log_b2', mean_log_b2, global_step)
                        writer.add_scalar('train/mean_alpha', mean_alpha, global_step)
                    else:
                        mean_log_b = unc_last.mean().item()
                        writer.add_scalar('train/mean_log_b', mean_log_b, global_step)

    avg_loss = total_loss / len(train_loader)
    avg_epe = total_epe / len(train_loader)

    return avg_loss, avg_epe


def log_validation_visualizations(writer, vol0, flow_gt, flow_pred, epoch,
                                   uncertainty=None):
    """Log mid-slice visualizations to TensorBoard.

    Creates a multi-panel figure showing input, error, and uncertainty
    (if available) at the middle z-slice of the first sample in the batch.
    Also logs uncertainty-specific scalar metrics.

    Args:
        writer: TensorBoard SummaryWriter
        vol0: (B, 1, D, H, W) input volume
        flow_gt: (B, 3, D, H, W) ground truth flow
        flow_pred: (B, 3, D, H, W) predicted flow
        epoch: current epoch number
        uncertainty: (B, 3, D, H, W) log_b values, or None
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # First sample → numpy
    v0 = vol0[0, 0].cpu().numpy()        # (D, H, W)
    fp = flow_pred[0].cpu().numpy()       # (3, D, H, W)
    fg = flow_gt[0].cpu().numpy()         # (3, D, H, W)
    mid = v0.shape[0] // 2

    # Error magnitude
    err = np.sqrt(((fp - fg) ** 2).sum(axis=0))  # (D, H, W)

    if uncertainty is not None:
        # Only take first 3 channels (log_b for NLL, log_b2 for MoL).
        # MoL outputs 4 channels (3 log_b2 + 1 logit_alpha); the 4th
        # channel is a mixing weight logit and must NOT be averaged in.
        log_b = uncertainty[0, :3].cpu().numpy()      # (3, D, H, W)
        b = np.exp(np.clip(log_b, -10, 10))           # numerical safety
        b_mag = b.mean(axis=0)                         # (D, H, W)

        # Compute feature density (optional, for enhanced visualization)
        try:
            feature_density = compute_feature_density(
                v0, threshold=0.1, sigma=3.0, min_density=0.0, max_density=1.0
            )
        except Exception as e:
            print(f"Warning: Failed to compute feature density: {e}")
            feature_density = None

        # === 2D Slice Visualization (existing multi-panel figure) ===
        fig_2d = create_uncertainty_comparison_figure(
            volume=v0,
            flow_pred=fp,
            flow_gt=fg,
            uncertainty=b_mag,
            feature_density=feature_density,
            slice_idx=mid,
            figsize=(18, 10) if feature_density is not None else (12, 10)
        )
        writer.add_figure('val/visualization_2d', fig_2d, epoch)
        plt.close(fig_2d)

        # === 3D Volume Rendering ===
        # Only render 3D every N epochs to save time (volume rendering is slow)
        render_3d_freq = 10  # render every 10 epochs
        if (epoch + 1) % render_3d_freq == 0 or epoch == 0:
            try:
                # Use auto backend (prefers PyVista, falls back to matplotlib)
                fig_3d = auto_render_uncertainty_volume(
                    volume=v0,
                    uncertainty=b_mag,
                    feature_density=feature_density,
                    backend='auto',  # auto-select best available backend
                    view_elev=30.0,
                    view_azim=45.0,
                    figsize=(10, 8),
                    title=f'Uncertainty Volume (Epoch {epoch + 1})'
                )

                # If backend returned a Figure (matplotlib), add to TensorBoard
                if isinstance(fig_3d, plt.Figure):
                    writer.add_figure('val/visualization_3d', fig_3d, epoch)
                    plt.close(fig_3d)
                # If backend returned numpy array (PyVista), convert to image
                elif isinstance(fig_3d, np.ndarray):
                    # PyVista returns (H, W, 3) RGB array
                    # TensorBoard expects (3, H, W) for add_image
                    img_tensor = torch.from_numpy(fig_3d).permute(2, 0, 1).float() / 255.0
                    writer.add_image('val/visualization_3d', img_tensor, epoch)

            except ImportError as e:
                # First time only: warn about missing PyVista
                if epoch == 0:
                    print(f"Note: 3D rendering with matplotlib only (install PyVista for better quality)")
                # Fallback to matplotlib only
                fig_3d = auto_render_uncertainty_volume(
                    volume=v0,
                    uncertainty=b_mag,
                    feature_density=feature_density,
                    backend='matplotlib',
                    view_elev=30.0,
                    view_azim=45.0,
                    figsize=(10, 8),
                    title=f'Uncertainty Volume (Epoch {epoch + 1})'
                )
                writer.add_figure('val/visualization_3d', fig_3d, epoch)
                plt.close(fig_3d)
            except Exception as e:
                print(f"Warning: Failed to render 3D volume: {e}")

        # === Scalar metrics ===
        # Compute Spearman correlation (full volume)
        err_vol = err.flatten()
        unc_vol = b_mag.flatten()
        rank_err = np.empty_like(err_vol)
        rank_unc = np.empty_like(unc_vol)
        rank_err[np.argsort(err_vol)] = np.arange(len(err_vol))
        rank_unc[np.argsort(unc_vol)] = np.arange(len(unc_vol))
        rho = np.corrcoef(rank_err, rank_unc)[0, 1]

        writer.add_scalar('val/uncertainty_correlation', rho, epoch)
        writer.add_scalar('val/uncertainty_mean', b_mag.mean(), epoch)
        writer.add_scalar('val/uncertainty_std', b_mag.std(), epoch)

    else:
        # Non-uncertainty model: Input, GT flow, Error
        gt_mag = np.sqrt((fg ** 2).sum(axis=0))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(v0[mid], cmap='gray')
        axes[0].set_title('Input vol0', fontsize=12)
        axes[0].axis('off')

        im1 = axes[1].imshow(gt_mag[mid], cmap='RdBu_r')
        axes[1].set_title('Flow GT magnitude', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax_err = max(np.percentile(err[mid], 99), 1e-6)
        im2 = axes[2].imshow(err[mid], cmap='hot', vmin=0, vmax=vmax_err)
        axes[2].set_title(f'Error (mean={err[mid].mean():.3f})', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.suptitle(f'Epoch {epoch + 1} | Mid-slice z={mid}', fontsize=14)
        plt.tight_layout()
        writer.add_figure('val/visualization', fig, epoch)
        plt.close(fig)


@torch.no_grad()
def validate(model, val_loader, config, epoch, writer, device):
    """Validate model."""
    model.eval()

    total_epe = 0
    total_log_b = 0.0
    total_alpha = 0.0
    n_samples = 0
    n_alpha_samples = 0
    vis_logged = False

    for batch in tqdm(val_loader, desc="Validation"):
        vol0 = batch['vol0'].to(device)
        vol1 = batch['vol1'].to(device)
        flow_gt = batch['flow'].to(device)

        # Use more iterations for validation
        # test_mode returns 2-tuple or 3-tuple (with uncertainty)
        output = model(
            vol0, vol1,
            iters=config['evaluation']['iters'],
            test_mode=True
        )
        flow_pred = output[1]  # high-res flow is always index 1

        epe = compute_epe(flow_pred, flow_gt)
        total_epe += epe

        # Accumulate uncertainty stats
        has_uncertainty = len(output) == 3
        if has_uncertainty:
            unc = output[2]
            # For NLL: 3 channels (log_b). For MoL: 4 channels (log_b2 + logit_alpha).
            total_log_b += unc[:, :3].mean().item() * vol0.shape[0]
            n_samples += vol0.shape[0]
            # MoL: also track mean alpha (mixing weight)
            if unc.shape[1] == 4:
                total_alpha += torch.sigmoid(unc[:, 3:4]).mean().item() * vol0.shape[0]
                n_alpha_samples += vol0.shape[0]

        # Log visualizations for first batch only
        if not vis_logged and writer is not None:
            uncertainty_map = output[2] if has_uncertainty else None
            try:
                log_validation_visualizations(
                    writer, vol0, flow_gt, flow_pred, epoch,
                    uncertainty=uncertainty_map
                )
            except Exception as e:
                print(f"  Warning: visualization failed: {e}")
            vis_logged = True

    avg_epe = total_epe / len(val_loader)

    if writer is not None:
        writer.add_scalar('val/EPE', avg_epe, epoch)
        if n_samples > 0:
            writer.add_scalar('val/mean_log_b',
                              total_log_b / n_samples, epoch)
        if n_alpha_samples > 0:
            writer.add_scalar('val/mean_alpha',
                              total_alpha / n_alpha_samples, epoch)

    return avg_epe


def main():
    parser = argparse.ArgumentParser(
        description='Train RAFT-DVC on confocal dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with 1/2 encoder and baseline training config
  python scripts/train_confocal.py \\
    --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \\
    --training-config configs/training/baseline_300ep.yaml

  # Resume training
  python scripts/train_confocal.py \\
    --model-config configs/models/raft_dvc_1_2_p2_r4.yaml \\
    --training-config configs/training/baseline_300ep.yaml \\
    --resume outputs/training/experiment/checkpoint_epoch_100.pth
        """
    )
    parser.add_argument(
        '--model-config',
        type=str,
        default='configs/models/raft_dvc_1_8_p4_r4.yaml',
        help='Path to model configuration file (default: 1/8 encoder)'
    )
    parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training/confocal_baseline.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"RAFT-DVC Training")
    print(f"{'='*70}")
    print(f"Model Config: {args.model_config}")
    print(f"Training Config: {args.training_config}")
    if args.resume:
        print(f"Resume From: {args.resume}")
    print(f"{'='*70}\n")

    # Load configurations
    print(f"Loading model configuration...")
    model_arch = load_model_config(args.model_config)

    print(f"Loading training configuration...")
    config = load_config(args.training_config)

    # Override resume path if provided
    if args.resume is not None:
        config['checkpoint']['resume'] = args.resume

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine uncertainty mode from config
    strategies_cfg = config.get('strategies', {})
    uncert_cfg = strategies_cfg.get('uncertainty', {})
    if uncert_cfg.get('enabled', False):
        loss_type = uncert_cfg.get('loss_type', 'nll')
        uncertainty_mode = loss_type  # "nll" or "mol"
    else:
        uncertainty_mode = "none"

    # Create model
    print("\nCreating model...")
    model = create_model(model_arch, uncertainty_mode=uncertainty_mode)
    model = model.to(device)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler - CyclicLR (matches volRAFT)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config['training']['base_lr'],
        max_lr=config['training']['max_lr'],
        step_size_up=config['training']['step_size_up'],
        step_size_down=config['training']['step_size_down'],
        mode=config['training']['mode'],
        cycle_momentum=False
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['training']['use_amp'] else None

    # Tensorboard writer
    writer = None
    if config['output']['tensorboard']:
        writer = SummaryWriter(output_dir / 'logs')

    # Resume or fine-tune from checkpoint
    start_epoch = 0
    best_epe = float('inf')

    finetune_path = config['checkpoint'].get('finetune')
    resume_path = config['checkpoint'].get('resume')

    if finetune_path is not None:
        # Fine-tune: load model weights only, fresh optimizer/scheduler/epoch
        # strict=False allows loading when architecture differs (e.g. added UncertaintyHead)
        print(f"Fine-tuning from {finetune_path}")
        checkpoint = torch.load(finetune_path)
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing:
            print(f"  New parameters (randomly initialized): {len(missing)} keys")
            for k in missing:
                print(f"    - {k}")
        if unexpected:
            print(f"  Ignored keys from checkpoint: {len(unexpected)} keys")
        print("  ✓ Loaded model weights (optimizer/scheduler reset for fine-tuning)")
    elif resume_path is not None:
        # Resume: restore full training state
        print(f"Resuming from {resume_path}")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Restored scheduler state")
        start_epoch = checkpoint['epoch'] + 1
        best_epe = checkpoint.get('best_epe', float('inf'))

    # Setup training strategies
    print("\nSetting up training strategies...")
    strategies = setup_strategies(config, device)
    if strategies:
        print(f"  Active strategies: {list(strategies.keys())}")
    else:
        print("  No strategies enabled (baseline training)")

    # Training loop
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Train
        train_loss, train_epe = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, epoch, writer, device,
            strategies=strategies
        )

        print(f"Train Loss: {train_loss:.4f}, Train EPE: {train_epe:.4f}")

        # Validate
        if (epoch + 1) % config['training']['val_freq'] == 0:
            val_epe = validate(model, val_loader, config, epoch, writer, device)
            print(f"Validation EPE: {val_epe:.4f}")

            # Save best model
            if val_epe < best_epe:
                best_epe = val_epe
                checkpoint_path = output_dir / 'checkpoint_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model_config': model.config.to_dict(),  # Save model architecture
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_epe': best_epe,
                    'config': config
                }, checkpoint_path)
                print(f"✓ Saved best model (EPE: {best_epe:.4f})")

        # Save checkpoint periodically
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'model_config': model.config.to_dict(),  # Save model architecture
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_epe': best_epe,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")

    print("\n" + "=" * 70)
    print(f"✓ Training complete! Best validation EPE: {best_epe:.4f}")
    print(f"  Model saved to: {output_dir / 'checkpoint_best.pth'}")

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
