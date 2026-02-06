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


def create_model(model_arch):
    """
    Create RAFT-DVC model from model architecture config.

    Args:
        model_arch: Model architecture configuration dict

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
        iters=model_arch.get('iters', 12)
    )

    print(f"\n{'='*70}")
    print(f"Model Architecture:")
    print(f"  Encoder Type: {model_config.encoder_type}")
    print(f"  Feature Dim: {model_config.feature_dim}")
    print(f"  Context Dim: {model_config.context_dim}")
    print(f"  Hidden Dim: {model_config.hidden_dim}")
    print(f"  Corr Levels: {model_config.corr_levels}")
    print(f"  Corr Radius: {model_config.corr_radius}")
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


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, epoch, writer, device):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_epe = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for i, batch in enumerate(pbar):
        vol0 = batch['vol0'].to(device)
        vol1 = batch['vol1'].to(device)
        flow_gt = batch['flow'].to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if config['training']['use_amp']:
            with torch.amp.autocast('cuda'):
                flow_predictions = model(vol0, vol1, iters=model.config.iters)
                loss = sequence_loss(flow_predictions, flow_gt, gamma=config['training']['gamma'])
        else:
            flow_predictions = model(vol0, vol1, iters=model.config.iters)
            loss = sequence_loss(flow_predictions, flow_gt, gamma=config['training']['gamma'])

        # Backward pass
        if config['training']['use_amp']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            optimizer.step()

        # Step scheduler per batch (CyclicLR steps per iteration)
        scheduler.step()

        # Compute metrics
        with torch.no_grad():
            epe = compute_epe(flow_predictions[-1], flow_gt)

        total_loss += loss.item()
        total_epe += epe

        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'EPE': epe})

        # Log to tensorboard
        if writer is not None and i % config['output']['log_freq'] == 0:
            global_step = epoch * len(train_loader) + i
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/EPE', epe, global_step)

    avg_loss = total_loss / len(train_loader)
    avg_epe = total_epe / len(train_loader)

    return avg_loss, avg_epe


@torch.no_grad()
def validate(model, val_loader, config, epoch, writer, device):
    """Validate model."""
    model.eval()

    total_epe = 0

    for batch in tqdm(val_loader, desc="Validation"):
        vol0 = batch['vol0'].to(device)
        vol1 = batch['vol1'].to(device)
        flow_gt = batch['flow'].to(device)

        # Use more iterations for validation
        _, flow_pred = model(vol0, vol1, iters=config['evaluation']['iters'], test_mode=True)

        epe = compute_epe(flow_pred, flow_gt)
        total_epe += epe

    avg_epe = total_epe / len(val_loader)

    if writer is not None:
        writer.add_scalar('val/EPE', avg_epe, epoch)

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

    # Create model
    print("\nCreating model...")
    model = create_model(model_arch)
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
    scaler = torch.amp.GradScaler('cuda') if config['training']['use_amp'] else None

    # Tensorboard writer
    writer = None
    if config['output']['tensorboard']:
        writer = SummaryWriter(output_dir / 'logs')

    # Resume from checkpoint if specified
    start_epoch = 0
    best_epe = float('inf')

    if config['checkpoint']['resume'] is not None:
        print(f"Resuming from {config['checkpoint']['resume']}")
        checkpoint = torch.load(config['checkpoint']['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Restored scheduler state")
        start_epoch = checkpoint['epoch'] + 1
        best_epe = checkpoint.get('best_epe', float('inf'))

    # Training loop
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Train
        train_loss, train_epe = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, config, epoch, writer, device
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
