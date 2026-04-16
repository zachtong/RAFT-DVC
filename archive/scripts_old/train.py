"""
Training script for RAFT-DVC

Usage:
    python scripts/train.py --config configs/training/default.yaml
    python scripts/train.py --data_dir /path/to/data --output_dir /path/to/output
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, random_split

from src.core import RAFTDVC, RAFTDVCConfig
from src.data import VolumePairDataset
from src.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train RAFT-DVC model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Path to output directory')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=96,
                        help='Hidden dimension for GRU')
    parser.add_argument('--context_dim', type=int, default=64,
                        help='Context dimension')
    parser.add_argument('--corr_levels', type=int, default=4,
                        help='Number of correlation pyramid levels')
    parser.add_argument('--corr_radius', type=int, default=4,
                        help='Correlation lookup radius')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of refinement iterations')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Sequence loss gamma')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    
    # Data arguments
    parser.add_argument('--patch_size', type=int, nargs=3, default=None,
                        help='Patch size for training (H W D)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    
    # Other arguments
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model config
    model_config = RAFTDVCConfig(
        input_channels=1,
        hidden_dim=args.hidden_dim,
        context_dim=args.context_dim,
        corr_levels=args.corr_levels,
        corr_radius=args.corr_radius,
        iters=args.iters,
        mixed_precision=args.mixed_precision
    )
    
    # Create model
    model = RAFTDVC(model_config)
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Create dataset
    patch_size = tuple(args.patch_size) if args.patch_size else None
    
    dataset = VolumePairDataset(
        root_dir=args.data_dir,
        augment=args.augment,
        patch_size=patch_size,
        has_flow=True
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ) if val_size > 0 else None
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gamma': args.gamma,
        'grad_clip': args.grad_clip,
        'iters': args.iters,
        'mixed_precision': args.mixed_precision,
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        config=train_config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
