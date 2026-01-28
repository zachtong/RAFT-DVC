"""
Trainer class for RAFT-DVC.

Manages the training loop, logging, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging

from ..core import RAFTDVC
from .loss import SequenceLoss, CombinedLoss


class Trainer:
    """
    Training manager for RAFT-DVC.
    
    Handles the training loop, validation, logging, and checkpointing.
    
    Args:
        model: RAFT-DVC model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation
        output_dir: Directory to save checkpoints and logs
        config: Training configuration dictionary
    """
    
    def __init__(
        self,
        model: RAFTDVC,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = './results',
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'epochs': 100,
            'lr': 4e-4,
            'weight_decay': 1e-4,
            'gamma': 0.8,  # Sequence loss gamma
            'grad_clip': 1.0,
            'iters': 12,
            'log_interval': 10,
            'val_interval': 1,
            'save_interval': 10,
            'mixed_precision': False,
        }
        if config is not None:
            self.config.update(config)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'],
            total_steps=total_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='linear'
        )
        
        # Setup loss
        self.criterion = SequenceLoss(gamma=self.config['gamma'])
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
        
        # Logging
        self.logger = logging.getLogger('RAFT-DVC')
        self._setup_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / 'training.log'
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            vol0 = batch['vol0'].to(self.device)
            vol1 = batch['vol1'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    flow_preds = self.model(vol0, vol1, iters=self.config['iters'])
                    loss = self.criterion(flow_preds, gt_flow)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                flow_preds = self.model(vol0, vol1, iters=self.config['iters'])
                loss = self.criterion(flow_preds, gt_flow)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()
            
            # Logging
            if batch_idx % self.config['log_interval'] == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} LR: {lr:.6f}"
                )
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        total_epe = 0.0
        
        for batch in self.val_loader:
            vol0 = batch['vol0'].to(self.device)
            vol1 = batch['vol1'].to(self.device)
            gt_flow = batch['flow'].to(self.device)
            
            flow_preds = self.model(vol0, vol1, iters=self.config['iters'])
            loss = self.criterion(flow_preds, gt_flow)
            
            # Compute EPE for final prediction
            final_flow = flow_preds[-1]
            epe = torch.sqrt(torch.sum((final_flow - gt_flow) ** 2, dim=1)).mean()
            
            total_loss += loss.item()
            total_epe += epe.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_epe = total_epe / len(self.val_loader)
        
        self.logger.info(
            f"Validation - Loss: {avg_loss:.4f} EPE: {avg_epe:.4f}"
        )
        
        return avg_loss
    
    def save_checkpoint(self, filename: str = 'checkpoint.pth', is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / filename
        
        self.model.save_checkpoint(
            str(checkpoint_path),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            config=self.config
        )
        
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            self.model.save_checkpoint(str(best_path), epoch=self.epoch)
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")
    
    def train(self):
        """Run full training."""
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        self.logger.info(f"Config: {self.config}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['epochs']):
            self.epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            if self.val_loader is not None and epoch % self.config['val_interval'] == 0:
                val_loss = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
            
            # Save periodic checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')
            
            # Always save latest
            self.save_checkpoint('latest.pth')
            
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Train Loss: {train_loss:.4f}"
            )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.1f} hours")
        
        return self.best_val_loss
