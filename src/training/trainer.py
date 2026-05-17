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
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR, CosineAnnealingLR
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import logging

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:  # tensorboard not installed
    SummaryWriter = None  # type: ignore[assignment]
    _TB_AVAILABLE = False

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
            'weight_decay': 5e-5,  # Reduced to match volRAFT (5e-5 instead of 1e-4)
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
        # Force CPU if CUDA compute capability is incompatible (e.g., RTX 5090 sm_120)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                # Test with operations that actually use CUDA kernels
                # Simple operations like torch.zeros() may work even when complex kernels fail
                x = torch.randn(2, 3, 4, 4, 4).cuda()
                y = torch.randn(2, 3, 4, 4, 4).cuda()
                z = torch.stack([x, y], dim=-1)  # This fails on incompatible GPUs
                _ = torch.nn.functional.conv3d(x, torch.randn(3, 3, 3, 3, 3).cuda())
                del x, y, z, _  # Clean up
            except RuntimeError as e:
                error_msg = str(e)
                if "no kernel image is available" in error_msg or "CUDA error" in error_msg:
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown'
                    print(f"\n{'='*60}")
                    print(f"WARNING: CUDA incompatible with your GPU")
                    print(f"GPU: {gpu_name}")
                    print(f"Error: {error_msg[:100]}...")
                    print(f"Falling back to CPU mode.")
                    print(f"For GPU support, wait for PyTorch with full sm_120 support.")
                    print(f"{'='*60}\n")
                    use_cuda = False

        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler — three supported types selected by `scheduler_type`:
        #   - cyclic   : CyclicLR(triangular2)   (legacy, volRAFT-inherited)
        #   - onecycle : OneCycleLR              (single warmup-then-anneal)
        #   - cosine   : CosineAnnealingLR       (monotonic cooldown)
        # `step_per` records whether to call scheduler.step() per iter or per epoch.
        sched_type = self.config.get('scheduler_type', 'cyclic')
        iters_per_epoch = max(1, len(self.train_loader))
        total_steps = int(self.config['epochs']) * iters_per_epoch

        if sched_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=float(self.config.get('max_lr', 4e-4)),
                total_steps=total_steps,
                pct_start=float(self.config.get('pct_start', 0.05)),
                anneal_strategy=str(self.config.get('anneal_strategy', 'cos')),
                div_factor=float(self.config.get('div_factor', 25.0)),
                final_div_factor=float(self.config.get('final_div_factor', 1e4)),
                cycle_momentum=False,
            )
            self._sched_step_per = 'iter'
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=int(self.config['epochs']),
                eta_min=float(self.config.get('min_lr', 1e-6)),
            )
            self._sched_step_per = 'epoch'
        else:  # cyclic (backward compatible default)
            base_lr = float(self.config.get('base_lr', 2e-5))
            max_lr = float(self.config.get('max_lr', 2e-4))
            step_size_up = int(self.config.get('step_size_up', 500))
            step_size_down = int(self.config.get('step_size_down', 1500))
            self.scheduler = CyclicLR(
                self.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size_up,
                step_size_down=step_size_down,
                mode='triangular2',
                cycle_momentum=False,
            )
            self._sched_step_per = 'iter'
        
        # Setup loss
        self.criterion = SequenceLoss(gamma=self.config['gamma'])

        # Mixed precision (new torch.amp API; falls back if old torch missing GradScaler)
        if self.config['mixed_precision']:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except TypeError:  # very old torch
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Logging
        self.logger = logging.getLogger('RAFT-DVC')
        self._setup_logging()

        # TensorBoard writer (best-effort -- skip silently if tb missing).
        self.tb_writer = None
        if _TB_AVAILABLE:
            tb_dir = self.output_dir / 'logs'
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))

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
            
            # `step_taken` tracks whether optimizer.step() actually ran this
            # iteration -- AMP scaler can skip it on inf/NaN gradients, in which
            # case we must NOT advance the LR scheduler (avoids the
            # "lr_scheduler.step() before optimizer.step()" warning + drift).
            if self.scaler is not None:
                # Pre-step LR snapshot so we can detect a real step via scaler.
                pre_step_scale = self.scaler.get_scale()
                with torch.amp.autocast('cuda'):
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
                # scaler.update() decreases the scale on bad steps; equal scale
                # implies optimizer.step() executed normally.
                step_taken = self.scaler.get_scale() >= pre_step_scale
            else:
                flow_preds = self.model(vol0, vol1, iters=self.config['iters'])
                loss = self.criterion(flow_preds, gt_flow)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
                self.optimizer.step()
                step_taken = True

            # Step scheduler per iteration for cyclic / onecycle; cosine steps per epoch.
            if step_taken and self._sched_step_per == 'iter':
                self.scheduler.step()
            self.global_step += 1
            total_loss += loss.item()

            # TensorBoard per-iter logging — keep LR only (needed to verify
            # OneCycleLR / CyclicLR shape).  Per-iter loss is dropped because
            # the IO cost dwarfs the signal value (250 writes/epoch).
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

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

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('val/loss', avg_loss, self.epoch)
            self.tb_writer.add_scalar('val/epe', avg_epe, self.epoch)

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
            training_config=self.config  # Save as training_config, not config
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
        
        # Checkpoint stores the *last completed* epoch (save_checkpoint is called
        # after train_epoch + validation finishes). The training loop iterates
        # range(self.epoch, max_epochs), so resume must start at last+1 to avoid
        # re-training the last completed epoch -- which also keeps scheduler step
        # count aligned with OneCycleLR's strict total_steps budget.
        last_completed = checkpoint.get('epoch', -1)
        self.epoch = last_completed + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.logger.info(
            f"Loaded checkpoint from {path} "
            f"(last completed epoch={last_completed}, resuming at epoch={self.epoch})"
        )
    
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

            # Per-epoch scheduler step (cosine; cyclic / onecycle step per-iter inside train_epoch).
            if self._sched_step_per == 'epoch':
                self.scheduler.step()

            # Validation
            if self.val_loader is not None and epoch % self.config['val_interval'] == 0:
                val_loss = self.validate()
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth', is_best=True)
            
            # Save periodic rotating checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch:04d}.pth')

            # Refresh latest.pth on a slower cadence -- writing every epoch
            # measurably slows OneDrive-synced repos and adds little safety
            # value (best_model.pth still saves on every val improvement).
            latest_interval = int(self.config.get('latest_interval', 10))
            if epoch % latest_interval == 0 or (epoch + 1) == self.config['epochs']:
                self.save_checkpoint('latest.pth')
            
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s - "
                f"Train Loss: {train_loss:.4f}"
            )

            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train/loss_epoch', train_loss, epoch)
                self.tb_writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.1f} hours")

        if self.tb_writer is not None:
            self.tb_writer.close()

        return self.best_val_loss
