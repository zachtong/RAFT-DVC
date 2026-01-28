"""
RAFT-DVC: Deep Learning for Digital Volume Correlation

Main model class that integrates all components for 3D optical flow estimation.
Based on VolRAFT (CVPR 2024 Workshop) with improvements for usability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, field

from .extractor import BasicEncoder, ContextEncoder
from .update import BasicUpdateBlock
from .corr import CorrBlock, coords_grid_3d, upflow_3d


@dataclass
class RAFTDVCConfig:
    """Configuration for RAFT-DVC model."""
    
    # Input/output configuration
    input_channels: int = 1  # Grayscale volume
    
    # Feature encoder configuration
    feature_dim: int = 128
    encoder_norm: str = 'instance'
    encoder_dropout: float = 0.0
    
    # Context encoder configuration
    hidden_dim: int = 96
    context_dim: int = 64
    context_norm: str = 'none'
    context_dropout: float = 0.0
    
    # Correlation configuration
    corr_levels: int = 4
    corr_radius: int = 4
    
    # Update block configuration
    use_sep_conv: bool = True
    
    # Inference configuration
    iters: int = 12
    
    # Training configuration
    mixed_precision: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_channels': self.input_channels,
            'feature_dim': self.feature_dim,
            'encoder_norm': self.encoder_norm,
            'encoder_dropout': self.encoder_dropout,
            'hidden_dim': self.hidden_dim,
            'context_dim': self.context_dim,
            'context_norm': self.context_norm,
            'context_dropout': self.context_dropout,
            'corr_levels': self.corr_levels,
            'corr_radius': self.corr_radius,
            'use_sep_conv': self.use_sep_conv,
            'iters': self.iters,
            'mixed_precision': self.mixed_precision,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAFTDVCConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class RAFTDVC(nn.Module):
    """
    RAFT-DVC: Recurrent All-Pairs Field Transforms for Digital Volume Correlation.
    
    A deep learning model for estimating 3D displacement fields between
    two volumetric images (reference and deformed volumes).
    
    Args:
        config: Model configuration. If None, uses default configuration.
    
    Example:
        >>> config = RAFTDVCConfig(iters=12)
        >>> model = RAFTDVC(config)
        >>> vol0 = torch.randn(1, 1, 64, 64, 64)  # Reference volume
        >>> vol1 = torch.randn(1, 1, 64, 64, 64)  # Deformed volume
        >>> flow_predictions = model(vol0, vol1)
        >>> final_flow = flow_predictions[-1]  # Shape: (1, 3, 64, 64, 64)
    """
    
    def __init__(self, config: Optional[RAFTDVCConfig] = None):
        super().__init__()
        
        self.config = config or RAFTDVCConfig()
        
        # Feature encoder (shared for both volumes)
        self.fnet = BasicEncoder(
            input_dim=self.config.input_channels,
            output_dim=self.config.feature_dim,
            norm_fn=self.config.encoder_norm,
            dropout=self.config.encoder_dropout
        )
        
        # Context encoder (only for reference volume)
        self.cnet = ContextEncoder(
            input_dim=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            norm_fn=self.config.context_norm,
            dropout=self.config.context_dropout
        )
        
        # Update block
        self.update_block = BasicUpdateBlock(
            corr_levels=self.config.corr_levels,
            corr_radius=self.config.corr_radius,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            use_sep_conv=self.config.use_sep_conv
        )
        
        # For mixed precision training
        self._use_amp = self.config.mixed_precision
    
    def freeze_bn(self):
        """Freeze batch normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()
    
    def _initialize_flow(
        self, 
        volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize flow coordinates.
        
        Flow is represented as difference between two coordinate grids.
        
        Args:
            volume: Input volume. Shape: (B, C, H, W, D)
        
        Returns:
            Tuple of (coords0, coords1) at 1/8 resolution
        """
        B, C, H, W, D = volume.shape
        
        # Feature maps are 1/8 resolution
        ht = int(np.ceil(H / 8))
        wd = int(np.ceil(W / 8))
        dp = int(np.ceil(D / 8))
        
        coords0 = coords_grid_3d(B, ht, wd, dp, device=volume.device)
        coords1 = coords_grid_3d(B, ht, wd, dp, device=volume.device)
        
        return coords0, coords1
    
    def _normalize_volumes(
        self, 
        vol0: torch.Tensor, 
        vol1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize volume pair to [0, 1] range.
        
        Uses shared min/max across both volumes for consistent normalization.
        
        Args:
            vol0, vol1: Input volumes. Shape: (B, C, H, W, D)
        
        Returns:
            Normalized volumes
        """
        # Stack volumes and compute global min/max
        stacked = torch.stack([vol0, vol1], dim=-1)
        vol_min = stacked.amin(dim=(2, 3, 4, 5), keepdim=True)
        vol_max = stacked.amax(dim=(2, 3, 4, 5), keepdim=True)
        
        # Squeeze last dimension and expand to volume shape
        vol_min = vol_min.squeeze(-1)
        vol_max = vol_max.squeeze(-1)
        
        # Normalize
        denom = vol_max - vol_min
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        
        vol0_norm = (vol0 - vol_min) / denom
        vol1_norm = (vol1 - vol_min) / denom
        
        return vol0_norm.contiguous(), vol1_norm.contiguous()
    
    def forward(
        self, 
        vol0: torch.Tensor, 
        vol1: torch.Tensor, 
        iters: Optional[int] = None,
        flow_init: Optional[torch.Tensor] = None,
        test_mode: bool = False
    ) -> List[torch.Tensor]:
        """
        Forward pass: estimate optical flow between two volumes.
        
        Args:
            vol0: Reference volume. Shape: (B, C, H, W, D)
            vol1: Deformed volume. Shape: (B, C, H, W, D)
            iters: Number of refinement iterations. If None, uses config value.
            flow_init: Initial flow estimate. If None, starts from zero.
            test_mode: If True, returns only final flow (more memory efficient).
        
        Returns:
            List of flow predictions at each iteration. Each has shape (B, 3, H, W, D).
            If test_mode=True, returns [(low_res_flow, high_res_flow)].
        """
        iters = iters or self.config.iters
        
        # Normalize input volumes
        vol0, vol1 = self._normalize_volumes(vol0, vol1)
        
        # Extract features
        if self._use_amp:
            with torch.cuda.amp.autocast():
                fmap0, fmap1 = self.fnet([vol0, vol1])
        else:
            fmap0, fmap1 = self.fnet([vol0, vol1])
        
        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        
        # Build correlation pyramid
        corr_fn = CorrBlock(
            fmap0, 
            fmap1, 
            num_levels=self.config.corr_levels, 
            radius=self.config.corr_radius
        )
        
        # Extract context features
        if self._use_amp:
            with torch.cuda.amp.autocast():
                net, context = self.cnet(vol0)
        else:
            net, context = self.cnet(vol0)
        
        # Initialize flow
        coords0, coords1 = self._initialize_flow(vol0)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init
        
        # Iterative refinement
        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            
            # Lookup correlation
            corr = corr_fn(coords1)
            
            # Current flow estimate
            flow = coords1 - coords0
            
            # Update step
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    net, delta_flow = self.update_block(net, context, corr, flow)
            else:
                net, delta_flow = self.update_block(net, context, corr, flow)
            
            # Update coordinates
            coords1 = coords1 + delta_flow
            
            # Upsample flow to full resolution
            flow_up = upflow_3d(
                coords1 - coords0, 
                target_shape=(vol0.shape[2], vol0.shape[3], vol0.shape[4])
            )
            
            flow_predictions.append(flow_up)
        
        if test_mode:
            return coords1 - coords0, flow_up
        
        return flow_predictions
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(
        self, 
        path: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            optimizer: Optimizer state to save
            scheduler: Scheduler state to save
            epoch: Current epoch number
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'epoch': epoch,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(
        cls, 
        path: str, 
        device: Optional[torch.device] = None
    ) -> Tuple['RAFTDVC', Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load model to
        
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device)
        
        config = RAFTDVCConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model, checkpoint
