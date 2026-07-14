"""
RAFT-DVC: Deep Learning for Digital Volume Correlation

Main model class that integrates all components for 3D optical flow estimation.
Based on VolRAFT (CVPR 2024 Workshop) with improvements for usability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, field

from .extractor import (
    BasicEncoder,
    MediumEncoder,
    ShallowEncoder,
    FullResEncoder,
    ContextEncoder
)
from .update import BasicUpdateBlock
from .corr import CorrBlock, coords_grid_3d, upflow_3d

# NOTE: ``CorrBlockOnTheFly`` and ``CorrBlockCUDA`` are imported lazily
# inside ``forward()`` so this module loads even when those optional
# files are absent (e.g. on TACC Vista where the CUDA OTF kernel and
# pure-PyTorch OTF source are not deployed because all training uses
# the standard CorrBlock).


@dataclass
class RAFTDVCConfig:
    """Configuration for RAFT-DVC model."""

    # Input/output configuration
    input_channels: int = 1  # Grayscale volume

    # Encoder type selection
    encoder_type: str = '1/8'  # Options: '1/1', '1/2', '1/4', '1/8'

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

    # Correlation implementation:
    #   "standard"     -- pre-compute full all-pairs corr (memory O(H^6))
    #   "on_the_fly"   -- compute corr per-lookup (memory O(C * H^3 * K))
    # See docs/plans/2026-05-14-on-the-fly-corr.md
    corr_impl: str = "standard"
    corr_chunk_size: int = 27
    corr_use_checkpoint: bool = True

    # Correlation-sampler coordinate convention:
    #   2 -- fixed convention (correct: querying (h, w, d) samples vol[h, w, d])
    #   1 -- legacy convention with a W<->D axis swap in bilinear_sampler_3d.
    #        Checkpoints trained before 2026-07-12 are co-adapted to the swap
    #        and only work with version 1 (load_checkpoint sets this
    #        automatically when the saved config lacks the field).
    corr_sampler_version: int = 2

    # Update block configuration
    use_sep_conv: bool = True

    # Gradient-checkpoint the per-iteration update block to cut the O(iters)
    # activation graph to one iteration's worth.  Lets the memory-heavy fm32
    # (1/2 encoder) case use the fast standard CorrBlock instead of the slow
    # on-the-fly path.  Default off (byte-identical forward; only affects backward).
    checkpoint_updates: bool = False

    # Gradient-checkpoint the correlation LOOKUP (grid_sample) each iteration, so
    # its O(iters) grad_input buffers are recomputed in backward instead of stored.
    # Together with checkpoint_updates this brings fm32 (1/2 @ input64) standard
    # correlation to ~24 GB at batch 1 (vs OOM).  Default off.
    checkpoint_corr: bool = False

    # Inference configuration
    iters: int = 12

    # Training configuration
    mixed_precision: bool = False

    # Uncertainty estimation
    # uncertainty_mode: "none" | "nll" | "mol"
    uncertainty_mode: str = "none"

    def __post_init__(self):
        """Validate uncertainty_mode and corr_impl."""
        if self.uncertainty_mode not in ("none", "nll", "mol"):
            raise ValueError(
                f"Invalid uncertainty_mode: {self.uncertainty_mode!r}. "
                f"Must be one of: 'none', 'nll', 'mol'"
            )
        if self.corr_impl not in (
            "standard", "on_the_fly",
            "cuda_otf", "cuda_otf_v3", "cuda_otf_v4", "cuda_otf_v5",
        ):
            raise ValueError(
                f"Invalid corr_impl: {self.corr_impl!r}. "
                f"Must be one of: 'standard', 'on_the_fly', 'cuda_otf', "
                f"'cuda_otf_v3', 'cuda_otf_v4', 'cuda_otf_v5'"
            )
        if self.corr_sampler_version not in (1, 2):
            raise ValueError(
                f"Invalid corr_sampler_version: {self.corr_sampler_version!r}. "
                f"Must be 1 (legacy W<->D-swapped) or 2 (fixed)."
            )
        if self.corr_sampler_version == 2 and self.corr_impl != "standard":
            raise ValueError(
                f"corr_impl={self.corr_impl!r} has not been ported to the "
                f"fixed sampler convention (corr_sampler_version=2). Use "
                f"corr_impl='standard', or corr_sampler_version=1 for legacy "
                f"checkpoints."
            )

    @property
    def predict_uncertainty(self) -> bool:
        """Derived property for backward compatibility."""
        return self.uncertainty_mode != "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_channels': self.input_channels,
            'encoder_type': self.encoder_type,
            'feature_dim': self.feature_dim,
            'encoder_norm': self.encoder_norm,
            'encoder_dropout': self.encoder_dropout,
            'hidden_dim': self.hidden_dim,
            'context_dim': self.context_dim,
            'context_norm': self.context_norm,
            'context_dropout': self.context_dropout,
            'corr_levels': self.corr_levels,
            'corr_radius': self.corr_radius,
            'corr_impl': self.corr_impl,
            'corr_chunk_size': self.corr_chunk_size,
            'corr_use_checkpoint': self.corr_use_checkpoint,
            'corr_sampler_version': self.corr_sampler_version,
            'use_sep_conv': self.use_sep_conv,
            'checkpoint_updates': self.checkpoint_updates,
            'checkpoint_corr': self.checkpoint_corr,
            'iters': self.iters,
            'mixed_precision': self.mixed_precision,
            'uncertainty_mode': self.uncertainty_mode,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAFTDVCConfig':
        """Create config from dictionary.

        Handles backward compatibility: old checkpoints store
        ``predict_uncertainty: True`` instead of ``uncertainty_mode``.
        """
        d = {k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__}
        # Migrate legacy predict_uncertainty flag
        if 'uncertainty_mode' not in d and config_dict.get('predict_uncertainty', False):
            d['uncertainty_mode'] = 'nll'
        return cls(**d)


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

        # Select encoder class based on encoder_type
        encoder_map = {
            '1/1': FullResEncoder,
            '1/2': ShallowEncoder,
            '1/4': MediumEncoder,
            '1/8': BasicEncoder
        }

        if self.config.encoder_type not in encoder_map:
            raise ValueError(
                f"Invalid encoder_type: {self.config.encoder_type}. "
                f"Must be one of: {list(encoder_map.keys())}"
            )

        encoder_class = encoder_map[self.config.encoder_type]

        # Feature encoder (shared for both volumes)
        self.fnet = encoder_class(
            input_dim=self.config.input_channels,
            output_dim=self.config.feature_dim,
            norm_fn=self.config.encoder_norm,
            dropout=self.config.encoder_dropout
        )

        # Context encoder (only for reference volume)
        # Uses the same encoder class as fnet
        self.cnet = ContextEncoder(
            input_dim=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            norm_fn=self.config.context_norm,
            dropout=self.config.context_dropout
        )
        # Override the encoder inside ContextEncoder to match fnet's type
        self.cnet.encoder = encoder_class(
            input_dim=self.config.input_channels,
            output_dim=self.config.hidden_dim + self.config.context_dim,
            norm_fn=self.config.context_norm,
            dropout=self.config.context_dropout
        )

        # Update block
        self.update_block = BasicUpdateBlock(
            corr_levels=self.config.corr_levels,
            corr_radius=self.config.corr_radius,
            hidden_dim=self.config.hidden_dim,
            context_dim=self.config.context_dim,
            use_sep_conv=self.config.use_sep_conv,
            uncertainty_mode=self.config.uncertainty_mode
        )

        # For mixed precision training
        self._use_amp = self.config.mixed_precision

        # Compute downsample factor based on encoder type
        self.downsample_factor = int(self.config.encoder_type.split('/')[1])
    
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
            Tuple of (coords0, coords1) at encoder's output resolution
        """
        B, C, H, W, D = volume.shape

        # Feature maps are at 1/downsample_factor resolution
        ht = int(np.ceil(H / self.downsample_factor))
        wd = int(np.ceil(W / self.downsample_factor))
        dp = int(np.ceil(D / self.downsample_factor))

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

        # Build correlation block — standard precomputed, pure-PyTorch
        # on-the-fly, or CUDA on-the-fly (fastest with low memory).
        if self.config.corr_impl == "cuda_otf_v5":
            from .corr_otf_cuda import CorrBlockCUDAv5
            corr_fn = CorrBlockCUDAv5(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
            )
        elif self.config.corr_impl == "cuda_otf_v4":
            from .corr_otf_cuda import CorrBlockCUDAv4
            corr_fn = CorrBlockCUDAv4(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
            )
        elif self.config.corr_impl == "cuda_otf_v3":
            from .corr_otf_cuda import CorrBlockCUDAv3
            corr_fn = CorrBlockCUDAv3(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
            )
        elif self.config.corr_impl == "cuda_otf":
            from .corr_otf_cuda import CorrBlockCUDA
            corr_fn = CorrBlockCUDA(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
            )
        elif self.config.corr_impl == "on_the_fly":
            from .corr_otf import CorrBlockOnTheFly
            corr_fn = CorrBlockOnTheFly(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
                chunk_size=self.config.corr_chunk_size,
                use_checkpoint=self.config.corr_use_checkpoint,
            )
        else:
            corr_fn = CorrBlock(
                fmap0,
                fmap1,
                num_levels=self.config.corr_levels,
                radius=self.config.corr_radius,
                legacy_wd_swap=(self.config.corr_sampler_version == 1),
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
        uncertainty_predictions = []
        target_shape = (vol0.shape[2], vol0.shape[3], vol0.shape[4])

        for _ in range(iters):
            coords1 = coords1.detach()

            # Lookup correlation (optionally gradient-checkpointed so the
            # per-iteration grid_sample backward buffers are not retained --
            # required to fit fm32 standard correlation on a 32 GB GPU).
            if (getattr(self.config, 'checkpoint_corr', False)
                    and self.training and torch.is_grad_enabled()):
                corr = _ckpt(corr_fn, coords1, use_reentrant=False)
            else:
                corr = corr_fn(coords1)

            # Current flow estimate
            flow = coords1 - coords0

            # Update step.  Optionally gradient-checkpoint the update block so the
            # O(iters) activation graph collapses to one iteration's worth -- this
            # lets the memory-heavy fm32 (1/2) case use the fast standard CorrBlock
            # instead of the slow on-the-fly path.  Default off.
            _do_ckpt = (getattr(self.config, 'checkpoint_updates', False)
                        and self.training and torch.is_grad_enabled())
            if self._use_amp:
                with torch.cuda.amp.autocast():
                    if _do_ckpt:
                        net, delta_flow, log_b = _ckpt(
                            self.update_block, net, context, corr, flow,
                            use_reentrant=False)
                    else:
                        net, delta_flow, log_b = self.update_block(
                            net, context, corr, flow
                        )
            else:
                if _do_ckpt:
                    net, delta_flow, log_b = _ckpt(
                        self.update_block, net, context, corr, flow,
                        use_reentrant=False)
                else:
                    net, delta_flow, log_b = self.update_block(
                        net, context, corr, flow
                    )

            # Update coordinates
            coords1 = coords1 + delta_flow

            # Upsample flow to full resolution
            flow_up = upflow_3d(coords1 - coords0, target_shape=target_shape)
            flow_predictions.append(flow_up)

            # Upsample uncertainty if present
            if log_b is not None:
                log_b_up = upflow_3d(log_b, target_shape=target_shape)
                uncertainty_predictions.append(log_b_up)

        if test_mode:
            if uncertainty_predictions:
                return (coords1 - coords0, flow_up,
                        uncertainty_predictions[-1])
            return coords1 - coords0, flow_up

        if uncertainty_predictions:
            return flow_predictions, uncertainty_predictions

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
            **kwargs: Additional data to save (e.g., training_config, best_val_loss)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config.to_dict(),  # Model architecture config
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

        # save_checkpoint writes the architecture under 'model_config'; very old
        # checkpoints used 'config'.  Accept both -- loading with the WRONG key
        # used to raise KeyError here, and silently mismatched configs are the
        # bug class that once shipped a random network in the 2D GUI.
        config_dict = checkpoint.get('model_config', checkpoint.get('config'))
        if config_dict is None:
            raise KeyError(
                f"Checkpoint {path!r} has neither 'model_config' nor 'config'; "
                f"available keys: {sorted(checkpoint.keys())}"
            )
        # Checkpoints saved before 2026-07-12 predate corr_sampler_version and
        # were trained with the legacy W<->D-swapped sampler; they are
        # co-adapted to it and must keep using it (version 1).  Only inject
        # here, in the checkpoint path -- fresh configs default to version 2.
        if 'corr_sampler_version' not in config_dict:
            config_dict = dict(config_dict, corr_sampler_version=1)
        config = RAFTDVCConfig.from_dict(config_dict)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        return model, checkpoint
