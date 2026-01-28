"""
Loss functions for RAFT-DVC training.

Implements sequence loss and end-point-error metrics for flow estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EPELoss(nn.Module):
    """
    End-Point-Error (EPE) Loss.
    
    Computes the L2 distance between predicted and ground truth flow.
    This is the standard metric for optical flow evaluation.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        pred_flow: torch.Tensor, 
        gt_flow: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute EPE loss.
        
        Args:
            pred_flow: Predicted flow. Shape: (B, 3, H, W, D)
            gt_flow: Ground truth flow. Shape: (B, 3, H, W, D)
            mask: Optional mask for valid regions. Shape: (B, 1, H, W, D)
        
        Returns:
            EPE loss (scalar)
        """
        # L2 distance
        epe = torch.sqrt(torch.sum((pred_flow - gt_flow) ** 2, dim=1, keepdim=True))
        
        if mask is not None:
            epe = epe * mask
            if self.reduction == 'mean':
                return epe.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return epe.sum()
            else:
                return epe
        else:
            if self.reduction == 'mean':
                return epe.mean()
            elif self.reduction == 'sum':
                return epe.sum()
            else:
                return epe


class SequenceLoss(nn.Module):
    """
    Sequence Loss for RAFT-DVC.
    
    Applies exponentially increasing weights to flow predictions
    at each iteration, emphasizing later (more refined) predictions.
    
    Args:
        gamma: Exponential weighting factor (default: 0.8)
        max_flow: Maximum flow value for gradient clipping (optional)
    """
    
    def __init__(
        self, 
        gamma: float = 0.8,
        max_flow: Optional[float] = None
    ):
        super().__init__()
        self.gamma = gamma
        self.max_flow = max_flow
        self.epe = EPELoss(reduction='none')
    
    def forward(
        self, 
        flow_preds: List[torch.Tensor], 
        gt_flow: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sequence loss.
        
        Args:
            flow_preds: List of flow predictions at each iteration.
                        Each has shape (B, 3, H, W, D)
            gt_flow: Ground truth flow. Shape: (B, 3, H, W, D)
            mask: Optional mask for valid regions. Shape: (B, 1, H, W, D)
        
        Returns:
            Total sequence loss (scalar)
        """
        n_predictions = len(flow_preds)
        total_loss = 0.0
        
        for i, flow_pred in enumerate(flow_preds):
            # Exponentially increasing weight
            weight = self.gamma ** (n_predictions - i - 1)
            
            # Compute EPE for this prediction
            epe = self.epe(flow_pred, gt_flow, mask)
            
            # Apply mask if provided
            if mask is not None:
                loss = (epe * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = epe.mean()
            
            total_loss += weight * loss
        
        return total_loss


class SmoothLoss(nn.Module):
    """
    Smoothness loss for flow regularization.
    
    Encourages spatially smooth displacement fields
    while preserving edges at intensity boundaries.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self, 
        flow: torch.Tensor, 
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute smoothness loss.
        
        Args:
            flow: Flow field. Shape: (B, 3, H, W, D)
            image: Optional reference image for edge-aware smoothing.
                   Shape: (B, C, H, W, D)
        
        Returns:
            Smoothness loss (scalar)
        """
        # Compute flow gradients
        flow_dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        flow_dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        flow_dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        
        if image is not None:
            # Edge-aware weighting
            img_dy = torch.abs(image[:, :, 1:, :, :] - image[:, :, :-1, :, :]).mean(dim=1, keepdim=True)
            img_dx = torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :]).mean(dim=1, keepdim=True)
            img_dz = torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1]).mean(dim=1, keepdim=True)
            
            weight_y = torch.exp(-img_dy)
            weight_x = torch.exp(-img_dx)
            weight_z = torch.exp(-img_dz)
            
            smooth_y = (weight_y * torch.abs(flow_dy)).mean()
            smooth_x = (weight_x * torch.abs(flow_dx)).mean()
            smooth_z = (weight_z * torch.abs(flow_dz)).mean()
        else:
            smooth_y = torch.abs(flow_dy).mean()
            smooth_x = torch.abs(flow_dx).mean()
            smooth_z = torch.abs(flow_dz).mean()
        
        return self.weight * (smooth_y + smooth_x + smooth_z)


class CombinedLoss(nn.Module):
    """
    Combined loss for RAFT-DVC training.
    
    Combines sequence EPE loss with optional smoothness regularization.
    """
    
    def __init__(
        self,
        gamma: float = 0.8,
        smooth_weight: float = 0.0
    ):
        super().__init__()
        self.sequence_loss = SequenceLoss(gamma=gamma)
        self.smooth_loss = SmoothLoss(weight=smooth_weight) if smooth_weight > 0 else None
    
    def forward(
        self,
        flow_preds: List[torch.Tensor],
        gt_flow: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            flow_preds: List of flow predictions
            gt_flow: Ground truth flow
            image: Reference image for smoothness (optional)
            mask: Valid region mask (optional)
        
        Returns:
            Dictionary with 'total', 'epe', and optionally 'smooth' losses
        """
        epe_loss = self.sequence_loss(flow_preds, gt_flow, mask)
        
        losses = {
            'total': epe_loss,
            'epe': epe_loss.detach()
        }
        
        if self.smooth_loss is not None and image is not None:
            smooth = self.smooth_loss(flow_preds[-1], image)
            losses['total'] = losses['total'] + smooth
            losses['smooth'] = smooth.detach()
        
        return losses
