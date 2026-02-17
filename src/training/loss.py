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

    Computes the L1 distance between predicted and ground truth flow.
    This matches volRAFT's implementation for more robust training.
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
        # L1 distance (absolute difference) - matches volRAFT implementation
        # Sum absolute differences across flow components (3 channels)
        epe = torch.sum(torch.abs(pred_flow - gt_flow), dim=1, keepdim=True)

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


class LaplacianSmoothLoss(nn.Module):
    """
    Multi-scale Laplacian smoothness loss using 3D Laplacian operator.

    Penalizes the absolute Laplacian (L1) of the flow field at one or
    more spatial scales, encouraging smooth predictions at all scales.

    For each stride s, builds a (2s+1)^3 kernel where:
        center = -6, 6 face-adjacent neighbors at distance s = +1.
    Applied to each flow component (u, v, w) independently via groups.

    Args:
        strides: List of spatial strides. Default [1, 2, 4] for multi-scale.
                 Each stride s compares voxels s apart (kernel size = 2s+1).
    """

    def __init__(self, strides=None):
        super().__init__()
        if strides is None:
            strides = [1, 2, 4]
        self.strides = strides

        for s in strides:
            kernel = self._build_kernel(s)
            # Repeat for 3 flow components (groups=3)
            kernel = kernel.repeat(3, 1, 1, 1, 1)  # (3, 1, k, k, k)
            self.register_buffer(f'kernel_s{s}', kernel)

    @staticmethod
    def _build_kernel(stride: int) -> torch.Tensor:
        """Build Laplacian kernel for given stride."""
        k = 2 * stride + 1
        c = stride  # center index
        kernel = torch.zeros(1, 1, k, k, k)
        kernel[0, 0, c, c, c - stride] = 1.0  # z-
        kernel[0, 0, c, c, c + stride] = 1.0  # z+
        kernel[0, 0, c, c - stride, c] = 1.0  # y-
        kernel[0, 0, c, c + stride, c] = 1.0  # y+
        kernel[0, 0, c - stride, c, c] = 1.0  # x-
        kernel[0, 0, c + stride, c, c] = 1.0  # x+
        kernel[0, 0, c, c, c] = -6.0           # center
        return kernel

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Compute mean absolute Laplacian (L1) across all scales.

        Args:
            flow: (B, 3, D, H, W) predicted flow

        Returns:
            Scalar smoothness loss (average over all scales)
        """
        total = 0.0
        for s in self.strides:
            kernel = getattr(self, f'kernel_s{s}')
            pad = s  # padding = stride to maintain spatial size
            laplacian = F.conv3d(flow, kernel, padding=pad, groups=3)
            total += laplacian.abs().mean()
        return total / len(self.strides)


class FeatureDensityWeight(nn.Module):
    """
    Compute per-voxel weight map based on input volume feature density.

    Voxels near bright features (beads) get weight ~1.0.
    Voxels in dark/featureless regions get weight = min_weight.
    Transition is smoothed by Gaussian blur of the binary feature mask.

    Args:
        threshold: Intensity above which a voxel is considered a "feature".
        sigma: Gaussian blur sigma (voxels) controlling influence spread.
        min_weight: Minimum weight for featureless regions.
    """

    def __init__(self, threshold: float = 0.1, sigma: float = 3.0,
                 min_weight: float = 0.2):
        super().__init__()
        self.threshold = threshold
        self.min_weight = min_weight

        # Build 1D Gaussian kernel for separable 3D blur
        kernel_size = 2 * int(2 * sigma + 0.5) + 1
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Store as (1, 1, K) for separable conv
        self.register_buffer('kernel_d', gauss_1d.view(1, 1, -1, 1, 1))
        self.register_buffer('kernel_h', gauss_1d.view(1, 1, 1, -1, 1))
        self.register_buffer('kernel_w', gauss_1d.view(1, 1, 1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, vol0: torch.Tensor) -> torch.Tensor:
        """
        Compute weight map from input volume.

        Args:
            vol0: (B, 1, D, H, W) input volume

        Returns:
            weight: (B, 1, D, H, W) in [min_weight, 1.0]
        """
        # 1. Binary feature mask
        mask = (vol0 > self.threshold).float()

        # 2. Separable 3D Gaussian blur
        p = self.pad
        blurred = F.conv3d(mask, self.kernel_d, padding=(p, 0, 0))
        blurred = F.conv3d(blurred, self.kernel_h, padding=(0, p, 0))
        blurred = F.conv3d(blurred, self.kernel_w, padding=(0, 0, p))

        # 3. Normalize to [0, 1]
        bmax = blurred.amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-8)
        density = blurred / bmax

        # 4. Scale to [min_weight, 1.0]
        weight = self.min_weight + (1.0 - self.min_weight) * density

        return weight


class MaskedSequenceLoss(nn.Module):
    """
    Sequence loss weighted by feature density.

    Per-voxel errors are multiplied by a weight map derived from vol0:
    bright regions (features) get higher weight, dark regions get lower weight.

    Args:
        gamma: Exponential weighting factor for sequence iterations.
        threshold: Intensity threshold for feature detection.
        sigma: Gaussian blur sigma for density map.
        min_weight: Minimum weight for featureless regions.
    """

    def __init__(self, gamma: float = 0.8, threshold: float = 0.1,
                 sigma: float = 3.0, min_weight: float = 0.2):
        super().__init__()
        self.gamma = gamma
        self.density_weight = FeatureDensityWeight(threshold, sigma, min_weight)

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        gt_flow: torch.Tensor,
        vol0: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute density-weighted sequence loss.

        Args:
            flow_preds: List of (B, 3, D, H, W) flow predictions
            gt_flow: (B, 3, D, H, W) ground truth flow
            vol0: (B, 1, D, H, W) input volume for density computation

        Returns:
            Scalar loss
        """
        weight = self.density_weight(vol0)  # (B, 1, D, H, W)

        n_predictions = len(flow_preds)
        total_loss = 0.0

        for i, flow_pred in enumerate(flow_preds):
            i_weight = self.gamma ** (n_predictions - i - 1)

            # Per-voxel L1 error summed over 3 flow components
            err = (flow_pred - gt_flow).abs().sum(dim=1, keepdim=True)  # (B, 1, D, H, W)

            # Weighted mean
            weighted_err = (err * weight).sum() / (weight.sum() + 1e-8)
            total_loss += i_weight * weighted_err

        return total_loss


class NLLSequenceLoss(nn.Module):
    """
    Sequence loss with Laplace NLL for uncertainty estimation.

    Instead of plain L1, uses the negative log-likelihood of a Laplace
    distribution:  L = |error| * exp(-log_b) + log_b
    where log_b is the learned per-voxel, per-component log-scale parameter.

    Args:
        gamma: Exponential weighting factor for sequence iterations.
        log_b_min: Optional floor for log_b. Prevents the model from being
            overconfident (b → 0). If None, log_b is unconstrained.
            Example: log_b_min=log(0.2)≈-1.6 means b >= 0.2.
    """

    def __init__(self, gamma: float = 0.8, log_b_min: float = None):
        super().__init__()
        self.gamma = gamma
        self.log_b_min = log_b_min

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        uncertainty_preds: List[torch.Tensor],
        gt_flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute NLL sequence loss.

        Args:
            flow_preds: List of (B, 3, D, H, W) flow predictions
            uncertainty_preds: List of (B, 3, D, H, W) log_b values
            gt_flow: (B, 3, D, H, W) ground truth flow

        Returns:
            Scalar loss
        """
        n_predictions = len(flow_preds)
        total_loss = 0.0

        for i, (fp, log_b) in enumerate(
            zip(flow_preds, uncertainty_preds)
        ):
            i_weight = self.gamma ** (n_predictions - i - 1)
            if self.log_b_min is not None:
                log_b = log_b.clamp(min=self.log_b_min)
            abs_err = (fp - gt_flow).abs()  # (B, 3, D, H, W)
            nll = abs_err * torch.exp(-log_b) + log_b
            total_loss += i_weight * nll.mean()

        return total_loss


class MoLSequenceLoss(nn.Module):
    """
    Mixture-of-Laplace sequence loss for uncertainty estimation (SEA-RAFT).

    Loss per voxel/component:
        L = -log[ alpha * Lap(err; b1) + (1-alpha) * Lap(err; b2) ]

    where Lap(x; b) = exp(-|x|/b) / (2b).

    - b1 is a fixed scale for the "ordinary" (L1-like) component.
    - b2 = exp(log_b2) is a learned, per-voxel scale for ambiguous pixels.
    - alpha = sigmoid(logit_alpha) is a learned mixing weight.

    The uncertainty head outputs (B, 4, D, H, W):
      channels 0-2 = log_b2 (per flow component),
      channel  3   = logit_alpha (shared across components, broadcast).

    All arithmetic is done in **log-space** via ``logaddexp`` so that
    no intermediate value underflows to float16 subnormal (which would
    be ~100x slower on GPU).  No float32 cast is needed.

    Args:
        gamma: Exponential weighting factor for sequence iterations.
        b1: Fixed scale for the ordinary component.
        log_b2_clamp: (min, max) clamp range for log_b2 (used when use_softplus=False).
        use_softplus: If True, use softplus instead of clamp for log_b2.
            softplus preserves gradients near 0, avoiding gradient death.
            log_b2 = softplus(raw) is always > 0, capped at log_b2_clamp[1].
    """

    def __init__(self, gamma: float = 0.8, b1: float = 1.0,
                 log_b2_clamp: tuple = (0.0, 10.0),
                 use_softplus: bool = False):
        super().__init__()
        self.gamma = gamma
        self.b1 = b1
        self.log_b2_clamp = log_b2_clamp
        self.use_softplus = use_softplus
        # Pre-compute constant log(2 * b1)
        import math
        self.log_2b1 = math.log(2.0 * b1)
        self.log_2 = math.log(2.0)

    def forward(
        self,
        flow_preds: List[torch.Tensor],
        uncertainty_preds: List[torch.Tensor],
        gt_flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MoL sequence loss in log-space.

        Args:
            flow_preds: List of (B, 3, D, H, W) flow predictions.
            uncertainty_preds: List of (B, 4, D, H, W) uncertainty outputs.
            gt_flow: (B, 3, D, H, W) ground truth flow.

        Returns:
            Scalar loss.
        """
        n_predictions = len(flow_preds)
        total_loss = 0.0

        for i, (fp, unc) in enumerate(zip(flow_preds, uncertainty_preds)):
            i_weight = self.gamma ** (n_predictions - i - 1)

            if self.use_softplus:
                # softplus(x) > 0 always, gradient = sigmoid(x) never zero
                log_b2 = F.softplus(unc[:, :3]).clamp(max=self.log_b2_clamp[1])
            else:
                log_b2 = unc[:, :3].clamp(*self.log_b2_clamp)
            logit_alpha = unc[:, 3:4]                                # (B, 1, D, H, W)
            b2 = torch.exp(log_b2)                                   # [1, exp(10)], safe in fp16

            abs_err = (fp - gt_flow).abs()  # (B, 3, D, H, W)

            # Log-space Laplace: log Lap(x; b) = -|x|/b - log(2b)
            # All values are regular-magnitude negatives, no subnormals.
            log_lap1 = -abs_err / self.b1 - self.log_2b1
            log_lap2 = -abs_err / b2 - self.log_2 - log_b2

            # Log-space mixing weights (numerically stable)
            log_alpha = F.logsigmoid(logit_alpha)
            log_1m_alpha = F.logsigmoid(-logit_alpha)

            # log(alpha*Lap1 + (1-alpha)*Lap2) via logaddexp (no exp of tiny values)
            nll = -torch.logaddexp(
                log_alpha + log_lap1,
                log_1m_alpha + log_lap2
            )
            total_loss += i_weight * nll.mean()

        return total_loss


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
