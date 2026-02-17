"""
Data augmentation strategies for sparse feature training.

Provides augmentation modules that can be toggled via training config.
All augmentations operate on GPU tensors in the training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CutoutAugmentation3D(nn.Module):
    """
    Random cuboid masking for 3D volumes with target keep fraction.

    Zeros out random cuboid regions in both vol0 and vol1 simultaneously,
    forcing the model to learn flow prediction from surrounding context.
    The flow GT remains unchanged.

    Each sample gets a random target keep fraction in [keep_min, keep_max],
    then cuboid regions are iteratively removed until the target is reached.

    Args:
        prob: Probability of applying cutout to each sample in the batch
        keep_min: Minimum fraction of voxels to keep (default 0.60)
        keep_max: Maximum fraction of voxels to keep (default 0.85)
        region_size_min: Each cuboid dimension as fraction of volume dim
        region_size_max: Each cuboid dimension as fraction of volume dim
    """

    def __init__(
        self,
        prob: float = 0.7,
        keep_min: float = 0.60,
        keep_max: float = 0.85,
        region_size_min: float = 0.2,
        region_size_max: float = 0.5
    ):
        super().__init__()
        self.prob = prob
        self.keep_min = keep_min
        self.keep_max = keep_max
        self.region_size_min = region_size_min
        self.region_size_max = region_size_max

    def forward(
        self,
        vol0: torch.Tensor,
        vol1: torch.Tensor
    ) -> tuple:
        """
        Apply cutout augmentation to volume pair.

        Same spatial regions are masked in both volumes so the model
        receives consistent (but incomplete) information.

        Args:
            vol0: Reference volume (B, 1, D, H, W)
            vol1: Deformed volume (B, 1, D, H, W)

        Returns:
            Tuple of (masked_vol0, masked_vol1)
        """
        if not self.training:
            return vol0, vol1

        B, C, D, H, W = vol0.shape
        total = D * H * W
        vol0 = vol0.clone()
        vol1 = vol1.clone()

        for b in range(B):
            if torch.rand(1).item() > self.prob:
                continue

            # Random target keep fraction for this sample
            target_keep = torch.empty(1).uniform_(
                self.keep_min, self.keep_max
            ).item()

            # Build mask on CPU (bool), then apply
            mask = torch.ones(D, H, W, dtype=torch.bool)

            for _ in range(200):  # safety limit
                if mask.sum().item() / total <= target_keep:
                    break

                # Random cuboid size
                frac_d = torch.empty(1).uniform_(
                    self.region_size_min, self.region_size_max).item()
                frac_h = torch.empty(1).uniform_(
                    self.region_size_min, self.region_size_max).item()
                frac_w = torch.empty(1).uniform_(
                    self.region_size_min, self.region_size_max).item()

                rd = max(1, int(D * frac_d))
                rh = max(1, int(H * frac_h))
                rw = max(1, int(W * frac_w))

                d0 = torch.randint(0, max(1, D - rd + 1), (1,)).item()
                h0 = torch.randint(0, max(1, H - rh + 1), (1,)).item()
                w0 = torch.randint(0, max(1, W - rw + 1), (1,)).item()

                mask[d0:d0+rd, h0:h0+rh, w0:w0+rw] = False

            # Apply mask to both volumes
            inv_mask = (~mask).to(vol0.device)
            vol0[b, :, inv_mask] = 0.0
            vol1[b, :, inv_mask] = 0.0

        return vol0, vol1

    def __repr__(self):
        return (
            f"CutoutAugmentation3D(prob={self.prob}, "
            f"keep=[{self.keep_min}, {self.keep_max}], "
            f"region_size=[{self.region_size_min}, {self.region_size_max}])"
        )


class GaussianBlur3D(nn.Module):
    """
    3D Gaussian blur for volume preprocessing.

    Uses separable 1D convolutions for efficiency.
    Applied identically to vol0 and vol1; flow GT is unchanged.

    Args:
        sigma: Gaussian sigma in voxels
        kernel_size: Kernel size (auto-computed from sigma if None)
    """

    def __init__(self, sigma: float = 1.0, kernel_size: int = None):
        super().__init__()
        self.sigma = sigma

        if kernel_size is None:
            kernel_size = 2 * math.ceil(2 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Register as buffers (move to device with model)
        # Shape for conv3d: (out_channels, in_channels/groups, kD, kH, kW)
        self.register_buffer(
            'kernel_d', kernel_1d.view(1, 1, -1, 1, 1)
        )
        self.register_buffer(
            'kernel_h', kernel_1d.view(1, 1, 1, -1, 1)
        )
        self.register_buffer(
            'kernel_w', kernel_1d.view(1, 1, 1, 1, -1)
        )

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D Gaussian blur using separable convolutions.

        Args:
            vol: Input volume (B, C, D, H, W)

        Returns:
            Blurred volume (B, C, D, H, W)
        """
        B, C, D, H, W = vol.shape
        pad = self.kernel_size // 2

        # Apply separable 1D convolutions along each axis
        # Process each channel independently using groups
        vol = vol.reshape(B * C, 1, D, H, W)

        vol = F.conv3d(vol, self.kernel_d, padding=(pad, 0, 0))
        vol = F.conv3d(vol, self.kernel_h, padding=(0, pad, 0))
        vol = F.conv3d(vol, self.kernel_w, padding=(0, 0, pad))

        vol = vol.reshape(B, C, D, H, W)
        return vol

    def __repr__(self):
        return f"GaussianBlur3D(sigma={self.sigma}, kernel_size={self.kernel_size})"
