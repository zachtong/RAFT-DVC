"""
Update Module for RAFT-DVC

Implements the iterative update mechanism using ConvGRU for flow refinement.
Based on VolRAFT (CVPR 2024) with improvements for 3D volumetric data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FlowHead(nn.Module):
    """
    Flow prediction head.

    Predicts 3D displacement (dx, dy, dz) from hidden features.
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, kernel_size=3, padding=1)  # 3 for (dx, dy, dz)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state. Shape: (B, C, H, W, D)

        Returns:
            Delta flow. Shape: (B, 3, H, W, D)
        """
        return self.conv2(self.relu(self.conv1(x)))


class UncertaintyHead(nn.Module):
    """
    Uncertainty prediction head (parallel to FlowHead).

    Predicts log_b (log-scale parameter of Laplace distribution) per flow
    component from the same GRU hidden state.  Initial bias = 0 so that
    b = exp(0) = 1.0 at the start of training.
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state. Shape: (B, C, H, W, D)

        Returns:
            log_b. Shape: (B, 3, H, W, D)
        """
        return self.conv2(self.relu(self.conv1(x)))


class MoLUncertaintyHead(nn.Module):
    """
    Mixture-of-Laplace uncertainty head.

    Outputs 4 channels: 3 for log_b2 (learnable scale per flow component)
    and 1 for logit_alpha (mixing weight).

    Initialization:
        - log_b2 bias = 0  ->  b2 starts at exp(0) = 1.0
        - logit_alpha bias = 2.0  ->  alpha starts at sigmoid(2) ~ 0.88,
          favoring the fixed L1 component early in training.
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 4, kernel_size=3, padding=1)  # 3 log_b2 + 1 logit_alpha
        self.relu = nn.ReLU(inplace=True)
        # Initialize biases: log_b2 channels = 0, logit_alpha = 2.0
        with torch.no_grad():
            self.conv2.bias[:3].zero_()      # log_b2
            self.conv2.bias[3].fill_(2.0)    # logit_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden state. Shape: (B, C, H, W, D)

        Returns:
            (B, 4, H, W, D): [log_b2(3ch), logit_alpha(1ch)]
        """
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU3D(nn.Module):
    """
    3D Convolutional GRU for iterative refinement.
    
    Standard GRU cell with 3D convolutions instead of linear layers.
    """
    
    def __init__(self, hidden_dim: int = 96, input_dim: int = 192):
        super().__init__()
        
        combined_dim = hidden_dim + input_dim
        
        self.convz = nn.Conv3d(combined_dim, hidden_dim, kernel_size=3, padding=1)
        self.convr = nn.Conv3d(combined_dim, hidden_dim, kernel_size=3, padding=1)
        self.convq = nn.Conv3d(combined_dim, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        GRU forward pass.
        
        Args:
            h: Hidden state. Shape: (B, hidden_dim, H, W, D)
            x: Input features. Shape: (B, input_dim, H, W, D)
        
        Returns:
            Updated hidden state. Shape: (B, hidden_dim, H, W, D)
        """
        hx = torch.cat([h, x], dim=1)
        
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        
        h = (1 - z) * h + z * q
        return h


class SepConvGRU3D(nn.Module):
    """
    Separable 3D Convolutional GRU.
    
    Uses separable convolutions along each axis for efficiency.
    Processes height, width, and depth dimensions sequentially.
    """
    
    def __init__(self, hidden_dim: int = 96, input_dim: int = 192):
        super().__init__()
        
        combined_dim = hidden_dim + input_dim
        
        # Height-wise convolutions
        self.convz_h = nn.Conv3d(combined_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0))
        self.convr_h = nn.Conv3d(combined_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0))
        self.convq_h = nn.Conv3d(combined_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0))
        
        # Width-wise convolutions
        self.convz_w = nn.Conv3d(combined_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0))
        self.convr_w = nn.Conv3d(combined_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0))
        self.convq_w = nn.Conv3d(combined_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0))
        
        # Depth-wise convolutions
        self.convz_d = nn.Conv3d(combined_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2))
        self.convr_d = nn.Conv3d(combined_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2))
        self.convq_d = nn.Conv3d(combined_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2))
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Separable GRU forward pass.
        
        Args:
            h: Hidden state. Shape: (B, hidden_dim, H, W, D)
            x: Input features. Shape: (B, input_dim, H, W, D)
        
        Returns:
            Updated hidden state. Shape: (B, hidden_dim, H, W, D)
        """
        # Height pass
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz_h(hx))
        r = torch.sigmoid(self.convr_h(hx))
        q = torch.tanh(self.convq_h(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        
        # Width pass
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz_w(hx))
        r = torch.sigmoid(self.convr_w(hx))
        q = torch.tanh(self.convq_w(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        
        # Depth pass
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz_d(hx))
        r = torch.sigmoid(self.convr_d(hx))
        q = torch.tanh(self.convq_d(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        
        return h


class MotionEncoder(nn.Module):
    """
    Motion encoder for RAFT-DVC.
    
    Encodes current flow and correlation features into motion features
    for the GRU update.
    """
    
    def __init__(
        self, 
        corr_levels: int = 4, 
        corr_radius: int = 4,
        hidden_dim: int = 80
    ):
        super().__init__()
        
        # Correlation feature dimension
        corr_planes = corr_levels * (2 * corr_radius + 1) ** 3
        
        # Correlation encoder
        self.convc1 = nn.Conv3d(corr_planes, 96, kernel_size=1, padding=0)
        
        # Flow encoder
        self.convf1 = nn.Conv3d(3, 64, kernel_size=7, padding=3)
        self.convf2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        
        # Combined encoder
        self.conv = nn.Conv3d(128, hidden_dim, kernel_size=3, padding=1)
    
    def forward(
        self, 
        flow: torch.Tensor, 
        corr: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode motion features.
        
        Args:
            flow: Current flow estimate. Shape: (B, 3, H, W, D)
            corr: Correlation features. Shape: (B, corr_dim, H, W, D)
        
        Returns:
            Motion features. Shape: (B, hidden_dim + 3, H, W, D)
        """
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    """
    Basic update block for RAFT-DVC.

    Combines motion encoding, GRU update, and flow prediction
    for iterative flow refinement.

    Args:
        corr_levels: Number of correlation pyramid levels
        corr_radius: Correlation lookup radius
        hidden_dim: GRU hidden dimension
        context_dim: Context feature dimension
        use_sep_conv: Whether to use separable convolutions in GRU
        uncertainty_mode: ``"none"`` | ``"nll"`` | ``"mol"``
    """

    def __init__(
        self,
        corr_levels: int = 4,
        corr_radius: int = 4,
        hidden_dim: int = 96,
        context_dim: int = 64,
        use_sep_conv: bool = True,
        uncertainty_mode: str = "none"
    ):
        super().__init__()

        motion_dim = 80 + 3  # MotionEncoder output

        self.encoder = MotionEncoder(
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_dim=80
        )

        gru_input_dim = motion_dim + context_dim

        if use_sep_conv:
            self.gru = SepConvGRU3D(
                hidden_dim=hidden_dim,
                input_dim=gru_input_dim
            )
        else:
            self.gru = ConvGRU3D(
                hidden_dim=hidden_dim,
                input_dim=gru_input_dim
            )

        self.flow_head = FlowHead(
            input_dim=hidden_dim,
            hidden_dim=128
        )

        if uncertainty_mode == "nll":
            self.uncertainty_head = UncertaintyHead(input_dim=hidden_dim, hidden_dim=128)
        elif uncertainty_mode == "mol":
            self.uncertainty_head = MoLUncertaintyHead(input_dim=hidden_dim, hidden_dim=128)
        else:
            self.uncertainty_head = None

    def forward(
        self,
        net: torch.Tensor,
        context: torch.Tensor,
        corr: torch.Tensor,
        flow: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Update block forward pass.

        Args:
            net: Hidden state. Shape: (B, hidden_dim, H, W, D)
            context: Context features. Shape: (B, context_dim, H, W, D)
            corr: Correlation features. Shape: (B, corr_dim, H, W, D)
            flow: Current flow estimate. Shape: (B, 3, H, W, D)

        Returns:
            Tuple of:
            - Updated hidden state. Shape: (B, hidden_dim, H, W, D)
            - Delta flow. Shape: (B, 3, H, W, D)
            - log_b (or None). Shape: (B, 3, H, W, D)
        """
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([context, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        log_b = None
        if self.uncertainty_head is not None:
            log_b = self.uncertainty_head(net)

        return net, delta_flow, log_b
