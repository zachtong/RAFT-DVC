"""
3D Feature Extractor for RAFT-DVC

Extracts multi-scale features from 3D volume pairs for correlation computation.
Based on VolRAFT (CVPR 2024) with improvements for clarity and flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class ResidualBlock3D(nn.Module):
    """3D Residual block with configurable normalization."""
    
    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        norm_fn: str = 'instance', 
        stride: int = 1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Normalization layers
        self.norm1 = self._get_norm_layer(norm_fn, planes)
        self.norm2 = self._get_norm_layer(norm_fn, planes)
        
        # Downsample if stride != 1
        if stride == 1:
            self.downsample = None
        else:
            self.norm3 = self._get_norm_layer(norm_fn, planes)
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3
            )
    
    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer based on type."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=planes // 8, num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class BottleneckBlock3D(nn.Module):
    """3D Bottleneck block for deeper networks."""
    
    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        norm_fn: str = 'instance', 
        stride: int = 1
    ):
        super().__init__()
        
        mid_planes = planes // 4
        
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv3d(mid_planes, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
        # Normalization layers
        self.norm1 = self._get_norm_layer(norm_fn, mid_planes)
        self.norm2 = self._get_norm_layer(norm_fn, mid_planes)
        self.norm3 = self._get_norm_layer(norm_fn, planes)
        
        # Downsample if needed
        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.norm4 = self._get_norm_layer(norm_fn, planes)
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm4
            )
    
    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer based on type."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=max(1, planes // 8), num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class BasicEncoder(nn.Module):
    """
    Basic 3D encoder for feature extraction.
    
    Takes 3D volume input and produces multi-scale feature maps.
    Output is 1/8 resolution of input.
    
    Args:
        input_dim: Number of input channels (default: 1 for grayscale volume)
        output_dim: Number of output feature channels
        norm_fn: Normalization type ('batch', 'instance', 'group', 'none')
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        input_dim: int = 1,
        output_dim: int = 128, 
        norm_fn: str = 'instance', 
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.output_dim = output_dim
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_dim, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = self._get_norm_layer(norm_fn, 32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Residual layers (output stride = 8)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)   # /2 (from conv1)
        self.layer2 = self._make_layer(64, stride=2)   # /4
        self.layer3 = self._make_layer(96, stride=2)   # /8
        
        # Output projection
        self.conv2 = nn.Conv3d(96, output_dim, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer based on type."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=max(1, planes // 8), num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")
    
    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer with two blocks."""
        layer1 = BottleneckBlock3D(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock3D(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input volume(s). Can be single tensor or list/tuple of two tensors.
               Shape: (B, C, H, W, D)
        
        Returns:
            Feature map(s). Same structure as input (single or tuple).
        """
        # Handle list/tuple input (for processing image pairs together)
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        # Forward through network
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.conv2(x)
        
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        
        # Split back if input was list/tuple
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        
        return x


class ContextEncoder(nn.Module):
    """
    Context encoder for RAFT-DVC.
    
    Similar to BasicEncoder but produces both hidden state and context features.
    Used to initialize the GRU hidden state and provide context for updates.
    """
    
    def __init__(
        self, 
        input_dim: int = 1,
        hidden_dim: int = 96,
        context_dim: int = 64,
        norm_fn: str = 'none', 
        dropout: float = 0.0
    ):
        super().__init__()
        
        output_dim = hidden_dim + context_dim
        self.encoder = BasicEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            norm_fn=norm_fn,
            dropout=dropout
        )
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input volume. Shape: (B, C, H, W, D)
        
        Returns:
            Tuple of (hidden_state, context) tensors
        """
        features = self.encoder(x)
        net, context = torch.split(features, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        context = torch.relu(context)
        return net, context


class MediumEncoder(nn.Module):
    """
    Medium 3D encoder with 1/4 resolution output.

    Balanced between computational efficiency and spatial resolution.
    Suitable for moderate displacement fields.

    Architecture:
        - Initial conv: stride=2 → 1/2 resolution
        - Layer1: stride=1 → 1/2 resolution
        - Layer2: stride=2 → 1/4 resolution
        - Layer3: stride=1 → 1/4 resolution (no further downsampling)

    Args:
        input_dim: Number of input channels (default: 1)
        output_dim: Number of output feature channels
        norm_fn: Normalization type ('batch', 'instance', 'group', 'none')
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 128,
        norm_fn: str = 'instance',
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.output_dim = output_dim

        # Initial convolution
        self.conv1 = nn.Conv3d(input_dim, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = self._get_norm_layer(norm_fn, 32)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual layers (output stride = 4)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)   # /2
        self.layer2 = self._make_layer(64, stride=2)   # /4
        self.layer3 = self._make_layer(96, stride=1)   # /4 (no change)

        # Output projection
        self.conv2 = nn.Conv3d(96, output_dim, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=max(1, planes // 8), num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer."""
        layer1 = BottleneckBlock3D(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock3D(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class ShallowEncoder(nn.Module):
    """
    Shallow 3D encoder with 1/2 resolution output.

    Higher spatial resolution for fine displacement estimation.
    Recommended for sub-voxel precision DVC measurements.

    Architecture:
        - Initial conv: stride=2 → 1/2 resolution
        - Layer1: stride=1 → 1/2 resolution
        - Layer2: stride=1 → 1/2 resolution (no downsampling)
        - Layer3: stride=1 → 1/2 resolution (no downsampling)

    Args:
        input_dim: Number of input channels (default: 1)
        output_dim: Number of output feature channels
        norm_fn: Normalization type ('batch', 'instance', 'group', 'none')
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 128,
        norm_fn: str = 'instance',
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.output_dim = output_dim

        # Initial convolution - only 1x downsampling
        self.conv1 = nn.Conv3d(input_dim, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = self._get_norm_layer(norm_fn, 32)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual layers (output stride = 2, all stride=1)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)   # /2
        self.layer2 = self._make_layer(64, stride=1)   # /2 (no change)
        self.layer3 = self._make_layer(96, stride=1)   # /2 (no change)

        # Output projection
        self.conv2 = nn.Conv3d(96, output_dim, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=max(1, planes // 8), num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer."""
        layer1 = BottleneckBlock3D(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock3D(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class FullResEncoder(nn.Module):
    """
    Full resolution 3D encoder with 1/1 resolution output (no downsampling).

    Maximum spatial resolution for ultra-fine displacement estimation.
    WARNING: Very high computational and memory cost!

    Architecture:
        - Initial conv: stride=1 → Full resolution
        - Layer1: stride=1 → Full resolution
        - Layer2: stride=1 → Full resolution
        - Layer3: stride=1 → Full resolution

    Args:
        input_dim: Number of input channels (default: 1)
        output_dim: Number of output feature channels
        norm_fn: Normalization type ('batch', 'instance', 'group', 'none')
        dropout: Dropout probability

    Note:
        For 128^3 volumes:
        - Feature map size: 128^3 = 2,097,152 voxels
        - Memory: ~20-30GB for batch_size=1
        - Training time: ~10x slower than BasicEncoder
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 128,
        norm_fn: str = 'instance',
        dropout: float = 0.0
    ):
        super().__init__()
        self.norm_fn = norm_fn
        self.output_dim = output_dim

        # Initial convolution - NO downsampling
        self.conv1 = nn.Conv3d(input_dim, 32, kernel_size=7, stride=1, padding=3)
        self.norm1 = self._get_norm_layer(norm_fn, 32)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual layers (all stride=1, full resolution maintained)
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)   # /1
        self.layer2 = self._make_layer(64, stride=1)   # /1
        self.layer3 = self._make_layer(96, stride=1)   # /1

        # Output projection
        self.conv2 = nn.Conv3d(96, output_dim, kernel_size=1)

        # Dropout
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _get_norm_layer(self, norm_fn: str, planes: int) -> nn.Module:
        """Create normalization layer."""
        if norm_fn == 'group':
            return nn.GroupNorm(num_groups=max(1, planes // 8), num_channels=planes)
        elif norm_fn == 'batch':
            return nn.BatchNorm3d(planes)
        elif norm_fn == 'instance':
            return nn.InstanceNorm3d(planes)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown norm_fn: {norm_fn}")

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        """Create a residual layer."""
        layer1 = BottleneckBlock3D(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock3D(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
