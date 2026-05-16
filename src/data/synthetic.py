"""
Synthetic Flow Generator for RAFT-DVC

Generates synthetic displacement fields for training data creation.
Based on VolRAFT flow generation with improvements for flexibility.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RegularGridInterpolator
import warnings


@dataclass
class FlowConfig:
    """Configuration for synthetic flow generation."""
    
    # Flow type weights
    translation_weight: float = 0.2
    rotation_weight: float = 0.2
    affine_weight: float = 0.2
    polynomial_weight: float = 0.2
    smooth_random_weight: float = 0.2
    
    # Flow magnitude limits (in voxels)
    max_translation: float = 10.0
    max_rotation_deg: float = 5.0
    max_scale: float = 0.05  # ±5% scaling
    max_shear: float = 0.02
    
    # Smooth random field parameters
    smooth_sigma_min: float = 10.0
    smooth_sigma_max: float = 30.0
    smooth_amplitude_min: float = 1.0
    smooth_amplitude_max: float = 5.0
    
    # Polynomial parameters
    poly_degree: int = 3
    poly_amplitude: float = 3.0
    
    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total = (self.translation_weight + self.rotation_weight + 
                 self.affine_weight + self.polynomial_weight + 
                 self.smooth_random_weight)
        if total > 0:
            self.translation_weight /= total
            self.rotation_weight /= total
            self.affine_weight /= total
            self.polynomial_weight /= total
            self.smooth_random_weight /= total


class SyntheticFlowGenerator:
    """
    Generate synthetic 3D displacement fields.
    
    Creates various types of displacement fields that simulate
    real-world deformations for DVC training:
    - Translation
    - Rotation
    - Affine transformations
    - Polynomial deformations
    - Smooth random fields
    
    Args:
        config: FlowConfig with generation parameters
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self, 
        config: Optional[FlowConfig] = None,
        seed: Optional[int] = None
    ):
        self.config = config or FlowConfig()
        self.config.normalize_weights()
        self.rng = np.random.RandomState(seed)
    
    def generate(
        self, 
        shape: Tuple[int, int, int],
        flow_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate a synthetic displacement field.
        
        Args:
            shape: Volume shape (H, W, D)
            flow_type: Type of flow to generate. If None, samples randomly 
                       based on config weights.
        
        Returns:
            Displacement field of shape (3, H, W, D) where channels are
            (dy, dx, dz) displacements in voxel units.
        """
        if flow_type is None:
            flow_type = self._sample_flow_type()
        
        if flow_type == 'translation':
            return self._generate_translation(shape)
        elif flow_type == 'rotation':
            return self._generate_rotation(shape)
        elif flow_type == 'affine':
            return self._generate_affine(shape)
        elif flow_type == 'polynomial':
            return self._generate_polynomial(shape)
        elif flow_type == 'smooth_random':
            return self._generate_smooth_random(shape)
        elif flow_type == 'combined':
            return self._generate_combined(shape)
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
    
    def _sample_flow_type(self) -> str:
        """Sample a flow type based on config weights."""
        types = ['translation', 'rotation', 'affine', 'polynomial', 'smooth_random']
        weights = [
            self.config.translation_weight,
            self.config.rotation_weight,
            self.config.affine_weight,
            self.config.polynomial_weight,
            self.config.smooth_random_weight
        ]
        return self.rng.choice(types, p=weights)
    
    def _generate_translation(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate uniform translation field."""
        H, W, D = shape
        
        # Random translation in each direction
        ty = self.rng.uniform(-1, 1) * self.config.max_translation
        tx = self.rng.uniform(-1, 1) * self.config.max_translation
        tz = self.rng.uniform(-1, 1) * self.config.max_translation
        
        flow = np.zeros((3, H, W, D), dtype=np.float32)
        flow[0] = ty  # y displacement
        flow[1] = tx  # x displacement
        flow[2] = tz  # z displacement
        
        return flow
    
    def _generate_rotation(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate rotation displacement field."""
        H, W, D = shape
        
        # Create coordinate grids centered at volume center
        cy, cx, cz = H / 2, W / 2, D / 2
        y, x, z = np.mgrid[0:H, 0:W, 0:D].astype(np.float32)
        y = y - cy
        x = x - cx
        z = z - cz
        
        # Random rotation angles
        max_rad = np.deg2rad(self.config.max_rotation_deg)
        angle_x = self.rng.uniform(-max_rad, max_rad)
        angle_y = self.rng.uniform(-max_rad, max_rad)
        angle_z = self.rng.uniform(-max_rad, max_rad)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        # Compute new positions
        coords = np.stack([y, x, z], axis=0).reshape(3, -1)
        new_coords = R @ coords
        
        # Displacement is new - old
        flow = (new_coords - coords).reshape(3, H, W, D).astype(np.float32)
        
        return flow
    
    def _generate_affine(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate affine transformation displacement field."""
        H, W, D = shape
        
        # Create coordinate grids centered at volume center
        cy, cx, cz = H / 2, W / 2, D / 2
        y, x, z = np.mgrid[0:H, 0:W, 0:D].astype(np.float32)
        y = y - cy
        x = x - cx
        z = z - cz
        
        # Random affine matrix (identity + small perturbation)
        A = np.eye(3, dtype=np.float32)
        
        # Random scaling
        scale = 1.0 + self.rng.uniform(-self.config.max_scale, self.config.max_scale, 3)
        A[0, 0] = scale[0]
        A[1, 1] = scale[1]
        A[2, 2] = scale[2]
        
        # Random shear
        shear = self.rng.uniform(-self.config.max_shear, self.config.max_shear, (3, 3))
        np.fill_diagonal(shear, 0)
        A = A + shear
        
        # Apply transformation
        coords = np.stack([y, x, z], axis=0).reshape(3, -1)
        new_coords = A @ coords
        
        # Add random translation
        t = self.rng.uniform(-1, 1, 3) * self.config.max_translation * 0.5
        new_coords = new_coords + t.reshape(3, 1)
        
        flow = (new_coords - coords).reshape(3, H, W, D).astype(np.float32)
        
        return flow
    
    def _generate_polynomial(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate polynomial deformation field."""
        H, W, D = shape
        
        # Normalized coordinates [-1, 1]
        y = np.linspace(-1, 1, H).astype(np.float32)
        x = np.linspace(-1, 1, W).astype(np.float32)
        z = np.linspace(-1, 1, D).astype(np.float32)
        yy, xx, zz = np.meshgrid(y, x, z, indexing='ij')
        
        flow = np.zeros((3, H, W, D), dtype=np.float32)
        
        # Generate random polynomial coefficients
        for dim in range(3):
            field = np.zeros((H, W, D), dtype=np.float32)
            for py in range(self.config.poly_degree + 1):
                for px in range(self.config.poly_degree + 1 - py):
                    for pz in range(self.config.poly_degree + 1 - py - px):
                        if py + px + pz > 0:  # Skip constant term
                            coef = self.rng.uniform(-1, 1) * self.config.poly_amplitude
                            coef /= (py + px + pz)  # Reduce higher-order term contribution
                            field += coef * (yy ** py) * (xx ** px) * (zz ** pz)
            flow[dim] = field
        
        return flow
    
    def _generate_smooth_random(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate smooth random displacement field."""
        H, W, D = shape
        
        # Random smoothing parameters
        sigma = self.rng.uniform(
            self.config.smooth_sigma_min, 
            self.config.smooth_sigma_max
        )
        amplitude = self.rng.uniform(
            self.config.smooth_amplitude_min,
            self.config.smooth_amplitude_max
        )
        
        flow = np.zeros((3, H, W, D), dtype=np.float32)
        
        for dim in range(3):
            # Generate random field
            field = self.rng.randn(H, W, D).astype(np.float32)
            # Smooth it
            field = gaussian_filter(field, sigma=sigma, mode='reflect')
            # Normalize and scale
            field = field / (field.std() + 1e-8) * amplitude
            flow[dim] = field
        
        return flow
    
    def _generate_combined(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Generate combined displacement field from multiple types."""
        flow = np.zeros((3, *shape), dtype=np.float32)
        
        # Randomly combine 2-3 types
        n_types = self.rng.randint(2, 4)
        types = ['rotation', 'affine', 'polynomial', 'smooth_random']
        selected = self.rng.choice(types, size=n_types, replace=False)
        
        for ft in selected:
            weight = self.rng.uniform(0.3, 0.7)
            flow += weight * self.generate(shape, flow_type=ft)
        
        return flow
    
    def warp_volume(
        self, 
        volume: np.ndarray, 
        flow: np.ndarray,
        order: int = 1
    ) -> np.ndarray:
        """
        Apply displacement field to warp a volume.
        
        Args:
            volume: Input volume of shape (H, W, D) or (C, H, W, D)
            flow: Displacement field of shape (3, H, W, D)
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        
        Returns:
            Warped volume with same shape as input
        """
        has_channels = volume.ndim == 4
        if has_channels:
            C = volume.shape[0]
            H, W, D = volume.shape[1:]
        else:
            H, W, D = volume.shape
        
        # Create sampling coordinates
        y, x, z = np.mgrid[0:H, 0:W, 0:D].astype(np.float32)
        
        # Add displacement to get source coordinates
        coords_y = y + flow[0]
        coords_x = x + flow[1]
        coords_z = z + flow[2]
        
        # Warp each channel
        if has_channels:
            warped = np.zeros_like(volume)
            for c in range(C):
                coords = np.array([coords_y, coords_x, coords_z])
                warped[c] = map_coordinates(
                    volume[c], coords, order=order, mode='constant', cval=0
                )
        else:
            coords = np.array([coords_y, coords_x, coords_z])
            warped = map_coordinates(
                volume, coords, order=order, mode='constant', cval=0
            )
        
        return warped.astype(np.float32)
    
    def generate_pair(
        self, 
        volume: np.ndarray,
        flow_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a volume pair with ground truth flow.
        
        Args:
            volume: Reference volume of shape (H, W, D) or (C, H, W, D)
            flow_type: Type of flow to generate
        
        Returns:
            Tuple of (vol0, vol1, flow) where vol1 = warp(vol0, flow)
        """
        if volume.ndim == 4:
            shape = volume.shape[1:]
        else:
            shape = volume.shape
        
        flow = self.generate(shape, flow_type=flow_type)
        vol1 = self.warp_volume(volume, flow)
        
        return volume, vol1, flow
