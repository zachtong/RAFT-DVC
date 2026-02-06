"""
Deformation field generators for synthetic DVC data.
Implements 4 types: Affine, B-spline, Localized, and Combined.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator


class DeformationGenerator:
    """Factory class for generating various deformation fields."""

    def __init__(self, volume_shape, config):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            config: Configuration dict from YAML
        """
        self.volume_shape = np.array(volume_shape)
        self.config = config

    def generate(self, deform_type=None):
        """
        Generate a random deformation field.

        Args:
            deform_type: 'affine', 'bspline', 'localized', or 'combined'
                        If None, randomly sample according to probabilities

        Returns:
            flow: numpy array of shape (3, D, H, W) - displacement field
            metadata: dict with generation parameters
        """
        if deform_type is None:
            # Sample type according to probabilities
            types = list(self.config['type_probabilities'].keys())
            probs = list(self.config['type_probabilities'].values())
            deform_type = np.random.choice(types, p=probs)

        if deform_type == 'affine':
            flow, metadata = self._generate_affine()
        elif deform_type == 'bspline':
            flow, metadata = self._generate_bspline()
        elif deform_type == 'localized':
            flow, metadata = self._generate_localized()
        elif deform_type == 'combined':
            flow, metadata = self._generate_combined()
        else:
            raise ValueError(f"Unknown deformation type: {deform_type}")

        metadata['type'] = deform_type
        return flow, metadata

    def _generate_affine(self):
        """Generate affine transformation (translation + rotation + scale + shear)."""
        cfg = self.config['affine']

        # Random parameters
        translation = np.random.uniform(-cfg['translation_max'],
                                       cfg['translation_max'], 3)
        rotation = np.random.uniform(-cfg['rotation_max'],
                                    cfg['rotation_max'], 3) * np.pi / 180
        scale = np.random.uniform(cfg['scale_min'], cfg['scale_max'], 3)
        shear = np.random.uniform(-cfg['shear_max'], cfg['shear_max'], 6)

        # Build affine matrix
        # Rotation matrices (around each axis)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(rotation[0]), -np.sin(rotation[0])],
                      [0, np.sin(rotation[0]), np.cos(rotation[0])]])

        Ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])],
                      [0, 1, 0],
                      [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])

        Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0],
                      [np.sin(rotation[2]), np.cos(rotation[2]), 0],
                      [0, 0, 1]])

        R = Rz @ Ry @ Rx

        # Scale matrix
        S = np.diag(scale)

        # Shear matrix
        H = np.array([[1, shear[0], shear[1]],
                     [shear[2], 1, shear[3]],
                     [shear[4], shear[5], 1]])

        # Combined linear transformation
        A = H @ S @ R

        # Create coordinate grids
        D, H, W = self.volume_shape
        z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W),
                              indexing='ij')

        # Center of volume
        center = self.volume_shape / 2

        # Compute displacement field
        # flow = A @ (coords - center) + center + translation - coords
        #      = (A - I) @ (coords - center) + translation

        coords = np.stack([z, y, x], axis=0)  # (3, D, H, W)
        coords_centered = coords - center[:, None, None, None]

        # Apply linear transformation
        flow = np.zeros_like(coords, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                flow[i] += (A[i, j] - (1 if i == j else 0)) * coords_centered[j]

        # Add translation
        flow += translation[:, None, None, None]

        metadata = {
            'translation': translation.tolist(),
            'rotation_deg': (rotation * 180 / np.pi).tolist(),
            'scale': scale.tolist(),
            'shear': shear.tolist()
        }

        return flow.astype(np.float32), metadata

    def _generate_bspline(self):
        """Generate smooth B-spline free-form deformation."""
        cfg = self.config['bspline']

        # Control grid size
        control_shape = np.array(cfg['control_grid_size'])

        # Random displacements on control points
        disp_magnitude = np.random.uniform(cfg['displacement_min'],
                                          cfg['displacement_max'])

        # Generate random displacements for each control point
        control_flow = np.random.randn(3, *control_shape) * disp_magnitude

        # Smooth the control point displacements
        sigma = cfg['smoothness'] * np.min(control_shape) / 4
        for i in range(3):
            control_flow[i] = gaussian_filter(control_flow[i], sigma=sigma,
                                             mode='nearest')

        # Interpolate to full resolution
        flow = self._interpolate_flow(control_flow, self.volume_shape)

        metadata = {
            'control_grid_size': control_shape.tolist(),
            'displacement_magnitude': float(disp_magnitude),
            'smoothness': cfg['smoothness']
        }

        return flow.astype(np.float32), metadata

    def _generate_localized(self):
        """Generate localized deformation (e.g., indentation, local compression)."""
        cfg = self.config['localized']

        # Number of deformation centers
        num_centers = np.random.randint(cfg['num_centers_min'],
                                       cfg['num_centers_max'] + 1)

        # Initialize flow field
        flow = np.zeros((3, *self.volume_shape), dtype=np.float32)

        centers = []
        for _ in range(num_centers):
            # Random center location
            center = np.random.rand(3) * self.volume_shape

            # Random displacement at center
            center_disp = np.random.randn(3) * cfg['center_displacement_max']

            # Radius of influence
            radius = np.random.uniform(cfg['radius_of_influence_min'],
                                      cfg['radius_of_influence_max'])

            # Create distance map from center
            D, H, W = self.volume_shape
            z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W),
                                  indexing='ij')

            dz = z - center[0]
            dy = y - center[1]
            dx = x - center[2]
            dist = np.sqrt(dz**2 + dy**2 + dx**2)

            # Radial decay function
            decay = np.exp(-(dist / radius) ** cfg['decay_exponent'])

            # Add localized displacement
            for i in range(3):
                flow[i] += center_disp[i] * decay

            centers.append({
                'location': center.tolist(),
                'displacement': center_disp.tolist(),
                'radius': float(radius)
            })

        metadata = {
            'num_centers': num_centers,
            'centers': centers
        }

        return flow, metadata

    def _generate_combined(self):
        """Generate combined deformation (affine + bspline)."""
        cfg = self.config['combined']

        # Generate both components
        flow_affine, meta_affine = self._generate_affine()
        flow_bspline, meta_bspline = self._generate_bspline()

        # Weighted combination
        flow = (cfg['affine_weight'] * flow_affine +
                cfg['bspline_weight'] * flow_bspline)

        metadata = {
            'affine_weight': cfg['affine_weight'],
            'bspline_weight': cfg['bspline_weight'],
            'affine_params': meta_affine,
            'bspline_params': meta_bspline
        }

        return flow.astype(np.float32), metadata

    def _interpolate_flow(self, control_flow, target_shape):
        """
        Interpolate flow field from control grid to target resolution.

        Args:
            control_flow: (3, Dc, Hc, Wc) control point displacements
            target_shape: (D, H, W) target volume shape

        Returns:
            flow: (3, D, H, W) interpolated flow field
        """
        control_shape = np.array(control_flow.shape[1:])

        # Control grid coordinates
        control_coords = [np.linspace(0, target_shape[i] - 1, control_shape[i])
                         for i in range(3)]

        # Target grid coordinates
        target_coords = [np.arange(target_shape[i]) for i in range(3)]

        # Interpolate each component
        flow = np.zeros((3, *target_shape), dtype=np.float32)
        for i in range(3):
            interpolator = RegularGridInterpolator(
                control_coords, control_flow[i],
                method='linear', bounds_error=False, fill_value=0
            )

            # Create target coordinate grid
            Z, Y, X = np.meshgrid(*target_coords, indexing='ij')
            points = np.stack([Z, Y, X], axis=-1)

            flow[i] = interpolator(points)

        return flow


def compute_flow_statistics(flow):
    """
    Compute statistics of a flow field.

    Args:
        flow: (3, D, H, W) numpy array

    Returns:
        stats: dict with magnitude mean/std/max, divergence, etc.
    """
    # Magnitude
    magnitude = np.sqrt(np.sum(flow**2, axis=0))

    # Divergence (approximate with finite differences)
    dz = np.gradient(flow[0], axis=0)
    dy = np.gradient(flow[1], axis=1)
    dx = np.gradient(flow[2], axis=2)
    divergence = dz + dy + dx

    stats = {
        'magnitude_mean': float(np.mean(magnitude)),
        'magnitude_std': float(np.std(magnitude)),
        'magnitude_max': float(np.max(magnitude)),
        'magnitude_min': float(np.min(magnitude)),
        'divergence_mean': float(np.mean(divergence)),
        'divergence_std': float(np.std(divergence))
    }

    return stats
