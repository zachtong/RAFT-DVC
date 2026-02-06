"""
Volume warping functions (forward warp with trilinear splatting).
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class ForwardWarper:
    """Forward warp beads using trilinear splatting."""

    def __init__(self, volume_shape):
        """
        Args:
            volume_shape: Tuple (D, H, W)
        """
        self.volume_shape = np.array(volume_shape)

    def warp_beads(self, beads, flow):
        """
        Warp beads using forward warping with trilinear splatting.

        Args:
            beads: dict with 'positions', 'radii', 'intensities'
            flow: (3, D, H, W) displacement field

        Returns:
            warped_beads: dict with warped bead positions
            num_visible: number of beads still in FOV after warping
        """
        positions = beads['positions']  # (N, 3)
        radii = beads['radii']
        intensities = beads['intensities']

        # Sample flow at bead positions
        displacements = self._sample_flow_at_positions(flow, positions)

        # Compute new positions
        new_positions = positions + displacements

        # Check which beads are still visible (within volume bounds)
        margin = 5  # voxels from boundary
        in_bounds = np.all(
            (new_positions >= margin) &
            (new_positions < self.volume_shape - margin),
            axis=1
        )

        # Keep only visible beads
        warped_beads = {
            'positions': new_positions[in_bounds],
            'radii': radii[in_bounds],
            'intensities': intensities[in_bounds],
            'num_beads': np.sum(in_bounds),
            'original_num_beads': len(positions)
        }

        return warped_beads, np.sum(in_bounds)

    def warp_volume(self, volume, beads_original, beads_warped):
        """
        Alternative: directly warp the volume using bead correspondences.
        This is more efficient than re-rendering.

        Args:
            volume: (D, H, W) original volume
            beads_original: original bead dict
            beads_warped: warped bead dict

        Returns:
            warped_volume: (D, H, W) warped volume
        """
        # For now, we'll use the simpler approach: re-render warped beads
        # This ensures consistent rendering with trilinear splatting
        from .beads import BeadRenderer

        renderer = BeadRenderer(self.volume_shape)
        warped_volume = renderer.render(beads_warped)

        return warped_volume

    def warp_volume_dense(self, volume, flow):
        """
        Warp entire volume using forward warping with trilinear splatting.
        This is more general but slower than bead-based warping.

        Args:
            volume: (D, H, W) source volume
            flow: (3, D, H, W) displacement field

        Returns:
            warped_volume: (D, H, W) warped volume
        """
        D, H, W = self.volume_shape
        warped = np.zeros((D, H, W), dtype=np.float32)
        weight = np.zeros((D, H, W), dtype=np.float32)

        # For each source voxel with non-zero intensity
        nz_indices = np.nonzero(volume > 0.01)  # Threshold to skip background

        for idx in range(len(nz_indices[0])):
            z, y, x = nz_indices[0][idx], nz_indices[1][idx], nz_indices[2][idx]
            intensity = volume[z, y, x]

            # Get displacement
            displacement = flow[:, z, y, x]
            target_pos = np.array([z, y, x]) + displacement

            # Trilinear splatting
            self._trilinear_splat(warped, weight, target_pos, intensity)

        # Normalize by accumulated weights
        mask = weight > 0
        warped[mask] /= weight[mask]

        return warped

    def _sample_flow_at_positions(self, flow, positions):
        """
        Sample flow field at arbitrary positions using trilinear interpolation.

        Args:
            flow: (3, D, H, W) flow field
            positions: (N, 3) array of positions

        Returns:
            displacements: (N, 3) array of sampled displacements
        """
        from scipy.ndimage import map_coordinates

        N = len(positions)
        displacements = np.zeros((N, 3), dtype=np.float32)

        # For each component
        for i in range(3):
            # map_coordinates expects (N, ndim) format where each row is (z, y, x)
            coords = positions.T  # (3, N)
            displacements[:, i] = map_coordinates(
                flow[i], coords, order=1, mode='nearest'
            )

        return displacements

    def _trilinear_splat(self, volume, weight, position, intensity):
        """
        Splat intensity to 8 neighboring voxels using trilinear interpolation.

        Args:
            volume: (D, H, W) target volume (modified in-place)
            weight: (D, H, W) weight accumulator (modified in-place)
            position: (3,) float array - target position
            intensity: float - intensity to splat
        """
        # Floor and ceil positions
        z0, y0, x0 = np.floor(position).astype(int)
        z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1

        # Check bounds
        if (z0 < 0 or z1 >= self.volume_shape[0] or
            y0 < 0 or y1 >= self.volume_shape[1] or
            x0 < 0 or x1 >= self.volume_shape[2]):
            return  # Out of bounds

        # Fractional parts
        dz = position[0] - z0
        dy = position[1] - y0
        dx = position[2] - x0

        # 8 trilinear weights
        w000 = (1 - dz) * (1 - dy) * (1 - dx)
        w001 = (1 - dz) * (1 - dy) * dx
        w010 = (1 - dz) * dy * (1 - dx)
        w011 = (1 - dz) * dy * dx
        w100 = dz * (1 - dy) * (1 - dx)
        w101 = dz * (1 - dy) * dx
        w110 = dz * dy * (1 - dx)
        w111 = dz * dy * dx

        # Accumulate to 8 neighbors
        volume[z0, y0, x0] += w000 * intensity
        volume[z0, y0, x1] += w001 * intensity
        volume[z0, y1, x0] += w010 * intensity
        volume[z0, y1, x1] += w011 * intensity
        volume[z1, y0, x0] += w100 * intensity
        volume[z1, y0, x1] += w101 * intensity
        volume[z1, y1, x0] += w110 * intensity
        volume[z1, y1, x1] += w111 * intensity

        # Accumulate weights
        weight[z0, y0, x0] += w000
        weight[z0, y0, x1] += w001
        weight[z0, y1, x0] += w010
        weight[z0, y1, x1] += w011
        weight[z1, y0, x0] += w100
        weight[z1, y0, x1] += w101
        weight[z1, y1, x0] += w110
        weight[z1, y1, x1] += w111


def compute_warp_statistics(beads_original, beads_warped):
    """
    Compute statistics about the warping process.

    Args:
        beads_original: original bead dict
        beads_warped: warped bead dict

    Returns:
        stats: dict with warping statistics
    """
    original_count = beads_original['num_beads']
    warped_count = beads_warped['num_beads']

    visible_ratio = warped_count / original_count if original_count > 0 else 0

    # Compute displacement statistics for visible beads
    if warped_count > 0:
        # Match original and warped positions
        # (Simplified: assume beads_warped corresponds to first N beads)
        orig_pos = beads_original['positions'][:warped_count]
        warp_pos = beads_warped['positions']

        displacements = np.linalg.norm(warp_pos - orig_pos, axis=1)

        disp_mean = float(np.mean(displacements))
        disp_std = float(np.std(displacements))
        disp_max = float(np.max(displacements))
    else:
        disp_mean = disp_std = disp_max = 0

    stats = {
        'original_count': int(original_count),
        'warped_count': int(warped_count),
        'visible_ratio': float(visible_ratio),
        'displacement_mean': disp_mean,
        'displacement_std': disp_std,
        'displacement_max': disp_max
    }

    return stats
