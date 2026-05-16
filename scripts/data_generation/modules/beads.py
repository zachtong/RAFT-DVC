"""
Bead/particle generation and rendering for synthetic confocal microscopy.
"""

import numpy as np

# Note: ``scipy.ndimage.gaussian_filter`` and ``scipy.spatial.cKDTree`` are
# imported lazily inside the rendering helpers so that environments missing
# scipy can still parse this module (the JIT site emits the real error).


class BeadGenerator:
    """Generate random fluorescent beads/particles."""

    def __init__(self, volume_shape, config):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            config: Configuration dict for particles
        """
        self.volume_shape = np.array(volume_shape)
        self.config = config

    def generate_beads(self):
        """
        Generate random beads with positions, radii, and intensities.

        Returns:
            beads: dict with keys 'positions', 'radii', 'intensities'
        """
        # Sample number of beads
        if self.config['count_stratified']:
            # Stratified sampling for uniform density distribution
            num_beads = self._stratified_count_sampling()
        else:
            num_beads = np.random.randint(self.config['count_min'],
                                         self.config['count_max'] + 1)

        # Generate bead positions (with minimum distance constraint)
        positions = self._generate_positions(num_beads)

        # Sample radii
        radii = self._sample_radii(num_beads)

        # Sample intensities
        intensities = self._sample_intensities(num_beads)

        beads = {
            'positions': positions,  # (N, 3) array
            'radii': radii,          # (N,) array
            'intensities': intensities,  # (N,) array
            'num_beads': num_beads
        }

        return beads

    def _stratified_count_sampling(self):
        """Sample bead count with stratification to ensure density diversity."""
        # Divide range into 4 tiers
        count_min = self.config['count_min']
        count_max = self.config['count_max']
        tier_size = (count_max - count_min) / 4

        # Randomly choose a tier
        tier = np.random.randint(0, 4)

        # Sample within that tier
        tier_min = int(count_min + tier * tier_size)
        tier_max = int(count_min + (tier + 1) * tier_size)

        return np.random.randint(tier_min, tier_max + 1)

    def _generate_positions(self, num_beads):
        """
        Generate bead positions with minimum distance constraint.

        Args:
            num_beads: Number of beads to generate

        Returns:
            positions: (N, 3) array of bead centers
        """
        # Start with random positions
        positions = []

        # Safety margin from boundaries (to avoid edge effects)
        # Can be configured in config, defaults to 5 voxels for backward compatibility
        margin = self.config.get('margin_voxels', 5)

        # Maximum attempts to place a bead
        max_attempts = num_beads * 20

        attempts = 0
        while len(positions) < num_beads and attempts < max_attempts:
            # Random position
            pos = np.random.rand(3) * (self.volume_shape - 2 * margin) + margin

            # Check minimum distance to existing beads
            if len(positions) == 0:
                positions.append(pos)
            else:
                positions_array = np.array(positions)
                distances = np.linalg.norm(positions_array - pos, axis=1)

                # Minimum distance is a factor of typical radius
                min_dist = self.config['min_distance_factor'] * \
                          (self.config['radius_min'] + self.config['radius_max']) / 2

                if np.min(distances) > min_dist:
                    positions.append(pos)

            attempts += 1

        if len(positions) < num_beads:
            print(f"Warning: Only placed {len(positions)}/{num_beads} beads "
                  f"(min distance constraint)")

        return np.array(positions)

    def _sample_radii(self, num_beads):
        """Sample bead radii."""
        if self.config['radius_distribution'] == 'lognormal':
            mean = self.config['radius_lognormal_mean']
            std = self.config['radius_lognormal_std']
            radii = np.random.lognormal(mean, std, num_beads)

            # Clip to valid range
            radii = np.clip(radii, self.config['radius_min'],
                           self.config['radius_max'])
        else:  # uniform
            radii = np.random.uniform(self.config['radius_min'],
                                     self.config['radius_max'],
                                     num_beads)

        return radii

    def _sample_intensities(self, num_beads):
        """Sample bead intensities."""
        intensities = np.random.normal(self.config['intensity_mean'],
                                      self.config['intensity_std'],
                                      num_beads)

        # Clip to valid range
        intensities = np.clip(intensities,
                             self.config['intensity_min'],
                             self.config['intensity_max'])

        return intensities


class BeadRenderer:
    """Render beads as 3D Gaussians into a volume."""

    def __init__(self, volume_shape, config=None):
        """
        Args:
            volume_shape: Tuple (D, H, W)
            config: Optional configuration dict for rendering parameters
        """
        self.volume_shape = volume_shape

        # Get Gaussian sharpness parameter (default 2.0 for backward compatibility)
        if config is not None:
            self.gaussian_sharpness = config.get('gaussian_sharpness', 2.0)
        else:
            self.gaussian_sharpness = 2.0

    def render(self, beads):
        """
        Render beads into a volume.

        Args:
            beads: dict with 'positions', 'radii', 'intensities'

        Returns:
            volume: (D, H, W) numpy array
        """
        volume = np.zeros(self.volume_shape, dtype=np.float32)

        positions = beads['positions']
        radii = beads['radii']
        intensities = beads['intensities']

        for i in range(len(positions)):
            self._render_single_bead(
                volume,
                positions[i],
                radii[i],
                intensities[i]
            )

        return volume

    def _render_single_bead(self, volume, position, radius, intensity):
        """
        Render a single bead as a 3D Gaussian.

        Args:
            volume: (D, H, W) array to render into (modified in-place)
            position: (3,) array - bead center
            radius: float - bead radius (sigma of Gaussian)
            intensity: float - peak intensity
        """
        # Determine bounding box (3 sigma)
        bbox_radius = int(np.ceil(3 * radius))

        # Integer bounding box
        z0 = max(0, int(position[0]) - bbox_radius)
        z1 = min(self.volume_shape[0], int(position[0]) + bbox_radius + 1)
        y0 = max(0, int(position[1]) - bbox_radius)
        y1 = min(self.volume_shape[1], int(position[1]) + bbox_radius + 1)
        x0 = max(0, int(position[2]) - bbox_radius)
        x1 = min(self.volume_shape[2], int(position[2]) + bbox_radius + 1)

        # Skip if bead is outside volume
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return

        # Create local coordinate grids
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing='ij'
        )

        # Distance from bead center
        dz = z_grid - position[0]
        dy = y_grid - position[1]
        dx = x_grid - position[2]
        dist_sq = dz**2 + dy**2 + dx**2

        # 3D Gaussian with configurable sharpness
        gaussian = intensity * np.exp(-dist_sq / (self.gaussian_sharpness * radius**2))

        # Add to volume (accumulate if overlap)
        volume[z0:z1, y0:y1, x0:x1] += gaussian


class HardSphereBeadRenderer:
    """Render beads as hard spheres with anti-aliased edges, then apply PSF blur.

    Physical model (matches fluorescent bead imaging more closely than the pure
    Gaussian renderer):

      1. Each bead is a solid sphere of uniform intensity inside its radius.
         A narrow linear taper at the sphere surface provides sub-voxel
         anti-aliasing so small-radius beads don't look like cubes.
      2. Bead overlaps are resolved by ``np.maximum`` (saturation) rather than
         ``+=``. This models the fact that stacking two fluorophores on the
         same voxel does not produce unbounded brightness.
      3. After all beads are drawn, a single Gaussian filter is applied to the
         whole volume to simulate the imaging PSF.

    Compared to :class:`BeadRenderer`, this decouples the *bead radius* from
    the *PSF width* and eliminates the r**3 fill-ratio blow-up that makes the
    pure-Gaussian renderer produce saturated volumes at large radii.

    Config keys (all optional):
        psf_sigma: Gaussian PSF sigma in voxels (default 0.8).
        edge_softness: half-width of the anti-aliasing taper in voxels
            (default 0.5). Set to 0 for a strict hard sphere.
    """

    def __init__(self, volume_shape, config=None):
        self.volume_shape = tuple(volume_shape)
        cfg = config or {}
        self.psf_sigma = float(cfg.get('psf_sigma', 0.8))
        self.edge_softness = float(cfg.get('edge_softness', 0.5))

    def render(self, beads):
        """Render all beads, then apply a single global PSF blur."""
        volume = np.zeros(self.volume_shape, dtype=np.float32)

        positions = beads['positions']
        radii = beads['radii']
        intensities = beads['intensities']

        for i in range(len(positions)):
            self._render_single_bead(
                volume,
                positions[i],
                float(radii[i]),
                float(intensities[i]),
            )

        if self.psf_sigma > 0:
            from scipy.ndimage import gaussian_filter
            volume = gaussian_filter(volume, sigma=self.psf_sigma)

        return volume

    def _render_single_bead(self, volume, position, radius, intensity):
        taper = self.edge_softness
        bbox_r = int(np.ceil(radius + taper)) + 1

        z0 = max(0, int(position[0]) - bbox_r)
        z1 = min(self.volume_shape[0], int(position[0]) + bbox_r + 1)
        y0 = max(0, int(position[1]) - bbox_r)
        y1 = min(self.volume_shape[1], int(position[1]) + bbox_r + 1)
        x0 = max(0, int(position[2]) - bbox_r)
        x1 = min(self.volume_shape[2], int(position[2]) + bbox_r + 1)

        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            return

        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(z0, z1),
            np.arange(y0, y1),
            np.arange(x0, x1),
            indexing='ij',
        )
        dz = z_grid - position[0]
        dy = y_grid - position[1]
        dx = x_grid - position[2]
        dist = np.sqrt(dz * dz + dy * dy + dx * dx)

        if taper > 0:
            # Linear taper: alpha = 1 at dist = r - taper, 0 at dist = r + taper.
            alpha = np.clip((radius + taper - dist) / (2.0 * taper), 0.0, 1.0)
        else:
            alpha = (dist <= radius).astype(np.float32)

        bead_value = (alpha * intensity).astype(np.float32)

        # Saturation composite: max, not sum.
        region = volume[z0:z1, y0:y1, x0:x1]
        np.maximum(region, bead_value, out=region)


def compute_bead_statistics(beads):
    """
    Compute statistics of generated beads.

    Args:
        beads: dict with bead properties

    Returns:
        stats: dict with statistics
    """
    positions = beads['positions']
    radii = beads['radii']
    intensities = beads['intensities']

    # Volume fraction (approximate)
    total_volume = np.prod([128, 128, 128])  # Assume 128^3
    bead_volume = np.sum((4/3) * np.pi * radii**3)
    volume_fraction = bead_volume / total_volume

    # Spatial distribution (check for clustering)
    if len(positions) > 1:
        # Average nearest neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        distances, _ = tree.query(positions, k=2)  # k=2 to exclude self
        nn_dist = np.mean(distances[:, 1])
    else:
        nn_dist = 0

    stats = {
        'num_beads': len(positions),
        'radius_mean': float(np.mean(radii)),
        'radius_std': float(np.std(radii)),
        'intensity_mean': float(np.mean(intensities)),
        'intensity_std': float(np.std(intensities)),
        'volume_fraction': float(volume_fraction),
        'nearest_neighbor_dist': float(nn_dist)
    }

    return stats
