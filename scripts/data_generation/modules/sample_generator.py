"""
Phase-1 sample generator: builds (I1, I2, flow) triples with full pipeline.

Per sample pipeline (see memory/dataset_construction.md):
  1. Place beads in an *expanded* volume (RSA with min-distance constraint).
     --- Fresh bead realization per sample (positions + radii + intensities).
  2. Sample a deformation field with max|u|_inf == fm_target * downsample_factor.
  3. Forward-warp bead centers: x_exp' = x_exp + u(x_crop),
     where x_crop = x_exp - buffer_total (cropped-frame coords).
  4. Render I1 at x_exp, I2 at x_exp' on the expanded volume.
  5. Center-crop both to the input size.
  6. Apply INDEPENDENT Poisson + Gaussian noise to I1 and I2.
  7. Return (I1, I2, flow, metadata).

Design parameters are fixed by the finalized memory files:
- synthetic_data_params.md (radius, density, PSF, edge softness, min-dist factor)
- displacement_warp_design.md (fm bounds, DIC shape functions, warp method)
- noise_model_design.md (gain=500, read_sigma=0.01, independent realization)
"""

from __future__ import annotations

import numpy as np

from .beads import HardSphereBeadRenderer
from .deformation_v2 import DeformationSampler


# -----------------------------------------------------------------------------
# Bead placement (grid-based RSA) --- logic lifted from preview_parameter_grid.py
# -----------------------------------------------------------------------------

def place_beads_grid(
    volume_size,
    target_count,
    radius,
    min_distance_factor,
    rng,
    max_attempts_per_bead=20,
    consecutive_failure_limit=400,
):
    """Grid-based rejection sampling (Poisson-disk-like) for bead centers.

    Returns
    -------
    positions : (n_placed, 3) float32
    info : dict with keys requested, placed, early_stopped.
    """
    if target_count <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            {"requested": 0, "placed": 0, "early_stopped": False},
        )

    if min_distance_factor <= 0:
        positions = rng.uniform(
            0, volume_size, size=(target_count, 3)
        ).astype(np.float32)
        return positions, {
            "requested": target_count,
            "placed": target_count,
            "early_stopped": False,
        }

    min_dist = float(min_distance_factor) * float(radius)
    min_dist_sq = min_dist * min_dist
    cell_size = min_dist

    grid = {}
    placed = np.empty((target_count, 3), dtype=np.float32)
    n_placed = 0
    attempts = 0
    consecutive_failures = 0
    max_total_attempts = max_attempts_per_bead * target_count
    early_stopped = False

    while n_placed < target_count and attempts < max_total_attempts:
        pos = rng.uniform(0, volume_size, size=3).astype(np.float32)
        ci = int(pos[0] / cell_size)
        cj = int(pos[1] / cell_size)
        ck = int(pos[2] / cell_size)

        too_close = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    key = (ci + dz, cj + dy, ck + dx)
                    neighbors = grid.get(key)
                    if neighbors is None:
                        continue
                    for existing in neighbors:
                        d0 = existing[0] - pos[0]
                        d1 = existing[1] - pos[1]
                        d2 = existing[2] - pos[2]
                        if d0 * d0 + d1 * d1 + d2 * d2 < min_dist_sq:
                            too_close = True
                            break
                    if too_close:
                        break
                if too_close:
                    break
            if too_close:
                break

        if not too_close:
            placed[n_placed] = pos
            grid.setdefault((ci, cj, ck), []).append(pos)
            n_placed += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= consecutive_failure_limit:
                early_stopped = True
                break

        attempts += 1

    positions = placed[:n_placed].copy()
    return positions, {
        "requested": target_count,
        "placed": n_placed,
        "early_stopped": early_stopped,
    }


def sample_bead_attributes(n, radius_center, rng):
    """Sample per-bead radii ~ U(r - 0.2, r + 0.2) and intensity ~ clipped N(1, 0.1)."""
    if n == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    radius_min = max(0.5, radius_center - 0.2)
    radius_max = radius_center + 0.2
    radii = rng.uniform(radius_min, radius_max, size=n).astype(np.float32)
    intensities = np.clip(
        rng.normal(1.0, 0.1, size=n), 0.5, 1.5
    ).astype(np.float32)
    return radii, intensities


# -----------------------------------------------------------------------------
# Noise model (Poisson shot + Gaussian read) --- per memory/noise_model_design.md
# -----------------------------------------------------------------------------

def apply_noise(volume, gain, read_sigma, rng):
    """Apply Poisson (shot) + Gaussian (read) noise.

    Args:
        volume: (D, H, W) float32.
        gain: expected photons per unit intensity.
        read_sigma: Gaussian read noise stdev in normalized units.
        rng: numpy.random.Generator.
    """
    lam = np.clip(volume, 0.0, None) * gain
    shot = rng.poisson(lam).astype(np.float32) / gain
    read = rng.normal(0.0, read_sigma, size=volume.shape).astype(np.float32)
    noisy = shot + read
    np.clip(noisy, 0.0, None, out=noisy)
    return noisy


# -----------------------------------------------------------------------------
# Main sample generator
# -----------------------------------------------------------------------------

class Phase1SampleGenerator:
    """Generate one (I1, I2, flow) sample following finalized Phase-1 design.

    Two displacement-range modes (mutually exclusive):

    1) FM-space mode (default, backward-compatible).  Specify ``fm_min``,
       ``fm_max``, ``feature_map_size``.  ``fm_target ~ U(fm_min, fm_max)``
       is drawn in feature-map voxels; input-space max disp is then
       ``fm_target * downsample_factor`` where
       ``downsample_factor = size / feature_map_size``.  This gives a
       constant FM-scale difficulty regardless of input resolution.

    2) Input-space mode (paper-1 architecture-ablation experiments).
       Specify ``input_disp_min`` and ``input_disp_max``.
       ``input_max_disp ~ U(input_disp_min, input_disp_max)`` is drawn
       directly in input voxels; FM-disp varies inversely with
       downsample factor.  Use this when comparing encoders at a fixed
       input observation -- all encoders see the same physical
       deformation regardless of architecture.
    """

    def __init__(
        self,
        size,
        radius,
        density_per_1000,
        fm_min=0.3,
        fm_max=3.0,
        feature_map_size=32,
        psf_sigma=0.8,
        edge_softness=0.5,
        min_distance_factor=1.3,
        noise_gain=500.0,
        noise_read_sigma=0.01,
        type_probs=(0.10, 0.45, 0.45),
        input_disp_min=None,
        input_disp_max=None,
    ):
        self.size = int(size)
        self.radius = int(radius)
        self.density = float(density_per_1000)
        self.feature_map_size = int(feature_map_size)
        self.downsample_factor = self.size / self.feature_map_size
        self.psf_sigma = float(psf_sigma)
        self.edge_softness = float(edge_softness)
        self.min_distance_factor = float(min_distance_factor)
        self.noise_gain = float(noise_gain)
        self.noise_read_sigma = float(noise_read_sigma)
        self.type_probs = tuple(type_probs)

        # Mode selection: input-disp explicit overrides fm-disp.
        if input_disp_min is not None or input_disp_max is not None:
            if input_disp_min is None or input_disp_max is None:
                raise ValueError(
                    "input_disp_min and input_disp_max must both be set."
                )
            self.disp_mode = "input"
            self.input_disp_min = float(input_disp_min)
            self.input_disp_max = float(input_disp_max)
            # Derived fm range (used only for metadata / buffer):
            self.fm_min = self.input_disp_min / self.downsample_factor
            self.fm_max = self.input_disp_max / self.downsample_factor
        else:
            self.disp_mode = "fm"
            self.fm_min = float(fm_min)
            self.fm_max = float(fm_max)
            self.input_disp_min = self.fm_min * self.downsample_factor
            self.input_disp_max = self.fm_max * self.downsample_factor

        # Rendering buffer: covers bead radius + AA taper + 3*PSF sigma.
        self.buffer_render = int(
            np.ceil(self.radius + self.edge_softness + 3.0 * self.psf_sigma)
        ) + 1

        # Displacement buffer: enough for the largest possible input-space
        # disp, regardless of which mode produced it.
        self.max_disp_buffer = int(np.ceil(self.input_disp_max)) + 1

        self.buffer_total = self.buffer_render + self.max_disp_buffer
        self.expanded_size = self.size + 2 * self.buffer_total

        # Pre-instantiate renderer (shape-bound) and deformation sampler.
        self._renderer = HardSphereBeadRenderer(
            (self.expanded_size, self.expanded_size, self.expanded_size),
            config={
                'psf_sigma': self.psf_sigma,
                'edge_softness': self.edge_softness,
            },
        )
        self._deform_sampler = DeformationSampler(
            volume_shape=(self.size, self.size, self.size),
            type_probs=self.type_probs,
        )

    # -------------------------------------------------------------------------

    def generate(self, seed):
        """Generate one sample.

        Returns
        -------
        sample : dict with keys
            I1: (size, size, size) float32, noisy reference image
            I2: (size, size, size) float32, noisy warped image (indep noise)
            flow: (3, size, size, size) float32, GT displacement on input grid
            metadata: dict
        """
        rng = np.random.default_rng(seed)

        # ---------- 1. Bead placement in expanded volume ----------
        expanded_voxels = self.expanded_size ** 3
        target_count = int(round(self.density * expanded_voxels / 1000.0))
        positions_exp, place_info = place_beads_grid(
            self.expanded_size,
            target_count,
            self.radius,
            min_distance_factor=self.min_distance_factor,
            rng=rng,
        )
        radii, intensities = sample_bead_attributes(
            len(positions_exp), self.radius, rng
        )

        # ---------- 2. Deformation sampling ----------
        # In "fm" mode draw fm_target then scale to input; in "input" mode
        # draw input_max_disp directly and derive fm_target inversely.
        if self.disp_mode == "input":
            input_max_disp = float(rng.uniform(self.input_disp_min,
                                               self.input_disp_max))
            fm_target = input_max_disp / self.downsample_factor
        else:
            fm_target = float(rng.uniform(self.fm_min, self.fm_max))
            input_max_disp = fm_target * self.downsample_factor
        flow_fn, flow_grid, deform_meta = self._deform_sampler.sample(
            target_max_disp=input_max_disp, rng=rng
        )

        # ---------- 3. Forward warp bead centers ----------
        # Flow lives in the cropped frame. Beads in expanded frame have
        # offset = buffer_total relative to the cropped frame.
        positions_crop = positions_exp - self.buffer_total
        displacements = flow_fn(positions_crop)  # (N, 3)
        positions_exp_warped = positions_exp + displacements

        # ---------- 4. Render I1 and I2 on expanded volume ----------
        beads_ref = {
            'positions': positions_exp.astype(np.float32),
            'radii': radii,
            'intensities': intensities,
            'num_beads': int(len(positions_exp)),
        }
        beads_def = {
            'positions': positions_exp_warped.astype(np.float32),
            'radii': radii,
            'intensities': intensities,
            'num_beads': int(len(positions_exp)),
        }
        I1_exp = self._renderer.render(beads_ref)
        I2_exp = self._renderer.render(beads_def)

        # ---------- 5. Center crop both to input size ----------
        b = self.buffer_total
        s = self.size
        I1 = I1_exp[b:b + s, b:b + s, b:b + s].copy()
        I2 = I2_exp[b:b + s, b:b + s, b:b + s].copy()

        # ---------- 6. Independent Poisson + Gaussian noise ----------
        # Derive two independent child RNGs so I1 / I2 get fully independent noise.
        seed_1 = int(rng.integers(0, 2**31 - 1))
        seed_2 = int(rng.integers(0, 2**31 - 1))
        I1_noisy = apply_noise(
            I1, self.noise_gain, self.noise_read_sigma, np.random.default_rng(seed_1)
        )
        I2_noisy = apply_noise(
            I2, self.noise_gain, self.noise_read_sigma, np.random.default_rng(seed_2)
        )

        # ---------- 7. Metadata ----------
        metadata = {
            'deform_type': deform_meta['deform_type'],
            'fm_target': float(fm_target),
            'input_max_disp': float(input_max_disp),
            'achieved_max_disp': float(deform_meta['achieved_max_disp']),
            'scale_factor': float(deform_meta['scale_factor']),
            'radius': int(self.radius),
            'density_per_1000': float(self.density),
            'size': int(self.size),
            'downsample_factor': float(self.downsample_factor),
            'feature_map_size': int(self.feature_map_size),
            'disp_mode': self.disp_mode,
            'input_disp_min': float(self.input_disp_min),
            'input_disp_max': float(self.input_disp_max),
            'seed': int(seed),
            'num_beads_placed': int(place_info['placed']),
            'num_beads_requested': int(place_info['requested']),
            'early_stopped': bool(place_info['early_stopped']),
            'buffer_total': int(self.buffer_total),
            'expanded_size': int(self.expanded_size),
            'noise_gain': float(self.noise_gain),
            'noise_read_sigma': float(self.noise_read_sigma),
            'noise_seed_I1': seed_1,
            'noise_seed_I2': seed_2,
            'coefficients': deform_meta['coefficients'],
        }

        return {
            'I1': I1_noisy.astype(np.float32),
            'I2': I2_noisy.astype(np.float32),
            'flow': flow_grid.astype(np.float32),
            'metadata': metadata,
        }
