"""
DIC shape-function deformation generators (0/1/2-order polynomial).

Finalized 2026-04-16 (see memory/displacement_warp_design.md):
- 0-order (translation, 3 params): pure rigid translation, no rotation.
- 1-order (affine, 12 params): t + (A - I)*(y - center). Includes rigid rotation,
  scale, shear as a special case.
- 2-order (quadratic, 30 params): 10 coefficients per component. Linearly-varying
  strain.

Unified magnitude normalization:
  1. Sample base coefficients from N(0, 1).
  2. Evaluate u on the input-volume integer-voxel grid.
  3. Compute max|u|_inf over all voxels and all 3 components.
  4. Rescale all coefficients by target_max / current_max.

This guarantees the output flow_grid has its max entry equal to target_max_disp
while preserving the random "shape" of the field.

The returned flow_fn is the continuous analytical form and can be evaluated at
arbitrary (off-grid) positions --- needed for forward bead warping where beads
live in the *expanded* coordinate frame.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Coefficient samplers (base distribution: N(0, 1))
# -----------------------------------------------------------------------------

def _sample_translation(rng):
    """Translation: t ∈ R^3."""
    return rng.normal(0.0, 1.0, size=3).astype(np.float32)


def _sample_affine(rng):
    """Affine: (t ∈ R^3, B ∈ R^{3x3}) where u(y) = t + B*(y - center)."""
    t = rng.normal(0.0, 1.0, size=3).astype(np.float32)
    B = rng.normal(0.0, 1.0, size=(3, 3)).astype(np.float32)
    return t, B


def _sample_quadratic(rng):
    """Quadratic: A ∈ R^{3x10}, 10 coeffs per component.

    For component i:
      u_i(y) = A[i, 0]
             + A[i, 1]*dz + A[i, 2]*dy + A[i, 3]*dx
             + A[i, 4]*dz^2 + A[i, 5]*dy^2 + A[i, 6]*dx^2
             + A[i, 7]*dz*dy + A[i, 8]*dz*dx + A[i, 9]*dy*dx
    where (dz, dy, dx) = y - center.
    """
    return rng.normal(0.0, 1.0, size=(3, 10)).astype(np.float32)


# -----------------------------------------------------------------------------
# Analytical flow_fn builders (evaluate at arbitrary positions)
# -----------------------------------------------------------------------------

def _make_translation_fn(t):
    t = np.asarray(t, dtype=np.float32)

    def fn(y):
        y = np.asarray(y, dtype=np.float32)
        out = np.broadcast_to(t, y.shape)
        return out.astype(np.float32, copy=True)
    return fn


def _make_affine_fn(t, B, center):
    t = np.asarray(t, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    def fn(y):
        y = np.asarray(y, dtype=np.float32)
        d = y - center  # (..., 3)
        return (d @ B.T + t).astype(np.float32)
    return fn


def _make_quadratic_fn(A, center):
    A = np.asarray(A, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    def fn(y):
        y = np.asarray(y, dtype=np.float32)
        d = y - center  # (N, 3)
        dz, dy_, dx_ = d[..., 0], d[..., 1], d[..., 2]
        ones = np.ones_like(dz)
        features = np.stack([
            ones,
            dz, dy_, dx_,
            dz * dz, dy_ * dy_, dx_ * dx_,
            dz * dy_, dz * dx_, dy_ * dx_,
        ], axis=-1)  # (N, 10)
        return (features @ A.T).astype(np.float32)
    return fn


# -----------------------------------------------------------------------------
# Grid evaluator
# -----------------------------------------------------------------------------

def eval_flow_on_grid(flow_fn, volume_shape):
    """Evaluate flow_fn on the integer-voxel grid of volume_shape.

    Args:
        flow_fn: callable (N, 3) -> (N, 3).
        volume_shape: (D, H, W).

    Returns:
        flow: (3, D, H, W) float32.
    """
    D, H, W = volume_shape
    zz, yy, xx = np.meshgrid(
        np.arange(D, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing='ij',
    )
    coords = np.stack([zz, yy, xx], axis=-1).reshape(-1, 3)  # (N, 3)
    u = flow_fn(coords)  # (N, 3)
    u = u.reshape(D, H, W, 3).transpose(3, 0, 1, 2)
    return u.astype(np.float32)


def _max_abs_inf(flow):
    """Max over all voxels of ||u||_inf (equivalently, max |flow_ijk|)."""
    return float(np.max(np.abs(flow))) if flow.size > 0 else 0.0


# -----------------------------------------------------------------------------
# Main sampler
# -----------------------------------------------------------------------------

class DeformationSampler:
    """Samples DIC shape-function deformations with unified magnitude norm.

    Per-sample draw:
      1. Sample deformation type with given probs.
      2. Sample base coefficients.
      3. Evaluate u on the input-volume grid; compute max|u|_inf.
      4. Rescale all coefficients by target_max / current_max.
      5. Return flow_fn (continuous, callable) + flow_grid + metadata.

    Args:
        volume_shape: (D, H, W) of the input volume (cropped output frame).
        type_probs: (p_translation, p_affine, p_quadratic). Default = 10/45/45.
    """

    TYPES = ('translation', 'affine', 'quadratic')

    def __init__(self, volume_shape, type_probs=(0.10, 0.45, 0.45)):
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.center = (np.array(self.volume_shape, dtype=np.float32) - 1.0) / 2.0
        self.type_probs = np.asarray(type_probs, dtype=np.float64)
        if not np.isclose(self.type_probs.sum(), 1.0):
            raise ValueError(
                f"type_probs must sum to 1 (got {self.type_probs.sum()})"
            )

    def sample(self, target_max_disp, rng):
        """Sample a deformation field achieving max|u|_inf == target_max_disp.

        Args:
            target_max_disp: desired maximum |displacement| in voxels.
            rng: numpy.random.Generator.

        Returns:
            flow_fn: callable (N, 3) -> (N, 3). Analytical form for arbitrary y.
            flow_grid: (3, D, H, W) float32. Evaluated on integer-voxel grid.
            metadata: dict.
        """
        type_idx = int(rng.choice(len(self.TYPES), p=self.type_probs))
        deform_type = self.TYPES[type_idx]

        if deform_type == 'translation':
            t = _sample_translation(rng)
            flow_fn_base = _make_translation_fn(t)
            base_coeffs = {'t': t.tolist()}
        elif deform_type == 'affine':
            t, B = _sample_affine(rng)
            flow_fn_base = _make_affine_fn(t, B, self.center)
            base_coeffs = {'t': t.tolist(), 'B': B.tolist()}
        else:  # quadratic
            A = _sample_quadratic(rng)
            flow_fn_base = _make_quadratic_fn(A, self.center)
            base_coeffs = {'A': A.tolist()}

        # Evaluate on the input-volume grid to measure current max
        flow_grid_base = eval_flow_on_grid(flow_fn_base, self.volume_shape)
        current_max = _max_abs_inf(flow_grid_base)
        if current_max < 1e-6:
            # Degenerate sample; re-draw by scaling from an epsilon
            current_max = 1e-6
        scale = float(target_max_disp) / current_max

        # Scale coefficients
        if deform_type == 'translation':
            t_s = t * scale
            flow_fn = _make_translation_fn(t_s)
            scaled_coeffs = {'t': t_s.tolist()}
        elif deform_type == 'affine':
            t_s, B_s = t * scale, B * scale
            flow_fn = _make_affine_fn(t_s, B_s, self.center)
            scaled_coeffs = {'t': t_s.tolist(), 'B': B_s.tolist()}
        else:  # quadratic
            A_s = A * scale
            flow_fn = _make_quadratic_fn(A_s, self.center)
            scaled_coeffs = {'A': A_s.tolist()}

        flow_grid = (flow_grid_base * scale).astype(np.float32)
        achieved_max = _max_abs_inf(flow_grid)

        metadata = {
            'deform_type': deform_type,
            'input_max_disp': float(target_max_disp),
            'achieved_max_disp': float(achieved_max),
            'scale_factor': float(scale),
            'type_probs': self.type_probs.tolist(),
            'center': self.center.tolist(),
            'coefficients': scaled_coeffs,
        }

        return flow_fn, flow_grid, metadata
