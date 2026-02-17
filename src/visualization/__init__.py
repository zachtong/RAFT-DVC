"""Visualization utilities for RAFT-DVC.

This module provides decoupled, reusable visualization functions for:
- 3D volume rendering (uncertainty, feature density, etc.)
- 2D slices and heatmaps
- Statistical plots (scatter, correlation, etc.)

All functions are designed to be:
- Framework-agnostic (work with numpy arrays)
- Backend-flexible (matplotlib, pyvista, etc.)
- Extensible (easy to add new viz types)
"""

from .volume_render import (
    render_uncertainty_volume_mpl,
    render_uncertainty_volume_pyvista,
    auto_render_uncertainty_volume,
    compute_feature_density,
)
from .slice_viz import (
    create_uncertainty_comparison_figure,
)
from .side_by_side_render import (
    render_side_by_side_pyvista,
    render_side_by_side_matplotlib,
    auto_render_side_by_side,
)
from .triple_volume_render import (
    render_triple_volume_pyvista,
)

__all__ = [
    'render_uncertainty_volume_mpl',
    'render_uncertainty_volume_pyvista',
    'auto_render_uncertainty_volume',
    'compute_feature_density',
    'create_uncertainty_comparison_figure',
    'render_side_by_side_pyvista',
    'render_side_by_side_matplotlib',
    'auto_render_side_by_side',
    'render_triple_volume_pyvista',
]
