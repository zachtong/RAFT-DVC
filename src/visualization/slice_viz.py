"""2D slice visualization utilities.

Provides functions for creating 2D heatmaps, scatter plots, and
multi-panel comparison figures.

Uses lazy imports to avoid dependency errors at module import time.
"""

import numpy as np
from typing import Optional, Tuple


def create_uncertainty_comparison_figure(
    volume: np.ndarray,
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    uncertainty: np.ndarray,
    feature_density: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """Create comprehensive uncertainty comparison figure.

    This is the main 2D visualization function, creating a multi-panel
    figure with:
        - Input volume (mid-slice)
        - Error magnitude heatmap
        - Uncertainty heatmap
        - Feature density heatmap (optional)
        - Error vs Uncertainty scatter plot
        - Spearman correlation coefficient

    Args:
        volume: (D, H, W) input volume
        flow_pred: (3, D, H, W) predicted flow
        flow_gt: (3, D, H, W) ground truth flow
        uncertainty: (D, H, W) uncertainty map (averaged over components)
        feature_density: (D, H, W) optional density map
        slice_idx: Z-slice index (default: middle slice)
        figsize: figure size (width, height)

    Returns:
        fig: matplotlib Figure with 2x3 or 2x2 subplots
    """
    # Lazy import
    import matplotlib.pyplot as plt

    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Compute error magnitude
    error = np.sqrt(((flow_pred - flow_gt) ** 2).sum(axis=0))  # (D, H, W)

    # Extract slices
    vol_slice = volume[slice_idx]
    err_slice = error[slice_idx]
    unc_slice = uncertainty[slice_idx]

    # Determine layout
    if feature_density is not None:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        density_slice = feature_density[slice_idx]
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)

    # === Panel 1: Input volume ===
    ax = axes[0, 0]
    ax.imshow(vol_slice, cmap='gray')
    ax.set_title('Input Volume', fontsize=12)
    ax.axis('off')

    # === Panel 2: Error heatmap ===
    ax = axes[0, 1]
    vmax_err = np.percentile(err_slice, 99)
    im1 = ax.imshow(err_slice, cmap='hot', vmin=0, vmax=vmax_err)
    ax.set_title(f'Error |pred - GT| (mean={err_slice.mean():.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # === Panel 3: Uncertainty heatmap ===
    if feature_density is not None:
        ax = axes[0, 2]
    else:
        ax = axes[1, 0]

    vmax_unc = np.percentile(unc_slice, 99)
    im2 = ax.imshow(unc_slice, cmap='hot', vmin=0, vmax=vmax_unc)
    ax.set_title(f'Uncertainty (mean={unc_slice.mean():.3f})', fontsize=12)
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # === Panel 4: Feature density (optional) ===
    if feature_density is not None:
        ax = axes[1, 0]
        vmax_density = np.percentile(density_slice, 99)
        im3 = ax.imshow(density_slice, cmap='Greens', vmin=0, vmax=vmax_density)
        ax.set_title(f'Feature Density (mean={density_slice.mean():.3f})', fontsize=12)
        ax.axis('off')
        plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    # === Panel 5: Error vs Uncertainty scatter ===
    if feature_density is not None:
        ax = axes[1, 1]
    else:
        ax = axes[1, 1]

    # Subsample for scatter plot (avoid overcrowding)
    err_flat = err_slice.flatten()
    unc_flat = unc_slice.flatten()
    n = len(err_flat)
    if n > 3000:
        idx = np.random.choice(n, 3000, replace=False)
        err_sample = err_flat[idx]
        unc_sample = unc_flat[idx]
    else:
        err_sample = err_flat
        unc_sample = unc_flat

    ax.scatter(unc_sample, err_sample, alpha=0.3, s=5, c='steelblue')
    ax.set_xlabel('Uncertainty', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Error vs Uncertainty', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Compute Spearman correlation (full volume, not just slice)
    err_vol = error.flatten()
    unc_vol = uncertainty.flatten()
    rank_err = np.empty_like(err_vol)
    rank_unc = np.empty_like(unc_vol)
    rank_err[np.argsort(err_vol)] = np.arange(len(err_vol))
    rank_unc[np.argsort(unc_vol)] = np.arange(len(unc_vol))
    rho = np.corrcoef(rank_err, rank_unc)[0, 1]

    ax.text(
        0.05, 0.95, f'Spearman ρ={rho:.3f}\n(full volume)',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    # === Panel 6: Correlation by density bins (optional) ===
    if feature_density is not None:
        ax = axes[1, 2]

        # Bin by density quartiles
        density_flat = feature_density.flatten()
        quartiles = np.percentile(density_flat, [25, 50, 75])

        bin_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        bin_rhos = []

        for i in range(4):
            if i == 0:
                mask = density_flat <= quartiles[0]
            elif i == 3:
                mask = density_flat > quartiles[2]
            else:
                mask = (density_flat > quartiles[i-1]) & (density_flat <= quartiles[i])

            if mask.sum() > 10:  # enough samples
                err_bin = err_vol[mask]
                unc_bin = unc_vol[mask]
                rank_err_bin = np.empty_like(err_bin)
                rank_unc_bin = np.empty_like(unc_bin)
                rank_err_bin[np.argsort(err_bin)] = np.arange(len(err_bin))
                rank_unc_bin[np.argsort(unc_bin)] = np.arange(len(unc_bin))
                rho_bin = np.corrcoef(rank_err_bin, rank_unc_bin)[0, 1]
                bin_rhos.append(rho_bin)
            else:
                bin_rhos.append(0.0)

        ax.bar(bin_labels, bin_rhos, color=['lightcoral', 'coral', 'lightgreen', 'green'])
        ax.set_ylabel('Spearman ρ', fontsize=11)
        ax.set_title('Correlation by Density Quartile', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle(f'Uncertainty Analysis (Z-slice {slice_idx})', fontsize=14, y=0.98)
    plt.tight_layout()

    return fig
