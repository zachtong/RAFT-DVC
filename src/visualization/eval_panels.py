"""Shared per-sample evaluation figure used by both evaluate_phase1
and evaluate_ood.

Builds a 4-row x 3-col grid at the z-mid slice:

  row 0: I1 (ref)      I2 (warped)      |EPE|
  row 1: GT U(z)       GT V(y)          GT W(x)
  row 2: pred U        pred V           pred W
  row 3: error U       error V          error W
        (= pred - GT, signed, separate scale)

Per-component GT and pred share a symmetric colour scale (RdBu_r), so any
spatial mismatch is visually obvious.  Error row uses its own scale to
make small residuals visible.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


_DEFAULT_COMP_LABELS: Tuple[str, str, str] = ("U (z)", "V (y)", "W (x)")


def render_flow_eval_4x3(
    vol0: np.ndarray,
    vol1: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    z: Optional[int] = None,
    comp_labels: Sequence[str] = _DEFAULT_COMP_LABELS,
    figsize: Tuple[float, float] = (12.0, 14.0),
):
    """Build the 4x3 evaluation figure for one sample.

    Args:
        vol0: (D, H, W)   reference volume.
        vol1: (D, H, W)   warped volume.
        gt:   (3, D, H, W) ground-truth displacement (U, V, W) = (z, y, x).
        pred: (3, D, H, W) predicted displacement.
        z:    z-slice index to display. ``None`` -> middle slice.
        comp_labels: labels for the 3 flow components.
        figsize: matplotlib figure size in inches.

    Returns:
        ``(fig, axes)``.  Caller is responsible for ``fig.suptitle(...)``
        and ``fig.tight_layout(rect=(0, 0, 1, 0.97))``.
    """
    # Lazy import keeps the module importable in headless / mpl-less envs.
    import matplotlib.pyplot as plt

    if z is None:
        z = int(vol0.shape[0]) // 2

    err_mag = np.sqrt(((pred - gt) ** 2).sum(axis=0))[z]  # (H, W)

    fig, axes = plt.subplots(4, 3, figsize=figsize)

    # ---- Row 0: I1, I2, |EPE| -------------------------------------------
    im00 = axes[0, 0].imshow(vol0[z], cmap="gray")
    axes[0, 0].set_title("I1 (ref)", fontsize=10)
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.02)

    im01 = axes[0, 1].imshow(vol1[z], cmap="gray")
    axes[0, 1].set_title("I2 (warped)", fontsize=10)
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.02)

    im02 = axes[0, 2].imshow(err_mag, cmap="inferno")
    axes[0, 2].set_title(
        f"|EPE| this slice  (mean={err_mag.mean():.3f})", fontsize=10,
    )
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.02)

    # ---- Rows 1-3: per-component GT / pred / error ----------------------
    for k, comp in enumerate(comp_labels):
        gt_c = gt[k, z]
        pred_c = pred[k, z]
        diff_c = pred_c - gt_c

        # Shared symmetric scale for GT and pred (so they're visually comparable)
        span = float(max(abs(gt_c).max(), abs(pred_c).max(), 1e-6))
        # Error has its own (typically tighter) scale
        diff_span = float(max(abs(diff_c).max(), 1e-6))

        im_gt = axes[1, k].imshow(
            gt_c, cmap="RdBu_r", vmin=-span, vmax=span,
        )
        axes[1, k].set_title(
            f"GT {comp}\n[min={gt_c.min():+.2f}, max={gt_c.max():+.2f}]",
            fontsize=9,
        )
        plt.colorbar(im_gt, ax=axes[1, k], fraction=0.046, pad=0.02)

        im_pr = axes[2, k].imshow(
            pred_c, cmap="RdBu_r", vmin=-span, vmax=span,
        )
        axes[2, k].set_title(
            f"pred {comp}\n[min={pred_c.min():+.2f}, max={pred_c.max():+.2f}]",
            fontsize=9,
        )
        plt.colorbar(im_pr, ax=axes[2, k], fraction=0.046, pad=0.02)

        im_df = axes[3, k].imshow(
            diff_c, cmap="seismic", vmin=-diff_span, vmax=diff_span,
        )
        axes[3, k].set_title(
            f"error {comp} = pred - GT\n"
            f"(MAE={abs(diff_c).mean():.3f}, max={diff_span:.2f})",
            fontsize=9,
        )
        plt.colorbar(im_df, ax=axes[3, k], fraction=0.046, pad=0.02)

    # Hide tick marks for cleanliness
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    return fig, axes


__all__ = ["render_flow_eval_4x3"]
