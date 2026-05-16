"""Per-dataset QC visualization.

For each generated dataset folder produce a ``visualizations/`` subdir
containing one ``sample_NN_overview.png`` per sample (default: first 10
samples of the train split).

Each overview PNG has three panels:

* Panel 1 (left, 3x3 grid)  - I1, I2, |flow| at three orthogonal mid-slices
  (xy, xz, yz).
* Panel 2 (middle, 2x2 grid) - displacement magnitude histogram, per-axis
  displacement histograms, |u| sorted scatter, and 2D heatmap of |u| at the
  z-mid slice.
* Panel 3 (right, 1 axes 3D) - 3D scatter of bright voxels (proxy for beads)
  with a subsampled quiver field showing the flow.

Backend: matplotlib only (no PyVista dependency).  Quality is sufficient for
QC: did we put the right disp magnitude, right type, right bead density?
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# -----------------------------------------------------------------------------
# matplotlib compatibility shim
# -----------------------------------------------------------------------------
# Some conda envs ship a numpy/matplotlib pair that breaks at import time
# (matplotlib<3.8 tries ``from numpy import VisibleDeprecationWarning`` which
# numpy>=2.0 removed; matplotlib>=3.8 tries ``from numpy.exceptions import
# VisibleDeprecationWarning`` which numpy<1.25 lacks).  Inject the symbol in
# both locations so either matplotlib branch can succeed.

def _install_numpy_visible_deprecation_shim() -> None:
    """Make ``numpy.VisibleDeprecationWarning`` and ``numpy.exceptions``
    available regardless of numpy version, so matplotlib import succeeds."""
    target = None
    try:
        from numpy.exceptions import VisibleDeprecationWarning as _Native
        target = _Native
    except Exception:
        target = getattr(np, "VisibleDeprecationWarning", None)
    if target is None:
        target = DeprecationWarning  # last-resort dummy
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = target  # type: ignore[attr-defined]
    try:
        import numpy.exceptions  # noqa: F401
    except Exception:
        import types
        ne = types.ModuleType("numpy.exceptions")
        ne.VisibleDeprecationWarning = target
        np.exceptions = ne  # type: ignore[attr-defined]
        import sys as _sys
        _sys.modules["numpy.exceptions"] = ne


def _try_import_matplotlib():
    """Import matplotlib lazily with a compatibility shim. Returns
    ``(plt, available, error_message)``.  When ``available`` is False the
    caller should skip visualization and emit a clear user-facing warning."""
    _install_numpy_visible_deprecation_shim()
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless rendering
        import matplotlib.pyplot as plt  # noqa: F401
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        return plt, True, ""
    except Exception as exc:  # noqa: BLE001
        return None, False, f"{type(exc).__name__}: {exc}"


_plt, _MPL_AVAILABLE, _MPL_ERR = _try_import_matplotlib()
if _MPL_AVAILABLE:
    plt = _plt  # type: ignore[assignment]
else:
    plt = None  # type: ignore[assignment]
    warnings.warn(
        "[dataset_viz] matplotlib import failed -- visualization will be "
        f"skipped. Underlying error: {_MPL_ERR}\n"
        "  Fix (in the offending env):\n"
        "    pip install --upgrade 'numpy>=1.25,<2.1' 'matplotlib>=3.8'\n"
        "  or pin matplotlib older:\n"
        "    pip install 'matplotlib<3.8'\n"
        "  Generation will continue without visualizations; rerun later\n"
        "  with `--viz` after the env is fixed.",
        RuntimeWarning,
        stacklevel=2,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_sample(npz_path: Path) -> dict:
    """Load one sample NPZ and parse the JSON metadata string."""
    data = np.load(npz_path, allow_pickle=True)
    meta_raw = data["metadata"]
    try:
        meta = json.loads(str(meta_raw))
    except Exception:
        meta = {"_raw_metadata_load_failed": True}
    return {
        "I1": data["I1"],
        "I2": data["I2"],
        "flow": data["flow"],
        "metadata": meta,
    }


def _flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Return |u| from a (3, D, H, W) flow tensor."""
    return np.sqrt(np.sum(flow.astype(np.float32) ** 2, axis=0))


def _bead_proxy_points(I1: np.ndarray, top_percentile: float = 99.0,
                       max_points: int = 1500) -> np.ndarray:
    """Pick bright voxels as bead-position proxies for 3D rendering."""
    thresh = np.percentile(I1, top_percentile)
    mask = I1 >= thresh
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    zs, ys, xs = np.where(mask)
    pts = np.stack([zs, ys, xs], axis=1).astype(np.float32)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), size=max_points,
                                              replace=False)
        pts = pts[idx]
    return pts


# -----------------------------------------------------------------------------
# Panel renderers
# -----------------------------------------------------------------------------

def _draw_slice_panel(axes_grid: Sequence[Sequence[plt.Axes]],
                      I1: np.ndarray, I2: np.ndarray, mag: np.ndarray,
                      size: int) -> None:
    """3 rows (I1, I2, |flow|) x 3 cols (xy, xz, yz mid-slices)."""
    mid = size // 2
    vmax_img = max(float(I1.max()), float(I2.max()), 1e-6)
    vmax_mag = float(mag.max()) + 1e-6

    panels = [
        ("I1",     I1,  "gray",   0.0, vmax_img),
        ("I2",     I2,  "gray",   0.0, vmax_img),
        ("|flow|", mag, "viridis", 0.0, vmax_mag),
    ]
    view_titles = ["xy (z=mid)", "xz (y=mid)", "yz (x=mid)"]

    for r, (label, vol, cmap, vmin, vmax) in enumerate(panels):
        slices = [vol[mid, :, :], vol[:, mid, :], vol[:, :, mid]]
        for c, (slc, view_title) in enumerate(zip(slices, view_titles)):
            ax = axes_grid[r][c]
            im = ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin="lower")
            if r == 0:
                ax.set_title(view_title, fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _draw_disp_panel(axes_grid: Sequence[Sequence[plt.Axes]],
                     flow: np.ndarray, mag: np.ndarray, size: int) -> None:
    """2x2: |u| hist, per-axis hist, |u| sorted, |u| heatmap at z=mid."""
    mag_flat = mag.ravel()
    uz, uy, ux = flow[0].ravel(), flow[1].ravel(), flow[2].ravel()

    ax = axes_grid[0][0]
    ax.hist(mag_flat, bins=40, color="C2", alpha=0.85)
    ax.set_title("|u| distribution", fontsize=8)
    ax.set_xlabel("|u| (voxels)", fontsize=7)
    ax.set_ylabel("count", fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes_grid[0][1]
    bins = 40
    for arr, label, color in [(uz, "uz", "C0"), (uy, "uy", "C1"), (ux, "ux", "C3")]:
        ax.hist(arr, bins=bins, alpha=0.5, label=label, color=color)
    ax.set_title("per-axis disp", fontsize=8)
    ax.set_xlabel("voxels", fontsize=7)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes_grid[1][0]
    n_show = min(2000, mag_flat.size)
    sample_idx = np.linspace(0, mag_flat.size - 1, n_show).astype(np.int64)
    ax.plot(np.sort(mag_flat[sample_idx]), color="C2")
    ax.set_title("|u| sorted (sampled)", fontsize=8)
    ax.set_xlabel("rank", fontsize=7)
    ax.set_ylabel("|u|", fontsize=7)
    ax.tick_params(labelsize=7)

    ax = axes_grid[1][1]
    mid = size // 2
    im = ax.imshow(mag[mid], cmap="viridis", origin="lower",
                   vmin=0.0, vmax=float(mag.max()) + 1e-6)
    ax.set_title(f"|u| at z={mid}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _draw_3d_panel(ax: plt.Axes, I1: np.ndarray, flow: np.ndarray,
                   size: int, meta: dict) -> None:
    """3D scatter of bead-proxy voxels + subsampled quiver of flow."""
    bead_pts = _bead_proxy_points(I1)
    ax.scatter(
        bead_pts[:, 2], bead_pts[:, 1], bead_pts[:, 0],
        s=4, c="C1", alpha=0.45, depthshade=False, label="bright voxels",
    )

    quiver_stride = max(1, size // 6)
    coords = np.arange(0, size, quiver_stride)
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing="ij")
    UZ = flow[0, Z, Y, X]
    UY = flow[1, Z, Y, X]
    UX = flow[2, Z, Y, X]
    mag_q = np.sqrt(UZ ** 2 + UY ** 2 + UX ** 2)
    scale = float(np.maximum(mag_q.max(), 1e-6))
    # Normalize arrow lengths to about quiver_stride for visibility.
    length = quiver_stride * 1.4
    ax.quiver(
        X, Y, Z, UX / scale, UY / scale, UZ / scale,
        length=length, color="C0", normalize=False, linewidth=0.6,
        arrow_length_ratio=0.4,
    )

    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)
    ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7)
    ax.set_zlabel("z", fontsize=7)
    ax.tick_params(labelsize=6)
    deform = meta.get("deform_type", "?")
    fm_t = meta.get("fm_target", float("nan"))
    in_max = meta.get("input_max_disp", float("nan"))
    ax.set_title(
        f"3D | {deform}  fm={fm_t:.2f}  in_max={in_max:.2f}",
        fontsize=9,
    )


# -----------------------------------------------------------------------------
# One-sample composite figure
# -----------------------------------------------------------------------------

def render_sample_overview(sample: dict, out_path: Path) -> None:
    """Render the 3-panel overview for one sample and save as PNG."""
    I1, I2, flow, meta = sample["I1"], sample["I2"], sample["flow"], sample["metadata"]
    size = int(I1.shape[-1])
    mag = _flow_magnitude(flow)

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(
        nrows=3, ncols=8,
        width_ratios=[1, 1, 1.2, 0.2, 1.1, 1.1, 0.2, 1.6],
        hspace=0.35, wspace=0.30,
    )

    # Panel 1: slices (3x3)
    slice_axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
    _draw_slice_panel(slice_axes, I1, I2, mag, size)

    # Panel 2: disp (2x2). Center the 2x2 block in rows 0-2 by leaving row 2 empty.
    disp_axes = [
        [fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[0, 5])],
        [fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[1, 5])],
    ]
    _draw_disp_panel(disp_axes, flow, mag, size)
    # Row 2 of panel 2 used for a small metadata text box.
    meta_ax = fig.add_subplot(gs[2, 4:6])
    meta_ax.axis("off")
    meta_lines = [
        f"deform_type   : {meta.get('deform_type', '?')}",
        f"fm_target     : {meta.get('fm_target', float('nan')):.3f}",
        f"input_max_disp: {meta.get('input_max_disp', float('nan')):.3f}",
        f"achieved_max  : {meta.get('achieved_max_disp', float('nan')):.3f}",
        f"size          : {meta.get('size', size)}",
        f"downsample    : {meta.get('downsample_factor', float('nan')):.1f}",
        f"feature_map   : {meta.get('feature_map_size', '?')}",
        f"radius        : {meta.get('radius', '?')}",
        f"density/1000  : {meta.get('density_per_1000', float('nan')):.2f}",
        f"num_beads     : {meta.get('num_beads_placed', '?')}",
        f"mix_choice    : {meta.get('mix_choice', '-')}",
    ]
    meta_ax.text(0.0, 1.0, "\n".join(meta_lines),
                 fontsize=8, family="monospace", va="top")

    # Panel 3: 3D scatter
    ax3d = fig.add_subplot(gs[:, 7], projection="3d")
    _draw_3d_panel(ax3d, I1, flow, size, meta)

    title = (
        f"{out_path.parent.parent.name} | sample {out_path.stem.split('_')[1]}"
        f" | seed={meta.get('seed', '?')}"
    )
    fig.suptitle(title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def visualize_dataset(cfg_dir: Path, n_samples: int = 10,
                      split: str = "train") -> int:
    """Render overview PNGs for the first ``n_samples`` of ``split``.

    If matplotlib is unavailable in the current environment, emits a single
    warning and returns 0 instead of raising -- generation must not be
    blocked by a viz-only dependency problem.

    Returns
    -------
    int : number of PNGs successfully written.
    """
    if not _MPL_AVAILABLE:
        print(
            f"  [viz] SKIP {cfg_dir.name}: matplotlib unavailable "
            f"({_MPL_ERR.splitlines()[0] if _MPL_ERR else 'unknown'})"
        )
        return 0
    cfg_dir = Path(cfg_dir)
    split_dir = cfg_dir / split
    if not split_dir.exists():
        return 0

    out_dir = cfg_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = sorted(split_dir.glob("sample_*.npz"))[:n_samples]
    n_done = 0
    for path in sample_paths:
        try:
            sample = _load_sample(path)
            out_png = out_dir / f"{path.stem}_overview.png"
            render_sample_overview(sample, out_png)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [viz] WARN: failed on {path.name}: {exc}")
    return n_done
