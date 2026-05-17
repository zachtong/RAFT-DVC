"""Generate a multi-page PDF preview of one or more phase-1 NPZ samples.

Useful for sanity-checking a dataset (bead size, density, displacement
magnitude, noise level) before/during training -- especially when the
sample metadata is sparse.

Usage:
    python scripts/preview_phase1_sample.py \
        --data-dir data_paper1_v2/r4_medium_size64/train \
        --indices 0 1 2 \
        --output reports/paper1_v2_r4md64_preview.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def load_sample(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load (I1, I2, flow, metadata) from a phase-1 sample NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    metadata_raw = data["metadata"]
    metadata = json.loads(str(metadata_raw)) if metadata_raw.size == 1 else {}
    return data["I1"], data["I2"], data["flow"], metadata


def render_sample_page(
    pdf: PdfPages,
    sample_idx: int,
    I1: np.ndarray,
    I2: np.ndarray,
    flow: np.ndarray,
    metadata: dict,
    title_prefix: str,
) -> None:
    """Render a single sample as one PDF page with central slices + flow."""
    size = I1.shape[0]
    mid = size // 2

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f"{title_prefix}  sample_{sample_idx:05d}   "
        f"input_disp range [{metadata.get('input_disp_min', '?')},"
        f" {metadata.get('input_disp_max', '?')}]   "
        f"this sample max disp = {metadata.get('input_max_disp', 0):.2f}",
        fontsize=12,
    )

    # Row 0: I1 central slices (3 axes)
    for col, (axis_idx, axis_name) in enumerate(zip([0, 1, 2], ["YZ", "XZ", "XY"])):
        slc = np.take(I1, mid, axis=axis_idx)
        im = axes[0, col].imshow(slc, cmap="gray", origin="lower")
        axes[0, col].set_title(f"I1  {axis_name} slice @ {axis_name[0]}={mid}")
        axes[0, col].set_xlabel("axis")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
    axes[0, 3].hist(I1.ravel(), bins=80, color="C0", alpha=0.7)
    axes[0, 3].set_title(f"I1 intensity histogram\nmean={I1.mean():.3f} max={I1.max():.3f}")
    axes[0, 3].set_xlabel("intensity")

    # Row 1: I2 central slices (3 axes)
    for col, (axis_idx, axis_name) in enumerate(zip([0, 1, 2], ["YZ", "XZ", "XY"])):
        slc = np.take(I2, mid, axis=axis_idx)
        im = axes[1, col].imshow(slc, cmap="gray", origin="lower")
        axes[1, col].set_title(f"I2  {axis_name} slice @ {axis_name[0]}={mid}")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)
    axes[1, 3].hist(I2.ravel(), bins=80, color="C1", alpha=0.7)
    axes[1, 3].set_title(f"I2 intensity histogram\nmean={I2.mean():.3f} max={I2.max():.3f}")
    axes[1, 3].set_xlabel("intensity")

    # Row 2: flow components + magnitude (all at z=mid central slice)
    flow_at_mid = flow[:, :, :, mid]  # (3, H, W)
    flow_mag = np.linalg.norm(flow, axis=0)
    for col, comp in enumerate(["Fx", "Fy", "Fz"]):
        slc = flow_at_mid[col]
        vmax = max(abs(slc.min()), abs(slc.max())) or 1.0
        im = axes[2, col].imshow(slc, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        axes[2, col].set_title(f"flow {comp}  XY slice @ z={mid}")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046)
    im = axes[2, 3].imshow(flow_mag[:, :, mid], cmap="viridis", origin="lower")
    axes[2, 3].set_title(
        f"|flow|  XY slice @ z={mid}\n"
        f"global max={flow_mag.max():.2f}  mean={flow_mag.mean():.2f}"
    )
    plt.colorbar(im, ax=axes[2, 3], fraction=0.046)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing sample_*.npz files.")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Which sample indices to render (default: 0 1 2).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output PDF path (e.g. reports/preview.pdf).")
    parser.add_argument("--title", type=str, default=None,
                        help="Override the title prefix (default: data-dir basename).")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    title_prefix = args.title or args.data_dir.parent.name

    with PdfPages(args.output) as pdf:
        for idx in args.indices:
            npz_path = args.data_dir / f"sample_{idx:05d}.npz"
            if not npz_path.exists():
                print(f"[skip] {npz_path} not found")
                continue
            I1, I2, flow, metadata = load_sample(npz_path)
            render_sample_page(pdf, idx, I1, I2, flow, metadata, title_prefix)
            print(f"[ok]  page added for sample_{idx:05d}")

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
