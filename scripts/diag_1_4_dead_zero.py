"""Diagnose paper-1 1/4 dead-zero failure: inspect data + encoder outputs.

Three-layer diagnostic with side-by-side 1/8 vs 1/4 comparison:

  1. Data: I1, I2 central slices, flow magnitude, statistics.
  2. Encoder output (random init, same seed):
      - Per-channel statistics (mean, variance) at fm scale
      - Per-channel slice visualization
  3. Gradient signal (the KEY diagnostic):
      - corr_at_zero(p) = <fmap1[p], fmap2[p]> / sqrt(C)
      - corr_at_truth(p) = <fmap1[p], fmap2[p + flow_fm[p]]> / sqrt(C)
      - Difference distribution: positive mean => learnable; near zero => dead.

If 1/4 corr_at_truth ~ corr_at_zero at random init, the encoder produces
features too similar across nearby spatial positions -- model has no
gradient signal to escape the zero-flow attractor.

Usage:
    python scripts/diag_1_4_dead_zero.py ^
        --data-dir data_paper1_v2/r4_medium_size64/train ^
        --indices 0 1 2 5 ^
        --output reports/diag_1_4_dead_zero_2026-05-18.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from src.core import RAFTDVC, RAFTDVCConfig
from src.core.corr import bilinear_sampler_3d


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_sample(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    raw = d["metadata"]
    try:
        md = json.loads(str(raw)) if raw.size == 1 else {}
    except Exception:
        md = {}
    return {
        "I1": d["I1"].astype(np.float32),
        "I2": d["I2"].astype(np.float32),
        "flow": d["flow"].astype(np.float32),
        "metadata": md,
        "path": npz_path,
    }


def build_random_encoder(yaml_path: Path, seed: int = 42) -> RAFTDVC:
    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg_data = yaml.safe_load(fh)
    cfg = RAFTDVCConfig.from_dict(cfg_data["architecture"])
    cfg.corr_impl = "standard"  # extractor itself is independent of corr_impl
    torch.manual_seed(seed)
    model = RAFTDVC(cfg).cuda().eval()
    return model


def get_fmap(model: RAFTDVC, vol: torch.Tensor) -> torch.Tensor:
    """Return (1, C, H_fm, W_fm, D_fm) encoded feature map."""
    with torch.no_grad():
        # RAFT-DVC's fnet takes 5D (B, 1, H, W, D), returns (B, C, H_fm, ...)
        return model.fnet(vol)


def lookup_fmap2_at(fmap2: torch.Tensor, coords_fm: torch.Tensor) -> torch.Tensor:
    """Sample fmap2 at coords_fm via trilinear interp.

    coords_fm channel order = (H, D, W) matching corr.py convention.
    Returns (B, C, H, W, D).
    """
    coords_perm = coords_fm.permute(0, 2, 3, 4, 1)  # (B, H, W, D, 3)
    return bilinear_sampler_3d(fmap2, coords_perm)


def compute_corr_signal(
    fmap1: torch.Tensor, fmap2: torch.Tensor, flow_fm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute corr at zero-flow and at true-flow for each spatial position.

    Returns:
        corr_at_zero: (H_fm, W_fm, D_fm)
        corr_at_truth: (H_fm, W_fm, D_fm)
    """
    B, C, H, W, D = fmap1.shape
    device = fmap1.device

    # Identity coords (H, D, W channel order to match flow_fm)
    base_h = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1, 1).expand(B, 1, H, W, D)
    base_w = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W, 1).expand(B, 1, H, W, D)
    base_d = torch.arange(D, device=device, dtype=torch.float32).view(1, 1, 1, 1, D).expand(B, 1, H, W, D)
    coords_id = torch.cat([base_h, base_d, base_w], dim=1)  # (1, 3, H, W, D) in (H, D, W) order

    coords_truth = coords_id + flow_fm  # add flow offsets

    with torch.no_grad():
        f2_id = lookup_fmap2_at(fmap2, coords_id)
        f2_truth = lookup_fmap2_at(fmap2, coords_truth)
        norm = math.sqrt(C)
        corr_at_zero = (fmap1 * f2_id).sum(1).squeeze(0) / norm     # (H, W, D)
        corr_at_truth = (fmap1 * f2_truth).sum(1).squeeze(0) / norm
    return corr_at_zero, corr_at_truth


def flow_to_fm_space(flow_input: torch.Tensor, ds_factor: int) -> torch.Tensor:
    """Convert flow (in input voxel units) to fm voxel units.

    Also downsamples spatially via avg pool to match fmap resolution.
    flow_input: (1, 3, H_in, W_in, D_in)
    Returns: (1, 3, H_fm, W_fm, D_fm)
    """
    # Downsample with avg pooling, scale magnitudes by 1/ds
    flow_fm = torch.nn.functional.avg_pool3d(
        flow_input, kernel_size=ds_factor, stride=ds_factor,
    ) / ds_factor
    return flow_fm


# -----------------------------------------------------------------------------
# PDF rendering
# -----------------------------------------------------------------------------

def render_input(pdf: PdfPages, sample: dict, sample_idx: int) -> None:
    I1, I2, flow = sample["I1"], sample["I2"], sample["flow"]
    md = sample["metadata"]
    size = I1.shape[0]
    mid = size // 2

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f"sample_{sample_idx:05d}  input-disp [{md.get('input_disp_min', '?')}, "
        f"{md.get('input_disp_max', '?')}]  this sample max_disp = "
        f"{md.get('input_max_disp', np.abs(flow).max()):.2f}",
        fontsize=12,
    )

    for col, (ax_idx, name) in enumerate(zip([0, 1, 2], ["YZ", "XZ", "XY"])):
        slc = np.take(I1, mid, axis=ax_idx)
        im = axes[0, col].imshow(slc, cmap="gray", origin="lower")
        axes[0, col].set_title(f"I1  {name} @ {name[0]}={mid}")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
    axes[0, 3].hist(I1.ravel(), bins=80, color="C0", alpha=0.7)
    axes[0, 3].set_title(f"I1 intensity\nmean={I1.mean():.3f} max={I1.max():.3f}")

    for col, (ax_idx, name) in enumerate(zip([0, 1, 2], ["YZ", "XZ", "XY"])):
        slc = np.take(I2, mid, axis=ax_idx)
        im = axes[1, col].imshow(slc, cmap="gray", origin="lower")
        axes[1, col].set_title(f"I2  {name} @ {name[0]}={mid}")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)
    axes[1, 3].hist(I2.ravel(), bins=80, color="C1", alpha=0.7)
    axes[1, 3].set_title(f"I2 intensity\nmean={I2.mean():.3f} max={I2.max():.3f}")

    flow_mag = np.linalg.norm(flow, axis=0)
    for col, comp in enumerate(["Fx", "Fy", "Fz"]):
        slc = flow[col, :, :, mid]
        vmax = max(abs(slc.min()), abs(slc.max())) or 1.0
        im = axes[2, col].imshow(slc, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        axes[2, col].set_title(f"flow {comp}  XY @ z={mid}")
        plt.colorbar(im, ax=axes[2, col], fraction=0.046)
    im = axes[2, 3].imshow(flow_mag[:, :, mid], cmap="viridis", origin="lower")
    axes[2, 3].set_title(
        f"|flow|  XY @ z={mid}\nmax={flow_mag.max():.2f}  mean={flow_mag.mean():.2f}"
    )
    plt.colorbar(im, ax=axes[2, 3], fraction=0.046)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def render_fmap_channels(
    pdf: PdfPages,
    fmap: torch.Tensor,
    label: str,
    sample_idx: int,
    n_channels: int = 12,
) -> None:
    """Show first n_channels of fmap as central slices."""
    fmap_np = fmap.squeeze(0).cpu().numpy()  # (C, H, W, D)
    C, H, W, D = fmap_np.shape
    mid = H // 2

    ncols = 4
    nrows = (n_channels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.ravel() if nrows * ncols > 1 else [axes]
    fig.suptitle(
        f"{label} encoder fmap (random init, sample_{sample_idx:05d})  "
        f"shape={fmap_np.shape}  first {n_channels} channels (XY @ z={mid})",
        fontsize=11,
    )

    for ch in range(n_channels):
        slc = fmap_np[ch, :, :, mid]
        vmax = max(abs(slc.min()), abs(slc.max())) or 1.0
        im = axes[ch].imshow(slc, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
        axes[ch].set_title(
            f"ch {ch}  std={slc.std():.3f}  range[{slc.min():.2f}, {slc.max():.2f}]",
            fontsize=9,
        )
        axes[ch].axis("off")
    for ch in range(n_channels, len(axes)):
        axes[ch].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def render_fmap_stats(pdf: PdfPages, fmap_1_8, fmap_1_4, sample_idx: int) -> None:
    """Side-by-side comparison of fmap statistics."""
    f_1_8 = fmap_1_8.squeeze(0).cpu().numpy()  # (C, H, W, D)
    f_1_4 = fmap_1_4.squeeze(0).cpu().numpy()

    # Per-channel statistics
    spatial_var_1_8 = f_1_8.reshape(f_1_8.shape[0], -1).var(axis=1)
    spatial_var_1_4 = f_1_4.reshape(f_1_4.shape[0], -1).var(axis=1)
    channel_mean_1_8 = f_1_8.reshape(f_1_8.shape[0], -1).mean(axis=1)
    channel_mean_1_4 = f_1_4.reshape(f_1_4.shape[0], -1).mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Encoder output statistics (random init), sample_{sample_idx:05d}",
        fontsize=12,
    )

    axes[0, 0].bar(range(len(spatial_var_1_8)), spatial_var_1_8, color="C0")
    axes[0, 0].set_title(
        f"1/8 encoder per-channel spatial variance\n"
        f"shape={f_1_8.shape}  mean_var={spatial_var_1_8.mean():.4f}"
    )
    axes[0, 0].set_xlabel("channel")
    axes[0, 0].set_ylabel("spatial var")

    axes[0, 1].bar(range(len(spatial_var_1_4)), spatial_var_1_4, color="C1")
    axes[0, 1].set_title(
        f"1/4 encoder per-channel spatial variance\n"
        f"shape={f_1_4.shape}  mean_var={spatial_var_1_4.mean():.4f}"
    )
    axes[0, 1].set_xlabel("channel")
    axes[0, 1].set_ylabel("spatial var")

    axes[1, 0].hist(f_1_8.ravel(), bins=80, color="C0", alpha=0.7)
    axes[1, 0].set_title(
        f"1/8 fmap value histogram\n"
        f"global std={f_1_8.std():.3f}  range[{f_1_8.min():.2f}, {f_1_8.max():.2f}]"
    )
    axes[1, 1].hist(f_1_4.ravel(), bins=80, color="C1", alpha=0.7)
    axes[1, 1].set_title(
        f"1/4 fmap value histogram\n"
        f"global std={f_1_4.std():.3f}  range[{f_1_4.min():.2f}, {f_1_4.max():.2f}]"
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def render_gradient_signal(
    pdf: PdfPages, fmap1_1_8, fmap2_1_8, fmap1_1_4, fmap2_1_4, flow, sample_idx,
) -> None:
    """⭐ KEY DIAGNOSTIC: corr_at_truth vs corr_at_zero for each encoder.

    If corr_at_truth - corr_at_zero ≈ 0 on average for 1/4, the model has
    no gradient signal to learn from -- predict zero is the local minimum.
    """
    flow_t = torch.from_numpy(flow).unsqueeze(0).cuda()  # (1, 3, H, W, D)

    flow_fm_1_8 = flow_to_fm_space(flow_t, ds_factor=8)
    flow_fm_1_4 = flow_to_fm_space(flow_t, ds_factor=4)

    cz_1_8, ct_1_8 = compute_corr_signal(fmap1_1_8, fmap2_1_8, flow_fm_1_8)
    cz_1_4, ct_1_4 = compute_corr_signal(fmap1_1_4, fmap2_1_4, flow_fm_1_4)

    diff_1_8 = (ct_1_8 - cz_1_8).cpu().numpy()
    diff_1_4 = (ct_1_4 - cz_1_4).cpu().numpy()
    cz_1_8 = cz_1_8.cpu().numpy(); ct_1_8 = ct_1_8.cpu().numpy()
    cz_1_4 = cz_1_4.cpu().numpy(); ct_1_4 = ct_1_4.cpu().numpy()

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    fig.suptitle(
        f"⭐ Gradient signal: corr_at_truth - corr_at_zero (sample_{sample_idx:05d})\n"
        f"random init encoders.  Positive mean => model can learn; ~zero => DEAD.",
        fontsize=12,
    )

    # Row 0: corr_at_zero spatial maps (mid-z slice)
    mid_8 = cz_1_8.shape[2] // 2
    mid_4 = cz_1_4.shape[2] // 2
    im = axes[0, 0].imshow(cz_1_8[:, :, mid_8], cmap="RdBu_r", origin="lower")
    axes[0, 0].set_title(f"1/8 corr_at_zero  (mean={cz_1_8.mean():.3f})")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
    im = axes[0, 1].imshow(cz_1_4[:, :, mid_4], cmap="RdBu_r", origin="lower")
    axes[0, 1].set_title(f"1/4 corr_at_zero  (mean={cz_1_4.mean():.3f})")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    # Row 1: corr_at_truth spatial maps
    im = axes[1, 0].imshow(ct_1_8[:, :, mid_8], cmap="RdBu_r", origin="lower")
    axes[1, 0].set_title(f"1/8 corr_at_truth  (mean={ct_1_8.mean():.3f})")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    im = axes[1, 1].imshow(ct_1_4[:, :, mid_4], cmap="RdBu_r", origin="lower")
    axes[1, 1].set_title(f"1/4 corr_at_truth  (mean={ct_1_4.mean():.3f})")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Row 2: difference histogram (the key diagnostic)
    bins = 60
    axes[2, 0].hist(diff_1_8.ravel(), bins=bins, color="C0", alpha=0.7, label="1/8")
    axes[2, 0].hist(diff_1_4.ravel(), bins=bins, color="C1", alpha=0.6, label="1/4")
    axes[2, 0].axvline(0, color="k", lw=1, ls="--")
    axes[2, 0].axvline(diff_1_8.mean(), color="C0", lw=2, label=f"1/8 mean={diff_1_8.mean():.3f}")
    axes[2, 0].axvline(diff_1_4.mean(), color="C1", lw=2, label=f"1/4 mean={diff_1_4.mean():.3f}")
    axes[2, 0].set_title("corr_at_truth - corr_at_zero  (combined histogram)")
    axes[2, 0].set_xlabel("Δ corr value")
    axes[2, 0].legend(fontsize=9)

    # Row 2 right: per-voxel ratio
    ratio_1_8 = diff_1_8.mean() / (np.abs(cz_1_8).mean() + 1e-8)
    ratio_1_4 = diff_1_4.mean() / (np.abs(cz_1_4).mean() + 1e-8)
    summary_text = (
        f"DIAGNOSTIC SUMMARY (random init)\n\n"
        f"=== 1/8 encoder (fm={cz_1_8.shape}) ===\n"
        f"  corr_at_zero  mean: {cz_1_8.mean():+.4f}   std: {cz_1_8.std():.4f}\n"
        f"  corr_at_truth mean: {ct_1_8.mean():+.4f}   std: {ct_1_8.std():.4f}\n"
        f"  Δ mean:           {diff_1_8.mean():+.4f}   std: {diff_1_8.std():.4f}\n"
        f"  Δ / |corr_at_zero|: {ratio_1_8:+.3f}\n\n"
        f"=== 1/4 encoder (fm={cz_1_4.shape}) ===\n"
        f"  corr_at_zero  mean: {cz_1_4.mean():+.4f}   std: {cz_1_4.std():.4f}\n"
        f"  corr_at_truth mean: {ct_1_4.mean():+.4f}   std: {ct_1_4.std():.4f}\n"
        f"  Δ mean:           {diff_1_4.mean():+.4f}   std: {diff_1_4.std():.4f}\n"
        f"  Δ / |corr_at_zero|: {ratio_1_4:+.3f}\n\n"
        f"VERDICT:\n"
        f"  If |Δ mean| << |corr_at_zero| mean for 1/4 but not 1/8,\n"
        f"  the 1/4 encoder produces features too similar across nearby\n"
        f"  positions -- there's NO gradient signal to escape zero-flow.\n\n"
        f"  Healthy case: Δ mean > 0.1 * |corr_at_zero|\n"
        f"  Dead case:   |Δ mean| < 0.05 * |corr_at_zero|"
    )
    axes[2, 1].text(
        0.02, 0.98, summary_text,
        transform=axes[2, 1].transAxes, fontsize=9,
        verticalalignment="top", family="monospace",
    )
    axes[2, 1].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--encoder-1-8", type=Path,
                        default=PROJECT_ROOT / "configs/models/raft_dvc_1_8_p4_r4.yaml")
    parser.add_argument("--encoder-1-4", type=Path,
                        default=PROJECT_ROOT / "configs/models/raft_dvc_1_4_p3_r4.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for encoder random init (same for both for fair compare)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device.")
        sys.exit(1)

    print(f"Building encoders (seed={args.seed})...")
    model_1_8 = build_random_encoder(args.encoder_1_8, seed=args.seed)
    model_1_4 = build_random_encoder(args.encoder_1_4, seed=args.seed)
    n1_8 = sum(p.numel() for p in model_1_8.fnet.parameters())
    n1_4 = sum(p.numel() for p in model_1_4.fnet.parameters())
    print(f"  1/8 fnet params: {n1_8:,}")
    print(f"  1/4 fnet params: {n1_4:,}")

    with PdfPages(args.output) as pdf:
        for idx in args.indices:
            npz_path = args.data_dir / f"sample_{idx:05d}.npz"
            if not npz_path.exists():
                print(f"[skip] {npz_path} not found")
                continue
            sample = load_sample(npz_path)
            print(f"[{idx}] {npz_path.name}: flow max={np.abs(sample['flow']).max():.2f}")

            # To GPU
            I1_t = torch.from_numpy(sample["I1"]).unsqueeze(0).unsqueeze(0).cuda()
            I2_t = torch.from_numpy(sample["I2"]).unsqueeze(0).unsqueeze(0).cuda()

            # Run encoders
            fmap1_1_8 = get_fmap(model_1_8, I1_t)
            fmap2_1_8 = get_fmap(model_1_8, I2_t)
            fmap1_1_4 = get_fmap(model_1_4, I1_t)
            fmap2_1_4 = get_fmap(model_1_4, I2_t)

            # Page 1: input + flow visualization
            render_input(pdf, sample, idx)
            # Page 2: fmap statistics (side-by-side)
            render_fmap_stats(pdf, fmap1_1_8, fmap1_1_4, idx)
            # Pages 3-4: fmap per-channel visualization
            render_fmap_channels(pdf, fmap1_1_8, "1/8", idx)
            render_fmap_channels(pdf, fmap1_1_4, "1/4", idx)
            # Page 5: KEY -- gradient signal
            render_gradient_signal(
                pdf, fmap1_1_8, fmap2_1_8, fmap1_1_4, fmap2_1_4,
                sample["flow"], idx,
            )

    print(f"\nWrote {args.output}")
    print("\nKey page: the LAST page ('Gradient signal') -- compare 1/8 vs 1/4 Δ mean.")
    print("If |Δ mean (1/4)| << |Δ mean (1/8)| relative to corr_at_zero scale,")
    print("the 1/4 encoder at random init has no learnable signal -> need init / architecture change.")


if __name__ == "__main__":
    main()
