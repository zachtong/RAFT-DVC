"""Phase-1 cross-config (OOD) evaluation wrapper.

Loads a single checkpoint and evaluates it on N data configs (typically the
9 single + 1 mixed configs at one input size), producing one consolidated
report instead of N separate ones.

Outputs::

    <output-dir>/ood_summary.csv          per-config summary stats (one row each)
    <output-dir>/ood_per_sample.csv       all per-sample EPE across configs
    <reports-dir>/phase1_<exp>_ood_<date>.pdf  consolidated PDF:
        Page 1  -- bar chart of EPE per config (in-distribution highlighted)
        Page 2  -- overlaid EPE histograms (small multiples)
        Page 3  -- 3x3 radius x density heatmap (single-configs)
        Page 4+ -- per-config viz (2 samples each: best + worst by EPE)

Typical usage::

    python scripts/phase1/evaluate_ood.py \\
        --checkpoint checkpoints/phase1/<run>/best_model.pth \\
        --train-config r4_medium_size32 \\
        --eval-configs r4_medium_size32,r4_sparse_size32,r4_dense_size32,\\
                       r2_medium_size32,r6_medium_size32,\\
                       r2_sparse_size32,r2_dense_size32,\\
                       r6_sparse_size32,r6_dense_size32,mixed_size32 \\
        --experiment-name conv_b8_800ep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


# -- Repo + script path setup -------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse the proven inference helpers from evaluate_phase1.
from evaluate_phase1 import (  # noqa: E402
    _aggregate,
    _evaluate_loader,
    _load_model,
    _resolve_data_root,
    _resolve_device,
)

from src.data import parse_data_config_name, safe_phase1_collate  # noqa: E402
from src.data.dataset import Phase1NPZDataset  # noqa: E402
from src.visualization import render_flow_eval_4x3  # noqa: E402


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-config (OOD) evaluation: one checkpoint -> N configs.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-config", type=str, required=True,
        help="Data-config that the checkpoint was trained on (highlighted "
             "as in-distribution in the report).",
    )
    parser.add_argument(
        "--eval-configs", type=str, required=True,
        help="Comma-separated list of data-configs to evaluate (include the "
             "train-config to keep an in-distribution baseline in the chart).",
    )
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=None, dest="data_root")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write the OOD CSVs.  Default: "
             "reports/phase1/<experiment>_ood/",
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=_REPO_ROOT / "reports",
        help="Root dir for the consolidated PDF.",
    )
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=None,
                        help="GRU iterations (default: from ckpt).")
    parser.add_argument(
        "--num-visual", type=int, default=2,
        help="Number of samples to visualise per config (default: 2 -- "
             "easiest + hardest).",
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Per-config evaluation
# -----------------------------------------------------------------------------

def _evaluate_one_config(
    model,
    data_root: Path,
    data_config: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    iters: Optional[int],
) -> Tuple[List[dict], list, dict]:
    """Run inference on one config; return (per-sample rows, viz cache,
    aggregate summary)."""
    split_root = data_root / data_config / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split dir missing: {split_root}")
    ds = Phase1NPZDataset(
        root_dir=str(split_root),
        augment=False,
        patch_size=None,
        include_metadata=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_phase1_collate,
    )
    rows, cache = _evaluate_loader(model, loader, device, iters)
    summary = _aggregate(rows)
    summary["data_config"] = data_config
    summary["split"] = split
    return rows, cache, summary


# -----------------------------------------------------------------------------
# Output: CSVs
# -----------------------------------------------------------------------------

def _write_csvs(
    summaries: List[dict],
    all_rows: List[Tuple[str, dict]],
    out_dir: Path,
    train_config: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-config summary
    fields = [
        "data_config", "is_train_config", "n_samples",
        "epe_mean", "epe_std", "epe_median",
        "epe_p90", "epe_p99", "epe_min", "epe_max",
    ]
    with open(out_dir / "ood_summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k) for k in fields}
            row["is_train_config"] = (s["data_config"] == train_config)
            writer.writerow(row)

    # Per-sample (all configs concatenated)
    if all_rows:
        # Take keys from first row + add 'data_config'
        sample_fields = list(all_rows[0][1].keys()) + ["data_config"]
        with open(out_dir / "ood_per_sample.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sample_fields)
            writer.writeheader()
            for cfg, sample_row in all_rows:
                out_row = dict(sample_row)
                out_row["data_config"] = cfg
                writer.writerow(out_row)


# -----------------------------------------------------------------------------
# PDF report
# -----------------------------------------------------------------------------

_RADII = (2, 4, 6)
_DENSITIES = ("sparse", "medium", "dense")


def _parse_or_none(name: str) -> Optional[Dict[str, Any]]:
    try:
        return parse_data_config_name(name)
    except ValueError:
        return None


def _write_pdf_report(
    summaries: List[dict],
    rows_by_cfg: Dict[str, List[dict]],
    cache_by_cfg: Dict[str, list],
    pdf_path: Path,
    title: str,
    train_config: str,
    num_visual: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(pdf_path)) as pdf:

        # ---------------------------------------------------------------------
        # Page 1: title + bar chart (EPE per config, in-dist highlighted)
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(11.5, 8.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.18, 1.0])

        # Title block
        ax_title = fig.add_subplot(gs[0, 0])
        ax_title.axis("off")
        ax_title.text(0.5, 0.6, title, ha="center", va="center",
                      fontsize=14, fontweight="bold")
        ax_title.text(0.5, 0.15,
                      f"Train config (in-distribution): {train_config}",
                      ha="center", va="center", fontsize=10, color="darkgreen")

        # Bar chart
        ax_bar = fig.add_subplot(gs[1, 0])
        cfgs = [s["data_config"] for s in summaries]
        means = np.array([s["epe_mean"] for s in summaries])
        stds = np.array([s["epe_std"] for s in summaries])
        p90 = np.array([s["epe_p90"] for s in summaries])
        x = np.arange(len(cfgs))
        colors = [
            "darkgreen" if c == train_config else "steelblue" for c in cfgs
        ]
        bars = ax_bar.bar(x, means, yerr=stds, color=colors,
                           edgecolor="black", capsize=4, alpha=0.85,
                           label="mean +/- std")
        ax_bar.scatter(x, p90, marker="x", color="darkorange", s=50, zorder=3,
                        label="p90")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(cfgs, rotation=35, ha="right", fontsize=9)
        ax_bar.set_ylabel("EPE (voxels)")
        ax_bar.set_title("EPE per data-config (in-distribution = green)")
        ax_bar.legend(loc="upper left")
        ax_bar.grid(axis="y", linestyle=":", alpha=0.5)
        for i, m in enumerate(means):
            ax_bar.text(i, m + (stds[i] if not np.isnan(stds[i]) else 0) + 0.005,
                         f"{m:.3f}", ha="center", fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------------------
        # Page 2: small-multiple histograms of EPE per config
        # ---------------------------------------------------------------------
        n_cfg = len(cfgs)
        n_cols = 3
        n_rows = int(np.ceil(n_cfg / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(11.5, 3.0 * n_rows),
                                  sharex=True)
        axes = np.atleast_2d(axes)
        global_max = max(
            max((r["epe"] for r in rows_by_cfg.get(c, [])), default=1.0)
            for c in cfgs
        )
        for i, c in enumerate(cfgs):
            ax = axes[i // n_cols, i % n_cols]
            data = np.array([r["epe"] for r in rows_by_cfg.get(c, [])])
            if data.size == 0:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                         transform=ax.transAxes)
            else:
                colour = "darkgreen" if c == train_config else "steelblue"
                ax.hist(data, bins=30, range=(0, global_max * 1.05),
                         color=colour, edgecolor="black", alpha=0.85)
                ax.axvline(data.mean(), color="red", linestyle="--",
                            linewidth=1, label=f"mean={data.mean():.3f}")
                ax.legend(fontsize=7, loc="upper right")
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(n_cfg, n_rows * n_cols):
            axes[j // n_cols, j % n_cols].set_visible(False)
        fig.suptitle("EPE distribution per config", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------------------
        # Page 3: 3x3 radius x density heatmap (single configs only)
        # ---------------------------------------------------------------------
        heatmap = np.full((3, 3), np.nan, dtype=np.float64)
        for s in summaries:
            parsed = _parse_or_none(s["data_config"])
            if parsed is None or parsed["mixed"]:
                continue
            try:
                ri = _RADII.index(parsed["radius"])
                di = _DENSITIES.index(parsed["density"])
            except ValueError:
                continue
            heatmap[ri, di] = s["epe_mean"]

        fig, ax = plt.subplots(figsize=(8, 7))
        valid_vals = heatmap[~np.isnan(heatmap)]
        if valid_vals.size == 0:
            ax.text(0.5, 0.5, "(no single-config data)",
                     ha="center", va="center", transform=ax.transAxes,
                     fontsize=12)
            ax.axis("off")
        else:
            vmin = float(valid_vals.min())
            vmax = float(valid_vals.max())
            im = ax.imshow(heatmap, cmap="YlOrRd", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(3))
            ax.set_xticklabels(_DENSITIES, fontsize=10)
            ax.set_yticks(range(3))
            ax.set_yticklabels([f"R{r}" for r in _RADII], fontsize=10)
            ax.set_xlabel("density")
            ax.set_ylabel("bead radius")
            for ri in range(3):
                for di in range(3):
                    v = heatmap[ri, di]
                    if np.isnan(v):
                        text = "(n/a)"
                    else:
                        text = f"{v:.3f}"
                    is_train = (
                        f"r{_RADII[ri]}_{_DENSITIES[di]}_size" in train_config
                    )
                    color = "white" if (
                        not np.isnan(v) and v > 0.6 * vmax + 0.4 * vmin
                    ) else "black"
                    weight = "bold" if is_train else "normal"
                    ax.text(di, ri, text + ("\n(train)" if is_train else ""),
                             ha="center", va="center",
                             color=color, fontsize=11, fontweight=weight)
            plt.colorbar(im, ax=ax, label="mean EPE (voxels)",
                          fraction=0.046, pad=0.04)
        ax.set_title("Generalisation matrix: mean EPE on each single config",
                      fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # ---------------------------------------------------------------------
        # Pages 4+: per-config viz (easiest + hardest by EPE)
        # ---------------------------------------------------------------------
        for c in cfgs:
            rows = rows_by_cfg.get(c, [])
            cache = cache_by_cfg.get(c, [])
            if not rows or not cache:
                continue
            epes = np.array([r["epe"] for r in rows])
            order = np.argsort(epes)
            if num_visual <= 0:
                continue
            picks_idx = []
            picks_idx.append(int(order[0]))                 # easiest
            if num_visual >= 2:
                picks_idx.append(int(order[-1]))            # hardest
            for k in range(1, num_visual - 1):
                slot = int(round(k * (len(order) - 1) / max(num_visual - 1, 1)))
                picks_idx.append(int(order[slot]))

            for pi in picks_idx:
                if pi >= len(cache):
                    continue
                vol0, vol1, gt, pred, name = cache[pi]
                fig, _ = render_flow_eval_4x3(vol0, vol1, gt, pred)
                is_train = (c == train_config)
                tag = " [IN-DIST]" if is_train else " [OOD]"
                fig.suptitle(
                    f"{c}{tag}  |  {name}  |  "
                    f"EPE={rows[pi]['epe']:.3f}  "
                    f"fm={rows[pi]['fm_target']:.2f}  "
                    f"deform={rows[pi]['deform_type']}",
                    fontsize=11,
                    color=("darkgreen" if is_train else "black"),
                )
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(fig); plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    eval_configs = [c.strip() for c in args.eval_configs.split(",") if c.strip()]
    if not eval_configs:
        raise SystemExit("--eval-configs is empty.")

    device = _resolve_device(args.device)
    data_root = _resolve_data_root(args.data_root)

    print(f"[ood] checkpoint    : {args.checkpoint}")
    print(f"[ood] train_config  : {args.train_config}")
    print(f"[ood] eval_configs  : {len(eval_configs)}")
    for c in eval_configs:
        print(f"                      - {c}")
    print(f"[ood] device        : {device}")

    model, _ = _load_model(args.checkpoint, device)
    print(f"[ood] model params  : {model.get_num_parameters():,}")

    summaries: List[dict] = []
    rows_by_cfg: Dict[str, List[dict]] = {}
    cache_by_cfg: Dict[str, list] = {}
    all_rows: List[Tuple[str, dict]] = []

    for cfg in eval_configs:
        print(f"\n[ood] evaluating: {cfg}")
        try:
            rows, cache, summary = _evaluate_one_config(
                model=model,
                data_root=data_root,
                data_config=cfg,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                iters=args.iters,
            )
        except FileNotFoundError as exc:
            print(f"    skip ({exc})")
            continue
        rows_by_cfg[cfg] = rows
        cache_by_cfg[cfg] = cache
        summaries.append(summary)
        for r in rows:
            all_rows.append((cfg, r))
        print(
            f"    n={summary['n_samples']:3d}  "
            f"mean={summary['epe_mean']:.4f}  "
            f"median={summary['epe_median']:.4f}  "
            f"p90={summary['epe_p90']:.4f}"
        )

    if not summaries:
        raise SystemExit("[ood] no configs evaluated; aborting.")

    # Output paths
    out_dir = args.output_dir or (
        _REPO_ROOT / "reports" / "phase1" / f"{args.experiment_name}_ood"
    )
    _write_csvs(summaries, all_rows, out_dir, args.train_config)
    print(f"\n[ood] CSVs           : {out_dir}/ood_summary.csv")
    print(f"[ood]                  {out_dir}/ood_per_sample.csv")

    pdf_name = (
        f"phase1_{args.experiment_name}_ood_{date.today().isoformat()}.pdf"
    )
    pdf_path = args.reports_dir / pdf_name
    _write_pdf_report(
        summaries=summaries,
        rows_by_cfg=rows_by_cfg,
        cache_by_cfg=cache_by_cfg,
        pdf_path=pdf_path,
        title=(
            f"Phase-1 OOD eval  |  {args.experiment_name}  |  "
            f"train={args.train_config}"
        ),
        train_config=args.train_config,
        num_visual=args.num_visual,
    )
    print(f"[ood] PDF report     : {pdf_path}")

    # ---- Compact headline table -----------------------------------------
    in_dist = next(
        (s for s in summaries if s["data_config"] == args.train_config), None,
    )
    if in_dist is not None:
        in_epe = in_dist["epe_mean"]
        ood_means = [
            s["epe_mean"] for s in summaries
            if s["data_config"] != args.train_config
        ]
        if ood_means:
            ood_mean = float(np.mean(ood_means))
            ood_max = float(np.max(ood_means))
            gap = ood_mean - in_epe
            print("\n" + "-" * 60)
            print(f"[ood] in-distribution EPE  = {in_epe:.4f}")
            print(f"[ood] avg OOD     EPE      = {ood_mean:.4f}")
            print(f"[ood] worst OOD   EPE      = {ood_max:.4f}")
            print(f"[ood] generalisation gap   = {gap:+.4f} "
                  f"(ratio {ood_mean/in_epe:.2f}x)")
            print("-" * 60)


if __name__ == "__main__":
    main()
