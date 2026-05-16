"""Phase-1 evaluation entry point.

Loads a trained checkpoint, evaluates on the `test/` split of a Phase-1
dataset configuration, and writes:

* ``<output-dir>/metrics.csv``       -- per-sample EPE + metadata
* ``<output-dir>/summary.json``      -- aggregate statistics
* ``reports/phase1_<experiment>_<data>_<YYYY-MM-DD>.pdf`` -- PDF report
  (field maps + error distribution + summary table)

CLI::

    python scripts/phase1/evaluate_phase1.py \
        --checkpoint  checkpoints/phase1/<run>/best_model.pth \
        --data-config r2_sparse_size32 \
        --experiment-name phase1_r2sparse32_enc1_1 \
        --num-visual 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


# -- Repo root on path --------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core import RAFTDVC, RAFTDVCConfig  # noqa: E402
from src.data import safe_phase1_collate  # noqa: E402
from src.data.dataset import Phase1NPZDataset  # noqa: E402
from src.visualization import render_flow_eval_4x3  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RAFT-DVC model on the Phase-1 test split.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to .pth checkpoint.")
    parser.add_argument("--data-config", type=str, required=True,
                        help="Phase-1 data configuration name, e.g. r2_sparse_size32.")
    parser.add_argument("--experiment-name", type=str, required=True,
                        help="Tag for output filenames.")
    parser.add_argument("--data-root", type=Path, default=None, dest="data_root",
                        help="Root Phase-1 data directory. "
                             "Default: $DATA_PHASE1_ROOT or <repo>/data_phase1.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write metrics + report. "
                             "Default: reports/phase1/<experiment>_<data>/")
    parser.add_argument("--reports-dir", type=Path,
                        default=_REPO_ROOT / "reports",
                        help="Root reports directory for the PDF (default: reports/).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=None,
                        help="Override GRU iteration count (default: use model's config).")
    parser.add_argument("--num-visual", type=int, default=6,
                        help="Number of samples to visualize in the PDF.")
    parser.add_argument("--device", type=str, default=None,
                        help="'cuda' / 'cpu'. Auto-detected when unset.")
    return parser.parse_args()


# -- Helpers ------------------------------------------------------------------
def _resolve_data_root(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.environ.get("DATA_PHASE1_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return (_REPO_ROOT / "data_phase1").resolve()


def _resolve_device(cli_value: str | None) -> torch.device:
    if cli_value:
        return torch.device(cli_value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[RAFTDVC, dict]:
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    # The checkpoint may or may not carry the full architecture config.
    arch = ckpt.get("model_config") or ckpt.get("config") or {}
    if not arch:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not embed a model config; "
            f"cannot reconstruct the architecture."
        )
    cfg = RAFTDVCConfig.from_dict(arch)
    model = RAFTDVC(cfg).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def _evaluate_loader(
    model: RAFTDVC,
    loader: DataLoader,
    device: torch.device,
    iters: int | None,
) -> tuple[list[dict], list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]]:
    """Returns (per-sample metrics, cached raw arrays for visualization)."""
    rows: list[dict] = []
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    iter_count = iters if iters is not None else model.config.iters

    for batch in loader:
        vol0 = batch["vol0"].to(device, non_blocking=True)
        vol1 = batch["vol1"].to(device, non_blocking=True)
        gt = batch["flow"].to(device, non_blocking=True)

        preds = model(vol0, vol1, iters=iter_count)
        pred = preds[-1]

        # EPE per voxel, then mean over voxels -> (B,) EPE per sample
        err = torch.sqrt(torch.sum((pred - gt) ** 2, dim=1))  # (B, D, H, W)
        epe_per_sample = err.mean(dim=(1, 2, 3)).cpu().numpy()
        max_err_per_sample = err.amax(dim=(1, 2, 3)).cpu().numpy()
        gt_mag = torch.sqrt(torch.sum(gt ** 2, dim=1))
        gt_mag_mean = gt_mag.mean(dim=(1, 2, 3)).cpu().numpy()

        for i in range(vol0.shape[0]):
            meta = batch.get("meta", [{}] * vol0.shape[0])
            meta_i = meta[i] if isinstance(meta, list) else {}
            rows.append({
                "filename": batch["filename"][i],
                "epe": float(epe_per_sample[i]),
                "max_err": float(max_err_per_sample[i]),
                "gt_flow_mag_mean": float(gt_mag_mean[i]),
                "fm_target": float(meta_i.get("fm_target", 0.0)) if meta_i else 0.0,
                "deform_type": str(meta_i.get("deform_type", "?")) if meta_i else "?",
                "radius": int(meta_i.get("radius", 0)) if meta_i else 0,
                "density_per_1000": float(meta_i.get("density_per_1000", 0.0)) if meta_i else 0.0,
            })
            if len(cache) < 64:  # keep up to 64 raw samples for plotting; cheap
                cache.append((
                    vol0[i].cpu().numpy().squeeze(0),  # (D, H, W)
                    vol1[i].cpu().numpy().squeeze(0),
                    gt[i].cpu().numpy(),               # (3, D, H, W)
                    pred[i].cpu().numpy(),
                    batch["filename"][i],
                ))
    return rows, cache


def _aggregate(rows: list[dict]) -> dict:
    epes = np.array([r["epe"] for r in rows], dtype=np.float64)
    return {
        "n_samples": int(len(rows)),
        "epe_mean": float(epes.mean()) if epes.size else float("nan"),
        "epe_std":  float(epes.std())  if epes.size else float("nan"),
        "epe_median": float(np.median(epes)) if epes.size else float("nan"),
        "epe_p90":  float(np.percentile(epes, 90)) if epes.size else float("nan"),
        "epe_p99":  float(np.percentile(epes, 99)) if epes.size else float("nan"),
        "epe_min":  float(epes.min()) if epes.size else float("nan"),
        "epe_max":  float(epes.max()) if epes.size else float("nan"),
    }


def _write_metrics(rows: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    if rows:
        with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def _write_pdf_report(
    summary: dict,
    rows: list[dict],
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]],
    pdf_path: Path,
    title: str,
    num_visual: int,
) -> None:
    # Headless-safe matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(pdf_path)) as pdf:
        # ---- Page 1: title + summary table -----------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.94, title, ha="center", va="center",
                fontsize=16, fontweight="bold", transform=ax.transAxes)
        y = 0.86
        for key in ["n_samples", "epe_mean", "epe_std", "epe_median",
                    "epe_p90", "epe_p99", "epe_min", "epe_max"]:
            val = summary[key]
            txt = f"{key:>14s} :  {val:.4f}" if isinstance(val, float) else f"{key:>14s} :  {val}"
            ax.text(0.08, y, txt, fontsize=11, family="monospace",
                    transform=ax.transAxes)
            y -= 0.035
        pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: EPE histogram + EPE vs fm_target ------------------------
        if rows:
            epes = np.array([r["epe"] for r in rows])
            fm = np.array([r["fm_target"] for r in rows])
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].hist(epes, bins=40, color="steelblue", edgecolor="black")
            axes[0].set_xlabel("EPE (voxels)")
            axes[0].set_ylabel("count")
            axes[0].set_title("EPE distribution (test set)")
            axes[1].scatter(fm, epes, s=12, alpha=0.6, color="darkorange")
            axes[1].set_xlabel("fm_target (voxels)")
            axes[1].set_ylabel("EPE (voxels)")
            axes[1].set_title("EPE vs displacement magnitude")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- Pages 3+: per-sample slice visualizations -----------------------
        # 4x3 layout (I1/I2/|EPE| + GT/pred/error per component) via the
        # shared ``src.visualization.render_flow_eval_4x3`` helper.
        n_vis = min(num_visual, len(cache))
        if n_vis > 0:
            epes = np.array([r["epe"] for r in rows])
            order = np.argsort(epes)
            picks = np.linspace(0, len(order) - 1, num=n_vis, dtype=int)
            chosen = [int(order[i]) for i in picks if int(order[i]) < len(cache)]
            seen = set()
            for idx in chosen:
                if idx in seen:
                    continue
                seen.add(idx)
                vol0, vol1, gt, pred, name = cache[idx]
                fig, _ = render_flow_eval_4x3(vol0, vol1, gt, pred)
                fig.suptitle(
                    f"{name}  --  EPE={rows[idx]['epe']:.3f}  "
                    f"fm_target={rows[idx]['fm_target']:.2f}  "
                    f"deform={rows[idx]['deform_type']}",
                    fontsize=11,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                pdf.savefig(fig); plt.close(fig)


# -- Main ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    device = _resolve_device(args.device)
    data_phase1_root = _resolve_data_root(args.data_root)
    data_root = data_phase1_root / args.data_config / args.split
    if not data_root.exists():
        raise FileNotFoundError(f"Split directory not found: {data_root}")

    print(f"[eval] checkpoint      : {args.checkpoint}")
    print(f"[eval] data            : {data_root}")
    print(f"[eval] device          : {device}")

    # Model
    model, _ckpt = _load_model(args.checkpoint, device)
    print(f"[eval] model params    : {model.get_num_parameters():,}")
    print(f"[eval] encoder_type    : {model.config.encoder_type}")

    # Data
    ds = Phase1NPZDataset(
        root_dir=str(data_root),
        augment=False,
        patch_size=None,
        include_metadata=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=safe_phase1_collate,
    )
    print(f"[eval] samples         : {len(ds)}")

    # Evaluate
    rows, cache = _evaluate_loader(model, loader, device, args.iters)
    summary = _aggregate(rows)
    summary["experiment_name"] = args.experiment_name
    summary["data_config"] = args.data_config
    summary["split"] = args.split
    summary["checkpoint"] = str(args.checkpoint)

    # Output metrics
    out_dir = args.output_dir or (
        _REPO_ROOT / "reports" / "phase1" / f"{args.experiment_name}_{args.data_config}"
    )
    _write_metrics(rows, summary, out_dir)
    print(f"[eval] metrics written : {out_dir}/metrics.csv")
    print(f"[eval] summary         : {out_dir}/summary.json")

    # PDF report (CLAUDE.md testing policy)
    pdf_name = f"phase1_{args.experiment_name}_{args.data_config}_{date.today().isoformat()}.pdf"
    pdf_path = args.reports_dir / pdf_name
    _write_pdf_report(
        summary=summary,
        rows=rows,
        cache=cache,
        pdf_path=pdf_path,
        title=f"Phase-1 eval  |  {args.experiment_name}  |  {args.data_config}",
        num_visual=args.num_visual,
    )
    print(f"[eval] PDF report      : {pdf_path}")

    # Print headline numbers
    print("-" * 60)
    print(f"[eval]  EPE mean  = {summary['epe_mean']:.4f}")
    print(f"[eval]  EPE median= {summary['epe_median']:.4f}")
    print(f"[eval]  EPE p90   = {summary['epe_p90']:.4f}")
    print(f"[eval]  EPE p99   = {summary['epe_p99']:.4f}")
    print(f"[eval]  EPE max   = {summary['epe_max']:.4f}")


if __name__ == "__main__":
    main()
