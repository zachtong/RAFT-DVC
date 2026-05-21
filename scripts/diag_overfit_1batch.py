"""Single-batch overfit smoke test for RAFT-DVC.

A healthy model architecture + loss + optimizer must be able to overfit
a single mini-batch to near-zero loss within ~200 iterations.  If loss
stays flat, the gradient signal is broken somewhere in the pipeline.

Usage:
    python scripts/diag_overfit_1batch.py \
        --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
        --data-root data_paper1_v2 \
        --data-config r4_medium_size64 \
        --iters 200 \
        --lr 1e-3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.core import RAFTDVC, RAFTDVCConfig  # noqa: E402
from src.training.loss import SequenceLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-config", type=str, required=True,
                        help="Subdirectory under data-root (e.g. r4_medium_size64)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters-update", type=int, default=12,
                        help="GRU update iterations inside RAFT")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_batch(data_dir: Path, count: int, start_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load `count` consecutive samples and stack into a mini-batch."""
    files = sorted(data_dir.glob("sample_*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ samples in {data_dir}")
    files = files[start_idx:start_idx + count]
    I1, I2, flow = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        I1.append(d["I1"])
        I2.append(d["I2"])
        flow.append(d["flow"])
    I1 = torch.from_numpy(np.stack(I1)).float()       # (B, D, H, W)
    I2 = torch.from_numpy(np.stack(I2)).float()
    flow = torch.from_numpy(np.stack(flow)).float()   # (B, 3, D, H, W)
    # Add channel dim to volumes for the model
    I1 = I1.unsqueeze(1)  # (B, 1, D, H, W)
    I2 = I2.unsqueeze(1)
    return I1, I2, flow


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Build model
    with open(args.model_config) as fh:
        model_yaml = yaml.safe_load(fh)
    arch = model_yaml.get("architecture", {})
    config = RAFTDVCConfig.from_dict(arch)
    model = RAFTDVC(config).to(device)
    model.train()

    print(f"[overfit] model: {args.model_config.name}")
    print(f"[overfit] params: {sum(p.numel() for p in model.parameters()):,}")

    # Load one fixed batch
    data_dir = args.data_root / args.data_config / args.split
    I1, I2, gt_flow = load_batch(data_dir, args.batch_size)
    I1 = I1.to(device)
    I2 = I2.to(device)
    gt_flow = gt_flow.to(device)
    print(f"[overfit] batch: I1={tuple(I1.shape)} flow={tuple(gt_flow.shape)}")
    print(f"[overfit] I1 stats: mean={I1.mean():.4f} std={I1.std():.4f}")
    print(f"[overfit] flow stats: |mean|={gt_flow.mean():.4f} max={gt_flow.abs().max():.4f}")

    # Optimizer + loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = SequenceLoss(gamma=0.8)

    print(f"\n[overfit] starting {args.iters} iters at lr={args.lr}")
    print(f"{'iter':>5} | {'loss':>10} | {'epe':>8} | {'gnorm':>10}")
    print("-" * 50)

    for it in range(args.iters):
        optimizer.zero_grad()
        flow_preds = model(I1, I2, iters=args.iters_update)
        loss = criterion(flow_preds, gt_flow)
        loss.backward()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
        optimizer.step()

        with torch.no_grad():
            final_pred = flow_preds[-1] if isinstance(flow_preds, list) else flow_preds
            epe = (final_pred - gt_flow).pow(2).sum(dim=1).sqrt().mean()

        if it < 10 or it % 10 == 0 or it == args.iters - 1:
            print(f"{it:>5d} | {loss.item():>10.4f} | {epe.item():>8.4f} | {gnorm.item():>10.4f}")

    print(f"\n[overfit] DONE. Final loss={loss.item():.4f} EPE={epe.item():.4f}")
    print("[overfit] Interpretation:")
    print("  - loss → < 0.5 within 200 iters     : gradient signal HEALTHY")
    print("  - loss stays > 10 the entire run    : gradient signal DEAD (broken pipeline)")
    print("  - loss decreases then plateaus high : optimizer or loss formulation issue")


if __name__ == "__main__":
    main()
