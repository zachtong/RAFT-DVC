"""Train "VolRAFT architecture, sampler FIXED" on OUR Phase-1/paper-1 data.

Architecture: the released VolRAFT variant (small=True encoder: fnet out 128,
hidden 96, context 64; corr_levels 4, corr_radius 3, iters 12, internal
per-patch-pair min-max normalization, loss max_flow clamp 24) with the
``bilinear_sampler`` axis-order bug FIXED (see volraft_fixed/model/utils/utils.py).

NOTE on architecture spec: the released checkpoint (volraft_config_120) was
verified to use the SMALL encoder (fnet output_dim=128), because upstream
``NetworkVolRAFT`` hardcodes ``args.small = True``.  We retrain the actual
released architecture, not the 256-feature "basic" variant.

Training recipe mirrors scripts/phase1/train_phase1.py + configs/phase1/_base.yaml:
  * 300 epochs x 250 optimizer steps (2000 train samples / batch 8, drop_last)
  * AdamW  lr = max_lr = 2e-4, weight_decay 5e-5
  * OneCycleLR, pct_start 0.2, cosine anneal, div_factor 25, final_div 1e4
  * fp16 autocast + GradScaler, grad clip 1.0
  * RAFT sequence loss, gamma 0.8 (our SequenceLoss; upstream's max_flow=24
    GT-magnitude exclusion is a no-op on this dataset -- max |u| ~ 8 voxels)
  * batch auto-halves on CUDA OOM during a pre-flight probe (start 8)

Data: G:/Zixiang_local_data/raft-dvc/data_paper1_v3/r4_medium_size64
      (train 2000 / val 200, 64^3, keys I1/I2/flow) via src.data.Phase1NPZDataset.
Output: C:/Zixiang_local_data/raft-dvc/paper1/phase1/volraft_fixed_r4md64/
      (training.log, latest.pth, best_model.pth, checkpoint_epoch_*.pth --
      written by src.training.Trainer).

Launch instructions: volraft_fixed/LAUNCH.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# -- Make repo root importable (this file lives at <repo>/volraft_fixed/) ----
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import safe_phase1_collate            # noqa: E402
from src.data.dataset import Phase1NPZDataset       # noqa: E402
from src.training import Trainer                    # noqa: E402

from volraft_fixed.model.volraft import ModuleVolRAFT  # noqa: E402

DEFAULT_DATA_DIR = Path(r"G:\Zixiang_local_data\raft-dvc\data_paper1_v3\r4_medium_size64")
DEFAULT_OUTPUT_DIR = Path(r"C:\Zixiang_local_data\raft-dvc\paper1\phase1\volraft_fixed_r4md64")


# ---------------------------------------------------------------------------
# Model wrapper: adapts ModuleVolRAFT to the Trainer's RAFTDVC-shaped API
# ---------------------------------------------------------------------------
class VolRAFTFixed(nn.Module):
    """ModuleVolRAFT (fixed sampler) with the interface our Trainer expects.

    Trainer calls: model(vol0, vol1, iters=N) -> list of (B, 3, D, H, W);
    model.get_num_parameters(); model.save_checkpoint(path, ...).
    """

    ARCH = {
        "network_name": "volraft_fixed",
        "small": True,           # matches the released checkpoint architecture
        "corr_levels": 4,
        "corr_radius": 3,
        "iters": 12,
        "proj_max_flow": 24.0,   # upstream loss clamp (no-op on our data)
        "fnet_output_dim": 128,
        "hidden_dim": 96,
        "context_dim": 64,
        "sampler": "fixed (see volraft_fixed/model/utils/utils.py)",
    }

    def __init__(self, mixed_precision: bool = True):
        super().__init__()
        args = argparse.Namespace(
            corr_levels=self.ARCH["corr_levels"],
            corr_radius=self.ARCH["corr_radius"],
            num_channels=1,
            small=True,
            dropout=0.0,
            iters=self.ARCH["iters"],
            # Inner autocast around fnet/cnet/update (upstream behavior);
            # Trainer's outer torch.amp.autocast + GradScaler complete AMP.
            mixed_precision=mixed_precision,
            alternate_corr=False,
            should_normalize=True,   # upstream per-patch-pair joint min-max
            flow_shape=None,         # set per forward (upflow8 target shape)
        )
        self.args = args
        self.module = ModuleVolRAFT(args)

    def forward(self, vol0: torch.Tensor, vol1: torch.Tensor, iters: int = 12):
        # upflow8 upsamples each iteration's flow to this full-res shape
        self.args.flow_shape = (vol0.shape[0], 3) + tuple(vol0.shape[2:])
        return self.module(vol0, vol1, iters=iters)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path, optimizer=None, scheduler=None,
                        epoch=None, **kwargs):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": dict(self.ARCH),
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, str(path))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    import random as _py_random
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    _py_random.seed(worker_seed)


def _build_dataloaders(data_dir: Path, batch_size: int, num_workers: int,
                       generator: torch.Generator):
    train_ds = Phase1NPZDataset(root_dir=str(data_dir / "train"),
                                augment=True, include_metadata=False)
    val_ds = Phase1NPZDataset(root_dir=str(data_dir / "val"),
                              augment=False, include_metadata=False)
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, drop_last=True,
        collate_fn=safe_phase1_collate, worker_init_fn=_seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent, drop_last=False,
        collate_fn=safe_phase1_collate, worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader


def _probe_batch_size(start_bs: int, vol_size: int, iters: int) -> int:
    """Find the largest batch <= start_bs that survives fwd+bwd on the GPU.

    Auto-halves on CUDA OOM (start 8 per the paper-1 recipe).  Runs a full
    forward + backward + optimizer step in autocast to approximate the peak
    training footprint.
    """
    if not torch.cuda.is_available():
        return start_bs

    bs = start_bs
    while bs >= 1:
        model = None
        try:
            model = VolRAFTFixed().cuda()
            optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            v0 = torch.rand(bs, 1, vol_size, vol_size, vol_size, device="cuda")
            v1 = torch.rand(bs, 1, vol_size, vol_size, vol_size, device="cuda")
            with torch.amp.autocast("cuda"):
                preds = model(v0, v1, iters=iters)
                loss = sum(p.float().abs().mean() for p in preds)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            del model, optim, scaler, v0, v1, preds, loss
            torch.cuda.empty_cache()
            print(f"[probe] batch {bs}: OK")
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"[probe] batch {bs}: CUDA OOM -> halving")
            del model
            torch.cuda.empty_cache()
            bs //= 2
    raise RuntimeError("Even batch size 1 OOMs -- cannot train on this GPU.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VolRAFT (fixed sampler) on our NPZ dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Dataset dir containing train/ and val/ NPZ splits.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Checkpoints + training.log destination.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Starting batch; auto-halved on OOM by a probe.")
    parser.add_argument("--max-lr", type=float, default=2.0e-4)
    parser.add_argument("--pct-start", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-probe", action="store_true",
                        help="Skip the OOM batch-size probe.")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seeding (mirror train_phase1.py: torch + numpy + python random)
    import random as _py_random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    _py_random.seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    data_dir = args.data_dir
    if not (data_dir / "train").exists():
        raise FileNotFoundError(f"train split not found under {data_dir}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Batch-size probe (auto-halve on OOM), then dataloaders
    batch_size = args.batch_size
    if not args.no_probe:
        # 64^3 volumes; probe with the real spatial size
        batch_size = _probe_batch_size(batch_size, vol_size=64, iters=args.iters)
    train_loader, val_loader = _build_dataloaders(
        data_dir, batch_size, args.num_workers, generator)

    mixed_precision = (not args.no_amp) and torch.cuda.is_available()
    model = VolRAFTFixed(mixed_precision=mixed_precision)

    train_config = {
        "epochs": args.epochs,
        "iters": args.iters,
        "lr": args.max_lr,
        "weight_decay": 5.0e-5,
        "gamma": 0.8,
        "grad_clip": 1.0,
        "mixed_precision": mixed_precision,
        "scheduler_type": "onecycle",
        "max_lr": args.max_lr,
        "pct_start": args.pct_start,
        "anneal_strategy": "cos",
        "div_factor": 25.0,
        "final_div_factor": 1.0e4,
        "grad_accum_steps": 1,
        "log_interval": 10,
        "val_interval": 5,
        "save_interval": 50,
        "latest_interval": 10,
    }

    steps_per_epoch = len(train_loader)
    print("=" * 72)
    print(f"[volraft_fixed] data       : {data_dir}")
    print(f"[volraft_fixed] output     : {output_dir}")
    print(f"[volraft_fixed] batch size : {batch_size} "
          f"({steps_per_epoch} optimizer steps/epoch, "
          f"{args.epochs * steps_per_epoch} total)")
    print(f"[volraft_fixed] params     : {model.get_num_parameters():,}")
    print(f"[volraft_fixed] amp        : {mixed_precision}")
    print(f"[volraft_fixed] cuda       : {torch.cuda.is_available()}")
    print("=" * 72, flush=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        config=train_config,
    )

    # Persist the resolved run configuration for reproducibility
    resolved = {
        "experiment_name": output_dir.name,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "batch_size": batch_size,
        "batch_size_requested": args.batch_size,
        "num_workers": args.num_workers,
        "training": train_config,
        "model_architecture": VolRAFTFixed.ARCH,
    }
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)

    if args.resume is not None:
        trainer.load_checkpoint(str(args.resume))

    best_val = trainer.train()
    print(f"[volraft_fixed] done. best_val_loss={best_val:.6f}")
    print(f"[volraft_fixed] checkpoints at: {output_dir}")


if __name__ == "__main__":
    # Survive console control signals during long detached runs
    # (mirrors scripts/phase1/train_phase1.py).
    import signal as _signal
    for _s in ("SIGINT", "SIGBREAK"):
        if hasattr(_signal, _s):
            try:
                _signal.signal(getattr(_signal, _s), _signal.SIG_IGN)
            except Exception:
                pass
    main()
