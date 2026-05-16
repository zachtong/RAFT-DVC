"""Phase-1 training entry point.

Thin wrapper around :class:`src.training.Trainer`.  All shared settings
live in ``configs/phase1/_base.yaml``; per-run variations (encoder, data
config) are selected on the command line.

Typical usage (local)::

    python scripts/phase1/train_phase1.py \
        --model-config configs/models/raft_dvc_1_8_p4_r4.yaml \
        --data-config  r2_sparse_size32 \
        --experiment-name phase1_r2sparse32_enc1_8

On TACC the data and checkpoint locations come from environment
variables set by ``scripts/slurm_train.sh``::

    export DATA_PHASE1_ROOT=$SCRATCH/raft-dvc/data_phase1
    export CHECKPOINTS_ROOT=$WORK/checkpoints/raft-dvc

The training-data directory is resolved as
``<DATA_PHASE1_ROOT>/<data-config>/{train,val}`` (flat layout -- the phase
is already in the directory name, so no redundant ``/phase1/`` subpath).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

# -- Ensure repo root is importable when invoked from any CWD -----------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core import RAFTDVC, RAFTDVCConfig  # noqa: E402
from src.data import safe_phase1_collate  # noqa: E402
from src.data.dataset import Phase1NPZDataset  # noqa: E402
from src.data.otf_dataset import build_otf_dataset  # noqa: E402
from src.training import Trainer  # noqa: E402


# -- Argparse -----------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RAFT-DVC model on a Phase-1 dataset configuration.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=_REPO_ROOT / "configs" / "phase1" / "_base.yaml",
        help="Path to the shared Phase-1 base YAML config.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to a model architecture YAML (e.g. configs/models/raft_dvc_1_8_p4_r4.yaml).",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Phase-1 data configuration name, e.g. r2_sparse_size32.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Unique tag for this run (becomes the checkpoint subdirectory).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        dest="data_root",
        help="Root Phase-1 data directory (holds <data-config>/<split>/). "
        "Default: $DATA_PHASE1_ROOT or <repo>/data_phase1.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root checkpoints directory. Default: $CHECKPOINTS_ROOT or <repo>/checkpoints.",
    )
    # Common overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision even if the base config enables it.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    # On-the-fly (OTF) generation: no disk reads, samples synthesised by
    # CPU workers each call.  Eliminates dataset memorisation at the cost
    # of ~10-50 ms/sample CPU per worker.
    parser.add_argument(
        "--otf", action="store_true",
        help="Use on-the-fly sample generation instead of reading NPZ files.",
    )
    parser.add_argument(
        "--otf-train-length", type=int, default=1000,
        help="Number of samples per training epoch when --otf is set "
             "(default: 1000 to match the NPZ split sizes).",
    )
    parser.add_argument(
        "--otf-val-length", type=int, default=100,
        help="Number of samples per validation pass when --otf is set "
             "(default: 100).",
    )
    parser.add_argument(
        "--feature-map-size", type=int, default=16,
        help="Feature-map size used by OTF disp scaling (default: 16). "
             "Must match the encoder's effective fm size.",
    )
    parser.add_argument(
        "--max-lr", type=float, default=None,
        help="Override max_lr (peak LR for OneCycleLR / CyclicLR) -- needed "
             "when scaling batch size from the _base.yaml default. "
             "Sqrt scaling rule: max_lr ~ 2e-4 * sqrt(batch/8).",
    )
    return parser.parse_args()


# -- Helpers ------------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_data_root(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.environ.get("DATA_PHASE1_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return (_REPO_ROOT / "data_phase1").resolve()


def _resolve_output_root(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.environ.get("CHECKPOINTS_ROOT")
    if env_value:
        return Path(env_value).resolve()
    return (_REPO_ROOT / "checkpoints").resolve()


def _build_dataloaders(
    data_cfg: dict,
    data_root: Path,
    split_batch_size: tuple[int, int],
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = Phase1NPZDataset(
        root_dir=str(data_root / "train"),
        augment=bool(data_cfg.get("augment_train", True)),
        patch_size=(
            tuple(data_cfg["patch_size"]) if data_cfg.get("patch_size") else None
        ),
        include_metadata=bool(data_cfg.get("include_metadata", False)),
    )
    val_ds = Phase1NPZDataset(
        root_dir=str(data_root / "val"),
        augment=bool(data_cfg.get("augment_val", False)),
        patch_size=(
            tuple(data_cfg["patch_size"]) if data_cfg.get("patch_size") else None
        ),
        include_metadata=bool(data_cfg.get("include_metadata", False)),
    )

    bs_train, bs_val = split_batch_size
    pin = bool(data_cfg.get("pin_memory", True))
    persistent = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=bs_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=True,
        collate_fn=safe_phase1_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent,
        drop_last=False,
        collate_fn=safe_phase1_collate,
    )
    return train_loader, val_loader


def _build_otf_dataloaders(
    data_cfg: dict,
    data_config_name: str,
    split_batch_size: tuple[int, int],
    num_workers: int,
    feature_map_size: int,
    train_length: int,
    val_length: int,
) -> tuple[DataLoader, DataLoader]:
    """Build OTF train/val loaders.  No disk reads; CPU workers synthesise
    each sample.  Train and val use different ``seed_base`` so the val pool
    is fully disjoint from training."""
    train_ds = build_otf_dataset(
        data_config_name=data_config_name,
        length=train_length,
        feature_map_size=feature_map_size,
        include_metadata=bool(data_cfg.get("include_metadata", False)),
        augment=bool(data_cfg.get("augment_train", True)),
        seed_base=42_000_000,
    )
    val_ds = build_otf_dataset(
        data_config_name=data_config_name,
        length=val_length,
        feature_map_size=feature_map_size,
        include_metadata=bool(data_cfg.get("include_metadata", False)),
        augment=False,
        seed_base=42_500_000,  # disjoint pool
    )

    bs_train, bs_val = split_batch_size
    pin = bool(data_cfg.get("pin_memory", True))
    persistent = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=bs_train, shuffle=False,  # shuffling pointless
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent, drop_last=True,
        collate_fn=safe_phase1_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs_val, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
        persistent_workers=persistent, drop_last=False,
        collate_fn=safe_phase1_collate,
    )
    return train_loader, val_loader


def _build_model(model_yaml: dict) -> RAFTDVC:
    arch = model_yaml.get("architecture", {})
    config = RAFTDVCConfig.from_dict(arch)
    return RAFTDVC(config)


def _build_trainer_config(base_cfg: dict, args: argparse.Namespace) -> dict:
    training = dict(base_cfg["training"])
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.no_amp:
        training["mixed_precision"] = False
    if args.max_lr is not None:
        # Override both fields together -- OneCycleLR uses ``max_lr``, but
        # the optimizer's initial lr also depends on ``lr`` (= peak before
        # div_factor scaling).  Keep them in lockstep.
        training["max_lr"] = float(args.max_lr)
        training["lr"] = float(args.max_lr)
    return training


def _resolve_batch_sizes(base_cfg: dict, cli_bs: int | None) -> tuple[int, int]:
    base_bs = int(cli_bs) if cli_bs is not None else int(base_cfg["data"]["batch_size"])
    val_bs = int(base_cfg.get("validation", {}).get("batch_size", base_bs))
    return base_bs, val_bs


def _resolve_num_workers(base_cfg: dict, cli_nw: int | None) -> int:
    if cli_nw is not None:
        return int(cli_nw)
    return int(base_cfg["data"].get("num_workers", 4))


# -- Main ---------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    base_cfg = _load_yaml(args.base_config)
    model_yaml = _load_yaml(args.model_config)

    # Seeding
    seed = (
        args.seed
        if args.seed is not None
        else int(base_cfg.get("run", {}).get("seed", 42))
    )
    torch.manual_seed(seed)

    # Path resolution
    data_phase1_root = _resolve_data_root(args.data_root)
    output_root = _resolve_output_root(args.output_root)
    output_dir = output_root / "phase1" / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders
    bs_train, bs_val = _resolve_batch_sizes(base_cfg, args.batch_size)
    num_workers = _resolve_num_workers(base_cfg, args.num_workers)

    if args.otf:
        # OTF mode: no on-disk dataset required; CPU workers synthesise.
        print(
            f"[phase1] OTF mode     : ON  "
            f"(train={args.otf_train_length}, val={args.otf_val_length}, "
            f"fm={args.feature_map_size})"
        )
        train_loader, val_loader = _build_otf_dataloaders(
            data_cfg=base_cfg["data"],
            data_config_name=args.data_config,
            split_batch_size=(bs_train, bs_val),
            num_workers=num_workers,
            feature_map_size=args.feature_map_size,
            train_length=args.otf_train_length,
            val_length=args.otf_val_length,
        )
        data_root = None  # not used
    else:
        data_root = data_phase1_root / args.data_config
        if not data_root.exists():
            raise FileNotFoundError(
                f"Phase-1 data directory not found: {data_root}\n"
                f"Check --data-config and --data-root (or $DATA_PHASE1_ROOT)."
            )
        train_loader, val_loader = _build_dataloaders(
            base_cfg["data"],
            data_root,
            split_batch_size=(bs_train, bs_val),
            num_workers=num_workers,
        )

    # Model
    model = _build_model(model_yaml)

    # Trainer config
    train_config = _build_trainer_config(base_cfg, args)

    data_summary = "OTF (in-memory)" if args.otf else f"{args.data_config}  ({data_root})"
    print("=" * 72)
    print(f"[phase1] experiment : {args.experiment_name}")
    print(f"[phase1] model      : {args.model_config.name}")
    print(f"[phase1] data       : {data_summary}")
    print(f"[phase1] batch size : train={bs_train}, val={bs_val}")
    print(f"[phase1] workers    : {num_workers}")
    print(f"[phase1] output     : {output_dir}")
    print(f"[phase1] device ok  : cuda={torch.cuda.is_available()}")
    print(f"[phase1] params     : {model.get_num_parameters():,}")
    print("=" * 72, flush=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        config=train_config,
    )

    # Persist the resolved config next to the checkpoints for reproducibility.
    resolved = {
        "base_config_path": str(args.base_config),
        "model_config_path": str(args.model_config),
        "data_config": args.data_config,
        "experiment_name": args.experiment_name,
        "data_phase1_root": str(data_phase1_root),
        "output_root": str(output_root),
        "seed": seed,
        "batch_size_train": bs_train,
        "batch_size_val": bs_val,
        "num_workers": num_workers,
        "training": train_config,
        "model_architecture": model_yaml.get("architecture", {}),
    }
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)

    if args.resume is not None:
        trainer.load_checkpoint(str(args.resume))

    best_val = trainer.train()
    print(f"[phase1] done. best_val_loss={best_val:.6f}")
    print(f"[phase1] checkpoints at: {output_dir}")


if __name__ == "__main__":
    main()
