# Archive — Pre-Phase-1 Experimental Snapshot

**Archived on:** 2026-04-16
**Archived by:** Zach Tong
**Reason:** Reorganization before starting the Phase-1 experimental campaign (40 training runs) for the RAFTcorr3D paper.

## What's here

This directory contains the experimental artifacts produced before 2026-04-16.
They are **read-only historical reference** and should not be imported by new code.

```
archive/
├── configs_training_old/      Old training YAMLs (confocal_128_*, confocal_32_*, etc.)
├── scripts_old/               Old entry-point scripts (train.py, train_confocal.py, test_confocal.py, infer.py)
├── docs_old/                  Old experimental docs (TRAINING_GUIDE, MoL uncertainty reports, etc.)
└── run.bat                    Old interactive Windows menu (referenced legacy scripts)
```

## What was KEPT in the live codebase

- `src/core/`, `src/data/`, `src/training/`, `src/utils/`, `src/visualization/` — core reusable library
- `src/legacy_inference/` (renamed from `src/inference/`) — retained for Phase-3 real-data inference; marked DEPRECATED for Phase-1/2
- `configs/models/` — 4 encoder variants (1/1, 1/2, 1/4, 1/8), still used
- `configs/data_generation/`, `configs/inference/` — still used
- `scripts/data_generation/`, `scripts/generate_phase1_dataset.py`, `scripts/preview_*.py`,
  `scripts/slurm_train.sh`, `scripts/tacc_setup.sh` — current infrastructure
- `docs/ARCHITECTURE.md`, `docs/CODEBASE_GUIDE_CN.md`, `docs/plans/` — still accurate

## Why we archived instead of deleting

- Git history alone is hard to navigate for "show me the old uncertainty experiment"
- A few legacy docs contain design rationale not yet captured elsewhere
- The old training configs show the evolution of hyper-parameter choices

## Rules for using archive/

- **Do not import from `archive/`** in new code
- **Do not modify** files here except to add links/notes pointing to their successors
- If you need to resurrect something, copy it out and rename it to avoid confusion
