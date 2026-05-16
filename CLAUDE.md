# RAFT-DVC Project Rules

## Language Policy
- **Communication**: Always use Chinese (中文) when talking to the user
- **Code & Comments**: Always use English for all code, comments, variable names, commit messages, and technical documentation

## Testing Policy — Visual PDF Reports

When a new feature is developed or an existing feature is significantly modified:

1. **Generate a visual PDF test report** and save it to `reports/` folder
2. The report filename should be descriptive: `reports/<feature_name>_report_<YYYY-MM-DD>.pdf`
3. The report must help the user intuitively understand the feature's:
   - **Effect** — field maps, displacement/strain visualizations, before/after comparisons
   - **Performance** — timing benchmarks, throughput metrics, convergence curves
   - **Boundary conditions** — behavior at edges, with noise, at extreme parameters
   - **Limitations** — failure cases, accuracy degradation, known constraints
4. Include as appropriate: field maps, error distributions, summary tables, comparison charts
5. After generating the report, **always tell the user the exact file path and name**

## Non-Regression Policy

When modifying functionality:
- **Never degrade core features** that the user has previously validated and locked down
- If a change risks affecting existing behavior, verify the old behavior still works before claiming completion
- When in doubt, run existing tests or create comparison tests against the previous implementation

## Memory Policy

- Save important project context, decisions, and user preferences to the memory system
- Memory directory: `~/.claude/projects/C--Users-13014-OneDrive---The-University-of-Texas-at-Austin-Documents-Python-Codes-RAFT-DVC/memory/`
- Update memories when learning new project facts or receiving user corrections

## Project Structure (as of 2026-04-16 reorganization)

```
RAFT-DVC/
├── src/                       Core library (reusable across phases)
│   ├── core/                  RAFT-DVC network (encoder, correlation, update)
│   ├── data/                  Dataset classes
│   ├── training/              Trainer, loss, augmentation strategies
│   ├── utils/                 IO, memory, config loaders
│   ├── visualization/         TensorBoard / PyVista / matplotlib renderers
│   └── legacy_inference/      DEPRECATED for Phase-1/2 — retained for Phase-3
│                              real-data inference. Do NOT import for new work.
├── configs/
│   ├── models/                Encoder variants (1/1, 1/2, 1/4, 1/8)
│   ├── data_generation/       Synthetic data YAMLs
│   ├── phase1/                Phase-1 training configs (to be added)
│   └── inference/             Visualization configs
├── scripts/
│   ├── data_generation/       Synthetic data modules (beads, deformation, noise)
│   ├── generate_phase1_dataset.py
│   ├── preview_*.py           Noise / parameter preview tools
│   ├── slurm_train.sh         TACC Vista SLURM template
│   ├── tacc_setup.sh          TACC first-time setup cheatsheet
│   ├── upload_dataset_to_tacc.sh  rsync local data_phase1/ to $SCRATCH
│   └── phase1/                train_phase1.py, evaluate_phase1.py
├── data_phase1/               Local Phase-1 synthetic dataset (~60 GB, git-ignored)
│                              Layout: data_phase1/<config>/{train,val,test}/sample_*.npz
│                              Future phases: data_phase2/, data_phase3/
├── reports/                   PDF test reports (git-ignored, local only)
├── docs/                      Architecture & codebase guides
├── archive/                   Pre-Phase-1 experimental snapshot (read-only)
│   ├── configs_training_old/
│   ├── scripts_old/
│   ├── docs_old/
│   └── run.bat
└── CLAUDE.md                  This file
```

## Storage Policy — TACC Vista

- `$WORK` (1 TB, persistent): code repo, conda env, curated `checkpoint_best.pth`
- `$SCRATCH` (tens of TB, 10-day purge): **all synthetic data** (Phase-1 ~60 GB,
  Phase-2/3 multi-TB 128³/256³), training run logs, TensorBoard events
- `$HOME` (~10 GB): `.bashrc`, SSH keys only

Datasets live on `$SCRATCH` because `$WORK` cannot hold multi-TB data. Env
vars pointing at the live dataset:
- `DATA_PHASE1_ROOT=$SCRATCH/raft-dvc/data_phase1` (SLURM script exports this)
- (future) `DATA_PHASE2_ROOT`, `DATA_PHASE3_ROOT`

Use a touch-based keepalive in the SLURM script to refresh atime and avoid
the 10-day purge.

## Archive Policy

- Do **not** import from `archive/` in new code
- Do **not** modify `archive/` files except to add pointers to successors
- If resurrecting something: copy out, rename to avoid confusion

## Legacy Inference Module

- `src/legacy_inference/` is DEPRECATED for Phase-1/2 pipelines
- It is retained for Phase-3 real-data inference (VolRAFT comparison,
  experimental confocal data) and must not be imported for synthetic
  val/test evaluation — use `scripts/phase1/evaluate_phase1.py` instead
