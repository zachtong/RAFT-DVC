# RAFT-DVC: Deep Learning for Digital Volume Correlation

A PyTorch implementation of 3D Digital Volume Correlation (DVC) using the RAFT
optical-flow architecture, with synthetic confocal-microscopy data generation
and a TACC-ready training pipeline.

**Based on**: [VolRAFT (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Wong_VolRAFT_Volumetric_Optical_Flow_Network_for_Digital_Volume_Correlation_of_CVPRW_2024_paper.html)

**Authors**: Zach Tong, Prof. Jin Yang (UT Austin), Lehu Bu

---

## 🧭 Project Status (2026-04-16)

This repository is in the middle of a reorganization. The live code targets
the **Phase-1 experimental campaign** for the RAFTcorr3D paper (40 training
runs across 9 data configurations × 4 encoder variants).

Pre-2026-04-16 experimental artifacts have been moved to [`archive/`](archive/)
as a read-only historical reference. See [`archive/README.md`](archive/README.md).

## 📁 Project Structure

```
RAFT-DVC/
├── src/                          Core library (reusable)
│   ├── core/                     RAFT-DVC network (encoder, correlation, update)
│   ├── data/                     Dataset classes
│   ├── training/                 Trainer, loss, augmentation strategies
│   ├── utils/                    IO, memory, config loaders
│   ├── visualization/            TensorBoard hooks, PyVista/matplotlib renders
│   └── legacy_inference/         DEPRECATED for Phase-1/2; retained for Phase-3
│
├── configs/
│   ├── models/                   Encoder variants (1/1, 1/2, 1/4, 1/8)
│   ├── data_generation/          Synthetic-data YAMLs
│   ├── phase1/                   Phase-1 training configs  (to be added)
│   └── inference/                Visualization configs
│
├── scripts/
│   ├── data_generation/          Synthetic-data modules
│   ├── generate_phase1_dataset.py
│   ├── preview_noise_model.py
│   ├── preview_parameter_grid.py
│   ├── slurm_train.sh            TACC Vista SLURM template
│   ├── tacc_setup.sh             TACC first-time setup cheatsheet
│   └── phase1/                   Phase-1 training/evaluation scripts  (to be added)
│
├── datasets/phase1/              Local Phase-1 dataset (not tracked by git)
├── reports/                      Auto-generated PDF test reports
├── docs/                         Architecture & codebase guide (Chinese + English)
├── archive/                      Pre-Phase-1 experimental snapshot (read-only)
└── CLAUDE.md                     Project rules for Claude Code
```

## 📦 Setup

### Local (Windows / Linux)

```bash
conda create -n raft-dvc python=3.10
conda activate raft-dvc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### TACC Vista (GH200)

See [`scripts/tacc_setup.sh`](scripts/tacc_setup.sh) — a copy-paste cheatsheet
for first-time setup of `$WORK` / `$SCRATCH` directories, conda env, and shell
aliases.

## 🚀 Phase-1 Workflow (in progress)

1. **Generate dataset locally** — `python scripts/generate_phase1_dataset.py`
   produces 1000/100/100 train/val/test NPZ samples per configuration.
2. **Upload dataset to TACC `$SCRATCH`** — see `scripts/upload_dataset_to_tacc.sh`
   (to be added).
3. **Train on Vista GH200** — `sbatch scripts/slurm_train.sh` (after editing
   the script to point at `train_phase1.py`).
4. **Evaluate** — `python scripts/phase1/evaluate_phase1.py --checkpoint ...`.

## 📚 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — core library design (authoritative)
- [`docs/CODEBASE_GUIDE_CN.md`](docs/CODEBASE_GUIDE_CN.md) — 中文代码库指南
- [`docs/plans/`](docs/plans/) — current experiment plans
- [`CLAUDE.md`](CLAUDE.md) — project conventions (language policy, testing policy)
- [`archive/`](archive/) — pre-Phase-1 experimental docs and configs

## 📖 Citation

```bibtex
@inproceedings{wong2024volraft,
  title={VolRAFT: Volumetric Optical Flow Network for Digital Volume Correlation},
  author={Wong, Chun Yin and Schanz, Daniel and Schr\"oder, Andreas and Geisler, Reinhard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
  pages={3621--3630}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE).
