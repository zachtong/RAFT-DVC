# RAFT-DVC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21578266.svg)](https://doi.org/10.5281/zenodo.21578266)

Resolution-aware learned digital volume correlation for particle-labeled
volumes. This repository contains the reference implementation, the three
trained solvers, the synthetic-volume generator, and the headless drivers for
the classical DVC baselines used in the accompanying paper.

Zixiang (Zach) Tong, Lehu Bu, Jin Yang — University of Texas at Austin.
Contact: <zachtong@utexas.edu>.

## What this is

A 3D adaptation of the RAFT optical-flow architecture for DVC. The
displacement field is solved on a coarse feature grid at 1/s of the input
resolution and interpolated back to the voxel grid, so the downsampling factor
`s` sets both the cost and the finest resolvable feature. Rather than ship one
network, we train three *arms* at `s = 2, 4, 8`, each matched to a particle
size and a displacement band, and give a rule for choosing between them.

| arm | downsample `s` | particle radius | displacement band | training volume |
|-----|----------------|-----------------|-------------------|-----------------|
| s2  | 2              | 2 voxel         | 2–4 voxel         | 32³             |
| s4  | 4              | 4 voxel         | 4–8 voxel         | 64³             |
| s8  | 8              | 8 voxel         | 8–16 voxel        | 128³            |

All three share one architecture (2 correlation levels, search radius 4, 12
update iterations) and one optimizer schedule, so differences between them come
from the input scale alone.

## Choosing an arm

Two constraints decide it.

**Resolvability.** The feature grid must still see the particles, which
requires the particle diameter to be at least about `s` voxel. This rules out
the coarse arms on fine-grained volumes.

**Reach.** The correlation pyramid searches `radius x 2^(levels-1) x s = 8s`
voxel nominally; measured collapse points are near 6, 13, and 16 voxel for s2,
s4, and s8. The displacement must stay inside that range.

Deploy the smallest `s` that resolves the particles and still reaches the
expected displacement. Within its band, an arm's error scales with the grid it
solves on, roughly `0.017 x s` voxel; picking a coarser arm than necessary
costs accuracy proportionally.

## Setup

```bash
conda create -n raft-dvc python=3.10
conda activate raft-dvc
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Trained weights for the three arms are attached to the tagged release. Each is
about 8 MB and carries its own `model_config`, so inference does not need a
separate architecture file.

## Reproducing the results

**Generate synthetic volumes.** The generator places Gaussian-rendered
particles and applies a deformation drawn from the DIC shape-function classes
(translation, affine, quadratic), then adds the imaging noise model:

```bash
python scripts/generate_phase1_dataset.py --config configs/data_generation/<cfg>.yaml
```

Volume size, particle radius and density, and the displacement band are all set
in the YAML. The benchmark volumes in the paper are not redistributed because
this generator reproduces them deterministically from the parameters tabulated
in the paper's appendix.

**Train.**

```bash
python scripts/phase1/train_phase1.py --model-config configs/models/raft_dvc_1_4_p2_r4.yaml ...
```

`scripts/phase1/slurm_*.sh` are the SLURM templates used on TACC Vista.

**Evaluate.**

```bash
python scripts/phase1/evaluate_phase1.py --checkpoint <arm>.pth --data <split>
```

**Classical baselines.** `traditional_DIC_codes/benchmark/` holds
non-interactive MATLAB drivers for local subset DVC, ALDVC, and finite-element
global DVC, plus the `.npz` to `.mat` converter. Every solver parameter used in
the paper is tabulated there and in the paper's appendix. `AXIS_CONVENTION.md`
documents the index mapping between the Python and MATLAB sides, which is worth
reading before comparing fields.

## The correlation-sampler test

`tests/test_corr_sampler.py` is the impulse test discussed in the paper. It
places a single non-zero voxel in the correlation volume and checks that the
sampler reads it back at the queried location. A defect that transposes two
axes passes ordinary shape checks and still trains, but shifts the correlation
peak with position; the test fails immediately on that defect. If you port RAFT
to 3D, run this first.

```bash
pytest tests/test_corr_sampler.py -v
```

## VolRAFT comparison

`volraft_asreleased/` and `volraft_fixed/` are ports of
[VolRAFT](https://github.com/hereon-mbs/VolRAFT) used for the architecture
comparison in the paper: the first reproduces the upstream sampler behaviour,
the second applies the axis-order correction. Both are derived work under the
upstream MIT license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{tong2026raftdvc,
  author  = {Tong, Zixiang and Bu, Lehu and Yang, Jin},
  title   = {{RAFT-DVC}: A resolution-aware framework for learned digital
             volume correlation in particle-labeled volumes},
  journal = {Acta Mechanica Sinica},
  year    = {2026}
}
```

The archived v1.0 release: doi:10.5281/zenodo.21578267 (concept DOI
10.5281/zenodo.21578266 always resolves to the latest version).

The real confocal indentation volume used for validation comes from the DVC
Challenge 2.0 dataset (doi:10.21203/rs.3.rs-9683321/v1).

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses are
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
