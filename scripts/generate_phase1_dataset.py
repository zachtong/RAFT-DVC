"""
Phase-1 dataset generation (RAFTcorr3D) -- fm=16 design.

Generates per-sample NPZ files under
``data_phase1/<config>/<split>/sample_NNNNN.npz``.

Each NPZ contains:
    I1       (D, H, W) float32 -- noisy reference image
    I2       (D, H, W) float32 -- noisy warped image (independent noise realization)
    flow     (3, D, H, W) float32 -- ground-truth displacement (uz, uy, ux)
    metadata (0-d object) -- JSON string with deform_type, fm_target,
                             feature_map_size, mix_choice (for mixed configs),
                             seed, etc.

Dataset scope (revised 2026-05-15 for fm=16):
    10 configs per size:
        9 single: r{2,4,6} x density{sparse=0.2, medium=0.6, dense=1.0}
        1 mixed : each sample randomly draws (radius, density) from the 9
    Sizes: 16, 32 by default (64 deferred per user request 2026-05-15;
                              128 deferred -- requires TACC SCRATCH storage).
    Per config per split: train=1000, val=100, test=100.
    Feature map: 16 (drives displacement scaling per
                     memory/displacement_warp_design.md, with fm_max=3,
                     input_max_disp = fm_target * size / feature_map_size).

CLI examples
------------
Default full generation (sizes 16, 32; fm=16; 10 configs each; with viz)::

    python scripts/generate_phase1_dataset.py

Quick smoke test (5 train + 2 val + 2 test on size 16, single + mixed)::

    python scripts/generate_phase1_dataset.py --sizes 16 \
        --train 5 --val 2 --test 2

Only single configs (no mixed)::

    python scripts/generate_phase1_dataset.py --no-mixed

Skip visualization to save time::

    python scripts/generate_phase1_dataset.py --no-viz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


# -----------------------------------------------------------------------------
# Environment sanity check (run BEFORE any heavy imports)
# -----------------------------------------------------------------------------
# Common failure mode in mixed conda/pip envs on Windows: numpy ends up
# half-installed and reports ``cannot import name '__version__' from numpy
# (unknown location)`` when scipy/matplotlib/torch try to use it.  No shim
# can rescue a numpy that cannot expose ``__version__``; we surface a clean
# actionable message instead of a deep stack trace from scipy.

def _check_numpy_or_die() -> None:
    try:
        import numpy as _np  # noqa: F401
        _ = _np.__version__  # scipy/matplotlib both read this attribute
        _ = _np.zeros((2,))  # smoke-check core dispatch
    except Exception as exc:
        print(
            "\n[FATAL] numpy is broken in the active Python environment.",
            file=sys.stderr,
        )
        print(f"  Underlying error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "  Fix (typical -- pick one):\n"
            "    1) Reinstall numpy in the current env:\n"
            "         pip uninstall -y numpy && pip install 'numpy>=1.25,<2.1'\n"
            "    2) If conda + pip are mixed, rebuild the env from scratch:\n"
            "         conda env remove -n <env>\n"
            "         conda create -n <env> python=3.10 numpy scipy matplotlib pytorch -c pytorch\n"
            "    3) Verify with:\n"
            "         python -c \"import numpy; print(numpy.__file__, numpy.__version__)\"\n"
            "  Then re-run this script.",
            file=sys.stderr,
        )
        raise SystemExit(2)


_check_numpy_or_die()

import numpy as np  # noqa: E402  -- safe after sanity check

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'data_generation'))

from modules.dataset_viz import visualize_dataset  # noqa: E402
from modules.sample_generator import Phase1SampleGenerator  # noqa: E402


# -----------------------------------------------------------------------------
# Dataset specification
# -----------------------------------------------------------------------------

RADII = [2, 4, 6]
DENSITIES = [('sparse', 0.2), ('medium', 0.6), ('dense', 1.0)]
DEFAULT_FEATURE_MAP_SIZE = 16
BASE_SEED = 20260416
MIXED_SEED_STRIDE = 31  # coprime with 9 so all (radius, density) combos appear
MIXED_NAME = 'mixed'


def config_hash(size, radius, density_name):
    """Stable 32-bit hash for (size, radius, density)."""
    s = f"size={size}|r={radius}|d={density_name}"
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:8], 16)


def sample_seed(size, radius, density_name, split, sample_idx):
    """Deterministic seed per (config, split, sample_idx)."""
    split_offset = {'train': 0, 'val': 10_000_000, 'test': 20_000_000}[split]
    ch = config_hash(size, radius, density_name) % 997
    return BASE_SEED + ch * 100_000_000 + split_offset + sample_idx


def mixed_seed(size, split, sample_idx):
    """Deterministic seed for the 'mixed' config.

    Independent of the single-config seed space (uses a different hash slot).
    """
    split_offset = {'train': 0, 'val': 10_000_000, 'test': 20_000_000}[split]
    ch = config_hash(size, 0, MIXED_NAME) % 997  # 0 -> reserved for mixed
    return BASE_SEED + ch * 100_000_000 + split_offset + sample_idx + 7_000_000


def single_config_dirname(size, radius, density_name):
    return f"r{radius}_{density_name}_size{size}"


def mixed_config_dirname(size):
    return f"{MIXED_NAME}_size{size}"


# -----------------------------------------------------------------------------
# Save helper
# -----------------------------------------------------------------------------

def save_sample(path, sample):
    """Save one sample as uncompressed NPZ. Metadata stored as JSON string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        I1=sample['I1'],
        I2=sample['I2'],
        flow=sample['flow'],
        metadata=np.asarray(json.dumps(sample['metadata']), dtype=object),
    )


# -----------------------------------------------------------------------------
# Generation loops -- single configs
# -----------------------------------------------------------------------------

def generate_split_single(generator, size, radius, density_name, split,
                          n_samples, output_root, progress_every=100):
    """Generate n_samples for one single (radius, density, size, split)."""
    out_dir = (
        output_root / single_config_dirname(size, radius, density_name) / split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(n_samples):
        seed = sample_seed(size, radius, density_name, split, i)
        sample = generator.generate(seed)
        save_sample(out_dir / f"sample_{i:05d}.npz", sample)

        if (i + 1) % progress_every == 0 or (i + 1) == n_samples:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            print(
                f"    [{split}] {i + 1:5d}/{n_samples}  "
                f"{rate:5.2f} samp/s  ({dt:5.1f}s)"
            )


def generate_single_configs(
    sizes,
    counts,
    output_root,
    feature_map_size,
    only_configs=None,
    input_disp_range=None,
):
    """Loop over sizes x (radius, density) configs x splits.

    Args:
        only_configs: Optional iterable of config names ("r4_medium" etc.) to
            restrict generation to a subset of the 9 (R, density) grid.
            ``None`` = generate all 9.
        input_disp_range: Optional ``(low, high)`` tuple of input-space
            displacement bounds (voxels).  When set, the generator uses
            "input" disp mode (fixed input deformation regardless of
            encoder downsample factor) instead of the default "fm" mode
            (fixed fm-scale difficulty).
    """
    if only_configs is not None:
        only_configs = set(only_configs)

    for size in sizes:
        for radius in RADII:
            for density_name, density_val in DENSITIES:
                cfg_id = f"r{radius}_{density_name}"
                if only_configs is not None and cfg_id not in only_configs:
                    continue
                name = single_config_dirname(size, radius, density_name)
                print(f"\n=== {name} ===")
                t_cfg_start = time.time()
                gen_kwargs = dict(
                    size=size,
                    radius=radius,
                    density_per_1000=density_val,
                    feature_map_size=feature_map_size,
                )
                if input_disp_range is not None:
                    gen_kwargs["input_disp_min"] = input_disp_range[0]
                    gen_kwargs["input_disp_max"] = input_disp_range[1]
                generator = Phase1SampleGenerator(**gen_kwargs)
                for split, n in counts.items():
                    if n > 0:
                        generate_split_single(
                            generator, size, radius, density_name,
                            split, n, output_root,
                        )
                dt_cfg = time.time() - t_cfg_start
                print(f"  [{name}] done in {dt_cfg:.1f}s")


# -----------------------------------------------------------------------------
# Generation loops -- mixed config
# -----------------------------------------------------------------------------

def _build_mixed_generator_pool(size, feature_map_size):
    """Pre-instantiate 9 generators (one per single config) for mixed mode."""
    pool = {}
    for radius in RADII:
        for density_name, density_val in DENSITIES:
            key = (radius, density_name)
            pool[key] = Phase1SampleGenerator(
                size=size,
                radius=radius,
                density_per_1000=density_val,
                feature_map_size=feature_map_size,
            )
    return pool


def _mix_choice_for_sample(size, split, sample_idx):
    """Deterministic (radius, density_name) pick for a given mixed sample."""
    # Hash-driven, stride-based cycling so the 9 combos appear roughly
    # uniformly across the split.
    ch = config_hash(size, 0, MIXED_NAME) % 997
    slot = (ch + sample_idx * MIXED_SEED_STRIDE) % 9
    radius = RADII[slot // 3]
    density_name, _ = DENSITIES[slot % 3]
    return radius, density_name


def generate_split_mixed(pool, size, split, n_samples, output_root,
                         progress_every=100):
    """Generate n_samples for the mixed config of one (size, split)."""
    out_dir = output_root / mixed_config_dirname(size) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i in range(n_samples):
        radius, density_name = _mix_choice_for_sample(size, split, i)
        generator = pool[(radius, density_name)]
        seed = mixed_seed(size, split, i)
        sample = generator.generate(seed)
        # Tag the metadata with the mix choice for downstream analysis.
        sample['metadata']['mix_choice'] = f"r{radius}_{density_name}"
        save_sample(out_dir / f"sample_{i:05d}.npz", sample)

        if (i + 1) % progress_every == 0 or (i + 1) == n_samples:
            dt = time.time() - t0
            rate = (i + 1) / max(dt, 1e-6)
            print(
                f"    [{split}] {i + 1:5d}/{n_samples}  "
                f"{rate:5.2f} samp/s  ({dt:5.1f}s)"
            )


def generate_mixed_configs(sizes, counts, output_root, feature_map_size):
    """Generate the mixed_size{X} dataset for each requested size."""
    for size in sizes:
        name = mixed_config_dirname(size)
        print(f"\n=== {name} ===")
        t_cfg_start = time.time()
        pool = _build_mixed_generator_pool(size, feature_map_size)
        for split, n in counts.items():
            if n > 0:
                generate_split_mixed(pool, size, split, n, output_root)
        dt_cfg = time.time() - t_cfg_start
        print(f"  [{name}] done in {dt_cfg:.1f}s")


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def run_visualization(sizes, output_root, include_mixed, n_viz=10):
    """Render overview PNGs for the first n_viz train samples of every config."""
    print(f"\n=== visualizations (first {n_viz} train samples per config) ===")
    t0 = time.time()
    total = 0
    for size in sizes:
        for radius in RADII:
            for density_name, _ in DENSITIES:
                cfg_dir = output_root / single_config_dirname(
                    size, radius, density_name
                )
                if cfg_dir.exists():
                    n = visualize_dataset(cfg_dir, n_samples=n_viz)
                    total += n
        if include_mixed:
            cfg_dir = output_root / mixed_config_dirname(size)
            if cfg_dir.exists():
                n = visualize_dataset(cfg_dir, n_samples=n_viz)
                total += n
    print(f"  rendered {total} PNGs in {time.time() - t0:.1f}s")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_sizes(s):
    return [int(x) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--sizes', type=_parse_sizes, default=[16, 32],
                        help='Comma-separated input sizes (default: 16,32)')
    parser.add_argument('--feature-map-size', type=int,
                        default=DEFAULT_FEATURE_MAP_SIZE,
                        help='Feature-map size that drives disp scaling '
                             '(default: 16)')
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--val', type=int, default=100)
    parser.add_argument('--test', type=int, default=100)
    parser.add_argument('--output-root', type=str, default='data_phase1',
                        help='Output directory relative to project root '
                             '(default: data_phase1)')
    parser.add_argument('--no-mixed', dest='with_mixed', action='store_false',
                        help='Skip mixed_size{X} generation')
    parser.add_argument('--no-viz', dest='with_viz', action='store_false',
                        help='Skip per-config visualization PNGs')
    parser.add_argument('--viz-samples', type=int, default=10,
                        help='Number of train samples to visualize per '
                             'config (default: 10)')
    parser.add_argument(
        '--only-config',
        type=str, default=None,
        help='Comma-separated config IDs (e.g. "r4_medium" or '
             '"r4_medium,r2_sparse") to restrict generation to a subset. '
             'Format: "r{R}_{density}" with R in {2,4,6} and density in '
             '{sparse,medium,dense}. When set, mixed is auto-skipped.',
    )
    parser.add_argument(
        '--input-disp-range',
        type=float, nargs=2, metavar=('LOW', 'HIGH'), default=None,
        help='Two floats: input-space displacement range in voxels '
             '(e.g. "0.6 6"). When set, the generator uses input-space '
             'mode: input_max_disp ~ U(LOW, HIGH); fm-disp is derived. '
             'Used for paper-1 architecture-ablation experiments where '
             'we fix the physical deformation regardless of encoder.',
    )
    parser.set_defaults(with_mixed=True, with_viz=True)
    args = parser.parse_args()

    # Parse --only-config into a set; auto-skip mixed when restricted.
    only_configs = None
    if args.only_config:
        only_configs = [c.strip() for c in args.only_config.split(',') if c.strip()]
        # Sanity-check each entry
        valid_density = {n for n, _ in DENSITIES}
        for cfg in only_configs:
            parts = cfg.split('_')
            if (len(parts) != 2 or not parts[0].startswith('r')
                    or parts[1] not in valid_density):
                raise SystemExit(
                    f"--only-config entry {cfg!r} invalid; expected "
                    f"'r{{2,4,6}}_{{sparse,medium,dense}}'."
                )
        args.with_mixed = False  # mixed makes no sense when restricting

    input_disp_range = None
    if args.input_disp_range is not None:
        lo, hi = args.input_disp_range
        if lo <= 0 or hi <= lo:
            raise SystemExit(
                f"--input-disp-range invalid: LOW={lo}, HIGH={hi} "
                f"(require 0 < LOW < HIGH)."
            )
        input_disp_range = (float(lo), float(hi))

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    counts = {'train': args.train, 'val': args.val, 'test': args.test}
    n_per_size = (len(only_configs) if only_configs is not None
                  else len(RADII) * len(DENSITIES))
    n_single = len(args.sizes) * n_per_size
    n_mixed = len(args.sizes) if args.with_mixed else 0
    disp_str = (f"input-space [{input_disp_range[0]}, {input_disp_range[1]}] voxels"
                if input_disp_range is not None
                else f"fm-space [0.3, 3] x downsample (default)")
    cfg_str = (",".join(only_configs) if only_configs is not None
               else "all 9 (r{2,4,6} x {sp,md,dn})")
    print(
        f"=== Generation plan ===\n"
        f"  sizes              : {args.sizes}\n"
        f"  feature_map_size   : {args.feature_map_size}\n"
        f"  disp mode          : {disp_str}\n"
        f"  single configs     : {cfg_str}  ({n_per_size} per size)\n"
        f"  mixed configs      : {n_mixed}\n"
        f"  train/val/test     : {counts['train']}/{counts['val']}/{counts['test']}\n"
        f"  total configs      : {n_single + n_mixed}\n"
        f"  total samples      : {(n_single + n_mixed) * sum(counts.values())}\n"
        f"  visualizations     : {'yes' if args.with_viz else 'no'}\n"
        f"  output_root        : {output_root}\n"
    )

    t0 = time.time()

    generate_single_configs(
        sizes=args.sizes, counts=counts, output_root=output_root,
        feature_map_size=args.feature_map_size,
        only_configs=only_configs,
        input_disp_range=input_disp_range,
    )

    if args.with_mixed:
        generate_mixed_configs(
            sizes=args.sizes, counts=counts, output_root=output_root,
            feature_map_size=args.feature_map_size,
        )

    if args.with_viz:
        run_visualization(
            sizes=args.sizes, output_root=output_root,
            include_mixed=args.with_mixed, n_viz=args.viz_samples,
        )

    total = time.time() - t0
    print(f"\nAll done in {total / 60:.1f} min.")
    print(f"Dataset at: {output_root}")


if __name__ == '__main__':
    main()
