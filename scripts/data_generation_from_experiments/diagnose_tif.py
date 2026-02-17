"""
Quick diagnostic script to check TIF file statistics.
Helps determine appropriate quality filtering thresholds.

Usage:
    python scripts/data_generation_from_experiments/diagnose_tif.py --tif_dir data_tif/your_experiment
"""

import argparse
import numpy as np
import tifffile
from pathlib import Path


def analyze_tif_file(tif_path):
    """Analyze a single TIF file."""
    print(f"\nAnalyzing: {tif_path.name}")
    print("-" * 60)

    # Load
    vol = tifffile.imread(tif_path)
    print(f"Shape: {vol.shape}")
    print(f"Dtype: {vol.dtype}")

    # Original stats
    print(f"\nOriginal values:")
    print(f"  Min:  {np.min(vol)}")
    print(f"  Max:  {np.max(vol)}")
    print(f"  Mean: {np.mean(vol):.2f}")
    print(f"  Std:  {np.std(vol):.2f}")

    # Normalize (same as in TifLoader)
    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        vol_norm = vol.astype(np.float32) / info.max
    else:
        vol_norm = vol.astype(np.float32)

    print(f"\nNormalized to [0, 1]:")
    print(f"  Min:  {np.min(vol_norm):.6f}")
    print(f"  Max:  {np.max(vol_norm):.6f}")
    print(f"  Mean: {np.mean(vol_norm):.6f}")
    print(f"  Std:  {np.std(vol_norm):.6f}")

    # Sample 128^3 regions
    print(f"\nSampling 5 random 128³ regions:")
    D, H, W = vol_norm.shape

    if D >= 128 and H >= 128 and W >= 128:
        for i in range(5):
            d = np.random.randint(0, D - 128)
            h = np.random.randint(0, H - 128)
            w = np.random.randint(0, W - 128)

            sample = vol_norm[d:d+128, h:h+128, w:w+128]
            print(f"  Sample {i+1}: mean={np.mean(sample):.6f}, std={np.std(sample):.6f}")
    else:
        print("  (Volume too small to sample 128³ regions)")

    return vol_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tif_dir', type=str, required=True,
                       help='Directory containing TIF files')
    args = parser.parse_args()

    tif_dir = Path(args.tif_dir)

    if not tif_dir.exists():
        print(f"Error: Directory not found: {tif_dir}")
        return

    # Find TIF files
    tif_files = list(tif_dir.glob('*.tif')) + list(tif_dir.glob('*.tiff'))

    if not tif_files:
        print(f"Error: No TIF files found in {tif_dir}")
        return

    print("=" * 60)
    print("TIF File Diagnostic Report")
    print("=" * 60)
    print(f"Directory: {tif_dir}")
    print(f"Found {len(tif_files)} TIF files")

    # Analyze each file
    all_means = []
    all_stds = []

    for tif_file in tif_files[:5]:  # Limit to first 5 files
        vol_norm = analyze_tif_file(tif_file)
        all_means.append(np.mean(vol_norm))
        all_stds.append(np.std(vol_norm))

    # Summary
    print("\n" + "=" * 60)
    print("Summary Statistics (normalized)")
    print("=" * 60)
    print(f"Mean intensity range: {min(all_means):.6f} - {max(all_means):.6f}")
    print(f"Std intensity range:  {min(all_stds):.6f} - {max(all_stds):.6f}")

    print("\nRecommended quality thresholds:")
    print(f"  min_mean_intensity: {min(all_means) * 0.5:.6f}  (50% of minimum mean)")
    print(f"  min_std_intensity:  {min(all_stds) * 0.5:.6f}  (50% of minimum std)")

    print("\nOr use these conservative values:")
    print(f"  min_mean_intensity: 0.0  (disable)")
    print(f"  min_std_intensity:  0.0  (disable)")


if __name__ == '__main__':
    main()
