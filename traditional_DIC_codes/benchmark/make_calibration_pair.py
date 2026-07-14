"""Generate the axis-calibration data for the headless DVC wrappers.

Creates a synthetic 64^3 random-blob volume (gaussian-filtered noise, soft
thresholded to [0,1]) and three deformed copies, each shifted by EXACTLY
+3 voxels along one numpy axis via np.roll:

    def_axisK = np.roll(ref, +3, axis=K)      (K = 0, 1, 2)

np.roll(+3, axis=K) means def[i_K] = ref[i_K - 3]: a feature at index p in the
reference appears at index p+3 in the deformed volume, i.e. the physical
displacement of features is +3 voxels along numpy axis K. Whichever DVC
displacement component (u,v,w) reports ~+3 identifies the mapping.

Also writes cfg JSONs for aldvc_headless (winsize [12,12,12], step [6,6,6]).
Paths inside cfgs are relative to the benchmark/ directory (MATLAB cwd).

Usage:  python make_calibration_pair.py
"""

import json
import os

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CFGS = os.path.join(HERE, "cfgs")


def make_blob_volume(n=64, sigma=2.0, seed=0):
    """Gaussian-filtered random noise, soft-thresholded, values in [0,1]."""
    rng = np.random.default_rng(seed)
    vol = gaussian_filter(rng.standard_normal((n, n, n)), sigma=sigma)
    vol = (vol - vol.min()) / (vol.max() - vol.min())          # -> [0,1]
    vol = np.clip((vol - 0.5) * 4.0 + 0.5, 0.0, 1.0)           # soft threshold
    return vol.astype(np.float64)


def main():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(CFGS, exist_ok=True)

    ref = make_blob_volume(n=64, sigma=2.0, seed=0)
    sio.savemat(os.path.join(DATA, "calib_ref.mat"), {"vol": ref})
    print(f"calib_ref.mat  shape={ref.shape}  min={ref.min():.3f} max={ref.max():.3f}")

    for axis in (0, 1, 2):
        dfm = np.roll(ref, +3, axis=axis)
        sio.savemat(os.path.join(DATA, f"calib_def_axis{axis}.mat"), {"vol": dfm})
        cfg = {
            "refFile": "data/calib_ref.mat",
            "defFile": f"data/calib_def_axis{axis}.mat",
            "winsize": [12, 12, 12],
            "winstepsize": [6, 6, 6],
            "outFile": f"data/calib_out_axis{axis}.mat",
        }
        cfg_path = os.path.join(CFGS, f"calib_axis{axis}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote calib_def_axis{axis}.mat and {cfg_path}")


if __name__ == "__main__":
    main()
