"""Read the axis-calibration outputs (data/calib_out_axis{0,1,2}.mat) produced
by aldvc_headless and report, for each numpy-axis shift, the median value of
each DVC displacement component (u,v,w) of U_local and U_aldvc.

The component reporting ~+3 for the +3-voxel np.roll along numpy axis K
establishes the numpy-axis <-> (u,v,w) mapping.

Usage:  python check_calibration.py
"""

import os

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def component_medians(U):
    U = np.asarray(U).ravel()
    return [float(np.median(U[0::3])), float(np.median(U[1::3])), float(np.median(U[2::3]))]


def main():
    comp_names = ["u (comp1)", "v (comp2)", "w (comp3)"]
    mapping = {}
    for axis in (0, 1, 2):
        path = os.path.join(DATA, f"calib_out_axis{axis}.mat")
        if not os.path.isfile(path):
            print(f"MISSING: {path}")
            continue
        m = sio.loadmat(path)
        print(f"\n=== np.roll(ref, +3, axis={axis}) ===")
        for name in ("U_local", "U_aldvc"):
            med = component_medians(m[name])
            print(f"  {name:8s} medians  u={med[0]:+.3f}  v={med[1]:+.3f}  w={med[2]:+.3f}")
        med = component_medians(m["U_aldvc"])
        k = int(np.argmax(np.abs(med)))
        mapping[axis] = (comp_names[k], med[k])

    print("\n=== Inferred mapping (component with |median| max, from U_aldvc) ===")
    for axis, (comp, val) in mapping.items():
        print(f"  numpy axis {axis}  ->  {comp}   (median {val:+.3f}, expected +3)")


if __name__ == "__main__":
    main()
