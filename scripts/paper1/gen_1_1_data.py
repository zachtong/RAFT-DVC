"""Generate training data for the ruler-law f=1 (1/1) experiments.

Two presets (select with --preset):
  r1_16  -- STRICT matched 4th point: size 16, radius 1, density 38.4,
            disp [1,2]; feature volume 16^3 (== the 32/64/128 arms), 157 beads.
            Beads sit at the resolvability edge (r1 = diam ~2 vox).
  r2_32  -- CLEAN f=1-vs-f=2 on the 1/2 arm's data distribution: size 32,
            radius 2, density 4.8, disp [2,4]; resolvable beads. The 1/1 model
            (encoder f=1) sees feature 32^3 on this data.

All params are also overridable individually (--size --radius --density
--disp-min --disp-max). feature_map_size is fixed to 16 (only affects metadata
in input-disp mode; the generated I1/I2/flow depend solely on input disp).

Layout: <root>/r{radius}_medium_size{size}/{train,val}/sample_{i:05d}.npz
matching the trainer's --data-config r{radius}_medium_size{size} convention.
Idempotent: skips a split whose files already exist (safe to re-run in SLURM).
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "data_generation"))
from modules.sample_generator import Phase1SampleGenerator  # noqa: E402

PRESETS = {
    "r1_16": dict(size=16, radius=1, density=38.4, disp=(1.0, 2.0)),
    "r2_32": dict(size=32, radius=2, density=4.8,  disp=(2.0, 4.0)),
}
FM = 16
BASE_SEED = 771_000_000
SPLIT_OFF = {"train": 0, "val": 10_000_000}


def gen(root, size, radius, density, disp, n_train, n_val, verify):
    g = Phase1SampleGenerator(
        size=size, radius=radius, density_per_1000=density,
        feature_map_size=FM, input_disp_min=disp[0], input_disp_max=disp[1])
    print(f"gen r{radius}@{size}: downsample={g.downsample_factor}, "
          f"fm-disp=[{g.fm_min:.2f},{g.fm_max:.2f}], disp={disp}", flush=True)
    cfg = f"r{radius}_medium_size{size}"
    for split, n in (("train", n_train), ("val", n_val)):
        outdir = root / cfg / split
        outdir.mkdir(parents=True, exist_ok=True)
        existing = len(list(outdir.glob("sample_*.npz")))
        if existing >= n and not verify:
            print(f"  [{split}] {existing} already present -- skip", flush=True)
            continue
        for i in range(n):
            fp = outdir / f"sample_{i:05d}.npz"
            if fp.exists() and not verify:
                continue
            s = g.generate(BASE_SEED + SPLIT_OFF[split] + i)
            if verify:
                print(f"  {split}/{i}: shape={s['I1'].shape} "
                      f"max|u|={np.abs(s['flow']).max():.2f}", flush=True)
                if i >= 2:
                    return
                continue
            np.savez_compressed(fp, I1=s["I1"].astype(np.float32),
                                I2=s["I2"].astype(np.float32),
                                flow=s["flow"].astype(np.float32),
                                metadata=s["metadata"])
        if not verify:
            print(f"  [{split}] wrote {n} -> {outdir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="r1_16")
    ap.add_argument("--root", default=r"C:\Zixiang_local_data\raft-dvc\data_paper1_axis2")
    ap.add_argument("--size", type=int)
    ap.add_argument("--radius", type=int)
    ap.add_argument("--density", type=float)
    ap.add_argument("--disp-min", type=float)
    ap.add_argument("--disp-max", type=float)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    p = PRESETS[a.preset]
    size = a.size or p["size"]
    radius = a.radius or p["radius"]
    density = a.density or p["density"]
    disp = (a.disp_min or p["disp"][0], a.disp_max or p["disp"][1])
    gen(Path(a.root), size, radius, density, disp,
        3 if a.verify else a.train, 0 if a.verify else a.val, a.verify)
