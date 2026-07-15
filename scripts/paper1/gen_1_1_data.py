"""Generate the matched 4th ruler-law point: the f=1 (1/1) arm's training data.

Design (matched to the 32/64/128 benchmark arms, which all hold feature volume
16^3 constant): size 16, radius 1, feature_map_size 16 -> downsample_factor 1;
input displacement [1,2] (= the arms' shared fm-disp band, here at f=1 so raw ==
fm); density 38.4/1000 -> 38.4*16^3/1000 = 157 beads (the arms' matched count).

In feature units this is IDENTICAL to the other three arms (16^3 grid, bead = 1
feature-voxel, disp [1,2] feature-voxel, 157 beads); only f differs.

Layout: <root>/r1_medium_size16/{train,val}/sample_{i:05d}.npz  (I1,I2,flow,metadata)
matching the trainer's --data-config r1_medium_size16 convention.
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

SIZE, RADIUS, DENSITY, FM = 16, 1, 38.4, 16
DISP = (1.0, 2.0)
BASE_SEED = 771_000_000       # distinct namespace for the 1/1 arm
SPLIT_OFF = {"train": 0, "val": 10_000_000}


def gen(root: Path, n_train: int, n_val: int, verify_only: bool):
    g = Phase1SampleGenerator(
        size=SIZE, radius=RADIUS, density_per_1000=DENSITY,
        feature_map_size=FM, input_disp_min=DISP[0], input_disp_max=DISP[1])
    print(f"downsample_factor = {g.downsample_factor} (want 1.0); "
          f"fm-disp band = [{g.fm_min:.2f}, {g.fm_max:.2f}] (want [1,2])",
          flush=True)
    cfg = f"r{RADIUS}_medium_size{SIZE}"
    for split, n in (("train", n_train), ("val", n_val)):
        outdir = root / cfg / split
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            seed = BASE_SEED + SPLIT_OFF[split] + i
            s = g.generate(seed)
            I1, flow = s["I1"], s["flow"]
            if verify_only:
                nz = float((I1 > I1.mean() + I1.std()).mean())
                print(f"  {split}/{i}: I1 shape={I1.shape} range=[{I1.min():.0f},"
                      f"{I1.max():.0f}] bright-frac={nz:.3f}  "
                      f"max|u|={np.abs(flow).max():.2f}  "
                      f"beads_meta={s['metadata'].get('num_beads_placed')}",
                      flush=True)
                continue
            np.savez_compressed(outdir / f"sample_{i:05d}.npz",
                                I1=I1.astype(np.float32),
                                I2=s["I2"].astype(np.float32),
                                flow=flow.astype(np.float32),
                                metadata=s["metadata"])
        if not verify_only:
            print(f"[{split}] wrote {n} -> {outdir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Zixiang_local_data\raft-dvc\data_paper1_axis2")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--verify", action="store_true", help="generate 3, print stats, no save")
    a = ap.parse_args()
    if a.verify:
        gen(Path(a.root), 3, 0, verify_only=True)
    else:
        gen(Path(a.root), a.train, a.val, verify_only=False)
