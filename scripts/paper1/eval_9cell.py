"""Nine-cell generalization study (prediction only): each of the three trained
solvers evaluated on a 3x3 grid of DVC images (bead radius x displacement range)
at a fixed 64^3 size. Each model's ID cell (matched bead+displacement) lies on
the diagonal; off-diagonal cells probe interpolation/extrapolation. Writes
per-(model,cell) raw-EPE mean/std to reports/phase1/nine_cell_epe.csv.

Usage:  python scripts/paper1/eval_9cell.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from src.core.raft_dvc import RAFTDVC  # noqa: E402

LOCAL = Path(r"C:\Zixiang_local_data\raft-dvc")
DATA = LOCAL / "_9cell"
DEV = torch.device("cuda")

MODELS = [  # (name, checkpoint, ID radius, ID disp)
    ("1/2", LOCAL / "paper1/phase1/paper1_axis2_1_2_at32/latest.pth", 2, "2-4"),
    ("1/4", LOCAL / "paper1/phase1/paper1_clean_1_4_r4md64_pw02/latest.pth", 4, "4-8"),
    ("1/8", LOCAL / "paper1/phase1/paper1_axis2_1_8_at128/latest.pth", 8, "8-16"),
]
BEADS = [2, 4, 8]
DISPS = [(2, 4), (4, 8), (8, 16)]


@torch.no_grad()
def cell_epe(model, cell_dir):
    epes = []
    for npz in sorted(cell_dir.glob("*.npz")):
        d = np.load(npz)
        gt = d["flow"].astype(np.float32)
        v0 = torch.from_numpy(d["I1"]).float()[None, None].to(DEV)
        v1 = torch.from_numpy(d["I2"]).float()[None, None].to(DEV)
        _, up = model(v0, v1, iters=12, test_mode=True)
        pred = up[0].cpu().numpy()
        epes.append(float(np.sqrt(((pred - gt) ** 2).sum(0)).mean()))
    return np.array(epes)


def main():
    rows = []
    for name, ck, id_r, id_d in MODELS:
        model, _ = RAFTDVC.load_checkpoint(str(ck), device=DEV)
        model.eval()
        for r in BEADS:
            for lo, hi in DISPS:
                cdir = DATA / f"d{lo}_{hi}" / f"r{r}_medium_size64" / "test"
                e = cell_epe(model, cdir)
                is_id = (r == id_r and f"{lo}-{hi}" == id_d)
                rows.append(dict(model=name, radius=r, disp=f"{lo}-{hi}",
                                 epe_mean=f"{e.mean():.4f}", epe_std=f"{e.std(ddof=1):.4f}",
                                 n=len(e), is_id=int(is_id)))
                print(f"{name}  r{r}  disp[{lo},{hi}]  EPE={e.mean():.3f}+/-{e.std(ddof=1):.3f}"
                      f"{'   <-- ID' if is_id else ''}", flush=True)
        del model; torch.cuda.empty_cache()
    out = _REPO / "reports" / "phase1" / "nine_cell_epe.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("saved:", out, flush=True)


if __name__ == "__main__":
    main()
