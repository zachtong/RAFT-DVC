"""Ruler-law money figure from the fixed-sampler retrained arms.
Reads reports/phase1/ruler_law_fixed.csv."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

REPO = Path(__file__).resolve().parents[2]
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10.5,
})
OK = {2: "#56B4E9", 4: "#009E73", 8: "#E69F00"}

rows = list(csv.DictReader(open(REPO / "reports/phase1/ruler_law_fixed.csv")))
f = np.array([int(r["f"]) for r in rows])
mu = np.array([float(r["rEPE_mean"]) for r in rows])
sd = np.array([float(r["rEPE_std"]) for r in rows])
fe = mu / f
fe_sd = sd / f
c = fe.mean()

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)

# (a) feature-EPE vs f: flat
ax = axes[0]
ax.axhline(c, color="gray", ls="--", lw=1.2)
ax.text(2.15, c * 1.06, f"$c \\approx {c:.3f}$", fontsize=10, color="gray")
for fi, m, s in zip(f, fe, fe_sd):
    ax.errorbar(fi, m, yerr=s, fmt="o", ms=8, capsize=4, color=OK[fi],
                mec="k", mew=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8], ["2", "4", "8"])
ax.set_ylim(0, 0.032)
ax.set_xlabel("downsampling factor $f$")
ax.set_ylabel("feature-EPE (feature voxel)")
ax.set_title("a   error in feature-volume units is constant")
ax.xaxis.set_minor_formatter(NullFormatter())

# (b) raw-EPE vs f: proportional
ax = axes[1]
ff = np.linspace(1.6, 10, 50)
ax.plot(ff, c * ff, "--", color="gray", lw=1.2, label=f"$c\\,f$,  $c={c:.3f}$")
for fi, m, s in zip(f, mu, sd):
    ax.errorbar(fi, m, yerr=s, fmt="o", ms=8, capsize=4, color=OK[fi],
                mec="k", mew=0.6, label=f"1/{fi} solver")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks([2, 4, 8], ["2", "4", "8"])
ax.set_yticks([0.03125, 0.0625, 0.125], ["0.031", "0.062", "0.125"])
ax.set_xlabel("downsampling factor $f$")
ax.set_ylabel("raw-EPE (voxel)")
ax.set_title("b   error in raw-volume units scales with $f$")
ax.legend(fontsize=8.5, loc="upper left")
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

for ax in axes:
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=4)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=4)

out = REPO / "paper_1" / "Figures" / "fig_ruler_law.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
print("saved:", out, " c =", round(c, 4))
