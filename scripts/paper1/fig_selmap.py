"""Selection-rule map (paper Fig: fig:selmap).

The (bead diameter) x (displacement magnitude) plane, colored by the solver
the rule deploys: smallest downsampling factor f whose resolvability
constraint (bead diameter >= f) holds and whose VALIDATED operating band
(the trained displacement range, [f, 2f] under the matched design) contains
the displacement. Dashed lines mark each arm's ARCHITECTURAL reach
(r * 2^(L-1) * f = 8 f), extendable by retraining with a wider range.
Benchmark scenarios S1-S5 are marked.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "Arial", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10.5,
})

OK = {2: "#56B4E9", 4: "#009E73", 8: "#E69F00"}   # Okabe-Ito per arm
GRAY = "#d9d9d9"

# --- rule ---------------------------------------------------------------
def rule(d, u):
    """Trained-toolkit deployment rule: smallest f that resolves the bead and
    whose validated band contains |u|.  Returns f or 0 (no trained arm)."""
    if u <= 4.0:
        f = 2
    elif u <= 8.0:
        f = 4
    elif u <= 16.0:
        f = 8
    else:
        return 0
    return f if d >= f else 0


d_edges = np.logspace(np.log2(1.5), np.log2(40), 240, base=2)
u_edges = np.logspace(np.log2(0.8), np.log2(40), 240, base=2)
D, U = np.meshgrid(d_edges, u_edges, indexing="ij")
Z = np.vectorize(rule)(D, U)

fig, ax = plt.subplots(figsize=(5.4, 4.6))
cmap = ListedColormap([GRAY, OK[2], OK[4], OK[8]])
zi = np.zeros_like(Z)
for k, f in enumerate((2, 4, 8), start=1):
    zi[Z == f] = k
ax.pcolormesh(D, U, zi, cmap=cmap, vmin=0, vmax=3, shading="auto",
              rasterized=True, alpha=0.85)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)

# architectural reach lines (8f), dashed, per arm color
for f in (2, 4, 8):
    ax.axhline(8 * f, color=OK[f], ls="--", lw=1.3)
ax.text(22, 16 * 1.08, "architectural reach 8f per arm\n"
        "(extendable by retraining)", fontsize=8, color="#555", ha="center")

# resolvability boundaries (vertical, at d = f)
for f in (4, 8):
    lo = {4: 4.0, 8: 8.0}[f]
    ax.plot([f, f], [lo, 2 * f], color="k", lw=1.0)

# scenario markers: (d, u_mid, label)
scen = [(4, 3, "S1"), (8, 6, "S2"), (16, 12, "S3"), (4, 12, "S4"), (16, 3, "S5")]
for d, u, lab in scen:
    ax.plot(d, u, "o", ms=7, mfc="white", mec="k", mew=1.4, zorder=5)
    ax.annotate(lab, (d, u), textcoords="offset points", xytext=(7, 4),
                fontsize=9, fontweight="bold")

ax.set_xlabel("bead diameter (voxels)")
ax.set_ylabel("displacement magnitude $\\|\\mathbf{u}\\|$ (voxels)")
ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
ax.set_yticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
ax.set_xlim(1.5, 40)
ax.set_ylim(0.8, 40)

handles = [Patch(fc=OK[2], alpha=0.85, label="deploy 1/2  (f = 2)"),
           Patch(fc=OK[4], alpha=0.85, label="deploy 1/4  (f = 4)"),
           Patch(fc=OK[8], alpha=0.85, label="deploy 1/8  (f = 8)"),
           Patch(fc=GRAY, label="no trained solver")]
ax.legend(handles=handles, loc="upper left", fontsize=8.2, framealpha=0.95)

# axis arrows
ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False, ms=4)
ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False, ms=4)

out = Path(__file__).resolve().parents[2] / "paper_1" / "Figures" / "fig_selmap.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
print("saved:", out)
