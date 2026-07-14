"""Paper-1 benchmark suite orchestrator (classical DVC baselines).

Scenario datasets (generated with scripts/generate_phase1_dataset.py, seeded)
live under G:\\Zixiang_local_data\\raft-dvc\\_bench\\S1..S7. Classical methods
are the headless MATLAB wrappers in traditional_DIC_codes/benchmark
(aldvc_headless -> U_local + U_aldvc; globaldvc_headless -> U_global).
Axis mapping npz<->MATLAB is identity, coordinatesFEM is 1-based, U is
interleaved [u1;v1;w1;u2;...] (see benchmark/HEADLESS_README.md).

Subcommands
-----------
gen-cfgs        Write per-sample MATLAB cfg JSONs into
                traditional_DIC_codes/benchmark/cfgs/bench/<Sx>/.
run-classical   Convert npz pairs to .mat and run both MATLAB wrappers per
                sample, sequentially (one MATLAB at a time). Resumable: skips
                samples whose output .mat already exists.
eval            Mean/median EPE at nodes vs the npz 'flow' GT (U_dvc ~ +flow,
                measured convention) -> reports/phase1/bench_classical.csv.
fig-matrix      STUB (needs learned-method results).
fig-replay      STUB (needs learned-method results).

Usage
-----
python scripts/paper1/benchmark_suite.py gen-cfgs
python scripts/paper1/benchmark_suite.py run-classical [--scenario S1] [--limit N]
python scripts/paper1/benchmark_suite.py eval [--scenario S1]

TODO: S7 (512^3, n=3) classical runs are EXCLUDED from the default
run-classical order (~25-56 min per sample per method, ~4 h total). Run
overnight with: python scripts/paper1/benchmark_suite.py run-classical --scenario S7
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
MATLAB_EXE = r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"
MATLAB_TIMEOUT_S = 6 * 3600
BENCH_MATLAB = REPO / "traditional_DIC_codes" / "benchmark"
CFG_ROOT = BENCH_MATLAB / "cfgs" / "bench"
BENCH_DATA = Path(r"G:\Zixiang_local_data\raft-dvc\_bench")
S7_SAMPLE0 = Path(r"G:\Zixiang_local_data\raft-dvc\_bench512"
                  r"\r8_medium_size512\test\sample_00000.npz")
CSV_OUT = REPO / "reports" / "phase1" / "bench_classical.csv"

sys.path.insert(0, str(BENCH_MATLAB))
from npz_to_mat import npz_to_mat  # noqa: E402


@dataclass(frozen=True)
class BeadParams:
    """Per-bead-size classical parameters (HEADLESS_README.md table)."""
    aldvc_win: int      # ALDVC subset size
    aldvc_step: int     # ALDVC subset step
    global_step: int    # Global-DVC finite element size
    alpha: float        # Global-DVC regularization


@dataclass(frozen=True)
class Scenario:
    name: str
    config_dir: str     # dataset folder name under <root>/<name>/
    bead: str           # key into BEAD_PARAMS
    n_samples: int
    note: str = ""

    def npz_path(self, idx: int) -> Path:
        if self.name == "S7" and idx == 0:
            return S7_SAMPLE0  # pre-existing sample, kept in place
        return (BENCH_DATA / self.name / self.config_dir / "test"
                / f"sample_{idx:05d}.npz")

    def work_dir(self, kind: str) -> Path:
        """kind in {'mat', 'out', 'logs'} -- scratch dirs on G: (large files)."""
        return BENCH_DATA / self.name / kind


BEAD_PARAMS = {
    "r2": BeadParams(aldvc_win=12, aldvc_step=6, global_step=8, alpha=0.8),
    "r4": BeadParams(aldvc_win=16, aldvc_step=8, global_step=10, alpha=1.0),
    "r8": BeadParams(aldvc_win=24, aldvc_step=12, global_step=12, alpha=1.2),
}

SCENARIOS = {
    "S1": Scenario("S1", "r2_medium_size64", "r2", 20, "core r2 U[2,4]"),
    "S2": Scenario("S2", "r4_medium_size64", "r4", 20, "core r4 U[4,8]"),
    "S3": Scenario("S3", "r8_medium_size64", "r8", 20,
                   "core r8 U[8,16]; win24@64^3 leaves few nodes (finding)"),
    "S4": Scenario("S4", "r2_medium_size64", "r2", 20, "stress r2 U[8,16]"),
    "S5": Scenario("S5", "r8_medium_size64", "r8", 20,
                   "stress r8 U[2,4]; win24@64^3 leaves few nodes (finding)"),
    "S6": Scenario("S6", "r4_medium_size128", "r4", 20, "deployment 128^3"),
    "S7": Scenario("S7", "r8_medium_size512", "r8", 3,
                   "deployment 512^3 demo (n=3, run overnight)"),
}
# S7 excluded by default: ~25-56 min per sample per method (see module TODO).
DEFAULT_RUN_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]

# method -> (output .mat suffix, U variable, timing stage fields to sum)
METHODS = {
    "local": ("aldvc", "U_local", ("integer_search_s", "local_icgn_s")),
    "aldvc": ("aldvc", "U_aldvc",
              ("integer_search_s", "local_icgn_s", "global_admm_s")),
    "global": ("global", "U_global",
               ("integer_search_s", "img_gradient_s", "global_icgn_s")),
}


def _mpath(p: Path) -> str:
    """Absolute forward-slash path for MATLAB cfg JSONs."""
    return str(p).replace("\\", "/")


def _select(scenario: str | None, default: list[str] | None = None) -> list[str]:
    if scenario is not None:
        if scenario not in SCENARIOS:
            raise SystemExit(f"unknown scenario {scenario!r}; "
                             f"choose from {sorted(SCENARIOS)}")
        return [scenario]
    return default if default is not None else list(SCENARIOS)


# ---------------------------------------------------------------------------
# gen-cfgs
# ---------------------------------------------------------------------------

def cmd_gen_cfgs(args: argparse.Namespace) -> None:
    n_written = 0
    for name in _select(args.scenario):
        sc = SCENARIOS[name]
        bp = BEAD_PARAMS[sc.bead]
        cfg_dir = CFG_ROOT / sc.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        mat_dir, out_dir = sc.work_dir("mat"), sc.work_dir("out")
        for i in range(sc.n_samples):
            stem = f"sample_{i:05d}"
            ref = _mpath(mat_dir / f"{stem}_ref.mat")
            dfm = _mpath(mat_dir / f"{stem}_def.mat")
            aldvc_cfg = {
                "refFile": ref, "defFile": dfm,
                "winsize": [bp.aldvc_win] * 3,
                "winstepsize": [bp.aldvc_step] * 3,
                "outFile": _mpath(out_dir / f"{stem}_aldvc.mat"),
            }
            global_cfg = {
                "refFile": ref, "defFile": dfm,
                "winstepsize": [bp.global_step] * 3,
                "alpha": bp.alpha,
                "outFile": _mpath(out_dir / f"{stem}_global.mat"),
            }
            for tag, cfg in (("aldvc", aldvc_cfg), ("global", global_cfg)):
                path = cfg_dir / f"{stem}_{tag}.json"
                path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                n_written += 1
        print(f"[{sc.name}] {sc.n_samples} samples x 2 cfgs -> {cfg_dir}")
    print(f"gen-cfgs: wrote {n_written} cfg files")


# ---------------------------------------------------------------------------
# run-classical
# ---------------------------------------------------------------------------

def _run_matlab(func: str, cfg_path: Path, log_path: Path) -> tuple[float, int]:
    """Run one headless wrapper via matlab -batch; returns (wall_s, returncode)."""
    cmd = [MATLAB_EXE, "-batch", f"{func}('{_mpath(cfg_path)}')"]
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.run(cmd, cwd=str(BENCH_MATLAB), stdout=log,
                                  stderr=subprocess.STDOUT,
                                  timeout=MATLAB_TIMEOUT_S)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\n[benchmark_suite] TIMEOUT after {MATLAB_TIMEOUT_S}s\n")
            rc = -9
    return time.perf_counter() - t0, rc


def _log_tail(log_path: Path, n: int = 12) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(f"    | {ln}" for ln in lines[-n:])
    except OSError:
        return "    | <no log>"


def cmd_run_classical(args: argparse.Namespace) -> None:
    failures: list[str] = []
    for name in _select(args.scenario, DEFAULT_RUN_ORDER):
        sc = SCENARIOS[name]
        mat_dir, out_dir, log_dir = (sc.work_dir(k) for k in ("mat", "out", "logs"))
        for d in (mat_dir, out_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)
        n_run = sc.n_samples if args.limit is None else min(args.limit, sc.n_samples)
        print(f"=== [{sc.name}] {sc.config_dir}  ({n_run} samples)  {sc.note} ===",
              flush=True)
        for i in range(n_run):
            stem = f"sample_{i:05d}"
            npz = sc.npz_path(i)
            if not npz.is_file():
                failures.append(f"{sc.name}/{stem}: npz missing ({npz})")
                print(f"  [{stem}] MISSING npz -- skipped", flush=True)
                continue
            ref_mat = mat_dir / f"{stem}_ref.mat"
            if not ref_mat.is_file() or not (mat_dir / f"{stem}_def.mat").is_file():
                npz_to_mat(str(npz), str(mat_dir), prefix=stem)
            for tag, func in (("aldvc", "aldvc_headless"),
                              ("global", "globaldvc_headless")):
                out_mat = out_dir / f"{stem}_{tag}.mat"
                if out_mat.is_file():
                    print(f"  [{stem}] {tag}: exists, skipped", flush=True)
                    continue
                cfg_path = CFG_ROOT / sc.name / f"{stem}_{tag}.json"
                if not cfg_path.is_file():
                    raise SystemExit(f"cfg missing: {cfg_path} (run gen-cfgs first)")
                log_path = log_dir / f"{stem}_{tag}.log"
                wall_s, rc = _run_matlab(func, cfg_path, log_path)
                ok = rc == 0 and out_mat.is_file()
                sidecar = out_dir / f"{stem}_{tag}_runtime.json"
                sidecar.write_text(json.dumps(
                    {"wall_s": round(wall_s, 2), "returncode": rc, "ok": ok}),
                    encoding="utf-8")
                status = "ok" if ok else f"FAILED (rc={rc})"
                print(f"  [{stem}] {tag}: {status}  wall={wall_s:.1f}s", flush=True)
                if not ok:
                    failures.append(f"{sc.name}/{stem}/{tag} rc={rc}")
                    print(_log_tail(log_path), flush=True)
    print(f"\nrun-classical done. failures: {len(failures)}")
    for f in failures:
        print(f"  FAIL {f}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _timing_struct(m: dict) -> dict[str, float]:
    t = m["timing"]
    return {k: float(np.squeeze(t[k][0, 0])) for k in t.dtype.names}


def _epe_at_nodes(u_flat: np.ndarray, coords: np.ndarray,
                  flow: np.ndarray) -> np.ndarray:
    """Per-node EPE. Measured convention: U_dvc(X) ~= +flow(X), identity axes,
    coordinatesFEM 1-based (numpy position = coords - 1)."""
    u = np.asarray(u_flat, dtype=np.float64).ravel()
    u_nodes = np.stack([u[0::3], u[1::3], u[2::3]], axis=1)  # (N, 3)
    pos = np.asarray(coords, dtype=np.float64) - 1.0
    idx = np.rint(pos).astype(np.int64)
    if not np.allclose(pos, idx, atol=1e-6):
        raise ValueError("non-integer node coordinates; expected ndgrid nodes")
    gt = flow[:, idx[:, 0], idx[:, 1], idx[:, 2]].T  # (N, 3)
    epe = np.linalg.norm(u_nodes - gt, axis=1)
    # Methods can emit NaN at isolated nodes (e.g. ALDVC ADMM FD step on a
    # starved 3x3x3 node grid) -- evaluate over the finite nodes only; the
    # reduced count is disclosed via n_nodes.
    return epe[np.isfinite(epe)]


def cmd_eval(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for name in _select(args.scenario):
        sc = SCENARIOS[name]
        out_dir = sc.work_dir("out")
        for i in range(sc.n_samples):
            stem = f"sample_{i:05d}"
            npz = sc.npz_path(i)
            mats = {tag: out_dir / f"{stem}_{tag}.mat" for tag in ("aldvc", "global")}
            if not any(p.is_file() for p in mats.values()):
                continue  # not run yet (e.g. S7)
            flow = np.asarray(np.load(npz)["flow"], dtype=np.float64)
            for method, (tag, var, stages) in METHODS.items():
                if not mats[tag].is_file():
                    print(f"  [{sc.name}/{stem}] {method}: output missing, skipped")
                    continue
                m = sio.loadmat(str(mats[tag]))
                timing = _timing_struct(m)
                epe = _epe_at_nodes(m[var], m["coordinatesFEM"], flow)
                if epe.size == 0:
                    print(f"  [{sc.name}/{stem}] {method}: all nodes NaN, skipped")
                    continue
                rows.append({
                    "scenario": sc.name, "sample": i, "method": method,
                    "epe_mean": round(float(epe.mean()), 4),
                    "epe_median": round(float(np.median(epe)), 4),
                    "n_nodes": int(epe.size),
                    "runtime_s": round(sum(timing[s] for s in stages), 2),
                })
            del flow
    if not rows:
        raise SystemExit("eval: no outputs found -- run run-classical first")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"eval: wrote {len(rows)} rows -> {CSV_OUT}\n")

    # Per-scenario summary (mean +/- s.d. of per-sample mean EPE).
    print(f"{'scenario':<9}{'method':<8}{'n':>3}  {'EPE mean+/-sd':<18}"
          f"{'median(med)':>12}{'nodes':>7}{'runtime_s':>10}")
    for name in sorted({r["scenario"] for r in rows}):
        for method in METHODS:
            sub = [r for r in rows
                   if r["scenario"] == name and r["method"] == method]
            if not sub:
                continue
            e = np.array([r["epe_mean"] for r in sub])
            med = np.median([r["epe_median"] for r in sub])
            rt = np.mean([r["runtime_s"] for r in sub])
            nodes = sub[0]["n_nodes"]
            print(f"{name:<9}{method:<8}{len(sub):>3}  "
                  f"{e.mean():.3f} +/- {e.std():.3f}   "
                  f"{med:>12.3f}{nodes:>7}{rt:>10.1f}")


# ---------------------------------------------------------------------------
# figure stubs (learned-method results not in yet)
# ---------------------------------------------------------------------------

def cmd_fig_matrix(args: argparse.Namespace) -> None:
    """STUB -- planned: scenario x method accuracy matrix (BASELINE_PLAN.md).

    Heatmap-style table: rows = S1..S7, columns = 3 RAFT-DVC arms + 5
    baselines (local, global, ALDVC, VolRAFT zero-shot, VolRAFT retrained);
    cell = mean +/- s.d. EPE over n samples; the rule-selected arm per
    scenario is highlighted. Requires learned-method results.
    """
    raise SystemExit("fig-matrix: stub -- learned-method results not in yet "
                     "(classical numbers: reports/phase1/bench_classical.csv)")


def cmd_fig_replay(args: argparse.Namespace) -> None:
    """STUB -- planned: full-page replay figures for S1/S2/S3 (BASELINE_PLAN.md).

    Per scenario: 5 columns (GT, rule-selected arm, VolRAFT-retrained, 2
    strongest classical) x 6 rows (U, U-err, V, V-err, W, W-err), signed
    error, shared diverging scale per row; sample = median-EPE-of-ours.
    Requires learned-method results.
    """
    raise SystemExit("fig-replay: stub -- learned-method results not in yet")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-cfgs", help="write MATLAB cfg JSONs")
    p.add_argument("--scenario", default=None)
    p.set_defaults(fn=cmd_gen_cfgs)

    p = sub.add_parser("run-classical",
                       help="run MATLAB wrappers per sample (resumable); "
                            "default order S1..S6, S7 only via --scenario S7")
    p.add_argument("--scenario", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="max samples per scenario (smoke testing)")
    p.set_defaults(fn=cmd_run_classical)

    p = sub.add_parser("eval", help="EPE vs npz flow GT -> bench_classical.csv")
    p.add_argument("--scenario", default=None)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("fig-matrix", help="STUB: scenario x method matrix")
    p.set_defaults(fn=cmd_fig_matrix)

    p = sub.add_parser("fig-replay", help="STUB: S1-S3 replay figures")
    p.set_defaults(fn=cmd_fig_replay)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
