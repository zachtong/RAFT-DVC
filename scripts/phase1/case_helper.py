"""Read a case_matrix.yaml entry and emit shell-evalable variables.

Used by SLURM array job scripts to extract per-task case parameters.

Usage in shell::

    eval $(python scripts/phase1/case_helper.py \\
        --matrix configs/phase1/case_matrix.yaml \\
        --index $SLURM_ARRAY_TASK_ID)
    # -> exports NAME, DATA, MODEL, BATCH

Other modes::

    --list        Print all cases (one per line, ``IDX  NAME  DATA  MODEL  BATCH``)
    --count       Print the total number of cases (useful for ``--array=0-$(...)``)
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import yaml


def load_cases(matrix_path: Path) -> list:
    with open(matrix_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cases = data.get("cases", [])
    if not cases:
        raise SystemExit(f"[case_helper] No 'cases' key (or empty) in {matrix_path}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True,
                        help="Path to case_matrix.yaml")
    parser.add_argument("--index", type=int, default=None,
                        help="Zero-based case index to emit as shell vars")
    parser.add_argument("--list", action="store_true",
                        help="Print all cases as a human-readable table")
    parser.add_argument("--count", action="store_true",
                        help="Print only the case count (integer)")
    args = parser.parse_args()

    if not args.matrix.exists():
        raise SystemExit(f"[case_helper] Matrix not found: {args.matrix}")

    cases = load_cases(args.matrix)

    if args.count:
        print(len(cases))
        return

    if args.list:
        print(f"{'idx':>4}  {'name':<22}  {'data':<22}  {'batch':>5}  model")
        for i, c in enumerate(cases):
            print(
                f"{i:>4}  {c['name']:<22}  {c['data']:<22}  "
                f"{c['batch']:>5}  {c['model']}"
            )
        return

    if args.index is None:
        raise SystemExit("[case_helper] Specify --index or --list or --count")

    if args.index < 0 or args.index >= len(cases):
        raise SystemExit(
            f"[case_helper] index {args.index} out of range [0, {len(cases) - 1}]"
        )

    case = cases[args.index]
    # Emit shell-evalable assignments (quote paths/names that may contain spaces).
    print(f"NAME={shlex.quote(str(case['name']))}")
    print(f"DATA={shlex.quote(str(case['data']))}")
    print(f"MODEL={shlex.quote(str(case['model']))}")
    print(f"BATCH={int(case['batch'])}")


if __name__ == "__main__":
    main()
