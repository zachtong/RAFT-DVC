"""Profile RAFT-DVC peak GPU memory across (encoder, size, batch, iters, corr_impl) configs.

Useful for planning training runs: which batch sizes fit, where OTF gives
meaningful memory savings, and how memory scales with input size + architecture.

Each profile run does:
  1. Build the model from a chosen yaml (overriding iters + corr_impl).
  2. Allocate (batch, 1, size, size, size) input tensors.
  3. Run forward + dummy backward (so backward activations are exercised).
  4. Record torch.cuda.max_memory_allocated() and max_memory_reserved().

Results written as CSV + pretty stdout table.  Optionally generates a PDF
with per-(size, corr_impl) heatmap (encoder x batch -> peak memory).

Usage:
    # Default sweep -- 3 encoders x 3 sizes x 6 batches x standard+OTF
    python scripts/profile_memory.py

    # Targeted sweep matching paper-1 use case
    python scripts/profile_memory.py \\
        --encoders 1_8_p4_r4 1_4_p3_r4 1_2_p2_r4 \\
        --sizes 64 \\
        --batches 1 4 8 12 16 24 \\
        --corr-impls standard on_the_fly \\
        --output reports/mem_profile_paper1.csv \\
        --pdf reports/mem_profile_paper1.pdf

    # FP32 sweep for AMP comparison
    python scripts/profile_memory.py --no-amp --output reports/mem_profile_fp32.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import RAFTDVC, RAFTDVCConfig  # noqa: E402


def load_arch_config(encoder_name: str, iters: int, corr_impl: str) -> RAFTDVCConfig:
    """Load yaml at configs/models/raft_dvc_<encoder_name>.yaml + override iters/corr_impl."""
    yaml_path = PROJECT_ROOT / "configs" / "models" / f"raft_dvc_{encoder_name}.yaml"
    with open(yaml_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    arch = data["architecture"]
    cfg = RAFTDVCConfig.from_dict(arch)
    cfg.iters = iters
    cfg.corr_impl = corr_impl
    return cfg


def profile_one(
    encoder_name: str,
    input_size: int,
    batch: int,
    iters: int,
    mixed_precision: bool,
    corr_impl: str,
    device: str = "cuda",
) -> dict:
    """Profile peak memory for one config.  Returns dict with status + numbers."""
    out = {
        "encoder": encoder_name,
        "size": input_size,
        "batch": batch,
        "iters": iters,
        "amp": mixed_precision,
        "corr_impl": corr_impl,
        "status": "?",
        "params_M": float("nan"),
        "alloc_GB": float("nan"),
        "reserved_GB": float("nan"),
        "error": "",
    }

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = load_arch_config(encoder_name, iters, corr_impl)
        model = RAFTDVC(cfg).to(device)
        model.train()
        out["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6

        vol1 = torch.randn(batch, 1, input_size, input_size, input_size, device=device)
        vol2 = torch.randn(batch, 1, input_size, input_size, input_size, device=device)
        target = torch.randn(batch, 3, input_size, input_size, input_size, device=device)

        if mixed_precision:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                preds = model(vol1, vol2)
                loss = sum((p - target).abs().mean() for p in preds)
        else:
            preds = model(vol1, vol2)
            loss = sum((p - target).abs().mean() for p in preds)

        loss.backward()

        out["alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
        out["reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
        out["status"] = "OK"

        del model, vol1, vol2, target, preds, loss

    except torch.cuda.OutOfMemoryError as exc:
        out["status"] = "OOM"
        out["error"] = str(exc).split("\n")[0][:160]
        try:
            out["alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
            out["reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
        except Exception:
            pass
    except Exception as exc:
        out["status"] = "ERR"
        out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    torch.cuda.empty_cache()
    return out


def write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "encoder", "size", "batch", "iters", "amp", "corr_impl",
        "status", "params_M", "alloc_GB", "reserved_GB", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def print_table(results: list[dict], gpu_total_GB: float) -> None:
    print()
    print("=" * 110)
    header = (
        f"{'encoder':<18} {'size':>4} {'bs':>3} {'it':>3} "
        f"{'corr':<11} {'amp':>5} {'status':<6} "
        f"{'params':>7} {'alloc_GB':>9} {'reserved':>9} {'%total':>6}"
    )
    print(header)
    print("-" * 110)
    for r in results:
        pct = (r["reserved_GB"] / gpu_total_GB * 100) if r["reserved_GB"] == r["reserved_GB"] else float("nan")
        print(
            f"{r['encoder']:<18} {r['size']:>4} {r['batch']:>3} {r['iters']:>3} "
            f"{r['corr_impl']:<11} {str(r['amp']):>5} {r['status']:<6} "
            f"{r['params_M']:>6.2f}M {r['alloc_GB']:>9.2f} {r['reserved_GB']:>9.2f} {pct:>5.0f}%"
        )
    print("=" * 110)


def write_pdf(results: list[dict], path: Path, gpu_name: str, gpu_total_GB: float) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import numpy as np  # noqa: F401
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print(f"[skip pdf] matplotlib not installed")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    encoders = sorted({r["encoder"] for r in results})
    sizes = sorted({r["size"] for r in results})
    batches = sorted({r["batch"] for r in results})
    corr_impls = sorted({r["corr_impl"] for r in results})
    amps = sorted({r["amp"] for r in results})

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for amp in amps:
            for size in sizes:
                for corr_impl in corr_impls:
                    grid = np.full((len(encoders), len(batches)), np.nan)
                    statuses = np.full((len(encoders), len(batches)), "", dtype=object)
                    for i, enc in enumerate(encoders):
                        for j, bs in enumerate(batches):
                            row = next(
                                (r for r in results
                                 if r["encoder"] == enc and r["size"] == size
                                 and r["batch"] == bs and r["corr_impl"] == corr_impl
                                 and r["amp"] == amp),
                                None,
                            )
                            if row is None:
                                continue
                            grid[i, j] = row["reserved_GB"]
                            statuses[i, j] = row["status"]

                    fig, ax = plt.subplots(figsize=(1.2 * len(batches) + 3, 0.6 * len(encoders) + 3))
                    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=gpu_total_GB)
                    ax.set_xticks(range(len(batches)))
                    ax.set_xticklabels(batches)
                    ax.set_yticks(range(len(encoders)))
                    ax.set_yticklabels(encoders)
                    ax.set_xlabel("batch size")
                    ax.set_ylabel("encoder")
                    ax.set_title(
                        f"{gpu_name} ({gpu_total_GB:.0f} GB) | "
                        f"size={size} | corr={corr_impl} | amp={amp}\n"
                        f"reserved GB (annotation = status; X = OOM, E = error)",
                        fontsize=10,
                    )
                    for i in range(len(encoders)):
                        for j in range(len(batches)):
                            val = grid[i, j]
                            status = statuses[i, j]
                            if status == "OOM":
                                label = "OOM"
                                color = "red"
                            elif status == "ERR":
                                label = "E"
                                color = "orange"
                            elif np.isnan(val):
                                label = "-"
                                color = "gray"
                            else:
                                label = f"{val:.1f}"
                                color = "white" if val < gpu_total_GB * 0.5 else "black"
                            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
                    plt.colorbar(im, ax=ax, label="reserved GB")
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=120)
                    plt.close(fig)
    print(f"[pdf]  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--encoders", nargs="+",
        default=["1_8_p4_r4", "1_4_p3_r4", "1_2_p2_r4"],
        help="Encoder yaml stems (looked up in configs/models/raft_dvc_<NAME>.yaml).",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 12, 16, 24])
    parser.add_argument("--iters", type=int, nargs="+", default=[12])
    parser.add_argument(
        "--corr-impls", nargs="+",
        default=["standard", "on_the_fly"],
        choices=["standard", "on_the_fly"],
    )
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--output", type=Path, default=Path("reports/memory_profile.csv"))
    parser.add_argument("--pdf", type=Path, default=None,
                        help="If set, also write a PDF heatmap report.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA device available.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_total_GB:.1f} GB total)")
    print(f"PyTorch: {torch.__version__}")
    print()

    configs = list(itertools.product(
        args.encoders, args.sizes, args.batches, args.iters, args.corr_impls,
    ))
    n = len(configs)
    print(f"Profiling {n} configurations (amp={args.amp})...")
    print()

    results = []
    skip_keys: set[tuple] = set()  # (encoder, size, corr_impl): minimum batch that OOM'd
    for idx, (enc, size, batch, iters, corr_impl) in enumerate(configs):
        skip_key = (enc, size, corr_impl, iters)
        prev_oom_batch = next(
            (k[-1] for k in skip_keys if k[:-1] == skip_key),
            None,
        )
        if prev_oom_batch is not None and batch > prev_oom_batch:
            if not args.quiet:
                print(
                    f"[{idx + 1:>3}/{n}] {enc} size={size} batch={batch:>2} "
                    f"iters={iters} corr={corr_impl} ... SKIP (>= OOM batch {prev_oom_batch})"
                )
            results.append({
                "encoder": enc, "size": size, "batch": batch, "iters": iters,
                "amp": args.amp, "corr_impl": corr_impl,
                "status": "SKIP", "params_M": float("nan"),
                "alloc_GB": float("nan"), "reserved_GB": float("nan"),
                "error": f"skipped after OOM at batch {prev_oom_batch}",
            })
            continue

        if not args.quiet:
            print(
                f"[{idx + 1:>3}/{n}] {enc} size={size} batch={batch:>2} "
                f"iters={iters} corr={corr_impl} ... ", end="", flush=True,
            )

        r = profile_one(enc, size, batch, iters, args.amp, corr_impl)
        results.append(r)

        if not args.quiet:
            if r["status"] == "OK":
                print(f"{r['alloc_GB']:.2f} / {r['reserved_GB']:.2f} GB")
            elif r["status"] == "OOM":
                print(f"OOM (max alloc {r['alloc_GB']:.2f} GB)")
                skip_keys.add((*skip_key, batch))
            else:
                print(f"ERROR: {r['error'][:60]}")

    write_csv(results, args.output)
    print()
    print(f"[csv]  {args.output}")

    if args.pdf is not None:
        write_pdf(results, args.pdf, gpu_name, gpu_total_GB)

    print_table(results, gpu_total_GB)


if __name__ == "__main__":
    main()
