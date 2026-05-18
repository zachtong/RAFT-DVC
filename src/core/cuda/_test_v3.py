"""Test v3 forward kernel: correctness vs v1 and standard CorrBlock + speed bench.

Run with:
    python src\\core\\cuda\\_test_v3.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.core.corr import CorrBlock
from src.core.corr_otf_cuda import (
    CorrBlockCUDA, CorrBlockCUDAv3, CorrBlockCUDAv4, _load_module,
)


DEVICE = "cuda" if torch.cuda.is_available() else None
if DEVICE is None:
    print("ERROR: no CUDA device.")
    sys.exit(1)


def _make_inputs(B=2, C=128, H=8, W=8, D=8, seed=42, max_flow=1.5):
    g = torch.Generator(device="cpu").manual_seed(seed)
    fmap1 = torch.randn(B, C, H, W, D, generator=g, dtype=torch.float32).to(DEVICE)
    fmap2 = torch.randn(B, C, H, W, D, generator=g, dtype=torch.float32).to(DEVICE)
    # Coords near identity grid with small flow
    coords_h = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1, 1).expand(B, 1, H, W, D)
    coords_w = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W, 1).expand(B, 1, H, W, D)
    coords_d = torch.arange(D, dtype=torch.float32).view(1, 1, 1, 1, D).expand(B, 1, H, W, D)
    base = torch.cat([coords_h, coords_d, coords_w], dim=1).contiguous().to(DEVICE)
    noise = (torch.rand(B, 3, H, W, D, generator=g) - 0.5) * 2 * max_flow
    coords = base + noise.to(DEVICE)
    return fmap1, fmap2, coords


def test_correctness():
    print("\n=== [correctness] v3 / v4 vs v1 vs standard CorrBlock ===")
    for (B, C, H) in [(1, 32, 8), (2, 128, 8), (2, 128, 16), (1, 256, 16)]:
        fmap1, fmap2, coords = _make_inputs(B=B, C=C, H=H, W=H, D=H)
        for L in (2, 3):
            std = CorrBlock(fmap1, fmap2, num_levels=L, radius=4)
            v1 = CorrBlockCUDA(fmap1, fmap2, num_levels=L, radius=4)
            v3 = CorrBlockCUDAv3(fmap1, fmap2, num_levels=L, radius=4)
            v4 = CorrBlockCUDAv4(fmap1, fmap2, num_levels=L, radius=4)
            out_std = std(coords)
            out_v1 = v1(coords)
            out_v3 = v3(coords)
            out_v4 = v4(coords)
            d_v1_std = (out_v1 - out_std).abs().max().item()
            d_v3_std = (out_v3 - out_std).abs().max().item()
            d_v4_std = (out_v4 - out_std).abs().max().item()
            ok3 = "P" if d_v3_std < 1e-4 else "F"
            ok4 = "P" if d_v4_std < 1e-4 else "F"
            print(
                f"  B={B} C={C} H={H} L={L}  "
                f"v1-std={d_v1_std:.2e}  v3-std={d_v3_std:.2e}[{ok3}]  "
                f"v4-std={d_v4_std:.2e}[{ok4}]"
            )


def bench(B, C, H, L=2, n_iters=12, n_warmup=3, n_runs=10):
    fmap1, fmap2, coords = _make_inputs(B=B, C=C, H=H, W=H, D=H)
    results = {}
    for name, ctor in [
        ("standard", CorrBlock),
        ("v1", CorrBlockCUDA),
        ("v3", CorrBlockCUDAv3),
        ("v4", CorrBlockCUDAv4),
    ]:
        try:
            corr = ctor(fmap1, fmap2, num_levels=L, radius=4)
        except Exception as exc:
            results[name] = f"build error: {exc}"
            continue
        # Warmup
        try:
            for _ in range(n_warmup):
                _ = corr(coords)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            results[name] = "OOM"
            continue

        # Time n_iters lookups (mimic 12 GRU iters), n_runs averaged
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            for _ in range(n_iters):
                _ = corr(coords)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms_per_step = (t1 - t0) * 1000 / n_runs
        peak_GB = torch.cuda.max_memory_reserved() / 1e9
        results[name] = (ms_per_step, peak_GB)
    return results


def benchmark():
    print("\n=== [bench] forward-only (12 lookup iters per step, fp32, no grad) ===")
    print(f"{'config':<22} {'impl':<10} {'ms/step':>10} {'peak_GB':>8}")
    print("-" * 56)
    for cfg in [
        (1, 128, 8),   # 1/8 fm
        (1, 128, 16),  # 1/4 fm
        (4, 128, 16),
        (8, 128, 16),
        (1, 128, 32),  # 1/2 fm
    ]:
        B, C, H = cfg
        cfg_str = f"B={B} C={C} fm={H}"
        results = bench(B, C, H, L=2)
        for impl in ("standard", "v1", "v3", "v4"):
            r = results.get(impl, "?")
            if isinstance(r, tuple):
                ms, peak = r
                print(f"{cfg_str:<22} {impl:<10} {ms:>10.2f} {peak:>7.2f}G")
            else:
                print(f"{cfg_str:<22} {impl:<10} {str(r):>10}")
        print()


if __name__ == "__main__":
    print("[setup] loading CUDA extension...")
    _load_module()
    print("[setup] OK")
    test_correctness()
    benchmark()
