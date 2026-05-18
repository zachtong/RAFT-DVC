"""CUDA-backed on-the-fly 3D correlation lookup.

Loads the compiled CUDA extension on first use (JIT via
``torch.utils.cpp_extension.load``).  Exposes:

* ``CorrLookupCUDAFunction`` — ``torch.autograd.Function`` that calls the
  forward / backward kernels.
* ``corr_lookup_cuda`` — friendly wrapper that loops over pyramid levels
  and stitches the output to match the standard CorrBlock interface
  ``(B, num_levels * (2r+1)^3, H, W, D)``.
* ``CorrBlockCUDA`` — drop-in CorrBlock equivalent backed by the CUDA
  kernel.  Same constructor signature, same ``__call__`` semantics, same
  output shape; can be swapped in by ``RAFTDVCConfig(corr_impl='cuda_otf')``.

The first import triggers the build; subsequent imports reuse the
cached compiled extension from ``$LOCALAPPDATA/torch_extensions``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F  # noqa: F401  (used by other CorrBlock variants)


# -----------------------------------------------------------------------------
# JIT build / lazy module load
# -----------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_CUDA_SRC = _THIS_DIR / "cuda" / "corr_otf_cuda.cu"

_MODULE = None


def _ensure_cuda_home() -> None:
    """Make sure CUDA_HOME points at a 12.4+ toolkit (matches torch cu124)."""
    candidates = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    ]
    cur = os.environ.get("CUDA_HOME", "")
    if "v12." not in cur:
        for c in candidates:
            if os.path.isdir(c):
                os.environ["CUDA_HOME"] = c
                bin_dir = os.path.join(c, "bin")
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break


def _try_import_cached_pyd():
    """Fast path: import the previously compiled .pyd directly, bypassing
    the torch JIT ninja-staleness check. Avoids the need for ninja/MSVC on
    PATH for environments that have already built the extension once.

    Cache directory follows torch's naming convention:
        %LOCALAPPDATA%\\torch_extensions\\torch_extensions\\Cache\\py<P>_cu<C>\\<ext>
    where <P> is e.g. 310 (Python 3.10), <C> is e.g. 124 / 128 (CUDA 12.4 / 12.8).
    We probe the actual PyTorch + Python versions at runtime so this works
    across PyTorch upgrades (e.g. cu124 -> cu128)."""
    import importlib.util
    import sys as _sys
    import torch as _torch
    py_tag = f"py{_sys.version_info.major}{_sys.version_info.minor}"
    cu_tag = f"cu{(_torch.version.cuda or '').replace('.', '')}" if _torch.version.cuda else "cpu"
    base = Path(os.path.expandvars(
        rf"%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\{py_tag}_{cu_tag}"
        r"\raft_dvc_corr_otf_cuda"
    ))
    candidates = [base]
    # Glob for siblings in case PyTorch minor versions create sub-dirs
    parent = base.parent.parent
    if parent.exists():
        for sib in parent.iterdir():
            if sib.is_dir() and sib.name.startswith(py_tag):
                p = sib / "raft_dvc_corr_otf_cuda"
                if p != base and p not in candidates:
                    candidates.append(p)
    for cache_dir in candidates:
        pyd = cache_dir / "raft_dvc_corr_otf_cuda.pyd"
        if not pyd.exists():
            continue
        spec = importlib.util.spec_from_file_location("raft_dvc_corr_otf_cuda", str(pyd))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not _CUDA_SRC.exists():
        raise FileNotFoundError(f"CUDA source not found: {_CUDA_SRC}")
    _ensure_cuda_home()
    try:
        cached = _try_import_cached_pyd()
    except Exception:
        cached = None
    if cached is not None:
        _MODULE = cached
        return _MODULE
    from torch.utils.cpp_extension import load
    # /Zc:preprocessor required by PyTorch 2.11+ headers (compiled_autograd.h
    # uses conforming preprocessor syntax).  Without it MSVC's legacy
    # preprocessor mis-parses `std::` qualified templates as ambiguous symbols.
    _MODULE = load(
        name="raft_dvc_corr_otf_cuda",
        sources=[str(_CUDA_SRC)],
        extra_cflags=["/Zc:preprocessor"],
        extra_cuda_cflags=[
            "-O2", "--ptxas-options=-v",
            "-Xcompiler", "/Zc:preprocessor",
        ],
        verbose=True,
    )
    return _MODULE


# -----------------------------------------------------------------------------
# autograd Function: one pyramid level
# -----------------------------------------------------------------------------

class _CorrOneLevelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale):
        mod = _load_module()
        # Kernels currently support fp32 only; cast and cast back if needed
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()
        out = mod.forward_one_level(f1, f2, c, int(radius), float(scale))
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = int(radius)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None


def _corr_one_level(fmap1, fmap2_i, coords, radius, scale):
    return _CorrOneLevelFn.apply(fmap1, fmap2_i, coords, radius, scale)


# -----------------------------------------------------------------------------
# V3 autograd Function: cooperative-tiled forward + v1 backward
# -----------------------------------------------------------------------------
# V3 forward kernel is ~10-50x faster than v1 (cooperative tiling + scatter
# trick).  V3 backward kernel is not yet implemented -- we fall back to v1
# backward in the meantime, which is correct (already bench-validated) but
# slower than its own forward.  Combined: v3 forward + v1 backward should
# still be ~3-10x faster than pure v1 in training (forward dominates).

class _CorrOneLevelV3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale):
        mod = _load_module()
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()
        # V3 wants channels-last (B, H, W, D, C) layout for coalesced loads.
        # Permute + contiguous() costs ~few ms but enables the kernel's
        # coalesced global reads, which dominate runtime.
        f1_cl = f1.permute(0, 2, 3, 4, 1).contiguous()
        f2_cl = f2.permute(0, 2, 3, 4, 1).contiguous()
        out = mod.forward_one_level_v3(f1_cl, f2_cl, c, int(radius), float(scale))
        # Save channels-first tensors for v1 backward (which expects them).
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = int(radius)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        # Use v1 backward (channels-first, already validated correct).
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None


def _corr_one_level_v3(fmap1, fmap2_i, coords, radius, scale):
    return _CorrOneLevelV3Fn.apply(fmap1, fmap2_i, coords, radius, scale)


# -----------------------------------------------------------------------------
# V4 autograd Function: v3 + neighbor-group parallelism + atomicAdd scatter
# -----------------------------------------------------------------------------
# V4 is v3's structure (channels-last cooperative tiling) plus an extra
# gridDim.z dimension splitting the (iy, ix, iz) iteration space into G
# groups.  This multiplies launched block count by G (default 8), letting
# the kernel saturate the GPU's many SMs.  Scatter uses atomicAdd to handle
# cross-block writes to the same output position.

class _CorrOneLevelV4Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale, n_groups=8):
        mod = _load_module()
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()
        f1_cl = f1.permute(0, 2, 3, 4, 1).contiguous()
        f2_cl = f2.permute(0, 2, 3, 4, 1).contiguous()
        out = mod.forward_one_level_v4(
            f1_cl, f2_cl, c, int(radius), float(scale), int(n_groups),
        )
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = int(radius)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Use v1 backward as fallback (v4 backward TBD)
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None, None


def _corr_one_level_v4(fmap1, fmap2_i, coords, radius, scale, n_groups=8):
    return _CorrOneLevelV4Fn.apply(fmap1, fmap2_i, coords, radius, scale, n_groups)


# -----------------------------------------------------------------------------
# V5 autograd Function: v4 + 4-warp blocks (128 threads per block)
# -----------------------------------------------------------------------------
# V4 was limited by 32 threads/block.  V5 uses 128 threads (4 warps) where
# all warps share the f1 shared-mem region but each has its own f2 slab.
# Each warp handles 1/4 of the iter range -> ~4x more concurrent work per
# block.  Default n_groups=4 (vs 8 in v4) since v5 has more intra-block
# parallelism.

class _CorrOneLevelV5Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale, n_groups=4):
        mod = _load_module()
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()
        f1_cl = f1.permute(0, 2, 3, 4, 1).contiguous()
        f2_cl = f2.permute(0, 2, 3, 4, 1).contiguous()
        out = mod.forward_one_level_v5(
            f1_cl, f2_cl, c, int(radius), float(scale), int(n_groups),
        )
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = int(radius)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None, None


def _corr_one_level_v5(fmap1, fmap2_i, coords, radius, scale, n_groups=4):
    return _CorrOneLevelV5Fn.apply(fmap1, fmap2_i, coords, radius, scale, n_groups)


# -----------------------------------------------------------------------------
# V5b: V5 + per-warp f1 (no cross-warp shared-mem bank conflict)
# -----------------------------------------------------------------------------

class _CorrOneLevelV5bFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale, n_groups=4):
        mod = _load_module()
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()
        f1_cl = f1.permute(0, 2, 3, 4, 1).contiguous()
        f2_cl = f2.permute(0, 2, 3, 4, 1).contiguous()
        out = mod.forward_one_level_v5b(
            f1_cl, f2_cl, c, int(radius), float(scale), int(n_groups),
        )
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = int(radius)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None, None


def _corr_one_level_v5b(fmap1, fmap2_i, coords, radius, scale, n_groups=4):
    return _CorrOneLevelV5bFn.apply(fmap1, fmap2_i, coords, radius, scale, n_groups)


# -----------------------------------------------------------------------------
# V6: trilinear-sample CUDA kernel + torch.matmul fp16
# -----------------------------------------------------------------------------
# Strategy: materialize sampled fmap2 as fp16 tensor (chunked), then use
# torch.matmul which dispatches to cuBLAS GEMM.  cuBLAS uses tensor cores
# for fp16 GEMMs where M, N, K >= 16.  Our per-source dot is M=1 which may
# fall back to gemv scalar path -- but fp16 storage alone halves memory
# bandwidth vs v5 fp32, giving ~2x speedup as memory-bound floor.

class _CorrOneLevelV6Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2_i, coords, radius, scale, chunk_size=512):
        mod = _load_module()
        f1 = fmap1.float().contiguous()
        f2 = fmap2_i.float().contiguous()
        c = coords.float().contiguous()

        B, C, H, W, D = f1.shape
        spatial = H * W * D
        radius_i = int(radius)
        scale_f = float(scale)
        n_neighbors = (2 * radius_i + 1) ** 3

        # Output: (B, n_neighbors, H, W, D) fp32 -- matches v1/v5 interface
        corr = torch.zeros(
            B, n_neighbors, H, W, D,
            device=f1.device, dtype=torch.float32,
        )
        # View as (B, n_neighbors, spatial) for indexed scatter
        corr_flat = corr.view(B, n_neighbors, spatial)

        # f1 as (B, spatial, C) fp16 once
        # (permute (B, C, H, W, D) -> (B, H, W, D, C) -> (B, spatial, C))
        f1_flat = f1.permute(0, 2, 3, 4, 1).contiguous().view(B, spatial, C).half()

        norm = 1.0 / math.sqrt(C)
        for chunk_start in range(0, spatial, chunk_size):
            actual_chunk = min(chunk_size, spatial - chunk_start)

            # Sample fmap2 at (chunk_size, n_neighbors) positions per batch
            # Output: (B, actual_chunk, n_neighbors, C) fp16
            # Note: kernel may produce extra rows if chunk_size > actual_chunk;
            # we always request actual_chunk so output is exactly (B, actual_chunk, N, C).
            sampled = mod.v6_sample(
                f2, c, radius_i, scale_f, chunk_start, actual_chunk,
            )  # (B, actual_chunk, n_neighbors, C) fp16

            # f1 chunk: (B, actual_chunk, C) fp16
            f1_chunk = f1_flat[:, chunk_start:chunk_start + actual_chunk, :]

            # Per-source per-neighbor dot product via matmul:
            #   out[b, s, n] = sum_c f1[b, s, c] * sampled[b, s, n, c]
            # Reshape: (B, S, 1, C) @ (B, S, C, N) -> (B, S, 1, N) -> (B, S, N)
            corr_chunk = torch.matmul(
                f1_chunk.unsqueeze(-2),                 # (B, S, 1, C)
                sampled.transpose(-1, -2),              # (B, S, C, N)
            ).squeeze(-2)                                # (B, S, N) fp16

            corr_chunk = corr_chunk.float() * norm

            # Scatter to corr_flat (B, N, spatial) -- need transpose (B, S, N) -> (B, N, S)
            corr_flat[:, :, chunk_start:chunk_start + actual_chunk] = corr_chunk.transpose(1, 2)

        # Save for backward (use v1 backward fallback)
        ctx.save_for_backward(f1, f2, c)
        ctx.radius = radius_i
        ctx.scale = scale_f
        return corr

    @staticmethod
    def backward(ctx, grad_out):
        mod = _load_module()
        f1, f2, c = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        gf1, gf2, gc = mod.backward_one_level(
            f1, f2, c, grad_out, ctx.radius, ctx.scale,
        )
        return gf1, gf2, gc, None, None, None


def _corr_one_level_v6(fmap1, fmap2_i, coords, radius, scale, chunk_size=512):
    return _CorrOneLevelV6Fn.apply(fmap1, fmap2_i, coords, radius, scale, chunk_size)


# -----------------------------------------------------------------------------
# Multi-level wrapper matching CorrBlock interface
# -----------------------------------------------------------------------------

class CorrBlockCUDA:
    """CUDA-backed 3D correlation block — drop-in for CorrBlock / CorrBlockOnTheFly.

    Stores only fmap1 and the avg-pool pyramid of fmap2; the correlation
    values themselves are computed inside each ``__call__`` by the CUDA
    kernel.  Memory footprint matches CorrBlockOnTheFly; speed is
    expected to be comparable to the standard CorrBlock.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(f"expected 5D fmaps; got {fmap1.ndim}D")
        if not (fmap1.is_cuda and fmap2.is_cuda):
            raise RuntimeError("CorrBlockCUDA requires CUDA tensors")

        self.num_levels = num_levels
        self.radius = radius
        self.fmap1 = fmap1

        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (B,3,H,W,D); got {tuple(coords.shape)}")
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level(
                self.fmap1, fmap2_i, coords, self.radius, scale,
            )  # (B, n_neighbors, H, W, D)
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()


class CorrBlockCUDAv3:
    """V3 cooperative-tiled CUDA correlation block.

    Uses v3 forward kernel (cooperative tiling, channels-last, 10-50x faster
    than v1) with v1 backward (already validated, until v3 backward lands).
    Drop-in for CorrBlock / CorrBlockOnTheFly / CorrBlockCUDA.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(f"expected 5D fmaps; got {fmap1.ndim}D")
        if not (fmap1.is_cuda and fmap2.is_cuda):
            raise RuntimeError("CorrBlockCUDAv3 requires CUDA tensors")
        if fmap1.shape[1] % 32 != 0:
            raise ValueError(
                f"v3 requires C divisible by 32 (got C={fmap1.shape[1]}). "
                f"Use 'cuda_otf' (v1) or 'on_the_fly' for arbitrary C."
            )

        self.num_levels = num_levels
        self.radius = radius
        self.fmap1 = fmap1

        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (B,3,H,W,D); got {tuple(coords.shape)}")
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level_v3(
                self.fmap1, fmap2_i, coords, self.radius, scale,
            )
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()


class CorrBlockCUDAv4:
    """V4 CUDA correlation block: cooperative tiling + neighbor-group parallelism.

    Same forward output as v3 / v1 / standard.  Uses atomicAdd scatter so
    extra gridDim.z dimension can split the inner loop across more blocks,
    achieving high GPU occupancy.  Backward uses v1 kernel as fallback.

    Args:
        n_groups: number of neighbor-iteration groups (gridDim.z).  Default 8
            balances atomic contention vs occupancy.  Tune via bench.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        n_groups: int = 8,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(f"expected 5D fmaps; got {fmap1.ndim}D")
        if not (fmap1.is_cuda and fmap2.is_cuda):
            raise RuntimeError("CorrBlockCUDAv4 requires CUDA tensors")
        if fmap1.shape[1] % 32 != 0:
            raise ValueError(
                f"v4 requires C divisible by 32 (got C={fmap1.shape[1]})."
            )

        self.num_levels = num_levels
        self.radius = radius
        self.n_groups = n_groups
        self.fmap1 = fmap1

        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (B,3,H,W,D); got {tuple(coords.shape)}")
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level_v4(
                self.fmap1, fmap2_i, coords, self.radius, scale, self.n_groups,
            )
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()


class CorrBlockCUDAv5b:
    """V5b: V5 with per-warp f1 region to eliminate bank conflicts.
    Same interface as v5; only the kernel differs."""

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        n_groups: int = 4,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.shape[1] % 32 != 0:
            raise ValueError(f"v5b requires C % 32 == 0 (got {fmap1.shape[1]})")
        self.num_levels = num_levels
        self.radius = radius
        self.n_groups = n_groups
        self.fmap1 = fmap1
        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level_v5b(
                self.fmap1, fmap2_i, coords, self.radius, scale, self.n_groups,
            )
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()


class CorrBlockCUDAv6:
    """V6: trilinear-sample CUDA kernel + torch.matmul fp16 path.

    Strategy:
      1. Custom CUDA kernel materializes sampled fmap2 (B, chunk, N, C) fp16.
      2. torch.matmul (cuBLAS gemm fp16) for per-source dot products.

    Backward: v1 kernel fallback.

    Args:
        chunk_size: number of source voxels per Python-loop iteration.
            Larger -> fewer Python overhead but more peak GPU memory.
            Default 512 balances both on 5090 32GB.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        chunk_size: int = 512,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(f"expected 5D fmaps; got {fmap1.ndim}D")
        if not (fmap1.is_cuda and fmap2.is_cuda):
            raise RuntimeError("CorrBlockCUDAv6 requires CUDA tensors")

        self.num_levels = num_levels
        self.radius = radius
        self.chunk_size = chunk_size
        self.fmap1 = fmap1
        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (B,3,H,W,D); got {tuple(coords.shape)}")
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level_v6(
                self.fmap1, fmap2_i, coords, self.radius, scale, self.chunk_size,
            )
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()


class CorrBlockCUDAv5:
    """V5: 4-warp cooperative tiling + neighbor-group parallelism.

    Same scatter trick + atomicAdd accumulation as v4, but with 128 threads
    (4 warps) per block.  All warps share one f1 region in smem; each warp
    has its private f2 region and processes 1/4 of the iter range.
    """

    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
        n_groups: int = 4,
    ) -> None:
        if fmap1.shape != fmap2.shape:
            raise ValueError(
                f"fmap1 and fmap2 must match; got {tuple(fmap1.shape)} vs "
                f"{tuple(fmap2.shape)}"
            )
        if fmap1.ndim != 5:
            raise ValueError(f"expected 5D fmaps; got {fmap1.ndim}D")
        if not (fmap1.is_cuda and fmap2.is_cuda):
            raise RuntimeError("CorrBlockCUDAv5 requires CUDA tensors")
        if fmap1.shape[1] % 32 != 0:
            raise ValueError(
                f"v5 requires C divisible by 32 (got C={fmap1.shape[1]})."
            )

        self.num_levels = num_levels
        self.radius = radius
        self.n_groups = n_groups
        self.fmap1 = fmap1

        self.fmap2_pyramid: List[torch.Tensor] = [fmap2]
        for _ in range(num_levels - 1):
            prev = self.fmap2_pyramid[-1]
            self.fmap2_pyramid.append(F.avg_pool3d(prev, kernel_size=2, stride=2))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 5 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (B,3,H,W,D); got {tuple(coords.shape)}")
        parts = []
        for level in range(self.num_levels):
            fmap2_i = self.fmap2_pyramid[level]
            scale = 1.0 / (2 ** level)
            corr_l = _corr_one_level_v5(
                self.fmap1, fmap2_i, coords, self.radius, scale, self.n_groups,
            )
            parts.append(corr_l)
        out = torch.cat(parts, dim=1)
        return out.contiguous().float()
