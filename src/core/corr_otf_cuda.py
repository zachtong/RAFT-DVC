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
import torch.nn.functional as F


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
    PATH for environments that have already built the extension once."""
    import importlib.util
    cache_dir = Path(os.path.expandvars(
        r"%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py310_cu124"
        r"\raft_dvc_corr_otf_cuda"
    ))
    pyd = cache_dir / "raft_dvc_corr_otf_cuda.pyd"
    if not pyd.exists():
        return None
    spec = importlib.util.spec_from_file_location("raft_dvc_corr_otf_cuda", str(pyd))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
