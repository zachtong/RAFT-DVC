# OTF v3: Cooperative-Tiled 3D CUDA Correlation Kernel

**Status**: In progress (2026-05-17). Forward kernel landed; backward + integration TBD.

## Motivation

Bench results from `src/core/cuda/_bench_std_vs_otf_training_out.txt`:

| Config | Standard CorrBlock | CUDA OTF v1 | Ratio |
|---|---|---|---|
| fm=16 batch=1 | 204 ms/step | 755 ms/step | v1 3.7x slower |
| fm=16 batch=4 | 680 ms/step | 2845 ms/step | v1 4.2x slower |
| fm=16 batch=8 | 34910 ms/step (paging) | 6140 ms/step | v1 5.7x faster |
| fm=32 batch=1 | OOM | 9485 ms/step | v1 only option |

v1 is correct (forward + backward pass equivalence tests) but **3-4x slower than standard**
at sizes that fit memory.  Root cause: v1's forward kernel uses **one block per source
voxel** (gridDim = B*H*W*D) with **256 threads** independently fetching scattered fmap2
positions.  No cooperative loading of either fmap1 or fmap2 — each thread issues its own
global memory transactions.

## Design (v3)

Follow princeton-vl/RAFT's `alt_cuda_corr` 2D kernel pattern, ported faithfully to 3D:

* **32 source voxels per block** (BLOCK_T = 32 = one warp) — *cooperative*
* **32-channel chunks** (CHANNEL_STRIDE = 32) — shared memory tiles
* **Channels-last (B, H, W, D, C) memory layout** — coalesced fmap loads
* **Bilinear/trilinear scatter at output write** — avoid materializing interpolated fmap2

### Block layout

```
gridDim  = (B, ceil(H*W*D / BLOCK_T))   <- 32x fewer blocks than v1
blockDim = BLOCK_T = 32                  <- one warp, lockstep
```

### Shared memory budget

```
f1[CHANNEL_STRIDE][BLOCK_T + 1]   = 32 * 33 * 4 = 4224 B
f2[CHANNEL_STRIDE][BLOCK_T + 1]   = 4224 B
coords_y/x/z[BLOCK_T]             = 3 * 128 = 384 B
Total per block                   ≈ 9 KB  (well under 48 KB default)
```

### Memory traffic comparison (1/4 batch=8 fm=16)

| Tensor | v1 reads | v3 reads | v3 reduction |
|---|---|---|---|
| fmap2 | ~24 G   | ~131 M  | 180x fewer |
| fmap1 | ~12 G   | ~4 M    | 3000x fewer |

### Bilinear scatter trick

For 2D RAFT (and 3D here), each source voxel needs the correlation at fractional
positions `coords[s] + (-r..r)^N`.  Naive: interpolate fmap2 (8 corner reads + dot)
per neighbor = 8 * N^3 * C reads.

Trick: iterate over **integer** offsets `(iy, ix, iz)` ∈ [0, rd+1)^3.  Compute the
un-interpolated dot product `dot(f1[s], f2[floor(coords[s]) + (iy-r, ix-r, iz-r)])`.
Each integer position contributes to up to 8 output neighbors via trilinear weights:

```
output[n_iy_off, n_ix_off, n_iz_off]  +=  dot * w(cy, cx, cz)
where n_iy_off = iy_loc - r - cy_corner, etc.
```

This shifts the interpolation algebra to the output write, reducing the dot product
to a clean integer-grid sample.

### Algorithm (forward)

```
for each block (b, source_block):
    load coords[block_offset..block_offset+32] to shared mem  (3 floats * 32)
    for each channel chunk c0 in [0, C) step 32:
        cooperatively load f1[c0:c0+32, block_offset:block_offset+32]   (4 KB to shared)
        for iy_loc in [0..rd]:           # rd = 9 for radius=4
            for ix_loc in [0..rd]:
                for iz_loc in [0..rd]:
                    cooperatively load f2[c0:c0+32, neighbor_pos] for all 32 sources  (4 KB)
                    for source `tid` in block:
                        dot = sum_k f1[k, tid] * f2[k, tid]
                        scatter dot * trilinear_weight to up to 8 output positions
```

## Correctness validation

Reference: v1 CUDA kernel (already in `corr_otf_cuda.cu`) is bench-tested numerically
equivalent to standard CorrBlock (fp32 1e-6, fp16 5e-3, gradients 1e-5).

Tests v3 forward against:
1. v1 forward — exact match (both are fp32 deterministic)
2. Standard CorrBlock — within fp32 tolerance

## Iteration plan

| Step | Deliverable | Status |
|---|---|---|
| 1 | Plan doc + v3 forward kernel + Python wrapper | **THIS COMMIT** |
| 2 | Forward correctness vs v1 (CI test) | Next |
| 3 | Forward benchmark vs v1 / std | Next |
| 4 | v3 backward kernel | TBD |
| 5 | Backward correctness vs v1 | TBD |
| 6 | Integration via `corr_impl='cuda_otf_v3'` | TBD |
| 7 | Paper-1 v3 1/2 case using OTF v3 | TBD |

## Non-goals (v3)

* **Tensor cores (wmma)**: deferred to v4.  v3 stays in fp32 to keep numerics
  identical to v1.  Expected v3 speedup over v1 is 10-50x even without tensor
  cores, which should already get us competitive with standard CorrBlock.
* **Mixed precision input**: also deferred to v4.

## Channel layout caveat

v3 requires channels-last memory (`(B, H, W, D, C)` not `(B, C, H, W, D)`).  The
Python wrapper does a `permute(0, 2, 3, 4, 1).contiguous()` before kernel launch.
Cost: ~few ms per call (fmap is small compared to corr volume).  Worth it for the
coalesced global loads.
