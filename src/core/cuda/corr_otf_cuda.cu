/* On-the-fly 3D correlation lookup — CUDA kernel.
 *
 * Adapted to 3D from princeton-vl/RAFT's alt_cuda_corr_kernel.cu.
 *
 * Functional spec:
 *   For each output voxel (b, h, w, d) at each pyramid level l, sample fmap2_l
 *   at (2r+1)^3 neighbor positions of `coords[b, :, h, w, d] / 2^l` via
 *   trilinear interpolation, dot-product each sample with fmap1[b, :, h, w, d],
 *   and normalize by 1/sqrt(C).  This matches the PyTorch OTF reference
 *   implementation in src/core/corr_otf.py.
 *
 *   Forward:  per-call output shape (B, n_neighbors, H, W, D), one level at a
 *             time.  The Python wrapper loops over pyramid levels and
 *             concatenates along the neighbor dim, then permutes to match
 *             the standard CorrBlock interface (B, L*n_neighbors, H, W, D).
 *
 *   Backward: gradients flow back to fmap1, fmap2, and coords (the latter
 *             because trilinear sampling is differentiable w.r.t. coords).
 */

// Minimal include set -- avoid <torch/extension.h> (umbrella) because PyTorch
// 2.11 dev's torch/csrc/dynamo/compiled_autograd.h fails with C2872 'std'
// ambiguous under nvcc + MSVC + CUDA 13.1.  We only need:
//   - at::Tensor & options
//   - CUDA stream getter
//   - pybind11 macros for the module entrypoint
//   - TORCH_CHECK / AT_CUDA_CHECK macros
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda.h>
#include <cuda_runtime.h>

// We use `torch::` qualified APIs throughout this file (TORCH_CHECK already
// works since c10/util/Exception.h provides it).  Map the few `torch::*`
// type/factory references to their at:: equivalents so we don't need
// torch/all.h.  Note ATen names the fp32 scalar type `kFloat` (not
// `kFloat32`); we alias it via a constexpr.
namespace torch {
    using ::at::Tensor;
    using ::at::empty;
    using ::at::zeros;
    using ::at::zeros_like;
    constexpr auto kFloat32 = ::at::kFloat;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIG(x)
#define CHECK_FP32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

#define BLOCK_THREADS 256

// =============================================================================
// Trilinear interpolation device helpers
// =============================================================================

// Precomputed trilinear sampling state: clamped corner indices + per-corner
// validity-aware weights.  Once filled, the per-channel hot loop is fully
// branch-free (always-valid memory access, weights of out-of-bounds corners
// are zero — matching grid_sample padding_mode='zeros').
struct TrilinearState {
    int y0, x0, z0, y1, x1, z1;          // clamped corner indices
    float w000, w001, w010, w011;         // 8 corner weights (validity baked in)
    float w100, w101, w110, w111;
    float my0, my1, mx0, mx1, mz0, mz1;  // raw validity masks (for backward)
    float fy, fx, fz;                     // fractional parts (for backward)
};

__device__ __forceinline__ TrilinearState make_trilinear_state(
    float y, float x, float z, int H, int W, int D
) {
    TrilinearState s;
    int y0 = (int)floorf(y);
    int x0 = (int)floorf(x);
    int z0 = (int)floorf(z);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    int z1 = z0 + 1;
    s.fy = y - (float)y0;
    s.fx = x - (float)x0;
    s.fz = z - (float)z0;

    s.my0 = (y0 >= 0 && y0 < H) ? 1.f : 0.f;
    s.my1 = (y1 >= 0 && y1 < H) ? 1.f : 0.f;
    s.mx0 = (x0 >= 0 && x0 < W) ? 1.f : 0.f;
    s.mx1 = (x1 >= 0 && x1 < W) ? 1.f : 0.f;
    s.mz0 = (z0 >= 0 && z0 < D) ? 1.f : 0.f;
    s.mz1 = (z1 >= 0 && z1 < D) ? 1.f : 0.f;

    s.y0 = max(0, min(y0, H - 1));
    s.y1 = max(0, min(y1, H - 1));
    s.x0 = max(0, min(x0, W - 1));
    s.x1 = max(0, min(x1, W - 1));
    s.z0 = max(0, min(z0, D - 1));
    s.z1 = max(0, min(z1, D - 1));

    float w_y0 = 1.f - s.fy;  float w_y1 = s.fy;
    float w_x0 = 1.f - s.fx;  float w_x1 = s.fx;
    float w_z0 = 1.f - s.fz;  float w_z1 = s.fz;

    s.w000 = w_y0 * w_x0 * w_z0 * s.my0 * s.mx0 * s.mz0;
    s.w001 = w_y0 * w_x0 * w_z1 * s.my0 * s.mx0 * s.mz1;
    s.w010 = w_y0 * w_x1 * w_z0 * s.my0 * s.mx1 * s.mz0;
    s.w011 = w_y0 * w_x1 * w_z1 * s.my0 * s.mx1 * s.mz1;
    s.w100 = w_y1 * w_x0 * w_z0 * s.my1 * s.mx0 * s.mz0;
    s.w101 = w_y1 * w_x0 * w_z1 * s.my1 * s.mx0 * s.mz1;
    s.w110 = w_y1 * w_x1 * w_z0 * s.my1 * s.mx1 * s.mz0;
    s.w111 = w_y1 * w_x1 * w_z1 * s.my1 * s.mx1 * s.mz1;
    return s;
}

// Branch-free per-channel trilinear sample using precomputed state.
__device__ __forceinline__ float trilinear_one_channel(
    const float* __restrict__ fmap2_b,
    int ch, int H, int W, int D,
    const TrilinearState& s
) {
    int stride_h = W * D;
    int stride_w = D;
    const float* base = fmap2_b + ch * H * stride_h;

    float c000 = base[s.y0 * stride_h + s.x0 * stride_w + s.z0];
    float c001 = base[s.y0 * stride_h + s.x0 * stride_w + s.z1];
    float c010 = base[s.y0 * stride_h + s.x1 * stride_w + s.z0];
    float c011 = base[s.y0 * stride_h + s.x1 * stride_w + s.z1];
    float c100 = base[s.y1 * stride_h + s.x0 * stride_w + s.z0];
    float c101 = base[s.y1 * stride_h + s.x0 * stride_w + s.z1];
    float c110 = base[s.y1 * stride_h + s.x1 * stride_w + s.z0];
    float c111 = base[s.y1 * stride_h + s.x1 * stride_w + s.z1];

    return s.w000 * c000 + s.w001 * c001 + s.w010 * c010 + s.w011 * c011
         + s.w100 * c100 + s.w101 * c101 + s.w110 * c110 + s.w111 * c111;
}


// =============================================================================
// Forward kernel — one pyramid level
// =============================================================================
// Block layout:
//   gridDim  = (B * H * W * D,)             — one block per output voxel
//   blockDim = (BLOCK_THREADS,)             — threads cooperate on neighbors
// Each block computes the n_neighbors = (2r+1)^3 correlation values for one
// (b, h, w, d) output voxel by sampling fmap2 at neighbors of coords/scale.

__global__ void corr_otf_forward_kernel(
    const float* __restrict__ fmap1,   // (B, C, H, W, D), contiguous
    const float* __restrict__ fmap2,   // (B, C, Hi, Wi, Di), contiguous
    const float* __restrict__ coords,  // (B, 3, H, W, D), contiguous (y, x, z order)
    float* __restrict__ corr,          // (B, n_neighbors, H, W, D), contiguous
    int B, int C,
    int H, int W, int D,
    int Hi, int Wi, int Di,
    int radius,
    float scale,    // 1.0 / 2^level — multiplied into coords
    float norm      // 1.0 / sqrt(C)
) {
    int spatial = H * W * D;
    int total = B * spatial;
    int idx = blockIdx.x;
    if (idx >= total) return;

    int b = idx / spatial;
    int s = idx % spatial;
    int h = s / (W * D);
    int wd = s % (W * D);
    int w = wd / D;
    int d = wd % D;

    int span = 2 * radius + 1;
    int n_neighbors = span * span * span;

    // Per-batch offsets
    const float* fmap1_b = fmap1 + b * C * spatial;
    const float* fmap2_b = fmap2 + b * C * Hi * Wi * Di;
    const float* coords_b = coords + b * 3 * spatial;
    float* corr_b = corr + b * n_neighbors * spatial;

    // Centroid of the lookup window at this pyramid level.
    // Convention matches src/core/corr.py's bilinear_sampler_3d: coords
    // channel 0 indexes H, channel 1 indexes D, channel 2 indexes W.
    // (The names mirror coords_grid_3d output, but bilinear_sampler_3d's
    // grid permutation effectively swaps W and D during sampling, so we
    // mirror that swap here.)
    float c_H = coords_b[0 * spatial + s] * scale;
    float c_D = coords_b[1 * spatial + s] * scale;
    float c_W = coords_b[2 * spatial + s] * scale;

    // Each thread iterates over a strided subset of neighbors
    for (int n = threadIdx.x; n < n_neighbors; n += blockDim.x) {
        // Decode delta channel order — same as coords (H, D, W)
        int idx_H = n / (span * span);
        int rem = n % (span * span);
        int idx_D = rem / span;
        int idx_W = rem % span;
        float d_H = (float)(idx_H - radius);
        float d_D = (float)(idx_D - radius);
        float d_W = (float)(idx_W - radius);

        float t_H = c_H + d_H;
        float t_D = c_D + d_D;
        float t_W = c_W + d_W;

        // Precompute trilinear state ONCE (clamped indices + masked weights)
        // so the channel loop below is fully branch-free.
        TrilinearState st = make_trilinear_state(t_H, t_W, t_D, Hi, Wi, Di);

        float acc = 0.f;
        for (int ch = 0; ch < C; ++ch) {
            float f1 = fmap1_b[ch * spatial + s];
            float f2 = trilinear_one_channel(fmap2_b, ch, Hi, Wi, Di, st);
            acc += f1 * f2;
        }
        corr_b[n * spatial + s] = acc * norm;
    }
}


// =============================================================================
// Backward kernel — one pyramid level
// =============================================================================
// Accumulates gradients w.r.t. fmap1, fmap2, and coords.  fmap2 uses atomicAdd
// because the same fmap2 voxel can be touched by many output voxels.
// coords uses thread-local accumulation since each (b, h, w, d) has its own
// 3 gradient outputs.

// Backward kernel — one pyramid level.  Multi-warp per-source-voxel layout
// (fp32 throughout, register accumulator + warp-shuffle reduction):
//   * ONE BLOCK per source voxel, 256 threads (8 warps) cooperating over
//     the 1458 neighbors (~6 per thread)
//   * Per-thread per-channel grad_fmap1 accumulator in registers, warp-
//     shuffle reduced, then ONE atomicAdd per warp per channel
//   * grad_fmap2 still uses per-iter atomicAdd (varies by neighbor)
//   * grad_coords accumulated per-thread, warp-shuffle reduced
//
// This is the stable production version of the kernel.  fp16/tensor-core
// experiments (see git history) gave no measurable speedup on Ada because
// the bottleneck is atomicAdd contention + memory latency, not fp32 compute.
//
// gridDim  = B * H * W * D
// blockDim = BLOCK_THREADS (256)
template <int C_TPL>
__global__ void corr_otf_backward_kernel(
    const float* __restrict__ fmap1,
    const float* __restrict__ fmap2,
    const float* __restrict__ coords,
    const float* __restrict__ grad_corr,  // (B, n_neighbors, H, W, D)
    float* __restrict__ grad_fmap1,       // (B, C, H, W, D)
    float* __restrict__ grad_fmap2,       // (B, C, Hi, Wi, Di)
    float* __restrict__ grad_coords,      // (B, 3, H, W, D)
    int B,
    int H, int W, int D,
    int Hi, int Wi, int Di,
    int radius,
    float scale,
    float norm
) {
    constexpr int C = C_TPL;
    int spatial = H * W * D;
    int total = B * spatial;
    int idx = blockIdx.x;
    if (idx >= total) return;

    int b = idx / spatial;
    int s = idx % spatial;

    int span = 2 * radius + 1;
    int n_neighbors = span * span * span;

    const float* fmap1_b = fmap1 + b * C * spatial;
    const float* fmap2_b = fmap2 + b * C * Hi * Wi * Di;
    const float* coords_b = coords + b * 3 * spatial;
    const float* grad_corr_b = grad_corr + b * n_neighbors * spatial;

    float* grad_fmap1_b = grad_fmap1 + b * C * spatial;
    float* grad_fmap2_b = grad_fmap2 + b * C * Hi * Wi * Di;
    float* grad_coords_b = grad_coords + b * 3 * spatial;

    // Shared-memory cache of fmap1[*, s] in fp32 — uniform across all
    // neighbors for this block, loaded cooperatively once.
    __shared__ float fmap1_cache[C];
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        fmap1_cache[i] = fmap1_b[i * spatial + s];
    }
    __syncthreads();

    float c_H = coords_b[0 * spatial + s] * scale;
    float c_D = coords_b[1 * spatial + s] * scale;
    float c_W = coords_b[2 * spatial + s] * scale;

    // Per-thread per-channel register accumulator (template C lets the
    // compiler keep the array in registers — no spilling at C=128).
    float grad_fmap1_acc[C];
    #pragma unroll
    for (int i = 0; i < C; ++i) grad_fmap1_acc[i] = 0.f;

    float local_g_H = 0.f, local_g_D = 0.f, local_g_W = 0.f;

    // 256 threads partition the 1458 neighbors (~6 per thread)
    for (int n = threadIdx.x; n < n_neighbors; n += blockDim.x) {
        int idx_H = n / (span * span);
        int rem = n % (span * span);
        int idx_D = rem / span;
        int idx_W = rem % span;
        float d_H = (float)(idx_H - radius);
        float d_D = (float)(idx_D - radius);
        float d_W = (float)(idx_W - radius);

        float t_H = c_H + d_H;
        float t_D = c_D + d_D;
        float t_W = c_W + d_W;

        // Build trilinear state (clamped indices + masked weights). For
        // backward we also need un-masked per-axis weights to compute the
        // d_sample/d_coord derivatives correctly.
        TrilinearState st = make_trilinear_state(t_H, t_W, t_D, Hi, Wi, Di);
        float go = grad_corr_b[n * spatial + s] * norm;

        // Bare per-axis weights (no validity mask) — used for axis-derivative
        // sign coefficients.  Validity is folded back in via st.m* below.
        float w_y0 = 1.f - st.fy, w_y1 = st.fy;
        float w_x0 = 1.f - st.fx, w_x1 = st.fx;
        float w_z0 = 1.f - st.fz, w_z1 = st.fz;

        int stride_h_f2 = Wi * Di;
        int stride_w_f2 = Di;

        #pragma unroll
        for (int ch = 0; ch < C; ++ch) {
            const float* base_ch = fmap2_b + ch * Hi * Wi * Di;
            float* gbase_ch = grad_fmap2_b + ch * Hi * Wi * Di;

            float c000 = base_ch[st.y0 * stride_h_f2 + st.x0 * stride_w_f2 + st.z0];
            float c001 = base_ch[st.y0 * stride_h_f2 + st.x0 * stride_w_f2 + st.z1];
            float c010 = base_ch[st.y0 * stride_h_f2 + st.x1 * stride_w_f2 + st.z0];
            float c011 = base_ch[st.y0 * stride_h_f2 + st.x1 * stride_w_f2 + st.z1];
            float c100 = base_ch[st.y1 * stride_h_f2 + st.x0 * stride_w_f2 + st.z0];
            float c101 = base_ch[st.y1 * stride_h_f2 + st.x0 * stride_w_f2 + st.z1];
            float c110 = base_ch[st.y1 * stride_h_f2 + st.x1 * stride_w_f2 + st.z0];
            float c111 = base_ch[st.y1 * stride_h_f2 + st.x1 * stride_w_f2 + st.z1];

            float f1 = fmap1_cache[ch];
            float sampled = st.w000 * c000 + st.w001 * c001 + st.w010 * c010 + st.w011 * c011
                          + st.w100 * c100 + st.w101 * c101 + st.w110 * c110 + st.w111 * c111;

            // Per-thread register accumulator — warp-shuffle reduced at end
            grad_fmap1_acc[ch] += go * sampled;

            // grad_fmap2 corners (skip invalid via validity masks)
            float gof1 = go * f1;
            if (st.my0 * st.mx0 * st.mz0 > 0.f)
                atomicAdd(gbase_ch + st.y0 * stride_h_f2 + st.x0 * stride_w_f2 + st.z0, gof1 * st.w000);
            if (st.my0 * st.mx0 * st.mz1 > 0.f)
                atomicAdd(gbase_ch + st.y0 * stride_h_f2 + st.x0 * stride_w_f2 + st.z1, gof1 * st.w001);
            if (st.my0 * st.mx1 * st.mz0 > 0.f)
                atomicAdd(gbase_ch + st.y0 * stride_h_f2 + st.x1 * stride_w_f2 + st.z0, gof1 * st.w010);
            if (st.my0 * st.mx1 * st.mz1 > 0.f)
                atomicAdd(gbase_ch + st.y0 * stride_h_f2 + st.x1 * stride_w_f2 + st.z1, gof1 * st.w011);
            if (st.my1 * st.mx0 * st.mz0 > 0.f)
                atomicAdd(gbase_ch + st.y1 * stride_h_f2 + st.x0 * stride_w_f2 + st.z0, gof1 * st.w100);
            if (st.my1 * st.mx0 * st.mz1 > 0.f)
                atomicAdd(gbase_ch + st.y1 * stride_h_f2 + st.x0 * stride_w_f2 + st.z1, gof1 * st.w101);
            if (st.my1 * st.mx1 * st.mz0 > 0.f)
                atomicAdd(gbase_ch + st.y1 * stride_h_f2 + st.x1 * stride_w_f2 + st.z0, gof1 * st.w110);
            if (st.my1 * st.mx1 * st.mz1 > 0.f)
                atomicAdd(gbase_ch + st.y1 * stride_h_f2 + st.x1 * stride_w_f2 + st.z1, gof1 * st.w111);

            // Coord gradients: d(sampled)/d(t_H, t_W, t_D)
            float c000e = c000 * st.my0 * st.mx0 * st.mz0;
            float c001e = c001 * st.my0 * st.mx0 * st.mz1;
            float c010e = c010 * st.my0 * st.mx1 * st.mz0;
            float c011e = c011 * st.my0 * st.mx1 * st.mz1;
            float c100e = c100 * st.my1 * st.mx0 * st.mz0;
            float c101e = c101 * st.my1 * st.mx0 * st.mz1;
            float c110e = c110 * st.my1 * st.mx1 * st.mz0;
            float c111e = c111 * st.my1 * st.mx1 * st.mz1;

            float d_dy = (-1.f) * (w_x0 * (w_z0 * c000e + w_z1 * c001e)
                                 + w_x1 * (w_z0 * c010e + w_z1 * c011e))
                       + ( 1.f) * (w_x0 * (w_z0 * c100e + w_z1 * c101e)
                                 + w_x1 * (w_z0 * c110e + w_z1 * c111e));
            float d_dx = w_y0 * ((-1.f) * (w_z0 * c000e + w_z1 * c001e)
                                + ( 1.f) * (w_z0 * c010e + w_z1 * c011e))
                       + w_y1 * ((-1.f) * (w_z0 * c100e + w_z1 * c101e)
                                + ( 1.f) * (w_z0 * c110e + w_z1 * c111e));
            float d_dz = w_y0 * (w_x0 * (-c000e + c001e) + w_x1 * (-c010e + c011e))
                       + w_y1 * (w_x0 * (-c100e + c101e) + w_x1 * (-c110e + c111e));

            local_g_H += gof1 * d_dy * scale;
            local_g_D += gof1 * d_dz * scale;
            local_g_W += gof1 * d_dx * scale;
        }
    }

    // Warp-shuffle reduction of grad_fmap1_acc — each warp's lane 0 does
    // ONE atomicAdd per channel to global.
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int ch = 0; ch < C; ++ch) {
        float v = grad_fmap1_acc[ch];
        v += __shfl_xor_sync(mask, v, 16);
        v += __shfl_xor_sync(mask, v, 8);
        v += __shfl_xor_sync(mask, v, 4);
        v += __shfl_xor_sync(mask, v, 2);
        v += __shfl_xor_sync(mask, v, 1);
        if (lane == 0) {
            atomicAdd(grad_fmap1_b + ch * spatial + s, v);
        }
    }

    // -------------------------------------------------------------------
    // Coord gradient: warp-shuffle reduce within each warp, lane 0 of each
    // warp atomicAdds to global.  Cheap (3 scalars per warp, 8 warps).
    // (Reuses `mask` and `lane` from the grad_fmap1 reduction above.)
    // -------------------------------------------------------------------
    local_g_H += __shfl_xor_sync(mask, local_g_H, 16);
    local_g_H += __shfl_xor_sync(mask, local_g_H, 8);
    local_g_H += __shfl_xor_sync(mask, local_g_H, 4);
    local_g_H += __shfl_xor_sync(mask, local_g_H, 2);
    local_g_H += __shfl_xor_sync(mask, local_g_H, 1);
    local_g_D += __shfl_xor_sync(mask, local_g_D, 16);
    local_g_D += __shfl_xor_sync(mask, local_g_D, 8);
    local_g_D += __shfl_xor_sync(mask, local_g_D, 4);
    local_g_D += __shfl_xor_sync(mask, local_g_D, 2);
    local_g_D += __shfl_xor_sync(mask, local_g_D, 1);
    local_g_W += __shfl_xor_sync(mask, local_g_W, 16);
    local_g_W += __shfl_xor_sync(mask, local_g_W, 8);
    local_g_W += __shfl_xor_sync(mask, local_g_W, 4);
    local_g_W += __shfl_xor_sync(mask, local_g_W, 2);
    local_g_W += __shfl_xor_sync(mask, local_g_W, 1);
    if (lane == 0) {
        atomicAdd(grad_coords_b + 0 * spatial + s, local_g_H);
        atomicAdd(grad_coords_b + 1 * spatial + s, local_g_D);
        atomicAdd(grad_coords_b + 2 * spatial + s, local_g_W);
    }
}


// =============================================================================
// C++ launchers — invoked from Python
// =============================================================================

torch::Tensor corr_otf_forward_one_level(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int64_t radius,
    double scale
) {
    CHECK_INPUT(fmap1); CHECK_FP32(fmap1);
    CHECK_INPUT(fmap2); CHECK_FP32(fmap2);
    CHECK_INPUT(coords); CHECK_FP32(coords);
    TORCH_CHECK(fmap1.dim() == 5, "fmap1 must be 5D");
    TORCH_CHECK(fmap2.dim() == 5, "fmap2 must be 5D");
    TORCH_CHECK(coords.dim() == 5 && coords.size(1) == 3, "coords must be (B,3,H,W,D)");

    int B = fmap1.size(0);
    int C = fmap1.size(1);
    int H = fmap1.size(2);
    int W = fmap1.size(3);
    int D = fmap1.size(4);
    int Hi = fmap2.size(2);
    int Wi = fmap2.size(3);
    int Di = fmap2.size(4);
    int span = 2 * (int)radius + 1;
    int n_neighbors = span * span * span;
    float norm = 1.f / sqrtf((float)C);

    auto corr = torch::empty({B, n_neighbors, H, W, D}, fmap1.options());

    int n_blocks = B * H * W * D;
    corr_otf_forward_kernel<<<n_blocks, BLOCK_THREADS, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
        fmap1.data_ptr<float>(),
        fmap2.data_ptr<float>(),
        coords.data_ptr<float>(),
        corr.data_ptr<float>(),
        B, C, H, W, D, Hi, Wi, Di,
        (int)radius, (float)scale, norm
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return corr;
}


std::vector<torch::Tensor> corr_otf_backward_one_level(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor grad_corr,
    int64_t radius,
    double scale
) {
    CHECK_INPUT(fmap1); CHECK_FP32(fmap1);
    CHECK_INPUT(fmap2); CHECK_FP32(fmap2);
    CHECK_INPUT(coords); CHECK_FP32(coords);
    CHECK_INPUT(grad_corr); CHECK_FP32(grad_corr);

    int B = fmap1.size(0);
    int C = fmap1.size(1);
    int H = fmap1.size(2);
    int W = fmap1.size(3);
    int D = fmap1.size(4);
    int Hi = fmap2.size(2);
    int Wi = fmap2.size(3);
    int Di = fmap2.size(4);
    float norm = 1.f / sqrtf((float)C);

    auto grad_fmap1 = torch::zeros_like(fmap1);
    auto grad_fmap2 = torch::zeros_like(fmap2);
    auto grad_coords = torch::zeros_like(coords);

    // Backward: one block (256 threads = 8 warps) per source voxel.
    // Dispatch by feature_dim — template + launch_bounds enable per-thread
    // register accumulator + warp-shuffle reduction.
    int n_blocks = B * H * W * D;
    auto launch_tpl = [&](auto C_const) {
        constexpr int C_VAL = decltype(C_const)::value;
        corr_otf_backward_kernel<C_VAL><<<n_blocks, BLOCK_THREADS, 0,
                                          at::cuda::getCurrentCUDAStream()>>>(
            fmap1.data_ptr<float>(),
            fmap2.data_ptr<float>(),
            coords.data_ptr<float>(),
            grad_corr.data_ptr<float>(),
            grad_fmap1.data_ptr<float>(),
            grad_fmap2.data_ptr<float>(),
            grad_coords.data_ptr<float>(),
            B, H, W, D, Hi, Wi, Di,
            (int)radius, (float)scale, norm
        );
    };
    if (C == 16)       launch_tpl(std::integral_constant<int, 16>{});
    else if (C == 64)  launch_tpl(std::integral_constant<int, 64>{});
    else if (C == 128) launch_tpl(std::integral_constant<int, 128>{});
    else if (C == 256) launch_tpl(std::integral_constant<int, 256>{});
    else TORCH_CHECK(false, "Unsupported feature_dim=", C);
    AT_CUDA_CHECK(cudaGetLastError());
    return {grad_fmap1, grad_fmap2, grad_coords};
}


// =============================================================================
// V3: Cooperative-tiled forward kernel (faithful 3D port of alt_cuda_corr)
// =============================================================================
//
// Design (see docs/plans/2026-05-17-otf-v3-cooperative-tiling.md):
//   * BLOCK_T = 32 source voxels per block (one warp)
//   * CHANNEL_STRIDE = 32 channels per chunk
//   * Channels-last fmap layout (B, H, W, D, C) for coalesced loads
//   * Bilinear scatter at output write (NOT fmap2 interp before dot)
//
// gridDim  = (B, ceil(spatial / BLOCK_T))
// blockDim = BLOCK_T = 32
//
// Memory savings vs v1: ~180x fewer fmap2 reads (cooperative loading +
// scatter trick eliminates 8 corner fetches per neighbor per source).

#define BLOCK_T 32
#define CHANNEL_STRIDE 32

__global__ void corr_otf_forward_kernel_v3(
    const float* __restrict__ fmap1_cl,    // (B, H,  W,  D,  C) channels-last
    const float* __restrict__ fmap2_cl,    // (B, Hi, Wi, Di, C) channels-last
    const float* __restrict__ coords,      // (B, 3, H, W, D)   (y, d, x channel order)
    float* __restrict__ corr,              // (B, n_neighbors, H, W, D)
    int B, int C,
    int H, int W, int D,
    int Hi, int Wi, int Di,
    int radius,
    float scale,
    float norm
) {
    const int b = blockIdx.x;
    const int block_offset = blockIdx.y * BLOCK_T;
    const int tid = threadIdx.x;
    const int spatial = H * W * D;
    const int spatial_i = Hi * Wi * Di;

    const int rd = 2 * radius + 1;
    const int rd2 = rd * rd;
    const int n_neighbors = rd2 * rd;

    __shared__ float f1_smem[CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float f2_smem[CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float coords_y_smem[BLOCK_T];
    __shared__ float coords_x_smem[BLOCK_T];
    __shared__ float coords_z_smem[BLOCK_T];

    // Each thread loads coords for its own source voxel (tid is source-in-block here)
    const int s_self = block_offset + tid;
    const bool valid_self = (s_self < spatial);
    if (valid_self) {
        // Coord channel order matches v1: (H, D, W)
        coords_y_smem[tid] = coords[b * 3 * spatial + 0 * spatial + s_self] * scale;
        coords_z_smem[tid] = coords[b * 3 * spatial + 1 * spatial + s_self] * scale;
        coords_x_smem[tid] = coords[b * 3 * spatial + 2 * spatial + s_self] * scale;
    } else {
        coords_y_smem[tid] = 0;
        coords_z_smem[tid] = 0;
        coords_x_smem[tid] = 0;
    }
    __syncthreads();

    // Initialize THIS thread's output positions to 0 (we use += during scatter)
    if (valid_self) {
        for (int n = 0; n < n_neighbors; ++n) {
            corr[b * n_neighbors * spatial + n * spatial + s_self] = 0.f;
        }
    }
    __syncthreads();

    // Per-thread fractional parts (own source's coords)
    const float fy_self = valid_self ? (coords_y_smem[tid] - floorf(coords_y_smem[tid])) : 0.f;
    const float fx_self = valid_self ? (coords_x_smem[tid] - floorf(coords_x_smem[tid])) : 0.f;
    const float fz_self = valid_self ? (coords_z_smem[tid] - floorf(coords_z_smem[tid])) : 0.f;
    const float w_y0 = 1.f - fy_self, w_y1 = fy_self;
    const float w_x0 = 1.f - fx_self, w_x1 = fx_self;
    const float w_z0 = 1.f - fz_self, w_z1 = fz_self;

    // Output base for THIS thread's source voxel
    float* corr_self = (s_self < spatial)
        ? (corr + b * n_neighbors * spatial + s_self)
        : nullptr;

    // Outer loop: channel chunks
    for (int c0 = 0; c0 < C; c0 += CHANNEL_STRIDE) {
        // Cooperatively load f1 for the channel chunk.
        // Thread `tid` plays the role of "channel within chunk" here, looping
        // over all 32 source voxels.  After this loop f1_smem[c][k] is the
        // value of fmap1 at channel (c0+c), source voxel block_offset+k.
        for (int k = 0; k < BLOCK_T; ++k) {
            int s_k = block_offset + k;
            float v = 0.f;
            if (s_k < spatial && (c0 + tid) < C) {
                // channels-last: offset = ((b * spatial + s_k) * C + ch)
                v = fmap1_cl[(long)b * spatial * C + (long)s_k * C + (c0 + tid)];
            }
            f1_smem[tid][k] = v;
        }
        __syncthreads();

        // Triple loop over integer neighbor positions
        for (int iy_loc = 0; iy_loc <= rd; ++iy_loc) {
            for (int ix_loc = 0; ix_loc <= rd; ++ix_loc) {
                for (int iz_loc = 0; iz_loc <= rd; ++iz_loc) {
                    // Cooperatively load f2 for this (iy_loc, ix_loc, iz_loc)
                    // across all 32 source voxels (each source has different
                    // neighbor position based on its own coords).
                    for (int k = 0; k < BLOCK_T; ++k) {
                        int s_k = block_offset + k;
                        float v = 0.f;
                        if (s_k < spatial && (c0 + tid) < C) {
                            int y_base = (int)floorf(coords_y_smem[k]);
                            int x_base = (int)floorf(coords_x_smem[k]);
                            int z_base = (int)floorf(coords_z_smem[k]);
                            int y2 = y_base - radius + iy_loc;
                            int x2 = x_base - radius + ix_loc;
                            int z2 = z_base - radius + iz_loc;
                            if (y2 >= 0 && y2 < Hi && x2 >= 0 && x2 < Wi
                                && z2 >= 0 && z2 < Di) {
                                long fmap2_idx =
                                    ((long)b * spatial_i + (long)y2 * Wi * Di
                                     + (long)x2 * Di + (long)z2) * C
                                    + (c0 + tid);
                                v = fmap2_cl[fmap2_idx];
                            }
                        }
                        f2_smem[tid][k] = v;
                    }
                    __syncthreads();

                    // Dot product for THIS thread's source voxel
                    if (corr_self != nullptr) {
                        float dot = 0.f;
                        #pragma unroll
                        for (int k = 0; k < CHANNEL_STRIDE; ++k) {
                            dot += f1_smem[k][tid] * f2_smem[k][tid];
                        }

                        // Scatter to up to 8 output positions with trilinear weights.
                        // Corner (cy, cx, cz) ∈ {0,1}^3 contributes to output
                        // neighbor offset (iy_loc - cy, ix_loc - cx, iz_loc - cz).
                        // Bounds: each component must be in [0, rd-1].
                        float dot_norm = dot * norm;

                        // Neighbor index layout: (H_idx, D_idx, W_idx) -- H slowest,
                        // W fastest -- matching v1's convention (see v1 decode at
                        // line 171-174).  iy_loc varies H, ix_loc varies W,
                        // iz_loc varies D.  So n_idx = H_idx * rd^2 + D_idx * rd + W_idx
                        //                          = (iy-cy)*rd2 + (iz-cz)*rd + (ix-cx).

                        if (iy_loc < rd && ix_loc < rd && iz_loc < rd) {
                            int n_idx = iy_loc * rd2 + iz_loc * rd + ix_loc;
                            corr_self[n_idx * spatial] += dot_norm * w_y0 * w_x0 * w_z0;
                        }
                        if (iy_loc < rd && ix_loc < rd && iz_loc > 0) {
                            int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + ix_loc;
                            corr_self[n_idx * spatial] += dot_norm * w_y0 * w_x0 * w_z1;
                        }
                        if (iy_loc < rd && ix_loc > 0 && iz_loc < rd) {
                            int n_idx = iy_loc * rd2 + iz_loc * rd + (ix_loc - 1);
                            corr_self[n_idx * spatial] += dot_norm * w_y0 * w_x1 * w_z0;
                        }
                        if (iy_loc < rd && ix_loc > 0 && iz_loc > 0) {
                            int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                            corr_self[n_idx * spatial] += dot_norm * w_y0 * w_x1 * w_z1;
                        }
                        if (iy_loc > 0 && ix_loc < rd && iz_loc < rd) {
                            int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + ix_loc;
                            corr_self[n_idx * spatial] += dot_norm * w_y1 * w_x0 * w_z0;
                        }
                        if (iy_loc > 0 && ix_loc < rd && iz_loc > 0) {
                            int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + ix_loc;
                            corr_self[n_idx * spatial] += dot_norm * w_y1 * w_x0 * w_z1;
                        }
                        if (iy_loc > 0 && ix_loc > 0 && iz_loc < rd) {
                            int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + (ix_loc - 1);
                            corr_self[n_idx * spatial] += dot_norm * w_y1 * w_x1 * w_z0;
                        }
                        if (iy_loc > 0 && ix_loc > 0 && iz_loc > 0) {
                            int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                            corr_self[n_idx * spatial] += dot_norm * w_y1 * w_x1 * w_z1;
                        }
                    }
                    __syncthreads();
                }
            }
        }
    }
}


// V3 launcher.  fmap1/fmap2 must already be in channels-last layout
// (the Python wrapper does the permute + contiguous() before calling).
torch::Tensor corr_otf_forward_one_level_v3(
    torch::Tensor fmap1_cl,
    torch::Tensor fmap2_cl,
    torch::Tensor coords,
    int64_t radius,
    double scale
) {
    CHECK_INPUT(fmap1_cl); CHECK_FP32(fmap1_cl);
    CHECK_INPUT(fmap2_cl); CHECK_FP32(fmap2_cl);
    CHECK_INPUT(coords);   CHECK_FP32(coords);
    TORCH_CHECK(fmap1_cl.dim() == 5, "fmap1_cl must be (B,H,W,D,C)");
    TORCH_CHECK(fmap2_cl.dim() == 5, "fmap2_cl must be (B,Hi,Wi,Di,C)");
    TORCH_CHECK(coords.dim() == 5 && coords.size(1) == 3, "coords must be (B,3,H,W,D)");

    int B  = (int)fmap1_cl.size(0);
    int H  = (int)fmap1_cl.size(1);
    int W  = (int)fmap1_cl.size(2);
    int D  = (int)fmap1_cl.size(3);
    int C  = (int)fmap1_cl.size(4);
    int Hi = (int)fmap2_cl.size(1);
    int Wi = (int)fmap2_cl.size(2);
    int Di = (int)fmap2_cl.size(3);
    TORCH_CHECK(C == (int)fmap2_cl.size(4), "fmap1/fmap2 must have same C");
    TORCH_CHECK(C % CHANNEL_STRIDE == 0, "C must be a multiple of 32 for v3 (got C=", C, ")");

    int span = 2 * (int)radius + 1;
    int n_neighbors = span * span * span;
    float norm = 1.f / sqrtf((float)C);
    int spatial = H * W * D;
    int n_blocks_y = (spatial + BLOCK_T - 1) / BLOCK_T;

    auto corr = torch::empty({B, n_neighbors, H, W, D}, fmap1_cl.options());

    dim3 grid(B, n_blocks_y, 1);
    dim3 block(BLOCK_T, 1, 1);

    corr_otf_forward_kernel_v3<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        fmap1_cl.data_ptr<float>(),
        fmap2_cl.data_ptr<float>(),
        coords.data_ptr<float>(),
        corr.data_ptr<float>(),
        B, C, H, W, D, Hi, Wi, Di,
        (int)radius, (float)scale, norm
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return corr;
}


// =============================================================================
// V4: V3 + neighbor-group parallelism + atomicAdd scatter
// =============================================================================
// V3 was correct after the (H,D,W) scatter fix, but only ~3% SM occupancy
// because we launched B * spatial/32 blocks with 32 threads each -- way too
// few work units to saturate the 5090's 170 SMs.
//
// V4 fix: split the (iy_loc, ix_loc, iz_loc) iteration space across an
// EXTRA grid dimension (gridDim.z = G groups).  Each block now handles
// 32 source voxels x 1/G of the 1000 inner iterations.  Total blocks
// multiplied by G -> better occupancy.
//
// Cost: different blocks may write to the same output position via the
// 8-corner scatter, so we use atomicAdd.  Output is pre-zeroed by host
// (torch::zeros), so atomics accumulate correctly from any initial state.
//
// Tuning param G (n_groups): launcher uses G=8 by default; bench-tuneable.
// G=8 turns the 1024 block kernel (fm=32) into 8192 blocks, easily filling
// the 5090's 170 SMs across multiple waves.

__global__ void corr_otf_forward_kernel_v4(
    const float* __restrict__ fmap1_cl,
    const float* __restrict__ fmap2_cl,
    const float* __restrict__ coords,
    float* __restrict__ corr,
    int B, int C,
    int H, int W, int D,
    int Hi, int Wi, int Di,
    int radius,
    float scale,
    float norm
) {
    const int b = blockIdx.x;
    const int block_offset = blockIdx.y * BLOCK_T;
    const int group_id = blockIdx.z;
    const int group_count = gridDim.z;
    const int tid = threadIdx.x;
    const int spatial = H * W * D;
    const int spatial_i = Hi * Wi * Di;

    const int rd = 2 * radius + 1;
    const int rd2 = rd * rd;
    const int n_neighbors = rd2 * rd;
    const int rd_plus = rd + 1;
    const int total_inner = rd_plus * rd_plus * rd_plus;
    const int iter_per_group = (total_inner + group_count - 1) / group_count;
    const int iter_start = group_id * iter_per_group;
    const int iter_end = min(iter_start + iter_per_group, total_inner);
    if (iter_start >= iter_end) return;

    __shared__ float f1_smem[CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float f2_smem[CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float coords_y_smem[BLOCK_T];
    __shared__ float coords_x_smem[BLOCK_T];
    __shared__ float coords_z_smem[BLOCK_T];

    const int s_self = block_offset + tid;
    const bool valid_self = (s_self < spatial);
    if (valid_self) {
        coords_y_smem[tid] = coords[b * 3 * spatial + 0 * spatial + s_self] * scale;
        coords_z_smem[tid] = coords[b * 3 * spatial + 1 * spatial + s_self] * scale;
        coords_x_smem[tid] = coords[b * 3 * spatial + 2 * spatial + s_self] * scale;
    } else {
        coords_y_smem[tid] = 0;
        coords_z_smem[tid] = 0;
        coords_x_smem[tid] = 0;
    }
    __syncthreads();

    // Output pre-zeroed by host (torch::zeros).  No init loop here.

    const float fy_self = valid_self ? (coords_y_smem[tid] - floorf(coords_y_smem[tid])) : 0.f;
    const float fx_self = valid_self ? (coords_x_smem[tid] - floorf(coords_x_smem[tid])) : 0.f;
    const float fz_self = valid_self ? (coords_z_smem[tid] - floorf(coords_z_smem[tid])) : 0.f;
    const float w_y0 = 1.f - fy_self, w_y1 = fy_self;
    const float w_x0 = 1.f - fx_self, w_x1 = fx_self;
    const float w_z0 = 1.f - fz_self, w_z1 = fz_self;

    float* corr_self = valid_self
        ? (corr + b * n_neighbors * spatial + s_self)
        : nullptr;

    for (int c0 = 0; c0 < C; c0 += CHANNEL_STRIDE) {
        // Load f1 chunk (cooperative)
        for (int k = 0; k < BLOCK_T; ++k) {
            int s_k = block_offset + k;
            float v = 0.f;
            if (s_k < spatial && (c0 + tid) < C) {
                v = fmap1_cl[(long)b * spatial * C + (long)s_k * C + (c0 + tid)];
            }
            f1_smem[tid][k] = v;
        }
        __syncthreads();

        // Iterate only over this group's slice of (iy_loc, ix_loc, iz_loc)
        for (int linear = iter_start; linear < iter_end; ++linear) {
            int iy_loc = linear / (rd_plus * rd_plus);
            int rem = linear % (rd_plus * rd_plus);
            int ix_loc = rem / rd_plus;
            int iz_loc = rem % rd_plus;

            // Load f2 chunk for this (iy, ix, iz) across all 32 sources
            for (int k = 0; k < BLOCK_T; ++k) {
                int s_k = block_offset + k;
                float v = 0.f;
                if (s_k < spatial && (c0 + tid) < C) {
                    int y_base = (int)floorf(coords_y_smem[k]);
                    int x_base = (int)floorf(coords_x_smem[k]);
                    int z_base = (int)floorf(coords_z_smem[k]);
                    int y2 = y_base - radius + iy_loc;
                    int x2 = x_base - radius + ix_loc;
                    int z2 = z_base - radius + iz_loc;
                    if (y2 >= 0 && y2 < Hi && x2 >= 0 && x2 < Wi
                        && z2 >= 0 && z2 < Di) {
                        long fmap2_idx =
                            ((long)b * spatial_i + (long)y2 * Wi * Di
                             + (long)x2 * Di + (long)z2) * C
                            + (c0 + tid);
                        v = fmap2_cl[fmap2_idx];
                    }
                }
                f2_smem[tid][k] = v;
            }
            __syncthreads();

            if (corr_self != nullptr) {
                float dot = 0.f;
                #pragma unroll
                for (int k = 0; k < CHANNEL_STRIDE; ++k) {
                    dot += f1_smem[k][tid] * f2_smem[k][tid];
                }
                float dot_norm = dot * norm;

                // 8-corner trilinear scatter via atomicAdd.  Different blocks
                // (different groups) may target the same (s, n_idx) so atomics
                // are required for correctness.
                if (iy_loc < rd && ix_loc < rd && iz_loc < rd) {
                    int n_idx = iy_loc * rd2 + iz_loc * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x0 * w_z0);
                }
                if (iy_loc < rd && ix_loc < rd && iz_loc > 0) {
                    int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x0 * w_z1);
                }
                if (iy_loc < rd && ix_loc > 0 && iz_loc < rd) {
                    int n_idx = iy_loc * rd2 + iz_loc * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x1 * w_z0);
                }
                if (iy_loc < rd && ix_loc > 0 && iz_loc > 0) {
                    int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x1 * w_z1);
                }
                if (iy_loc > 0 && ix_loc < rd && iz_loc < rd) {
                    int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x0 * w_z0);
                }
                if (iy_loc > 0 && ix_loc < rd && iz_loc > 0) {
                    int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x0 * w_z1);
                }
                if (iy_loc > 0 && ix_loc > 0 && iz_loc < rd) {
                    int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x1 * w_z0);
                }
                if (iy_loc > 0 && ix_loc > 0 && iz_loc > 0) {
                    int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x1 * w_z1);
                }
            }
            __syncthreads();
        }
    }
}


torch::Tensor corr_otf_forward_one_level_v4(
    torch::Tensor fmap1_cl,
    torch::Tensor fmap2_cl,
    torch::Tensor coords,
    int64_t radius,
    double scale,
    int64_t n_groups
) {
    CHECK_INPUT(fmap1_cl); CHECK_FP32(fmap1_cl);
    CHECK_INPUT(fmap2_cl); CHECK_FP32(fmap2_cl);
    CHECK_INPUT(coords);   CHECK_FP32(coords);
    TORCH_CHECK(fmap1_cl.dim() == 5, "fmap1_cl must be (B,H,W,D,C)");
    TORCH_CHECK(fmap2_cl.dim() == 5, "fmap2_cl must be (B,Hi,Wi,Di,C)");
    TORCH_CHECK(coords.dim() == 5 && coords.size(1) == 3, "coords must be (B,3,H,W,D)");
    TORCH_CHECK(n_groups >= 1, "n_groups must be >= 1");

    int B  = (int)fmap1_cl.size(0);
    int H  = (int)fmap1_cl.size(1);
    int W  = (int)fmap1_cl.size(2);
    int D  = (int)fmap1_cl.size(3);
    int C  = (int)fmap1_cl.size(4);
    int Hi = (int)fmap2_cl.size(1);
    int Wi = (int)fmap2_cl.size(2);
    int Di = (int)fmap2_cl.size(3);
    TORCH_CHECK(C == (int)fmap2_cl.size(4), "fmap1/fmap2 must have same C");
    TORCH_CHECK(C % CHANNEL_STRIDE == 0, "C must be a multiple of 32 for v4");

    int span = 2 * (int)radius + 1;
    int n_neighbors = span * span * span;
    float norm = 1.f / sqrtf((float)C);
    int spatial = H * W * D;
    int n_blocks_y = (spatial + BLOCK_T - 1) / BLOCK_T;

    // Pre-zero output (atomicAdd accumulates from zero)
    auto corr = torch::zeros({B, n_neighbors, H, W, D}, fmap1_cl.options());

    dim3 grid(B, n_blocks_y, (int)n_groups);
    dim3 block(BLOCK_T, 1, 1);

    corr_otf_forward_kernel_v4<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        fmap1_cl.data_ptr<float>(),
        fmap2_cl.data_ptr<float>(),
        coords.data_ptr<float>(),
        corr.data_ptr<float>(),
        B, C, H, W, D, Hi, Wi, Di,
        (int)radius, (float)scale, norm
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return corr;
}


// =============================================================================
// V5: V4 + multi-warp cooperative tiling (4 warps per block)
// =============================================================================
// V4 used 32 threads/block (1 warp) which limits per-block work parallelism.
// V5 uses 128 threads/block (4 warps) where:
//   * All 4 warps SHARE one f1 region in smem (loaded by warp 0 once per chunk)
//   * Each warp has its OWN f2 region (4 private slabs)
//   * 4 warps partition this block's (iy, ix, iz) iterations 4-ways
// Cross-warp scatter conflicts resolved via atomicAdd (same as v4).
//
// Expected speedup over v4: ~4x at fm=16/32 (more concurrent work per block).
// gridDim same as v4 (B, spatial/32, n_groups); blockDim 128 instead of 32.

#define V5_NUM_WARPS 4

__global__ void corr_otf_forward_kernel_v5(
    const float* __restrict__ fmap1_cl,
    const float* __restrict__ fmap2_cl,
    const float* __restrict__ coords,
    float* __restrict__ corr,
    int B, int C,
    int H, int W, int D,
    int Hi, int Wi, int Di,
    int radius,
    float scale,
    float norm
) {
    const int b = blockIdx.x;
    const int block_offset = blockIdx.y * BLOCK_T;
    const int group_id = blockIdx.z;
    const int group_count = gridDim.z;

    const int tid = threadIdx.x;          // 0..127
    const int warp_id = tid >> 5;          // 0..3
    const int lane = tid & 31;             // 0..31 (== source index within block)

    const int spatial = H * W * D;
    const int spatial_i = Hi * Wi * Di;

    const int rd = 2 * radius + 1;
    const int rd2 = rd * rd;
    const int n_neighbors = rd2 * rd;
    const int rd_plus = rd + 1;
    const int total_inner = rd_plus * rd_plus * rd_plus;
    const int iter_per_group = (total_inner + group_count - 1) / group_count;
    const int group_start = group_id * iter_per_group;
    const int group_end = min(group_start + iter_per_group, total_inner);
    if (group_start >= group_end) return;

    const int iter_per_warp = (group_end - group_start + V5_NUM_WARPS - 1) / V5_NUM_WARPS;
    const int warp_iter_start = group_start + warp_id * iter_per_warp;
    const int warp_iter_end = min(warp_iter_start + iter_per_warp, group_end);

    // Shared memory layout:
    //   f1_smem:  shared by all warps (32 channels x 32 sources)
    //   f2_smem:  per-warp (NUM_WARPS x 32 channels x 32 sources)
    //   coords:   3 arrays of 32 sources
    __shared__ float f1_smem[CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float f2_smem[V5_NUM_WARPS][CHANNEL_STRIDE][BLOCK_T + 1];
    __shared__ float coords_y_smem[BLOCK_T];
    __shared__ float coords_x_smem[BLOCK_T];
    __shared__ float coords_z_smem[BLOCK_T];

    // Load coords (warp 0 only -- lanes 0..31 each load one source)
    if (warp_id == 0) {
        int s_w0 = block_offset + lane;
        if (s_w0 < spatial) {
            coords_y_smem[lane] = coords[b * 3 * spatial + 0 * spatial + s_w0] * scale;
            coords_z_smem[lane] = coords[b * 3 * spatial + 1 * spatial + s_w0] * scale;
            coords_x_smem[lane] = coords[b * 3 * spatial + 2 * spatial + s_w0] * scale;
        } else {
            coords_y_smem[lane] = 0.f;
            coords_z_smem[lane] = 0.f;
            coords_x_smem[lane] = 0.f;
        }
    }
    __syncthreads();

    // Per-lane fractional weights (lane = source index)
    const int s_self = block_offset + lane;
    const bool valid_self = (s_self < spatial);
    const float fy_self = valid_self ? (coords_y_smem[lane] - floorf(coords_y_smem[lane])) : 0.f;
    const float fx_self = valid_self ? (coords_x_smem[lane] - floorf(coords_x_smem[lane])) : 0.f;
    const float fz_self = valid_self ? (coords_z_smem[lane] - floorf(coords_z_smem[lane])) : 0.f;
    const float w_y0 = 1.f - fy_self, w_y1 = fy_self;
    const float w_x0 = 1.f - fx_self, w_x1 = fx_self;
    const float w_z0 = 1.f - fz_self, w_z1 = fz_self;

    float* corr_self = valid_self
        ? (corr + b * n_neighbors * spatial + s_self)
        : nullptr;

    for (int c0 = 0; c0 < C; c0 += CHANNEL_STRIDE) {
        // Load f1: warp 0 fills f1_smem[channel][source]
        // (lane plays "channel" role here; loops over BLOCK_T sources)
        if (warp_id == 0) {
            for (int k = 0; k < BLOCK_T; ++k) {
                int s_k = block_offset + k;
                float v = 0.f;
                if (s_k < spatial && (c0 + lane) < C) {
                    v = fmap1_cl[(long)b * spatial * C + (long)s_k * C + (c0 + lane)];
                }
                f1_smem[lane][k] = v;
            }
        }
        __syncthreads();

        // This warp's slice of inner iterations
        for (int linear = warp_iter_start; linear < warp_iter_end; ++linear) {
            int iy_loc = linear / (rd_plus * rd_plus);
            int rem = linear % (rd_plus * rd_plus);
            int ix_loc = rem / rd_plus;
            int iz_loc = rem % rd_plus;

            // Load f2 into this warp's private slab
            for (int k = 0; k < BLOCK_T; ++k) {
                int s_k = block_offset + k;
                float v = 0.f;
                if (s_k < spatial && (c0 + lane) < C) {
                    int y_base = (int)floorf(coords_y_smem[k]);
                    int x_base = (int)floorf(coords_x_smem[k]);
                    int z_base = (int)floorf(coords_z_smem[k]);
                    int y2 = y_base - radius + iy_loc;
                    int x2 = x_base - radius + ix_loc;
                    int z2 = z_base - radius + iz_loc;
                    if (y2 >= 0 && y2 < Hi && x2 >= 0 && x2 < Wi
                        && z2 >= 0 && z2 < Di) {
                        long fmap2_idx =
                            ((long)b * spatial_i + (long)y2 * Wi * Di
                             + (long)x2 * Di + (long)z2) * C
                            + (c0 + lane);
                        v = fmap2_cl[fmap2_idx];
                    }
                }
                f2_smem[warp_id][lane][k] = v;
            }
            __syncwarp();  // intra-warp sync after f2 load

            if (corr_self != nullptr) {
                float dot = 0.f;
                #pragma unroll
                for (int k = 0; k < CHANNEL_STRIDE; ++k) {
                    dot += f1_smem[k][lane] * f2_smem[warp_id][k][lane];
                }
                float dot_norm = dot * norm;

                // 8-corner trilinear scatter via atomicAdd
                if (iy_loc < rd && ix_loc < rd && iz_loc < rd) {
                    int n_idx = iy_loc * rd2 + iz_loc * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x0 * w_z0);
                }
                if (iy_loc < rd && ix_loc < rd && iz_loc > 0) {
                    int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x0 * w_z1);
                }
                if (iy_loc < rd && ix_loc > 0 && iz_loc < rd) {
                    int n_idx = iy_loc * rd2 + iz_loc * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x1 * w_z0);
                }
                if (iy_loc < rd && ix_loc > 0 && iz_loc > 0) {
                    int n_idx = iy_loc * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y0 * w_x1 * w_z1);
                }
                if (iy_loc > 0 && ix_loc < rd && iz_loc < rd) {
                    int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x0 * w_z0);
                }
                if (iy_loc > 0 && ix_loc < rd && iz_loc > 0) {
                    int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + ix_loc;
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x0 * w_z1);
                }
                if (iy_loc > 0 && ix_loc > 0 && iz_loc < rd) {
                    int n_idx = (iy_loc - 1) * rd2 + iz_loc * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x1 * w_z0);
                }
                if (iy_loc > 0 && ix_loc > 0 && iz_loc > 0) {
                    int n_idx = (iy_loc - 1) * rd2 + (iz_loc - 1) * rd + (ix_loc - 1);
                    atomicAdd(&corr_self[n_idx * spatial], dot_norm * w_y1 * w_x1 * w_z1);
                }
            }
            __syncwarp();
        }
        __syncthreads();  // before next channel chunk's f1 load
    }
}


torch::Tensor corr_otf_forward_one_level_v5(
    torch::Tensor fmap1_cl,
    torch::Tensor fmap2_cl,
    torch::Tensor coords,
    int64_t radius,
    double scale,
    int64_t n_groups
) {
    CHECK_INPUT(fmap1_cl); CHECK_FP32(fmap1_cl);
    CHECK_INPUT(fmap2_cl); CHECK_FP32(fmap2_cl);
    CHECK_INPUT(coords);   CHECK_FP32(coords);
    TORCH_CHECK(fmap1_cl.dim() == 5, "fmap1_cl must be (B,H,W,D,C)");
    TORCH_CHECK(fmap2_cl.dim() == 5, "fmap2_cl must be (B,Hi,Wi,Di,C)");
    TORCH_CHECK(coords.dim() == 5 && coords.size(1) == 3, "coords must be (B,3,H,W,D)");
    TORCH_CHECK(n_groups >= 1, "n_groups must be >= 1");

    int B  = (int)fmap1_cl.size(0);
    int H  = (int)fmap1_cl.size(1);
    int W  = (int)fmap1_cl.size(2);
    int D  = (int)fmap1_cl.size(3);
    int C  = (int)fmap1_cl.size(4);
    int Hi = (int)fmap2_cl.size(1);
    int Wi = (int)fmap2_cl.size(2);
    int Di = (int)fmap2_cl.size(3);
    TORCH_CHECK(C == (int)fmap2_cl.size(4), "fmap1/fmap2 must have same C");
    TORCH_CHECK(C % CHANNEL_STRIDE == 0, "C must be a multiple of 32 for v5");

    int span = 2 * (int)radius + 1;
    int n_neighbors = span * span * span;
    float norm = 1.f / sqrtf((float)C);
    int spatial = H * W * D;
    int n_blocks_y = (spatial + BLOCK_T - 1) / BLOCK_T;

    auto corr = torch::zeros({B, n_neighbors, H, W, D}, fmap1_cl.options());

    dim3 grid(B, n_blocks_y, (int)n_groups);
    dim3 block(V5_NUM_WARPS * 32, 1, 1);   // 128 threads = 4 warps

    corr_otf_forward_kernel_v5<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        fmap1_cl.data_ptr<float>(),
        fmap2_cl.data_ptr<float>(),
        coords.data_ptr<float>(),
        corr.data_ptr<float>(),
        B, C, H, W, D, Hi, Wi, Di,
        (int)radius, (float)scale, norm
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return corr;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_one_level",  &corr_otf_forward_one_level,
          "On-the-fly 3D correlation forward (single pyramid level, v1)");
    m.def("backward_one_level", &corr_otf_backward_one_level,
          "On-the-fly 3D correlation backward (single pyramid level, v1)");
    m.def("forward_one_level_v3", &corr_otf_forward_one_level_v3,
          "V3 cooperative-tiled forward (channels-last fmaps)");
    m.def("forward_one_level_v4", &corr_otf_forward_one_level_v4,
          "V4 cooperative-tiled forward with neighbor-group parallelism");
    m.def("forward_one_level_v5", &corr_otf_forward_one_level_v5,
          "V5 cooperative-tiled forward with multi-warp + neighbor-group parallelism");
}
