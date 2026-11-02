#include "mhc.cuh"

// two input gates, two output gates, 2x2 mixing matrix
#define MHC_N_LOGITS 8

static __device__ __forceinline__ float mhc_lse2(const float x, const float y) {
    const float mx = fmaxf(x, y);
    return mx + logf(expf(x - mx) + expf(y - mx));
}

// one block per row, see ggml_mhc_route for the packed output layout
template <int block_size>
static __global__ void mhc_route_f32(
        const float * a,
        const float * b,
        const float * w,
        const float * bias,
        float       * dst,
        const int64_t d,
        const int64_t n_tok,
        const int64_t sa1,
        const int64_t sa2,
        const int64_t sb1,
        const int64_t sb2,
        const int     n_iter) {
    ggml_cuda_pdl_lc();

    constexpr int n_warps = block_size/WARP_SIZE;

    __shared__ float s_red[MHC_N_LOGITS + 1][n_warps];
    __shared__ float s_pre[2];

    const int64_t t   = blockIdx.x;
    const int64_t s   = blockIdx.y;
    const int     tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const float * va = a + t*sa1 + s*sa2;
    const float * vb = b + t*sb1 + s*sb2;
    float       * vd = dst + (s*n_tok + t)*(d + 6);

    // acc[0..7]: unnormalized logits, acc[8]: sum of squares over both streams
    float acc[MHC_N_LOGITS + 1];
#pragma unroll
    for (int k = 0; k <= MHC_N_LOGITS; ++k) {
        acc[k] = 0.0f;
    }

    for (int64_t i = tid; i < d; i += block_size) {
        const float xa = va[i];
        const float xb = vb[i];
        acc[MHC_N_LOGITS] += xa*xa + xb*xb;
#pragma unroll
        for (int k = 0; k < MHC_N_LOGITS; ++k) {
            acc[k] += w[k*2*d + i]*xa + w[k*2*d + d + i]*xb;
        }
    }

#pragma unroll
    for (int k = 0; k <= MHC_N_LOGITS; ++k) {
        acc[k] = warp_reduce_sum(acc[k]);
    }

    if constexpr (n_warps > 1) {
        const int warp_id = tid / WARP_SIZE;
        const int lane_id = tid % WARP_SIZE;

        if (lane_id == 0) {
#pragma unroll
            for (int k = 0; k <= MHC_N_LOGITS; ++k) {
                s_red[k][warp_id] = acc[k];
            }
        }
        __syncthreads();

        if (warp_id == 0) {
#pragma unroll
            for (int k = 0; k <= MHC_N_LOGITS; ++k) {
                acc[k] = warp_reduce_sum(lane_id < n_warps ? s_red[k][lane_id] : 0.0f);
            }
        }
    }

    if (tid == 0) {
        const float scale = rsqrtf(acc[MHC_N_LOGITS]/(2.0f*d) + 1e-6f);

        float lg[MHC_N_LOGITS];
#pragma unroll
        for (int k = 0; k < MHC_N_LOGITS; ++k) {
            lg[k] = acc[k]*scale + bias[k];
        }

        s_pre[0] = 1.0f/(1.0f + expf(-lg[0]));
        s_pre[1] = 1.0f/(1.0f + expf(-lg[1]));

        // sinkhorn in log space: normalize the rows, then the columns
        float m00 = lg[4];
        float m01 = lg[5];
        float m10 = lg[6];
        float m11 = lg[7];
        for (int it = 0; it < n_iter; ++it) {
            float lse;
            lse = mhc_lse2(m00, m01); m00 -= lse; m01 -= lse;
            lse = mhc_lse2(m10, m11); m10 -= lse; m11 -= lse;
            lse = mhc_lse2(m00, m10); m00 -= lse; m10 -= lse;
            lse = mhc_lse2(m01, m11); m01 -= lse; m11 -= lse;
        }

        vd[d + 0] = 2.0f/(1.0f + expf(-lg[2]));
        vd[d + 1] = 2.0f/(1.0f + expf(-lg[3]));
        vd[d + 2] = expf(m00);
        vd[d + 3] = expf(m01);
        vd[d + 4] = expf(m10);
        vd[d + 5] = expf(m11);
    }
    __syncthreads();

    const float pre0 = s_pre[0];
    const float pre1 = s_pre[1];

    for (int64_t i = tid; i < d; i += block_size) {
        vd[i] = pre0*va[i] + pre1*vb[i];
    }
}

static __global__ void mhc_merge_f32(
        const float * a,
        const float * b,
        const float * x,
        const float * p,
        float       * dst,
        const int64_t d,
        const int64_t n_tok,
        const int64_t n_rows,
        const int64_t sa1,
        const int64_t sa2,
        const int64_t sb1,
        const int64_t sb2,
        const int64_t sx1,
        const int64_t sx2,
        const int64_t sp1,
        const int64_t sp2) {
    ggml_cuda_pdl_lc();

    const int64_t ir = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (ir >= d*n_rows) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i   = ir % d;
    const int64_t row = ir / d;
    const int64_t t   = row % n_tok;
    const int64_t s   = row / n_tok;

    // gates follow the routed branch in the packed router output
    const float * vp = p + t*sp1 + s*sp2 + d;

    const float va = a[t*sa1 + s*sa2 + i];
    const float vb = b[t*sb1 + s*sb2 + i];
    const float vx = x[t*sx1 + s*sx2 + i];

    float * vd = dst + row*2*d;

    vd[i]     = vp[2]*va + vp[3]*vb + vp[0]*vx;
    vd[d + i] = vp[4]*va + vp[5]*vb + vp[1]*vx;
}

void ggml_cuda_op_mhc_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a    = dst->src[0];
    const ggml_tensor * b    = dst->src[1];
    const ggml_tensor * w    = dst->src[2];
    const ggml_tensor * bias = dst->src[3];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32 && bias->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_is_contiguous(bias) && ggml_is_contiguous(dst));

    const int64_t d     = a->ne[0];
    const int64_t n_tok = a->ne[1];
    const int64_t n_str = a->ne[2];

    GGML_ASSERT(w->ne[0] == 2*d && w->ne[1] == MHC_N_LOGITS);
    GGML_ASSERT(n_str <= 65535);

    const int n_iter = ggml_get_op_params_i32(dst, 0);

    constexpr int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims(n_tok, n_str, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(mhc_route_f32<block_size>, launch_params,
            (const float *) a->data, (const float *) b->data, (const float *) w->data, (const float *) bias->data, (float *) dst->data,
            d, n_tok,
            a->nb[1] / sizeof(float), a->nb[2] / sizeof(float),
            b->nb[1] / sizeof(float), b->nb[2] / sizeof(float),
            n_iter);
}

void ggml_cuda_op_mhc_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    const ggml_tensor * x = dst->src[2];
    const ggml_tensor * p = dst->src[3];

    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && p->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float));
    GGML_ASSERT(x->nb[0] == sizeof(float) && p->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t d      = a->ne[0];
    const int64_t n_tok  = a->ne[1];
    const int64_t n_rows = n_tok*a->ne[2];

    constexpr int block_size = 256;
    const int64_t n_blocks = (d*n_rows + block_size - 1) / block_size;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims(n_blocks, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(mhc_merge_f32, launch_params,
            (const float *) a->data, (const float *) b->data, (const float *) x->data, (const float *) p->data, (float *) dst->data,
            d, n_tok, n_rows,
            a->nb[1] / sizeof(float), a->nb[2] / sizeof(float),
            b->nb[1] / sizeof(float), b->nb[2] / sizeof(float),
            x->nb[1] / sizeof(float), x->nb[2] / sizeof(float),
            p->nb[1] / sizeof(float), p->nb[2] / sizeof(float));
}
