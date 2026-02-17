//
//  rmsnorm.metal
//  QwenInferenceEngine
//

#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_q4(
    device const half* input  [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output       [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]
){
    float thread_sum_sq = 0.0f;
    for (uint i = tid; i < 1536; i += threads_per_tg) {
        float val = (float)input[i];
        thread_sum_sq += val * val;
    }

    thread_sum_sq = simd_sum(thread_sum_sq);

    threadgroup float shared_sums[32]; // Max 32 SIMD groups for Apple Silicon
    uint simd_id = tid / 32;

    if (tid % 32 == 0) {
        shared_sums[simd_id] = thread_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float final_scale;
    if (tid == 0){
        float total_sum_sq = 0.0f;
        uint num_simd = (threads_per_tg + 31) / 32;
        for (uint i = 0; i < num_simd; i++) {
            total_sum_sq += shared_sums[i];
        }
        final_scale = rsqrt(total_sum_sq / 1536.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < 1536; i += threads_per_tg) {
        output[i] = (half)((float)input[i] * final_scale) * weight[i];
    }
}
