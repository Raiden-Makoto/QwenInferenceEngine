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
    uint tid [[thread_position_in_grid]]
){
    float val = (float)input[tid];
    float sq_val = val * val;
    float sum_sq = simd_sum(sq_val);
    
    // Fixed array size to match 1536 / 32
    threadgroup float shared_sums[48];

    if ((tid % 32) == 0) {
        shared_sums[tid / 32] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float final_scale = 1.0f;
    if (tid == 0){
        float total_sum_sq = 0.0f;
        for (int i = 0; i < 48; i++) total_sum_sq += shared_sums[i];
        
        // Fixed division precision and Qwen 1e-6 epsilon
        final_scale = rsqrt(total_sum_sq / 1536.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    output[tid] = (half)(val * final_scale) * weight[tid];
}
