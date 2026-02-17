//
//  attention.metal
//  QwenInferenceEngine
//

#include <metal_stdlib>
using namespace metal;

kernel void attention_scores(
    device const half* q_vec         [[buffer(0)]],
    device const half* k_cache       [[buffer(1)]],
    device float* scores             [[buffer(2)]],
    device const uint* seq_len       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint q_head = tid;
    uint k_head = q_head / 6;
    device const half* q_ptr = q_vec + (q_head * 128);
    
    uint current_seq_len = seq_len[0];
    
    for (uint i = 0; i < current_seq_len; i++) {
        // Swift already offset k_cache to the correct layer. We just index the token and head.
        device const half* k_ptr = k_cache + (i * 256) + (k_head * 128);
        
        float dot = 0.0f;
        for (uint d = 0; d < 128; d++) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        scores[q_head * current_seq_len + i] = dot * 0.088388f; // Qwen 2.5 scaling factor
    }
}

kernel void attn_weighted_sum(
    device const float* scores       [[buffer(0)]],
    device const half* v_cache       [[buffer(1)]],
    device half* output              [[buffer(2)]],
    device const uint* seq_len       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint current_seq_len = seq_len[0];
    uint q_head = tid / 128;
    uint head_dim_idx = tid % 128;
    uint v_head = q_head / 6;
    
    float weighted_sum = 0.0f;
    
    for (uint i = 0; i < current_seq_len; i++) {
        float attn_weight = scores[q_head * current_seq_len + i];
        
        // Swift already offset v_cache.
        device const half* v_ptr = v_cache + (i * 256) + (v_head * 128) + head_dim_idx;
        weighted_sum += attn_weight * (float)(*v_ptr);
    }
    output[tid] = (half)weighted_sum;
}

kernel void residual_add(
    device const half* x [[buffer(0)]],
    device const half* attn_x [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    output[tid] = x[tid] + attn_x[tid];
}

kernel void write_kv_cache(
    device const half* k_in          [[buffer(0)]],
    device const half* v_in          [[buffer(1)]],
    device half* k_cache             [[buffer(2)]],
    device half* v_cache             [[buffer(3)]],
    device const uint* seq_len       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 256) return; // QwenConfig.kDim
    
    uint position = seq_len[0] - 1;
    if (position >= 1024) return; // Max context window guard
    
    uint cache_idx = (position * 256) + tid;
    
    k_cache[cache_idx] = k_in[tid];
    v_cache[cache_idx] = v_in[tid];
}
