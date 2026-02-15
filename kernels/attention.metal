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
    constant uint& layer_idx         [[buffer(4)]], // Added to prevent overwrite
    uint tid [[thread_position_in_grid]])
{
    uint q_head = tid;
    uint k_head = q_head / 6;
    device const half* q_ptr = q_vec + (q_head * 128);
    
    // Isolate the layer in the cache
    uint layer_offset = layer_idx * 1024 * 256;
    uint current_seq_len = seq_len[0];
    
    for (uint i = 0; i < current_seq_len; i++) {
        device const half* k_ptr = k_cache + layer_offset + (i * 256) + (k_head * 128);
        
        float dot = 0.0f;
        for (uint d = 0; d < 128; d++) {
            dot += (float)q_ptr[d] * (float)k_ptr[d];
        }
        scores[q_head * current_seq_len + i] = dot * 0.088388f;
    }
}

kernel void attn_weighted_sum(
    device const float* scores       [[buffer(0)]],
    device const half* v_cache       [[buffer(1)]],
    device half* output              [[buffer(2)]],
    device const uint* seq_len       [[buffer(3)]],
    constant uint& layer_idx         [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint current_seq_len = seq_len[0];
    uint q_head = tid / 128;
    uint head_dim_idx = tid % 128;
    uint v_head = q_head / 6;
    
    uint layer_offset = layer_idx * 1024 * 256;
    float weighted_sum = 0.0f;
    
    for (uint i = 0; i < current_seq_len; i++) {
        float attn_weight = scores[q_head * current_seq_len + i];
        
        device const half* v_ptr = v_cache + layer_offset + (i * 256) + (v_head * 128) + head_dim_idx;
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
