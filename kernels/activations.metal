//
//  activations.metal
//  QwenInferenceEngine
//

#include <metal_stdlib>
using namespace metal;

kernel void swiglu(
    device half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    float x = (float)gate[tid];
    
    // Numerically Stable SiLU
    float silu;
    if (x > 15.0f) {
        silu = x;
    } else if (x < -15.0f) {
        silu = 0.0f;
    } else {
        silu = x / (1.0f + exp(-x));
    }
    
    gate[tid] = (half)(silu * (float)up[tid]);
}

kernel void softmax(
    device float* scores       [[buffer(0)]],
    device const uint* seq_len [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    uint current_seq_len = seq_len[0];
    uint q_head = tid;
    
    uint start_idx = q_head * current_seq_len;
    
    float max_val = -1e9f;
    for (uint i = 0; i < current_seq_len; i++) {
        max_val = max(max_val, scores[start_idx + i]);
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < current_seq_len; i++) {
        float e = exp(scores[start_idx + i] - max_val);
        scores[start_idx + i] = e;
        sum += e;
    }
    
    if (sum == 0.0f) sum = 1e-9f;
    for (uint i = 0; i < current_seq_len; i++) {
        scores[start_idx + i] /= sum;
    }
}
