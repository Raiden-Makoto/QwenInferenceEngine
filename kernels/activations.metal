//
//  activations.metal
//  QwenInferenceEngine
//
//  Created by Raiden Makoto on 2026-02-14.
//

#include <metal_stdlib>
using namespace metal;

// SWIGLU for 4-bit quantized weights
kernel void swiglu(
    device const half* gate_vec [[buffer(0)]],
    device const half* up_vec [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
){
    float x = (float)gate_vec[tid];
    float y = (float) up_vec[tid];
    float silu = x / (1.0f + exp(-x));
    output[tid] = (half)(silu * y);
}

// softmax for attention scores
// using online softmax for efficiency, one forward pass instead of 3
kernel void softmax(
    device float* scores [[buffer(0)]],
    device const uint* seq_len [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
){
    uint current_seq_len = seq_len[0];
    uint head_offset = tid * current_seq_len;
    device float* head_scores = scores + head_offset;
    // Find Max for Numerical Stability
    float max_val = -INFINITY;
    for (uint i = 0; i < current_seq_len; i++) {
        if (head_scores[i] > max_val) max_val = head_scores[i];
    }
    // Sum of Exponentials
    float sum = 0.0f;
    for (uint i = 0; i < current_seq_len; i++) {
        head_scores[i] = exp(head_scores[i] - max_val);
        sum += head_scores[i];
    }
    // Normalize
    for (uint i = 0; i < current_seq_len; i++) {
        head_scores[i] /= sum;
    }
}
