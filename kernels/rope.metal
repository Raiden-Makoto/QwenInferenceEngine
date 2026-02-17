//
//  rope.metal
//  QwenInferenceEngine
//

#include <metal_stdlib>
using namespace metal;

#include <metal_stdlib>
using namespace metal;

kernel void apply_rope_q4(
    device half* vec [[buffer(0)]],
    constant uint* seq_len [[buffer(1)]], // Now accepts the position from Swift
    uint tid [[thread_position_in_grid]]
){
    uint head_idx = tid / 64;
    uint pair_idx = tid % 64;
    
    // Qwen's mandatory rotate_half layout
    uint idx0 = (head_idx * 128) + pair_idx;
    uint idx1 = idx0 + 64;
    
    // Dynamic position tracking
    float m = (float)(seq_len[0] - 1);
    
    float theta_base = 1000000.0f; // Qwen 2.5 context window requirement
    float head_dim = 128.0f;
    float theta = m * pow(theta_base, -((float)(pair_idx * 2) / head_dim));

    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    float x1 = (float)vec[idx0];
    float x2 = (float)vec[idx1];

    vec[idx0] = (half)(x1 * cos_theta - x2 * sin_theta);
    vec[idx1] = (half)(x1 * sin_theta + x2 * cos_theta);
}
