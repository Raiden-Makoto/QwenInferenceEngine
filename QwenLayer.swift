//
//  QwenLayer.swift
//  QwenInferenceEngine
//
//  Created by Raiden Makoto on 2026-02-14.
//

import Foundation
import Metal

struct QwenLayer{
    let index: Int
    // RMSNorm Weights (FP16)
    let inputNorm: MTLBuffer
    let postAttnNorm: MTLBuffer
    // QKV Projections
    let qProj: MTLBuffer
    let qScales: MTLBuffer
    let kProj: MTLBuffer
    let kScales: MTLBuffer
    let vProj: MTLBuffer
    let vScales: MTLBuffer
    // Output Projection
    let oProj: MTLBuffer
    let oScales: MTLBuffer
    // MLP Projections (Gate, Up, Down)
    let gateProj: MTLBuffer
    let gateScales: MTLBuffer
    let upProj: MTLBuffer
    let upScales: MTLBuffer
    let downProj: MTLBuffer
    let downScales: MTLBuffer
    
    // Initializer
    init(device: MTLDevice, loader: WeightLoader, index: Int) throws {
        self.index = index
        self.inputNorm = try loader.loadLayerWeight(layer: index, name: "layer\(index)_norm")
        self.postAttnNorm = try loader.loadLayerWeight(layer: index, name: "layer\(index)_post_attn_norm")
        // Load QKV projections
        self.qProj = try loader.loadLayerWeight(layer: index, name: "q_proj_4bit")
        self.qScales = try loader.loadLayerWeight(layer: index, name: "q_proj_scales")
        self.kProj = try loader.loadLayerWeight(layer: index, name: "k_proj_4bit")
        self.kScales = try loader.loadLayerWeight(layer: index, name: "k_proj_scales")
        self.vProj = try loader.loadLayerWeight(layer: index, name: "v_proj_4bit")
        self.vScales = try loader.loadLayerWeight(layer: index, name: "v_proj_scales")
        // Load Output Projections
        self.oProj = try loader.loadLayerWeight(layer: index, name: "o_proj_4bit")
        self.oScales  = try loader.loadLayerWeight(layer: index, name: "o_proj_scales")
        // MLP Projections
        self.gateProj = try loader.loadLayerWeight(layer: index, name: "gate_proj_4bit")
        self.gateScales = try loader.loadLayerWeight(layer: index, name: "gate_proj_scales")
        self.upProj = try loader.loadLayerWeight(layer: index, name: "up_proj_4bit")
        self.upScales = try loader.loadLayerWeight(layer: index, name: "up_proj_scales")
        self.downProj = try loader.loadLayerWeight(layer: index, name: "down_proj_4bit")
        self.downScales = try loader.loadLayerWeight(layer: index, name: "down_proj_scales")
    }
}
