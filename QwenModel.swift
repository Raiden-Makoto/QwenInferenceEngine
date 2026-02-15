//
//  QwenModel.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-14.
//

import Foundation
import Metal

class QwenModel{
    let device: MTLDevice
    let loader: WeightLoader
    // Layer-Agnostic Weights
    let tokenEmbeddings: MTLBuffer
    let finalNormWeight: MTLBuffer
    let LMHeadPacked: MTLBuffer
    let LHMHeadScales: MTLBuffer
    // Stores the 28 Transformer Blocks
    var layers: [QwenLayer] = []
    // Initializer
    init(device: MTLDevice, weightsPath: String) throws {
        self.device = device
        self.loader = WeightLoader(device: device, weightsDir: weightsPath)
        self.tokenEmbeddings = try self.loader.loadBuffer(relativePath: "global/embed_tokens.bin")
        self.finalNormWeight = try self.loader.loadBuffer(relativePath: "global/final_norm.bin")
        self.LMHeadPacked = try self.loader.loadBuffer(relativePath: "global/embed_tokens_4bit.bin")
        self.LHMHeadScales = try self.loader.loadBuffer(relativePath: "global/embed_tokens_scales.bin")
        // Load the 28 Transformer Blocks
        for i in 0..<QwenConfig.numLayers{
            let layer = try QwenLayer(device: device,loader: self.loader, index: i)
            self.layers.append(layer)
        }
        print("Model successfully loaded global weights and all 28 transformer blocks.")
    }
}
