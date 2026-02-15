//
//  InferenceManager.swift
//  QwenInferenceEngine
//
//  Created by Raiden Makoto on 2026-02-14.
//

import Foundation
import Metal

class InferenceManager {
    let device: MTLDevice
    // --- Intermediate Activations (FP16 = 2 bytes) ---
    let residual: MTLBuffer
    let x: MTLBuffer
    let q: MTLBuffer
    let k: MTLBuffer
    let v: MTLBuffer
    let attnOut: MTLBuffer
    let gateOut: MTLBuffer
    let upOut: MTLBuffer
    let swigluOut: MTLBuffer
    let mlpOut: MTLBuffer
    // --- Logits (FP16) ---
    let logits: MTLBuffer
    // --- Attention Scores (FP32 = 4 bytes) ---
    let scores: MTLBuffer
    // --- The KV Cache (FP16) ---
    // Size: 28 layers * 1024 tokens * 256 dim
    let kvCacheK: MTLBuffer
    let kvCacheV: MTLBuffer
    // --- Meta/Control Buffers (UInt32 = 4 bytes) ---
    let seqLen: MTLBuffer
    let layerIdx: MTLBuffer
    let gemvConstants1536: MTLBuffer
    let gemvConstants8960: MTLBuffer
    init(device: MTLDevice) throws {
        self.device = device
        // Helper to safely allocate shared memory buffers
        func makeBuffer(length: Int) -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
                fatalError("Failed to allocate state buffer of size \(length) bytes.")
            }
            return buffer
        }
        let hiddenSize = QwenConfig.hiddenDim * 2
        self.residual = makeBuffer(length: hiddenSize)
        self.x = makeBuffer(length: hiddenSize)
        self.attnOut = makeBuffer(length: hiddenSize)
        self.mlpOut = makeBuffer(length: hiddenSize)
        self.q = makeBuffer(length: hiddenSize) // 1536 dim
        self.k = makeBuffer(length: QwenConfig.kDim * 2) // 256 dim
        self.v = makeBuffer(length: QwenConfig.kDim * 2) // 256 dim
        
        let intermediateSize = QwenConfig.mlpIntermediate * 2
        self.gateOut = makeBuffer(length: intermediateSize)
        self.upOut = makeBuffer(length: intermediateSize)
        self.swigluOut = makeBuffer(length: intermediateSize)
        // 2. Output and Scores
        self.logits = makeBuffer(length: QwenConfig.actualVocabSize * 2)
        self.scores = makeBuffer(length: QwenConfig.numAttnHeads * QwenConfig.maxContextWindow * 4) // FP32
        // 3. The KV Cache
        let cacheSize = QwenConfig.numLayers * QwenConfig.maxContextWindow * QwenConfig.kDim * 2
        self.kvCacheK = makeBuffer(length: cacheSize)
        self.kvCacheV = makeBuffer(length: cacheSize)
        // 4. Meta Buffers for Kernel Arguments
        self.seqLen = makeBuffer(length: 4)
        self.layerIdx = makeBuffer(length: 4)
        // Constants array for GEMV: [K, simd_groups, threads]
        let threads = UInt32(QwenConfig.GEMVThreads)
        let simdGroups = threads / 32
        let const1536: [UInt32] = [UInt32(QwenConfig.hiddenDim), simdGroups, threads]
        self.gemvConstants1536 = device.makeBuffer(bytes: const1536, length: 12, options: .storageModeShared)!
        let const8960: [UInt32] = [UInt32(QwenConfig.mlpIntermediate), simdGroups, threads]
        self.gemvConstants8960 = device.makeBuffer(bytes: const8960, length: 12, options: .storageModeShared)!
    }
}
