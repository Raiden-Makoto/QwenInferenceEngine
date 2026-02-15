//
//  QwenConfig.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-13.
//

import Foundation

struct QwenConfig{
    static let hiddenDim = 1536
    static let mlpIntermediate = 8960
    static let numLayers = 28
    static let numAttnHeads = 12
    static let numKVHeads = 2 // Grouped-Query Attention
    static let headDim = 128
    static let vocabSize = 151936
    static let actualVocabSize = 151666
    static let maxContextWindow = 1024
    static let kDim = numKVHeads * headDim
    static let GEMVThreads = 768
}
