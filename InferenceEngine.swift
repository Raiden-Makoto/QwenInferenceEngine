//
//  InferenceEngine.swift
//  QwenInferenceEngine
//

import Foundation
import Metal

class QwenEngine {
    let device: MTLDevice
    let model: QwenModel
    let state: InferenceManager
    let manager: MetalManager
    let tokenizer: QwenTokenizer
    
    init(weightsPath: String, kernelsPath: String) async throws {
        guard let defaultDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device.")
        }
        self.device = defaultDevice
        self.model = try QwenModel(device: device, weightsPath: weightsPath)
        self.state = try InferenceManager(device: device)
        self.manager = try MetalManager(kernelsDir: kernelsPath)
        self.tokenizer = try await QwenTokenizer(weightsDir: weightsPath)
        
        print("QwenEngine Initialized and Ready.")
    }
    
    func generate(prompt: String, maxTokens: Int = 64) {
        let tokens = tokenizer.encode(prompt: prompt)
        fflush(stdout)
        
        var currentToken = 0
        for i in 0..<tokens.count {
            currentToken = forward(tokenID: tokens[i], position: i)
        }
        
        var currentPos = tokens.count
        
        for _ in 0..<maxTokens {
            if currentToken == 151645 || currentToken == 151643 {
                break
            }
            let word = tokenizer.decode(tokens: [currentToken])
            print(word, terminator: "")
            fflush(stdout)
            
            currentToken = forward(tokenID: currentToken, position: currentPos)
            currentPos += 1
        }
        print("\n\n--- Generation Complete ---")
    }
    
    private func forward(tokenID: Int, position: Int) -> Int {
        let safePosition = min(position, QwenConfig.maxContextWindow - 1)
        
        let hiddenDimBytes = QwenConfig.hiddenDim * MemoryLayout<Float16>.stride
        let embeddingOffset = tokenID * hiddenDimBytes
        let sourcePtr = model.tokenEmbeddings.contents().advanced(by: embeddingOffset)
        memcpy(state.x.contents(), sourcePtr, hiddenDimBytes)
        
        guard let commandBuffer = manager.commandQueue.makeCommandBuffer() else {
            fatalError("Failed to create command buffer.")
        }
        
        let seqLenPtr = state.seqLen.contents().assumingMemoryBound(to: UInt32.self)
        seqLenPtr[0] = UInt32(safePosition + 1)
        
        var const1536: [UInt32] = [
            UInt32(QwenConfig.hiddenDim),
            UInt32(QwenConfig.GEMVThreads / 32),
            UInt32(QwenConfig.GEMVThreads)
        ]
        var const8960: [UInt32] = [
            UInt32(QwenConfig.mlpIntermediate),
            UInt32(QwenConfig.GEMVThreads / 32),
            UInt32(QwenConfig.GEMVThreads)
        ]
        let buf1536 = device.makeBuffer(bytes: &const1536, length: MemoryLayout<UInt32>.stride * 3, options: .storageModeShared)!
        let buf8960 = device.makeBuffer(bytes: &const8960, length: MemoryLayout<UInt32>.stride * 3, options: .storageModeShared)!
        
        let maxBiasBytes = QwenConfig.mlpIntermediate * MemoryLayout<Float16>.stride
        guard let zeroBiasBuffer = device.makeBuffer(length: maxBiasBytes, options: .storageModeShared) else {
            fatalError("Failed to create zero bias buffer.")
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Failed to create compute encoder.")
        }
        
        // --- FIXED RMS NORM DISPATCHES ---
        let normGrid = MTLSize(width: 1, height: 1, depth: 1)
        let normGroup = MTLSize(width: 256, height: 1, depth: 1)
        
        let gemvGroup = MTLSize(width: QwenConfig.GEMVThreads, height: 1, depth: 1)
        let qGrid = MTLSize(width: QwenConfig.hiddenDim, height: 1, depth: 1)
        let kvGrid = MTLSize(width: QwenConfig.kDim, height: 1, depth: 1)
        let mlpGrid = MTLSize(width: QwenConfig.mlpIntermediate, height: 1, depth: 1)

        let kOffset = QwenConfig.hiddenDim * MemoryLayout<Float16>.stride
        let vOffset = (QwenConfig.hiddenDim + QwenConfig.kDim) * MemoryLayout<Float16>.stride

        for i in 0..<QwenConfig.numLayers {
            let layer = model.layers[i]
            
            // a. RMSNorm (Input)
            encoder.setComputePipelineState(manager.pipelines["rms_norm_q4"]!)
            encoder.setBuffer(state.x, offset: 0, index: 0)
            encoder.setBuffer(layer.inputNorm, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.dispatchThreadgroups(normGrid, threadsPerThreadgroup: normGroup) // <-- FIXED
            
            // b. Q, K, V Projections
            encoder.setComputePipelineState(manager.pipelines["gemv_q4_0"]!)
            
            encoder.setBuffer(layer.qProj, offset: 0, index: 0)
            encoder.setBuffer(layer.qScales, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.setBuffer(state.q, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(layer.qkvBias, offset: 0, index: 5)
            encoder.dispatchThreadgroups(qGrid, threadsPerThreadgroup: gemvGroup)
            
            encoder.setBuffer(layer.kProj, offset: 0, index: 0)
            encoder.setBuffer(layer.kScales, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.setBuffer(state.k, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(layer.qkvBias, offset: kOffset, index: 5)
            encoder.dispatchThreadgroups(kvGrid, threadsPerThreadgroup: gemvGroup)
            
            encoder.setBuffer(layer.vProj, offset: 0, index: 0)
            encoder.setBuffer(layer.vScales, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.setBuffer(state.v, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(layer.qkvBias, offset: vOffset, index: 5)
            encoder.dispatchThreadgroups(kvGrid, threadsPerThreadgroup: gemvGroup)
            
            // c. RoPE
            encoder.setComputePipelineState(manager.pipelines["apply_rope_q4"]!)
            encoder.setBuffer(state.q, offset: 0, index: 0)
            encoder.setBuffer(state.seqLen, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.hiddenDim / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            encoder.setBuffer(state.k, offset: 0, index: 0)
            encoder.setBuffer(state.seqLen, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.kDim / 2, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            
            // d. KV Cache Memory Copy
            let kvSize = QwenConfig.kDim * MemoryLayout<Float16>.stride
            let layerCacheOffset = i * QwenConfig.maxContextWindow * kvSize
            
            encoder.setComputePipelineState(manager.pipelines["write_kv_cache"]!)
            encoder.setBuffer(state.k, offset: 0, index: 0)
            encoder.setBuffer(state.v, offset: 0, index: 1)
            encoder.setBuffer(state.kvCacheK, offset: layerCacheOffset, index: 2)
            encoder.setBuffer(state.kvCacheV, offset: layerCacheOffset, index: 3)
            encoder.setBuffer(state.seqLen, offset: 0, index: 4)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.kDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            // e. Attention Scores
            encoder.setComputePipelineState(manager.pipelines["attention_scores"]!)
            encoder.setBuffer(state.q, offset: 0, index: 0)
            encoder.setBuffer(state.kvCacheK, offset: layerCacheOffset, index: 1)
            encoder.setBuffer(state.scores, offset: 0, index: 2)
            encoder.setBuffer(state.seqLen, offset: 0, index: 3)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.numAttnHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 12, height: 1, depth: 1))
            
            // f. Softmax
            encoder.setComputePipelineState(manager.pipelines["softmax"]!)
            encoder.setBuffer(state.scores, offset: 0, index: 0)
            encoder.setBuffer(state.seqLen, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.numAttnHeads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 12, height: 1, depth: 1))
            
            // g. Attention Weighted Sum
            encoder.setComputePipelineState(manager.pipelines["attn_weighted_sum"]!)
            encoder.setBuffer(state.scores, offset: 0, index: 0)
            encoder.setBuffer(state.kvCacheV, offset: layerCacheOffset, index: 1)
            encoder.setBuffer(state.attnOut, offset: 0, index: 2)
            encoder.setBuffer(state.seqLen, offset: 0, index: 3)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            // h. Output Projection
            encoder.setComputePipelineState(manager.pipelines["gemv_q4_0"]!)
            encoder.setBuffer(layer.oProj, offset: 0, index: 0)
            encoder.setBuffer(layer.oScales, offset: 0, index: 1)
            encoder.setBuffer(state.attnOut, offset: 0, index: 2)
            encoder.setBuffer(state.q, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(layer.oBias, offset: 0, index: 5)
            encoder.dispatchThreadgroups(qGrid, threadsPerThreadgroup: gemvGroup)
            
            // i. Residual Add
            encoder.setComputePipelineState(manager.pipelines["residual_add"]!)
            encoder.setBuffer(state.x, offset: 0, index: 0)
            encoder.setBuffer(state.q, offset: 0, index: 1)
            encoder.setBuffer(state.x, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // j. Post-Attention RMSNorm
            encoder.setComputePipelineState(manager.pipelines["rms_norm_q4"]!)
            encoder.setBuffer(state.x, offset: 0, index: 0)
            encoder.setBuffer(layer.postAttnNorm, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.dispatchThreadgroups(normGrid, threadsPerThreadgroup: normGroup) // <-- FIXED
            
            // k. MLP Gate and Up Projections
            encoder.setComputePipelineState(manager.pipelines["gemv_q4_0"]!)
            encoder.setBuffer(layer.gateProj, offset: 0, index: 0)
            encoder.setBuffer(layer.gateScales, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.setBuffer(state.gateOut, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(zeroBiasBuffer, offset: 0, index: 5)
            encoder.dispatchThreadgroups(mlpGrid, threadsPerThreadgroup: gemvGroup)
            
            encoder.setBuffer(layer.upProj, offset: 0, index: 0)
            encoder.setBuffer(layer.upScales, offset: 0, index: 1)
            encoder.setBuffer(state.residual, offset: 0, index: 2)
            encoder.setBuffer(state.upOut, offset: 0, index: 3)
            encoder.setBuffer(buf1536, offset: 0, index: 4)
            encoder.setBuffer(zeroBiasBuffer, offset: 0, index: 5)
            encoder.dispatchThreadgroups(mlpGrid, threadsPerThreadgroup: gemvGroup)
            
            // l. SwiGLU Activation
            encoder.setComputePipelineState(manager.pipelines["swiglu"]!)
            encoder.setBuffer(state.gateOut, offset: 0, index: 0)
            encoder.setBuffer(state.upOut, offset: 0, index: 1)
            encoder.dispatchThreads(mlpGrid, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
            // m. MLP Down Projection
            encoder.setComputePipelineState(manager.pipelines["gemv_q4_0"]!)
            encoder.setBuffer(layer.downProj, offset: 0, index: 0)
            encoder.setBuffer(layer.downScales, offset: 0, index: 1)
            encoder.setBuffer(state.gateOut, offset: 0, index: 2)
            encoder.setBuffer(state.mlpOut, offset: 0, index: 3)
            encoder.setBuffer(buf8960, offset: 0, index: 4)
            encoder.setBuffer(layer.mlpDownBias, offset: 0, index: 5)
            encoder.dispatchThreadgroups(qGrid, threadsPerThreadgroup: gemvGroup)
            
            // n. Residual Add
            encoder.setComputePipelineState(manager.pipelines["residual_add"]!)
            encoder.setBuffer(state.x, offset: 0, index: 0)
            encoder.setBuffer(state.mlpOut, offset: 0, index: 1)
            encoder.setBuffer(state.x, offset: 0, index: 2)
            encoder.dispatchThreads(MTLSize(width: QwenConfig.hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            
        } // --- END OF 28-LAYER LOOP ---
        
        // 4. FINAL RMS NORM
        encoder.setComputePipelineState(manager.pipelines["rms_norm_q4"]!)
        encoder.setBuffer(state.x, offset: 0, index: 0)
        encoder.setBuffer(model.finalNormWeight, offset: 0, index: 1)
        encoder.setBuffer(state.residual, offset: 0, index: 2)
        encoder.dispatchThreadgroups(normGrid, threadsPerThreadgroup: normGroup) // <-- FIXED
        
        // 5. LM HEAD (Logits Projection)
        encoder.setComputePipelineState(manager.pipelines["gemv_fp16"]!)
        encoder.setBuffer(model.LMHeadFP16, offset: 0, index: 0)
        encoder.setBuffer(state.residual, offset: 0, index: 1)
        encoder.setBuffer(state.logits, offset: 0, index: 2)
        encoder.setBuffer(buf1536, offset: 0, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: QwenConfig.vocabSize, height: 1, depth: 1), threadsPerThreadgroup: gemvGroup)
        
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return extractNextTokenID(from: state.logits)
    }
    
    private func extractNextTokenID(from logitsBuffer: MTLBuffer, temperature: Float = 0.7, topP: Float = 0.8) -> Int {
        let logitsPtr = logitsBuffer.contents().assumingMemoryBound(to: Float16.self)
        let vocabSize = QwenConfig.actualVocabSize
        
        // 0. Fallback to greedy decoding if temperature is 0
        if temperature <= 0.0 {
            var maxVal: Float16 = -Float16.infinity
            var maxIdx = 0
            for i in 0..<vocabSize {
                let logit = logitsPtr[i]
                if logit.isNaN { continue }
                if logit > maxVal {
                    maxVal = logit
                    maxIdx = i
                }
            }
            if maxVal == -Float16.infinity {
                print("\n\n[WARNING: GPU Network Collapsed to NaN. Forcing Engine Halt.]")
                return 151645
            }
            return maxIdx
        }
        
        // 1. Extract, apply Temperature, and find Max for stable Softmax
        var logits = [Float](repeating: 0.0, count: vocabSize)
        var maxLogit: Float = -Float.infinity
        
        for i in 0..<vocabSize {
            let val = Float(logitsPtr[i])
            if val.isNaN { continue }
            
            let scaled = val / temperature
            logits[i] = scaled
            if scaled > maxLogit {
                maxLogit = scaled
            }
        }
        
        // NaN protection
        if maxLogit == -Float.infinity {
            print("\n\n[WARNING: GPU Network Collapsed to NaN. Forcing Engine Halt.]")
            return 151645
        }
        
        // 2. Softmax (to convert raw logits into percentages adding up to 1.0)
        var sumExp: Float = 0.0
        for i in 0..<vocabSize {
            let expVal = exp(logits[i] - maxLogit)
            logits[i] = expVal
            sumExp += expVal
        }
        for i in 0..<vocabSize {
            logits[i] /= sumExp
        }
        
        // 3. Top-P (Nucleus) Sorting
        // Create an array mapping the probability back to its original token ID
        var probsWithIndices = [(prob: Float, index: Int)]()
        probsWithIndices.reserveCapacity(vocabSize)
        for i in 0..<vocabSize {
            probsWithIndices.append((logits[i], i))
        }
        
        // Sort descending by highest probability
        probsWithIndices.sort { $0.prob > $1.prob }
        
        // 4. Calculate Top-P Cutoff
        var cumulativeProb: Float = 0.0
        var cutoffIndex = 0
        for i in 0..<vocabSize {
            cumulativeProb += probsWithIndices[i].prob
            if cumulativeProb >= topP {
                cutoffIndex = i
                break
            }
        }
        
        // Ensure we always have at least one token to sample from
        cutoffIndex = max(1, cutoffIndex + 1)
        
        // 5. Sample from the filtered distribution (The Weighted Dice Roll)
        let filteredProbs = probsWithIndices[0..<cutoffIndex]
        let newSum = filteredProbs.reduce(0.0) { $0 + $1.prob }
        
        let randomVal = Float.random(in: 0..<newSum)
        var runningSum: Float = 0.0
        
        for item in filteredProbs {
            runningSum += item.prob
            if randomVal <= runningSum {
                return item.index
            }
        }
        
        return filteredProbs.last?.index ?? 0
    }
}
