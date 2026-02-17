//
//  MetalManager.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-14.
//

import Metal
import Foundation

// Custom error handling
enum MetalSetupError: Error {
    case deviceInitializationFailed
    case fileReadFailed(String)
    case compilationFailed(String)
    case functionNotFound(String)
    case pipelineCreationFailed(String)
}


class MetalManager {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    // Dictionary that holds the compiled, ready-to-run kernels
    var pipelines: [String: MTLComputePipelineState] = [:]

    init(kernelsDir: String) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalSetupError.deviceInitializationFailed
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MetalSetupError.deviceInitializationFailed
        }
        self.commandQueue = queue
        try setupPipelines(kernelsDir: kernelsDir)
    }

    private func setupPipelines(kernelsDir: String) throws {
        let dirURL = URL(fileURLWithPath: kernelsDir)
        // Map the physical files to the kernel functions inside them
        let fileToFunctions: [String: [String]] = [
            "activations.metal": ["swiglu", "softmax"],
            "attention.metal":   ["attention_scores", "attn_weighted_sum", "residual_add", "write_kv_cache"],
            "matmul.metal":      ["gemv_q4_0", "gemv_fp16"],
            "rmsnorm.metal":     ["rms_norm_q4"],
            "rope.metal":        ["apply_rope_q4"]
        ]
        
        for (fileName, functionNames) in fileToFunctions {
            let fileURL = dirURL.appendingPathComponent(fileName)
            let source: String
            do {
                source = try String(contentsOf: fileURL, encoding: .utf8)
            } catch {
                throw MetalSetupError.fileReadFailed(fileURL.path)
            }
            let library: MTLLibrary
            do {
                library = try device.makeLibrary(source: source, options: nil)
            } catch {
                print("Compilation failed in \(fileName): \(error)")
                throw MetalSetupError.compilationFailed(fileName)
            }
            for funcName in functionNames {
                guard let mtlFunction = library.makeFunction(name: funcName) else {
                    throw MetalSetupError.functionNotFound(funcName)
                }
                
                do {
                    pipelines[funcName] = try device.makeComputePipelineState(function: mtlFunction)
                } catch {
                    throw MetalSetupError.pipelineCreationFailed(funcName)
                }
            }
        }
        print("All Metal kernels compiled successfully.")
    }
}
