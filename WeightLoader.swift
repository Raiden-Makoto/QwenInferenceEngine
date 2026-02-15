//
//  WeightLoader.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-14.
//

import Foundation
import Metal

enum WeightLoadingError: Error{
    case fileNotFound(String)
    case bufferCreationFailed(String)
}

class WeightLoader{
    let device: MTLDevice
    let weightsDir: URL
    
    init(device: MTLDevice, weightsDir: String) {
        self.device = device
        self.weightsDir = URL(fileURLWithPath: weightsDir)
    }
    
    // Maps a .bin file directly to a Metal Buffer using memoy mapping
    func loadBuffer(relativePath: String) throws -> MTLBuffer {
        let fileURL = weightsDir.appendingPathComponent(relativePath)
        //print("Loading: \(fileURL.lastPathComponent)") // Debugging print
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw WeightLoadingError.fileNotFound(fileURL.path)
        }
        do {
            let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
            return try data.withUnsafeBytes { pointer in
                guard let baseAddress = pointer.baseAddress else {
                    throw WeightLoadingError.bufferCreationFailed(relativePath)
                }
                guard let buffer = device.makeBuffer(bytes: baseAddress, length: data.count, options: .storageModeShared) else {
                    throw WeightLoadingError.bufferCreationFailed(relativePath)
                }
                return buffer
            }
        } catch {
            throw error
        }
    }
    
    // Convenient method to load a specific layer weight
    func loadLayerWeight(layer: Int, name: String) throws -> MTLBuffer{
        let path = "layer\(layer)/\(name).bin"
        return try loadBuffer(relativePath: path)
    }
}
