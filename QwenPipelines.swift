//
//  QwenPipelines.swift
//  QwenInferenceEngine
//
import Foundation
import Accelerate

class QwenPipeline {
    let modelID = "Qwen/Qwen2.5-1.5B-Instruct"
    let targetDir: URL
    let weightsDir: URL
    let blockSize = 32
    
    // Hardcoded dimensions to guarantee safe bias padding if missing
    let hiddenDim = 1536
    let kDim = 256
    let mlpIntermediate = 8960
    
    init() {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        self.targetDir = cwd.appendingPathComponent("downloads")
        self.weightsDir = cwd.appendingPathComponent("weights")
    }
    
    func run() async throws {
        print("Starting Qwen 2.5 Metal Pipeline...")
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)
        
        let safetensorsURL = targetDir.appendingPathComponent("model.safetensors")
        
        // 1. Download
        if !FileManager.default.fileExists(atPath: safetensorsURL.path) {
            print("Downloading model.safetensors (This will take a while)...")
            try await downloadFile(filename: "model.safetensors")
            try await downloadFile(filename: "config.json")
            try await downloadFile(filename: "tokenizer.json")
            try await downloadFile(filename: "tokenizer_config.json")
            try await downloadFile(filename: "vocab.json")
        } else {
            print("model.safetensors already exists. Skipping download.")
        }
        
        // 2. Parse Safetensors Once (Memory Mapped)
        print("🧠 Memory-mapping Safetensors...")
        let safetensorsData = try Data(contentsOf: safetensorsURL, options: .alwaysMapped)
        
        let headerSizeData = safetensorsData.subdata(in: 0..<8)
        let headerSize = Int(headerSizeData.withUnsafeBytes { $0.load(as: UInt64.self) })
        
        let headerData = safetensorsData.subdata(in: 8..<(8 + headerSize))
        let header = try JSONSerialization.jsonObject(with: headerData) as! [String: Any]
        
        let tensorStartOffset = 8 + headerSize
        
        // --- EXTRACTION HELPERS ---
        
        func extractTensorAsFloat32(name: String) -> [Float]? {
            guard let meta = header[name] as? [String: Any],
                  let offsets = meta["data_offsets"] as? [Int],
                  let dtype = meta["dtype"] as? String else { return nil }
            
            let start = tensorStartOffset + offsets[0]
            let length = offsets[1] - offsets[0]
            let rawData = safetensorsData.subdata(in: start..<(start + length))
            
            if dtype == "BF16" {
                // Safely convert BFloat16 to Float32
                let count = length / 2
                let bf16Array = rawData.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                var f32Array = [Float](repeating: 0, count: count)
                f32Array.withUnsafeMutableBytes { f32Ptr in
                    let u32Ptr = f32Ptr.bindMemory(to: UInt32.self)
                    for i in 0..<count { u32Ptr[i] = UInt32(bf16Array[i]) << 16 }
                }
                return f32Array
            } else if dtype == "F16" {
                let count = length / 2
                let f16Array = rawData.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
                return f16Array.map { Float($0) }
            } else {
                fatalError("Unsupported dtype: \(dtype) for \(name)")
            }
        }
        
        func saveAsFloat16(tensor: [Float]?, defaultCount: Int, saveTo path: URL) throws {
            let f16Array: [Float16]
            if let t = tensor {
                f16Array = t.map { Float16($0) }
            } else {
                f16Array = [Float16](repeating: 0.0, count: defaultCount) // Fallback for missing biases
            }
            let data = f16Array.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: path)
        }
        
        func packAndSave4Bit(tensor: [Float]?, saveTo pathPrefix: URL) throws {
            guard let weights = tensor else { return }
            
            let numWeights = weights.count
            let numBlocks = (numWeights + blockSize - 1) / blockSize
            
            var packedData = Data(count: numBlocks * (blockSize / 2))
            var scalesData = Data(count: numBlocks * MemoryLayout<Float16>.stride)
            
            packedData.withUnsafeMutableBytes { packedPtr in
                scalesData.withUnsafeMutableBytes { scalesPtr in
                    let packed = packedPtr.assumingMemoryBound(to: UInt8.self)
                    let scales = scalesPtr.assumingMemoryBound(to: Float16.self)
                    
                    for b in 0..<numBlocks {
                        let startIdx = b * blockSize
                        var maxAbs: Float = 0
                        
                        for i in 0..<blockSize {
                            let idx = startIdx + i
                            if idx < numWeights { maxAbs = max(maxAbs, abs(weights[idx])) }
                        }
                        
                        let scale = maxAbs / 7.0
                        let invScale = scale != 0 ? 1.0 / scale : 0
                        scales[b] = Float16(scale)
                        
                        for i in stride(from: 0, to: blockSize, by: 2) {
                            let idxLow = startIdx + i
                            let idxHigh = startIdx + i + 1
                            
                            let wLow = idxLow < numWeights ? weights[idxLow] : 0
                            let wHigh = idxHigh < numWeights ? weights[idxHigh] : 0
                            
                            let qLow = Int8(clamping: Int(round(wLow * invScale)))
                            let qHigh = Int8(clamping: Int(round(wHigh * invScale)))
                            
                            let uLow = UInt8(bitPattern: qLow.clamped(min: -8, max: 7) + 8)
                            let uHigh = UInt8(bitPattern: qHigh.clamped(min: -8, max: 7) + 8)
                            
                            packed[(b * (blockSize / 2)) + (i / 2)] = uLow | (uHigh << 4)
                        }
                    }
                }
            }
            try packedData.write(to: pathPrefix.appendingPathExtension("bin"))
            let scalesURL = pathPrefix.deletingLastPathComponent().appendingPathComponent("\(pathPrefix.lastPathComponent.replacingOccurrences(of: "_4bit", with: "_scales")).bin")
            try scalesData.write(to: scalesURL)
        }

        // --- 3. PROCESS GLOBAL WEIGHTS ---
        print("📦 Processing Global Weights...")
        let globalDir = weightsDir.appendingPathComponent("global")
        try FileManager.default.createDirectory(at: globalDir, withIntermediateDirectories: true)
        
        let embedTokens = extractTensorAsFloat32(name: "model.embed_tokens.weight")
        try saveAsFloat16(tensor: embedTokens, defaultCount: 151936 * hiddenDim, saveTo: globalDir.appendingPathComponent("embed_tokens.bin"))
        try packAndSave4Bit(tensor: embedTokens, saveTo: globalDir.appendingPathComponent("embed_tokens_4bit"))
        
        let finalNorm = extractTensorAsFloat32(name: "model.norm.weight")
        try saveAsFloat16(tensor: finalNorm, defaultCount: hiddenDim, saveTo: globalDir.appendingPathComponent("final_norm.bin"))
        
        // --- 4. PROCESS 28 LAYERS ---
        for layer in 0..<28 {
            print("Processing Layer \(layer)/27...")
            let layerDir = weightsDir.appendingPathComponent("layer\(layer)")
            try FileManager.default.createDirectory(at: layerDir, withIntermediateDirectories: true)
            
            // Norms
            try saveAsFloat16(tensor: extractTensorAsFloat32(name: "model.layers.\(layer).input_layernorm.weight"), defaultCount: hiddenDim, saveTo: layerDir.appendingPathComponent("layer\(layer)_norm.bin"))
            try saveAsFloat16(tensor: extractTensorAsFloat32(name: "model.layers.\(layer).post_attention_layernorm.weight"), defaultCount: hiddenDim, saveTo: layerDir.appendingPathComponent("layer\(layer)_post_attn_norm.bin"))
            
            // Attn Matrices (4-bit)
            let attnKeys = ["q_proj", "k_proj", "v_proj", "o_proj"] // No c_proj in Qwen2.5!
            for key in attnKeys {
                let w = extractTensorAsFloat32(name: "model.layers.\(layer).self_attn.\(key).weight")
                try packAndSave4Bit(tensor: w, saveTo: layerDir.appendingPathComponent("\(key)_4bit"))
            }
            
            // Attn Biases (Concatenated into one buffer for Swift)
            let qBias = extractTensorAsFloat32(name: "model.layers.\(layer).self_attn.q_proj.bias") ?? [Float](repeating: 0, count: hiddenDim)
            let kBias = extractTensorAsFloat32(name: "model.layers.\(layer).self_attn.k_proj.bias") ?? [Float](repeating: 0, count: kDim)
            let vBias = extractTensorAsFloat32(name: "model.layers.\(layer).self_attn.v_proj.bias") ?? [Float](repeating: 0, count: kDim)
            let concatenatedQKVBias = qBias + kBias + vBias
            try saveAsFloat16(tensor: concatenatedQKVBias, defaultCount: hiddenDim + (kDim * 2), saveTo: layerDir.appendingPathComponent("qkv_bias.bin"))
            
            // Output Bias
            try saveAsFloat16(tensor: extractTensorAsFloat32(name: "model.layers.\(layer).self_attn.o_proj.bias"), defaultCount: hiddenDim, saveTo: layerDir.appendingPathComponent("o_bias.bin"))
            
            // MLP Matrices (4-bit)
            let mlpKeys = ["gate_proj", "up_proj", "down_proj"]
            for key in mlpKeys {
                let w = extractTensorAsFloat32(name: "model.layers.\(layer).mlp.\(key).weight")
                try packAndSave4Bit(tensor: w, saveTo: layerDir.appendingPathComponent("\(key)_4bit"))
            }
            
            // MLP Bias
            try saveAsFloat16(tensor: extractTensorAsFloat32(name: "model.layers.\(layer).mlp.down_proj.bias"), defaultCount: hiddenDim, saveTo: layerDir.appendingPathComponent("mlp_down_bias.bin"))
        }
        
        print("\nSwift Pipeline Complete. All weights successfully quantized, mapped, and ready for Metal.")
    }
    
    // --- SIMPLE DOWNLOADER ---
    private func downloadFile(filename: String) async throws {
        let url = URL(string: "https://huggingface.co/\(modelID)/resolve/main/\(filename)")!
        let dest = targetDir.appendingPathComponent(filename)
        
        let (tempURL, response) = try await URLSession.shared.download(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw NSError(domain: "Download", code: 404, userInfo: [NSLocalizedDescriptionKey: "Failed to download \(filename)"])
        }
        try FileManager.default.moveItem(at: tempURL, to: dest)
        if filename == "config.json" || filename == "tokenizer.json" {
            // Also copy config files directly to weights dir for the tokenizer to find later
            try? FileManager.default.copyItem(at: dest, to: weightsDir.appendingPathComponent(filename))
        }
    }
}

extension Comparable {
    func clamped(min: Self, max: Self) -> Self {
        if self < min { return min }
        if self > max { return max }
        return self
    }
}
