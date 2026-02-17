//
//  QwenTokenizer.swift
//  QwenInferenceEngine
//
//  Created by Raiden Makoto on 2026-02-14.
//

import Foundation
import Tokenizers // from package dependency

class QwenTokenizer {
    let tokenizer: Tokenizer
    
    // The rolling buffer for incomplete UTF-8 byte sequences
    private var decodeBuffer: [Int] = []

    init(weightsDir: String) async throws {
        let modelFolderURL = URL(fileURLWithPath: weightsDir)
        let tokenizerFileURL = modelFolderURL.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerFileURL.path) else {
            fatalError("Tokenizer weights file not found at \(tokenizerFileURL.path)")
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)
    }

    func encode(prompt: String) -> [Int] {
        var tokens = [Int]()
        
        // 1. Manually parse ChatML tags to bypass the tokenizer's default string-shredding
        let imStartParts = prompt.components(separatedBy: "<|im_start|>")
        
        for (i, startPart) in imStartParts.enumerated() {
            if i > 0 {
                tokens.append(151644) // <|im_start|> ID
            }
            
            let imEndParts = startPart.components(separatedBy: "<|im_end|>")
            for (j, endPart) in imEndParts.enumerated() {
                if j > 0 {
                    tokens.append(151645) // <|im_end|> ID
                }
                
                if !endPart.isEmpty {
                    // Let the underlying BPE tokenizer handle the raw English/Code
                    tokens.append(contentsOf: tokenizer.encode(text: endPart))
                }
            }
        }
        
        return tokens
    }
    
    func decode(tokens: [Int]) -> String {
        // 2. Append the incoming token(s) to our rolling buffer
        decodeBuffer.append(contentsOf: tokens)
        
        // Attempt to decode the entire buffer
        let decoded = tokenizer.decode(tokens: decodeBuffer)
        
        // Incomplete UTF-8 byte sequences decode into the Unicode Replacement Character ().
        // If we see it, we hold the bytes in the buffer and return an empty string.
        if decoded.contains("\u{FFFD}") {
            return ""
        }
        
        // Otherwise, we successfully completed a valid string!
        // Clear the buffer and return the string to the Inference Engine.
        decodeBuffer.removeAll()
        return decoded
    }
}
