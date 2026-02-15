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

    init(weightsDir: String) async throws {
        let modelFolderURL = URL(fileURLWithPath: weightsDir)
        let tokenizerFileURL = modelFolderURL.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerFileURL.path) else {
            fatalError("Tokenizer weights file not found at \(tokenizerFileURL.path)")
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelFolderURL)
    }

    func encode(prompt: String) -> [Int] {
        return tokenizer.encode(text: prompt)
    }
    func decode(tokens: [Int]) -> String {
        return tokenizer.decode(tokens: tokens)
    }
}
