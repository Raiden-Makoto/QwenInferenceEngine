//
//  main.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-13.
//

import Foundation
import Metal

func formatPrompt(userQuery: String) -> String {
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\(userQuery)<|im_end|>\n<|im_start|>assistant\n"
}

func runInference() {
    // A semaphore keeps the command-line tool alive while the async Task runs
    let sema = DispatchSemaphore(value: 0)
    
    Task {
        print("Booting Qwen 2.5 Metal Inference Engine...")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Resolve absolute paths for the weights and kernels directories
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let weightsPath = cwd.appendingPathComponent("weights").path
        let kernelsPath = cwd.appendingPathComponent("kernels").path
        
        do {
            // 1. Initialize the Engine
            // This loads the 4-bit weights into VRAM, compiles the .metal files, and parses tokenizer.json
            print("Loading weights, compiling kernels, and spinning up the tokenizer...")
            let engine = try await QwenEngine(weightsPath: weightsPath, kernelsPath: kernelsPath)
            
            let loadTime = CFAbsoluteTimeGetCurrent() - startTime
            print("Engine fully operational in \(String(format: "%.2f", loadTime)) seconds.\n")
            
            // 2. Set the Prompt
            print("> Enter a prompt: ", terminator: "")
            guard let input = readLine() else { return }
            let prompt = formatPrompt(userQuery: input)
            print("> Thinking...")
            
            let genStartTime = CFAbsoluteTimeGetCurrent()
            
            // 3. Trigger Autoregressive Generation
            // The engine will print the prompt and instantly stream newly predicted tokens
            engine.generate(prompt: prompt, maxTokens: 96)
            
            let genTime = CFAbsoluteTimeGetCurrent() - genStartTime
            print("\n-------------------------")
            print("Generation completed in \(String(format: "%.2f", genTime)) seconds.")
            
        } catch {
            print("\nFatal Error: \(error)")
        }
        sema.signal()
    }
    sema.wait()
}

// Execute the inference script
runInference()

// Only run once to download
/*
Task {
    do {
        let pipeline = QwenPipeline()
        try await pipeline.run()
        exit(0)
    } catch {
        print("Fatal Error: \(error)")
        exit(1)
    }
}
RunLoop.main.run()
*/
