//
//  main.swift
//  QwenInferenceEngine
//
//  Created by Raiden-Makoto on 2026-02-13.
//

import Foundation
import Metal

func testModelLoading(){
    guard let device = MTLCreateSystemDefaultDevice() else {
        fatalError("Metal is not supported on this device")
    }
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let weightsPath = cwd.appendingPathComponent("weights").path
    print("Starting Model Load Test with Device \(device.name) and Weights at \(weightsPath)")
    let startTime = CFAbsoluteTimeGetCurrent()
    do {
        let _ = try QwenModel(device: device, weightsPath: weightsPath)
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Model Loaded in \(String(format: "%.4f", timeElapsed))s")
    } catch WeightLoadingError.fileNotFound(let path) {
        print("Loading Failed: file not found at \(path)")
    } catch {
        print("Loading Failed: \(error.localizedDescription)")
    }
}

func testKernelLoading() {
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    let kernelsPath = cwd.appendingPathComponent("kernels").path
    print("Starting Kernel Compilation Test")
    let startTime = CFAbsoluteTimeGetCurrent()
    do {
        let _ = try MetalManager(kernelsDir: kernelsPath)
        let timeElapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("Kernels compiled in \(String(format: "%.4f", timeElapsed))s")
    } catch MetalSetupError.fileReadFailed(let path) {
        print("Compilation Failed: Could not read file at path: \(path)")
    } catch MetalSetupError.compilationFailed(let name) {
        print("Compilation Failed: Syntax error in \(name).")
    } catch MetalSetupError.functionNotFound(let name) {
        print("Compilation Failed: Could not find function '\(name)'.")
    } catch {
        print("Initialization Failed: \(error.localizedDescription)")
    }
}

func testTokenizer(prompt: String){
    let sema = DispatchSemaphore(value: 0)
    Task {
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let weightsPath = cwd.appendingPathComponent("weights").path
        print("Starting Tokenizer Test")
        do {
            let TK = try await QwenTokenizer(weightsDir: weightsPath)
            let tokens = TK.encode(prompt: prompt)
            print("Tokenized prompt: \(tokens)")
        } catch {
            print("Tokenizer Test Failed: \(error.localizedDescription)")
        }
        sema.signal()
    }
    sema.wait()
}

testModelLoading()
testKernelLoading()
testTokenizer(prompt: "Song of the Welkin Moon")

