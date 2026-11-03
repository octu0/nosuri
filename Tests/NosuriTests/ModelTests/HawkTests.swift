import XCTest
@testable import Nosuri
#if canImport(MLX)
import MLX
#endif

/// Tests for the Hawk network, streaming state consistency, SafeTensors serialization, and BPTT training.
final class HawkTests: XCTestCase {
    private func makeFeatures(frames: Int, dim: Int, phase: Int = 0) -> [[Float]] {
        var seq: [[Float]] = []
        var t = 0
        while t < frames {
            var frame = [Float](repeating: 0.0, count: dim)
            var d = 0
            while d < dim {
                frame[d] = sin(Float((t + phase) * 7 + d * 13) * 0.37)
                d += 1
            }
            seq.append(frame)
            t += 1
        }
        return seq
    }

    private func smallCPUWeights(convRight: Int = 0) -> HawkWeights {
        return HawkWeights.makeRandom(
            inputDim: 12,
            modelDim: 16,
            rnnDim: 24,
            mlpDim: 32,
            heads: 3,
            convWidth: 4,
            convRight: convRight,
            numLayers: 2,
            outputDim: 7,
            seed: 7
        )
    }

    /// Chunked CPU inference with carried state matches full-sequence inference.
    func testChunkedStreamingMatchesFullSequence() {
        for (convRight, chunk) in [(0, 1), (0, 5), (2, 4), (1, 7)] {
            let weights = smallCPUWeights(convRight: convRight)
            let net = HawkNetwork(weights: weights)
            let features = makeFeatures(frames: 31, dim: 12)
            let full = net.logProbs(features: features, chunkFrames: 64)
            let chunked = net.logProbs(features: features, chunkFrames: chunk)
            XCTAssertEqual(full.count, chunked.count)
            for t in 0..<full.count {
                for c in 0..<full[t].count {
                    XCTAssertEqual(full[t][c], chunked[t][c], accuracy: 1e-4)
                }
            }
        }
    }

    func testWeightsRoundTripThroughFile() throws {
        let weights = smallCPUWeights(convRight: 1)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("hawk_roundtrip_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try weights.save(to: url)
        let loaded = try HawkWeights.load(from: url)
        XCTAssertEqual(loaded, weights)
    }

    func testCPUClassificationTrainingReducesLoss() {
        let weights = smallCPUWeights(convRight: 0)
        let trainer = HawkCPUTrainer(weights: weights, learningRate: 0.01)
        let batch: [[[Float]]] = [makeFeatures(frames: 12, dim: 12), makeFeatures(frames: 10, dim: 12, phase: 5)]
        let targets: [[Int]] = [Array(repeating: 2, count: 12), Array(repeating: 5, count: 10)]
        let first = trainer.trainBatchClassification(featuresBatch: batch, targetsBatch: targets)
        var last = first
        var step = 0
        while step < 25 {
            last = trainer.trainBatchClassification(featuresBatch: batch, targetsBatch: targets)
            step += 1
        }
        XCTAssertFalse(last.isNaN)
        XCTAssertLessThan(last, first * 0.5)
        XCTAssertNotEqual(trainer.weights.layers[0].wA, weights.layers[0].wA)
    }
}
