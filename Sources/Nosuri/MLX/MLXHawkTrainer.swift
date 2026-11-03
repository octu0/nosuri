#if canImport(MLX)
import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Sequence classification training for the Hawk reaction model. RG-LRU is a linear recurrence,
/// so gradients flow through the whole comment stream sequence without truncation.
public final class MLXHawkTrainer: @unchecked Sendable {
    public let network: MLXHawkNetwork
    public let optimizer: ArrayLRAdam

    /// Gradient norm of the most recent batch, before clipping.
    public private(set) var lastGradientNorm: Float = 0.0

    /// Compiled training steps keyed by padded sequence length (frames rounded up to a multiple of 32).
    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// MLX buffer cache limit at the default width.
    public static let cacheLimitBytes = 24 << 30
    public static let minCacheLimitBytes = 6 << 30

    /// Adam updates since the moments were last zeroed; ArrayLRAdam has no bias correction, so it is applied
    /// to the learning rate here.
    private var adamSteps = 0

    public init(network: MLXHawkNetwork, learningRate: Float) {
        self.network = network
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        let scaled = Self.cacheLimitBytes / max(1, network.modelDim / HawkShape.defaultModelDim)
        MLX.GPU.set(cacheLimit: max(Self.minCacheLimitBytes, scaled))
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// Restores saved weights and resets the Adam moments.
    public func rollback(to weights: HawkWeights) {
        network.importWeights(from: weights)
        optimizer.resetState()
        adamSteps = 0
        compiledSteps.removeAll()
        eval(network, optimizer)
    }

    /// Inference logits [B, T, outputDim].
    public func logits(features: MLXArray) -> MLXArray {
        return network(features)
    }

    /// Runs one frame-level cross-entropy classification training step over comment sequences and returns the loss.
    /// `targetsBatch[b][t]` is the target reaction index in `0..<outputDim` for frame `t` of sequence `b`.
    public func trainBatchClassification(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        compiled: Bool = true
    ) -> Float {
        var validFeatures: [[[Float]]] = []
        var validTargets: [[Int]] = []
        var b = 0
        while b < min(featuresBatch.count, targetsBatch.count) {
            let frames = min(featuresBatch[b].count, targetsBatch[b].count)
            if 0 < frames {
                validFeatures.append(Array(featuresBatch[b].prefix(frames)))
                validTargets.append(Array(targetsBatch[b].prefix(frames)))
            }
            b += 1
        }
        if validFeatures.isEmpty {
            return 0.0
        }
        let (features, maxT) = padded(validFeatures)
        var targetFlat = [Int32](repeating: 0, count: validFeatures.count * maxT)
        var maskFlat = [Float](repeating: 0.0, count: validFeatures.count * maxT)
        var frameCounts = [Int32](repeating: 0, count: validFeatures.count)
        b = 0
        while b < validFeatures.count {
            let frames = validFeatures[b].count
            frameCounts[b] = Int32(frames)
            var t = 0
            while t < frames {
                targetFlat[b * maxT + t] = Int32(validTargets[b][t])
                maskFlat[b * maxT + t] = 1.0
                t += 1
            }
            b += 1
        }
        let targets = MLXArray(targetFlat, [validFeatures.count, maxT])
        let mask = MLXArray(maskFlat, [validFeatures.count, maxT])
        let counts = MLXArray(frameCounts)

        let lossFn = { (model: MLXHawkNetwork, arrays: [MLXArray]) -> [MLXArray] in
            let logits = model(arrays[0], frameCounts: arrays[3])
            let logp = logSoftmax(logits, axis: -1)
            let picked = takeAlong(logp, arrays[1].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let validMask = arrays[2]
            return [-sum(picked * validMask) / maximum(sum(validMask), MLXArray(Float(1.0)))]
        }
        let lg = valueAndGrad(model: network, lossFn)
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (lossValues, grads) = lg(self.network, Array(arrays[0..<4]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: 5.0)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[4]
            self.optimizer.update(model: self.network, gradients: clipped)
            self.optimizer.learningRate = outer
            return lossValues + [norm]
        }
        var stepFn: ([MLXArray]) -> [MLXArray] = step
        if compiled {
            switch compiledSteps[maxT] {
            case .some(let cached):
                stepFn = cached
            case .none:
                let fn = compile(inputs: [network, optimizer], outputs: [network, optimizer], step)
                compiledSteps[maxT] = fn
                stepFn = fn
            }
        }
        adamSteps += 1
        let t = Float(adamSteps)
        let lr = optimizer.learningRate * ((1.0 - pow(optimizer.betas.1, t)).squareRoot() / (1.0 - pow(optimizer.betas.0, t)))
        let lossValues = stepFn([features, targets, mask, counts, lr])
        eval(network, optimizer, lossValues)
        lastGradientNorm = lossValues[1].item(Float.self)
        return lossValues[0].item(Float.self)
    }

    /// Stacks sequences into [B, T, inputDim], zero-padding T up to a multiple of 32 to limit distinct compiled shapes.
    private func padded(_ batch: [[[Float]]]) -> (MLXArray, Int) {
        var maxT = 0
        for f in batch {
            maxT = max(maxT, f.count)
        }
        maxT = ((maxT + 31) / 32) * 32
        let inDim = network.inputDim
        var flat = [Float](repeating: 0.0, count: batch.count * maxT * inDim)
        var b = 0
        while b < batch.count {
            var t = 0
            while t < batch[b].count {
                let offset = (b * maxT + t) * inDim
                let row = batch[b][t]
                var d = 0
                while d < min(inDim, row.count) {
                    flat[offset + d] = row[d]
                    d += 1
                }
                t += 1
            }
            b += 1
        }
        return (MLXArray(flat, [batch.count, maxT, inDim]), maxT)
    }
}
#endif

