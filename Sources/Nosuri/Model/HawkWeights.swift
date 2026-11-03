import Foundation

/// Default model shape for the Hawk streaming model, shared by training and inference.
public enum HawkShape {
    public static let numLayers = 4
    public static let modelDim = 288
    /// Width the memory budgets are calibrated at (batch sizes and the MLX buffer cache scale down from it).
    public static let defaultModelDim = 288
    public static let rnnDim = 384
    public static let mlpDim = 864
    /// Number of diagonal blocks in the RG-LRU gates.
    public static let heads = 8
    public static let convWidth = 4
    /// Future frames per layer. Strictly causal (0) by default for low-latency live reaction streaming.
    public static let convRight = 0
    /// Griffin-style local attention blocks, each run after the recurrent layer with this index.
    public static let attentionAfter = [1, 3]
    /// Query heads of the local attention; keys and values share one head (multi-query).
    public static let attentionHeads = 8
    /// Frames each query may look back over, the current frame included. Strictly causal.
    public static let attentionWindow = 64
}

/// Weights of the Hawk sequence model (the recurrent-only variant of Griffin, De et al. 2024).
/// Each layer is RMSNorm → recurrent block → residual, then RMSNorm → gated MLP → residual.
/// The recurrent block multiplies a GeLU branch with a linear → causal conv → RG-LRU branch, then projects back.
/// All matrices are row-major [input, output] (applied as x · W).
public struct HawkWeights: Sendable, Equatable {
    public let inputDim: Int
    public let modelDim: Int
    /// Width of the recurrent block.
    public let rnnDim: Int
    /// Hidden width of the MLP.
    public let mlpDim: Int
    /// Number of diagonal blocks in the RG-LRU gate projections.
    public let heads: Int
    public let convWidth: Int
    /// Future frames seen by the convolution (lookahead). Adds this much latency per layer.
    public let convRight: Int
    public let outputDim: Int
    /// Query heads of the local attention blocks (0 when there are none).
    public let attentionHeads: Int
    /// Causal window of the local attention blocks in frames.
    public let attentionWindow: Int

    public let wIn: [Float]          // [inputDim, modelDim]
    /// Input bias. Keeps the residual stream nonzero on all-zero (idle) feature frames; otherwise each
    /// RMSNorm backward pass amplifies gradients by up to 1/√eps and deep stacks overflow.
    public let bIn: [Float]          // [modelDim]
    public let layers: [HawkLayerWeights]
    /// Local attention blocks, in order of `after`.
    public let attention: [HawkAttentionLayerWeights]
    public let finalNorm: [Float]    // [modelDim]
    public let wOut: [Float]         // [modelDim, outputDim]
    public let bOut: [Float]         // [outputDim]

    /// RG-LRU decay exponent c: a_t = σ(Λ)^(c·r_t).
    public static let decayExponent: Float = 8.0
    /// Epsilon inside RMSNorm, shared by the MLX trainer and the CPU network.
    public static let rmsNormEpsilon: Float = 1e-5
    /// Initial time constants (frames) spread log-uniformly across channels at r = 0.5.
    public static let initialTimeConstants: ClosedRange<Float> = 1.0...100.0

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        convRight: Int,
        outputDim: Int,
        wIn: [Float],
        bIn: [Float],
        layers: [HawkLayerWeights],
        finalNorm: [Float],
        wOut: [Float],
        bOut: [Float],
        attention: [HawkAttentionLayerWeights] = [],
        attentionHeads: Int = 0,
        attentionWindow: Int = 0
    ) {
        self.inputDim = inputDim
        self.attention = attention
        self.attentionHeads = attentionHeads
        self.attentionWindow = attentionWindow
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        self.outputDim = outputDim
        self.wIn = wIn
        self.bIn = bIn
        self.layers = layers
        self.finalNorm = finalNorm
        self.wOut = wOut
        self.bOut = bOut
    }

    public var numLayers: Int {
        return layers.count
    }

    /// Attention head width (modelDim / attentionHeads); 0 without attention.
    public var attentionHeadDim: Int {
        if attentionHeads <= 0 {
            return 0
        }
        return modelDim / attentionHeads
    }

    /// Convolution taps: `convWidth - 1` past, the current frame, and `convRight` future.
    public var convTaps: Int {
        return convWidth + convRight
    }

    /// Input-to-output latency in frames.
    public var lookaheadFrames: Int {
        return convRight * numLayers
    }

    public var headDim: Int {
        return rnnDim / heads
    }

    public var parameterCount: Int {
        var n = wIn.count + bIn.count + finalNorm.count + wOut.count + bOut.count
        for layer in layers {
            n += layer.parameterCount
        }
        for block in attention {
            n += block.parameterCount
        }
        return n
    }

    /// Initial Λ such that σ(Λ)^(c/2) = exp(-1/τ) for τ spread over `initialTimeConstants`.
    public static func initialLambda(rnnDim: Int) -> [Float] {
        let lo = logf(initialTimeConstants.lowerBound)
        let hi = logf(initialTimeConstants.upperBound)
        var values = [Float](repeating: 0.0, count: rnnDim)
        var n = 0
        while n < rnnDim {
            var frac: Float = 0.0
            if 1 < rnnDim {
                frac = Float(n) / Float(rnnDim - 1)
            }
            let tau = expf(lo + (hi - lo) * frac)
            let a = expf(-1.0 / (tau * decayExponent * 0.5))
            values[n] = logf(a / (1.0 - a))
            n += 1
        }
        return values
    }

    /// Saves as safetensors: tensors named after the fields (`layers.<i>.<field>`, `attention.<j>.<field>`) with the
    /// shapes in the field comments, and the sizes and `attention.<j>.after` in the metadata.
    public func save(to url: URL) throws {
        typealias T = SafeTensors.Tensor
        var tensors: [(name: String, tensor: T)] = [
            ("wIn", T(shape: [inputDim, modelDim], values: wIn)),
            ("bIn", T(shape: [modelDim], values: bIn)),
        ]
        var metadata = [
            "inputDim": "\(inputDim)", "modelDim": "\(modelDim)", "rnnDim": "\(rnnDim)", "mlpDim": "\(mlpDim)",
            "heads": "\(heads)", "convWidth": "\(convWidth)", "convRight": "\(convRight)", "outputDim": "\(outputDim)",
            "attentionHeads": "\(attentionHeads)", "attentionWindow": "\(attentionWindow)",
            "numLayers": "\(layers.count)", "attentionBlocks": "\(attention.count)",
        ]
        for (i, layer) in layers.enumerated() {
            let p = "layers.\(i)."
            tensors += [
                (p + "norm1", T(shape: [modelDim], values: layer.norm1)),
                (p + "wGate", T(shape: [modelDim, rnnDim], values: layer.wGate)),
                (p + "wX", T(shape: [modelDim, rnnDim], values: layer.wX)),
                (p + "convW", T(shape: [convTaps, rnnDim], values: layer.convW)),
                (p + "convB", T(shape: [rnnDim], values: layer.convB)),
                (p + "wA", T(shape: [heads, headDim, headDim], values: layer.wA)),
                (p + "bA", T(shape: [rnnDim], values: layer.bA)),
                (p + "wI", T(shape: [heads, headDim, headDim], values: layer.wI)),
                (p + "bI", T(shape: [rnnDim], values: layer.bI)),
                (p + "lambda", T(shape: [rnnDim], values: layer.lambda)),
                (p + "wO", T(shape: [rnnDim, modelDim], values: layer.wO)),
                (p + "norm2", T(shape: [modelDim], values: layer.norm2)),
                (p + "wMLPGate", T(shape: [modelDim, mlpDim], values: layer.wMLPGate)),
                (p + "wMLPUp", T(shape: [modelDim, mlpDim], values: layer.wMLPUp)),
                (p + "wMLPDown", T(shape: [mlpDim, modelDim], values: layer.wMLPDown)),
            ]
        }
        for (j, block) in attention.enumerated() {
            let p = "attention.\(j)."
            metadata[p + "after"] = "\(block.after)"
            tensors += [
                (p + "norm1", T(shape: [modelDim], values: block.norm1)),
                (p + "wQ", T(shape: [modelDim, attentionHeads * attentionHeadDim], values: block.wQ)),
                (p + "wK", T(shape: [modelDim, attentionHeadDim], values: block.wK)),
                (p + "wV", T(shape: [modelDim, attentionHeadDim], values: block.wV)),
                (p + "wO", T(shape: [attentionHeads * attentionHeadDim, modelDim], values: block.wO)),
                (p + "norm2", T(shape: [modelDim], values: block.norm2)),
                (p + "wMLPGate", T(shape: [modelDim, mlpDim], values: block.wMLPGate)),
                (p + "wMLPUp", T(shape: [modelDim, mlpDim], values: block.wMLPUp)),
                (p + "wMLPDown", T(shape: [mlpDim, modelDim], values: block.wMLPDown)),
            ]
        }
        tensors += [
            ("finalNorm", T(shape: [modelDim], values: finalNorm)),
            ("wOut", T(shape: [modelDim, outputDim], values: wOut)),
            ("bOut", T(shape: [outputDim], values: bOut)),
        ]
        try SafeTensors.write(tensors, metadata: metadata, to: url)
    }

    public static func load(from url: URL) throws -> HawkWeights {
        let (tensors, metadata) = try SafeTensors.read(from: url)
        func size(_ key: String) throws -> Int {
            guard let text = metadata[key], let value = Int(text) else {
                throw SafeTensors.Failure.badHeader
            }
            return value
        }
        func values(_ name: String) throws -> [Float] {
            guard let tensor = tensors[name] else {
                throw SafeTensors.Failure.badShape(name)
            }
            return tensor.values
        }
        var layers: [HawkLayerWeights] = []
        var i = 0
        while i < (try size("numLayers")) {
            let p = "layers.\(i)."
            layers.append(HawkLayerWeights(
                norm1: try values(p + "norm1"), wGate: try values(p + "wGate"), wX: try values(p + "wX"),
                convW: try values(p + "convW"), convB: try values(p + "convB"), wA: try values(p + "wA"), bA: try values(p + "bA"),
                wI: try values(p + "wI"), bI: try values(p + "bI"), lambda: try values(p + "lambda"), wO: try values(p + "wO"),
                norm2: try values(p + "norm2"), wMLPGate: try values(p + "wMLPGate"), wMLPUp: try values(p + "wMLPUp"),
                wMLPDown: try values(p + "wMLPDown")))
            i += 1
        }
        var attention: [HawkAttentionLayerWeights] = []
        var j = 0
        while j < (try size("attentionBlocks")) {
            let p = "attention.\(j)."
            attention.append(HawkAttentionLayerWeights(
                after: try size(p + "after"), norm1: try values(p + "norm1"), wQ: try values(p + "wQ"), wK: try values(p + "wK"),
                wV: try values(p + "wV"), wO: try values(p + "wO"), norm2: try values(p + "norm2"),
                wMLPGate: try values(p + "wMLPGate"), wMLPUp: try values(p + "wMLPUp"), wMLPDown: try values(p + "wMLPDown")))
            j += 1
        }
        return HawkWeights(
            inputDim: try size("inputDim"), modelDim: try size("modelDim"), rnnDim: try size("rnnDim"), mlpDim: try size("mlpDim"),
            heads: try size("heads"), convWidth: try size("convWidth"), convRight: try size("convRight"),
            outputDim: try size("outputDim"), wIn: try values("wIn"), bIn: try values("bIn"), layers: layers,
            finalNorm: try values("finalNorm"), wOut: try values("wOut"), bOut: try values("bOut"),
            attention: attention,
            attentionHeads: try size("attentionHeads"), attentionWindow: try size("attentionWindow"))
    }
}

/// Weights of one Hawk layer.
public struct HawkLayerWeights: Sendable, Equatable {
    public let norm1: [Float]       // [modelDim]
    public let wGate: [Float]       // [modelDim, rnnDim] GeLU branch
    public let wX: [Float]          // [modelDim, rnnDim] recurrent branch
    public let convW: [Float]       // [convWidth + convRight, rnnDim]; row convWidth - 1 is the current frame, later rows are future
    public let convB: [Float]       // [rnnDim]
    public let wA: [Float]          // [heads, headDim, headDim] recurrence gate (block-diagonal)
    public let bA: [Float]          // [rnnDim]
    public let wI: [Float]          // [heads, headDim, headDim] input gate (block-diagonal)
    public let bI: [Float]          // [rnnDim]
    public let lambda: [Float]      // [rnnDim] logit Λ of the decay σ(Λ)
    public let wO: [Float]          // [rnnDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(
        norm1: [Float], wGate: [Float], wX: [Float], convW: [Float], convB: [Float],
        wA: [Float], bA: [Float], wI: [Float], bI: [Float], lambda: [Float], wO: [Float],
        norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]
    ) {
        self.norm1 = norm1
        self.wGate = wGate
        self.wX = wX
        self.convW = convW
        self.convB = convB
        self.wA = wA
        self.bA = bA
        self.wI = wI
        self.bI = bI
        self.lambda = lambda
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wGate.count + wX.count + convW.count + convB.count + wA.count + bA.count
            + wI.count + bI.count + lambda.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }
}

/// Weights of one Griffin-style local attention block (RMSNorm → multi-query attention over the last
/// `attentionWindow` frames with rotary positions → residual, then RMSNorm → gated MLP → residual).
public struct HawkAttentionLayerWeights: Sendable, Equatable {
    /// Index of the recurrent layer this block follows.
    public let after: Int
    public let norm1: [Float]       // [modelDim]
    public let wQ: [Float]          // [modelDim, heads × headDim]
    public let wK: [Float]          // [modelDim, headDim] shared key head
    public let wV: [Float]          // [modelDim, headDim] shared value head
    public let wO: [Float]          // [heads × headDim, modelDim]
    public let norm2: [Float]       // [modelDim]
    public let wMLPGate: [Float]    // [modelDim, mlpDim]
    public let wMLPUp: [Float]      // [modelDim, mlpDim]
    public let wMLPDown: [Float]    // [mlpDim, modelDim]

    public init(after: Int, norm1: [Float], wQ: [Float], wK: [Float], wV: [Float], wO: [Float],
                norm2: [Float], wMLPGate: [Float], wMLPUp: [Float], wMLPDown: [Float]) {
        self.after = after
        self.norm1 = norm1
        self.wQ = wQ
        self.wK = wK
        self.wV = wV
        self.wO = wO
        self.norm2 = norm2
        self.wMLPGate = wMLPGate
        self.wMLPUp = wMLPUp
        self.wMLPDown = wMLPDown
    }

    public var parameterCount: Int {
        return norm1.count + wQ.count + wK.count + wV.count + wO.count + norm2.count + wMLPGate.count + wMLPUp.count + wMLPDown.count
    }

    /// Rotary position angle for pair `i` of `headDim / 2` at frame `position` (base 10000).
    public static func rotaryAngle(position: Int, pair: Int, headDim: Int) -> Float {
        let exponent = -2.0 * Float(pair) / Float(headDim)
        return Float(position) * powf(10000.0, exponent)
    }
}

