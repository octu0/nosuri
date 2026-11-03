#if canImport(MLX)
import Foundation
import MLX
import MLXNN

/// Linear recurrence h_t = a_t h_{t-1} + x_t (h_{-1} = 0) over [B, T, R], one Metal thread per channel.
/// The backward pass is a reverse-time scan kernel, so graph size does not grow with sequence length.
public enum MLXLinearScan {
    static let forwardKernel = MLXFast.metalKernel(
        name: "hawk_linear_scan_forward",
        inputNames: ["a", "x"],
        outputNames: ["h"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float state = 0.0f;
                for (int t = 0; t < T; ++t) {
                    size_t idx = base + size_t(t) * R;
                    state = a[idx] * state + x[idx];
                    h[idx] = state;
                }
            }
            """
    )

    /// δ_t = g_t + a_{t+1} δ_{t+1}, dx_t = δ_t, da_t = δ_t h_{t-1}.
    static let backwardKernel = MLXFast.metalKernel(
        name: "hawk_linear_scan_backward",
        inputNames: ["a", "h", "g"],
        outputNames: ["da", "dx"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float carry = 0.0f;
                for (int t = T - 1; 0 <= t; --t) {
                    size_t idx = base + size_t(t) * R;
                    float d = g[idx] + carry;
                    dx[idx] = d;
                    float previous = 0.0f;
                    if (0 < t) {
                        previous = h[idx - R];
                    }
                    da[idx] = d * previous;
                    carry = d * a[idx];
                }
            }
            """
    )

    static func launch(_ shape: [Int]) -> (template: [(String, any KernelTemplateArg)], grid: (Int, Int, Int), group: (Int, Int, Int)) {
        let threads = shape[0] * shape[2]
        return (
            template: [("B", shape[0]), ("T", shape[1]), ("R", shape[2])],
            grid: (threads, 1, 1),
            group: (min(256, threads), 1, 1)
        )
    }

    static func forward(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        let p = launch(a.shape)
        return forwardKernel(
            [a, x], template: p.template, grid: p.grid, threadGroup: p.group,
            outputShapes: [a.shape], outputDTypes: [.float32]
        )[0]
    }

    nonisolated(unsafe) static let function = CustomFunction {
        Forward { inputs in
            return [MLXLinearScan.forward(inputs[0], inputs[1])]
        }
        VJP { primals, cotangents in
            let a = primals[0]
            let h = MLXLinearScan.forward(a, primals[1])
            let p = launch(a.shape)
            return backwardKernel(
                [a, h, cotangents[0]], template: p.template, grid: p.grid, threadGroup: p.group,
                outputShapes: [a.shape, a.shape], outputDTypes: [.float32, .float32]
            )
        }
    }

    /// a, x: [B, T, R] (float32). Returns h: [B, T, R].
    public static func scan(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        return function([a.asType(.float32), x.asType(.float32)])[0]
    }
}

/// Causal local multi-query attention: out[b, t, h] = Σ_o p_o v[b, t-o] over the last W frames (0 ≤ o < W, t-o ≥ 0),
/// p = softmax_o(q[b, t, h] · k[b, t-o]). q: [B, T, H, D] (already scaled), k, v: [B, T, D].
/// One SIMD group per (b, t, h) row; each lane holds every 32nd channel and every 32nd offset's score.
public enum MLXLocalAttention {
    static let forwardKernel = MLXFast.metalKernel(
        name: "hawk_local_attention_forward",
        inputNames: ["q", "k", "v"],
        outputNames: ["out", "lse"],
        source: """
            constexpr int ND = (D + 31) / 32;
            constexpr int NW = (W + 31) / 32;
            uint row = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            if (uint(B * T * H) <= row) {
                return;
            }
            uint bt = row / H;
            int t = int(bt % T);
            size_t frame0 = size_t(bt - t) * D;
            int span = min(W, t + 1);
            float qr[ND];
            for (int j = 0; j < ND; ++j) {
                int d = int(lane) + 32 * j;
                qr[j] = d < D ? q[size_t(row) * D + d] : 0.0f;
            }
            float sc[NW];
            float top = -INFINITY;
            for (int o = 0; o < span; ++o) {
                size_t kb = frame0 + size_t(t - o) * D;
                float part = 0.0f;
                for (int j = 0; j < ND; ++j) {
                    int d = int(lane) + 32 * j;
                    if (d < D) {
                        part += qr[j] * k[kb + d];
                    }
                }
                float s = simd_sum(part);
                if ((o & 31) == int(lane)) {
                    sc[o >> 5] = s;
                }
                top = max(top, s);
            }
            float local = 0.0f;
            for (int j = 0; j < NW; ++j) {
                int o = int(lane) + 32 * j;
                if (o < span) {
                    local += exp(sc[j] - top);
                }
            }
            float logZ = top + log(simd_sum(local));
            float acc[ND];
            for (int j = 0; j < ND; ++j) {
                acc[j] = 0.0f;
            }
            for (int o = 0; o < span; ++o) {
                float p = exp(simd_shuffle(sc[o >> 5], ushort(o & 31)) - logZ);
                size_t vb = frame0 + size_t(t - o) * D;
                for (int j = 0; j < ND; ++j) {
                    int d = int(lane) + 32 * j;
                    if (d < D) {
                        acc[j] += p * v[vb + d];
                    }
                }
            }
            for (int j = 0; j < ND; ++j) {
                int d = int(lane) + 32 * j;
                if (d < D) {
                    out[size_t(row) * D + d] = acc[j];
                }
            }
            if (lane == 0) {
                lse[row] = logZ;
            }
            """
    )

    /// Per row: dq, and the probabilities p and score gradients ds = p (dp - Σ p dp) for every offset.
    static let backwardRowKernel = MLXFast.metalKernel(
        name: "hawk_local_attention_backward_rows",
        inputNames: ["q", "k", "v", "out", "lse", "g"],
        outputNames: ["dq", "prob", "dscore"],
        source: """
            constexpr int ND = (D + 31) / 32;
            uint row = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            if (uint(B * T * H) <= row) {
                return;
            }
            uint bt = row / H;
            int t = int(bt % T);
            size_t frame0 = size_t(bt - t) * D;
            int span = min(W, t + 1);
            float qr[ND];
            float gr[ND];
            float dotOut = 0.0f;
            for (int j = 0; j < ND; ++j) {
                int d = int(lane) + 32 * j;
                qr[j] = 0.0f;
                gr[j] = 0.0f;
                if (d < D) {
                    size_t i = size_t(row) * D + d;
                    qr[j] = q[i];
                    gr[j] = g[i];
                    dotOut += gr[j] * out[i];
                }
            }
            float delta = simd_sum(dotOut);
            float logZ = lse[row];
            float acc[ND];
            for (int j = 0; j < ND; ++j) {
                acc[j] = 0.0f;
            }
            for (int o = 0; o < span; ++o) {
                size_t kb = frame0 + size_t(t - o) * D;
                float sPart = 0.0f;
                float dpPart = 0.0f;
                for (int j = 0; j < ND; ++j) {
                    int d = int(lane) + 32 * j;
                    if (d < D) {
                        sPart += qr[j] * k[kb + d];
                        dpPart += gr[j] * v[kb + d];
                    }
                }
                float p = exp(simd_sum(sPart) - logZ);
                float ds = p * (simd_sum(dpPart) - delta);
                for (int j = 0; j < ND; ++j) {
                    int d = int(lane) + 32 * j;
                    if (d < D) {
                        acc[j] += ds * k[kb + d];
                    }
                }
                if (lane == 0) {
                    prob[size_t(row) * W + o] = p;
                    dscore[size_t(row) * W + o] = ds;
                }
            }
            for (int j = 0; j < ND; ++j) {
                int d = int(lane) + 32 * j;
                if (d < D) {
                    dq[size_t(row) * D + d] = acc[j];
                }
            }
            """
    )

    /// Per key frame s: dk[s] = Σ_{o,h} ds[s+o, h, o] q[s+o, h], dv[s] = Σ_{o,h} p[s+o, h, o] g[s+o, h].
    static let backwardKeyKernel = MLXFast.metalKernel(
        name: "hawk_local_attention_backward_keys",
        inputNames: ["q", "g", "prob", "dscore"],
        outputNames: ["dk", "dv"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (uint(B * T * D) <= gid) {
                return;
            }
            int d = int(gid % D);
            uint bs = gid / D;
            int s = int(bs % T);
            size_t b0 = size_t(bs - s);
            float accK = 0.0f;
            float accV = 0.0f;
            int span = min(W, T - s);
            for (int o = 0; o < span; ++o) {
                size_t row0 = (b0 + s + o) * H;
                for (int h = 0; h < H; ++h) {
                    size_t row = row0 + h;
                    accK += dscore[row * W + o] * q[row * D + d];
                    accV += prob[row * W + o] * g[row * D + d];
                }
            }
            dk[gid] = accK;
            dv[gid] = accV;
            """
    )

    static func template(_ q: MLXArray, window: Int) -> [(String, any KernelTemplateArg)] {
        return [("B", q.dim(0)), ("T", q.dim(1)), ("H", q.dim(2)), ("D", q.dim(3)), ("W", window)]
    }

    static func forward(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, window: Int) -> [MLXArray] {
        let rows = q.dim(0) * q.dim(1) * q.dim(2)
        return forwardKernel(
            [q, k, v], template: template(q, window: window), grid: (rows * 32, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [q.shape, [rows]], outputDTypes: [.float32, .float32]
        )
    }

    nonisolated(unsafe) static var functions: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    static let lock = NSLock()

    static func function(window: Int) -> ([MLXArray]) -> [MLXArray] {
        lock.lock()
        defer { lock.unlock() }
        if let f = functions[window] {
            return f
        }
        let f = CustomFunction {
            Forward { inputs in
                return [MLXLocalAttention.forward(inputs[0], inputs[1], inputs[2], window: window)[0]]
            }
            VJP { primals, cotangents in
                let q = primals[0]
                let k = primals[1]
                let v = primals[2]
                let forwardOutputs = MLXLocalAttention.forward(q, k, v, window: window)
                let rows = q.dim(0) * q.dim(1) * q.dim(2)
                let tmpl = MLXLocalAttention.template(q, window: window)
                let rowGrads = backwardRowKernel(
                    [q, k, v, forwardOutputs[0], forwardOutputs[1], cotangents[0]], template: tmpl,
                    grid: (rows * 32, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [q.shape, [rows, window], [rows, window]], outputDTypes: [.float32, .float32, .float32]
                )
                let keyCount = k.size
                let keyGrads = backwardKeyKernel(
                    [q, cotangents[0], rowGrads[1], rowGrads[2]], template: tmpl,
                    grid: (keyCount, 1, 1), threadGroup: (min(256, keyCount), 1, 1),
                    outputShapes: [k.shape, v.shape], outputDTypes: [.float32, .float32]
                )
                return [rowGrads[0], keyGrads[0], keyGrads[1]]
            }
        }
        functions[window] = f
        return f
    }

    /// q: [B, T, H, D], k, v: [B, T, D] (float32). Returns [B, T, H, D].
    public static func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, window: Int) -> MLXArray {
        return function(window: window)([q.asType(.float32), k.asType(.float32), v.asType(.float32)])[0]
    }
}

/// GeLU approximation x σ(1.702 x); must match the CPU inference path.
@inline(__always)
func hawkGelu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(1.702 * x)
}

@inline(__always)
func hawkRMSNorm(_ x: MLXArray, _ gain: MLXArray) -> MLXArray {
    return x * rsqrt(mean(x * x, axis: -1, keepDims: true) + HawkWeights.rmsNormEpsilon) * gain
}

/// One Hawk layer. See `HawkLayerWeights` for the meaning of each weight.
public final class MLXHawkLayer: Module, @unchecked Sendable {
    public var norm1: MLXArray
    public var wGate: MLXArray
    public var wX: MLXArray
    public var convW: MLXArray
    public var convB: MLXArray
    public var wA: MLXArray
    public var bA: MLXArray
    public var wI: MLXArray
    public var bI: MLXArray
    public var lambda: MLXArray
    public var wO: MLXArray
    public var norm2: MLXArray
    public var wMLPGate: MLXArray
    public var wMLPUp: MLXArray
    public var wMLPDown: MLXArray

    let heads: Int
    let convWidth: Int
    let convRight: Int

    init(modelDim: Int, rnnDim: Int, mlpDim: Int, heads: Int, convWidth: Int, convRight: Int, numLayers: Int) {
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        let headDim = rnnDim / heads
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        // Scale residual output projections by 1/√(2L) so the initial residual variance is independent of depth.
        let residualScale = 1.0 / sqrt(Float(2 * numLayers))
        self.norm1 = MLXArray.ones([modelDim])
        self.wGate = uniform(modelDim, [modelDim, rnnDim])
        self.wX = uniform(modelDim, [modelDim, rnnDim])
        self.convW = uniform(convWidth + convRight, [convWidth + convRight, rnnDim])
        self.convB = MLXArray.zeros([rnnDim])
        self.wA = uniform(headDim, [heads, headDim, headDim])
        self.bA = MLXArray.zeros([rnnDim])
        self.wI = uniform(headDim, [heads, headDim, headDim])
        self.bI = MLXArray.zeros([rnnDim])
        self.lambda = MLXArray(HawkWeights.initialLambda(rnnDim: rnnDim), [rnnDim])
        self.wO = uniform(rnnDim, [rnnDim, modelDim], scale: residualScale)
        self.norm2 = MLXArray.ones([modelDim])
        self.wMLPGate = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPUp = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPDown = uniform(mlpDim, [mlpDim, modelDim], scale: residualScale)
        super.init()
    }

    /// x: [B, T, modelDim] → [B, T, modelDim]. `frameMask` [B, T, 1] zeroes the convolution input past each
    /// sequence end, matching the zero padding the CPU path uses to flush lookahead.
    func callAsFunction(_ x: MLXArray, frameMask: MLXArray?) -> MLXArray {
        let batch = x.shape[0]
        let frames = x.shape[1]
        let rnnDim = wX.shape[1]
        let headDim = rnnDim / heads

        let n = hawkRMSNorm(x, norm1)
        let branch = hawkGelu(matmul(n, wGate))
        var raw = matmul(n, wX)
        if let mask = frameMask {
            raw = raw * mask
        }
        // Zero-pad convWidth - 1 frames before and convRight after; each frame sees
        // convWidth - 1 past frames, itself, and convRight future frames.
        let paddedRaw = concatenated(
            [MLXArray.zeros([batch, convWidth - 1, rnnDim]), raw, MLXArray.zeros([batch, convRight, rnnDim])], axis: 1)
        var u = convB + MLXArray.zeros([batch, frames, rnnDim])
        var k = 0
        while k < convWidth + convRight {
            u = u + paddedRaw[0..., k..<(k + frames), 0...] * convW[k]
            k += 1
        }
        // Block-diagonal gates as one batched matmul per head, [heads, B·T, headDim] · [heads, headDim, headDim].
        let blocks = u.reshaped([batch * frames, heads, headDim]).transposed(1, 0, 2)
        func gate(_ w: MLXArray, _ bias: MLXArray) -> MLXArray {
            return sigmoid(matmul(blocks, w).transposed(1, 0, 2).reshaped([batch, frames, rnnDim]) + bias)
        }
        let r = gate(wA, bA)
        let i = gate(wI, bI)
        // log a_t = c · r_t · log σ(Λ)
        let logA = (HawkWeights.decayExponent * r) * logSigmoid(lambda)
        let a = exp(logA)
        let input = sqrt(1.0 - exp(2.0 * logA) + 1e-6) * (i * u)
        let h = MLXLinearScan.scan(a, input)
        let mixed = x + matmul(branch * h, wO)

        let m = hawkRMSNorm(mixed, norm2)
        return mixed + matmul(hawkGelu(matmul(m, wMLPGate)) * matmul(m, wMLPUp), wMLPDown)
    }
}

/// Rotary position embedding on the last axis (half-split pairs). x: [..., T, ..., headDim] with the time axis
/// given by `cos`/`sin` shaped to broadcast ([T, 1, half] for [B, T, H, D], [T, half] for [B, T, D]).
@inline(__always)
func hawkRotary(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
    let half = x.shape[x.ndim - 1] / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half..<(2 * half)]
    return concatenated([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1)
}

/// Griffin-style local attention block: multi-query attention over the last `window` frames (causal), then a
/// gated MLP. See `HawkAttentionLayerWeights`.
public final class MLXHawkAttentionLayer: Module, @unchecked Sendable {
    public var norm1: MLXArray
    public var wQ: MLXArray
    public var wK: MLXArray
    public var wV: MLXArray
    public var wO: MLXArray
    public var norm2: MLXArray
    public var wMLPGate: MLXArray
    public var wMLPUp: MLXArray
    public var wMLPDown: MLXArray

    let after: Int
    let heads: Int
    let window: Int

    init(after: Int, modelDim: Int, mlpDim: Int, heads: Int, window: Int, residualScale: Float) {
        self.after = after
        self.heads = heads
        self.window = window
        let headDim = modelDim / heads
        func uniform(_ fanIn: Int, _ shape: [Int], scale: Float = 1.0) -> MLXArray {
            let bound = scale * sqrt(3.0 / Float(fanIn))
            return MLXRandom.uniform(low: -bound, high: bound, shape)
        }
        self.norm1 = MLXArray.ones([modelDim])
        self.wQ = uniform(modelDim, [modelDim, heads * headDim])
        self.wK = uniform(modelDim, [modelDim, headDim])
        self.wV = uniform(modelDim, [modelDim, headDim])
        self.wO = uniform(heads * headDim, [heads * headDim, modelDim], scale: residualScale)
        self.norm2 = MLXArray.ones([modelDim])
        self.wMLPGate = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPUp = uniform(modelDim, [modelDim, mlpDim])
        self.wMLPDown = uniform(mlpDim, [mlpDim, modelDim], scale: residualScale)
        super.init()
    }

    /// x: [B, T, modelDim] → [B, T, modelDim].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let frames = x.shape[1]
        let headDim = wK.shape[1]
        let half = headDim / 2
        var angles = [Float](repeating: 0.0, count: frames * half)
        var t = 0
        while t < frames {
            var i = 0
            while i < half {
                angles[t * half + i] = HawkAttentionLayerWeights.rotaryAngle(position: t, pair: i, headDim: headDim)
                i += 1
            }
            t += 1
        }
        let theta = MLXArray(angles, [frames, half])
        let cosT = cos(theta)
        let sinT = sin(theta)

        let n = hawkRMSNorm(x, norm1)
        let q = hawkRotary(matmul(n, wQ).reshaped([batch, frames, heads, headDim]),
                           cos: cosT.reshaped([frames, 1, half]), sin: sinT.reshaped([frames, 1, half])) * (1.0 / sqrt(Float(headDim)))
        let k = hawkRotary(matmul(n, wK), cos: cosT, sin: sinT)
        let v = matmul(n, wV)
        let out = MLXLocalAttention.attend(q, k, v, window: window)
        let mixed = x + matmul(out.reshaped([batch, frames, heads * headDim]), wO)
        let m = hawkRMSNorm(mixed, norm2)
        return mixed + matmul(hawkGelu(matmul(m, wMLPGate)) * matmul(m, wMLPUp), wMLPDown)
    }
}

/// Hawk sequence model on MLX: input projection → layers (with optional local attention blocks) → RMSNorm → readout.
public final class MLXHawkNetwork: Module, @unchecked Sendable {
    public let inputDim: Int
    public let modelDim: Int
    public let rnnDim: Int
    public let mlpDim: Int
    public let heads: Int
    public let convWidth: Int
    public let convRight: Int
    public let outputDim: Int
    public let attentionHeads: Int
    public let attentionWindow: Int

    public var wIn: MLXArray
    public var bIn: MLXArray
    public var layers: [MLXHawkLayer]
    public var attention: [MLXHawkAttentionLayer]
    public var finalNorm: MLXArray
    public var wOut: MLXArray
    public var bOut: MLXArray

    public init(
        inputDim: Int,
        modelDim: Int,
        rnnDim: Int,
        mlpDim: Int,
        heads: Int,
        convWidth: Int,
        convRight: Int,
        numLayers: Int,
        outputDim: Int,
        attentionAfter: [Int] = [],
        attentionHeads: Int = HawkShape.attentionHeads,
        attentionWindow: Int = HawkShape.attentionWindow
    ) {
        self.inputDim = inputDim
        self.attentionHeads = attentionAfter.isEmpty ? 0 : attentionHeads
        self.attentionWindow = attentionAfter.isEmpty ? 0 : attentionWindow
        self.modelDim = modelDim
        self.rnnDim = rnnDim
        self.mlpDim = mlpDim
        self.heads = heads
        self.convWidth = convWidth
        self.convRight = convRight
        self.outputDim = outputDim
        let inBound = sqrt(3.0 / Float(inputDim))
        self.wIn = MLXRandom.uniform(low: -inBound, high: inBound, [inputDim, modelDim])
        // Bias with RMS 0.1; see `HawkWeights.bIn`.
        let biasBound = 0.1 * sqrt(Float(3.0))
        self.bIn = MLXRandom.uniform(low: -biasBound, high: biasBound, [modelDim])
        var built: [MLXHawkLayer] = []
        var l = 0
        while l < numLayers {
            built.append(MLXHawkLayer(modelDim: modelDim, rnnDim: rnnDim, mlpDim: mlpDim, heads: heads, convWidth: convWidth, convRight: convRight, numLayers: numLayers))
            l += 1
        }
        self.layers = built
        let blockScale = 1.0 / sqrt(Float(2 * (numLayers + attentionAfter.count)))
        self.attention = attentionAfter.sorted().map { after in
            MLXHawkAttentionLayer(after: after, modelDim: modelDim, mlpDim: mlpDim, heads: attentionHeads, window: attentionWindow, residualScale: blockScale)
        }
        self.finalNorm = MLXArray.ones([modelDim])
        let outBound = sqrt(3.0 / Float(modelDim))
        self.wOut = MLXRandom.uniform(low: -outBound, high: outBound, [modelDim, outputDim])
        self.bOut = MLXArray.zeros([outputDim])
        super.init()
    }

    /// Network with the default shape (`HawkShape`) on the standard comment stream feature input.
    public convenience init(outputDim: Int = ReactionCatalog.numBaseClasses) {
        self.init(
            inputDim: CommentStreamFrontEnd.inputDim,
            modelDim: HawkShape.modelDim,
            rnnDim: HawkShape.rnnDim,
            mlpDim: HawkShape.mlpDim,
            heads: HawkShape.heads,
            convWidth: HawkShape.convWidth,
            convRight: HawkShape.convRight,
            numLayers: HawkShape.numLayers,
            outputDim: outputDim,
            attentionAfter: HawkShape.attentionAfter
        )
    }

    public convenience init(weights: HawkWeights) {
        self.init(
            inputDim: weights.inputDim,
            modelDim: weights.modelDim,
            rnnDim: weights.rnnDim,
            mlpDim: weights.mlpDim,
            heads: weights.heads,
            convWidth: weights.convWidth,
            convRight: weights.convRight,
            numLayers: weights.numLayers,
            outputDim: weights.outputDim,
            attentionAfter: weights.attention.map { $0.after },
            attentionHeads: max(1, weights.attentionHeads),
            attentionWindow: max(1, weights.attentionWindow)
        )
        importWeights(from: weights)
    }

    public var numLayers: Int {
        return layers.count
    }

    /// features: [B, T, inputDim] → logits [B, T, outputDim]. With `frameCounts` [B], frames past each sequence
    /// end are zeroed at the convolution input, matching CPU inference with lookahead.
    public func callAsFunction(_ features: MLXArray, frameCounts: MLXArray? = nil) -> MLXArray {
        let x = hidden(features, frameCounts: frameCounts, layerCount: layers.count)
        return matmul(hawkRMSNorm(x, finalNorm), wOut) + bOut
    }

    /// Residual stream [B, T, modelDim] after the first `layerCount` layers.
    public func hidden(_ features: MLXArray, frameCounts: MLXArray? = nil, layerCount: Int) -> MLXArray {
        var x = matmul(features, wIn) + bIn
        var frameMask: MLXArray? = nil
        if let counts = frameCounts, 0 < convRight {
            let positions = MLXArray(0..<features.shape[1]).reshaped([1, features.shape[1], 1])
            frameMask = (positions .< counts.reshaped([-1, 1, 1])).asType(.float32)
        }
        var l = 0
        while l < layerCount {
            x = layers[l](x, frameMask: frameMask)
            for block in attention where block.after == l {
                x = block(x)
            }
            l += 1
        }
        return x
    }

    /// Loads weights through `update(parameters:)` rather than reassigning properties, so the module tree
    /// (and anything compiled against it) keeps the same parameter slots.
    public func importWeights(from w: HawkWeights) {
        load(w, readout: true)
    }

    /// Loads everything except the readout, which keeps its current values.
    public func importEncoder(from w: HawkWeights) {
        load(w, readout: false)
    }

    private func load(_ w: HawkWeights, readout: Bool) {
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        var items: [(String, MLXArray)] = [
            ("wIn", MLXArray(w.wIn, [w.inputDim, d])),
            ("bIn", MLXArray(w.bIn, [d])),
            ("finalNorm", MLXArray(w.finalNorm, [d]))
        ]
        if readout {
            items.append(("wOut", MLXArray(w.wOut, [d, w.outputDim])))
            items.append(("bOut", MLXArray(w.bOut, [w.outputDim])))
        }
        var l = 0
        while l < min(layers.count, w.layers.count) {
            let lw = w.layers[l]
            let p = "layers.\(l)."
            items.append((p + "norm1", MLXArray(lw.norm1, [d])))
            items.append((p + "wGate", MLXArray(lw.wGate, [d, r])))
            items.append((p + "wX", MLXArray(lw.wX, [d, r])))
            items.append((p + "convW", MLXArray(lw.convW, [w.convTaps, r])))
            items.append((p + "convB", MLXArray(lw.convB, [r])))
            items.append((p + "wA", MLXArray(lw.wA, [w.heads, hd, hd])))
            items.append((p + "bA", MLXArray(lw.bA, [r])))
            items.append((p + "wI", MLXArray(lw.wI, [w.heads, hd, hd])))
            items.append((p + "bI", MLXArray(lw.bI, [r])))
            items.append((p + "lambda", MLXArray(lw.lambda, [r])))
            items.append((p + "wO", MLXArray(lw.wO, [r, d])))
            items.append((p + "norm2", MLXArray(lw.norm2, [d])))
            items.append((p + "wMLPGate", MLXArray(lw.wMLPGate, [d, m])))
            items.append((p + "wMLPUp", MLXArray(lw.wMLPUp, [d, m])))
            items.append((p + "wMLPDown", MLXArray(lw.wMLPDown, [m, d])))
            l += 1
        }
        var b = 0
        while b < min(attention.count, w.attention.count) {
            let aw = w.attention[b]
            let p = "attention.\(b)."
            let hd = w.attentionHeadDim
            items.append((p + "norm1", MLXArray(aw.norm1, [d])))
            items.append((p + "wQ", MLXArray(aw.wQ, [d, w.attentionHeads * hd])))
            items.append((p + "wK", MLXArray(aw.wK, [d, hd])))
            items.append((p + "wV", MLXArray(aw.wV, [d, hd])))
            items.append((p + "wO", MLXArray(aw.wO, [w.attentionHeads * hd, d])))
            items.append((p + "norm2", MLXArray(aw.norm2, [d])))
            items.append((p + "wMLPGate", MLXArray(aw.wMLPGate, [d, m])))
            items.append((p + "wMLPUp", MLXArray(aw.wMLPUp, [d, m])))
            items.append((p + "wMLPDown", MLXArray(aw.wMLPDown, [m, d])))
            b += 1
        }
        update(parameters: ModuleParameters.unflattened(items))
        eval(self)
    }

    public func exportWeights() -> HawkWeights {
        eval(self)
        func values(_ x: MLXArray) -> [Float] {
            return x.asArray(Float.self)
        }
        let exported = layers.map { layer in
            HawkLayerWeights(
                norm1: values(layer.norm1), wGate: values(layer.wGate), wX: values(layer.wX),
                convW: values(layer.convW), convB: values(layer.convB),
                wA: values(layer.wA), bA: values(layer.bA), wI: values(layer.wI), bI: values(layer.bI),
                lambda: values(layer.lambda), wO: values(layer.wO), norm2: values(layer.norm2),
                wMLPGate: values(layer.wMLPGate), wMLPUp: values(layer.wMLPUp), wMLPDown: values(layer.wMLPDown)
            )
        }
        return HawkWeights(
            inputDim: inputDim,
            modelDim: modelDim,
            rnnDim: rnnDim,
            mlpDim: mlpDim,
            heads: heads,
            convWidth: convWidth,
            convRight: convRight,
            outputDim: outputDim,
            wIn: values(wIn),
            bIn: values(bIn),
            layers: exported,
            finalNorm: values(finalNorm),
            wOut: values(wOut),
            bOut: values(bOut),
            attention: attention.map { a in
                HawkAttentionLayerWeights(
                    after: a.after, norm1: values(a.norm1), wQ: values(a.wQ), wK: values(a.wK), wV: values(a.wV),
                    wO: values(a.wO), norm2: values(a.norm2), wMLPGate: values(a.wMLPGate), wMLPUp: values(a.wMLPUp),
                    wMLPDown: values(a.wMLPDown)
                )
            },
            attentionHeads: attentionHeads,
            attentionWindow: attentionWindow
        )
    }
}
#endif

