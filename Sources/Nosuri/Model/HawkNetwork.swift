#if canImport(Accelerate)
import Accelerate
#endif
import Foundation

/// CPU inference for the Hawk sequence model. Processes feature frames chunk by chunk, carrying the
/// RG-LRU memory and buffered convolution frames across chunks.
/// Linear layers run as batched matrix products over the chunk; gates and the recurrence use SIMD8.
/// `modelDim`, `rnnDim` and `mlpDim` must be multiples of 8.
public final class HawkNetwork: @unchecked Sendable {
    public let weights: HawkWeights
    /// Per layer, c · log σ(Λ), so that log a_t = r_t · logDecay.
    public let logDecay: [[Float]]
    /// Rotary frequencies of the attention blocks, one per pair of a head.
    private let attentionFrequencies: [Double]

    /// Streaming state for a continuous session. With lookahead (`convRight`), each layer holds frames until
    /// their right context arrives, so output lags input by `convRight × numLayers` frames.
    public final class State {
        /// Per-layer RG-LRU memory, [rnnDim].
        public var memory: [[Float]]
        /// Per-layer convolution input buffer, [rows, rnnDim]. The first `convWidth - 1` rows are past
        /// frames; the remaining rows are frames not yet emitted.
        public var convBuffer: [[Float]]
        /// Per-layer residual stream for frames waiting on the convolution, [rows, modelDim].
        public var pending: [[Float]]
        /// Per attention block, rotated keys and values of the last `attentionWindow - 1` frames, [rows, headDim].
        public var attentionKeys: [[Float]]
        public var attentionValues: [[Float]]
        /// Per attention block, the absolute position of the next frame.
        public var attentionPosition: [Int]

        init(weights: HawkWeights) {
            memory = [[Float]](repeating: [Float](repeating: 0.0, count: weights.rnnDim), count: weights.numLayers)
            convBuffer = [[Float]](repeating: [Float](repeating: 0.0, count: (weights.convWidth - 1) * weights.rnnDim), count: weights.numLayers)
            pending = [[Float]](repeating: [], count: weights.numLayers)
            attentionKeys = [[Float]](repeating: [], count: weights.attention.count)
            attentionValues = [[Float]](repeating: [], count: weights.attention.count)
            attentionPosition = [Int](repeating: 0, count: weights.attention.count)
        }
    }

    public init(weights: HawkWeights) {
        precondition(weights.modelDim % 8 == 0 && weights.rnnDim % 8 == 0 && weights.mlpDim % 8 == 0, "widths must be multiples of 8")
        self.weights = weights
        let half = weights.attentionHeadDim / 2
        self.attentionFrequencies = (0..<half).map { pow(10000.0, -2.0 * Double($0) / Double(weights.attentionHeadDim)) }
        self.logDecay = weights.layers.map { layer in
            layer.lambda.map { lambda -> Float in
                // Numerically stable log σ(Λ).
                var logSigmoid = -log1pf(expf(-lambda))
                if lambda < 0.0 {
                    logSigmoid = lambda - log1pf(expf(lambda))
                }
                return HawkWeights.decayExponent * logSigmoid
            }
        }
    }

    public func makeState() -> State {
        return State(weights: weights)
    }

    /// Log-probabilities [T][outputDim] for a sequence of feature frames.
    public func logProbs(features: [[Float]], chunkFrames: Int = 128) -> [[Float]] {
        let state = makeState()
        var result: [[Float]] = []
        result.reserveCapacity(features.count)
        var t0 = 0
        while t0 < features.count {
            let t1 = min(features.count, t0 + max(1, chunkFrames))
            result += push(Array(features[t0..<t1]), state: state)
            t0 = t1
        }
        result += finish(state: state)
        return result
    }

    /// Feeds feature frames and returns log-probabilities for the frames whose right context is complete.
    /// With lookahead the output lags the input by `weights.lookaheadFrames`; `finish` drains the rest.
    public func push(_ features: [[Float]], state: State) -> [[Float]] {
        if features.isEmpty {
            return []
        }
        let inDim = weights.inputDim
        var chunk = [Float](repeating: 0.0, count: features.count * inDim)
        var t = 0
        while t < features.count {
            let row = features[t]
            let base = t * inDim
            var d = 0
            while d < min(inDim, row.count) {
                chunk[base + d] = row[d]
                d += 1
            }
            t += 1
        }
        return logSoftmaxRows(forward(chunk: chunk, frames: features.count, state: state))
    }

    /// Ends the sequence: pads every layer's right context with zeros and returns the remaining frames.
    public func finish(state: State) -> [[Float]] {
        let w = weights
        var x: [Float] = []
        var l = 0
        while l < w.numLayers {
            ingest(layer: l, input: x, state: state)
            state.convBuffer[l] += [Float](repeating: 0.0, count: w.convRight * w.rnnDim)
            x = produce(layer: l, state: state)
            x = attendAfter(layer: l, input: x, state: state)
            l += 1
        }
        return logSoftmaxRows(readout(x))
    }

    private func forward(chunk: [Float], frames: Int, state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        var x = [Float](repeating: 0.0, count: frames * d)
        var f = 0
        while f < frames {
            var c = 0
            while c < d {
                x[f * d + c] = w.bIn[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(chunk, w.wIn, &x, m: frames, n: d, k: w.inputDim, beta: 1.0)
        var l = 0
        while l < w.numLayers {
            ingest(layer: l, input: x, state: state)
            x = produce(layer: l, state: state)
            x = attendAfter(layer: l, input: x, state: state)
            l += 1
        }
        return readout(x)
    }

    /// Runs the attention blocks that follow recurrent layer `l` on its output frames.
    private func attendAfter(layer l: Int, input: [Float], state: State) -> [Float] {
        var x = input
        var b = 0
        while b < weights.attention.count {
            if weights.attention[b].after == l {
                x = attend(block: b, input: x, state: state)
            }
            b += 1
        }
        return x
    }

    /// Causal multi-query attention over the last `attentionWindow` frames, then the gated MLP. [rows, modelDim].
    private func attend(block b: Int, input: [Float], state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        let m = w.mlpDim
        let heads = w.attentionHeads
        let hd = w.attentionHeadDim
        let half = hd / 2
        let rows = input.count / d
        if rows <= 0 {
            return input
        }
        let aw = w.attention[b]
        var x = input
        var normed = [Float](repeating: 0.0, count: rows * d)
        Self.rmsNorm(x, aw.norm1, &normed, frames: rows, width: d)
        var q = [Float](repeating: 0.0, count: rows * heads * hd)
        var k = [Float](repeating: 0.0, count: rows * hd)
        var v = [Float](repeating: 0.0, count: rows * hd)
        Self.gemm(normed, aw.wQ, &q, m: rows, n: heads * hd, k: d)
        Self.gemm(normed, aw.wK, &k, m: rows, n: hd, k: d)
        Self.gemm(normed, aw.wV, &v, m: rows, n: hd, k: d)
        let start = state.attentionPosition[b]
        // Rotary positions; angles in Double so hours-long streams keep their relative phases.
        var cosines = [Float](repeating: 0.0, count: half)
        var sines = [Float](repeating: 0.0, count: half)
        func rotate(_ buf: inout [Float], base: Int) {
            var i = 0
            while i < half {
                let x1 = buf[base + i]
                let x2 = buf[base + half + i]
                buf[base + i] = x1 * cosines[i] - x2 * sines[i]
                buf[base + half + i] = x1 * sines[i] + x2 * cosines[i]
                i += 1
            }
        }
        var t = 0
        while t < rows {
            var i = 0
            while i < half {
                let angle = Double(start + t) * attentionFrequencies[i]
                cosines[i] = Float(cos(angle))
                sines[i] = Float(sin(angle))
                i += 1
            }
            var h = 0
            while h < heads {
                rotate(&q, base: (t * heads + h) * hd)
                h += 1
            }
            rotate(&k, base: t * hd)
            t += 1
        }
        let cachedRows = state.attentionKeys[b].count / hd
        let keys = state.attentionKeys[b] + k
        let values = state.attentionValues[b] + v
        let cacheStart = start - cachedRows
        let scale = 1.0 / sqrtf(Float(hd))
        var out = [Float](repeating: 0.0, count: rows * heads * hd)
        var scores = [Float](repeating: 0.0, count: heads * w.attentionWindow)
        q.withUnsafeBufferPointer { qp in
        keys.withUnsafeBufferPointer { kp in
        values.withUnsafeBufferPointer { vp in
        out.withUnsafeMutableBufferPointer { op in
        scores.withUnsafeMutableBufferPointer { sp in
            var t = 0
            while t < rows {
                let position = start + t
                let first = max(0, max(cacheStart, position - w.attentionWindow + 1))
                let count = position - first + 1
                let window = first - cacheStart
                let qBase = qp.baseAddress! + t * heads * hd
                let kBase = kp.baseAddress! + window * hd
                let vBase = vp.baseAddress! + window * hd
                let oBase = op.baseAddress! + t * heads * hd
                let sBase = sp.baseAddress!
                var h = 0
                while h < heads {
                    let qRow = qBase + h * hd
                    let sRow = sBase + h * count
                    var peak: Float = -Float.greatestFiniteMagnitude
                    var c = 0
                    while c < count {
                        let kRow = kBase + c * hd
                        let dot = VectorOperations.dotProduct(a: qRow, b: kRow, count: hd) * scale
                        sRow[c] = dot
                        if peak < dot {
                            peak = dot
                        }
                        c += 1
                    }
                    var total: Float = 0.0
                    c = 0
                    while c < count {
                        let e = expf(sRow[c] - peak)
                        sRow[c] = e
                        total += e
                        c += 1
                    }
                    let inv = 1.0 / total
                    let oRow = oBase + h * hd
                    for dIdx in 0..<hd {
                        oRow[dIdx] = 0.0
                    }
                    c = 0
                    while c < count {
                        let p = sRow[c] * inv
                        sRow[c] = p
                        let vRow = vBase + c * hd
                        var dIdx = 0
                        while dIdx < hd {
                            oRow[dIdx] += p * vRow[dIdx]
                            dIdx += 1
                        }
                        c += 1
                    }
                    h += 1
                }
                t += 1
            }
        }}}}}
        let keep = min(w.attentionWindow - 1, cachedRows + rows)
        state.attentionKeys[b] = Array(keys[((cachedRows + rows - keep) * hd)...])
        state.attentionValues[b] = Array(values[((cachedRows + rows - keep) * hd)...])
        state.attentionPosition[b] = start + rows
        Self.gemm(out, aw.wO, &x, m: rows, n: d, k: heads * hd, beta: 1.0)

        var mlpGate = [Float](repeating: 0.0, count: rows * m)
        var mlpUp = [Float](repeating: 0.0, count: rows * m)
        Self.rmsNorm(x, aw.norm2, &normed, frames: rows, width: d)
        Self.gemm(normed, aw.wMLPGate, &mlpGate, m: rows, n: m, k: d)
        Self.gemm(normed, aw.wMLPUp, &mlpUp, m: rows, n: m, k: d)
        Self.geluInPlace(&mlpGate)
        Self.multiplyInPlace(&mlpGate, mlpUp)
        Self.gemm(mlpGate, aw.wMLPDown, &x, m: rows, n: d, k: m, beta: 1.0)
        return x
    }

    private func logSoftmaxRows(_ logits: [Float]) -> [[Float]] {
        let outDim = weights.outputDim
        var rows: [[Float]] = []
        rows.reserveCapacity(logits.count / outDim)
        var f = 0
        while f < logits.count / outDim {
            rows.append(Self.logSoftmax(Array(logits[(f * outDim)..<((f + 1) * outDim)])))
            f += 1
        }
        return rows
    }

    /// Feeds frames [rows, modelDim] into a layer and buffers the convolution input (RMSNorm → linear).
    private func ingest(layer l: Int, input: [Float], state: State) {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let rows = input.count / d
        if rows <= 0 {
            return
        }
        let lw = w.layers[l]
        var normed = [Float](repeating: 0.0, count: rows * d)
        Self.rmsNorm(input, lw.norm1, &normed, frames: rows, width: d)
        var z = [Float](repeating: 0.0, count: rows * r)
        Self.gemm(normed, lw.wX, &z, m: rows, n: r, k: d)
        state.convBuffer[l] += z
        state.pending[l] += input
    }

    /// Runs the rest of the layer on frames whose right context is complete; returns [rows, modelDim].
    private func produce(layer l: Int, state: State) -> [Float] {
        let w = weights
        let d = w.modelDim
        let r = w.rnnDim
        let m = w.mlpDim
        let hd = w.headDim
        let taps = w.convTaps
        let buffered = state.convBuffer[l].count / r
        let frames = buffered - (taps - 1)
        if frames <= 0 {
            return []
        }
        let lw = w.layers[l]
        var x = Array(state.pending[l][0..<(frames * d)])
        state.pending[l].removeFirst(frames * d)

        var normed = [Float](repeating: 0.0, count: frames * d)
        var branch = [Float](repeating: 0.0, count: frames * r)
        var u = [Float](repeating: 0.0, count: frames * r)
        var keepPre = [Float](repeating: 0.0, count: frames * r)
        var admitPre = [Float](repeating: 0.0, count: frames * r)
        var mlpGate = [Float](repeating: 0.0, count: frames * m)
        var mlpUp = [Float](repeating: 0.0, count: frames * m)

        Self.rmsNorm(x, lw.norm1, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wGate, &branch, m: frames, n: r, k: d)
        Self.geluInPlace(&branch)
        Self.conv(state.convBuffer[l], lw.convW, lw.convB, &u, frames: frames, width: r, taps: taps)
        state.convBuffer[l].removeFirst(frames * r)
        // Block-diagonal gate projections.
        var h = 0
        while h < w.heads {
            let offset = h * hd * hd
            u.withUnsafeBufferPointer { ub in
                lw.wA.withUnsafeBufferPointer { wa in
                    keepPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
                lw.wI.withUnsafeBufferPointer { wi in
                    admitPre.withUnsafeMutableBufferPointer { out in
                        Self.gemmRaw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                    }
                }
            }
            h += 1
        }
        scan(layer: l, frames: frames, u: u, keepPre: keepPre, admitPre: admitPre, branch: &branch, memory: &state.memory[l])
        Self.gemm(branch, lw.wO, &x, m: frames, n: d, k: r, beta: 1.0)

        Self.rmsNorm(x, lw.norm2, &normed, frames: frames, width: d)
        Self.gemm(normed, lw.wMLPGate, &mlpGate, m: frames, n: m, k: d)
        Self.gemm(normed, lw.wMLPUp, &mlpUp, m: frames, n: m, k: d)
        Self.geluInPlace(&mlpGate)
        Self.multiplyInPlace(&mlpGate, mlpUp)
        Self.gemm(mlpGate, lw.wMLPDown, &x, m: frames, n: d, k: m, beta: 1.0)
        return x
    }

    /// Final RMSNorm and output projection: [rows, modelDim] → logits [rows, outputDim].
    private func readout(_ x: [Float]) -> [Float] {
        let w = weights
        let d = w.modelDim
        let frames = x.count / d
        if frames <= 0 {
            return []
        }
        var normed = [Float](repeating: 0.0, count: frames * d)
        Self.rmsNorm(x, w.finalNorm, &normed, frames: frames, width: d)
        var logits = [Float](repeating: 0.0, count: frames * w.outputDim)
        var f = 0
        while f < frames {
            var c = 0
            while c < w.outputDim {
                logits[f * w.outputDim + c] = w.bOut[c]
                c += 1
            }
            f += 1
        }
        Self.gemm(normed, w.wOut, &logits, m: frames, n: w.outputDim, k: d, beta: 1.0)
        return logits
    }

    /// RG-LRU: a = exp(r · c · log σ(Λ)), h = a h + √(1 - a²) (i u). Multiplies h into the GeLU branch in place.
    private func scan(layer: Int, frames: Int, u: [Float], keepPre: [Float], admitPre: [Float], branch: inout [Float], memory: inout [Float]) {
        let lw = weights.layers[layer]
        let r = weights.rnnDim
        let one = SIMD8<Float>(repeating: 1.0)
        let eps = SIMD8<Float>(repeating: 1e-6)
        logDecay[layer].withUnsafeBufferPointer { decay in
        lw.bA.withUnsafeBufferPointer { bA in
        lw.bI.withUnsafeBufferPointer { bI in
        u.withUnsafeBufferPointer { ub in
        keepPre.withUnsafeBufferPointer { kp in
        admitPre.withUnsafeBufferPointer { ap in
        branch.withUnsafeMutableBufferPointer { out in
        memory.withUnsafeMutableBufferPointer { mem in
            let dp = decay.baseAddress!
            let bap = bA.baseAddress!
            let bip = bI.baseAddress!
            let up = ub.baseAddress!
            let kpp = kp.baseAddress!
            let app = ap.baseAddress!
            let op = out.baseAddress!
            let mp = mem.baseAddress!
            var c = 0
            while c < r {
                let logA = Self.load(dp + c)
                let keepBias = Self.load(bap + c)
                let admitBias = Self.load(bip + c)
                var state = Self.load(mp + c)
                var t = 0
                while t < frames {
                    let idx = t * r + c
                    let keep = SIMDMath.sigmoid(Self.load(kpp + idx) + keepBias)
                    let admit = SIMDMath.sigmoid(Self.load(app + idx) + admitBias)
                    let a = SIMDMath.exp(keep * logA)
                    let scale = (one - a * a + eps).squareRoot()
                    state = a * state + scale * (admit * Self.load(up + idx))
                    Self.store(state * Self.load(op + idx), op + idx)
                    t += 1
                }
                Self.store(state, mp + c)
                c += 8
            }
        }}}}}}}}
    }

    // MARK: - Helpers

    /// c = a · b (+ beta · c), row-major: a [m, k], b [k, n], c [m, n].
    public static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    gemmRaw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    public static func gemmRaw(_ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int,
                               _ c: UnsafeMutablePointer<Float>, _ ldc: Int, m: Int, n: Int, k: Int, beta: Float) {
        #if canImport(Accelerate)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    1.0, a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc))
        #else
        let nSimd = n - (n % 8)
        var row = 0
        while row < m {
            let cRow = c + row * ldc
            let aRow = a + row * lda
            if beta == 0.0 {
                var col = 0
                while col < nSimd {
                    store(SIMD8<Float>(repeating: 0.0), cRow + col)
                    col += 8
                }
                while col < n {
                    cRow[col] = 0.0
                    col += 1
                }
            } else if beta != 1.0 {
                let bVec = SIMD8<Float>(repeating: beta)
                var col = 0
                while col < nSimd {
                    store(load(cRow + col) * bVec, cRow + col)
                    col += 8
                }
                while col < n {
                    cRow[col] *= beta
                    col += 1
                }
            }
            var p = 0
            while p < k {
                let aVal = aRow[p]
                if aVal != 0.0 {
                    let aVec = SIMD8<Float>(repeating: aVal)
                    let bRow = b + p * ldb
                    var col = 0
                    while col < nSimd {
                        store(load(cRow + col) + aVec * load(bRow + col), cRow + col)
                        col += 8
                    }
                    while col < n {
                        cRow[col] += aVal * bRow[col]
                        col += 1
                    }
                }
                p += 1
            }
            row += 1
        }
        #endif
    }

    public static func rmsNorm(_ x: [Float], _ gain: [Float], _ out: inout [Float], frames: Int, width: Int) {
        var f = 0
        while f < frames {
            let base = f * width
            var sumSq: Float = 0.0
            var c = 0
            while c < width {
                sumSq += x[base + c] * x[base + c]
                c += 1
            }
            let scale = 1.0 / sqrtf(sumSq / Float(width) + HawkWeights.rmsNormEpsilon)
            c = 0
            while c < width {
                out[base + c] = x[base + c] * scale * gain[c]
                c += 1
            }
            f += 1
        }
    }

    /// Sigmoid approximation of GeLU: x σ(1.702 x).
    public static func geluInPlace(_ x: inout [Float]) {
        let k = SIMD8<Float>(repeating: 1.702)
        x.withUnsafeMutableBufferPointer { p in
            let base = p.baseAddress!
            var i = 0
            while i + 8 <= p.count {
                let v = load(base + i)
                store(v * SIMDMath.sigmoid(k * v), base + i)
                i += 8
            }
            while i < p.count {
                p[i] = p[i] / (1.0 + expf(-1.702 * p[i]))
                i += 1
            }
        }
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    public static func multiplyInPlace(_ x: inout [Float], _ y: [Float]) {
        var i = 0
        while i < x.count {
            x[i] *= y[i]
            i += 1
        }
    }

    /// Depthwise temporal convolution. `buffer` holds at least [frames + taps - 1, width];
    /// output frame t reads rows t ..< t + taps.
    public static func conv(_ buffer: [Float], _ convW: [Float], _ convB: [Float], _ out: inout [Float], frames: Int, width: Int, taps: Int) {
        buffer.withUnsafeBufferPointer { bp in
        convW.withUnsafeBufferPointer { wp in
        convB.withUnsafeBufferPointer { cb in
        out.withUnsafeMutableBufferPointer { op in
            let b = bp.baseAddress!
            let w = wp.baseAddress!
            let bias = cb.baseAddress!
            let o = op.baseAddress!
            var t = 0
            while t < frames {
                var c = 0
                while c < width {
                    var acc = load(bias + c)
                    var j = 0
                    while j < taps {
                        acc += load(w + j * width + c) * load(b + (t + j) * width + c)
                        j += 1
                    }
                    store(acc, o + t * width + c)
                    c += 8
                }
                t += 1
            }
        }}}}
    }

    public static func logSoftmax(_ logits: [Float]) -> [Float] {
        var peak = -Float.greatestFiniteMagnitude
        for v in logits {
            if peak < v {
                peak = v
            }
        }
        var total: Float = 0.0
        for v in logits {
            total += expf(v - peak)
        }
        let norm = peak + logf(total)
        return logits.map { $0 - norm }
    }
}
