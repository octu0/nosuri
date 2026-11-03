import Foundation

/// Deterministic SplitMix64 pseudorandom number generator for reproducible weight initialization and shuffling.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    public mutating func uniform(low: Float, high: Float) -> Float {
        let u = Float(next() >> 40) / Float(1 << 24)
        return low + (high - low) * u
    }
}

extension HawkWeights {
    /// Initializes a fresh `HawkWeights` model with scaled uniform weights matching `MLXHawkNetwork`.
    public static func makeRandom(
        inputDim: Int = CommentStreamFrontEnd.inputDim,
        modelDim: Int = 64,
        rnnDim: Int = 64,
        mlpDim: Int = 128,
        heads: Int = 4,
        convWidth: Int = 4,
        convRight: Int = 0,
        numLayers: Int = 2,
        outputDim: Int = ReactionCatalog.numBaseClasses,
        seed: UInt64 = 1
    ) -> HawkWeights {
        var rng = SplitMix64(seed: seed)
        func uniformArray(_ fanIn: Int, count: Int, scale: Float = 1.0) -> [Float] {
            let bound = scale * sqrtf(3.0 / Float(max(1, fanIn)))
            var out = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                out[i] = rng.uniform(low: -bound, high: bound)
            }
            return out
        }
        let residualScale = 1.0 / sqrtf(Float(max(1, 2 * numLayers)))
        let headDim = rnnDim / heads
        let convTaps = convWidth + convRight
        var layers: [HawkLayerWeights] = []
        for _ in 0..<numLayers {
            layers.append(HawkLayerWeights(
                norm1: [Float](repeating: 1.0, count: modelDim),
                wGate: uniformArray(modelDim, count: modelDim * rnnDim),
                wX: uniformArray(modelDim, count: modelDim * rnnDim),
                convW: uniformArray(convTaps, count: convTaps * rnnDim),
                convB: [Float](repeating: 0.0, count: rnnDim),
                wA: uniformArray(headDim, count: heads * headDim * headDim),
                bA: [Float](repeating: 0.0, count: rnnDim),
                wI: uniformArray(headDim, count: heads * headDim * headDim),
                bI: [Float](repeating: 0.0, count: rnnDim),
                lambda: HawkWeights.initialLambda(rnnDim: rnnDim),
                wO: uniformArray(rnnDim, count: rnnDim * modelDim, scale: residualScale),
                norm2: [Float](repeating: 1.0, count: modelDim),
                wMLPGate: uniformArray(modelDim, count: modelDim * mlpDim),
                wMLPUp: uniformArray(modelDim, count: modelDim * mlpDim),
                wMLPDown: uniformArray(mlpDim, count: mlpDim * modelDim, scale: residualScale)
            ))
        }
        let biasBound = 0.1 * sqrtf(3.0)
        var bIn = [Float](repeating: 0.0, count: modelDim)
        for i in 0..<modelDim {
            bIn[i] = rng.uniform(low: -biasBound, high: biasBound)
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
            wIn: uniformArray(inputDim, count: inputDim * modelDim),
            bIn: bIn,
            layers: layers,
            finalNorm: [Float](repeating: 1.0, count: modelDim),
            wOut: uniformArray(modelDim, count: modelDim * outputDim),
            bOut: [Float](repeating: 0.0, count: outputDim)
        )
    }
}

/// Cross-platform analytical BPTT trainer for `HawkWeights` (runs on both Linux and macOS CPU).
public final class HawkCPUTrainer: @unchecked Sendable {
    public private(set) var weights: HawkWeights
    public var learningRate: Float
    public private(set) var lastGradientNorm: Float = 0.0

    private let beta1: Float = 0.9
    private let beta2: Float = 0.999
    private let eps: Float = 1e-8
    private var adamSteps: Int = 0

    // Flattened Adam moments
    private var m: [Float]
    private var v: [Float]

    public init(weights: HawkWeights, learningRate: Float = 0.002) {
        self.weights = weights
        self.learningRate = learningRate
        let total = weights.parameterCount
        self.m = [Float](repeating: 0.0, count: total)
        self.v = [Float](repeating: 0.0, count: total)
    }

    public func updateLearningRate(_ lr: Float) {
        self.learningRate = lr
    }

    private struct LayerCache {
        var xIn: [Float]
        var rms1: [Float]
        var normed1: [Float]
        var gatePre: [Float]
        var branch: [Float]
        var zPad: [Float]
        var u: [Float]
        var keep: [Float]
        var admit: [Float]
        var logDecay: [Float]
        var a: [Float]
        var scale: [Float]
        var state: [Float]
        var gated: [Float]
        var xMid: [Float]
        var rms2: [Float]
        var normed2: [Float]
        var mlpGatePre: [Float]
        var mlpGateAct: [Float]
        var mlpUp: [Float]
        var mlpProd: [Float]
    }

    /// Runs one mini-batch of sequences through analytical BPTT with hard class indices and updates `weights` with Adam.
    public func trainBatchClassification(
        featuresBatch: [[[Float]]],
        targetsBatch: [[Int]],
        labelSmoothing: Float = 0.02
    ) -> Float {
        let outDim = max(1, weights.outputDim)
        let smooth = labelSmoothing / Float(outDim)
        let onTarget = (1.0 - labelSmoothing) + smooth
        let distributions: [[[Float]]] = targetsBatch.map { seq in
            seq.map { target in
                var dist = [Float](repeating: smooth, count: outDim)
                let clamped = max(0, min(outDim - 1, target))
                dist[clamped] = onTarget
                return dist
            }
        }
        return trainBatchSoftTargets(featuresBatch: featuresBatch, targetDistributionsBatch: distributions)
    }

    /// Runs one mini-batch of sequences through analytical BPTT against soft target distributions `[batch][frames][outputDim]` and updates `weights` with Adam.
    public func trainBatchSoftTargets(
        featuresBatch: [[[Float]]],
        targetDistributionsBatch: [[[Float]]]
    ) -> Float {
        let w = weights
        let inDim = w.inputDim
        let d = w.modelDim
        let r = w.rnnDim
        let mDim = w.mlpDim
        let heads = w.heads
        let hd = w.headDim
        let taps = w.convWidth
        let outDim = w.outputDim
        let numLayers = w.numLayers

        var gwIn = [Float](repeating: 0.0, count: w.wIn.count)
        var gbIn = [Float](repeating: 0.0, count: w.bIn.count)
        var gFinalNorm = [Float](repeating: 0.0, count: w.finalNorm.count)
        var gwOut = [Float](repeating: 0.0, count: w.wOut.count)
        var gbOut = [Float](repeating: 0.0, count: w.bOut.count)

        struct LayerGrad {
            var norm1: [Float]
            var wGate: [Float]
            var wX: [Float]
            var convW: [Float]
            var convB: [Float]
            var wA: [Float]
            var bA: [Float]
            var wI: [Float]
            var bI: [Float]
            var lambda: [Float]
            var wO: [Float]
            var norm2: [Float]
            var wMLPGate: [Float]
            var wMLPUp: [Float]
            var wMLPDown: [Float]
        }

        var gLayers: [LayerGrad] = w.layers.map { lw in
            LayerGrad(
                norm1: [Float](repeating: 0.0, count: lw.norm1.count),
                wGate: [Float](repeating: 0.0, count: lw.wGate.count),
                wX: [Float](repeating: 0.0, count: lw.wX.count),
                convW: [Float](repeating: 0.0, count: lw.convW.count),
                convB: [Float](repeating: 0.0, count: lw.convB.count),
                wA: [Float](repeating: 0.0, count: lw.wA.count),
                bA: [Float](repeating: 0.0, count: lw.bA.count),
                wI: [Float](repeating: 0.0, count: lw.wI.count),
                bI: [Float](repeating: 0.0, count: lw.bI.count),
                lambda: [Float](repeating: 0.0, count: lw.lambda.count),
                wO: [Float](repeating: 0.0, count: lw.wO.count),
                norm2: [Float](repeating: 0.0, count: lw.norm2.count),
                wMLPGate: [Float](repeating: 0.0, count: lw.wMLPGate.count),
                wMLPUp: [Float](repeating: 0.0, count: lw.wMLPUp.count),
                wMLPDown: [Float](repeating: 0.0, count: lw.wMLPDown.count)
            )
        }

        var totalLoss: Float = 0.0
        var totalFrames = 0
        for b in 0..<min(featuresBatch.count, targetDistributionsBatch.count) {
            totalFrames += min(featuresBatch[b].count, targetDistributionsBatch[b].count)
        }
        if totalFrames == 0 {
            return 0.0
        }
        let invTotalFrames = 1.0 / Float(totalFrames)

        for b in 0..<min(featuresBatch.count, targetDistributionsBatch.count) {
            let seq = featuresBatch[b]
            let targetDists = targetDistributionsBatch[b]
            let frames = min(seq.count, targetDists.count)
            if frames <= 0 {
                continue
            }

            var flatIn = [Float](repeating: 0.0, count: frames * inDim)
            for t in 0..<frames {
                let row = seq[t]
                let base = t * inDim
                for c in 0..<min(inDim, row.count) {
                    flatIn[base + c] = row[c]
                }
            }

            // 1. Input projection
            var x = [Float](repeating: 0.0, count: frames * d)
            for t in 0..<frames {
                for c in 0..<d {
                    x[t * d + c] = w.bIn[c]
                }
            }
            HawkNetwork.gemm(flatIn, w.wIn, &x, m: frames, n: d, k: inDim, beta: 1.0)

            // 2. Layers forward
            var caches: [LayerCache] = []
            caches.reserveCapacity(numLayers)

            for l in 0..<numLayers {
                let lw = w.layers[l]
                let xIn = x
                let (rms1, normed1) = Self.rmsNormForward(xIn, gain: lw.norm1, frames: frames, width: d)

                var gatePre = [Float](repeating: 0.0, count: frames * r)
                HawkNetwork.gemm(normed1, lw.wGate, &gatePre, m: frames, n: r, k: d)
                var branch = gatePre
                HawkNetwork.geluInPlace(&branch)

                var z = [Float](repeating: 0.0, count: frames * r)
                HawkNetwork.gemm(normed1, lw.wX, &z, m: frames, n: r, k: d)

                let padRows = taps - 1
                let zPad = [Float](repeating: 0.0, count: padRows * r) + z
                var u = [Float](repeating: 0.0, count: frames * r)
                HawkNetwork.conv(zPad, lw.convW, lw.convB, &u, frames: frames, width: r, taps: taps)

                var keepPre = [Float](repeating: 0.0, count: frames * r)
                var admitPre = [Float](repeating: 0.0, count: frames * r)
                for h in 0..<heads {
                    let offset = h * hd * hd
                    u.withUnsafeBufferPointer { ub in
                        lw.wA.withUnsafeBufferPointer { wa in
                            keepPre.withUnsafeMutableBufferPointer { out in
                                HawkNetwork.gemmRaw(ub.baseAddress! + h * hd, r, wa.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                            }
                        }
                        lw.wI.withUnsafeBufferPointer { wi in
                            admitPre.withUnsafeMutableBufferPointer { out in
                                HawkNetwork.gemmRaw(ub.baseAddress! + h * hd, r, wi.baseAddress! + offset, hd, out.baseAddress! + h * hd, r, m: frames, n: hd, k: hd, beta: 0.0)
                            }
                        }
                    }
                }

                var keep = [Float](repeating: 0.0, count: frames * r)
                var admit = [Float](repeating: 0.0, count: frames * r)
                let logDecay = lw.lambda.map { lam -> Float in
                    let ls = lam < 0.0 ? (lam - log1pf(expf(lam))) : -log1pf(expf(-lam))
                    return HawkWeights.decayExponent * ls
                }
                var a = [Float](repeating: 0.0, count: frames * r)
                var scale = [Float](repeating: 0.0, count: frames * r)
                var state = [Float](repeating: 0.0, count: frames * r)
                var gated = [Float](repeating: 0.0, count: frames * r)

                for c in 0..<r {
                    var prevH: Float = 0.0
                    let ld = logDecay[c]
                    let ba = lw.bA[c]
                    let bi = lw.bI[c]
                    for t in 0..<frames {
                        let idx = t * r + c
                        let kVal = 1.0 / (1.0 + expf(-(keepPre[idx] + ba)))
                        let iVal = 1.0 / (1.0 + expf(-(admitPre[idx] + bi)))
                        let aVal = expf(kVal * ld)
                        let scVal = sqrtf(max(1e-6, 1.0 - aVal * aVal + 1e-6))
                        let hVal = aVal * prevH + scVal * (iVal * u[idx])
                        keep[idx] = kVal
                        admit[idx] = iVal
                        a[idx] = aVal
                        scale[idx] = scVal
                        state[idx] = hVal
                        gated[idx] = branch[idx] * hVal
                        prevH = hVal
                    }
                }

                var xMid = xIn
                HawkNetwork.gemm(gated, lw.wO, &xMid, m: frames, n: d, k: r, beta: 1.0)

                let (rms2, normed2) = Self.rmsNormForward(xMid, gain: lw.norm2, frames: frames, width: d)
                var mlpGatePre = [Float](repeating: 0.0, count: frames * mDim)
                var mlpUp = [Float](repeating: 0.0, count: frames * mDim)
                HawkNetwork.gemm(normed2, lw.wMLPGate, &mlpGatePre, m: frames, n: mDim, k: d)
                HawkNetwork.gemm(normed2, lw.wMLPUp, &mlpUp, m: frames, n: mDim, k: d)
                var mlpGateAct = mlpGatePre
                HawkNetwork.geluInPlace(&mlpGateAct)
                var mlpProd = mlpGateAct
                HawkNetwork.multiplyInPlace(&mlpProd, mlpUp)

                var xOut = xMid
                HawkNetwork.gemm(mlpProd, lw.wMLPDown, &xOut, m: frames, n: d, k: mDim, beta: 1.0)
                x = xOut

                caches.append(LayerCache(
                    xIn: xIn, rms1: rms1, normed1: normed1, gatePre: gatePre, branch: branch,
                    zPad: zPad, u: u, keep: keep, admit: admit, logDecay: logDecay, a: a,
                    scale: scale, state: state, gated: gated, xMid: xMid, rms2: rms2,
                    normed2: normed2, mlpGatePre: mlpGatePre, mlpGateAct: mlpGateAct,
                    mlpUp: mlpUp, mlpProd: mlpProd
                ))
            }

            // 3. Final RMSNorm & Readout
            let (rmsFinal, normedFinal) = Self.rmsNormForward(x, gain: w.finalNorm, frames: frames, width: d)
            var logits = [Float](repeating: 0.0, count: frames * outDim)
            for t in 0..<frames {
                for c in 0..<outDim {
                    logits[t * outDim + c] = w.bOut[c]
                }
            }
            HawkNetwork.gemm(normedFinal, w.wOut, &logits, m: frames, n: outDim, k: d, beta: 1.0)

            // 4. Loss & dLogits
            var dLogits = [Float](repeating: 0.0, count: frames * outDim)
            for t in 0..<frames {
                let row = Array(logits[(t * outDim)..<((t + 1) * outDim)])
                let logP = HawkNetwork.logSoftmax(row)
                let dist = targetDists[t]
                var stepLoss: Float = 0.0
                for c in 0..<outDim {
                    let weight = c < dist.count ? dist[c] : 0.0
                    stepLoss -= weight * logP[c]
                    let prob = expf(logP[c])
                    dLogits[t * outDim + c] = (prob - weight) * invTotalFrames
                }
                totalLoss += stepLoss * invTotalFrames
                for c in 0..<outDim {
                    gbOut[c] += dLogits[t * outDim + c]
                }
            }

            // Backward through readout
            Self.gemmTransA(normedFinal, dLogits, &gwOut, m: d, n: outDim, k: frames)
            var dNormedFinal = [Float](repeating: 0.0, count: frames * d)
            Self.gemmTransB(dLogits, w.wOut, &dNormedFinal, m: frames, n: d, k: outDim)
            var dx = Self.rmsNormBackward(x, rms: rmsFinal, gain: w.finalNorm, dOut: dNormedFinal, dGain: &gFinalNorm, frames: frames, width: d)

            // Backward through layers in reverse order
            var l = numLayers - 1
            while 0 <= l {
                let lw = w.layers[l]
                let cache = caches[l]

                // MLP backward
                Self.gemmTransA(cache.mlpProd, dx, &gLayers[l].wMLPDown, m: mDim, n: d, k: frames)
                var dMlpProd = [Float](repeating: 0.0, count: frames * mDim)
                Self.gemmTransB(dx, lw.wMLPDown, &dMlpProd, m: frames, n: mDim, k: d)

                var dMlpGatePre = [Float](repeating: 0.0, count: frames * mDim)
                var dMlpUp = [Float](repeating: 0.0, count: frames * mDim)
                for i in 0..<(frames * mDim) {
                    let dp = dMlpProd[i]
                    dMlpUp[i] = dp * cache.mlpGateAct[i]
                    let zVal = cache.mlpGatePre[i]
                    let s = 1.0 / (1.0 + expf(-1.702 * zVal))
                    let geluGrad = s * (1.0 + 1.702 * zVal * (1.0 - s))
                    dMlpGatePre[i] = dp * cache.mlpUp[i] * geluGrad
                }

                Self.gemmTransA(cache.normed2, dMlpGatePre, &gLayers[l].wMLPGate, m: d, n: mDim, k: frames)
                Self.gemmTransA(cache.normed2, dMlpUp, &gLayers[l].wMLPUp, m: d, n: mDim, k: frames)

                var dNormed2 = [Float](repeating: 0.0, count: frames * d)
                Self.gemmTransB(dMlpGatePre, lw.wMLPGate, &dNormed2, m: frames, n: d, k: mDim, beta: 0.0)
                Self.gemmTransB(dMlpUp, lw.wMLPUp, &dNormed2, m: frames, n: d, k: mDim, beta: 1.0)

                let dxMidFromMLP = Self.rmsNormBackward(cache.xMid, rms: cache.rms2, gain: lw.norm2, dOut: dNormed2, dGain: &gLayers[l].norm2, frames: frames, width: d)
                var dxMid = dx
                for i in 0..<(frames * d) {
                    dxMid[i] += dxMidFromMLP[i]
                }

                // Recurrent output projection backward
                Self.gemmTransA(cache.gated, dxMid, &gLayers[l].wO, m: r, n: d, k: frames)
                var dGated = [Float](repeating: 0.0, count: frames * r)
                Self.gemmTransB(dxMid, lw.wO, &dGated, m: frames, n: r, k: d)

                var dBranch = [Float](repeating: 0.0, count: frames * r)
                var dState = [Float](repeating: 0.0, count: frames * r)
                for i in 0..<(frames * r) {
                    dBranch[i] = dGated[i] * cache.state[i]
                    dState[i] = dGated[i] * cache.branch[i]
                }

                // Reverse-time scan through RG-LRU
                var dKeepPre = [Float](repeating: 0.0, count: frames * r)
                var dAdmitPre = [Float](repeating: 0.0, count: frames * r)
                var du = [Float](repeating: 0.0, count: frames * r)

                for c in 0..<r {
                    var carry: Float = 0.0
                    let ld = cache.logDecay[c]
                    let lam = lw.lambda[c]
                    let dLogDecay_dLambda = HawkWeights.decayExponent * (1.0 / (1.0 + expf(lam)))
                    var t = frames - 1
                    while 0 <= t {
                        let idx = t * r + c
                        let dh = dState[idx] + carry
                        let prevH: Float = (0 < t) ? cache.state[(t - 1) * r + c] : 0.0
                        let aVal = cache.a[idx]
                        let scVal = cache.scale[idx]
                        let kVal = cache.keep[idx]
                        let iVal = cache.admit[idx]
                        let uVal = cache.u[idx]
                        let iu = iVal * uVal

                        // h = a * prevH + scale * iu, where scale = sqrt(1 - a^2 + eps)
                        // d(scale)/da = -a / scale
                        let da = dh * (prevH - (aVal / scVal) * iu)
                        let dLogA = da * aVal
                        let dKeep = dLogA * ld
                        let dkPre = dKeep * kVal * (1.0 - kVal)
                        dKeepPre[idx] = dkPre
                        gLayers[l].bA[c] += dkPre
                        gLayers[l].lambda[c] += dLogA * kVal * dLogDecay_dLambda

                        let dIU = dh * scVal
                        let dAdmit = dIU * uVal
                        let diPre = dAdmit * iVal * (1.0 - iVal)
                        dAdmitPre[idx] = diPre
                        gLayers[l].bI[c] += diPre

                        du[idx] = dIU * iVal
                        carry = dh * aVal
                        t -= 1
                    }
                }

                // Backward through block-diagonal gate projections wA, wI
                for h in 0..<heads {
                    let offset = h * hd * hd
                    for t in 0..<frames {
                        let base = t * r + h * hd
                        for q in 0..<hd {
                            let uVal = cache.u[base + q]
                            var duAcc: Float = 0.0
                            for p in 0..<hd {
                                let kp = dKeepPre[base + p]
                                let ap = dAdmitPre[base + p]
                                gLayers[l].wA[offset + q * hd + p] += uVal * kp
                                gLayers[l].wI[offset + q * hd + p] += uVal * ap
                                duAcc += kp * lw.wA[offset + q * hd + p] + ap * lw.wI[offset + q * hd + p]
                            }
                            du[base + q] += duAcc
                        }
                    }
                }

                // Backward through depthwise causal convolution
                var dz = [Float](repeating: 0.0, count: frames * r)
                let padRows = taps - 1
                for t in 0..<frames {
                    for c in 0..<r {
                        let duVal = du[t * r + c]
                        gLayers[l].convB[c] += duVal
                        for j in 0..<taps {
                            gLayers[l].convW[j * r + c] += duVal * cache.zPad[(t + j) * r + c]
                            let srcFrame = t + j - padRows
                            if 0 <= srcFrame && srcFrame < frames {
                                dz[srcFrame * r + c] += duVal * lw.convW[j * r + c]
                            }
                        }
                    }
                }

                // Backward through GeLU branch
                var dGatePre = [Float](repeating: 0.0, count: frames * r)
                for i in 0..<(frames * r) {
                    let zVal = cache.gatePre[i]
                    let s = 1.0 / (1.0 + expf(-1.702 * zVal))
                    let geluGrad = s * (1.0 + 1.702 * zVal * (1.0 - s))
                    dGatePre[i] = dBranch[i] * geluGrad
                }

                Self.gemmTransA(cache.normed1, dGatePre, &gLayers[l].wGate, m: d, n: r, k: frames)
                Self.gemmTransA(cache.normed1, dz, &gLayers[l].wX, m: d, n: r, k: frames)

                var dNormed1 = [Float](repeating: 0.0, count: frames * d)
                Self.gemmTransB(dGatePre, lw.wGate, &dNormed1, m: frames, n: d, k: r, beta: 0.0)
                Self.gemmTransB(dz, lw.wX, &dNormed1, m: frames, n: d, k: r, beta: 1.0)

                let dxInFromRec = Self.rmsNormBackward(cache.xIn, rms: cache.rms1, gain: lw.norm1, dOut: dNormed1, dGain: &gLayers[l].norm1, frames: frames, width: d)
                dx = dxMid
                for i in 0..<(frames * d) {
                    dx[i] += dxInFromRec[i]
                }
                l -= 1
            }

            // Backward through input projection
            for t in 0..<frames {
                for c in 0..<d {
                    gbIn[c] += dx[t * d + c]
                }
            }
            Self.gemmTransA(flatIn, dx, &gwIn, m: inDim, n: d, k: frames)
        }

        // Flatten all gradients, clip norm to 5.0, and apply Adam update
        var flatGrad: [Float] = []
        flatGrad.reserveCapacity(w.parameterCount)
        flatGrad.append(contentsOf: gwIn)
        flatGrad.append(contentsOf: gbIn)
        for lg in gLayers {
            flatGrad.append(contentsOf: lg.norm1)
            flatGrad.append(contentsOf: lg.wGate)
            flatGrad.append(contentsOf: lg.wX)
            flatGrad.append(contentsOf: lg.convW)
            flatGrad.append(contentsOf: lg.convB)
            flatGrad.append(contentsOf: lg.wA)
            flatGrad.append(contentsOf: lg.bA)
            flatGrad.append(contentsOf: lg.wI)
            flatGrad.append(contentsOf: lg.bI)
            flatGrad.append(contentsOf: lg.lambda)
            flatGrad.append(contentsOf: lg.wO)
            flatGrad.append(contentsOf: lg.norm2)
            flatGrad.append(contentsOf: lg.wMLPGate)
            flatGrad.append(contentsOf: lg.wMLPUp)
            flatGrad.append(contentsOf: lg.wMLPDown)
        }
        flatGrad.append(contentsOf: gFinalNorm)
        flatGrad.append(contentsOf: gwOut)
        flatGrad.append(contentsOf: gbOut)

        var normSq: Float = 0.0
        for g in flatGrad {
            normSq += g * g
        }
        let gradNorm = sqrtf(normSq)
        lastGradientNorm = gradNorm
        let clipScale: Float = (gradNorm > 5.0) ? (5.0 / gradNorm) : 1.0

        adamSteps += 1
        let tFloat = Float(adamSteps)
        let biasCorr1 = 1.0 - powf(beta1, tFloat)
        let biasCorr2 = 1.0 - powf(beta2, tFloat)
        let stepLR = learningRate * sqrtf(biasCorr2) / biasCorr1

        var offset = 0
        func updateSlice(_ param: [Float]) -> [Float] {
            var updated = param
            for i in 0..<param.count {
                let idx = offset + i
                let g = flatGrad[idx] * clipScale
                let mNew = beta1 * m[idx] + (1.0 - beta1) * g
                let vNew = beta2 * v[idx] + (1.0 - beta2) * (g * g)
                m[idx] = mNew
                v[idx] = vNew
                updated[i] -= stepLR * mNew / (sqrtf(vNew) + eps)
            }
            offset += param.count
            return updated
        }

        let newWIn = updateSlice(w.wIn)
        let newBIn = updateSlice(w.bIn)
        var newLayers: [HawkLayerWeights] = []
        newLayers.reserveCapacity(w.numLayers)
        for lw in w.layers {
            let n1 = updateSlice(lw.norm1)
            let wg = updateSlice(lw.wGate)
            let wx = updateSlice(lw.wX)
            let cw = updateSlice(lw.convW)
            let cb = updateSlice(lw.convB)
            let wa = updateSlice(lw.wA)
            let ba = updateSlice(lw.bA)
            let wi = updateSlice(lw.wI)
            let bi = updateSlice(lw.bI)
            let lam = updateSlice(lw.lambda)
            let wo = updateSlice(lw.wO)
            let n2 = updateSlice(lw.norm2)
            let mg = updateSlice(lw.wMLPGate)
            let mu = updateSlice(lw.wMLPUp)
            let md = updateSlice(lw.wMLPDown)
            newLayers.append(HawkLayerWeights(
                norm1: n1, wGate: wg, wX: wx, convW: cw, convB: cb,
                wA: wa, bA: ba, wI: wi, bI: bi, lambda: lam, wO: wo,
                norm2: n2, wMLPGate: mg, wMLPUp: mu, wMLPDown: md
            ))
        }
        let newFinalNorm = updateSlice(w.finalNorm)
        let newWOut = updateSlice(w.wOut)
        let newBOut = updateSlice(w.bOut)

        weights = HawkWeights(
            inputDim: w.inputDim, modelDim: w.modelDim, rnnDim: w.rnnDim, mlpDim: w.mlpDim,
            heads: w.heads, convWidth: w.convWidth, convRight: w.convRight, outputDim: w.outputDim,
            wIn: newWIn, bIn: newBIn, layers: newLayers, finalNorm: newFinalNorm,
            wOut: newWOut, bOut: newBOut
        )

        return totalLoss
    }

    // MARK: - Matrix & RMSNorm Backward Helpers

    private static func rmsNormForward(_ x: [Float], gain: [Float], frames: Int, width: Int) -> (rms: [Float], normed: [Float]) {
        var rms = [Float](repeating: 0.0, count: frames)
        var out = [Float](repeating: 0.0, count: frames * width)
        for f in 0..<frames {
            let base = f * width
            var sumSq: Float = 0.0
            for c in 0..<width {
                sumSq += x[base + c] * x[base + c]
            }
            let scale = 1.0 / sqrtf(sumSq / Float(width) + HawkWeights.rmsNormEpsilon)
            rms[f] = scale
            for c in 0..<width {
                out[base + c] = x[base + c] * scale * gain[c]
            }
        }
        return (rms, out)
    }

    private static func rmsNormBackward(
        _ x: [Float],
        rms: [Float],
        gain: [Float],
        dOut: [Float],
        dGain: inout [Float],
        frames: Int,
        width: Int
    ) -> [Float] {
        var dx = [Float](repeating: 0.0, count: frames * width)
        let invWidth = 1.0 / Float(width)
        for f in 0..<frames {
            let base = f * width
            let r = rms[f]
            var dot: Float = 0.0
            for c in 0..<width {
                let g = dOut[base + c]
                dGain[c] += g * x[base + c] * r
                dot += g * gain[c] * x[base + c]
            }
            let r3 = r * r * r * invWidth
            for c in 0..<width {
                dx[base + c] = r * dOut[base + c] * gain[c] - x[base + c] * r3 * dot
            }
        }
        return dx
    }

    /// c += a^T · b, where a is [k, m], b is [k, n], c is [m, n].
    private static func gemmTransA(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int) {
        let nSimd = n - (n % 8)
        a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
        c.withUnsafeMutableBufferPointer { cp in
            let aBase = ap.baseAddress!
            let bBase = bp.baseAddress!
            let cBase = cp.baseAddress!
            for p in 0..<k {
                let aRow = aBase + p * m
                let bRow = bBase + p * n
                for row in 0..<m {
                    let aVal = aRow[row]
                    if aVal == 0.0 { continue }
                    let aVec = SIMD8<Float>(repeating: aVal)
                    let cRow = cBase + row * n
                    var col = 0
                    while col < nSimd {
                        HawkNetwork.store(HawkNetwork.load(cRow + col) + aVec * HawkNetwork.load(bRow + col), cRow + col)
                        col += 8
                    }
                    while col < n {
                        cRow[col] += aVal * bRow[col]
                        col += 1
                    }
                }
            }
        }}}
    }

    /// c = beta * c + a · b^T, where a is [m, k], b is [n, k], c is [m, n].
    private static func gemmTransB(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
        c.withUnsafeMutableBufferPointer { cp in
            let aBase = ap.baseAddress!
            let bBase = bp.baseAddress!
            let cBase = cp.baseAddress!
            for row in 0..<m {
                let aRow = aBase + row * k
                let cRow = cBase + row * n
                for col in 0..<n {
                    let bRow = bBase + col * k
                    let dot = VectorOperations.dotProduct(a: aRow, b: bRow, count: k)
                    if beta == 0.0 {
                        cRow[col] = dot
                    } else {
                        cRow[col] = beta * cRow[col] + dot
                    }
                }
            }
        }}}
    }
}
