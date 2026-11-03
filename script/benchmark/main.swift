import Foundation
import Nosuri

print("==================================================")
print("=== nosuri Hawk Streaming Benchmark ===")
print("==================================================")

let inDim = CommentStreamFrontEnd.inputDim
let modelDim = HawkShape.modelDim
let rnnDim = HawkShape.rnnDim
let mlpDim = HawkShape.mlpDim
let heads = HawkShape.heads
let convWidth = HawkShape.convWidth
let outDim = ReactionStateSpec.numBaseClasses

func makeLayer() -> HawkLayerWeights {
    return HawkLayerWeights(
        norm1: [Float](repeating: 1.0, count: modelDim),
        wGate: [Float](repeating: 0.01, count: modelDim * rnnDim),
        wX: [Float](repeating: 0.01, count: modelDim * rnnDim),
        convW: [Float](repeating: 0.05, count: convWidth * rnnDim),
        convB: [Float](repeating: 0.0, count: rnnDim),
        wA: [Float](repeating: 0.01, count: heads * (rnnDim / heads) * (rnnDim / heads)),
        bA: [Float](repeating: 0.0, count: rnnDim),
        wI: [Float](repeating: 0.01, count: heads * (rnnDim / heads) * (rnnDim / heads)),
        bI: [Float](repeating: 0.0, count: rnnDim),
        lambda: HawkWeights.initialLambda(rnnDim: rnnDim),
        wO: [Float](repeating: 0.01, count: rnnDim * modelDim),
        norm2: [Float](repeating: 1.0, count: modelDim),
        wMLPGate: [Float](repeating: 0.01, count: modelDim * mlpDim),
        wMLPUp: [Float](repeating: 0.01, count: modelDim * mlpDim),
        wMLPDown: [Float](repeating: 0.01, count: mlpDim * modelDim)
    )
}

let weights = HawkWeights(
    inputDim: inDim,
    modelDim: modelDim,
    rnnDim: rnnDim,
    mlpDim: mlpDim,
    heads: heads,
    convWidth: convWidth,
    convRight: 0,
    outputDim: outDim,
    wIn: [Float](repeating: 0.01, count: inDim * modelDim),
    bIn: [Float](repeating: 0.01, count: modelDim),
    layers: (0..<HawkShape.numLayers).map { _ in makeLayer() },
    finalNorm: [Float](repeating: 1.0, count: modelDim),
    wOut: [Float](repeating: 0.01, count: modelDim * outDim),
    bOut: [Float](repeating: 0.0, count: outDim)
)

let engine = ReactionStreamEngine(network: HawkNetwork(weights: weights))
let totalEvents = 1000
var events: [CommentEvent] = []
events.reserveCapacity(totalEvents)
for i in 0..<totalEvents {
    let t = Double(i) * 0.1
    let text = (i % 3 == 0) ? "" : "草www なんでそうなるの！？神引ききたー！"
    let sc = (i % 50 == 0) ? 5000 : 0
    events.append(CommentEvent(timestamp: t, text: text, superChatJPY: sc))
}

let startTime = Date().timeIntervalSinceReferenceDate
var checksum = 0
for ev in events {
    let clip = engine.push(ev)
    checksum &+= clip.baseIndex
}
let elapsed = Date().timeIntervalSinceReferenceDate - startTime
let eventsPerSec = Double(totalEvents) / max(1e-9, elapsed)
let latencyMs = (elapsed * 1000.0) / Double(totalEvents)

print("Model parameters: \(weights.parameterCount)")
print("Events processed: \(totalEvents)")
print("Total time:       \(String(format: "%.4f", elapsed)) s")
print("Throughput:       \(String(format: "%.1f", eventsPerSec)) events/s")
print("Latency / event:  \(String(format: "%.3f", latencyMs)) ms")
print("Checksum:         \(checksum)")
print("==================================================")

