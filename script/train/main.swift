import Foundation
import Nosuri
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

print("==================================================")
print("=== nosuri train / eval ===")
print("==================================================")

enum Defaults {
    static let lrMax: Float = 0.003
    static let lrMin: Float = 0.0003
    static let warmupStepDivisor = 20
    static let minWarmupSteps = 5
    static let batchSize = 16
    static let seed: UInt64 = 42
}

var epochs = 30
var batchSize = Defaults.batchSize
var datasetPaths: [String] = []
var exportWeightsPath: String? = nil
var importWeightsPath: String? = nil
var stepGain: Float = 0.22

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("Usage: train [-d <dataset_or_dir> ...] [options]")
        print("Options:")
        print("  -d, --dataset <path>         Dataset file (*.json) or directory (default: .tmp)")
        print("  -b, --batch-size <Int>       Mini-batch size (default: 16)")
        print("  -e, --epochs <Int>           Number of epochs (default: 30; 0 = eval only)")
        print("  --step-gain <Float>          Emotion slope filter step gain (default: 0.22)")
        print("  --export-weights <path>      Output path for the trained weights (.safetensors)")
        print("  --import-weights <path>      Initial weights (.safetensors)")
        exit(0)
    case "-e", "--epochs":
        if (argIdx + 1) < args.count, let val = Int(args[argIdx + 1]) {
            epochs = max(0, val)
            argIdx += 1
        }
    case "-b", "--batch-size":
        if (argIdx + 1) < args.count, let val = Int(args[argIdx + 1]) {
            batchSize = max(1, val)
            argIdx += 1
        }
    case "--step-gain":
        if (argIdx + 1) < args.count, let val = Float(args[argIdx + 1]) {
            stepGain = max(0.05, min(1.0, val))
            argIdx += 1
        }
    case "-d", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPaths.append(args[argIdx + 1])
            argIdx += 1
        }
    case "--export-weights":
        if (argIdx + 1) < args.count {
            exportWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--import-weights":
        if (argIdx + 1) < args.count {
            importWeightsPath = args[argIdx + 1]
            argIdx += 1
        }
    default:
        if arg.hasPrefix("-") != true {
            datasetPaths.append(arg)
        }
    }
    argIdx += 1
}

if datasetPaths.isEmpty {
    datasetPaths = [".tmp"]
}

let datasets = CommentEvent.loadScenarioDatasets(from: datasetPaths)
if datasets.isEmpty {
    print("error: no dataset files (*.json) found in \(datasetPaths.joined(separator: ", "))")
    exit(1)
}

print("dataset files (\(datasets.count)):")
var totalScenarioCount = 0
var totalEventCount = 0
for ds in datasets {
    let evCount = ds.scenarios.reduce(0) { $0 + $1.data.count }
    print("  - \(ds.source) (\(ds.scenarios.count) scenarios, \(evCount) comments)")
    totalScenarioCount += ds.scenarios.count
    totalEventCount += evCount
}

let frontEnd = CommentStreamFrontEnd()
let slopeFilter = EmotionSlopeFilter(
    numClasses: ReactionCatalog.numBaseClasses,
    stepGain: stepGain
)

var sequences: [[[Float]]] = []
var targetDistributions: [[[Float]]] = []
var dominantTargets: [[Int]] = []

for ds in datasets {
    let built = slopeFilter.buildTrainingEpisodes(scenarios: ds.scenarios, frontEnd: frontEnd)
    for ep in built {
        sequences.append(ep.frames)
        targetDistributions.append(ep.distributions)
        dominantTargets.append(ep.dominants)
    }
}

print("loaded \(totalScenarioCount) scenarios (\(totalEventCount) comments) -> \(sequences.count) training episodes")
print("inputDim: \(CommentStreamFrontEnd.inputDim) / reaction classes: \(ReactionCatalog.numBaseClasses)")

var initialWeights = HawkWeights.makeRandom(
    inputDim: CommentStreamFrontEnd.inputDim,
    modelDim: 64,
    rnnDim: 64,
    mlpDim: 128,
    heads: 4,
    convWidth: 4,
    convRight: 0,
    numLayers: 2,
    outputDim: ReactionCatalog.numBaseClasses,
    seed: Defaults.seed
)

if let impPath = importWeightsPath {
    guard let loaded = try? HawkWeights.load(from: URL(fileURLWithPath: impPath)) else {
        print("error: cannot load weights from \(impPath)")
        exit(1)
    }
    initialWeights = loaded
    print("imported weights: \(impPath) (\(loaded.parameterCount) parameters)")
} else {
    print("initialized Hawk network (\(initialWeights.parameterCount) parameters)")
}

let trainer = HawkCPUTrainer(weights: initialWeights, learningRate: Defaults.lrMax)

if 0 < epochs {
    let batchesPerEpoch = max(1, (sequences.count + batchSize - 1) / batchSize)
    let totalSteps = max(1, epochs * batchesPerEpoch)
    let warmupSteps = min(max(Defaults.minWarmupSteps, totalSteps / Defaults.warmupStepDivisor), max(1, totalSteps / 2))
    let scheduler = CosineLRScheduler(lrMax: Defaults.lrMax, lrMin: Defaults.lrMin)

    var rng = SplitMix64(seed: Defaults.seed)
    var indices = Array(0..<sequences.count)

    var step = 0
    var ep = 1
    while ep <= epochs {
        if indices.count > 1 {
            var i = indices.count - 1
            while i > 0 {
                let j = Int(rng.next() % UInt64(i + 1))
                indices.swapAt(i, j)
                i -= 1
            }
        }

        var lossSum: Float = 0.0
        var count = 0
        var b0 = 0
        while b0 < indices.count {
            let b1 = min(indices.count, b0 + batchSize)
            let batchSeqs = indices[b0..<b1].map { sequences[$0] }
            let batchDists = indices[b0..<b1].map { targetDistributions[$0] }
            step += 1
            let lr = scheduler.learningRate(step: step, totalSteps: totalSteps, warmupSteps: warmupSteps)
            trainer.updateLearningRate(lr)
            let loss = trainer.trainBatchSoftTargets(
                featuresBatch: batchSeqs,
                targetDistributionsBatch: batchDists
            )
            lossSum += loss
            count += 1
            b0 = b1
        }
        let avgLoss = lossSum / Float(max(1, count))
        print("  Epoch [\(ep)/\(epochs)] - loss: \(String(format: "%.4f", avgLoss))")
        ep += 1
    }
}

let exported = trainer.weights
if let expPath = exportWeightsPath {
    let url = URL(fileURLWithPath: expPath)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    do {
        try exported.save(to: url)
        print("✓ weights written: \(expPath)")
    } catch {
        print("✕ failed to write weights: \(error)")
        exit(1)
    }
}

let cpu = HawkNetwork(weights: exported)
var correct = 0
var totalCount = 0
for seqIdx in 0..<sequences.count {
    let logProbs = cpu.logProbs(features: sequences[seqIdx])
    let targets = dominantTargets[seqIdx]
    for i in 0..<min(logProbs.count, targets.count) {
        let row = logProbs[i]
        var bestIdx = 0
        var bestVal = -Float.greatestFiniteMagnitude
        for c in 0..<row.count {
            if bestVal < row[c] {
                bestVal = row[c]
                bestIdx = c
            }
        }
        if bestIdx == targets[i] {
            correct += 1
        }
        totalCount += 1
    }
}
let acc = Float(correct) * 100.0 / Float(max(1, totalCount))
print("Evaluation accuracy (against slope filter targets): \(String(format: "%.2f", acc))% (\(correct)/\(totalCount))")
