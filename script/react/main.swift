import Foundation
import Nosuri

var weightsPath = ".tmp/weights.safetensors"
var datasetPath = ""
var resourcesDir = ReactionCatalog.defaultResourcesDirectory

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("Usage: react -d <dataset> [-w <weights.safetensors>] [--resources <dir>]")
        exit(0)
    case "-w", "--weights", "--import-weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "-d", "--dataset":
        if (argIdx + 1) < args.count {
            datasetPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--resources":
        if (argIdx + 1) < args.count {
            resourcesDir = args[argIdx + 1]
            argIdx += 1
        }
    default:
        if arg.hasPrefix("-") != true {
            datasetPath = arg
        }
    }
    argIdx += 1
}

if datasetPath.isEmpty {
    print("Usage: react -d <dataset> [-w <weights.safetensors>] [--resources <dir>]")
    exit(1)
}

guard let scenarios = CommentEvent.loadScenarios(fromFile: datasetPath), scenarios.isEmpty != true else {
    print("error: cannot load scenarios from \(datasetPath)")
    exit(1)
}

guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: weightsPath)) else {
    print("error: cannot load weights from \(weightsPath)")
    exit(1)
}

print("==================================================")
print("=== nosuri live reaction stream ===")
print("==================================================")
print("weights:   \(weightsPath) (\(weights.parameterCount) parameters)")
print("scenarios: \(datasetPath) (\(scenarios.count) scenarios)")

let catalog = ReactionCatalog(directory: resourcesDir)
let engine = ReactionStreamEngine(network: HawkNetwork(weights: weights), catalog: catalog)

var prevScenario: ReactionScenario? = nil
for scenario in scenarios {
    let continuesChain = if let prev = prevScenario {
        scenario.fromState.baseIndex != 0 && prev.toState == scenario.fromState
    } else {
        false
    }
    if !continuesChain {
        engine.reset()
    }
    prevScenario = scenario

    let noteSuffix = scenario.note.isEmpty ? "" : " — \(scenario.note)"
    print("\n--- [\(scenario.reactionState)]\(noteSuffix) ---")

    var lastTs: Double = 0.0
    for ev in scenario.data {
        lastTs = ev.timestamp
        let pred = engine.ingest(ev)
        let clip = pred.clip
        let userPrefix = ev.user.isEmpty ? "" : "@\(ev.user): "
        let scText = ev.superChatJPY > 0 ? " [¥\(ev.superChatJPY)]" : ""
        let phaseText = pred.phase == "decaying" ? " [decay \(Int(pred.intensity * 100))%]" : (pred.escalated ? " (\(pred.baseClip.id)->\(clip.id))" : "")
        print(String(format: "[t=%5.2fs | u=%2d | v=%4.2f/s] %@%@%@ -> %@ (%@)%@",
                     ev.timestamp, pred.uniqueUsers, pred.velocity, userPrefix, ev.text, scText,
                     clip.id, clip.filename, phaseText))
    }

    // Query time-based decay at +3s, +6s, +9s after the last comment without feeding any dummy events.
    if scenario.toState.baseIndex > 0 {
        for offset in [3.0, 6.0, 9.0, 12.0] {
            let queryTs = lastTs + offset
            let pred = engine.state(at: queryTs)
            let clip = pred.clip
            let phaseText = pred.phase == "decaying" ? " [decay \(Int(pred.intensity * 100))%]" : ""
            print(String(format: "[t=%5.2fs | u=%2d | v=%4.2f/s] (state@+%.0fs) -> %@ (%@)%@",
                         queryTs, pred.uniqueUsers, pred.velocity, offset,
                         clip.id, clip.filename, phaseText))
            if pred.clip.baseIndex == 0 { break }
        }
    }
}
