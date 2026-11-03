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

guard let events = CommentEvent.loadList(fromFile: datasetPath), events.isEmpty != true else {
    print("error: cannot load dataset from \(datasetPath)")
    exit(1)
}

guard let weights = try? HawkWeights.load(from: URL(fileURLWithPath: weightsPath)) else {
    print("error: cannot load weights from \(weightsPath)")
    exit(1)
}

print("==================================================")
print("=== nosuri live reaction stream ===")
print("==================================================")
print("weights: \(weightsPath) (\(weights.parameterCount) parameters)")
print("dataset: \(datasetPath) (\(events.count) events)")

let catalog = ReactionCatalog(directory: resourcesDir)
let engine = ReactionStreamEngine(network: HawkNetwork(weights: weights), catalog: catalog)

for ev in events {
    let pred = engine.pushDetailed(ev)
    let clip = pred.clip
    let userPrefix = ev.user.isEmpty ? "" : "@\(ev.user): "
    let scText = ev.superChatJPY > 0 ? " [¥\(ev.superChatJPY)]" : ""
    let phaseText = pred.phase == "decaying" ? " [decay \(Int(pred.intensity * 100))%]" : (pred.escalated ? " (\(pred.baseClip.id)->\(clip.id))" : "")
    let commentDisplay = ev.text.isEmpty ? "(idle)" : "\(userPrefix)\(ev.text)"
    print(String(format: "[t=%5.2fs | u=%2d | v=%4.2f/s] %@%@ -> %@ (%@)%@",
                 ev.timestamp, pred.uniqueUsers, pred.velocity, commentDisplay, scText,
                 clip.id, clip.filename, phaseText))
}
