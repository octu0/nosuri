import Foundation

public struct CommentEvent: Codable, Sendable, Equatable {
    public let timestamp: Double
    public let user: String
    public let text: String
    public let superChatJPY: Int
    public let reactionIndex: Int?

    public init(
        timestamp: Double,
        user: String = "",
        text: String,
        superChatJPY: Int = 0,
        reactionIndex: Int? = nil
    ) {
        self.timestamp = timestamp
        self.user = user
        self.text = text
        self.superChatJPY = superChatJPY
        self.reactionIndex = reactionIndex
    }

    /// Resolves one or more file or directory paths into individual dataset file paths,
    /// preferring `*_comments.jsonl` files when scanning a directory.
    public static func resolveDatasetFiles(from paths: [String]) -> [String] {
        var resolved: [String] = []
        let fm = FileManager.default
        for rawPath in paths {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: rawPath, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let entries = (try? fm.contentsOfDirectory(atPath: rawPath))?.sorted() ?? []
                let commentJsonls = entries.filter { $0.hasSuffix("_comments.jsonl") }
                let selected: [String] = if commentJsonls.isEmpty != true {
                    commentJsonls
                } else {
                    entries.filter {
                        let lower = $0.lowercased()
                        return lower.hasSuffix(".jsonl") || lower.hasSuffix(".json") || lower.hasSuffix(".tsv") || lower.hasSuffix(".csv")
                    }
                }
                for name in selected {
                    resolved.append((rawPath as NSString).appendingPathComponent(name))
                }
            } else {
                resolved.append(rawPath)
            }
        }
        return resolved
    }

    /// Loads multiple dataset files (or directories containing `*_comments.jsonl` files) separately.
    public static func loadDatasets(from paths: [String]) -> [(source: String, events: [CommentEvent])] {
        let files = resolveDatasetFiles(from: paths)
        var results: [(source: String, events: [CommentEvent])] = []
        for file in files {
            if let events = loadList(fromFile: file), events.isEmpty != true {
                results.append((source: file, events: events))
            }
        }
        return results
    }

    /// Loads comment events from JSONL, JSON array, TSV, CSV, line-delimited text, or a directory of `*_comments.jsonl` files.
    public static func loadList(fromFile path: String) -> [CommentEvent]? {
        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            let files = resolveDatasetFiles(from: [path])
            if files.isEmpty { return nil }
            var combined: [CommentEvent] = []
            for file in files {
                if let list = loadList(fromFile: file) {
                    combined.append(contentsOf: list)
                }
            }
            return combined
        }
        guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
            return nil
        }
        let trimmedAll = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedAll.isEmpty {
            return []
        }

        if trimmedAll.hasPrefix("[") {
            if let data = trimmedAll.data(using: .utf8),
               let rawArray = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                var events: [CommentEvent] = []
                var autoTime = 0.0
                for dict in rawArray {
                    if let ev = parseDictionary(dict, defaultTimestamp: autoTime) {
                        autoTime = ev.timestamp + 0.5
                        events.append(ev)
                    }
                }
                return events
            }
        }

        let lines = content.components(separatedBy: .newlines)
        var events: [CommentEvent] = []
        var autoTime = 0.0
        let ext = (path as NSString).pathExtension.lowercased()

        for (lineIdx, rawLine) in lines.enumerated() {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }

            if line.hasPrefix("{") {
                guard let data = line.data(using: .utf8),
                      let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let ev = parseDictionary(dict, defaultTimestamp: autoTime) else {
                    return nil
                }
                autoTime = ev.timestamp + 0.5
                events.append(ev)
                continue
            }

            let delimiter: Character? = line.contains("\t") ? "\t" : ((ext == "csv" || line.contains(",")) ? "," : nil)
            if let delim = delimiter {
                let cols = splitDelimited(line, delimiter: delim)
                if lineIdx == 0 && isHeaderRow(cols) {
                    continue
                }
                if let ev = parseColumns(cols, defaultTimestamp: autoTime) {
                    autoTime = ev.timestamp + 0.5
                    events.append(ev)
                    continue
                }
            }

            let ev = CommentEvent(timestamp: autoTime, text: line, superChatJPY: 0, reactionIndex: nil)
            autoTime += 0.5
            events.append(ev)
        }

        return events
    }

    private static func parseReactionTag(_ value: Any?) -> Int? {
        if let n = value as? Int {
            return n
        }
        if let d = value as? Double {
            return Int(d)
        }
        if let s = value as? String {
            let cleaned = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if cleaned.isEmpty { return nil }
            let withoutHash = cleaned.hasPrefix("#") ? String(cleaned.dropFirst()) : cleaned
            let head = withoutHash.split(separator: "_").first.map(String.init) ?? withoutHash
            return Int(head)
        }
        return nil
    }

    private static func parseDictionary(_ dict: [String: Any], defaultTimestamp: Double) -> CommentEvent? {
        let ts: Double = if let v = (dict["timestamp"] ?? dict["time"] ?? dict["t"] ?? dict["offset"]) as? Double {
            v
        } else if let v = (dict["timestamp"] ?? dict["time"] ?? dict["t"] ?? dict["offset"]) as? Int {
            Double(v)
        } else if let s = (dict["timestamp"] ?? dict["time"] ?? dict["t"] ?? dict["offset"]) as? String, let v = Double(s) {
            v
        } else {
            defaultTimestamp
        }

        let user = (dict["user"] ?? dict["userId"] ?? dict["author"] ?? dict["name"]) as? String ?? ""
        let text = (dict["text"] ?? dict["comment"] ?? dict["message"] ?? dict["body"]) as? String ?? ""

        let sc: Int = if let v = (dict["superChatJPY"] ?? dict["superchat"] ?? dict["amount"] ?? dict["sc"]) as? Int {
            v
        } else if let v = (dict["superChatJPY"] ?? dict["superchat"] ?? dict["amount"] ?? dict["sc"]) as? Double {
            Int(v)
        } else if let s = (dict["superChatJPY"] ?? dict["superchat"] ?? dict["amount"] ?? dict["sc"]) as? String, let v = Int(s) {
            v
        } else {
            0
        }

        let reactionIdx = parseReactionTag(dict["reactionIndex"] ?? dict["reaction"] ?? dict["label"] ?? dict["emotion"] ?? dict["target"])

        return CommentEvent(timestamp: ts, user: user, text: text, superChatJPY: sc, reactionIndex: reactionIdx)
    }

    private static func isHeaderRow(_ cols: [String]) -> Bool {
        guard let first = cols.first?.lowercased() else { return false }
        return first == "timestamp" || first == "time" || first == "t" || first == "user" || first == "text" || first == "comment"
    }

    private static func splitDelimited(_ line: String, delimiter: Character) -> [String] {
        var result: [String] = []
        var current = ""
        var inQuotes = false
        for ch in line {
            if ch == "\"" {
                inQuotes.toggle()
            } else if ch == delimiter && inQuotes != true {
                result.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(ch)
            }
        }
        result.append(current.trimmingCharacters(in: .whitespaces))
        return result
    }

    private static func parseColumns(_ cols: [String], defaultTimestamp: Double) -> CommentEvent? {
        if cols.count >= 5, let ts = Double(cols[0]) {
            let user = cols[1]
            let text = cols[2]
            let sc = Int(cols[3]) ?? 0
            let reaction = parseReactionTag(cols[4])
            return CommentEvent(timestamp: ts, user: user, text: text, superChatJPY: sc, reactionIndex: reaction)
        }
        if cols.count >= 4, let ts = Double(cols[0]) {
            let text = cols[1]
            let sc = Int(cols[2]) ?? 0
            let reaction = parseReactionTag(cols[3])
            return CommentEvent(timestamp: ts, text: text, superChatJPY: sc, reactionIndex: reaction)
        }
        if cols.count == 2 {
            if let reaction = parseReactionTag(cols[0]) {
                return CommentEvent(timestamp: defaultTimestamp, text: cols[1], superChatJPY: 0, reactionIndex: reaction)
            }
            if let reaction = parseReactionTag(cols[1]) {
                return CommentEvent(timestamp: defaultTimestamp, text: cols[0], superChatJPY: 0, reactionIndex: reaction)
            }
        }
        return nil
    }
}

/// Converts incoming `CommentEvent` items into 80-dim feature frames `[inputDim]`, tracking sliding-window
/// comment velocity, unique user participation, SuperChat momentum, and hashed character N-grams across steps.
public final class CommentStreamFrontEnd: @unchecked Sendable {
    public static let inputDim = 80
    private static let metadataChannels = 8
    private static let hashBuckets = inputDim - metadataChannels

    private struct RecentEntry {
        let timestamp: Double
        let user: String
    }

    private var recentEntries: [RecentEntry] = []
    private var velocityEMA: Float = 0.0
    private var superChatMomentum: Float = 0.0
    private var crowdMomentum: Float = 0.0
    private var activeCrowdUsers: Float = 0.0
    private var autoAnonCounter: Int = 0
    private var lastEventTime: Double? = nil
    private var lastActiveCommentTime: Double? = nil
    private let windowSeconds: Double

    public private(set) var currentVelocity: Float = 0.0
    public private(set) var currentUniqueUsers: Int = 0

    public init(windowSeconds: Double = 12.0) {
        self.windowSeconds = max(0.5, windowSeconds)
    }

    public func reset() {
        recentEntries.removeAll()
        velocityEMA = 0.0
        superChatMomentum = 0.0
        crowdMomentum = 0.0
        activeCrowdUsers = 0.0
        autoAnonCounter = 0
        lastEventTime = nil
        lastActiveCommentTime = nil
        currentVelocity = 0.0
        currentUniqueUsers = 0
    }

    /// Emits one `[inputDim]` frame and updates sliding-window statistics.
    public func extractFrame(for event: CommentEvent) -> [Float] {
        var frame = [Float](repeating: 0.0, count: Self.inputDim)
        let now = event.timestamp
        if let last = lastEventTime, now < last {
            reset()
        }
        lastEventTime = now
        let isIdleTick = event.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && event.superChatJPY <= 0

        let cutoff = now - windowSeconds
        while let first = recentEntries.first, first.timestamp < cutoff {
            recentEntries.removeFirst()
        }

        let scLog = log1pf(Float(max(0, event.superChatJPY))) / 10.0
        superChatMomentum = 0.85 * superChatMomentum + scLog

        var sameUserRepeatCount = 0
        var dt: Double = 0.5

        if isIdleTick {
            let crowdScale = min(1.0, log1pf(max(0.0, activeCrowdUsers - 1.0)) / log1pf(30.0))
            let idleRetention: Float = 0.38 + 0.48 * crowdScale
            crowdMomentum *= idleRetention
            if crowdMomentum < 0.01 {
                crowdMomentum = 0.0
            }
            activeCrowdUsers = max(0.0, activeCrowdUsers * (0.60 + 0.28 * crowdScale) - 0.3)
            velocityEMA *= (0.55 + 0.30 * crowdScale)
            currentVelocity = velocityEMA
            currentUniqueUsers = Int(activeCrowdUsers.rounded(.up))
            dt = 3.0
        } else {
            if let prevActive = lastActiveCommentTime {
                dt = max(0.05, now - prevActive)
            }
            if dt > 0.75 {
                let crowdScale = min(1.0, log1pf(max(0.0, activeCrowdUsers - 1.0)) / log1pf(30.0))
                let gapDecay = powf(0.82 + 0.12 * crowdScale, Float(min(4.0, dt - 0.65)))
                crowdMomentum *= gapDecay
            }
            lastActiveCommentTime = now

            let trimmedUser = event.user.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmedUser.isEmpty {
                autoAnonCounter += 1
            }
            let userId = trimmedUser.isEmpty ? "viewer_\(autoAnonCounter)" : trimmedUser

            sameUserRepeatCount = recentEntries.filter { $0.user == userId }.count
            recentEntries.append(RecentEntry(timestamp: now, user: userId))

            let uniqueUsers = Set(recentEntries.map(\.user)).count
            if sameUserRepeatCount == 0 {
                activeCrowdUsers = max(Float(uniqueUsers), activeCrowdUsers + 1.0)
            } else {
                activeCrowdUsers = max(Float(uniqueUsers), activeCrowdUsers)
            }
            let effectiveUsers = max(uniqueUsers, Int(activeCrowdUsers.rounded()))
            currentUniqueUsers = effectiveUsers

            let shortWin = min(windowSeconds, 3.0)
            let shortCount = recentEntries.filter { $0.timestamp >= now - shortWin }.count
            let instVelocity = Float(shortCount) / Float(max(0.5, shortWin))
            velocityEMA = 0.5 * velocityEMA + 0.5 * instVelocity
            currentVelocity = velocityEMA

            let userWeight: Float = (sameUserRepeatCount == 0)
                ? 1.0
                : (1.0 / (1.0 + 3.5 * Float(sameUserRepeatCount)))
            let paceFactor = min(1.2, max(0.65, Float(0.75 / max(0.15, dt))))
            let stepIncrement = (0.20 * paceFactor + 0.08 * scLog) * userWeight
            crowdMomentum = min(1.0, crowdMomentum * 0.94 + stepIncrement)
        }

        let scalars = Array(event.text.unicodeScalars)
        let scalarCount = scalars.count

        // [0..<8]: stream dynamics & metadata features.
        let crowdScale = min(1.0, log1pf(max(0.0, activeCrowdUsers)) / log1pf(30.0))
        frame[0] = min(1.0, velocityEMA / 5.0)
        frame[1] = min(1.0, scLog)
        frame[2] = isIdleTick ? 1.0 : 0.0
        frame[3] = min(1.0, superChatMomentum / 3.0)
        frame[4] = crowdMomentum
        frame[5] = crowdScale
        frame[6] = isIdleTick ? 0.0 : (sameUserRepeatCount == 0 ? 1.0 : (1.0 / Float(1 + sameUserRepeatCount)))
        frame[7] = min(1.0, Float(dt) / 2.0)

        // [8..<80]: L2-normalized signed character 1-gram, 2-gram, and 3-gram projections (72 channels).
        if scalarCount > 0 {
            let buckets = UInt32(Self.hashBuckets)
            let baseOffset = Self.metadataChannels
            var i = 0
            while i < scalarCount {
                let u1 = scalars[i].value
                let h1a = u1 &* 2654435761
                let h1b = (u1 ^ 0x9E3779B9) &* 2246822519
                frame[baseOffset + Int(h1a % buckets)] += ((h1a & 1) == 0) ? 1.0 : -1.0
                frame[baseOffset + Int(h1b % buckets)] += ((h1b & 2) == 0) ? 0.5 : -0.5

                if i + 1 < scalarCount {
                    let u2 = scalars[i + 1].value
                    let c2 = (u1 &* 16777619) ^ u2
                    let h2a = c2 &* 2654435761
                    let h2b = (c2 ^ 0x85EBCA6B) &* 3266489917
                    frame[baseOffset + Int(h2a % buckets)] += ((h2a & 1) == 0) ? 0.85 : -0.85
                    frame[baseOffset + Int(h2b % buckets)] += ((h2b & 2) == 0) ? 0.45 : -0.45
                }

                if i + 2 < scalarCount {
                    let u2 = scalars[i + 1].value
                    let u3 = scalars[i + 2].value
                    let c3 = (((u1 &* 16777619) ^ u2) &* 16777619) ^ u3
                    let h3 = c3 &* 2654435761
                    frame[baseOffset + Int(h3 % buckets)] += ((h3 & 1) == 0) ? 0.65 : -0.65
                }
                i += 1
            }

            var normSq: Float = 0.0
            var b = 0
            while b < Self.hashBuckets {
                let v = frame[baseOffset + b]
                normSq += v * v
                b += 1
            }
            if normSq > 1e-6 {
                let invNorm = 1.0 / sqrtf(normSq)
                b = 0
                while b < Self.hashBuckets {
                    frame[baseOffset + b] *= invNorm
                    b += 1
                }
            }
        }

        return frame
    }

    public func extractFrames(for events: [CommentEvent]) -> [[Float]] {
        reset()
        return events.map { extractFrame(for: $0) }
    }
}
