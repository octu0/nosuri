import Foundation

public struct ReactionClip: Sendable, Equatable {
    public let id: String
    public let baseIndex: Int
    public let tier: Int
    public let filename: String

    public init(id: String, baseIndex: Int, tier: Int, filename: String) {
        self.id = id
        self.baseIndex = baseIndex
        self.tier = tier
        self.filename = filename
    }
}

/// Discovers `#0*`..`#38*` reaction clips from a directory (default `.tmp/resources`) and resolves
/// the clip for a predicted class index and stream intensity.
public struct ReactionCatalog: Sendable {
    public static let numBaseClasses = 39
    public static let defaultResourcesDirectory = ".tmp/resources"
    public static let defaultEscalationVelocity: Float = 1.0
    public static let defaultEscalationSuperChatJPY: Int = 5000

    public let idleClips: [ReactionClip]
    public let baseReactions: [ReactionClip]
    public let escalatedVariants: [Int: [ReactionClip]]

    public init(directory: String = ReactionCatalog.defaultResourcesDirectory) {
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: directory)) ?? []
        let mp4Files = entries.filter { $0.hasPrefix("#") && $0.lowercased().hasSuffix(".mp4") }.sorted()

        var discoveredIdle: [ReactionClip] = []
        var discoveredBase: [Int: ReactionClip] = [:]
        var discoveredEscalated: [Int: [ReactionClip]] = [:]

        for file in mp4Files {
            guard let parsed = Self.parseFilename(file) else { continue }
            if parsed.baseIndex == 0 {
                discoveredIdle.append(parsed)
                if parsed.tier == 1 && discoveredBase[0] == nil {
                    discoveredBase[0] = parsed
                }
            } else if parsed.tier == 1 {
                if discoveredBase[parsed.baseIndex] == nil {
                    discoveredBase[parsed.baseIndex] = parsed
                }
            } else {
                discoveredEscalated[parsed.baseIndex, default: []].append(parsed)
            }
        }

        // Put tier-1 #0 clip first in idle rotation.
        discoveredIdle.sort { a, b in
            if a.tier != b.tier { return a.tier < b.tier }
            return a.filename < b.filename
        }
        if discoveredIdle.isEmpty {
            discoveredIdle = [ReactionClip(id: "#0", baseIndex: 0, tier: 1, filename: "#0.mp4")]
        }
        if discoveredBase[0] == nil {
            discoveredBase[0] = discoveredIdle[0]
        }

        var bases: [ReactionClip] = []
        bases.reserveCapacity(Self.numBaseClasses)
        for i in 0..<Self.numBaseClasses {
            if let clip = discoveredBase[i] {
                bases.append(clip)
            } else {
                bases.append(ReactionClip(id: "#\(i)", baseIndex: i, tier: 1, filename: "#\(i).mp4"))
            }
        }

        self.idleClips = discoveredIdle
        self.baseReactions = bases
        self.escalatedVariants = discoveredEscalated
    }

    /// Parses `#<baseIndex>[_<tier>][_<hash>].mp4`.
    public static func parseFilename(_ filename: String) -> ReactionClip? {
        guard filename.hasPrefix("#"), filename.lowercased().hasSuffix(".mp4") else { return nil }
        let stem = String(filename.dropFirst().dropLast(4))
        let parts = stem.split(separator: "_", omittingEmptySubsequences: true)
        guard let first = parts.first,
              let baseIndex = Int(first),
              0 <= baseIndex && baseIndex < numBaseClasses else {
            return nil
        }

        if parts.count >= 2, parts[1].count <= 2, let sub = Int(parts[1]), sub >= 2 {
            return ReactionClip(id: "#\(baseIndex)_\(sub)", baseIndex: baseIndex, tier: sub, filename: filename)
        }
        return ReactionClip(id: "#\(baseIndex)", baseIndex: baseIndex, tier: 1, filename: filename)
    }

    public var allClips: [ReactionClip] {
        var result = idleClips
        for clip in baseReactions where clip.baseIndex != 0 {
            result.append(clip)
            if let variants = escalatedVariants[clip.baseIndex] {
                result.append(contentsOf: variants)
            }
        }
        return result
    }

    public func resolveClip(
        baseIndex: Int,
        commentVelocity: Float,
        superChatJPY: Int = 0,
        idleStep: Int = 0,
        virtualTier: Int? = nil,
        escalationVelocity: Float = ReactionCatalog.defaultEscalationVelocity,
        escalationSuperChatJPY: Int = ReactionCatalog.defaultEscalationSuperChatJPY
    ) -> ReactionClip {
        let clamped = max(0, min(Self.numBaseClasses - 1, baseIndex))
        if clamped == 0 {
            let idx = abs(idleStep) % idleClips.count
            return idleClips[idx]
        }

        let targetTier: Int = if let vt = virtualTier {
            max(1, vt)
        } else if escalationVelocity <= commentVelocity || escalationSuperChatJPY <= superChatJPY {
            2
        } else {
            1
        }

        if targetTier >= 2 {
            if let variants = escalatedVariants[clamped], variants.isEmpty != true {
                let variantIdx = (virtualTier == nil && escalationSuperChatJPY <= superChatJPY && 1 < variants.count)
                    ? 1
                    : min(targetTier - 2, variants.count - 1)
                let chosen = variants[max(0, variantIdx)]
                if chosen.tier == targetTier {
                    return chosen
                }
                return ReactionClip(
                    id: "#\(clamped)_\(targetTier)",
                    baseIndex: clamped,
                    tier: targetTier,
                    filename: chosen.filename
                )
            }
            let base = baseReactions[clamped]
            return ReactionClip(
                id: "#\(clamped)_\(targetTier)",
                baseIndex: clamped,
                tier: targetTier,
                filename: base.filename
            )
        }
        return baseReactions[clamped]
    }

    public static let shared = ReactionCatalog()
}
