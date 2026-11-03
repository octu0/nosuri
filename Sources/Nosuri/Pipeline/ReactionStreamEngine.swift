import Foundation

public struct ReactionCandidate: Sendable, Equatable {
    public let baseIndex: Int
    public let clip: ReactionClip
    public let probability: Float

    public init(baseIndex: Int, clip: ReactionClip, probability: Float) {
        self.baseIndex = baseIndex
        self.clip = clip
        self.probability = probability
    }
}

public struct ReactionPrediction: Sendable, Equatable {
    public let clip: ReactionClip
    public let baseClip: ReactionClip
    public let escalated: Bool
    public let phase: String
    public let intensity: Float
    public let velocity: Float
    public let uniqueUsers: Int
    public let superChatJPY: Int
    public let topCandidates: [ReactionCandidate]

    public init(
        clip: ReactionClip,
        baseClip: ReactionClip,
        escalated: Bool,
        phase: String = "idle",
        intensity: Float = 0.0,
        velocity: Float,
        uniqueUsers: Int = 0,
        superChatJPY: Int,
        topCandidates: [ReactionCandidate]
    ) {
        self.clip = clip
        self.baseClip = baseClip
        self.escalated = escalated
        self.phase = phase
        self.intensity = intensity
        self.velocity = velocity
        self.uniqueUsers = uniqueUsers
        self.superChatJPY = superChatJPY
        self.topCandidates = topCandidates
    }
}

/// Stateful streaming inference wrapper: extracts one feature frame per `CommentEvent`, advances
/// `HawkNetwork.State`, resolves virtual `#X_y` tiers (`#X`, `#X_2`, `#X_3`), and tracks emotional decay.
public final class ReactionStreamEngine: @unchecked Sendable {
    public let network: HawkNetwork
    public let frontEnd: CommentStreamFrontEnd
    public let catalog: ReactionCatalog
    public private(set) var state: HawkNetwork.State
    public let escalationProbability: Float
    private var idleStepCount: Int = 0
    private var lastTimestamp: Double? = nil
    private var previousBaseIndex: Int = 0
    private var previousIntensity: Float = 0.0
    private var previousTier: Int = 1

    public init(
        network: HawkNetwork,
        catalog: ReactionCatalog = .shared,
        windowSeconds: Double = 12.0,
        escalationProbability: Float = 0.62
    ) {
        self.network = network
        self.catalog = catalog
        self.frontEnd = CommentStreamFrontEnd(windowSeconds: windowSeconds)
        self.state = network.makeState()
        self.escalationProbability = escalationProbability
    }

    public func reset() {
        frontEnd.reset()
        state = network.makeState()
        idleStepCount = 0
        lastTimestamp = nil
        previousBaseIndex = 0
        previousIntensity = 0.0
        previousTier = 1
    }

    public func push(_ event: CommentEvent) -> ReactionClip {
        return pushDetailed(event).clip
    }

    public func pushDetailed(_ event: CommentEvent, topK: Int = 5) -> ReactionPrediction {
        if let last = lastTimestamp, event.timestamp < last {
            frontEnd.reset()
            state = network.makeState()
            previousBaseIndex = 0
            previousIntensity = 0.0
            previousTier = 1
        }
        lastTimestamp = event.timestamp
        let frame = frontEnd.extractFrame(for: event)
        let isIdleTick = event.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && event.superChatJPY <= 0

        let logProbs = network.push([frame], state: state)
        if isIdleTick {
            idleStepCount += 1
        }

        let velocity = frontEnd.currentVelocity
        let uniqueUsers = frontEnd.currentUniqueUsers
        let classCount = ReactionCatalog.numBaseClasses

        guard let scores = logProbs.last, scores.isEmpty != true else {
            let idleClip = catalog.resolveClip(
                baseIndex: 0,
                commentVelocity: velocity,
                superChatJPY: event.superChatJPY,
                idleStep: idleStepCount
            )
            return ReactionPrediction(
                clip: idleClip,
                baseClip: idleClip,
                escalated: false,
                phase: "idle",
                intensity: 0.0,
                velocity: velocity,
                uniqueUsers: uniqueUsers,
                superChatJPY: event.superChatJPY,
                topCandidates: [ReactionCandidate(baseIndex: 0, clip: idleClip, probability: 1.0)]
            )
        }

        let n = min(scores.count, classCount)
        var probs = [Float](repeating: 0.0, count: n)
        var indexed: [(index: Int, prob: Float)] = []
        indexed.reserveCapacity(n)
        var bestIdx = 0
        var bestProb: Float = -1.0
        var bestNonIdleIdx = 0
        var bestNonIdleProb: Float = -1.0

        for i in 0..<n {
            let p = expf(scores[i])
            probs[i] = p
            indexed.append((i, p))
            if bestProb < p {
                bestProb = p
                bestIdx = i
            }
            if i > 0 && bestNonIdleProb < p {
                bestNonIdleProb = p
                bestNonIdleIdx = i
            }
        }
        indexed.sort { $0.prob > $1.prob }

        // Preserve active reaction inertia during SuperChat or gradual decay tail (#X_2 -> #X -> #0).
        if bestIdx == 0 && previousBaseIndex > 0 && previousBaseIndex < n {
            let prevClassProb = probs[previousBaseIndex]
            if event.superChatJPY > 0 && (prevClassProb >= 0.16 || bestNonIdleProb >= 0.22) {
                if prevClassProb >= bestNonIdleProb * 0.75 {
                    bestIdx = previousBaseIndex
                    bestProb = max(prevClassProb, previousIntensity * 0.90)
                } else {
                    bestIdx = bestNonIdleIdx
                    bestProb = bestNonIdleProb
                }
            } else if prevClassProb >= 0.24 {
                bestIdx = previousBaseIndex
                bestProb = prevClassProb
            }
        }

        let isDecaying = bestIdx > 0 && (isIdleTick || (bestIdx == previousBaseIndex && bestProb < previousIntensity - 0.03))

        let virtualTier: Int = if bestIdx <= 0 {
            1
        } else if isDecaying {
            if bestProb >= 0.65 {
                min(previousTier, 2)
            } else {
                1
            }
        } else if bestProb >= 0.80 && (uniqueUsers >= 5 || event.superChatJPY >= 1000) {
            3
        } else if bestProb >= escalationProbability && (uniqueUsers >= 4 || event.superChatJPY >= 1000) {
            2
        } else if bestIdx == previousBaseIndex && previousTier >= 2 && bestProb >= 0.55 && uniqueUsers >= 4 {
            previousTier
        } else {
            1
        }

        let phase: String = if bestIdx <= 0 {
            "idle"
        } else if isDecaying {
            "decaying"
        } else if virtualTier >= 2 {
            "peak"
        } else {
            "rising"
        }

        previousBaseIndex = bestIdx
        previousIntensity = bestIdx > 0 ? bestProb : 0.0
        previousTier = virtualTier

        let k = max(1, min(topK, indexed.count))
        let topCandidates: [ReactionCandidate] = indexed.prefix(k).map { item in
            let itemTier = (item.index == bestIdx) ? virtualTier : 1
            let candidateClip = catalog.resolveClip(
                baseIndex: item.index,
                commentVelocity: velocity,
                superChatJPY: event.superChatJPY,
                idleStep: idleStepCount,
                virtualTier: itemTier
            )
            return ReactionCandidate(baseIndex: item.index, clip: candidateClip, probability: item.prob)
        }

        let baseClip = catalog.resolveClip(
            baseIndex: bestIdx,
            commentVelocity: 0.0,
            superChatJPY: 0,
            idleStep: idleStepCount,
            virtualTier: 1
        )
        let resolvedClip = catalog.resolveClip(
            baseIndex: bestIdx,
            commentVelocity: velocity,
            superChatJPY: event.superChatJPY,
            idleStep: idleStepCount,
            virtualTier: virtualTier
        )

        return ReactionPrediction(
            clip: resolvedClip,
            baseClip: baseClip,
            escalated: resolvedClip.baseIndex > 0 && resolvedClip.tier >= 2,
            phase: phase,
            intensity: bestIdx > 0 ? bestProb : probs[0],
            velocity: velocity,
            uniqueUsers: uniqueUsers,
            superChatJPY: event.superChatJPY,
            topCandidates: topCandidates
        )
    }
}
