import Foundation

public struct ReactionCandidate: Sendable, Equatable {
    public let state: ReactionStateSpec
    public let probability: Float

    public var baseIndex: Int { state.baseIndex }

    public init(state: ReactionStateSpec, probability: Float) {
        self.state = state
        self.probability = probability
    }
}

public struct ReactionPrediction: Sendable, Equatable {
    public let state: ReactionStateSpec
    public let baseState: ReactionStateSpec
    public let escalated: Bool
    public let phase: String
    public let intensity: Float
    public let velocity: Float
    public let uniqueUsers: Int
    public let superChatJPY: Int
    public let topCandidates: [ReactionCandidate]

    public init(
        state: ReactionStateSpec,
        baseState: ReactionStateSpec,
        escalated: Bool,
        phase: String = "idle",
        intensity: Float = 0.0,
        velocity: Float,
        uniqueUsers: Int = 0,
        superChatJPY: Int,
        topCandidates: [ReactionCandidate]
    ) {
        self.state = state
        self.baseState = baseState
        self.escalated = escalated
        self.phase = phase
        self.intensity = intensity
        self.velocity = velocity
        self.uniqueUsers = uniqueUsers
        self.superChatJPY = superChatJPY
        self.topCandidates = topCandidates
    }
}

/// Stateful streaming inference wrapper: ingests incoming `CommentEvent` items into `HawkNetwork.State`,
/// and evaluates the reaction state (including time-based emotional decay from the last comment timestamp)
/// at any requested timestamp via `state(at:)`.
public final class ReactionStreamEngine: @unchecked Sendable {
    public let network: HawkNetwork
    public let frontEnd: CommentStreamFrontEnd
    public private(set) var state: HawkNetwork.State
    public let escalationProbability: Float
    private var idleStepCount: Int = 0
    private var lastCommentTimestamp: Double? = nil
    private var previousBaseIndex: Int = 0
    private var previousIntensity: Float = 0.0
    private var previousTier: Int = 1
    private var lastVelocity: Float = 0.0
    private var lastUniqueUsers: Int = 0
    private var lastSuperChatJPY: Int = 0
    private var lastProbabilities: [Float] = []

    public init(
        network: HawkNetwork,
        windowSeconds: Double = 12.0,
        escalationProbability: Float = 0.64
    ) {
        self.network = network
        self.frontEnd = CommentStreamFrontEnd(windowSeconds: windowSeconds)
        self.state = network.makeState()
        self.escalationProbability = escalationProbability
    }

    public func reset() {
        frontEnd.reset()
        state = network.makeState()
        idleStepCount = 0
        lastCommentTimestamp = nil
        previousBaseIndex = 0
        previousIntensity = 0.0
        previousTier = 1
        lastVelocity = 0.0
        lastUniqueUsers = 0
        lastSuperChatJPY = 0
        lastProbabilities = []
    }

    public func push(_ event: CommentEvent) -> ReactionStateSpec {
        return pushDetailed(event).state
    }

    /// Ingests a comment and returns the reaction state at `event.timestamp`.
    public func ingest(_ event: CommentEvent, topK: Int = 5) -> ReactionPrediction {
        return pushDetailed(event, topK: topK)
    }

    /// Computes the reaction state at an arbitrary `timestamp` without feeding dummy idle events.
    /// Uses `timestamp - lastCommentTimestamp` and the active crowd size to calculate emotional decay.
    public func state(at timestamp: Double, topK: Int = 5) -> ReactionPrediction {
        guard let lastTs = lastCommentTimestamp else {
            return ReactionPrediction(
                state: .idle,
                baseState: .idle,
                escalated: false,
                phase: "idle",
                intensity: 1.0,
                velocity: 0.0,
                uniqueUsers: 0,
                superChatJPY: 0,
                topCandidates: [ReactionCandidate(state: .idle, probability: 1.0)]
            )
        }

        let elapsed = max(0.0, timestamp - lastTs)
        let crowdScale = min(1.0, log1pf(max(0.0, Float(lastUniqueUsers) - 1.0)) / log1pf(30.0))

        if previousBaseIndex <= 0 || previousIntensity <= 0.0 {
            return ReactionPrediction(
                state: .idle,
                baseState: .idle,
                escalated: false,
                phase: "idle",
                intensity: 1.0,
                velocity: 0.0,
                uniqueUsers: lastUniqueUsers,
                superChatJPY: 0,
                topCandidates: [ReactionCandidate(state: .idle, probability: 1.0)]
            )
        }

        let retentionPer3s: Float = 0.38 + 0.48 * crowdScale
        let decaySteps: Float = elapsed > 0.8 ? Float(elapsed - 0.8) / 3.0 : 0.0
        let factor: Float = decaySteps > 0.0 ? powf(retentionPer3s, decaySteps) : 1.0
        let decayedIntensity = previousIntensity * factor
        let decayedVelocity = lastVelocity * powf(0.55 + 0.30 * crowdScale, decaySteps)
        let decayedUsers = max(0, Int((Float(lastUniqueUsers) * powf(0.60 + 0.28 * crowdScale, decaySteps)).rounded()))

        if decayedIntensity < 0.24 {
            let idleProb = max(0.5, 1.0 - decayedIntensity)
            let prevState = ReactionStateSpec(baseIndex: previousBaseIndex, tier: 1)
            return ReactionPrediction(
                state: .idle,
                baseState: .idle,
                escalated: false,
                phase: "idle",
                intensity: idleProb,
                velocity: decayedVelocity,
                uniqueUsers: decayedUsers,
                superChatJPY: 0,
                topCandidates: [
                    ReactionCandidate(state: .idle, probability: idleProb),
                    ReactionCandidate(state: prevState, probability: decayedIntensity)
                ]
            )
        }

        let isDecaying = elapsed > 0.8
        let tier: Int = if isDecaying {
            if decayedIntensity >= 0.82 {
                min(previousTier, 3)
            } else if decayedIntensity >= 0.60 {
                min(previousTier, 2)
            } else {
                1
            }
        } else {
            previousTier
        }

        let phase: String = if isDecaying {
            "decaying"
        } else if tier >= 2 {
            "peak"
        } else {
            "rising"
        }

        let baseState = ReactionStateSpec(baseIndex: previousBaseIndex, tier: 1)
        let resolvedState = ReactionStateSpec(baseIndex: previousBaseIndex, tier: tier)

        var candidates: [ReactionCandidate] = [
            ReactionCandidate(state: resolvedState, probability: decayedIntensity),
            ReactionCandidate(state: .idle, probability: max(0.0, 1.0 - decayedIntensity))
        ]
        if lastProbabilities.isEmpty != true {
            for i in 1..<lastProbabilities.count where i != previousBaseIndex && candidates.count < max(1, topK) {
                let p = lastProbabilities[i] * factor
                if p > 0.01 {
                    candidates.append(ReactionCandidate(state: ReactionStateSpec(baseIndex: i, tier: 1), probability: p))
                }
            }
        }
        candidates.sort { $0.probability > $1.probability }

        return ReactionPrediction(
            state: resolvedState,
            baseState: baseState,
            escalated: resolvedState.baseIndex > 0 && resolvedState.tier >= 2,
            phase: phase,
            intensity: decayedIntensity,
            velocity: decayedVelocity,
            uniqueUsers: decayedUsers,
            superChatJPY: isDecaying ? 0 : lastSuperChatJPY,
            topCandidates: Array(candidates.prefix(max(1, topK)))
        )
    }

    public func pushDetailed(_ event: CommentEvent, topK: Int = 5) -> ReactionPrediction {
        let isIdleQuery = event.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && event.superChatJPY <= 0
        if isIdleQuery {
            idleStepCount += 1
            let queryTime = if let lastTs = lastCommentTimestamp, event.timestamp <= lastTs {
                lastTs + 3.0 * Double(idleStepCount)
            } else {
                event.timestamp
            }
            return state(at: queryTime, topK: topK)
        }

        var hadSilenceGap = false
        if let lastTs = lastCommentTimestamp {
            if event.timestamp < lastTs {
                reset()
            } else {
                let dt = event.timestamp - lastTs
                if dt > 2.2 {
                    hadSilenceGap = true
                    let crowdScale = min(1.0, log1pf(max(0.0, Float(lastUniqueUsers) - 1.0)) / log1pf(30.0))
                    let retentionPer3s: Float = 0.38 + 0.48 * crowdScale
                    let decayFactor = powf(retentionPer3s, Float(dt - 1.8) / 3.0)
                    state.attenuate(by: decayFactor)
                    previousIntensity *= decayFactor
                    if previousIntensity < 0.24 {
                        previousBaseIndex = 0
                        previousIntensity = 0.0
                        previousTier = 1
                    } else if previousIntensity < 0.60 {
                        previousTier = 1
                    } else if previousIntensity < 0.82 {
                        previousTier = min(previousTier, 2)
                    }
                }
            }
        }

        lastCommentTimestamp = event.timestamp
        idleStepCount = 0

        let frame = frontEnd.extractFrame(for: event)
        let logProbs = network.push([frame], state: state)

        let velocity = frontEnd.currentVelocity
        let uniqueUsers = frontEnd.currentUniqueUsers
        let classCount = ReactionStateSpec.numBaseClasses

        guard let scores = logProbs.last, scores.isEmpty != true else {
            return state(at: event.timestamp, topK: topK)
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

        // Preserve active reaction inertia while comments continue without a silence gap.
        if bestIdx == 0 && previousBaseIndex > 0 && previousBaseIndex < n && !hadSilenceGap {
            let prevClassProb = probs[previousBaseIndex]
            if event.superChatJPY > 0 && (prevClassProb >= 0.14 || bestNonIdleProb >= 0.22) {
                if prevClassProb >= bestNonIdleProb * 0.70 {
                    bestIdx = previousBaseIndex
                    bestProb = prevClassProb
                } else {
                    bestIdx = bestNonIdleIdx
                    bestProb = bestNonIdleProb
                }
            } else if prevClassProb >= 0.18 {
                bestIdx = previousBaseIndex
                bestProb = prevClassProb
            }
        }

        let canEscalate = uniqueUsers >= 2 || event.superChatJPY > 0
        if bestIdx > 0 && bestIdx == previousBaseIndex && !hadSilenceGap && canEscalate {
            let crowdBoost: Float = uniqueUsers > lastUniqueUsers ? 0.03 : 0.0
            let scBoost: Float = event.superChatJPY > 0 ? 0.05 : 0.0
            bestProb = max(bestProb, min(0.96, previousIntensity + crowdBoost + scBoost))
            probs[bestIdx] = bestProb
            if let pos = indexed.firstIndex(where: { $0.index == bestIdx }) {
                indexed[pos].prob = bestProb
            }
        }
        indexed.sort { $0.prob > $1.prob }

        let rawTier: Int = if bestIdx <= 0 || !canEscalate {
            1
        } else if bestProb >= 0.82 && (uniqueUsers >= 4 || event.superChatJPY >= 1000) {
            3
        } else if bestProb >= escalationProbability && (uniqueUsers >= 3 || event.superChatJPY >= 500) {
            2
        } else {
            1
        }

        let virtualTier: Int = if bestIdx > 0 && bestIdx == previousBaseIndex && !hadSilenceGap && canEscalate {
            max(rawTier, previousTier)
        } else {
            rawTier
        }

        let phase: String = if bestIdx <= 0 {
            "idle"
        } else if virtualTier >= 2 {
            "peak"
        } else {
            "rising"
        }

        previousBaseIndex = bestIdx
        previousIntensity = bestIdx > 0 ? bestProb : 0.0
        previousTier = virtualTier
        lastVelocity = velocity
        lastUniqueUsers = uniqueUsers
        lastSuperChatJPY = event.superChatJPY
        lastProbabilities = probs

        let k = max(1, min(topK, indexed.count))
        let topCandidates: [ReactionCandidate] = indexed.prefix(k).map { item in
            let itemTier = (item.index == bestIdx) ? virtualTier : 1
            return ReactionCandidate(
                state: ReactionStateSpec(baseIndex: item.index, tier: itemTier),
                probability: item.prob
            )
        }

        let baseState = ReactionStateSpec(baseIndex: bestIdx, tier: 1)
        let resolvedState = ReactionStateSpec(baseIndex: bestIdx, tier: virtualTier)

        return ReactionPrediction(
            state: resolvedState,
            baseState: baseState,
            escalated: resolvedState.baseIndex > 0 && resolvedState.tier >= 2,
            phase: phase,
            intensity: bestIdx > 0 ? bestProb : probs[0],
            velocity: velocity,
            uniqueUsers: uniqueUsers,
            superChatJPY: event.superChatJPY,
            topCandidates: topCandidates
        )
    }
}
