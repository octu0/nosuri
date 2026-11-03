import Foundation

/// Training target filter that converts a sequence of discrete `CommentEvent` annotations into
/// smooth probability distributions over `0..<numClasses`, modeling user-diversity emotional buildup
/// (dampening single-user spam, scaling with unique users) and crowd-size-dependent idle decay.
public struct EmotionSlopeFilter: Sendable {
    public let numClasses: Int
    public let windowSeconds: Double
    public let stepGain: Float
    public let activeRetention: Float
    public let otherDecay: Float
    public let minIdleRetention: Float
    public let maxIdleRetention: Float
    public let maxSuperChatBonus: Float
    public let maxVelocityBonus: Float

    private struct WindowEntry {
        let timestamp: Double
        let user: String
        let target: Int
    }

    public init(
        numClasses: Int = ReactionStateSpec.numBaseClasses,
        windowSeconds: Double = 12.0,
        stepGain: Float = 0.22,
        activeRetention: Float = 0.96,
        otherDecay: Float = 0.85,
        minIdleRetention: Float = 0.38,
        maxIdleRetention: Float = 0.86,
        maxSuperChatBonus: Float = 0.12,
        maxVelocityBonus: Float = 0.04
    ) {
        self.numClasses = max(2, numClasses)
        self.windowSeconds = max(1.0, windowSeconds)
        self.stepGain = stepGain
        self.activeRetention = activeRetention
        self.otherDecay = otherDecay
        self.minIdleRetention = minIdleRetention
        self.maxIdleRetention = maxIdleRetention
        self.maxSuperChatBonus = maxSuperChatBonus
        self.maxVelocityBonus = maxVelocityBonus
    }

    /// Computes soft target distributions `[frames][numClasses]` for a sequence of `CommentEvent`s.
    public func filter(events: [CommentEvent]) -> [[Float]] {
        var levels = [Float](repeating: 0.0, count: numClasses)
        var distributions: [[Float]] = []
        distributions.reserveCapacity(events.count)

        var recentComments: [WindowEntry] = []
        var activeCrowdUsers: Float = 0.0
        var prevTimestamp: Double? = nil
        var prevActiveTimestamp: Double? = nil
        var autoAnonCounter = 0

        for event in events {
            let now = event.timestamp
            if let prev = prevTimestamp, now < prev {
                levels = [Float](repeating: 0.0, count: numClasses)
                recentComments.removeAll()
                activeCrowdUsers = 0.0
                prevActiveTimestamp = nil
            }
            prevTimestamp = now

            let cutoff = now - windowSeconds
            while let first = recentComments.first, first.timestamp < cutoff {
                recentComments.removeFirst()
            }

            let target = max(0, min(numClasses - 1, event.reactionIndex ?? 0))
            let trimmed = event.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let isIdle = (trimmed.isEmpty && event.superChatJPY <= 0) || target == 0

            if isIdle {
                let crowdScale = min(1.0, log1pf(max(0.0, activeCrowdUsers - 1.0)) / log1pf(30.0))
                let idleRetention = minIdleRetention + (maxIdleRetention - minIdleRetention) * crowdScale
                for c in 1..<numClasses {
                    levels[c] *= idleRetention
                    if levels[c] < 0.01 {
                        levels[c] = 0.0
                    }
                }
                activeCrowdUsers = max(0.0, activeCrowdUsers * (0.50 + 0.30 * crowdScale) - 0.4)
            } else {
                let dt: Double = if let prevActive = prevActiveTimestamp {
                    max(0.05, now - prevActive)
                } else {
                    0.5
                }
                prevActiveTimestamp = now

                if dt > 0.75 {
                    let crowdScale = min(1.0, log1pf(max(0.0, activeCrowdUsers - 1.0)) / log1pf(30.0))
                    let gapDecay = powf(0.82 + 0.12 * crowdScale, Float(min(4.0, dt - 0.65)))
                    for c in 1..<numClasses {
                        levels[c] *= gapDecay
                        if levels[c] < 0.01 {
                            levels[c] = 0.0
                        }
                    }
                }

                let trimmedUser = event.user.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedUser.isEmpty {
                    autoAnonCounter += 1
                }
                let userId = trimmedUser.isEmpty ? "viewer_\(autoAnonCounter)" : trimmedUser

                let sameUserCount = recentComments.filter { $0.user == userId }.count
                recentComments.append(WindowEntry(timestamp: now, user: userId, target: target))

                let uniqueUsers = Set(recentComments.map(\.user)).count
                if sameUserCount == 0 {
                    activeCrowdUsers = max(Float(uniqueUsers), activeCrowdUsers + 1.0)
                } else {
                    activeCrowdUsers = max(Float(uniqueUsers), activeCrowdUsers)
                }
                let effectiveUsers = max(uniqueUsers, Int(activeCrowdUsers.rounded()))

                let userWeight: Float = (sameUserCount == 0)
                    ? 1.0
                    : (1.0 / (1.0 + 3.5 * Float(sameUserCount)))

                let crowdFactor: Float = if effectiveUsers <= 1 {
                    0.60
                } else if effectiveUsers == 2 {
                    0.85
                } else {
                    min(1.40, 1.0 + 0.12 * log1pf(Float(effectiveUsers - 3)))
                }

                let maxReachable: Float = if event.superChatJPY >= 5000 {
                    0.98
                } else if effectiveUsers <= 1 {
                    0.34
                } else if effectiveUsers == 2 {
                    0.62
                } else if effectiveUsers == 3 {
                    0.73
                } else {
                    0.98
                }

                let paceFactor: Float = if sameUserCount > 0 {
                    1.0
                } else if dt <= 0.20 {
                    1.18
                } else if dt <= 0.65 {
                    1.0
                } else {
                    max(0.68, 1.0 - Float(dt - 0.65) * 0.35)
                }

                let scBonus = min(maxSuperChatBonus, log1pf(Float(max(0, event.superChatJPY))) / 80.0)
                let velBonus = (dt < 0.5 && sameUserCount == 0) ? min(maxVelocityBonus, Float(0.5 - dt) * 0.08) : 0.0
                let delta = (stepGain * crowdFactor * paceFactor + scBonus + velBonus) * userWeight
                let retention: Float = (sameUserCount > 0 && effectiveUsers <= 1) ? 0.90 : activeRetention

                for c in 1..<numClasses {
                    if c == target {
                        levels[c] = min(maxReachable, levels[c] * retention + delta)
                    } else {
                        levels[c] *= otherDecay
                        if levels[c] < 0.01 {
                            levels[c] = 0.0
                        }
                    }
                }
            }

            var dist = [Float](repeating: 0.0, count: numClasses)
            var activeSum: Float = 0.0
            for c in 1..<numClasses {
                activeSum += levels[c]
            }

            if activeSum <= 0.0 {
                dist[0] = 1.0
            } else if activeSum < 1.0 {
                dist[0] = 1.0 - activeSum
                for c in 1..<numClasses {
                    dist[c] = levels[c]
                }
            } else {
                dist[0] = 0.0
                let inv = 1.0 / activeSum
                for c in 1..<numClasses {
                    dist[c] = levels[c] * inv
                }
            }

            distributions.append(dist)
        }

        return distributions
    }

    /// Returns the dominant class index (`argmax`) at each step of the filtered distributions.
    public func dominantTargets(events: [CommentEvent]) -> [Int] {
        return filter(events: events).map { Self.argmax($0) }
    }

    /// Computes soft target distributions `[frames][numClasses]` for a `ReactionScenario` (`"reaction_state": "A->B"`).
    public func filter(scenario: ReactionScenario) -> [[Float]] {
        let events = scenario.data
        let n = events.count
        if n == 0 { return [] }

        let states = scenario.states.count >= 2
            ? scenario.states
            : [ReactionStateSpec(baseIndex: 0), scenario.toState]
        let numSegments = max(1, states.count - 1)

        var distributions: [[Float]] = []
        distributions.reserveCapacity(n)

        var seenUsers: [String: Int] = [:]
        var prevTs: Double? = nil
        var autoUserCounter = 0

        for segIdx in 0..<numSegments {
            let startIdx = (segIdx * n) / numSegments
            let endIdx = ((segIdx + 1) * n) / numSegments
            if endIdx <= startIdx { continue }

            let startSpec = states[segIdx]
            let endSpec = states[segIdx + 1]
            let segCount = endIdx - startIdx

            var weights = [Float](repeating: 0.0, count: segCount)
            var weightSum: Float = 0.0

            for k in 0..<segCount {
                let ev = events[startIdx + k]
                let dt: Double = if let prev = prevTs {
                    max(0.05, ev.timestamp - prev)
                } else {
                    0.5
                }
                prevTs = ev.timestamp

                let trimmedUser = ev.user.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedUser.isEmpty {
                    autoUserCounter += 1
                }
                let userId = trimmedUser.isEmpty ? "viewer_\(autoUserCounter)" : trimmedUser
                let repeatCount = seenUsers[userId, default: 0]
                seenUsers[userId] = repeatCount + 1

                let userWeight: Float = repeatCount == 0
                    ? 1.0
                    : (1.0 / (1.0 + 3.5 * Float(repeatCount)))
                let paceFactor: Float = if repeatCount > 0 {
                    1.0
                } else if dt <= 0.20 {
                    1.20
                } else if dt <= 0.65 {
                    1.0
                } else {
                    max(0.68, 1.0 - Float(dt - 0.65) * 0.35)
                }
                let scBonus = min(0.35, log1pf(Float(max(0, ev.superChatJPY))) / 35.0)
                let w = max(0.05, (paceFactor + scBonus) * userWeight)
                weights[k] = w
                weightSum += w
            }

            let invTotal = 1.0 / max(1e-5, weightSum)
            var cumWeight: Float = 0.0

            for k in 0..<segCount {
                cumWeight += weights[k]
                let progress = min(1.0, max(0.0, cumWeight * invTotal))
                var levels = [Float](repeating: 0.0, count: numClasses)

                let fromIdx = max(0, min(numClasses - 1, startSpec.baseIndex))
                let toIdx = max(0, min(numClasses - 1, endSpec.baseIndex))

                if fromIdx == 0 && toIdx == 0 {
                    // #0 -> #0 (e.g. single-user spam or scattered non-escalating comments)
                    if let hint = events[startIdx + k].reactionIndex, hint > 0 && hint < numClasses {
                        levels[hint] = min(0.22, 0.18 * progress)
                    }
                } else if fromIdx == toIdx {
                    // Same base class transition (e.g. #5_2 -> #5_3 or #5_3 -> #5_2)
                    levels[toIdx] = startSpec.targetIntensity + (endSpec.targetIntensity - startSpec.targetIntensity) * progress
                } else {
                    if fromIdx > 0 {
                        levels[fromIdx] = startSpec.targetIntensity * max(0.0, 1.0 - 0.90 * progress)
                    }
                    if toIdx > 0 {
                        levels[toIdx] = endSpec.targetIntensity * progress
                    }
                }

                distributions.append(Self.makeDistribution(from: levels, numClasses: numClasses))
            }
        }

        return distributions
    }

    public func dominantTargets(scenario: ReactionScenario) -> [Int] {
        return filter(scenario: scenario).map { Self.argmax($0) }
    }

    /// Converts a list of `ReactionScenario` items into training episodes (including chained multi-stage episodes
    /// whenever consecutive scenarios share a non-idle boundary state `prev.toState == next.fromState`).
    public func buildTrainingEpisodes(
        scenarios: [ReactionScenario],
        frontEnd: CommentStreamFrontEnd
    ) -> [(frames: [[Float]], distributions: [[Float]], dominants: [Int])] {
        var episodes: [(frames: [[Float]], distributions: [[Float]], dominants: [Int])] = []
        var chainGroup: [ReactionScenario] = []

        func flushChain(_ group: [ReactionScenario]) {
            guard group.count >= 2 else { return }
            var combinedEvents: [CommentEvent] = []
            var combinedDists: [[Float]] = []
            var timeCursor: Double = 0.0

            for sc in group {
                let scDists = filter(scenario: sc)
                guard scDists.count == sc.data.count else { continue }
                let firstTs = sc.data.first?.timestamp ?? 1.0
                let offset = (combinedEvents.isEmpty || firstTs > timeCursor) ? 0.0 : (timeCursor - firstTs + 0.45)
                for (idx, ev) in sc.data.enumerated() {
                    let adjustedTs = ev.timestamp + offset
                    timeCursor = max(timeCursor, adjustedTs)
                    combinedEvents.append(CommentEvent(
                        timestamp: adjustedTs,
                        user: ev.user,
                        text: ev.text,
                        superChatJPY: ev.superChatJPY,
                        reactionIndex: ev.reactionIndex
                    ))
                    combinedDists.append(scDists[idx])
                }
            }
            if combinedEvents.isEmpty != true {
                let frames = frontEnd.extractFrames(for: combinedEvents)
                let doms = combinedDists.map { Self.argmax($0) }
                episodes.append((frames: frames, distributions: combinedDists, dominants: doms))
            }
        }

        for sc in scenarios where sc.data.isEmpty != true {
            let frames = frontEnd.extractFrames(for: sc.data)
            let dists = filter(scenario: sc)
            let doms = dists.map { Self.argmax($0) }
            episodes.append((frames: frames, distributions: dists, dominants: doms))

            if let prev = chainGroup.last {
                if sc.fromState.baseIndex != 0 && prev.toState == sc.fromState {
                    chainGroup.append(sc)
                } else {
                    flushChain(chainGroup)
                    chainGroup = [sc]
                }
            } else {
                chainGroup = [sc]
            }
        }
        flushChain(chainGroup)

        return episodes
    }

    private static func makeDistribution(from levels: [Float], numClasses: Int) -> [Float] {
        var dist = [Float](repeating: 0.0, count: numClasses)
        var activeSum: Float = 0.0
        for c in 1..<numClasses {
            activeSum += levels[c]
        }
        if activeSum <= 0.0 {
            dist[0] = 1.0
        } else if activeSum < 1.0 {
            dist[0] = 1.0 - activeSum
            for c in 1..<numClasses {
                dist[c] = levels[c]
            }
        } else {
            dist[0] = 0.0
            let inv = 1.0 / activeSum
            for c in 1..<numClasses {
                dist[c] = levels[c] * inv
            }
        }
        return dist
    }

    private static func argmax(_ dist: [Float]) -> Int {
        var bestIdx = 0
        var bestVal: Float = -1.0
        for c in 0..<dist.count {
            if bestVal < dist[c] {
                bestVal = dist[c]
                bestIdx = c
            }
        }
        return bestIdx
    }
}
