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
        numClasses: Int = ReactionCatalog.numBaseClasses,
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
        return filter(events: events).map { dist in
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
}
