import XCTest
@testable import Nosuri

final class ReactionCatalogTests: XCTestCase {
    func testReactionStateSpecAndStreamEngineSustainDecay() throws {
        XCTAssertEqual(ReactionStateSpec.numBaseClasses, 39)
        XCTAssertEqual(ReactionStateSpec.idle.id, "#0")
        XCTAssertEqual(ReactionStateSpec(baseIndex: 9, tier: 1).id, "#9")
        XCTAssertEqual(ReactionStateSpec(baseIndex: 9, tier: 2).id, "#9_2")
        XCTAssertEqual(ReactionStateSpec(baseIndex: 9, tier: 3).id, "#9_3")

        let engine = ReactionStreamEngine(
            network: HawkNetwork(weights: .makeRandom())
        )
        let initial = engine.state(at: 0.0)
        XCTAssertEqual(initial.state, ReactionStateSpec.idle)
        XCTAssertEqual(initial.state.baseIndex, 0)
    }

    func testMultiFormatDatasetLoading() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("dataset_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tsvPath = tempDir.appendingPathComponent("sample.tsv")
        let tsvContent = "timestamp\ttext\tsuperChatJPY\treaction\n1.0\tなんでやねん\t0\t#3\n1.5\tスパチャ！\t5000\t#9_2\n"
        try tsvContent.write(to: tsvPath, atomically: true, encoding: .utf8)

        let events = CommentEvent.loadList(fromFile: tsvPath.path)
        XCTAssertEqual(events?.count, 2)
        XCTAssertEqual(events?[0].reactionIndex, 3)
        XCTAssertEqual(events?[1].reactionIndex, 9)
        XCTAssertEqual(events?[1].superChatJPY, 5000)
    }

    func testCommentStreamFrontEndDimensionsAndVelocity() {
        let frontEnd = CommentStreamFrontEnd(windowSeconds: 2.0)
        let ev1 = CommentEvent(timestamp: 1.0, text: "なんでそうなるのwww", superChatJPY: 0)
        let ev2 = CommentEvent(timestamp: 1.5, text: "草", superChatJPY: 5000)
        let f1 = frontEnd.extractFrame(for: ev1)
        let f2 = frontEnd.extractFrame(for: ev2)
        XCTAssertEqual(f1.count, CommentStreamFrontEnd.inputDim)
        XCTAssertEqual(f2.count, CommentStreamFrontEnd.inputDim)
        XCTAssertGreaterThan(frontEnd.currentVelocity, 0.5)
    }

    func testEmotionSlopeFilterAndCategorizedDatasetLoading() throws {
        let filter = EmotionSlopeFilter(numClasses: 39, stepGain: 0.22)

        // 1. Same user spamming 5 comments should stay in #0 (idle)
        let soloEvents = (1...5).map { i in
            CommentEvent(timestamp: Double(i) * 0.3, user: "user_solo", text: "なんでやねんw", superChatJPY: 0, reactionIndex: 10)
        }
        let soloDoms = filter.dominantTargets(events: soloEvents)
        XCTAssertTrue(soloDoms.allSatisfy { $0 == 0 })

        // 2. Multiple distinct users posting toward #10 gradually rise and cross into #10
        let crowdEvents = [
            CommentEvent(timestamp: 1.0, user: "", text: "", superChatJPY: 0, reactionIndex: 0),
            CommentEvent(timestamp: 1.8, user: "u1", text: "なんでやねんw", superChatJPY: 0, reactionIndex: 10),
            CommentEvent(timestamp: 2.3, user: "u2", text: "おいおい嘘つくなw", superChatJPY: 0, reactionIndex: 10),
            CommentEvent(timestamp: 2.7, user: "u3", text: "苦しい言い訳きたwww", superChatJPY: 0, reactionIndex: 10),
            CommentEvent(timestamp: 3.0, user: "u4", text: "目が泳いでるぞwww", superChatJPY: 0, reactionIndex: 10),
            CommentEvent(timestamp: 6.0, user: "", text: "", superChatJPY: 0, reactionIndex: 0)
        ]
        let dists = filter.filter(events: crowdEvents)
        let doms = filter.dominantTargets(events: crowdEvents)
        XCTAssertEqual(dists.count, 6)
        XCTAssertEqual(doms[0], 0)
        XCTAssertEqual(doms[1], 0)
        XCTAssertEqual(doms[2], 0)
        XCTAssertEqual(doms[3], 10)
        XCTAssertEqual(doms[4], 10)

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("split_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let aPath = tempDir.appendingPathComponent("laugh_comments.jsonl")
        let bPath = tempDir.appendingPathComponent("tsukkomi_comments.jsonl")
        try "{\"timestamp\":1.0,\"user\":\"u1\",\"text\":\"草\",\"superChatJPY\":0,\"reactionIndex\":9}\n".write(to: aPath, atomically: true, encoding: .utf8)
        try "{\"timestamp\":1.0,\"user\":\"u2\",\"text\":\"なんでやねん\",\"superChatJPY\":0,\"reactionIndex\":10}\n".write(to: bPath, atomically: true, encoding: .utf8)

        let datasets = CommentEvent.loadDatasets(from: [tempDir.path])
        XCTAssertEqual(datasets.count, 2)
    }

    func testScenarioJSONLoadingAndTimeBasedDecay() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("scenario_json_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let jsonPath = tempDir.appendingPathComponent("tease.json")
        let jsonContent = """
        [
          {
            "reaction_state": "#0->#5_2",
            "note": "複数人がポンコツいじりコメントを投稿し、#0から#5_2に変化する",
            "data": [
              {"timestamp": 1.0, "user": "u1", "superchat": 0, "text": "またポンコツムーブしてるーw"},
              {"timestamp": 1.4, "user": "u2", "superchat": 0, "text": "むすっとしてるぞw"},
              {"timestamp": 1.8, "user": "u3", "superchat": 0, "text": "ほっぺ膨らませてぷく顔になってるぞ！"},
              {"timestamp": 2.2, "user": "u4", "superchat": 0, "text": "完全に拗ねてるw"}
            ]
          },
          {
            "reaction_state": "#5_2->#5_3",
            "note": "拗ね状態からさらにいじりとスパチャが重なり#5_3へ変化する",
            "data": [
              {"timestamp": 2.6, "user": "u5", "superchat": 1000, "text": "ガチおこぷんぷん丸で草💢"},
              {"timestamp": 3.0, "user": "u6", "superchat": 0, "text": "いじりすぎてガチ拗ねモード突入w"}
            ]
          }
        ]
        """
        try jsonContent.write(to: jsonPath, atomically: true, encoding: .utf8)

        let scenarioDatasets = CommentEvent.loadScenarioDatasets(from: [tempDir.path])
        XCTAssertEqual(scenarioDatasets.count, 1)
        XCTAssertEqual(scenarioDatasets[0].scenarios.count, 2)
        XCTAssertEqual(scenarioDatasets[0].scenarios[0].fromState, ReactionStateSpec(baseIndex: 0))
        XCTAssertEqual(scenarioDatasets[0].scenarios[0].toState, ReactionStateSpec(baseIndex: 5, tier: 2))
        XCTAssertEqual(scenarioDatasets[0].scenarios[1].fromState, ReactionStateSpec(baseIndex: 5, tier: 2))
        XCTAssertEqual(scenarioDatasets[0].scenarios[1].toState, ReactionStateSpec(baseIndex: 5, tier: 3))

        let filter = EmotionSlopeFilter(numClasses: 39)
        let frontEnd = CommentStreamFrontEnd()
        let episodes = filter.buildTrainingEpisodes(scenarios: scenarioDatasets[0].scenarios, frontEnd: frontEnd)
        // 2 standalone episodes + 1 chained multi-stage episode (#0 -> #5_2 -> #5_3)
        XCTAssertEqual(episodes.count, 3)
        XCTAssertEqual(episodes[0].dominants.first, 0)
        XCTAssertEqual(episodes[0].dominants.last, 5)
        XCTAssertEqual(episodes[1].dominants.last, 5)
    }
}
