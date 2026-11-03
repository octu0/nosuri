import XCTest
@testable import Nosuri

final class ReactionCatalogTests: XCTestCase {
    func testDynamicDirectoryDiscoveryAndEscalation() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("catalog_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        for name in ["#0_idleA.mp4", "#0_2_idleB.mp4", "#9_normal.mp4", "#9_2_escalated.mp4"] {
            _ = FileManager.default.createFile(atPath: tempDir.appendingPathComponent(name).path, contents: Data())
        }

        let catalog = ReactionCatalog(directory: tempDir.path)
        XCTAssertEqual(catalog.baseReactions.count, 39)
        XCTAssertEqual(catalog.idleClips.count, 2)

        let base9 = catalog.resolveClip(baseIndex: 9, commentVelocity: 0.4, superChatJPY: 0)
        XCTAssertEqual(base9.id, "#9")
        XCTAssertEqual(base9.filename, "#9_normal.mp4")

        let fast9 = catalog.resolveClip(baseIndex: 9, commentVelocity: 1.5, superChatJPY: 0)
        XCTAssertEqual(fast9.id, "#9_2")
        XCTAssertEqual(fast9.filename, "#9_2_escalated.mp4")

        let superChat9 = catalog.resolveClip(baseIndex: 9, commentVelocity: 0.2, superChatJPY: 10000)
        XCTAssertEqual(superChat9.id, "#9_2")

        let idle0 = catalog.resolveClip(baseIndex: 0, commentVelocity: 0.0, idleStep: 0)
        let idle1 = catalog.resolveClip(baseIndex: 0, commentVelocity: 0.0, idleStep: 1)
        XCTAssertEqual(idle0.baseIndex, 0)
        XCTAssertEqual(idle1.baseIndex, 0)
        XCTAssertNotEqual(idle0.filename, idle1.filename)
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
}
