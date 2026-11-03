import XCTest
@testable import Nosuri

/// The safetensors layout: header length, padded JSON header, raw float32 data at the declared offsets.
final class SafeTensorsTests: XCTestCase {
    func testWrittenFileFollowsTheLayoutAndReadsBack() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("st_\(UUID().uuidString).safetensors")
        let a = SafeTensors.Tensor(shape: [2, 3], values: [1, 2, 3, 4, 5, 6])
        let b = SafeTensors.Tensor(shape: [1], values: [-0.5])
        try SafeTensors.write([("a", a), ("b", b)], metadata: ["note": "かな"], to: url)
        let data = try Data(contentsOf: url)
        let length = data.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        XCTAssertEqual((8 + length) % 8, 0)
        XCTAssertEqual(data.count, 8 + length + 4 * 7)
        let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + length))) as! [String: Any]
        XCTAssertEqual((header["b"] as! [String: Any])["data_offsets"] as! [Int], [24, 28])
        let last = data.subdata(in: (8 + length + 24)..<(8 + length + 28)).withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        XCTAssertEqual(last, -0.5)
        let (tensors, metadata) = try SafeTensors.read(from: url)
        XCTAssertEqual(tensors["a"], a)
        XCTAssertEqual(tensors["b"], b)
        XCTAssertEqual(metadata["note"], "かな")
    }
}

