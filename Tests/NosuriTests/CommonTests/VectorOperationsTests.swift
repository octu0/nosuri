import XCTest
@testable import Nosuri

final class VectorOperationsTests: XCTestCase {
    func testDotProductEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            var b = [Float](repeating: 0.0, count: count)
            var i = 0
            while i < count {
                a[i] = sin(Float(i) * 0.1) * 2.5
                b[i] = cos(Float(i) * 0.2) * 1.8
                i += 1
            }
            var scalarSum: Float = 0.0
            i = 0
            while i < count {
                scalarSum += a[i] * b[i]
                i += 1
            }
            var simdSum: Float = 0.0
            if 0 < count {
                simdSum = a.withUnsafeBufferPointer { aPtr in
                    b.withUnsafeBufferPointer { bPtr in
                        VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
                    }
                }
            }
            let diff = abs(simdSum - scalarSum)
            let maxVal = max(abs(scalarSum), 1.0)
            let relError = diff / maxVal
            XCTAssertLessThan(relError, 1e-4)
            tIdx += 1
        }
    }

    func testSumOfSquaresEquivalence() {
        let testCounts = [0, 1, 7, 8, 9, 15, 16, 17, 319, 320, 321, 512, 1024]
        var tIdx = 0
        while tIdx < testCounts.count {
            let count = testCounts[tIdx]
            var a = [Float](repeating: 0.0, count: count)
            var i = 0
            while i < count {
                a[i] = sin(Float(i) * 0.3) * 1.5
                i += 1
            }
            var scalarSum: Float = 0.0
            i = 0
            while i < count {
                scalarSum += a[i] * a[i]
                i += 1
            }
            var simdSum: Float = 0.0
            if 0 < count {
                simdSum = a.withUnsafeBufferPointer { aPtr in
                    VectorOperations.sumOfSquares(ptr: aPtr.baseAddress!, count: count)
                }
            }
            let diff = abs(simdSum - scalarSum)
            let maxVal = max(abs(scalarSum), 1.0)
            let relError = diff / maxVal
            XCTAssertLessThan(relError, 1e-4)
            tIdx += 1
        }
    }
}

