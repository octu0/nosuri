import Foundation
import Dispatch
import Nosuri
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

private func sha1(_ data: [UInt8]) -> [UInt8] {
    var h0: UInt32 = 0x67452301
    var h1: UInt32 = 0xEFCDAB89
    var h2: UInt32 = 0x98BADCFE
    var h3: UInt32 = 0x10325476
    var h4: UInt32 = 0xC3D2E1F0

    var msg = data
    let bitLen = UInt64(msg.count) * 8
    msg.append(0x80)
    while (msg.count % 64) != 56 {
        msg.append(0x00)
    }
    for shift in stride(from: 56, through: 0, by: -8) {
        msg.append(UInt8((bitLen >> UInt64(shift)) & 0xFF))
    }

    @inline(__always)
    func rotl(_ x: UInt32, _ n: UInt32) -> UInt32 {
        return (x << n) | (x >> (32 - n))
    }

    var w = [UInt32](repeating: 0, count: 80)
    var chunkStart = 0
    while chunkStart < msg.count {
        for i in 0..<16 {
            let b = chunkStart + i * 4
            w[i] = (UInt32(msg[b]) << 24)
                 | (UInt32(msg[b + 1]) << 16)
                 | (UInt32(msg[b + 2]) << 8)
                 | UInt32(msg[b + 3])
        }
        for i in 16..<80 {
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1)
        }

        var a = h0
        var b = h1
        var c = h2
        var d = h3
        var e = h4

        for i in 0..<80 {
            let (f, k): (UInt32, UInt32) = if i < 20 {
                ((b & c) | ((~b) & d), 0x5A827999)
            } else if i < 40 {
                (b ^ c ^ d, 0x6ED9EBA1)
            } else if i < 60 {
                ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
            } else {
                (b ^ c ^ d, 0xCA62C1D6)
            }
            let temp = rotl(a, 5) &+ f &+ e &+ k &+ w[i]
            e = d
            d = c
            c = rotl(b, 30)
            b = a
            a = temp
        }

        h0 = h0 &+ a
        h1 = h1 &+ b
        h2 = h2 &+ c
        h3 = h3 &+ d
        h4 = h4 &+ e
        chunkStart += 64
    }

    var digest: [UInt8] = []
    digest.reserveCapacity(20)
    for h in [h0, h1, h2, h3, h4] {
        digest.append(UInt8((h >> 24) & 0xFF))
        digest.append(UInt8((h >> 16) & 0xFF))
        digest.append(UInt8((h >> 8) & 0xFF))
        digest.append(UInt8(h & 0xFF))
    }
    return digest
}

private func webSocketAcceptKey(for clientKey: String) -> String {
    let magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    let combined = Array((clientKey.trimmingCharacters(in: .whitespacesAndNewlines) + magic).utf8)
    return Data(sha1(combined)).base64EncodedString()
}

private func sendAll(_ fd: Int32, _ bytes: [UInt8]) -> Bool {
    var sent = 0
    while sent < bytes.count {
        let n = bytes.withUnsafeBufferPointer { buf -> Int in
            guard let base = buf.baseAddress else { return -1 }
            #if canImport(Glibc)
            return Glibc.send(fd, base + sent, bytes.count - sent, Int32(MSG_NOSIGNAL))
            #else
            return Darwin.send(fd, base + sent, bytes.count - sent, 0)
            #endif
        }
        if n <= 0 {
            return false
        }
        sent += n
    }
    return true
}

private func recvExact(_ fd: Int32, count: Int) -> [UInt8]? {
    if count == 0 { return [] }
    var buf = [UInt8](repeating: 0, count: count)
    var readTotal = 0
    while readTotal < count {
        let n = buf.withUnsafeMutableBufferPointer { ptr -> Int in
            guard let base = ptr.baseAddress else { return -1 }
            return recv(fd, base + readTotal, count - readTotal, 0)
        }
        if n <= 0 {
            return nil
        }
        readTotal += n
    }
    return buf
}

private func sendWebSocketText(_ fd: Int32, _ text: String) -> Bool {
    let payload = Array(text.utf8)
    var frame: [UInt8] = [0x81]
    if payload.count < 126 {
        frame.append(UInt8(payload.count))
    } else if payload.count <= 0xFFFF {
        frame.append(126)
        frame.append(UInt8((payload.count >> 8) & 0xFF))
        frame.append(UInt8(payload.count & 0xFF))
    } else {
        frame.append(127)
        let len = UInt64(payload.count)
        for shift in stride(from: 56, through: 0, by: -8) {
            frame.append(UInt8((len >> UInt64(shift)) & 0xFF))
        }
    }
    frame.append(contentsOf: payload)
    return sendAll(fd, frame)
}

private func sendWebSocketPong(_ fd: Int32, payload: [UInt8]) -> Bool {
    var frame: [UInt8] = [0x8A, UInt8(min(125, payload.count))]
    frame.append(contentsOf: payload.prefix(125))
    return sendAll(fd, frame)
}

private struct IncomingWSMessage: Decodable {
    let type: String
    let user: String?
    let text: String?
    let superChatJPY: Int?
    let deltaSeconds: Double?
}

private struct ClipDTO: Encodable {
    let id: String
    let baseIndex: Int
    let tier: String
    let filename: String
    let videoUrl: String

    init(_ clip: ReactionClip) {
        self.id = clip.id
        self.baseIndex = clip.baseIndex
        if clip.baseIndex == 0 {
            self.tier = "idle"
        } else if clip.tier >= 2 {
            self.tier = "escalated"
        } else {
            self.tier = "normal"
        }
        self.filename = clip.filename
        let encodedName = clip.filename.replacingOccurrences(of: "#", with: "%23")
        self.videoUrl = "/resources/\(encodedName)"
    }
}

private struct CandidateDTO: Encodable {
    let baseIndex: Int
    let id: String
    let probability: Float
}

private struct OutgoingWSMessage: Encodable {
    let type: String
    let sessionId: String
    let timestamp: Double
    let user: String
    let commentText: String
    let superChatJPY: Int
    let isIdleTick: Bool
    let escalated: Bool
    let phase: String
    let intensity: Float
    let velocity: Float
    let uniqueUsers: Int
    let inferenceMs: Double
    let clip: ClipDTO
    let baseClip: ClipDTO
    let topCandidates: [CandidateDTO]
}

private struct CatalogResponseDTO: Encodable {
    let status: String
    let modelParameters: Int
    let numBaseClasses: Int
    let totalVideoClips: Int
    let clips: [ClipDTO]
}

var port: UInt16 = UInt16(ProcessInfo.processInfo.environment["PORT"] ?? "8080") ?? 8080
var weightsPath = ProcessInfo.processInfo.environment["WEIGHTS_PATH"] ?? ".tmp/weights.safetensors"
var resourcesDir = ProcessInfo.processInfo.environment["RESOURCES_DIR"] ?? ReactionCatalog.defaultResourcesDirectory
var webDir = ProcessInfo.processInfo.environment["WEB_DIR"] ?? ".tmp/public"

var argIdx = 1
let args = CommandLine.arguments
while argIdx < args.count {
    let arg = args[argIdx]
    switch arg {
    case "-h", "--help":
        print("Usage: server [-p <port>] [-w <weights.safetensors>] [--resources <dir>] [--web <dir>]")
        exit(0)
    case "-p", "--port":
        if (argIdx + 1) < args.count, let v = UInt16(args[argIdx + 1]) {
            port = v
            argIdx += 1
        }
    case "-w", "--weights":
        if (argIdx + 1) < args.count {
            weightsPath = args[argIdx + 1]
            argIdx += 1
        }
    case "--resources":
        if (argIdx + 1) < args.count {
            resourcesDir = args[argIdx + 1]
            argIdx += 1
        }
    case "--web":
        if (argIdx + 1) < args.count {
            webDir = args[argIdx + 1]
            argIdx += 1
        }
    default:
        break
    }
    argIdx += 1
}

let modelWeights: HawkWeights = if FileManager.default.fileExists(atPath: weightsPath),
   let loaded = try? HawkWeights.load(from: URL(fileURLWithPath: weightsPath)) {
    loaded
} else {
    HawkWeights.makeRandom()
}
print("✓ Hawk weights ready (\(modelWeights.parameterCount) parameters)")

let sharedNetwork = HawkNetwork(weights: modelWeights)
let resolvedResourcesDir = resourcesDir
let resolvedWebDir = webDir
let serverCatalog = ReactionCatalog(directory: resolvedResourcesDir)

private func sendHTTPResponse(
    _ fd: Int32,
    status: String,
    contentType: String,
    extraHeaders: [String] = [],
    body: [UInt8]
) {
    var header = "HTTP/1.1 \(status)\r\n"
    header += "Content-Type: \(contentType)\r\n"
    header += "Content-Length: \(body.count)\r\n"
    header += "Access-Control-Allow-Origin: *\r\n"
    for h in extraHeaders {
        header += "\(h)\r\n"
    }
    header += "Connection: close\r\n\r\n"
    _ = sendAll(fd, Array(header.utf8))
    if body.isEmpty != true {
        _ = sendAll(fd, body)
    }
}

private func handleWebSocketSession(fd: Int32) {
    let sessionId = String(UUID().uuidString.prefix(8))
    let engine = ReactionStreamEngine(network: sharedNetwork, catalog: serverCatalog)
    var streamTime: Double = 0.0
    let encoder = JSONEncoder()
    let decoder = JSONDecoder()

    func makeMessage(event: CommentEvent, isIdle: Bool, pred: ReactionPrediction, elapsedMs: Double) -> OutgoingWSMessage {
        return OutgoingWSMessage(
            type: "reaction",
            sessionId: sessionId,
            timestamp: streamTime,
            user: event.user,
            commentText: event.text,
            superChatJPY: event.superChatJPY,
            isIdleTick: isIdle,
            escalated: pred.escalated,
            phase: pred.phase,
            intensity: pred.intensity,
            velocity: pred.velocity,
            uniqueUsers: pred.uniqueUsers,
            inferenceMs: elapsedMs,
            clip: ClipDTO(pred.clip),
            baseClip: ClipDTO(pred.baseClip),
            topCandidates: pred.topCandidates.map {
                CandidateDTO(baseIndex: $0.baseIndex, id: $0.clip.id, probability: $0.probability)
            }
        )
    }

    let initEvent = CommentEvent(timestamp: 0.0, text: "", superChatJPY: 0)
    let initPred = engine.pushDetailed(initEvent)
    engine.reset()
    if let data = try? encoder.encode(makeMessage(event: initEvent, isIdle: true, pred: initPred, elapsedMs: 0.1)),
       let str = String(data: data, encoding: .utf8) {
        _ = sendWebSocketText(fd, str)
    }

    while true {
        guard let header = recvExact(fd, count: 2) else { break }
        let opcode = header[0] & 0x0F
        let masked = (header[1] & 0x80) != 0
        var payloadLen = Int(header[1] & 0x7F)

        if payloadLen == 126 {
            guard let ext = recvExact(fd, count: 2) else { break }
            payloadLen = (Int(ext[0]) << 8) | Int(ext[1])
        } else if payloadLen == 127 {
            guard let ext = recvExact(fd, count: 8) else { break }
            var len64: UInt64 = 0
            for b in ext {
                len64 = (len64 << 8) | UInt64(b)
            }
            if len64 > 1_048_576 { break }
            payloadLen = Int(len64)
        }

        var maskKey: [UInt8] = [0, 0, 0, 0]
        if masked {
            guard let m = recvExact(fd, count: 4) else { break }
            maskKey = m
        }

        guard var payload = recvExact(fd, count: payloadLen) else { break }
        if masked {
            for i in 0..<payload.count {
                payload[i] ^= maskKey[i & 3]
            }
        }

        if opcode == 0x08 {
            break
        } else if opcode == 0x09 {
            if sendWebSocketPong(fd, payload: payload) != true { break }
            continue
        } else if opcode != 0x01 {
            continue
        }

        guard let text = String(bytes: payload, encoding: .utf8),
              let incoming = try? decoder.decode(IncomingWSMessage.self, from: Data(text.utf8)) else {
            continue
        }

        if incoming.type == "reset" {
            engine.reset()
            streamTime = 0.0
            let resetEvent = CommentEvent(timestamp: 0.0, text: "", superChatJPY: 0)
            let pred = engine.pushDetailed(resetEvent)
            engine.reset()
            if let data = try? encoder.encode(makeMessage(event: resetEvent, isIdle: true, pred: pred, elapsedMs: 0.1)),
               let str = String(data: data, encoding: .utf8) {
                if sendWebSocketText(fd, str) != true { break }
            }
            continue
        }

        let isIdle = (incoming.type == "idle_tick") || ((incoming.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && (incoming.superChatJPY ?? 0) <= 0)
        let dt = incoming.deltaSeconds ?? (isIdle ? 2.5 : 0.35)
        streamTime += max(0.05, dt)

        let event = CommentEvent(
            timestamp: streamTime,
            user: isIdle ? "" : (incoming.user ?? ""),
            text: isIdle ? "" : (incoming.text ?? ""),
            superChatJPY: max(0, incoming.superChatJPY ?? 0)
        )

        let t0 = Date().timeIntervalSinceReferenceDate
        let pred = engine.pushDetailed(event, topK: 5)
        let elapsedMs = (Date().timeIntervalSinceReferenceDate - t0) * 1000.0

        if let data = try? encoder.encode(makeMessage(event: event, isIdle: isIdle, pred: pred, elapsedMs: elapsedMs)),
           let str = String(data: data, encoding: .utf8) {
            if sendWebSocketText(fd, str) != true { break }
        }
    }
}

private func handleClientConnection(fd: Int32) {
    defer { _ = close(fd) }

    var requestBytes: [UInt8] = []
    var chunk = [UInt8](repeating: 0, count: 4096)
    while requestBytes.count < 16384 {
        let n = chunk.withUnsafeMutableBufferPointer { ptr -> Int in
            guard let base = ptr.baseAddress else { return -1 }
            return recv(fd, base, 4096, 0)
        }
        if n <= 0 { return }
        requestBytes.append(contentsOf: chunk[0..<n])
        if let str = String(bytes: requestBytes, encoding: .utf8), str.contains("\r\n\r\n") {
            break
        }
    }

    guard let requestString = String(bytes: requestBytes, encoding: .utf8) else { return }
    let lines = requestString.components(separatedBy: "\r\n")
    guard let requestLine = lines.first else { return }
    let parts = requestLine.split(separator: " ")
    guard parts.count >= 2 else { return }
    let method = String(parts[0])
    let rawPath = String(parts[1])
    let pathOnly = rawPath.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? rawPath

    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
        if line.isEmpty { break }
        if let colon = line.firstIndex(of: ":") {
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let val = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[key] = val
        }
    }

    if pathOnly == "/ws",
       let wsKey = headers["sec-websocket-key"],
       (headers["upgrade"]?.lowercased().contains("websocket") ?? false) {
        let accept = webSocketAcceptKey(for: wsKey)
        let resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: \(accept)\r\n\r\n"
        if sendAll(fd, Array(resp.utf8)) {
            handleWebSocketSession(fd: fd)
        }
        return
    }

    if method != "GET" && method != "HEAD" {
        sendHTTPResponse(fd, status: "405 Method Not Allowed", contentType: "text/plain; charset=utf-8", body: Array("Method Not Allowed".utf8))
        return
    }

    if pathOnly == "/healthz" {
        sendHTTPResponse(fd, status: "200 OK", contentType: "application/json; charset=utf-8", body: Array("{\"status\":\"ok\"}".utf8))
        return
    }

    if pathOnly == "/" || pathOnly == "/index.html" {
        let indexPath = "\(resolvedWebDir)/index.html"
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: indexPath)) else {
            sendHTTPResponse(fd, status: "404 Not Found", contentType: "text/plain; charset=utf-8", body: Array("index.html not found".utf8))
            return
        }
        sendHTTPResponse(fd, status: "200 OK", contentType: "text/html; charset=utf-8", body: [UInt8](data))
        return
    }

    if pathOnly == "/assets/nosuri-logo.svg" {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: "assets/nosuri-logo.svg")) else {
            sendHTTPResponse(fd, status: "404 Not Found", contentType: "text/plain; charset=utf-8", body: Array("logo not found".utf8))
            return
        }
        sendHTTPResponse(fd, status: "200 OK", contentType: "image/svg+xml", body: [UInt8](data))
        return
    }

    if pathOnly == "/api/catalog" {
        let dto = CatalogResponseDTO(
            status: "ok",
            modelParameters: modelWeights.parameterCount,
            numBaseClasses: ReactionCatalog.numBaseClasses,
            totalVideoClips: serverCatalog.allClips.count,
            clips: serverCatalog.allClips.map { ClipDTO($0) }
        )
        let data = (try? JSONEncoder().encode(dto)) ?? Data("{}".utf8)
        sendHTTPResponse(fd, status: "200 OK", contentType: "application/json; charset=utf-8", body: [UInt8](data))
        return
    }

    if pathOnly.hasPrefix("/resources/") {
        let rawFilename = String(pathOnly.dropFirst("/resources/".count))
        let decodedFilename = rawFilename.removingPercentEncoding ?? rawFilename
        if decodedFilename.contains("..") || decodedFilename.contains("/") || decodedFilename.contains("\\") {
            sendHTTPResponse(fd, status: "403 Forbidden", contentType: "text/plain; charset=utf-8", body: Array("Forbidden".utf8))
            return
        }
        let filePath = "\(resolvedResourcesDir)/\(decodedFilename)"
        guard let fileData = try? Data(contentsOf: URL(fileURLWithPath: filePath)) else {
            sendHTTPResponse(fd, status: "404 Not Found", contentType: "text/plain; charset=utf-8", body: Array("Video not found: \(decodedFilename)".utf8))
            return
        }
        let total = fileData.count
        if let rangeHeader = headers["range"], rangeHeader.hasPrefix("bytes=") {
            let spec = String(rangeHeader.dropFirst("bytes=".count))
            let bounds = spec.split(separator: "-", omittingEmptySubsequences: false)
            let start = (bounds.first.flatMap { Int($0) }) ?? 0
            let end = (bounds.count > 1 && bounds[1].isEmpty != true) ? (Int(bounds[1]) ?? (total - 1)) : (total - 1)
            let clampedStart = max(0, min(start, max(0, total - 1)))
            let clampedEnd = max(clampedStart, min(end, max(0, total - 1)))
            let slice = [UInt8](fileData[clampedStart...clampedEnd])
            sendHTTPResponse(
                fd,
                status: "206 Partial Content",
                contentType: "video/mp4",
                extraHeaders: [
                    "Accept-Ranges: bytes",
                    "Content-Range: bytes \(clampedStart)-\(clampedEnd)/\(total)",
                    "Cache-Control: public, max-age=3600"
                ],
                body: method == "HEAD" ? [] : slice
            )
            return
        }

        sendHTTPResponse(
            fd,
            status: "200 OK",
            contentType: "video/mp4",
            extraHeaders: [
                "Accept-Ranges: bytes",
                "Cache-Control: public, max-age=3600"
            ],
            body: method == "HEAD" ? [] : [UInt8](fileData)
        )
        return
    }

    sendHTTPResponse(fd, status: "404 Not Found", contentType: "text/plain; charset=utf-8", body: Array("Not Found".utf8))
}

#if canImport(Glibc)
signal(SIGPIPE, SIG_IGN)
let serverFD = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
#else
signal(SIGPIPE, SIG_IGN)
let serverFD = socket(AF_INET, SOCK_STREAM, 0)
#endif

if serverFD < 0 {
    perror("socket")
    exit(1)
}

var reuse: Int32 = 1
setsockopt(serverFD, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

var addr = sockaddr_in()
addr.sin_family = sa_family_t(AF_INET)
addr.sin_port = port.bigEndian
addr.sin_addr = in_addr(s_addr: INADDR_ANY.bigEndian)

let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
    return ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
        return bind(serverFD, saPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
    }
}

if bindResult < 0 {
    perror("bind")
    exit(1)
}

if listen(serverFD, 128) < 0 {
    perror("listen")
    exit(1)
}

print("Listening on http://0.0.0.0:\(port) (ws://0.0.0.0:\(port)/ws)")

while true {
    var clientAddr = sockaddr_in()
    var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
    let clientFD = withUnsafeMutablePointer(to: &clientAddr) { ptr -> Int32 in
        return ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { saPtr in
            return accept(serverFD, saPtr, &clientLen)
        }
    }
    if clientFD < 0 {
        continue
    }
    DispatchQueue.global(qos: .userInitiated).async {
        handleClientConnection(fd: clientFD)
    }
}
