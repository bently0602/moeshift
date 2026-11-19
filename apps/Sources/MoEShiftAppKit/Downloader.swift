import CryptoKit
import Foundation

/// Downloads one file straight into its destination folder, hashing it as it arrives.
///
/// URLSession's download tasks write to a temporary file on the startup disk first, which would
/// write every model file to the internal SSD once more; a data task streams to `<name>.part` in
/// the chosen folder instead. A stopped download resumes from the bytes already there (they are
/// hashed again first), and the file gets its final name only when its size and SHA-256 match.
final class Downloader: NSObject, URLSessionDataDelegate {
    struct Progress {
        var done: Int64
        var total: Int64
        var bytesPerSecond: Double
    }

    private let url: URL
    private let destination: URL
    private let size: Int64
    private let sha256: String
    private let onProgress: (Progress) -> Void
    private var partURL: URL { destination.appendingPathExtension("part") }

    private var handle: FileHandle?
    private var hasher = SHA256()
    private var offset: Int64 = 0
    private var startOffset: Int64 = 0
    private var startTime = Date()
    private var lastReport = Date.distantPast
    private var continuation: CheckedContinuation<Void, Error>?
    private var session: URLSession?
    private var failure: Error?

    init(url: URL, destination: URL, size: Int64, sha256: String, onProgress: @escaping (Progress) -> Void) {
        self.url = url
        self.destination = destination
        self.size = size
        self.sha256 = sha256
        self.onProgress = onProgress
    }

    func run() async throws {
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: partURL.path) {
            FileManager.default.createFile(atPath: partURL.path, contents: nil)
        }
        let handle = try FileHandle(forUpdating: partURL)
        self.handle = handle
        offset = try rehashExisting(handle)
        try handle.seek(toOffset: UInt64(offset))
        if offset < size {
            try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    self.continuation = continuation
                    let queue = OperationQueue()
                    queue.maxConcurrentOperationCount = 1
                    let session = URLSession(configuration: .ephemeral, delegate: self, delegateQueue: queue)
                    self.session = session
                    var request = URLRequest(url: url)
                    if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
                    startOffset = offset
                    startTime = Date()
                    session.dataTask(with: request).resume()
                }
            } onCancel: {
                self.session?.invalidateAndCancel()
            }
        }
        try handle.synchronize()
        try handle.close()
        self.handle = nil
        guard offset == size else { throw AppError("\(destination.lastPathComponent): got \(offset) of \(size) bytes.") }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard digest == sha256 else {
            try? FileManager.default.removeItem(at: partURL)
            throw AppError("\(destination.lastPathComponent) failed its checksum and was removed; download it again.")
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partURL, to: destination)
    }

    /// Hashes the bytes a stopped download left, so the final checksum covers the whole file.
    private func rehashExisting(_ handle: FileHandle) throws -> Int64 {
        let existing = Int64(try handle.seekToEnd())
        guard existing > 0 else { return 0 }
        if existing > size {
            try handle.truncate(atOffset: 0)
            return 0
        }
        try handle.seek(toOffset: 0)
        var done: Int64 = 0
        while done < existing {
            let chunk = try handle.read(upToCount: 8 << 20) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
            done += Int64(chunk.count)
            if Date().timeIntervalSince(lastReport) > 0.25 {
                lastReport = Date()
                onProgress(Progress(done: done, total: size, bytesPerSecond: 0))
            }
        }
        return done
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status == 206 {
            completionHandler(.allow)
        } else if status == 200 {
            // The server ignored the range: start over.
            do {
                try handle?.truncate(atOffset: 0)
                try handle?.seek(toOffset: 0)
                hasher = SHA256()
                offset = 0
                startOffset = 0
                completionHandler(.allow)
            } catch {
                failure = error
                completionHandler(.cancel)
            }
        } else {
            failure = AppError("Hugging Face answered HTTP \(status) for \(destination.lastPathComponent).")
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle?.write(contentsOf: data)
        } catch {
            failure = error
            dataTask.cancel()
            return
        }
        hasher.update(data: data)
        offset += Int64(data.count)
        if Date().timeIntervalSince(lastReport) > 0.25 {
            lastReport = Date()
            let elapsed = max(Date().timeIntervalSince(startTime), 0.001)
            onProgress(Progress(done: offset, total: size, bytesPerSecond: Double(offset - startOffset) / elapsed))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        session.finishTasksAndInvalidate()
        if let failure {
            continuation?.resume(throwing: failure)
        } else if let error {
            continuation?.resume(throwing: (error as NSError).code == NSURLErrorCancelled ? CancellationError() : error)
        } else {
            continuation?.resume()
        }
        continuation = nil
    }
}
