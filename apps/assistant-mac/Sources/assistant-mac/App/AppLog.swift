import Foundation

/// Minimal append-only file logger for the app (Swift) side.
///
/// The backend writes backend.log / backend.out.log, but the app itself only ever used NSLog →
/// Console.app — so there was no app-side log a user could `cat`. This writes timestamped lines to
/// `logs/app.log`, next to the backend's own logs. Best-effort: writes happen off the main thread
/// and every failure is swallowed, so logging can never block or crash the UI.
enum AppLog {
    private static let queue = DispatchQueue(label: "assistant.app.log")
    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        return f
    }()

    static func log(_ message: String) {
        let now = Date()
        // Format + write on the serial queue so the (non-thread-safe) DateFormatter and the file
        // handle are only ever touched from one thread.
        queue.async {
            let line = "\(formatter.string(from: now)) app: \(message)\n"
            guard let data = line.data(using: .utf8) else { return }
            let dir = Bootstrap.logsDir()
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let url = dir.appendingPathComponent("app.log")
            if let handle = try? FileHandle(forWritingTo: url) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
            } else {
                try? data.write(to: url)  // first write creates the file
            }
        }
    }
}
