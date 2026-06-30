import Foundation

enum ClientError: Error, CustomStringConvertible {
    case http(Int)
    case server(Int, String)  // carries the backend's `detail` so the UI can show why
    case badURL

    var description: String {
        switch self {
        case .http(let code): return "HTTP \(code)"
        case .server(_, let detail): return detail
        case .badURL: return "bad URL"
        }
    }
}

/// Thin async client over the Python backend's HTTP API. Mirrors omlx's app↔server
/// transport: REST for commands, SSE for streaming chat.
struct AssistantClient {
    var baseURL: URL
    private let session: URLSession = .shared

    // --- REST ---

    func status() async throws -> StatusDTO { try await get("/status") }
    func models() async throws -> ModelsDTO { try await get("/models") }
    func skills() async throws -> SkillsDTO { try await get("/skills") }
    func skill(_ name: String) async throws -> SkillBodyDTO { try await get("/skills/\(name)") }
    func memory() async throws -> MemoryListDTO { try await get("/memory") }

    func searchMemory(_ query: String) async throws -> MemorySearchDTO {
        let q = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        return try await get("/memory/search?q=\(q)")
    }

    func defaultModel() async throws -> DefaultModelDTO { try await get("/models/default") }
    func setDefaultModel(_ id: String) async throws {
        try await sendJSONNoDecode("PUT", "/models/default", body: ["model": id])
    }
    func fusion() async throws -> FusionConfigDTO { try await get("/fusion") }
    func setFusion(enabled: Bool, panel: [String], judge: String) async throws {
        try await sendJSONNoDecode(
            "PUT", "/fusion", body: ["enabled": enabled, "panel": panel, "judge": judge]
        )
    }
    func modelSettings(_ id: String) async throws -> ModelSettingsDTO {
        try await get("/models/\(id)/settings")
    }
    func setModelSettings(_ id: String, _ settings: [String: Any]) async throws {
        try await sendJSONNoDecode("PUT", "/models/\(id)/settings", body: ["settings": settings])
    }
    func loadModel(_ id: String) async throws { try await post("/models/\(id)/load") }
    func unloadModel(_ id: String) async throws { try await post("/models/\(id)/unload") }
    func deleteModel(_ id: String) async throws {
        try await sendJSONNoDecode("DELETE", "/models/\(id)", body: [:])
    }
    func reloadSkills() async throws -> ReloadDTO { try await postDecoding("/skills/reload") }

    func createSkill(name: String, description: String, body: String) async throws {
        try await sendJSONNoDecode(
            "POST", "/skills",
            body: ["name": name, "description": description, "body": body]
        )
    }

    func updateSkill(_ name: String, description: String, body: String) async throws {
        try await sendJSONNoDecode(
            "PUT", "/skills/\(name)", body: ["description": description, "body": body]
        )
    }

    func deleteSkill(_ name: String) async throws {
        try await sendJSONNoDecode("DELETE", "/skills/\(name)", body: [:])
    }

    func importSkill(content: String) async throws {
        try await sendJSONNoDecode("POST", "/skills/import", body: ["content": content])
    }

    func createMemory(content: String, tags: [String]) async throws {
        try await sendJSONNoDecode("POST", "/memory", body: ["content": content, "tags": tags])
    }

    func updateMemory(_ id: String, content: String, tags: [String]) async throws {
        try await sendJSONNoDecode("PUT", "/memory/\(id)", body: ["content": content, "tags": tags])
    }

    func deleteMemory(_ id: String) async throws {
        try await sendJSONNoDecode("DELETE", "/memory/\(id)", body: [:])
    }

    func generateImage(prompt: String) async throws -> ImageResultDTO {
        try await postJSON("/images/generate", body: ["prompt": prompt])
    }

    func downloads() async throws -> DownloadsDTO { try await get("/downloads") }

    func startDownload(repoId: String) async throws {
        try await postJSONNoDecode("/models/download", body: ["repo_id": repoId])
    }

    func cancelDownload(repoId: String) async throws {
        try await postJSONNoDecode("/models/download/cancel", body: ["repo_id": repoId])
    }

    func retryDownload(repoId: String) async throws {
        try await postJSONNoDecode("/models/download/retry", body: ["repo_id": repoId])
    }

    func removeDownload(repoId: String) async throws {
        try await postJSONNoDecode("/models/download/remove", body: ["repo_id": repoId])
    }

    func approveTool(token: String, decision: Bool) async throws {
        try await postJSONNoDecode("/chat/approve", body: ["token": token, "decision": decision])
    }

    func preflight() async throws -> PreflightDTO { try await get("/preflight") }

    func listSessions() async throws -> SessionListDTO { try await get("/sessions") }
    func createSession(model: String?) async throws -> SessionSummaryDTO {
        var body: [String: Any] = [:]
        if let model { body["model"] = model }
        return try await postJSON("/sessions", body: body)
    }
    func sessionDetail(_ id: String) async throws -> SessionDetailDTO {
        try await get("/sessions/\(id)")
    }
    func deleteSession(_ id: String) async throws {
        try await sendJSONNoDecode("DELETE", "/sessions/\(id)", body: [:])
    }

    func installTool(feature: String, upgrade: Bool = false) async throws {
        try await postJSONNoDecode("/setup/install", body: ["feature": feature, "upgrade": upgrade])
    }

    func getConfig() async throws -> ConfigDTO { try await get("/config") }

    func putConfig(
        modelsDir: String? = nil, downloadDir: String? = nil,
        extraModelDirs: [String]? = nil, hfCache: Bool? = nil,
        backendHost: String? = nil, backendPort: Int? = nil,
        modelBackend: String? = nil, maxOutputTokens: Int? = nil,
        maxToolIters: Int? = nil, turnTimeoutS: Double? = nil, memCeilingGb: Double? = nil,
        telegramToken: String? = nil, telegramAllowedUsers: [Int]? = nil
    ) async throws {
        var body: [String: Any] = [:]
        if let modelsDir { body["models_dir"] = modelsDir }
        if let downloadDir { body["download_dir"] = downloadDir }
        if let extraModelDirs { body["extra_model_dirs"] = extraModelDirs }
        if let hfCache { body["hf_cache"] = hfCache }
        if let backendHost { body["backend_host"] = backendHost }
        if let backendPort { body["backend_port"] = backendPort }
        if let modelBackend { body["model_backend"] = modelBackend }
        if let maxOutputTokens { body["max_output_tokens"] = maxOutputTokens }
        if let maxToolIters { body["max_tool_iters"] = maxToolIters }
        if let turnTimeoutS { body["turn_timeout_s"] = turnTimeoutS }
        if let memCeilingGb { body["mem_ceiling_gb"] = memCeilingGb }
        if let telegramToken { body["telegram_token"] = telegramToken }
        if let telegramAllowedUsers { body["telegram_allowed_users"] = telegramAllowedUsers }
        try await sendJSONNoDecode("PUT", "/config", body: body)
    }

    // --- streaming chat (SSE) ---

    func chat(
        message: String, model: String, sessionId: String?, interactiveApproval: Bool = true
    ) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: url("/chat"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    var body: [String: Any] = [
                        "message": message, "model": model,
                        "interactive_approval": interactiveApproval,
                    ]
                    if let sessionId { body["session_id"] = sessionId }
                    request.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await session.bytes(for: request)
                    if let http = response as? HTTPURLResponse, http.statusCode >= 400 {
                        throw ClientError.http(http.statusCode)
                    }
                    for try await line in bytes.lines {
                        if let event = AssistantClient.parseSSELine(line) {
                            continuation.yield(event)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // --- helpers ---

    func url(_ path: String) -> URL {
        let base = baseURL.absoluteString.hasSuffix("/")
            ? String(baseURL.absoluteString.dropLast())
            : baseURL.absoluteString
        return URL(string: base + path) ?? baseURL
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let (data, response) = try await session.data(from: url(path))
        try check(data, response)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func post(_ path: String) async throws {
        var request = URLRequest(url: url(path))
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        try check(data, response)
    }

    private func postDecoding<T: Decodable>(_ path: String) async throws -> T {
        var request = URLRequest(url: url(path))
        request.httpMethod = "POST"
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func postJSON<T: Decodable>(_ path: String, body: [String: Any]) async throws -> T {
        var request = URLRequest(url: url(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        try check(data, response)
        return try JSONDecoder().decode(T.self, from: data)
    }

    private func postJSONNoDecode(_ path: String, body: [String: Any]) async throws {
        try await sendJSONNoDecode("POST", path, body: body)
    }

    private func sendJSONNoDecode(_ method: String, _ path: String, body: [String: Any]) async throws {
        var request = URLRequest(url: url(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await session.data(for: request)
        try check(data, response)
    }

    /// Surface the backend's JSON `detail` on 4xx/5xx so the UI shows *why* (e.g. a
    /// non-chat model rejected by load), not just an opaque status code.
    private func check(_ data: Data, _ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        if let error = AssistantClient.mapError(statusCode: http.statusCode, data: data) {
            throw error
        }
    }

    /// Map an HTTP status + body to a `ClientError`, or nil when it isn't an error (< 400).
    /// Prefers the backend's JSON `detail`. Factored out of `check` for unit testing.
    static func mapError(statusCode: Int, data: Data) -> ClientError? {
        guard statusCode >= 400 else { return nil }
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let detail = obj["detail"] as? String, !detail.isEmpty {
            return .server(statusCode, detail)
        }
        return .http(statusCode)
    }

    /// Parse one SSE line (`data: {json}`) into a `ChatEvent`. Returns nil for non-data
    /// lines, empty payloads, or undecodable JSON. Factored out of `chat` for unit testing.
    static func parseSSELine(_ line: String) -> ChatEvent? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        guard !payload.isEmpty, let data = payload.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(ChatEvent.self, from: data)
    }
}
