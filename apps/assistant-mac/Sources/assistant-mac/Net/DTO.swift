import Foundation

// Codable mirrors of the backend's JSON. snake_case keys are mapped explicitly.

struct StatusDTO: Decodable {
    let backend: String
    let modelBackend: String?
    let omlx: OmlxStatusDTO

    enum CodingKeys: String, CodingKey {
        case backend, omlx
        case modelBackend = "model_backend"
    }
}

struct OmlxStatusDTO: Decodable {
    let state: String
    let detail: String
    let baseURL: String?
    let reachable: Bool

    enum CodingKeys: String, CodingKey {
        case state, detail, reachable
        case baseURL = "base_url"
    }
}

struct ModelDTO: Decodable, Identifiable, Hashable {
    let id: String
    let type: String?
    let loaded: Bool
    let source: String?
    let sizeBytes: Int?
    /// Weak at agentic tool calls (reasoning/thinking models) — rendered as ⚠️ in the picker so
    /// the user doesn't pick one for coding. Optional so an older backend without the field decodes.
    let weakAtTools: Bool?

    enum CodingKeys: String, CodingKey {
        case id, type, loaded, source
        case sizeBytes = "size_bytes"
        case weakAtTools = "weak_at_tools"
    }
}

struct ModelsDTO: Decodable {
    let models: [ModelDTO]
    let reachable: Bool
}

/// The backend-authoritative default chat model (GET /models/default). Shared with Telegram.
struct DefaultModelDTO: Decodable {
    let model: String?
    enum CodingKeys: String, CodingKey { case model = "default" }
}

/// Fusion config (GET/PUT /fusion): the panel models + judge for the virtual "fusion" model.
struct FusionConfigDTO: Decodable {
    let enabled: Bool
    let panel: [String]
    let judge: String?
}

/// One model's saved generation overrides (GET /models/{id}/settings). All optional — an
/// unset field means the global default applies.
struct ModelSettingsDTO: Decodable {
    let settings: Values
    struct Values: Decodable {
        let temperature: Double?
        let topP: Double?
        let topK: Int?
        let maxTokens: Int?
        enum CodingKeys: String, CodingKey {
            case temperature
            case topP = "top_p"
            case topK = "top_k"
            case maxTokens = "max_tokens"
        }
    }
}

struct SkillDTO: Decodable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let description: String
    let editable: Bool?  // user-authored skills are editable; bundled ones are read-only
}

struct SkillsDTO: Decodable {
    let skills: [SkillDTO]
}

struct SkillBodyDTO: Decodable {
    let name: String
    let body: String
    let description: String?
    let editable: Bool?
}

struct ReloadDTO: Decodable {
    let added: [String]
    let removed: [String]
    let unchanged: [String]
    let total: Int
}

struct MemoryEntryDTO: Decodable, Identifiable, Hashable {
    let id: String
    let content: String
    let tags: [String]
    let createdAt: String?

    enum CodingKeys: String, CodingKey {
        case id, content, tags
        case createdAt = "created_at"
    }
}

struct MemoryListDTO: Decodable {
    let memories: [MemoryEntryDTO]
}

struct MemorySearchDTO: Decodable {
    let results: [MemoryEntryDTO]
}

struct ImageResultDTO: Decodable {
    let path: String
}

struct DownloadDTO: Decodable, Identifiable, Hashable {
    var id: String { repoId }
    let repoId: String
    let status: String
    let totalBytes: Int
    let downloadedBytes: Int
    let etaSeconds: Int?
    let rateBps: Double?  // current transfer speed (10s-window average), bytes/s
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status, error
        case repoId = "repo_id"
        case totalBytes = "total_bytes"
        case downloadedBytes = "downloaded_bytes"
        case etaSeconds = "eta_seconds"
        case rateBps = "rate_bps"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repoId = try c.decode(String.self, forKey: .repoId)
        status = try c.decode(String.self, forKey: .status)
        totalBytes = try c.decodeIfPresent(Int.self, forKey: .totalBytes) ?? 0
        downloadedBytes = try c.decodeIfPresent(Int.self, forKey: .downloadedBytes) ?? 0
        // Tolerate an out-of-range ETA: a stalled download's remaining/rate can overflow Int64.
        // Treat that as "unknown" instead of throwing, which would corrupt the whole downloads
        // list decode (observed: "Number 6114835989602713600… is not representable in Swift").
        etaSeconds = (try? c.decodeIfPresent(Int.self, forKey: .etaSeconds)) ?? nil
        rateBps = try c.decodeIfPresent(Double.self, forKey: .rateBps)
        error = try c.decodeIfPresent(String.self, forKey: .error)
    }

    /// 0…1 when the total size is known; nil → indeterminate bar.
    var fraction: Double? {
        totalBytes > 0 ? min(1, Double(downloadedBytes) / Double(totalBytes)) : nil
    }
    var isActive: Bool { status == "queued" || status == "downloading" }
    var isResumable: Bool { status == "error" || status == "cancelled" }
}

struct DownloadsDTO: Decodable {
    let downloads: [DownloadDTO]
}

// --- preflight / setup ---

struct PreflightDTO: Decodable {
    let venv: String
    let python: String
    let configPath: String
    let configExists: Bool
    let downloadDir: String
    let paths: [PathCheckDTO]
    let tools: [ToolCheckDTO]
    let models: ModelsSummaryDTO
    let installs: [InstallDTO]

    enum CodingKeys: String, CodingKey {
        case venv, python, paths, tools, models, installs
        case configPath = "config_path"
        case configExists = "config_exists"
        case downloadDir = "download_dir"
    }
}

struct PathCheckDTO: Decodable, Identifiable, Hashable {
    var id: String { name }
    let name: String
    let path: String
    let exists: Bool
}

struct ToolCheckDTO: Decodable, Identifiable, Hashable {
    var id: String { feature }
    let feature: String
    let package: String
    let label: String
    let installed: Bool
    let version: String?       // installed version, if known
    let latest: String?        // PyPI latest (nil for source-overridden tools)
    let source: String?        // configured install source (patched build), if any
    let updateAvailable: Bool  // gates the "更新套件" button (N5)

    enum CodingKeys: String, CodingKey {
        case feature, package, label, installed, version, latest, source
        case updateAvailable = "update_available"
    }

    // Custom decode so an older backend (stale managed venv) that doesn't yet emit the new
    // fields still decodes — missing version info simply means "no update offered".
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        feature = try c.decode(String.self, forKey: .feature)
        package = try c.decode(String.self, forKey: .package)
        label = try c.decode(String.self, forKey: .label)
        installed = (try? c.decode(Bool.self, forKey: .installed)) ?? false
        version = try? c.decode(String.self, forKey: .version)
        latest = try? c.decode(String.self, forKey: .latest)
        source = try? c.decode(String.self, forKey: .source)
        updateAvailable = (try? c.decode(Bool.self, forKey: .updateAvailable)) ?? false
    }
}

struct ModelsSummaryDTO: Decodable, Hashable {
    let dir: String
    let exists: Bool
    let count: Int
    let hfCache: Bool

    enum CodingKeys: String, CodingKey {
        case dir, exists, count
        case hfCache = "hf_cache"
    }
}

struct InstallDTO: Decodable, Identifiable, Hashable {
    var id: String { feature }
    let feature: String
    let package: String
    let status: String
    let error: String?
}

struct ConfigDTO: Decodable {
    let modelsDir: String
    let downloadDir: String
    let extraModelDirs: [String]
    let hfCache: Bool
    let backendHost: String
    let backendPort: Int
    let modelBackend: String
    let maxOutputTokens: Int
    let maxToolIters: Int
    let turnTimeoutS: Double?  // per-turn wall-clock budget in seconds; nil/None = unlimited
    let memCeilingGb: Double?  // engine-pool memory-admission ceiling in GB; nil/None = no ceiling
    // Model-download tunables. hfHubDisableXet is the big one — Xet was measured throttling to a
    // few KB/s on some networks.
    let hfHubDisableXet: Bool
    let hfHubDownloadTimeout: Int
    let hfDownloadMaxWorkers: Int
    let configPath: String
    // Gateways (S9): the token is masked by the backend; full secret never crosses the wire.
    let telegramConfigured: Bool
    let telegramTokenMasked: String?
    let telegramAllowedUsers: [Int]
    let telegramRunning: Bool
    let telegramError: String?

    enum CodingKeys: String, CodingKey {
        case configPath = "config_path"
        case modelsDir = "models_dir"
        case downloadDir = "download_dir"
        case extraModelDirs = "extra_model_dirs"
        case hfCache = "hf_cache"
        case backendHost = "backend_host"
        case backendPort = "backend_port"
        case modelBackend = "model_backend"
        case maxOutputTokens = "max_output_tokens"
        case maxToolIters = "max_tool_iters"
        case turnTimeoutS = "turn_timeout_s"
        case memCeilingGb = "mem_ceiling_gb"
        case hfHubDisableXet = "hf_hub_disable_xet"
        case hfHubDownloadTimeout = "hf_hub_download_timeout"
        case hfDownloadMaxWorkers = "hf_download_max_workers"
        case telegramConfigured = "telegram_configured"
        case telegramTokenMasked = "telegram_token_masked"
        case telegramAllowedUsers = "telegram_allowed_users"
        case telegramRunning = "telegram_running"
        case telegramError = "telegram_error"
    }
}

/// A single Server-Sent Event from `/chat`. Decoded loosely: only `type` is always
/// present; the rest are populated per event kind.
struct ChatEvent: Decodable {
    let type: String
    let content: String?
    let sessionId: String?
    let name: String?
    let ok: Bool?
    let detail: String?
    let token: String?  // approval_request: the id to POST back to /chat/approve
    let usage: Usage?   // done: token accounting for the just-finished turn
    let summary: String?  // turn_diff: "N files changed (+x/-y)"
    let diff: String?     // turn_diff: the unified diff of files the turn changed
    let steps: [PlanStep]?  // plan: the agent's current checklist for this turn (SA.3)

    /// Estimated token counts carried by the terminal `done` event (backend tokens.py).
    struct Usage: Decodable {
        let contextTokens: Int?
        let outputTokens: Int?

        enum CodingKeys: String, CodingKey {
            case contextTokens = "context_tokens"
            case outputTokens = "output_tokens"
        }
    }

    enum CodingKeys: String, CodingKey {
        case type, content, name, ok, detail, token, usage, summary, diff, steps
        case sessionId = "session_id"
    }
}

/// One item in the agent's per-turn plan checklist (`plan` event). `status` is
/// "pending" | "in_progress" | "completed".
struct PlanStep: Decodable, Identifiable {
    let title: String
    let status: String
    var id: String { title }
}

// --- sessions (persisted conversations, S1) ---

struct SessionListDTO: Decodable {
    let sessions: [SessionSummaryDTO]
}

struct SessionSummaryDTO: Decodable, Identifiable, Hashable {
    let id: String
    let title: String
    let model: String?
    let messageCount: Int
    let lastAccessedAt: Double

    enum CodingKeys: String, CodingKey {
        case id, title, model
        case messageCount = "message_count"
        case lastAccessedAt = "last_accessed_at"
    }
}

struct SessionDetailDTO: Decodable {
    let id: String
    let model: String?
    let title: String
    let messages: [SessionMessageDTO]
}

struct SessionMessageDTO: Decodable {
    let role: String
    let content: String?

    enum CodingKeys: String, CodingKey { case role, content }

    // Tolerant decode: tool/system messages may carry non-string content; capture only
    // plain-string content (user/assistant) and treat anything else as empty rather than
    // failing the whole session load.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        role = (try? c.decode(String.self, forKey: .role)) ?? ""
        content = try? c.decode(String.self, forKey: .content)
    }
}
