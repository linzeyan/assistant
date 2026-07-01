import Foundation
import Testing

@testable import assistant_mac

/// The DTOs are the app↔backend contract. These lock the snake_case→camelCase mapping
/// and the tolerant decoders so a backend field rename is caught here, not at runtime.
struct DTODecodingTests {
    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    @Test func modelsDTOMapsSnakeCase() throws {
        let dto = try decode(
            ModelsDTO.self,
            """
            {"models":[{"id":"org/m","type":"llm","loaded":true,"source":"hf_cache","size_bytes":1024}],
             "reachable":true}
            """)
        #expect(dto.reachable)
        #expect(dto.models.count == 1)
        #expect(dto.models[0].id == "org/m")
        #expect(dto.models[0].source == "hf_cache")
        #expect(dto.models[0].sizeBytes == 1024)
    }

    @Test func modelDTOToleratesMissingOptionals() throws {
        let dto = try decode(ModelDTO.self, #"{"id":"m","loaded":false}"#)
        #expect(dto.type == nil)
        #expect(dto.sizeBytes == nil)
        #expect(dto.loaded == false)
    }

    @Test func configDTOMapsAllSnakeCaseKeys() throws {
        let dto = try decode(
            ConfigDTO.self,
            """
            {"models_dir":"/m","download_dir":"/d","extra_model_dirs":["/e"],"hf_cache":true,
             "backend_host":"127.0.0.1","backend_port":9981,"model_backend":"mlx",
             "max_output_tokens":8192,"max_tool_iters":16,"turn_timeout_s":300,"mem_ceiling_gb":110,
             "hf_hub_disable_xet":true,"hf_hub_download_timeout":120,"hf_download_max_workers":4,
             "config_path":"/c",
             "telegram_configured":true,"telegram_token_masked":"1234…wxyz",
             "telegram_allowed_users":[7,8],"telegram_running":true,"telegram_error":null}
            """)
        #expect(dto.modelsDir == "/m")
        #expect(dto.downloadDir == "/d")
        #expect(dto.extraModelDirs == ["/e"])
        #expect(dto.hfCache)
        #expect(dto.backendPort == 9981)
        #expect(dto.modelBackend == "mlx")
        #expect(dto.maxOutputTokens == 8192)
        #expect(dto.maxToolIters == 16)
        #expect(dto.turnTimeoutS == 300)
        #expect(dto.memCeilingGb == 110)
        #expect(dto.hfHubDisableXet)
        #expect(dto.hfHubDownloadTimeout == 120)
        #expect(dto.hfDownloadMaxWorkers == 4)
        #expect(dto.telegramConfigured)
        #expect(dto.telegramTokenMasked == "1234…wxyz")
        #expect(dto.telegramAllowedUsers == [7, 8])
        #expect(dto.telegramRunning)
        #expect(dto.telegramError == nil)
    }

    @Test func gatewayAllowlistParsesCommaSeparatedIds() throws {
        #expect(SettingsScreen.parseAllowlist("12345, 67890") == [12345, 67890])
        #expect(SettingsScreen.parseAllowlist(" 7 ,bad, 9 ") == [7, 9])  // skips non-numeric
        #expect(SettingsScreen.parseAllowlist("") == [])
    }

    @Test func downloadDTOMapsProgressFields() throws {
        let dto = try decode(
            DownloadDTO.self,
            """
            {"repo_id":"org/m","status":"downloading","total_bytes":1000,
             "downloaded_bytes":250,"eta_seconds":12,"error":null}
            """)
        #expect(dto.repoId == "org/m")
        #expect(dto.status == "downloading")
        #expect(dto.totalBytes == 1000)
        #expect(dto.downloadedBytes == 250)
        #expect(dto.etaSeconds == 12)
        #expect(dto.fraction == 0.25)
        #expect(dto.isActive)
        #expect(!dto.isResumable)
    }

    @Test func chatEventMapsSessionId() throws {
        let event = try decode(ChatEvent.self, #"{"type":"session","session_id":"abc"}"#)
        #expect(event.type == "session")
        #expect(event.sessionId == "abc")
        #expect(event.content == nil)
    }

    @Test func statusDTONestedOmlx() throws {
        let dto = try decode(
            StatusDTO.self,
            """
            {"backend":"ok","model_backend":"mlx",
             "omlx":{"state":"local","detail":"ready","base_url":null,"reachable":true}}
            """)
        #expect(dto.modelBackend == "mlx")
        #expect(dto.omlx.state == "local")
        #expect(dto.omlx.baseURL == nil)
        #expect(dto.omlx.reachable)
    }

    @Test func sessionDetailTolerantMessageDecode() throws {
        // tool/system messages may carry non-string content; those become nil content
        // (role preserved) rather than failing the whole session load.
        let dto = try decode(
            SessionDetailDTO.self,
            """
            {"id":"s1","model":"m","title":"t","messages":[
              {"role":"user","content":"hi"},
              {"role":"assistant","content":"yo"},
              {"role":"tool","content":{"nested":1}}
            ]}
            """)
        #expect(dto.messages.count == 3)
        #expect(dto.messages[0].content == "hi")
        #expect(dto.messages[2].role == "tool")
        #expect(dto.messages[2].content == nil)  // non-string content tolerated → nil
    }

    @Test func sessionSummaryMapsCountAndTimestamp() throws {
        let dto = try decode(
            SessionSummaryDTO.self,
            #"{"id":"s","title":"hello","model":"m","message_count":4,"last_accessed_at":1.5}"#)
        #expect(dto.messageCount == 4)
        #expect(abs(dto.lastAccessedAt - 1.5) < 0.0001)
    }
}
