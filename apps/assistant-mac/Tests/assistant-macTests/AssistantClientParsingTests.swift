import Foundation
import Testing

@testable import assistant_mac

/// Pure parsing/mapping helpers extracted from `AssistantClient` so they can be tested
/// without a live backend: SSE line framing, HTTP→ClientError mapping (which surfaces
/// the backend's `detail`), and trailing-slash-safe URL joining.
struct AssistantClientParsingTests {
    // --- SSE line framing ---

    @Test func parsesDataLine() {
        let event = AssistantClient.parseSSELine(#"data: {"type":"assistant_delta","content":"hi"}"#)
        #expect(event?.type == "assistant_delta")
        #expect(event?.content == "hi")
    }

    @Test func ignoresNonDataLine() {
        #expect(AssistantClient.parseSSELine(": keep-alive") == nil)
        #expect(AssistantClient.parseSSELine("event: ping") == nil)
    }

    @Test func ignoresEmptyPayload() {
        #expect(AssistantClient.parseSSELine("data:") == nil)
        #expect(AssistantClient.parseSSELine("data:    ") == nil)
    }

    @Test func ignoresMalformedJSON() {
        #expect(AssistantClient.parseSSELine("data: {not json}") == nil)
    }

    // --- HTTP → ClientError mapping ---

    @Test func mapErrorReturnsNilBelow400() {
        #expect(AssistantClient.mapError(statusCode: 200, data: Data()) == nil)
        #expect(AssistantClient.mapError(statusCode: 302, data: Data()) == nil)
    }

    @Test func mapErrorPrefersBackendDetail() {
        let data = Data(#"{"detail":"model not found"}"#.utf8)
        guard case .server(let code, let detail)? = AssistantClient.mapError(statusCode: 404, data: data)
        else {
            Issue.record("expected .server")
            return
        }
        #expect(code == 404)
        #expect(detail == "model not found")
    }

    @Test func mapErrorFallsBackToStatusWhenNoDetail() {
        guard case .http(let code)? = AssistantClient.mapError(statusCode: 500, data: Data()) else {
            Issue.record("expected .http")
            return
        }
        #expect(code == 500)
    }

    @Test func clientErrorDescriptionSurfacesDetail() {
        #expect(ClientError.server(404, "why").description == "why")
        #expect(ClientError.http(500).description == "HTTP 500")
    }

    // --- URL joining ---

    @Test func urlJoinHandlesTrailingSlash() {
        let noSlash = AssistantClient(baseURL: URL(string: "http://127.0.0.1:9981")!)
        #expect(noSlash.url("/status").absoluteString == "http://127.0.0.1:9981/status")
        let withSlash = AssistantClient(baseURL: URL(string: "http://127.0.0.1:9981/")!)
        #expect(withSlash.url("/status").absoluteString == "http://127.0.0.1:9981/status")
    }
}
