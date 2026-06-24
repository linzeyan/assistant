import Testing

@testable import assistant_mac

/// The auto-restart backoff schedule (S8). The point: a backend that keeps crashing must back
/// off (not hammer respawns), grow geometrically, and cap — so the policy is pinned here.
struct BackendSupervisionTests {
    @Test func backoffStartsAtOneSecond() {
        #expect(BackendController.nextBackoffNanos(0) == 1_000_000_000)
    }

    @Test func backoffDoubles() {
        #expect(BackendController.nextBackoffNanos(1_000_000_000) == 2_000_000_000)
        #expect(BackendController.nextBackoffNanos(2_000_000_000) == 4_000_000_000)
    }

    @Test func backoffCapsAtThirtySeconds() {
        // 16s would double to 32s — clamp to the 30s ceiling, and stay there.
        #expect(BackendController.nextBackoffNanos(16_000_000_000) == 30_000_000_000)
        #expect(BackendController.nextBackoffNanos(30_000_000_000) == 30_000_000_000)
    }
}
