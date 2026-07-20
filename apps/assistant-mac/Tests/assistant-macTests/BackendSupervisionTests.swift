import Testing

@testable import assistant_mac

/// The auto-restart backoff schedule (S8) and the restart-on-failed-probe policy (N93).
/// The point: a backend that keeps crashing must back off (not hammer respawns), grow
/// geometrically, and cap; and a live-but-busy backend must never be executed on a single
/// failed probe — so both policies are pinned here.
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

    // N93: an exited child is a crash — recovery must not wait for more strikes.
    @Test func deadChildRestartsOnFirstFailedProbe() {
        #expect(BackendController.shouldRestart(processAlive: false, consecutiveFailures: 1))
    }

    // N93: a live child failing a probe is busy (event loop starved by a heavy turn), not
    // dead. One or two failures must NOT trigger a restart — that SIGTERMed a working
    // backend mid-generation.
    @Test func liveChildToleratesEarlyFailedProbes() {
        #expect(!BackendController.shouldRestart(processAlive: true, consecutiveFailures: 1))
        #expect(
            !BackendController.shouldRestart(
                processAlive: true,
                consecutiveFailures: BackendController.probeStrikeLimit - 1))
    }

    // N93: a sustained outage still escalates — a genuinely wedged backend must recover.
    @Test func liveChildRestartsAtStrikeLimit() {
        #expect(
            BackendController.shouldRestart(
                processAlive: true,
                consecutiveFailures: BackendController.probeStrikeLimit))
    }
}
