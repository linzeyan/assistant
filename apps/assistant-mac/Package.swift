// swift-tools-version: 5.9
import PackageDescription

// Built as a SwiftPM executable (not an .xcodeproj) so it compiles and runs from the
// command line under Command Line Tools — no full Xcode required for development.
// Proper .app bundling / signing is deferred to the packaging phase.
let package = Package(
    name: "assistant-mac",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "assistant-mac",
            path: "Sources/assistant-mac"
        ),
        // Unit tests for the pure logic (DTO decoding, SSE reducer, segment parser,
        // client framing / error mapping). Run locally: `swift test` or `make app-test`.
        .testTarget(
            name: "assistant-macTests",
            dependencies: ["assistant-mac"],
            path: "Tests/assistant-macTests"
        ),
    ]
)
