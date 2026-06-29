#!/usr/bin/env swift
//
// Generate Resources/AppIcon.icns from the menu-bar glyph (SF Symbol "brain.head.profile"),
// so the Dock / app-switcher icon matches the status-bar item. Rendered offscreen (no Xcode
// asset catalog — this project is CLT-only), then packed with iconutil.
//
// Run: swift apps/assistant-mac/scripts/make-icon.swift
//
import AppKit
import Foundation

let symbolName = "brain.head.profile"
let scriptDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let resourcesDir = scriptDir.deletingLastPathComponent().appendingPathComponent("Resources")
let iconsetDir = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon.iconset")
let icnsOut = resourcesDir.appendingPathComponent("AppIcon.icns")

/// Render one square icon at `px` pixels: a vertical graphite gradient rounded square (macOS
/// "squircle"-ish corner radius) with the white symbol centered. Returns PNG data.
func renderIcon(px: Int) -> Data {
    let size = CGFloat(px)
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    )!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    // Leave a small transparent margin so the rounded square reads as an app icon, not a
    // full-bleed tile. ~10% inset matches Apple's macOS icon grid roughly.
    let inset = size * 0.092
    let rect = NSRect(x: inset, y: inset, width: size - inset * 2, height: size - inset * 2)
    let radius = rect.width * 0.225
    let squircle = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    let gradient = NSGradient(
        starting: NSColor(calibratedRed: 0.20, green: 0.21, blue: 0.24, alpha: 1),
        ending: NSColor(calibratedRed: 0.11, green: 0.11, blue: 0.13, alpha: 1)
    )!
    gradient.draw(in: squircle, angle: -90)

    // White symbol, centered, sized to ~56% of the canvas.
    let cfg = NSImage.SymbolConfiguration(pointSize: size * 0.5, weight: .regular)
    if let symbol = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil)?
        .withSymbolConfiguration(cfg) {
        let s = symbol.size
        let scale = (size * 0.56) / max(s.width, s.height)
        let drawW = s.width * scale, drawH = s.height * scale
        let dst = NSRect(x: (size - drawW) / 2, y: (size - drawH) / 2, width: drawW, height: drawH)
        let tinted = NSImage(size: NSSize(width: drawW, height: drawH))
        tinted.lockFocus()
        symbol.draw(in: NSRect(x: 0, y: 0, width: drawW, height: drawH))
        NSColor.white.set()
        NSRect(x: 0, y: 0, width: drawW, height: drawH).fill(using: .sourceAtop)
        tinted.unlockFocus()
        tinted.draw(in: dst)
    }

    NSGraphicsContext.restoreGraphicsState()
    return rep.representation(using: .png, properties: [:])!
}

// macOS iconset members: each logical size at @1x and @2x.
let members: [(name: String, px: Int)] = [
    ("icon_16x16", 16), ("icon_16x16@2x", 32),
    ("icon_32x32", 32), ("icon_32x32@2x", 64),
    ("icon_128x128", 128), ("icon_128x128@2x", 256),
    ("icon_256x256", 256), ("icon_256x256@2x", 512),
    ("icon_512x512", 512), ("icon_512x512@2x", 1024),
]

try? FileManager.default.removeItem(at: iconsetDir)
try! FileManager.default.createDirectory(at: iconsetDir, withIntermediateDirectories: true)
for m in members {
    let data = renderIcon(px: m.px)
    try! data.write(to: iconsetDir.appendingPathComponent("\(m.name).png"))
}

let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconsetDir.path, "-o", icnsOut.path]
try! iconutil.run()
iconutil.waitUntilExit()
guard iconutil.terminationStatus == 0 else {
    FileHandle.standardError.write("iconutil failed\n".data(using: .utf8)!)
    exit(1)
}
print("wrote \(icnsOut.path)")
