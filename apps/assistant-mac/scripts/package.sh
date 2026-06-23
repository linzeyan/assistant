#!/usr/bin/env bash
#
# Build, bundle, and code-sign the macOS app into dist/Assistant.app.
#
# Signing identity (env CODESIGN_IDENTITY):
#   unset / "-"  -> ad-hoc signature: runs on THIS machine, no Apple account needed.
#   "Developer ID Application: Name (TEAMID)" -> distributable; enables the hardened
#                 runtime (required before notarization, see `make app-notarize`).
#
set -euo pipefail

APP_NAME="Assistant"
BIN_NAME="assistant-mac"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # apps/assistant-mac
ROOT_DIR="$(cd "$APP_DIR/../.." && pwd)"
DIST="$ROOT_DIR/dist"
BUNDLE="$DIST/$APP_NAME.app"
IDENTITY="${CODESIGN_IDENTITY:--}"
ENTITLEMENTS="$APP_DIR/Resources/assistant.entitlements"

echo "==> Building release binary"
( cd "$APP_DIR" && swift build -c release )

echo "==> Assembling $BUNDLE"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS" "$BUNDLE/Contents/Resources"
cp "$APP_DIR/.build/release/$BIN_NAME" "$BUNDLE/Contents/MacOS/$BIN_NAME"
cp "$APP_DIR/Resources/Info.plist" "$BUNDLE/Contents/Info.plist"
# Optional icon: drop an AppIcon.icns into Resources/ to brand the app.
if [ -f "$APP_DIR/Resources/AppIcon.icns" ]; then
    cp "$APP_DIR/Resources/AppIcon.icns" "$BUNDLE/Contents/Resources/AppIcon.icns"
fi

# Thin-app backend payload: the wheel + bootstrap script the GUI installs into a
# managed venv on first run (the app ships no Python itself).
echo "==> Building + bundling backend wheel"
BACKEND_RES="$BUNDLE/Contents/Resources/backend"
mkdir -p "$BACKEND_RES"
( cd "$ROOT_DIR" && uv build --wheel -o "$BACKEND_RES" >/dev/null )
rm -f "$BACKEND_RES/.gitignore"  # uv drops one in the output dir; keep the bundle clean
cp "$APP_DIR/scripts/bootstrap.sh" "$BACKEND_RES/bootstrap.sh"
chmod +x "$BACKEND_RES/bootstrap.sh"

echo "==> Signing (identity: $IDENTITY)"
SIGN_ARGS=(--force --sign "$IDENTITY")
if [ "$IDENTITY" != "-" ]; then
    # Real Developer ID: hardened runtime + secure timestamp (needed to notarize).
    SIGN_ARGS+=(--options runtime --timestamp)
fi
if [ -f "$ENTITLEMENTS" ]; then
    SIGN_ARGS+=(--entitlements "$ENTITLEMENTS")
fi
codesign "${SIGN_ARGS[@]}" "$BUNDLE"

echo "==> Verifying signature"
codesign --verify --strict --verbose=2 "$BUNDLE"
codesign --display --verbose=2 "$BUNDLE" 2>&1 | sed 's/^/    /'

echo "==> Done: $BUNDLE"
if [ "$IDENTITY" = "-" ]; then
    echo "    Ad-hoc signed. First launch: right-click -> Open (Gatekeeper) or"
    echo "    sign with a Developer ID + notarize for distribution."
fi
echo "    open \"$BUNDLE\""
