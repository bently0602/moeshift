#!/bin/sh
# Build the engine and both MoEShift apps into build/apps/ (at the repository root):
#   MoEShift Flash Next.app   Qwen 3.8 Flash Next, port 8080
#   MoEShift A3B.app          Qwen3.6 35B A3B, port 8081
# Each bundle holds the moeshift binary and a copy of configs/moeshift.json, the tuned profiles.
# Needs only the Apple Command Line Tools and CMake (no Xcode). The apps are signed for this Mac
# only (ad hoc); copy them to /Applications if you like.
#
# Usage: apps/build.sh [--skip-engine]
#   --skip-engine  reuse build/bin/moeshift instead of running scripts/build-moeshift.sh
set -eu

cd "$(dirname "$0")/.."
root="$(pwd)"
out="$root/build/apps"

if [ "${1-}" != --skip-engine ]; then
  scripts/build-moeshift.sh
fi
[ -x build/bin/moeshift ] || { echo "build/bin/moeshift is missing; run without --skip-engine" >&2; exit 1; }

swift build -c release --package-path apps
bin="$(swift build -c release --package-path apps --show-bin-path)"
version="$(git describe --always --dirty 2>/dev/null || echo unknown)"
mkdir -p "$out"

make_app() {
  exe="$1"; name="$2"; id="$3"
  app="$out/$name.app"
  rm -rf "$app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cp "$bin/$exe" "$app/Contents/MacOS/$exe"
  cp build/bin/moeshift "$app/Contents/MacOS/moeshift"
  cp configs/moeshift.json "$app/Contents/Resources/moeshift.json"
  cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>$exe</string>
  <key>CFBundleIdentifier</key><string>$id</string>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
</dict>
</plist>
EOF
  codesign --force --sign - "$app/Contents/MacOS/moeshift"
  codesign --force --sign - "$app"
  echo "Built $app"
}

make_app FlashNextApp "MoEShift Flash Next" local.moeshift.flashnext
make_app A3BApp "MoEShift A3B" local.moeshift.a3b
