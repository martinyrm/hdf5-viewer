#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 /path/to/hdf5_plotter /path/to/HDF5 ImPlot Viewer.app" >&2
    exit 2
fi

binary=$1
app=$2
app_name=${APP_NAME:-HDF5 ImPlot Viewer}
bundle_id=${BUNDLE_ID:-org.local.hdf5-imgui-plotter}

if [[ ! -x "$binary" ]]; then
    echo "binary is not executable: $binary" >&2
    exit 1
fi

rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Frameworks" "$app/Contents/Resources"
cp -p "$binary" "$app/Contents/MacOS/hdf5_plotter"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>hdf5_plotter</string>
  <key>CFBundleIdentifier</key>
  <string>${bundle_id}</string>
  <key>CFBundleName</key>
  <string>${app_name}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${VERSION:-0.0.0}</string>
  <key>CFBundleVersion</key>
  <string>${VERSION:-0.0.0}</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

linked_deps() {
    otool -L "$1" | awk 'NR > 1 { print $1 }' | grep -E '^(/opt/homebrew|/usr/local)/' || true
}

queue=$(mktemp)
copied=$(mktemp)
trap 'rm -f "$queue" "$copied" "$queue.tmp"' EXIT

linked_deps "$app/Contents/MacOS/hdf5_plotter" > "$queue"

while [[ -s "$queue" ]]; do
    dep=$(head -n 1 "$queue")
    tail -n +2 "$queue" > "$queue.tmp"
    mv "$queue.tmp" "$queue"

    if grep -Fxq "$dep" "$copied"; then
        continue
    fi
    echo "$dep" >> "$copied"

    base=$(basename "$dep")
    target="$app/Contents/Frameworks/$base"
    if [[ ! -e "$target" ]]; then
        cp -pL "$dep" "$target"
        chmod u+w "$target"
    fi

    linked_deps "$target" | while IFS= read -r child; do
        if [[ -n "$child" ]] && ! grep -Fxq "$child" "$copied" && ! grep -Fxq "$child" "$queue"; then
            echo "$child" >> "$queue"
        fi
    done
done

install_name_tool -add_rpath "@executable_path/../Frameworks" "$app/Contents/MacOS/hdf5_plotter" 2>/dev/null || true

find "$app/Contents/Frameworks" -type f -name '*.dylib' -print0 | while IFS= read -r -d '' lib; do
    install_name_tool -id "@rpath/$(basename "$lib")" "$lib"
done

while IFS= read -r target; do
    linked_deps "$target" | while IFS= read -r dep; do
        base=$(basename "$dep")
        if [[ -e "$app/Contents/Frameworks/$base" ]]; then
            install_name_tool -change "$dep" "@rpath/$base" "$target"
        fi
    done
done < <(printf '%s\n' "$app/Contents/MacOS/hdf5_plotter"; find "$app/Contents/Frameworks" -type f -name '*.dylib')

codesign --force --deep --sign - "$app" >/dev/null 2>&1 || true
