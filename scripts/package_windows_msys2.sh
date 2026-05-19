#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 /path/to/hdf5_plotter.exe /path/to/package-dir" >&2
    exit 2
fi

exe=$1
out_dir=$2

if [[ ! -f "$exe" ]]; then
    echo "executable not found: $exe" >&2
    exit 1
fi
if [[ -z "${MINGW_PREFIX:-}" ]]; then
    echo "MINGW_PREFIX is not set; run this from an MSYS2 MinGW/UCRT shell" >&2
    exit 1
fi

rm -rf "$out_dir"
mkdir -p "$out_dir"
cp -p "$exe" "$out_dir/hdf5_plotter.exe"
cp -p README.md "$out_dir/README.md"
cp -p docs/releasing.md "$out_dir/RELEASING.md"

cat > "$out_dir/RUNTIME-DEPS.txt" <<DEPS
This Windows zip is self-contained for the MSYS2/MinGW runtime libraries used by
the app. It includes SDL2, HDF5, FreeType, libstdc++, libgcc, winpthread, and
their MSYS2 UCRT64 DLL dependencies next to hdf5_plotter.exe.

The host still needs a working Windows OpenGL driver. If the app starts but plot
rendering is slow or blank, update the GPU driver from Intel, AMD, NVIDIA, or
the computer vendor.
DEPS

declare -a queue=("$out_dir/hdf5_plotter.exe")
declare -a seen=()

already_seen() {
    local path=$1
    local item
    for item in "${seen[@]}"; do
        [[ "$item" == "$path" ]] && return 0
    done
    return 1
}

queue_dep() {
    local path=$1
    [[ "$path" == "${MINGW_PREFIX}/bin/"*.dll ]] || return 0
    already_seen "$path" && return 0
    local item
    for item in "${queue[@]}"; do
        [[ "$item" == "$path" ]] && return 0
    done
    queue+=("$path")
}

read_deps() {
    local target=$1
    ldd "$target" | awk '{ for (i = 1; i <= NF; ++i) if ($i ~ /^\//) print $i }'
}

index=0
while (( index < ${#queue[@]} )); do
    target=${queue[$index]}
    index=$((index + 1))
    already_seen "$target" && continue
    seen+=("$target")

    if [[ "$target" == "${MINGW_PREFIX}/bin/"*.dll ]]; then
        cp -p "$target" "$out_dir/$(basename "$target")"
    fi

    while IFS= read -r dep; do
        queue_dep "$dep"
    done < <(read_deps "$target")
done

{
    echo "Packaged files:"
    find "$out_dir" -maxdepth 1 -type f -printf '%f\n' | sort
    echo
    echo "ldd hdf5_plotter.exe:"
    ldd "$out_dir/hdf5_plotter.exe"
} > "$out_dir/LINKED-DLLS.txt"
