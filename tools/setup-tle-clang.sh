#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
set -euo pipefail

# TLE raw CUDA needs Clang 20+. Reuse a supported runner compiler or unpack
# checksum-pinned LLVM packages into a private job toolchain directory.
find_clang() {
    local candidate path version major
    for candidate in "${CLANG:-}" clang clang-23 clang-22 clang-21 clang-20; do
        [[ -n "$candidate" ]] || continue
        path="$(command -v -- "$candidate" 2>/dev/null)" || continue
        [[ -f "$path" && -x "$path" ]] || continue
        version="$("$path" --version 2>/dev/null)" || continue
        major="$(sed -nE 's/.*clang version ([0-9]+).*/\1/p' <<< "$version" | head -1)"
        if [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 20 )); then
            readlink -f -- "$path"
            return 0
        fi
    done
    return 1
}

if compiler="$(find_clang)"; then
    "$compiler" --version >&2
    printf '%s\n' "$compiler"
    exit 0
fi

[[ "$(uname -s)-$(uname -m)" == Linux-x86_64 ]] || {
    echo 'LLVM bootstrap requires Linux x86_64; install Clang 20+ on this runner.' >&2
    exit 1
}
prefix="${1:?Expected a private toolchain directory}"
cache="${2:-${HOME:?}/.cache/flaggems-vllm/tle-clang22.1.8}"
mkdir -p -- "$prefix" "$cache"
prefix="$(cd -- "$prefix" && pwd -P)"
cache="$(cd -- "$cache" && pwd -P)"

# Concurrent jobs share only verified archives. Partial files and extraction
# roots are private, and no compiler path is published before validation.
exec 9>"${cache}/.lock"
flock 9
partial=""
toolchain=""
cleanup() {
    [[ -z "$partial" ]] || rm -f -- "$partial"
    [[ -z "$toolchain" ]] || rm -rf -- "$toolchain"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

debs=()
check_package() {
    local name=$1 checksum=$2 url=$3 mirror=$4 deb
    deb="${cache}/${name}.deb"
    if [[ ! -f "$deb" ]] || ! printf '%s  %s\n' "$checksum" "$deb" | sha256sum --check --status; then
        partial="$(mktemp "${cache}/.${name}.part.XXXXXX")"
        for source in "$url" "$mirror"; do
            echo "Downloading ${name} from ${source}" >&2
            if curl --fail --silent --show-error --location --retry 2 \
                --retry-all-errors --connect-timeout 12 --max-time 90 \
                --retry-max-time 100 --output "$partial" "$source" \
                && printf '%s  %s\n' "$checksum" "$partial" | sha256sum --check --status; then
                mv -f -- "$partial" "$deb"
                partial=""
                break
            fi
        done
        [[ -z "$partial" ]] || {
            echo "Unable to obtain checksum-verified ${name}" >&2
            return 1
        }
    fi
    debs+=("$deb")
}

archive='https://apt.llvm.org/jammy/pool/main/l/llvm-toolchain-22'
mirror='https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/jammy/pool/main/l/llvm-toolchain-22'
version='22.1.8~++20260613092327+e80beda6e255-1~exp1~20260613092437.81'
while read -r package checksum; do
    filename="${package}_${version}_amd64.deb"
    check_package "$package" "$checksum" "${archive}/${filename}" "${mirror}/${filename}"
done <<'PACKAGES'
clang-22 0153d60799f66344f4b1d13e0de28edaf25aed384453b3ac2e5dd15e06ba70e7
libclang-common-22-dev d3a8cd0089d4756f045aea2f320c35dd372ca88c774e3e90c9645b6a404110f7
libclang-cpp22 819cccae27885e70ecb23c4a42065ce50bba8c62740b6728f637698de76faa3d
libllvm22 4bd592cab11e90e5b8625b5e3a1ceb5936e843c63bf7394f6f6506aa41302164
PACKAGES
z3='pool/universe/z/z3/libz3-4_4.8.12-1_amd64.deb'
check_package libz3-4 d83fda4b2e58d9b30f0f85b02a3eb48ca7d33ca88e9d4584249e698bc2e451cf \
    "https://archive.ubuntu.com/ubuntu/${z3}" "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/${z3}"

toolchain="$(mktemp -d "${prefix}/llvm22.XXXXXX")"
for deb in "${debs[@]}"; do
    dpkg-deb --extract "$deb" "$toolchain"
done
cat >"${toolchain}/clang" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export LD_LIBRARY_PATH="${root}/usr/lib/x86_64-linux-gnu:${root}/usr/lib/llvm-22/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${root}/usr/lib/llvm-22/bin/clang" "$@"
WRAPPER
chmod +x "${toolchain}/clang"
compiler="${toolchain}/clang"
"$compiler" --version >&2
toolchain=""
printf '%s\n' "$compiler"
