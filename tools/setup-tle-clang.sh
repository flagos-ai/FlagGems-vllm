#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
set -euo pipefail

# Extract the Jammy Clang 22.1.8 device compiler and its runtime dependency
# packages from LLVM's official APT archive. TLE uses clang to emit device IR,
# so it does not need a linker or system-level package installation.
mode="${1:?Expected download or extract}"
packages="${2:?Expected a package directory}"
case "$mode" in
    download) mkdir -p -- "$packages" ;;
    extract) toolchain="$(cd -- "${3:?Expected a private toolchain directory}" && pwd -P)" ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
archive="https://apt.llvm.org/jammy/pool/main/l/llvm-toolchain-22"
llvm_mirror="https://mirrors.tuna.tsinghua.edu.cn/llvm-apt/jammy/pool/main/l/llvm-toolchain-22"
version="22.1.8~++20260613092327+e80beda6e255-1~exp1~20260613092437.81"

download_package() {
    curl --fail --silent --show-error --location --retry 2 \
        --connect-timeout 12 --max-time 90 --retry-max-time 100 \
        --output "$2" "$1"
}

check_package() {
    local url=$1 mirror=$2 checksum=$3 name=$4 deb
    deb="${packages}/${name}.deb"
    if [[ "${mode}" == download ]]; then
        echo "Downloading ${name}" >&2
        download_package "$url" "$deb" || download_package "$mirror" "$deb"
    fi
    printf '%s  %s\n' "$checksum" "$deb" | sha256sum --check --quiet >&2
    if [[ "${mode}" == extract ]]; then
        dpkg-deb --extract "$deb" "$toolchain"
    fi
}

while read -r package checksum; do
    filename="${package}_${version}_amd64.deb"
    check_package "${archive}/${filename}" "${llvm_mirror}/${filename}" "$checksum" "$package"
done <<'PACKAGES'
clang-22 0153d60799f66344f4b1d13e0de28edaf25aed384453b3ac2e5dd15e06ba70e7
libclang-common-22-dev d3a8cd0089d4756f045aea2f320c35dd372ca88c774e3e90c9645b6a404110f7
libclang-cpp22 819cccae27885e70ecb23c4a42065ce50bba8c62740b6728f637698de76faa3d
libllvm22 4bd592cab11e90e5b8625b5e3a1ceb5936e843c63bf7394f6f6506aa41302164
PACKAGES
# Ubuntu Jammy runtime dependency of libllvm22.
z3_path="pool/universe/z/z3/libz3-4_4.8.12-1_amd64.deb"
check_package "https://archive.ubuntu.com/ubuntu/${z3_path}" \
    "https://mirrors.tuna.tsinghua.edu.cn/ubuntu/${z3_path}" \
    d83fda4b2e58d9b30f0f85b02a3eb48ca7d33ca88e9d4584249e698bc2e451cf libz3-4

if [[ "${mode}" == download ]]; then
    exit 0
fi

cat >"${toolchain}/clang" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd -P)"
export LD_LIBRARY_PATH="${root}/usr/lib/x86_64-linux-gnu:${root}/usr/lib/llvm-22/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${root}/usr/lib/llvm-22/bin/clang" "$@"
WRAPPER
chmod +x "${toolchain}/clang"
"${toolchain}/clang" --version >&2
printf '%s\n' "${toolchain}/clang"
