#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
set -euo pipefail

# Extract the Jammy Clang 22.1.8 device compiler and its runtime dependency
# packages from LLVM's official APT archive. TLE uses clang to emit device IR,
# so it does not need a linker or system-level package installation.
toolchain="$(cd -- "${1:?Expected a private toolchain directory}" && pwd -P)"
archive="https://apt.llvm.org/jammy/pool/main/l/llvm-toolchain-22"
version="22.1.8~++20260613092327+e80beda6e255-1~exp1~20260613092437.81"

extract_package() {
    local url=$1 checksum=$2 deb
    deb="${toolchain}/${url##*/}"
    curl --fail --silent --show-error --location --retry 3 --output "$deb" "$url"
    printf '%s  %s\n' "$checksum" "$deb" | sha256sum --check --quiet >&2
    dpkg-deb --extract "$deb" "$toolchain"
    rm -f -- "$deb"
}

while read -r package checksum; do
    extract_package "${archive}/${package}_${version}_amd64.deb" "$checksum"
done <<'PACKAGES'
clang-22 0153d60799f66344f4b1d13e0de28edaf25aed384453b3ac2e5dd15e06ba70e7
libclang-common-22-dev d3a8cd0089d4756f045aea2f320c35dd372ca88c774e3e90c9645b6a404110f7
libclang-cpp22 819cccae27885e70ecb23c4a42065ce50bba8c62740b6728f637698de76faa3d
libllvm22 4bd592cab11e90e5b8625b5e3a1ceb5936e843c63bf7394f6f6506aa41302164
PACKAGES
# Ubuntu Jammy runtime dependency of libllvm22.
extract_package "https://archive.ubuntu.com/ubuntu/pool/universe/z/z3/libz3-4_4.8.12-1_amd64.deb" \
    d83fda4b2e58d9b30f0f85b02a3eb48ca7d33ca88e9d4584249e698bc2e451cf

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
