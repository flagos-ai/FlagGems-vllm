#!/usr/bin/env bash
# Build only M3 CANN + independent dispatcher into the operator package.
set -euo pipefail
soc=${1:?usage: bash tools/build_minimax_m3_ascend.sh ascend910_93 [build-dir]}
case "$soc" in ascend910b|ascend910_93) ;; *) echo "Unsupported SOC: $soc" >&2; exit 2;; esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
native="$repo/csrc/minimax_m3_ascend"
build_dir=${2:-"$repo/build/minimax_m3_ascend"}
python_bin=${PYTHON:-python3}
package="$repo/src/flaggems_vllm/_minimax_m3_ascend"
mkdir -p "$build_dir" "$package/opp"
# CANN project build outputs stay inside this dedicated native source subtree.
cd "$native"
bash build.sh --pkg --ops=flaggems_m3_sparse_attention_score --soc="$soc" --vendor_name=flaggems_m3
shopt -s nullglob
installers=(build/cann-ops-transformer*.run)
if [[ ${#installers[@]} != 1 ]]; then echo "Expected one M3 OPP package" >&2; exit 3; fi
bash "${installers[0]}" --install-path="$package/opp"
torch_prefix=$("$python_bin" -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S "$native/bridge" -B "$build_dir/bridge" \
  -DCMAKE_PREFIX_PATH="$torch_prefix" -DPython_EXECUTABLE="$python_bin" \
  -DASCEND_HOME_PATH="${ASCEND_HOME_PATH:?source the CANN environment first}" \
  -DCMAKE_INSTALL_PREFIX="$package"
cmake --build "$build_dir/bridge" --parallel "${MAX_JOBS:-4}"
cmake --install "$build_dir/bridge"
printf '%s\n' "$soc" > "$package/SOC_VERSION"
echo "Built isolated FlagGems M3 payload: $package"
