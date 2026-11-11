#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Multi-backend development environment bootstrap for FlagGems-vllm.
#
# The backend list and every vendor-specific version live in
# `src/flaggems_vllm/backends.yaml`. This script is a thin orchestrator on top
# of that registry, mirroring FlagGems' `setup.sh <backend>` workflow:
#
#   uv-managed Python/venv -> vendor env -> vendor torch stack -> vLLM ->
#   FlagGems -> FlagGems-vllm (editable) -> compiler -> smoke checks
#
# Two installation modes are supported:
#
#   * virtualenv mode (default): create/refresh a uv virtual environment,
#     install everything into it, and write the vendor environment into
#     `.venv/bin/activate` so a later `source .venv/bin/activate` is enough.
#
#   * `--no-venv`: install this repository and its test dependencies into the
#     *current* Python environment. Vendor torch/triton/vLLM packages are left
#     untouched by default. This is the supported path inside vendor-provided
#     vLLM images (Ascend, MetaX, Hygon, Moore Threads, ...), where the image
#     already ships a matched torch/triton/vLLM stack.
#
# Usage:
#   bash tools/setup.sh [BACKEND] [OPTIONS]
#   source tools/setup.sh [BACKEND] [OPTIONS]
#
# Run `bash tools/setup.sh --help` for the full option and environment list.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
BACKENDS_FILE="${REPO_ROOT}/src/flaggems_vllm/backends.yaml"

ok()   { printf " ${GREEN}[OK]${NC}\n"; }
warn() { printf " ${YELLOW}[WARN]${NC} %s\n" "$1"; }
die()  { printf " ${RED}[FAILED]${NC} %s\n" "$1" >&2; exit 1; }
is_sourced() { [ "${BASH_SOURCE[0]}" != "${0}" ]; }

# `uv`'s --torch-backend=auto probes the local CUDA driver and silently drops
# torch when the driver is newer than uv knows about. FlagGems-vllm therefore
# installs the vendor-qualified torch build from the backend registry instead
# of relying on that probe. These exports make uv installs deterministic.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

# ── Defaults (all overridable through the environment) ──────────────────────
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
VENV_DIR="${VENV_DIR:-.venv}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"
FLAG_GEMS_VERSION="${FLAG_GEMS_VERSION:-}"
FLAG_GEMS_INDEX="${FLAG_GEMS_INDEX:-}"
VLLM_VERSION="${VLLM_VERSION:-}"
VLLM_PACKAGE="${VLLM_PACKAGE:-}"
FLAGTREE_VERSION="${FLAGTREE_VERSION:-}"
TRITON_PACKAGE="${TRITON_PACKAGE:-}"
COMPILER="${COMPILER:-}"
USE_FLAGTREE="${USE_FLAGTREE:-}"
CUDA_HOME="${CUDA_HOME:-}"
NO_VENV="${NO_VENV:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_COMPILER="${SKIP_COMPILER:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_BACKEND_DEPS="${SKIP_BACKEND_DEPS:-}"
SKIP_FLAGGEMS="${SKIP_FLAGGEMS:-0}"
SKIP_PROJECT="${SKIP_PROJECT:-0}"
STRICT_TRITON_IMPORT="${STRICT_TRITON_IMPORT:-0}"
STRICT_PIP_CHECK="${STRICT_PIP_CHECK:-0}"
UV_REFRESH="${UV_REFRESH:-0}"
DRY_RUN="${DRY_RUN:-0}"
VENV_CREATED=0
BACKEND_ARG=""
BACKEND=""
BACKEND_VENDOR=""
BACKEND_VENDOR_FROM_KEY=""

UV_VERSION="${UV_VERSION:-0.11.22}"
UV_MIRROR="${UV_MIRROR:-https://resource.flagos.net/repository/flagos-filestore/utils}"
UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone}"

UV_REFRESH_FLAG=()
if [ "${UV_REFRESH}" = "1" ]; then
  UV_REFRESH_FLAG=(--refresh)
fi

BUILD_DEPS=(
  "setuptools>=64,<82"
  "wheel"
  "packaging"
  "scikit-build-core==0.12.2"
  "pybind11==3.0.3"
  "cmake>=3.20,<4"
  "ninja==1.13.0"
  "PyYAML>=6.0"
)

# ── Help ────────────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $(basename "${SCRIPT_PATH}") [BACKEND] [OPTIONS]

Bootstraps a multi-backend development environment for FlagGems-vllm. The
backend registry lives in src/flaggems_vllm/backends.yaml; every version pin
can be overridden with the environment variables listed below.

Backends:
  Run 'tools/setup.sh --list' for the full list. Common choices:
    nvidia       -> nvidia-cuda133 (default)
    nvidia-cuda128
    ascend       -> ascend-cann900
    ascend-cann850
    hygon        -> hygon-dtk2604
    metax        -> metax-maca3810
    iluvatar     -> iluvatar-corex450
    mthreads     -> mthreads-musa520
    cambricon    -> cambricon-neuware472
    enflame      -> enflame-tops1106
    kunlunxin    -> kunlunxin-xre5371
    thead        -> thead-ppu210
    tsingmicro   -> tsingmicro-tsm260610
    biren, spacemit, sunrise, amd

Options:
  -h, --help            Show this help and exit.
  --list                List backend profiles and exit.
  --no-venv             Install into the current Python environment instead
                        of creating a virtual environment. Vendor deps and the
                        compiler are left untouched by default.
  --skip-vllm           Do not install vLLM. Useful when the vendor image
                        already provides it.
  --skip-compiler       Do not replace/install Triton or FlagTree.
  --skip-smoke          Skip import and pytest collection checks.
  --dry-run             Print the commands without executing them.

Environment:
  PYTHON=${PYTHON_BIN}                 Python used by --no-venv.
  PYTHON_VERSION=${PYTHON_VERSION:-<from backend>}
                        Python version for the virtual environment.
  VENV_DIR=${VENV_DIR}                 Virtual environment directory
                        (relative paths are resolved against the repo root).
  PIP_INDEX_URL=${PIP_INDEX_URL:-<registry mirror>}
                        Primary package index.
  FLAG_GEMS_VERSION=${FLAG_GEMS_VERSION:-<from registry>}
                        Pin flag-gems; empty uses the registry default.
  FLAG_GEMS_INDEX=${FLAG_GEMS_INDEX:-<from registry>}
                        Index that publishes flag-gems.
  VLLM_VERSION=${VLLM_VERSION:-<from backend>}
                        Override the backend's 'vllm==...' pin.
  VLLM_PACKAGE=${VLLM_PACKAGE:-<from backend>}
                        Complete vLLM requirement (may be a wheel URL). Wins
                        over VLLM_VERSION.
  COMPILER=${COMPILER:-<auto>}         flagtree | triton | keep.
  USE_FLAGTREE=${USE_FLAGTREE:-<auto>}  1 = force FlagTree, 0 = force Triton.
  FLAGTREE_VERSION=${FLAGTREE_VERSION:-<from backend>}
                        Override the FlagTree version.
  TRITON_PACKAGE=${TRITON_PACKAGE:-<from backend>}
                        Override the Triton requirement.
  SKIP_BACKEND_DEPS=${SKIP_BACKEND_DEPS:-<auto>}
                        1 = do not install the vendor torch stack. Defaults to
                        1 in --no-venv mode, 0 otherwise.
  SKIP_FLAGGEMS=${SKIP_FLAGGEMS}      1 = do not install flag-gems.
  SKIP_PROJECT=${SKIP_PROJECT}        1 = do not install FlagGems-vllm.
  STRICT_TRITON_IMPORT=${STRICT_TRITON_IMPORT}
                        1 = fail if 'import triton' fails.
  STRICT_PIP_CHECK=${STRICT_PIP_CHECK}
                        1 = fail if dependency resolution reports conflicts.
  UV_REFRESH=${UV_REFRESH}            1 = force uv cache revalidation.
  CUDA_HOME=${CUDA_HOME:-<auto>}      CUDA toolkit path for NVIDIA backends.

Examples:
  bash tools/setup.sh nvidia
  bash tools/setup.sh ascend
  bash tools/setup.sh --no-venv ascend        # inside a vllm-ascend image
  source tools/setup.sh nvidia                # activate .venv in this shell
  bash tools/setup.sh --list
EOF
}

# ── Bare-metal YAML helpers (run before a venv/PyYAML exists) ───────────────
list_backends() {
  awk '/^  [a-z0-9][a-z0-9-]*:$/ {
         key = $0
         sub(/^  /, "", key)
         sub(/:$/, "", key)
         print key
       }' "${BACKENDS_FILE}"
}

backend_exists() {
  awk -v backend="$1" '
    $0 == "  " backend ":" { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "${BACKENDS_FILE}"
}

backend_python_for() {
  awk -v backend="$1" '
    $0 == "  " backend ":" { in_backend = 1; next }
    in_backend && /^  [a-z0-9][a-z0-9-]*:$/ { exit }
    in_backend && /^    python:/ {
      sub(/^    python:[[:space:]]*/, "")
      print
      exit
    }
  ' "${BACKENDS_FILE}" | tr -d '"'
}

global_value() {
  awk -v key="$1" '
    $0 ~ "^" key ":" {
      sub("^" key ":[[:space:]]*", "")
      print
      exit
    }
  ' "${BACKENDS_FILE}" | tr -d '"'
}

alias_for() {
  awk -v name="$1" '
    /^aliases:/ { in_aliases = 1; next }
    /^[^[:space:]]/ { in_aliases = 0 }
    in_aliases && $1 == name ":" { print $2; exit }
  ' "${BACKENDS_FILE}"
}

list_backends_detailed() {
  local key python
  printf "Available backends (from src/flaggems_vllm/backends.yaml):\n"
  while IFS= read -r key; do
    python="$(backend_python_for "${key}")"
    [ -n "${python}" ] || python="$(global_value python)"
    printf "  %-24s vendor=%-12s python=%s\n" "${key}" "${key%%-*}" "${python}"
  done < <(list_backends)
  printf "\nAliases:\n"
  awk '
    /^aliases:/ { in_aliases = 1; next }
    /^[^[:space:]]/ { in_aliases = 0 }
    in_aliases && $1 ~ /:$/ {
      name = $1
      sub(/:$/, "", name)
      printf "  %-24s -> %s\n", name, $2
    }
  ' "${BACKENDS_FILE}"
  printf "\nDefault: %s\n" "$(global_value default_backend)"
}

resolve_backend() {
  local requested="${BACKEND_ARG:-${DNN_VENDOR:-${FLAGGEMS_VENDOR:-}}}"
  local default_backend alias target matches

  default_backend="$(global_value default_backend)"
  if [ -z "${default_backend}" ]; then
    die "src/flaggems_vllm/backends.yaml is missing default_backend"
  fi

  if [ -z "${requested}" ]; then
    BACKEND="${default_backend}"
  elif [ "${requested}" = "default" ]; then
    BACKEND="${default_backend}"
  elif backend_exists "${requested}"; then
    BACKEND="${requested}"
  else
    alias="$(alias_for "${requested}")"
    if [ -n "${alias}" ]; then
      BACKEND="${alias}"
    else
      matches="$(list_backends | awk -v name="${requested}" '$0 ~ "^" name "-"')"
      if [ "$(printf '%s\n' "${matches}" | sed '/^$/d' | wc -l)" -eq 1 ]; then
        BACKEND="${matches}"
      else
        printf "Unknown backend '%s'.\n\n" "${requested}" >&2
        list_backends_detailed >&2
        exit 1
      fi
    fi
  fi

  backend_exists "${BACKEND}" \
    || die "src/flaggems_vllm/backends.yaml has no entry for '${BACKEND}'"

  BACKEND_VENDOR_FROM_KEY="${BACKEND%%-*}"
  [ -n "${BACKEND_VENDOR_FROM_KEY}" ] || BACKEND_VENDOR_FROM_KEY="${BACKEND}"
}

# ── Command helpers ─────────────────────────────────────────────────────────
install_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    printf " $(uv --version)"
    ok
    return 0
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    printf " [dry-run] install uv %s\n" "${UV_VERSION}"
    return 0
  fi
  printf " not found, installing ...\n"
  local arch
  arch="$(uname -m)"
  mkdir -p "${HOME}/.local/bin"
  curl -sSf "${UV_MIRROR}/uv-${arch}-${UV_VERSION}-linux-gnu.tar.gz" \
    | tar xz --strip-components=1 -C "${HOME}/.local/bin" 2>/dev/null \
    || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  command -v uv >/dev/null 2>&1 || die "uv installation failed"
  printf "Installed $(uv --version)"
  ok
  # Persist PATH for subsequent GitHub Actions steps.
  if [ -n "${GITHUB_PATH:-}" ]; then
    echo "${HOME}/.local/bin" >> "${GITHUB_PATH}"
  fi
}

install_python() {
  printf "Installing Python ${PYTHON_VERSION} ..."
  if [ "${DRY_RUN}" = "1" ]; then
    printf " [dry-run] uv python install %s\n" "${PYTHON_VERSION}"
    return 0
  fi
  UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR}" \
    uv python install "${PYTHON_VERSION}" \
      --python-preference only-managed -q \
    || die "unable to install Python ${PYTHON_VERSION}"
  ok
}

create_venv() {
  printf "Creating virtual environment (%s) ..." "${VENV_DIR}"
  if [ "${DRY_RUN}" = "1" ]; then
    printf " [dry-run] uv venv %s --python %s\n" "${VENV_DIR}" "${PYTHON_VERSION}"
    return 0
  fi
  uv venv "${VENV_DIR}" \
    --python "${PYTHON_VERSION}" \
    --python-preference only-managed \
    --seed \
    -q \
    || die "unable to create virtual environment at ${VENV_DIR}"
  VENV_CREATED=1
  ok
}

# Install packages into the active venv (uv) or the current interpreter (pip).
# `--reinstall` is translated to pip's `--force-reinstall`.
#
# `--index-mode=flagos` prefers the per-vendor FlagOS index and falls back to
# the mirror. It is used for the vendor torch stack and the compiler. All other
# installs (build tools, flag-gems, vLLM, this repository) use the mirror only:
# the FlagOS nvidia index publishes vLLM variants that can shadow the plain
# PyPI release and make the request unsatisfiable.
install_packages() {
  local force=0 index_mode="mirror" arg
  local -a args=()
  for arg in "$@"; do
    case "${arg}" in
      --reinstall) force=1 ;;
      --index-mode=flagos) index_mode="flagos" ;;
      --index-mode=flag-gems) index_mode="flag-gems" ;;
      --index-mode=mirror) index_mode="mirror" ;;
      *) args+=("${arg}") ;;
    esac
  done
  [ "${#args[@]}" -gt 0 ] || return 0

  if [ "${DRY_RUN}" = "1" ]; then
    printf "[dry-run] install (%s index)" "${index_mode}"
    printf " %q" "${args[@]}"
    printf "\n"
    return 0
  fi

  if [ "${NO_VENV}" = "1" ]; then
    local -a extra=()
    if [ "${force}" = "1" ]; then
      extra+=(--force-reinstall)
    fi
    local -a index_args=(--index-url "${MIRROR}")
    if [ "${index_mode}" = "flagos" ]; then
      index_args=(--index-url "${FLAGOS_PYPI}" --extra-index-url "${MIRROR}")
    elif [ "${index_mode}" = "flag-gems" ]; then
      index_args=(--index-url "${FLAG_GEMS_INDEX}" --extra-index-url "${MIRROR}")
    fi
    "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
      "${index_args[@]}" \
      "${extra[@]+"${extra[@]}"}" \
      "${args[@]}"
  else
    local -a extra=("${UV_REFRESH_FLAG[@]+"${UV_REFRESH_FLAG[@]}"}")
    if [ "${force}" = "1" ]; then
      extra+=(--reinstall)
    fi
    local -a index_args=(--default-index "${MIRROR}")
    if [ "${index_mode}" = "flagos" ]; then
      index_args=(
        --default-index "${FLAGOS_PYPI}"
        --index "${MIRROR}"
      )
    elif [ "${index_mode}" = "flag-gems" ]; then
      index_args=(
        --default-index "${FLAG_GEMS_INDEX}"
        --index "${MIRROR}"
      )
    fi
    uv pip install --python "${PYTHON_BIN}" --no-config \
      "${index_args[@]}" \
      "${extra[@]+"${extra[@]}"}" \
      "${args[@]}"
  fi
}

uninstall_packages() {
  [ "${#}" -gt 0 ] || return 0
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  local pkg
  for pkg in "$@"; do
    if [ "${NO_VENV}" = "1" ]; then
      "${PYTHON_BIN}" -m pip uninstall -y "${pkg}" >/dev/null 2>&1 || true
    else
      uv pip uninstall --python "${PYTHON_BIN}" --no-config "${pkg}" \
        >/dev/null 2>&1 || true
    fi
  done
}

# Verify that the installed Triton/FlagTree package is complete and importable.
# uv's exit code only says the dist-info was written; it does not catch a
# truncated install whose RECORD files never landed on disk. That leaves
# `import triton` degraded to an empty namespace package (no triton.Config).
verify_triton_install() {
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
import site
import sys

# Restrict the search to this interpreter's own site-packages. A global
# compiler side dir added through PYTHONPATH must not shadow the venv install.
venv_site_packages = site.getsitepackages()[0]

# Vendor compiler distributions use different names (triton, flagtree,
# triton-ascend, triton-gcu, ...), so identify the provider by its files rather
# than by a hard-coded distribution name.
provider = None
for dist in md.distributions(path=[venv_site_packages]):
    for recorded in dist.files or []:
        if str(recorded).startswith("triton/"):
            provider = dist
            break
    if provider is not None:
        break

if provider is not None:
    missing = [str(f) for f in provider.files or [] if not f.locate().exists()]
    if missing:
        print(f"{provider.metadata['Name']}: {len(missing)} recorded file(s) "
              f"missing, e.g. {missing[:5]}")
        sys.exit(1)

# On Ascend, the Triton backend imports torch/torch_npu as a side effect while
# triton is still mid-import; importing torch first avoids a circular import.
import torch  # noqa: F401

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

import triton

if triton.__file__ is None or not hasattr(triton, "Config"):
    print(f"triton import incomplete (__file__={triton.__file__})")
    sys.exit(1)

print("triton install verified OK")
PY
}

# ── Backend configuration ───────────────────────────────────────────────────
parse_backend() {
  local yaml_python
  yaml_python="${PYTHON_BIN}"
  if [ "${DRY_RUN}" = "1" ] && [ "${NO_VENV}" != "1" ]; then
    # No venv is created in dry-run mode; use the host interpreter.
    yaml_python="${PYTHON_BIN}"
  fi

  BACKEND_DEPS=()
  BACKEND_VLLM=()
  BACKEND_TRITON_POST=()
  BACKEND_ENV_SOURCE=()
  BACKEND_ENV_KEYS=()
  BACKEND_ENV_VALS=()
  BACKEND_TEST_DEPS=()
  PROJECT_TEST_DEPS=()
  PROJECT_RUNTIME_DEPS=()

  eval "$("${yaml_python}" - "${BACKENDS_FILE}" "${BACKEND}" "${CONSTRAINTS_FILE}" <<'PY'
import re
import shlex
import sys

import yaml

path, backend, constraints_path = sys.argv[1:4]
with open(path) as f:
    cfg = yaml.safe_load(f)
entry = cfg["backends"][backend]


def emit_value(name, value):
    if value is None:
        value = ""
    print(f"{name}={shlex.quote(str(value))}")


def emit_array(name, values):
    values = [str(v) for v in (values or [])]
    print(f"{name}=(")
    for value in values:
        print(f"  {shlex.quote(value)}")
    print(")")


emit_value("BACKEND_PYTHON", entry.get("python", cfg.get("python", "3.12")))
emit_value("BACKEND_VENDOR", entry.get("vendor", backend.split("-")[0]))
emit_value("BACKEND_TRITON", entry.get("triton", ""))
emit_value("BACKEND_FLAGTREE", entry.get("flagtree", ""))
emit_value("BACKEND_VLLM_NOTE", entry.get("vllm_note", ""))
emit_array("BACKEND_DEPS", entry.get("deps"))
emit_array("BACKEND_VLLM", entry.get("vllm"))
emit_array("BACKEND_TRITON_POST", entry.get("triton_post_install"))

emit_array("BACKEND_TEST_DEPS", entry.get("test_deps"))
emit_array("PROJECT_TEST_DEPS", cfg.get("test_deps"))
emit_array("PROJECT_RUNTIME_DEPS", cfg.get("runtime_deps"))

emit_array("BACKEND_ENV_SOURCE", entry.get("env_source"))
env = entry.get("env") or {}
emit_array("BACKEND_ENV_KEYS", list(env.keys()))
emit_array("BACKEND_ENV_VALS", list(env.values()))

# Vendor torch/accelerator pins become constraints so that installing vLLM,
# FlagGems or this repository cannot silently replace the vendor stack.
constraints = ["setuptools>=64,<82", "cmake>=3.20,<4"]
for dep in entry.get("deps") or []:
    dep = str(dep).strip()
    # Skip editable paths, local files and options; registry deps are PEP 508
    # requirement strings or direct URLs.
    if not dep or dep.startswith("-") or dep.startswith("/"):
        continue
    if " @ " in dep:
        continue
    constraints.append(dep)

with open(constraints_path, "w") as f:
    f.write("\n".join(constraints) + "\n")
PY
)"
}

apply_env() {
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
  export C_INCLUDE_PATH="${C_INCLUDE_PATH:-}"
  export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:-}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${LIBRARY_PATH:-}"
  export PYTHONPATH="${PYTHONPATH:-}"
  export VIRTUAL_ENV="${VIRTUAL_ENV:-}"

  local i key value script
  for i in "${!BACKEND_ENV_KEYS[@]}"; do
    key="${BACKEND_ENV_KEYS[$i]}"
    value="${BACKEND_ENV_VALS[$i]}"
    # The registry values intentionally contain $PATH / $LD_LIBRARY_PATH.
    # eval expands them; the registry is repository-owned input.
    eval "export ${key}=${value}"
  done

  for script in "${BACKEND_ENV_SOURCE[@]+"${BACKEND_ENV_SOURCE[@]}"}"; do
    if [ ! -f "${script}" ]; then
      warn "vendor environment script not found: ${script}"
      continue
    fi
    if [ "${DRY_RUN}" = "1" ]; then
      printf "[dry-run] source %s\n" "${script}"
      continue
    fi
    # Vendor scripts frequently reference unset variables; relax nounset while
    # they run. Keep errexit so a broken vendor install is visible.
    set +u
    # shellcheck disable=SC1090
    source "${script}" \
      || die "failed to source vendor environment script: ${script}"
    set -u
  done
}

setup_cuda_env() {
  [ "${BACKEND_VENDOR_FROM_KEY}" = "nvidia" ] || return 0

  if [ -z "${CUDA_HOME}" ]; then
    local candidate
    for candidate in /usr/local/cuda /usr/local/cuda-*; do
      if [ -x "${candidate}/bin/nvcc" ]; then
        CUDA_HOME="${candidate}"
        break
      fi
    done
  fi

  if [ -n "${CUDA_HOME}" ]; then
    export CUDA_HOME
    export PATH="${CUDA_HOME}/bin:${PATH}"
    if [ -d "${CUDA_HOME}/lib64" ]; then
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
  fi

  export USE_FLAGCX="${USE_FLAGCX:-OFF}"
}

write_activate_env() {
  if [ "${NO_VENV}" = "1" ] || [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  [ "${VENV_CREATED}" = "1" ] || return 0

  local activate="${VENV_DIR}/bin/activate"
  local helper="${SCRIPT_DIR}/source_env_bash.sh"
  local i key value script

  # Remove a block written by a previous run so activation stays idempotent.
  sed -i \
    '/^# --- FlagGems-vllm environment (/,/^# --- end FlagGems-vllm environment ---/d' \
    "${activate}" 2>/dev/null || true

  {
    printf "\n# --- FlagGems-vllm environment (%s) ---\n" "${BACKEND}"
    # Keep the block safe when the caller enables `set -u`.
    printf ': "${LD_LIBRARY_PATH:=}"\n'
    printf ': "${LIBRARY_PATH:=}"\n'
    printf ': "${PYTHONPATH:=}"\n'
    printf ': "${CMAKE_PREFIX_PATH:=}"\n'
    printf "export FLAGGEMS_VENDOR=%q\n" "${BACKEND_VENDOR}"
    printf "export DNN_VENDOR=%q\n" "${BACKEND_VENDOR}"
    for i in "${!BACKEND_ENV_KEYS[@]}"; do
      key="${BACKEND_ENV_KEYS[$i]}"
      value="${BACKEND_ENV_VALS[$i]}"
      printf "export %s=\"%s\"\n" "${key}" "${value}"
    done
    for script in "${BACKEND_ENV_SOURCE[@]+"${BACKEND_ENV_SOURCE[@]}"}"; do
      printf '[ -f "%s" ] && eval "$(bash "%s" "%s")" || true\n' \
        "${script}" "${helper}" "${script}"
    done
    printf "# --- end FlagGems-vllm environment ---\n"
  } >>"${activate}"
}

# ── Checks ──────────────────────────────────────────────────────────────────
run_smoke_checks() {
  if [ "${SKIP_SMOKE}" = "1" ]; then
    warn "smoke checks skipped (SKIP_SMOKE=1)"
    return 0
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  printf "Verifying imports ..."
  "${PYTHON_BIN}" - "${BACKEND_VENDOR}" <<'PY' || die "import verification failed"
import importlib
import importlib.util
import sys

vendor = sys.argv[1].lower()

# On Ascend, torch_npu must be initialized before triton to avoid a circular
# import (triton -> torch_npu -> torch._dynamo -> triton).
if vendor.startswith("ascend"):
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass

import torch

try:
    import flag_gems
except ImportError:
    flag_gems = None
import flaggems_vllm

print(" torch:", torch.__version__)
print(" torch cuda:", getattr(torch.version, "cuda", None))
if flag_gems is None:
    print(" flag_gems: not installed")
else:
    print(" flag_gems:", getattr(flag_gems, "__version__", "unknown"))
print(" flaggems_vllm:", getattr(flaggems_vllm, "__version__", "unknown"))
print(" vendor:", getattr(flaggems_vllm, "vendor_name", "unknown"))
print(" device:", getattr(flaggems_vllm, "device", "unknown"))

if importlib.util.find_spec("vllm") is not None:
    importlib.import_module("vllm")
    print(" vllm:", getattr(sys.modules["vllm"], "__version__", "unknown"))
else:
    print(" vllm: not installed (vLLM-dependent tests will skip)")
PY
  ok

  if [ "${COMPILER}" = "flagtree" ]; then
    printf "Checking FlagTree TLE support ..."
    "${PYTHON_BIN}" - <<'PY' || die "FlagTree TLE support is unavailable"
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if not has_triton_tle(3, 6, 0):
    raise SystemExit("FlagTree TLE support is unavailable")
print(" has_triton_tle(3, 6, 0): True")
PY
    ok
  fi

  printf "Collecting tests ..."
  local collect_log
  collect_log="$(mktemp "${TMPDIR:-/tmp}/flaggems_vllm_collect.XXXXXX")"
  if ( cd "${REPO_ROOT}" && "${PYTHON_BIN}" -m pytest tests --collect-only -q ) \
      >"${collect_log}" 2>&1; then
    tail -1 "${collect_log}"
    ok
  else
    printf "\n"
    cat "${collect_log}"
    rm -f "${collect_log}"
    die "pytest collection failed"
  fi
  rm -f "${collect_log}"

  printf "Running dependency check ..."
  if [ "${NO_VENV}" = "1" ]; then
    "${PYTHON_BIN}" -m pip check
  else
    uv pip check --python "${PYTHON_BIN}"
  fi && ok || {
    printf "\n"
    if [ "${STRICT_PIP_CHECK}" = "1" ]; then
      die "dependency check failed (STRICT_PIP_CHECK=1)"
    else
      warn "dependency check failed; set STRICT_PIP_CHECK=1 to make it fatal"
    fi
  }
}

# ── Argument parsing ────────────────────────────────────────────────────────
parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        return 10
        ;;
      --list)
        list_backends_detailed
        return 10
        ;;
      --no-venv)
        NO_VENV=1
        ;;
      --skip-vllm)
        SKIP_VLLM=1
        ;;
      --skip-compiler)
        SKIP_COMPILER=1
        ;;
      --skip-smoke)
        SKIP_SMOKE=1
        ;;
      --dry-run)
        DRY_RUN=1
        ;;
      --)
        shift
        break
        ;;
      -*)
        die "unknown option: $1 (try --help)"
        ;;
      *)
        if [ -n "${BACKEND_ARG}" ]; then
          die "multiple backend arguments: '${BACKEND_ARG}' and '$1'"
        fi
        BACKEND_ARG="$1"
        ;;
    esac
    shift
  done

  if [ "$#" -gt 0 ]; then
    BACKEND_ARG="$1"
  fi
  return 0
}

# ── Main ────────────────────────────────────────────────────────────────────
main() {
  local rc=0
  parse_args "$@" || rc=$?
  if [ "${rc}" -eq 10 ]; then
    return 0
  fi

  [ -f "${REPO_ROOT}/pyproject.toml" ] \
    || die "pyproject.toml not found under ${REPO_ROOT}"
  [ -d "${REPO_ROOT}/src/flaggems_vllm" ] \
    || die "src/flaggems_vllm not found under ${REPO_ROOT}"
  [ -f "${BACKENDS_FILE}" ] \
    || die "backend registry not found: ${BACKENDS_FILE}"

  resolve_backend

  case "${VENV_DIR}" in
    /*) ;;
    *) VENV_DIR="${REPO_ROOT}/${VENV_DIR}" ;;
  esac

  MIRROR="${PIP_INDEX_URL:-$(global_value mirror)}"
  [ -n "${MIRROR}" ] || MIRROR="https://pypi.org/simple"
  PYPI_BASE="$(global_value pypi_base)"
  [ -n "${PYPI_BASE}" ] || PYPI_BASE="https://pypi.org/simple"
  BACKEND_VENDOR="${BACKEND_VENDOR_FROM_KEY}"
  FLAGOS_PYPI="${PYPI_BASE//\{vendor\}/${BACKEND_VENDOR}}"

  if [ -z "${PYTHON_VERSION}" ]; then
    PYTHON_VERSION="$(backend_python_for "${BACKEND}")"
    [ -n "${PYTHON_VERSION}" ] || PYTHON_VERSION="$(global_value python)"
    [ -n "${PYTHON_VERSION}" ] || PYTHON_VERSION="3.12"
  fi
  if [ -z "${FLAG_GEMS_VERSION}" ]; then
    FLAG_GEMS_VERSION="$(global_value flag_gems_version)"
  fi
  if [ -z "${FLAG_GEMS_INDEX}" ]; then
    FLAG_GEMS_INDEX="$(global_value flag_gems_index)"
  fi
  [ -n "${FLAG_GEMS_INDEX}" ] || FLAG_GEMS_INDEX="${FLAGOS_PYPI}"

  printf "Backend: ${CYAN}%s${NC} (vendor: %s, python: %s)" \
    "${BACKEND}" "${BACKEND_VENDOR}" "${PYTHON_VERSION}"
  ok

  CONSTRAINTS_FILE="$(mktemp "${TMPDIR:-/tmp}/flaggems_vllm_constraints.XXXXXX")"
  trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT

  if [ "${NO_VENV}" = "1" ]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
      || die "Python interpreter not found: ${PYTHON_BIN}"
    printf "Mode: current environment (%s)\n" \
      "$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "${PYTHON_BIN}")"
    if ! "${PYTHON_BIN}" -c 'import yaml' >/dev/null 2>&1; then
      printf "PyYAML is missing; installing it first ..."
      install_packages "PyYAML>=6.0"
      ok
    fi
  else
    install_uv
    install_python
    create_venv
    if [ "${DRY_RUN}" != "1" ]; then
      PYTHON_BIN="${VENV_DIR}/bin/python"
      [ -x "${PYTHON_BIN}" ] || die "virtual environment is missing ${PYTHON_BIN}"
    fi
    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    printf "Python: %s" "$("${PYTHON_BIN}" --version 2>&1 || true)"
    ok
    printf "Installing build tools ..."
    install_packages "${BUILD_DEPS[@]}"
    ok
  fi

  parse_backend
  [ -n "${BACKEND_VENDOR}" ] || BACKEND_VENDOR="${BACKEND_VENDOR_FROM_KEY}"
  FLAGOS_PYPI="${PYPI_BASE//\{vendor\}/${BACKEND_VENDOR}}"

  setup_cuda_env
  export FLAGGEMS_VENDOR="${BACKEND_VENDOR}"
  export DNN_VENDOR="${BACKEND_VENDOR}"
  apply_env

  if [ -z "${SKIP_BACKEND_DEPS}" ]; then
    if [ "${NO_VENV}" = "1" ]; then
      SKIP_BACKEND_DEPS=1
    else
      SKIP_BACKEND_DEPS=0
    fi
  fi

  if [ "${SKIP_BACKEND_DEPS}" = "1" ]; then
    warn "vendor torch/accelerator deps left untouched (SKIP_BACKEND_DEPS=1)"
  elif [ "${#BACKEND_DEPS[@]}" -eq 0 ]; then
    if [ "${NO_VENV}" != "1" ]; then
      die "No vendor torch stack is configured for '${BACKEND}'. Install the vendor PyTorch/Triton stack first and rerun with --no-venv, or add 'deps' to src/flaggems_vllm/backends.yaml."
    fi
    warn "no vendor torch stack configured for ${BACKEND}"
  else
    printf "Installing vendor dependencies for ${BACKEND} ...\n"
    install_packages --index-mode=flagos "${BACKEND_DEPS[@]}" \
      --constraint "${CONSTRAINTS_FILE}"
    ok
  fi

  if [ "${SKIP_FLAGGEMS}" = "1" ]; then
    warn "skipping flag-gems installation (SKIP_FLAGGEMS=1)"
  elif [ "${NO_VENV}" = "1" ] && "${PYTHON_BIN}" -c 'import flag_gems' >/dev/null 2>&1; then
    printf "flag_gems already available in the current environment"
    ok
  else
    printf "Installing FlagGems%s ..." \
      "${FLAG_GEMS_VERSION:+==${FLAG_GEMS_VERSION}}"
    # --no-deps keeps flag-gems' strict transitive pins (numpy, packaging,
    # sqlalchemy) from replacing the vendor stack. The runtime dependencies we
    # actually need are installed explicitly below.
    install_packages --index-mode=flag-gems --no-deps \
      "flag-gems${FLAG_GEMS_VERSION:+==${FLAG_GEMS_VERSION}}" \
      --constraint "${CONSTRAINTS_FILE}"
    ok
  fi

  if [ "${#PROJECT_RUNTIME_DEPS[@]}" -gt 0 ]; then
    printf "Installing pure-Python runtime dependencies ..."
    install_packages "${PROJECT_RUNTIME_DEPS[@]}" \
      --constraint "${CONSTRAINTS_FILE}"
    ok
  fi

  if [ "${SKIP_VLLM}" = "1" ] || { [ "${NO_VENV}" = "1" ] && [ -z "${VLLM_PACKAGE}" ]; }; then
    warn "vLLM installation skipped; vLLM-dependent tests will skip"
    if [ -n "${BACKEND_VLLM_NOTE}" ]; then
      warn "${BACKEND_VLLM_NOTE}"
    fi
  else
    local -a vllm_specs=()
    if [ -n "${VLLM_PACKAGE}" ]; then
      # Treat the override as one PEP 508 requirement, so a direct reference
      # such as "pkg @ https://.../pkg.whl" is not split on spaces.
      vllm_specs=("${VLLM_PACKAGE}")
    elif [ -n "${VLLM_VERSION}" ] && [ "${BACKEND_VLLM[0]:-}" != "" ] \
        && [[ "${BACKEND_VLLM[0]}" == vllm==* ]]; then
      vllm_specs=("vllm==${VLLM_VERSION}")
    else
      vllm_specs=("${BACKEND_VLLM[@]+"${BACKEND_VLLM[@]}"}")
    fi

    if [ "${#vllm_specs[@]}" -eq 0 ]; then
      warn "no vLLM package is configured for ${BACKEND}"
      if [ -n "${BACKEND_VLLM_NOTE}" ]; then
        warn "${BACKEND_VLLM_NOTE}"
      fi
    else
      printf "Installing vLLM packages ...\n"
      install_packages "${vllm_specs[@]}" --constraint "${CONSTRAINTS_FILE}"
      ok
    fi
  fi

  if [ "${SKIP_PROJECT}" = "1" ]; then
    warn "skipping FlagGems-vllm installation (SKIP_PROJECT=1)"
  else
    printf "Installing FlagGems-vllm editable (--no-deps) ..."
    # Install the project without resolving its dependencies: flag-gems is
    # already installed, and its metadata pins conflict with several vendor
    # numpy versions. Runtime and test dependencies are installed explicitly.
    install_packages -e "${REPO_ROOT}" --no-build-isolation --no-deps \
      --constraint "${CONSTRAINTS_FILE}"
    ok

    if [ "${#PROJECT_TEST_DEPS[@]}" -gt 0 ] || [ "${#BACKEND_TEST_DEPS[@]}" -gt 0 ]; then
      printf "Installing test dependencies ..."
      install_packages \
        "${PROJECT_TEST_DEPS[@]+"${PROJECT_TEST_DEPS[@]}"}" \
        "${BACKEND_TEST_DEPS[@]+"${BACKEND_TEST_DEPS[@]}"}" \
        --constraint "${CONSTRAINTS_FILE}"
      ok
    fi
  fi

  if [ "${SKIP_COMPILER}" = "1" ]; then
    COMPILER="keep"
  fi
  if [ -z "${COMPILER}" ]; then
    if [ "${NO_VENV}" = "1" ]; then
      COMPILER="keep"
    elif [ "${USE_FLAGTREE}" = "0" ]; then
      COMPILER="triton"
    elif [ "${USE_FLAGTREE}" = "1" ]; then
      COMPILER="flagtree"
    elif [ -n "${BACKEND_FLAGTREE}" ]; then
      COMPILER="flagtree"
    elif [ -n "${BACKEND_TRITON}" ]; then
      COMPILER="triton"
    else
      COMPILER="keep"
    fi
  fi

  local flagtree_pkgs triton_pkgs site_packages attempt
  flagtree_pkgs="${FLAGTREE_VERSION:+flagtree==${FLAGTREE_VERSION}}"
  [ -n "${flagtree_pkgs}" ] || flagtree_pkgs="${BACKEND_FLAGTREE}"
  triton_pkgs="${TRITON_PACKAGE:-${BACKEND_TRITON}}"
  site_packages="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"

  case "${COMPILER}" in
    keep)
      warn "compiler left untouched (COMPILER=keep); FLAGGEMS_VENDOR=${BACKEND_VENDOR}"
      ;;
    flagtree)
      [ -n "${flagtree_pkgs}" ] \
        || die "COMPILER=flagtree but ${BACKEND} has no flagtree package"
      printf "Installing FlagTree (%s) ...\n" "${flagtree_pkgs}"
      uninstall_packages triton
      for attempt in 1 2 3; do
        if install_packages --index-mode=flagos --reinstall "${flagtree_pkgs}" \
             --constraint "${CONSTRAINTS_FILE}" \
           && verify_triton_install; then
          ok
          break
        fi
        printf " ${RED}[incomplete]${NC}, cleaning cache and retrying ...\n"
        uninstall_packages flagtree
        [ "${NO_VENV}" = "1" ] || uv cache clean flagtree >/dev/null 2>&1 || true
        if [ -n "${site_packages}" ]; then
          rm -rf "${site_packages}/triton"
        fi
        if [ "${attempt}" -eq 3 ]; then
          die "FlagTree installation failed"
        fi
      done
      ;;
    triton)
      [ -n "${triton_pkgs}" ] \
        || die "COMPILER=triton but ${BACKEND} has no triton package"
      printf "Installing Triton (%s) ...\n" "${triton_pkgs}"
      uninstall_packages flagtree
      for attempt in 1 2 3; do
        if install_packages --index-mode=flagos --reinstall "${triton_pkgs}" \
             --constraint "${CONSTRAINTS_FILE}" \
           && verify_triton_install; then
          ok
          break
        fi
        printf " ${RED}[incomplete]${NC}, cleaning cache and retrying ...\n"
        uninstall_packages triton
        [ "${NO_VENV}" = "1" ] || uv cache clean triton >/dev/null 2>&1 || true
        if [ -n "${site_packages}" ]; then
          rm -rf "${site_packages}/triton"
        fi
        if [ "${attempt}" -eq 3 ]; then
          die "Triton installation failed"
        fi
      done
      if [ "${#BACKEND_TRITON_POST[@]}" -gt 0 ]; then
        printf "Installing Triton post-install packages ...\n"
        install_packages --index-mode=flagos "${BACKEND_TRITON_POST[@]}" \
          --constraint "${CONSTRAINTS_FILE}"
      fi
      ;;
    *)
      die "unknown COMPILER='${COMPILER}' (expected flagtree, triton or keep)"
      ;;
  esac

  if [ "${STRICT_TRITON_IMPORT}" = "1" ] && [ "${COMPILER}" != "keep" ]; then
    printf "Strictly checking Triton import ..."
    "${PYTHON_BIN}" -c 'import triton; print(" triton module:", triton.__file__)' \
      || die "import triton failed (STRICT_TRITON_IMPORT=1)"
    ok
  fi

  run_smoke_checks
  write_activate_env

  printf "\n${GREEN}FlagGems-vllm setup complete for %s.${NC}\n" "${BACKEND}"
  if [ "${NO_VENV}" = "1" ]; then
    printf "Installed into the current Python environment.\n"
    printf "Run tests with: PYTHONPATH=%s/src python -m pytest tests -q --quick\n" \
      "${REPO_ROOT}"
  else
    printf "Activate with: source %s/bin/activate\n" "${VENV_DIR}"
    printf "\nQuick validation:\n"
    printf "  pytest tests --collect-only\n"
    printf "  pytest tests -q --quick\n"
    printf "  pytest benchmark --collect-only\n"
    if is_sourced && [ "${DRY_RUN}" != "1" ]; then
      # shellcheck disable=SC1091
      source "${VENV_DIR}/bin/activate"
    fi
  fi
}

main "$@"
