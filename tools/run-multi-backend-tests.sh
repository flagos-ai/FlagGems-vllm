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

set -eo pipefail

BACKEND="${1:?usage: $0 <backend> [pr-id]}"
PR_ID="${2:-}"
VENV_PATH="${FLAGGEMS_VENV:-.venv}"

source "${VENV_PATH}/bin/activate"

# DeviceDetector consumes a vendor name rather than the full setup profile.
export FLAGGEMS_VENDOR="${BACKEND%%-*}"
export PYTHONPATH="${GITHUB_WORKSPACE:-$(pwd)}/src${PYTHONPATH:+:${PYTHONPATH}}"
set -u

echo "Backend: ${BACKEND}"
echo "PR ID: ${PR_ID:-n/a}"

if [[ -n "${CHANGED_FILES:-}" ]]; then
  export CI_TARGETS_JSON="${CHANGED_FILES}"
else
  export CI_TARGETS_JSON='{"schema_version":1,"mode":"skip","tests":[],"benchmarks":[]}'
fi
exec python tools/run_ci_targets.py \
  --backend "${BACKEND}" \
  --capabilities .github/backend-capabilities.json
