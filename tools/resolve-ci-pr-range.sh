#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

set -euo pipefail

# Both CI jobs check GitHub's synthetic merge. Pin the target to that checkout:
# the event base can be stale, and fetching main would race the checked tree.
read -r merge_sha target_base actual_head extra < <(git rev-list --parents -n 1 HEAD)
if [[ "${merge_sha}" != "${GITHUB_SHA}" || -z "${target_base}" || \
      -z "${actual_head}" || -n "${extra}" || "${actual_head}" != "${PR_HEAD_SHA}" ]]; then
  echo "::error::Expected the checked-out two-parent PR merge with the PR head as second parent."
  exit 1
fi
merge_base="$(git merge-base "${target_base}" "${PR_HEAD_SHA}")"
{
  echo "target_base=${target_base}"
  echo "head=${PR_HEAD_SHA}"
  echo "merge_base=${merge_base}"
} >> "${GITHUB_OUTPUT}"
