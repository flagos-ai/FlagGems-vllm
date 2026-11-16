#!/usr/bin/env python3
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

"""Check that CI workflows follow the shared-runner and gate-job conventions.

Rules:

1. Every job that runs on a GitHub-hosted ``ubuntu-latest`` runner declares
   ``timeout-minutes``, so a wedged step cannot hold a shared runner.
2. ``basic-ci.yml`` has a gate job (``unittest-required``) with
   ``if: always()`` whose ``needs`` list covers every other job. Most jobs are
   skipped for many PRs, so branch protection must require the gate instead of
   the individual jobs. A job with job-level ``continue-on-error: true`` is in
   an observation period and may stay out of the gate.

Exit code is non-zero when a rule is violated.
"""

import argparse
import sys
from pathlib import Path

import yaml

GATE_WORKFLOW = "basic-ci.yml"
GATE_JOB = "unittest-required"
HOSTED_RUNNER = "ubuntu-latest"


def load(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def check_timeouts(name, workflow):
    errors = []
    for job_id, job in (workflow.get("jobs") or {}).items():
        runs_on = job.get("runs-on")
        if HOSTED_RUNNER not in as_list(runs_on):
            continue
        if "timeout-minutes" not in job:
            errors.append(
                f"{name}: job '{job_id}' runs on {HOSTED_RUNNER} "
                "without timeout-minutes"
            )
    return errors


def check_gate(name, workflow):
    jobs = workflow.get("jobs") or {}
    gate = jobs.get(GATE_JOB)
    if gate is None:
        return [f"{name}: missing gate job '{GATE_JOB}'"]

    errors = []
    condition = str(gate.get("if", "")).replace(" ", "")
    if condition not in ("always()", "${{always()}}"):
        errors.append(
            f"{name}: gate job '{GATE_JOB}' must use 'if: always()' "
            "so it reports when a needed job fails or is skipped"
        )

    needs = set(as_list(gate.get("needs")))
    for job_id, job in jobs.items():
        if job_id == GATE_JOB or job.get("continue-on-error") is True:
            continue
        if job_id not in needs:
            errors.append(
                f"{name}: job '{job_id}' is not in the needs of '{GATE_JOB}'; "
                "add it, or mark it continue-on-error while it is observed"
            )
    for job_id in sorted(needs - set(jobs)):
        errors.append(f"{name}: gate needs unknown job '{job_id}'")
    return errors


def check_workflows(workflow_dir):
    errors = []
    paths = sorted(
        path
        for path in Path(workflow_dir).iterdir()
        if path.suffix in (".yml", ".yaml")
    )
    for path in paths:
        workflow = load(path)
        errors.extend(check_timeouts(path.name, workflow))
        if path.name == GATE_WORKFLOW:
            errors.extend(check_gate(path.name, workflow))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflows", default=".github/workflows")
    args = parser.parse_args()

    errors = check_workflows(args.workflows)
    for error in errors:
        print(f"::error::{error}")
    if errors:
        return 1
    print("workflow policy: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
