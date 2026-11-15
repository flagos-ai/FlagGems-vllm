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

import tempfile
import textwrap
import unittest
from pathlib import Path

from check_workflow_policy import check_workflows

GOOD = """
jobs:
  select:
    runs-on: ubuntu-latest
    timeout-minutes: 5
  tests:
    needs: select
    uses: ./.github/workflows/backend-test.yaml
  observed:
    runs-on: some-runner
    continue-on-error: true
  unittest-required:
    if: always()
    needs: [select, tests]
    runs-on: ubuntu-latest
    timeout-minutes: 5
"""


class WorkflowPolicyTest(unittest.TestCase):
    def run_policy(self, text, name="basic-ci.yml"):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / name).write_text(textwrap.dedent(text), encoding="utf-8")
            return check_workflows(tmp)

    def test_compliant_workflow_passes(self):
        self.assertEqual(self.run_policy(GOOD), [])

    def test_hosted_job_without_timeout_fails(self):
        text = GOOD.replace("    timeout-minutes: 5\n  tests", "  tests", 1)
        errors = self.run_policy(text)
        self.assertTrue(any("'select'" in e and "timeout" in e for e in errors))

    def test_self_hosted_job_may_omit_timeout(self):
        self.assertEqual(self.run_policy(GOOD), [])

    def test_job_missing_from_gate_fails(self):
        text = GOOD.replace("needs: [select, tests]", "needs: [select]")
        errors = self.run_policy(text)
        self.assertTrue(any("'tests'" in e and "needs" in e for e in errors))

    def test_observed_job_may_stay_out_of_gate(self):
        errors = self.run_policy(GOOD)
        self.assertFalse(any("'observed'" in e for e in errors))

    def test_gate_must_run_always(self):
        text = GOOD.replace("if: always()", "if: success()")
        errors = self.run_policy(text)
        self.assertTrue(any("always()" in e for e in errors))

    def test_missing_gate_fails(self):
        text = "jobs:\n  a:\n    runs-on: x\n"
        errors = self.run_policy(text)
        self.assertTrue(any("missing gate" in e for e in errors))

    def test_gate_with_unknown_need_fails(self):
        text = GOOD.replace("[select, tests]", "[select, tests, ghost]")
        errors = self.run_policy(text)
        self.assertTrue(any("'ghost'" in e for e in errors))

    def test_gate_rule_only_applies_to_basic_ci(self):
        text = "jobs:\n  a:\n    runs-on: x\n"
        self.assertEqual(self.run_policy(text, name="other.yml"), [])

    def test_repository_workflows_comply(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(check_workflows(root / ".github" / "workflows"), [])


if __name__ == "__main__":
    unittest.main()
