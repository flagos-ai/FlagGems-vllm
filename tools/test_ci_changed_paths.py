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

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class ChangedPathsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.git("init", "-b", "main")
        self.git("config", "user.email", "ci@example.invalid")
        self.git("config", "user.name", "CI Test")
        (self.root / "base.py").write_text("# base\n")
        (self.root / "removed.py").write_text("# removed\n")
        self.base = self.commit("base")
        self.workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/basic-ci.yml"
        ).read_text()

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.root, text=True, stderr=subprocess.PIPE
        ).strip()

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def run_selection(self, name, env):
        section = self.workflow.split("- name: " + name + "\n", 1)[1]
        script = textwrap.dedent(
            section.split("        run: |\n", 1)[1].split("\n\n", 1)[0]
        )
        # Run the workflow's actual Git selection, before downstream processing.
        script = script.split("files=()", 1)[0]
        result = subprocess.run(
            ["bash", "-e", "-c", script],
            cwd=self.root,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = "changed_py.bin" if "changed_py.bin" in script else "changed_files.bin"
        return set((self.root / output).read_bytes().rstrip(b"\0").split(b"\0"))

    def selections(self, env):
        return {
            name: self.run_selection(name, env)
            for name in (
                "check no torch compute fallback in changed files",
                "collect ci selection inputs",
            )
        }

    def test_pr_ignores_main_updates_in_merge_checkout(self):
        self.git("checkout", "-b", "topic")
        (self.root / "test 空格.py").write_text("# topic\n")
        (self.root / "removed.py").unlink()
        head = self.commit("topic changes")
        self.git("checkout", "main")
        (self.root / "unrelated_moe.py").write_text("# new main change\n")
        self.commit("main advanced after PR event")
        self.git("merge", "--no-ff", "topic", "-m", "synthetic merge")
        results = self.selections(
            dict(
                EVENT_NAME="pull_request",
                PR_BASE_SHA=self.base,
                PR_HEAD_SHA=head,
                GITHUB_SHA=self.git("rev-parse", "HEAD"),
                BEFORE_SHA="",
            )
        )
        self.assertEqual(
            results["check no torch compute fallback in changed files"],
            {"test 空格.py".encode()},
        )
        self.assertEqual(
            results["collect ci selection inputs"],
            {"test 空格.py".encode(), b"removed.py"},
        )

    def test_push_preserves_before_to_head_diff(self):
        (self.root / "push.py").write_text("# push\n")
        head = self.commit("push")
        for paths in self.selections(
            dict(
                EVENT_NAME="push",
                BEFORE_SHA=self.base,
                GITHUB_SHA=head,
                PR_BASE_SHA="",
                PR_HEAD_SHA="",
            )
        ).values():
            self.assertEqual(paths, {b"push.py"})

    def test_initial_push_uses_commit_files(self):
        (self.root / "new.py").write_text("# new\n")
        head = self.commit("initial push")
        for paths in self.selections(
            dict(
                EVENT_NAME="push",
                BEFORE_SHA="0" * 40,
                GITHUB_SHA=head,
                PR_BASE_SHA="",
                PR_HEAD_SHA="",
            )
        ).values():
            self.assertEqual(paths, {b"new.py"})
