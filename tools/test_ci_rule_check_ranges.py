# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

"""Run both CI workflows' shell steps against real divergent Git history."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash") and shutil.which("git"),
    "requires POSIX bash and git",
)
class RuleCheckRangesTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    WORKFLOW_PATH = ROOT / ".github/workflows/rule-check.yaml"
    UNIT_WORKFLOW_PATH = ROOT / ".github/workflows/unittest.yaml"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo = Path(self.temporary_directory.name) / "repo"
        self.repo.mkdir()
        self.bin = Path(self.temporary_directory.name) / "bin"
        self.bin.mkdir()
        self.calls = Path(self.temporary_directory.name) / "calls.jsonl"
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "CI_RULE_CALLS": str(self.calls),
            "CI_RULE_PYTHON": sys.executable,
            "CI_RULE_CHECKS": str(self.ROOT / "tools/ci_checks"),
        }
        self.steps = self.workflow_steps(self.WORKFLOW_PATH, "rule-check-required")
        self.unit_steps = self.workflow_steps(self.UNIT_WORKFLOW_PATH, "preprocess")
        self.outputs = {}
        self._install_capture_commands()
        self.git("init", "--initial-branch=main")
        for key, value in {
            "user.name": "CI Range Test",
            "user.email": "ci-range@example.invalid",
            "commit.gpgsign": "false",
            "core.autocrlf": "false",
        }.items():
            self.git("config", key, value)

    @staticmethod
    def workflow_steps(path, job):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {
            step["name"]: step
            for step in workflow["jobs"][job]["steps"]
            if "run" in step and "name" in step
        }

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, env=self.env, text=True
        ).strip()

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def history(self, kind="safe"):
        self.write("README.md", "Initial history.\n")
        self.write("conf/operators.yaml", "ops: []\n")
        self.write(
            "tools/resolve-ci-pr-range.sh",
            (self.ROOT / "tools/resolve-ci-pr-range.sh").read_text(encoding="utf-8"),
        )
        self.stale_base = self.commit("Old PR event base")
        self.write(
            "src/flaggems_vllm/ops/common_legacy.py",
            "import torch\ndef common(x):\n    return torch.add(x, x)\n",
        )
        shared = "# PR note: base\nimport torch\n" + "\n" * 12
        shared += "def merged(x):\n    return x\n"
        if kind == "shared":
            self.write("src/flaggems_vllm/ops/shared.py", shared)
        self.fork = self.commit("Main changes inherited by both branches")
        self.git("branch", "pr")
        self.write(
            "src/flaggems_vllm/ops/main_only.py",
            "import torch\ndef main_only(x):\n    return torch.mul(x, x)\n",
        )
        if kind == "path":
            self.write(
                "src/flaggems_vllm/runtime/backend/_metax/ops/main_only.py",
                "import torch\ndef main_only(x):\n    return torch.mul(x, x)\n",
            )
        if kind == "shared":
            self.write(
                "src/flaggems_vllm/ops/shared.py",
                shared.replace("return x", "return torch.add(x, x)"),
            )
        self.target = self.commit("Later main-only change")
        self.git("checkout", "-q", "pr")
        self.pr_file = "src/flaggems_vllm/ops/pr_change.py"
        if kind == "path":
            self.pr_file = "src/flaggems_vllm/ops/pr change\nwith newline.py"
        if kind == "shared":
            self.pr_file = "src/flaggems_vllm/ops/shared.py"
            self.write(
                self.pr_file, shared.replace("PR note: base", "PR note: changed")
            )
        else:
            body = "torch.add(x, x)" if kind == "forbidden" else "x"
            self.write(
                self.pr_file, f"import torch\ndef pr_change(x):\n    return {body}\n"
            )
        self.head = self.commit("Actual PR change")
        self.git("checkout", "-q", "main")
        self.git("merge", "--no-ff", "-m", "Synthetic PR merge", self.head)
        self.merge = self.git("rev-parse", "HEAD")

    def _install_capture_commands(self):
        script = f"#!{sys.executable}\n" + textwrap.dedent(
            """\
            import json, os, subprocess, sys
            from pathlib import Path

            args = sys.argv[1:]
            call = {'tool': Path(sys.argv[0]).name, 'args': args}
            if call['tool'] == 'pre-commit':
                if '--all-files' in args:
                    command = ['git', 'ls-files', '-z']
                else:
                    base = args[args.index('--from-ref') + 1]
                    head = args[args.index('--to-ref') + 1]
                    command = ['git', 'diff', '--name-only', '--diff-filter=d', '-z', base, head]
                call['files'] = subprocess.check_output(command).decode().strip('\\0').split('\\0')
            with open(os.environ['CI_RULE_CALLS'], 'a') as handle:
                handle.write(json.dumps(call) + '\\n')
            if call['tool'] != 'pre-commit':
                # Only the formatter is stubbed. Run both real production checkers
                # against the synthetic merge checkout's files and Git history.
                target = Path(os.environ['CI_RULE_CHECKS']) / Path(args[0]).name
                assert target.name in ['check_no_torch_fallback.py', 'derive_changed_operators.py']
                raise SystemExit(subprocess.call([os.environ['CI_RULE_PYTHON'], str(target), *args[1:]]))
            """
        )
        for name in ["pre-commit", "python", "python3"]:
            path = self.bin / name
            path.write_text(script, encoding="utf-8")
            path.chmod(0o755)

    def run_step(self, name, event="pull_request", before="", steps=None, **overrides):
        context = {
            "github.event_name": event,
            "github.event.pull_request.base.sha": self.stale_base,
            "github.event.pull_request.head.sha": self.head,
            "github.event.before": before,
            **{
                f"steps.pr-range.outputs.{key}": self.outputs.get(key, "")
                for key in ["target_base", "merge_base", "head"]
            },
            **overrides,
        }

        def render(value):
            return re.sub(
                r"\$\{\{\s*(.*?)\s*\}\}",
                lambda match: context[match[1]],
                str(value),
            )

        output = Path(self.temporary_directory.name) / "step-output"
        output.write_text("")
        step = (self.steps if steps is None else steps)[name]
        env = {
            **self.env,
            "GITHUB_SHA": overrides.get("github.sha", self.merge),
            "GITHUB_OUTPUT": str(output),
            **{key: render(value) for key, value in step.get("env", {}).items()},
        }
        process = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-e",
                "-o",
                "pipefail",
                "-c",
                render(step["run"]),
            ],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
        )
        outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return process, outputs

    def run_pr_checks(self):
        # The absence path also lets the unchanged workflow demonstrate its
        # actual wrong-file failure in the fixed negative-control run.
        if "Resolve pull request check range" in self.steps:
            result, self.outputs = self.run_step("Resolve pull request check range")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        precommit, _ = self.run_step("Run pre-commit (code style, changed files)")
        self.assertEqual(precommit.returncode, 0, precommit.stdout + precommit.stderr)
        fallback, _ = self.run_step("check no torch compute fallback in changed files")
        derive, operators = self.run_step("Derive changed operators")
        self.assertEqual(derive.returncode, 0, derive.stdout + derive.stderr)
        return fallback, operators

    def run_unit_collect(self):
        if "Resolve pull request check range" in self.unit_steps:
            result, self.outputs = self.run_step(
                "Resolve pull request check range", steps=self.unit_steps
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result, _ = self.run_step("collect ci selection inputs", steps=self.unit_steps)
        return result

    def selection_paths(self):
        data = (self.repo / "changed_files.bin").read_bytes()
        self.assertTrue(data.endswith(b"\0"), data)
        return [path.decode() for path in data[:-1].split(b"\0")]

    def captured(self, tool):
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        matches = [call for call in calls if call["tool"] == tool]
        self.assertEqual(len(matches), 1, matches)
        return matches[0]

    def test_pr_checks_exclude_stale_base_and_merge_only_files(self):
        self.history()
        fallback, operators = self.run_pr_checks()
        self.assertEqual(fallback.returncode, 0, fallback.stdout + fallback.stderr)
        formatter = self.captured("pre-commit")
        self.assertEqual(formatter["files"], [self.pr_file])
        args = formatter["args"]
        self.assertEqual(args[args.index("--from-ref") + 1], self.fork)
        self.assertEqual(args[args.index("--to-ref") + 1], self.head)
        self.assertEqual(self.captured("python")["args"][1:], [self.pr_file])
        derive_args = self.captured("python3")["args"]
        self.assertEqual(derive_args[derive_args.index("--base") + 1], self.target)
        self.assertEqual(derive_args[derive_args.index("--head") + 1], self.head)
        self.assertEqual(json.loads(operators["changed_files"]), [self.pr_file])
        self.assertEqual(json.loads(operators["changed_operators"]), ["pr_change"])

    def test_pr_forbidden_compute_is_still_rejected(self):
        self.history("forbidden")
        fallback, _ = self.run_pr_checks()
        self.assertEqual(fallback.returncode, 1, fallback.stdout + fallback.stderr)
        self.assertIn(
            f"{self.pr_file}:3: forbidden torch compute call torch.add()",
            fallback.stdout,
        )
        self.assertEqual(self.captured("python")["args"][1:], [self.pr_file])

    def test_selected_file_is_checked_in_merged_checkout(self):
        self.history("shared")
        self.assertNotIn("torch.add", self.git("show", f"{self.head}:{self.pr_file}"))
        self.assertIn("torch.add", (self.repo / self.pr_file).read_text())
        fallback, _ = self.run_pr_checks()
        self.assertEqual(fallback.returncode, 1, fallback.stdout + fallback.stderr)
        self.assertIn("forbidden torch compute call torch.add()", fallback.stdout)
        self.assertEqual(self.captured("python")["args"][1:], [self.pr_file])

    def invalid_checkouts(self):
        octopus = self.git(
            "commit-tree",
            "HEAD^{tree}",
            "-p",
            self.target,
            "-p",
            self.head,
            "-p",
            self.stale_base,
            "-m",
            "Unexpected extra merge parent",
        )
        return [
            (self.head, self.head, self.head),
            (self.merge, self.merge, self.target),
            (self.merge, self.head, self.head),
            (octopus, octopus, self.head),
        ]

    def test_unexpected_pr_checkout_is_rejected(self):
        self.history()
        for checkout, event_sha, event_head in self.invalid_checkouts():
            with self.subTest(
                checkout=checkout, event_sha=event_sha, event_head=event_head
            ):
                self.git("checkout", "-q", checkout)
                result, outputs = self.run_step(
                    "Resolve pull request check range",
                    **{
                        "github.sha": event_sha,
                        "github.event.pull_request.head.sha": event_head,
                    },
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(outputs, {})
        self.assertFalse(self.calls.exists())

    def test_push_ranges_and_all_file_precommit_are_unchanged(self):
        self.history()
        result, _ = self.run_step(
            "Run pre-commit (code style, all files)", event="push"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--all-files", self.captured("pre-commit")["args"])
        self.assertIn(
            "src/flaggems_vllm/ops/main_only.py", self.captured("pre-commit")["files"]
        )
        for checkout, before in [
            (self.merge, self.target),
            (self.head, "0000000000000000000000000000000000000000"),
            (self.head, ""),
        ]:
            with self.subTest(checkout=checkout, before=before):
                self.git("checkout", "-q", checkout)
                self.calls.write_text("")
                result, _ = self.run_step(
                    "check no torch compute fallback in changed files",
                    event="push",
                    before=before,
                    **{"github.sha": checkout},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.captured("python")["args"][1:], [self.pr_file])

    def test_unittest_collects_only_pr_paths_with_nul_delimiters(self):
        self.history("path")
        result = self.run_unit_collect()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.selection_paths(), [self.pr_file])
        self.assertEqual(self.outputs["target_base"], self.target)
        self.assertEqual(self.outputs["merge_base"], self.fork)
        self.assertEqual(self.outputs["head"], self.head)

    def test_unittest_rejects_unexpected_pr_checkout(self):
        self.history()
        for checkout, event_sha, event_head in self.invalid_checkouts():
            with self.subTest(
                checkout=checkout, event_sha=event_sha, event_head=event_head
            ):
                self.git("checkout", "-q", checkout)
                result, outputs = self.run_step(
                    "Resolve pull request check range",
                    steps=self.unit_steps,
                    **{
                        "github.sha": event_sha,
                        "github.event.pull_request.head.sha": event_head,
                    },
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(outputs, {})
        self.assertFalse((self.repo / "changed_files.bin").exists())

    def test_unittest_push_collection_keeps_previous_ranges(self):
        self.history()
        for checkout, before in [
            (self.merge, self.target),
            (self.head, "0000000000000000000000000000000000000000"),
            (self.head, ""),
        ]:
            with self.subTest(checkout=checkout, before=before):
                self.git("checkout", "-q", checkout)
                result, _ = self.run_step(
                    "collect ci selection inputs",
                    event="push",
                    before=before,
                    steps=self.unit_steps,
                    **{"github.sha": checkout},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.selection_paths(), [self.pr_file])


if __name__ == "__main__":
    unittest.main()
