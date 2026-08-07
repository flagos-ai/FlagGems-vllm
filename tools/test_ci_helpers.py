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

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import check_ci_pins
import run_ci_targets
import select_backends
import select_tests


class TemporaryRepositoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary_directory.name)
        (self.repo_root / "tests/deep/nested").mkdir(parents=True)
        (self.repo_root / "benchmark/deep/nested").mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_file(self, relative_path: str) -> None:
        path = self.repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class SelectTestsTest(TemporaryRepositoryTestCase):
    def test_environment_and_operator_changes_are_combined(self):
        self.make_file("tests/test_mul.py")

        mode, tests, benchmarks = select_tests.select_targets(
            self.repo_root,
            ["pyproject.toml", "src/flaggems_vllm/ops/mul.py"],
        )

        self.assertEqual(mode, "smoke")
        self.assertEqual(tests, ["tests/test_mul.py"])
        self.assertEqual(benchmarks, [])

    def test_flaggems_environment_helper_triggers_preflight(self):
        self.assertEqual(
            select_tests.select_targets(
                self.repo_root, ["tools/prepare-flaggems-ci-env.sh"]
            ),
            ("smoke", [], []),
        )

    def test_unknown_operator_change_selects_all_tests(self):
        self.make_file("tests/test_first.py")
        self.make_file("tests/deep/nested/test_second.py")

        mode, tests, _ = select_tests.select_targets(
            self.repo_root,
            ["src/flaggems_vllm/ops/unknown_operator.py"],
        )

        self.assertEqual(mode, "selected")
        self.assertEqual(
            tests,
            ["tests/deep/nested/test_second.py", "tests/test_first.py"],
        )

    def test_backend_operator_change_selects_matching_test(self):
        self.make_file("tests/test_mul.py")

        mode, tests, benchmarks = select_tests.select_targets(
            self.repo_root,
            ["src/flaggems_vllm/runtime/backend/_iluvatar/ops/mul.py"],
        )

        self.assertEqual(mode, "smoke")
        self.assertEqual(tests, ["tests/test_mul.py"])
        self.assertEqual(benchmarks, [])

    def test_documentation_change_is_skipped(self):
        self.assertEqual(
            select_tests.select_targets(self.repo_root, ["docs/guide.md"]),
            ("skip", [], []),
        )

    def test_nested_target_with_spaces_is_preserved(self):
        target = "tests/deep/nested/test_space name.py"
        self.make_file(target)

        mode, tests, _ = select_tests.select_targets(self.repo_root, [target])

        self.assertEqual(mode, "selected")
        self.assertEqual(tests, [target])

    def test_reads_null_delimited_unicode_paths(self):
        changed_files = self.repo_root / "changed.bin"
        changed_files.write_bytes("tests/test 空格.py\0docs/a.md\0".encode())
        self.assertEqual(
            select_tests.read_changed_files(str(changed_files)),
            ["tests/test 空格.py", "docs/a.md"],
        )

    def test_missing_changed_files_input_fails(self):
        with self.assertRaises(FileNotFoundError):
            select_tests.read_changed_files(str(self.repo_root / "missing"))


class SelectBackendsTest(unittest.TestCase):
    registry = [
        {
            "backend": "ascend-cann850",
            "runner_label": "ascend",
            "label": "vendor/Ascend",
            "gpu_check": "tools/gpu_check_ascend.sh",
            "enabled": True,
        },
        {
            "backend": "kunlunxin",
            "runner_label": "kunlunxin",
            "label": "vendor/Kunlunxin",
            "gpu_check": "tools/gpu_check_kunlunxin.sh",
            "enabled": True,
        },
        {
            "backend": "nvidia-cuda133",
            "runner_label": "h20",
            "label": "vendor/NVIDIA",
            "gpu_check": "tools/gpu_check_nvidia.sh",
            "enabled": True,
        },
    ]

    def test_pr_label_selects_only_matching_backend(self):
        selected = select_backends.select_backends(
            self.registry, {"vendor/Ascend"}, all_enabled=False
        )
        self.assertEqual([entry["backend"] for entry in selected], ["ascend-cann850"])

    def test_all_enabled_still_excludes_nvidia(self):
        selected = select_backends.select_backends(
            self.registry, set(), all_enabled=True
        )
        self.assertEqual(
            [entry["backend"] for entry in selected],
            ["ascend-cann850", "kunlunxin"],
        )

    def test_backend_source_change_selects_vendor_without_a_label(self):
        selected = select_backends.select_backends(
            self.registry,
            set(),
            all_enabled=False,
            changed_files={"src/flaggems_vllm/runtime/backend/_kunlunxin/ops/mul.py"},
        )
        self.assertEqual(
            [entry["backend"] for entry in selected],
            ["kunlunxin"],
        )

    def test_unrelated_source_change_does_not_select_a_vendor(self):
        selected = select_backends.select_backends(
            self.registry,
            set(),
            all_enabled=False,
            changed_files={"src/flaggems_vllm/ops/mul.py"},
        )
        self.assertEqual(selected, [])

    def test_label_json_requires_string_array(self):
        with self.assertRaises(ValueError):
            select_backends.parse_labels('{"vendor": "Ascend"}')


class RunCiTargetsTest(TemporaryRepositoryTestCase):
    def test_json_round_trip_and_safe_argv(self):
        target = "tests/deep/nested/test_$(touch pwned) 空格.py"
        self.make_file(target)
        raw = json.dumps(
            {
                "schema_version": 1,
                "mode": "selected",
                "tests": [target],
                "benchmarks": [],
            }
        )

        targets = run_ci_targets.load_targets(raw)
        validated = run_ci_targets.validate_target(self.repo_root, target, "tests")
        command = run_ci_targets.build_commands(targets)[0]

        self.assertEqual(validated, target)
        self.assertEqual(command[-1], target)
        self.assertNotIn("touch", command)

    def test_policy_is_fail_closed_by_default(self):
        targets = {"tests": ["tests/test_op.py"], "benchmarks": []}
        policy = {
            "allow_all_tests": False,
            "tests_allow": [],
            "benchmarks_enabled": False,
            "allow_all_benchmarks": False,
            "benchmarks_allow": [],
        }
        self.assertEqual(
            run_ci_targets.apply_policy(targets, policy),
            {"tests": [], "benchmarks": []},
        )

    def test_iluvatar_policy_approves_only_mul_correctness(self):
        repo_root = Path(__file__).resolve().parents[1]
        policy = run_ci_targets.load_policy(
            repo_root / ".github/backend-capabilities.json", "iluvatar"
        )
        targets = {
            "tests": ["tests/test_mul.py", "tests/test_flash_mla.py"],
            "benchmarks": ["benchmark/test_mul.py"],
        }

        self.assertEqual(
            run_ci_targets.apply_policy(targets, policy),
            {"tests": ["tests/test_mul.py"], "benchmarks": []},
        )

    def test_benchmark_command_uses_bounded_iterations(self):
        commands = run_ci_targets.build_commands(
            {"tests": [], "benchmarks": ["benchmark/test_op.py"]}
        )
        self.assertIn("--warmup", commands[0])
        self.assertEqual(commands[0][commands[0].index("--warmup") + 1], "1")
        self.assertIn("--iter", commands[0])
        self.assertEqual(commands[0][commands[0].index("--iter") + 1], "1")

    def test_rejects_unsafe_or_malformed_targets(self):
        with self.assertRaises(ValueError):
            run_ci_targets.load_targets(
                '{"schema_version":1,"tests":"tests/test_op.py","benchmarks":[]}'
            )
        with self.assertRaises(ValueError):
            run_ci_targets.validate_target(
                self.repo_root, "tests/../secrets.py", "tests"
            )


class IluvatarForkSelectionIntegrationTest(TemporaryRepositoryTestCase):
    def test_pr50_like_change_runs_only_mul_correctness(self):
        self.make_file("tests/test_mul.py")
        self.make_file("tests/test_flash_mla.py")
        self.make_file("benchmark/test_mul.py")
        changed_files = [
            "benchmark/core_shapes.yaml",
            "benchmark/test_mul.py",
            "conf/operators.yaml",
            "src/flaggems_vllm/runtime/backend/_iluvatar/ops/__init__.py",
            "src/flaggems_vllm/runtime/backend/_iluvatar/ops/mul.py",
            "tests/test_mul.py",
        ]
        registry = [
            {
                "backend": "iluvatar",
                "runner_label": "iluvatar",
                "label": "vendor/Iluvatar",
                "gpu_check": "tools/gpu_check_iluvatar.sh",
                "enabled": True,
            }
        ]

        selected_backends = select_backends.select_backends(
            registry,
            set(),
            all_enabled=False,
            changed_files=set(changed_files),
        )
        _, selected_tests, _ = select_tests.select_targets(
            self.repo_root, changed_files
        )
        policy = run_ci_targets.load_policy(
            Path(__file__).resolve().parents[1] / ".github/backend-capabilities.json",
            "iluvatar",
        )
        approved = run_ci_targets.apply_policy(
            # Normal pull requests pass --no-benchmarks in basic-ci.yml.
            {"tests": selected_tests, "benchmarks": []},
            policy,
        )

        self.assertEqual(
            [entry["backend"] for entry in selected_backends], ["iluvatar"]
        )
        self.assertEqual(
            approved,
            {"tests": ["tests/test_mul.py"], "benchmarks": []},
        )


class CiPinsTest(unittest.TestCase):
    def test_three_flaggems_pins_are_identical(self):
        repo_root = Path(__file__).resolve().parents[1]
        pins = check_ci_pins.extract_pins(repo_root)
        self.assertEqual(len(pins), 3)
        self.assertEqual(len(set(pins)), 1)


class BackendTuneConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "src/flaggems_vllm/runtime/backend/backend_utils.py"
        cls.module_name = "_flaggems_backend_utils_ci_test"
        spec = importlib.util.spec_from_file_location(cls.module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load backend utils from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[cls.module_name] = module
        spec.loader.exec_module(module)
        cls.backend_utils = module

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.module_name, None)

    def write_config(self, content: str) -> str:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "tune_configs.yaml"
        path.write_text(content, encoding="utf-8")
        return temporary_directory.name

    def test_comment_only_tune_config_is_an_empty_mapping(self):
        directory = self.write_config("# This backend has no tuned kernels yet.\n")
        self.assertEqual(
            self.backend_utils.get_tune_config(file_path=directory),
            {},
        )

    def test_iluvatar_comment_only_tune_config_loads(self):
        self.assertEqual(self.backend_utils.get_tune_config("iluvatar"), {})

    def test_non_empty_tune_config_is_preserved(self):
        directory = self.write_config("mul:\n  BLOCK_SIZE: 128\n")
        self.assertEqual(
            self.backend_utils.get_tune_config(file_path=directory),
            {"mul": {"BLOCK_SIZE": 128}},
        )

    def test_malformed_tune_config_is_rejected(self):
        directory = self.write_config("mul: [\n")
        with self.assertRaises(ValueError):
            self.backend_utils.get_tune_config(file_path=directory)

    def test_non_mapping_tune_config_is_rejected(self):
        directory = self.write_config("- mul\n")
        with self.assertRaises(ValueError):
            self.backend_utils.get_tune_config(file_path=directory)


class PrepareFlagGemsCiEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.helper = cls.repo_root / "tools/prepare-flaggems-ci-env.sh"

    def run_helper(
        self,
        backend: str,
        *,
        inherited_home_is_valid: bool = True,
        passwd_home_is_valid: bool = True,
        uv_overrides_are_invalid: bool = False,
    ):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        workspace = root / "FlagGems-vllm"
        runner_temp = root / "runner-temp"
        github_env = root / "github-env"
        workspace.mkdir()
        runner_temp.mkdir()
        inherited_home = root / "runner-home"
        if inherited_home_is_valid:
            inherited_home.mkdir()
        else:
            inherited_home.write_text("not a directory", encoding="utf-8")

        passwd_home = root / "passwd-home"
        if passwd_home_is_valid:
            passwd_home.mkdir()

        fake_bin = root / "fake-bin"
        fake_bin.mkdir()
        fake_getent = fake_bin / "getent"
        fake_getent.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' 'runner:x:1000:1000::{passwd_home}:/bin/bash'\n",
            encoding="utf-8",
        )
        fake_getent.chmod(0o755)

        environment = {
            **os.environ,
            "HOME": str(inherited_home),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_ENV": str(github_env),
            "GITHUB_RUN_ID": "12345",
            "GITHUB_RUN_ATTEMPT": "2",
        }
        for name in (
            "XDG_CACHE_HOME",
            "XDG_DATA_HOME",
            "UV_CACHE_DIR",
            "UV_PYTHON_INSTALL_DIR",
        ):
            environment.pop(name, None)
        if uv_overrides_are_invalid:
            invalid_cache = root / "invalid-uv-cache"
            invalid_python = root / "invalid-uv-python"
            invalid_cache.write_text("not a directory", encoding="utf-8")
            invalid_python.write_text("not a directory", encoding="utf-8")
            environment["UV_CACHE_DIR"] = str(invalid_cache)
            environment["UV_PYTHON_INSTALL_DIR"] = str(invalid_python)

        result = subprocess.run(
            ["bash", str(self.helper), backend],
            cwd=self.repo_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        values = {}
        if github_env.exists():
            for line in github_env.read_text(encoding="utf-8").splitlines():
                key, value = line.split("=", maxsplit=1)
                values[key] = value
        return (
            result,
            values,
            workspace.resolve(),
            runner_temp.resolve(),
            inherited_home.resolve(),
            passwd_home.resolve(),
        )

    def test_preserves_a_writable_runner_home_and_uv_defaults(self):
        result, values, workspace, _, inherited_home, _ = self.run_helper("iluvatar")
        self.assertEqual(result.returncode, 0, result.stderr)

        expected = {
            "HOME": inherited_home,
            "FLAGGEMS_DIR": workspace / ".ci/flaggems",
            "FLAGGEMS_VENV": workspace / ".ci/flaggems/.venv",
        }
        self.assertEqual(set(values), set(expected))
        for name, path in expected.items():
            self.assertEqual(Path(values[name]), path)
        self.assertTrue((inherited_home / ".local/bin").is_dir())
        self.assertTrue((inherited_home / ".cache/uv").is_dir())
        self.assertTrue((inherited_home / ".local/share/uv/python").is_dir())
        self.assertEqual(list(inherited_home.rglob(".flaggems-ci-write.*")), [])
        self.assertNotIn("UV_CACHE_DIR", values)
        self.assertNotIn("UV_PYTHON_INSTALL_DIR", values)

    def test_replaces_only_unusable_uv_overrides(self):
        result, values, _, _, inherited_home, _ = self.run_helper(
            "iluvatar", uv_overrides_are_invalid=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(values["UV_CACHE_DIR"]), inherited_home / ".cache/uv")
        self.assertEqual(
            Path(values["UV_PYTHON_INSTALL_DIR"]),
            inherited_home / ".local/share/uv/python",
        )

    def test_uses_passwd_home_when_inherited_home_is_invalid(self):
        result, values, _, _, _, passwd_home = self.run_helper(
            "iluvatar", inherited_home_is_valid=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(values["HOME"]), passwd_home)
        self.assertIn("Using passwd FlagGems CI HOME", result.stdout)
        self.assertNotIn("UV_CACHE_DIR", values)
        self.assertNotIn("UV_PYTHON_INSTALL_DIR", values)

    def test_uses_job_local_home_only_as_last_resort(self):
        result, values, _, runner_temp, _, _ = self.run_helper(
            "iluvatar",
            inherited_home_is_valid=False,
            passwd_home_is_valid=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected_home = runner_temp / "flaggems-vllm-12345-2-iluvatar" / "home"
        self.assertEqual(Path(values["HOME"]), expected_home)
        self.assertIn("Using job-local fallback FlagGems CI HOME", result.stdout)

    def test_rejects_an_unsafe_backend_path(self):
        result, values, _, _, _, _ = self.run_helper("../../iluvatar")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid FlagGems backend profile", result.stderr)
        self.assertEqual(values, {})


class SetupFlagGemsActionContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.action = (
            repo_root / ".github/actions/setup-flaggems/action.yml"
        ).read_text(encoding="utf-8")
        cls.test_script = (repo_root / "tools/run-multi-backend-tests.sh").read_text(
            encoding="utf-8"
        )

    def test_uses_explicit_caller_checkout_and_vendor_venv_paths(self):
        self.assertIn("tools/prepare-flaggems-ci-env.sh", self.action)
        self.assertIn('cd "${FLAGGEMS_DIR}"', self.action)
        self.assertIn('source "${FLAGGEMS_VENV}/bin/activate"', self.action)
        self.assertIn('-e "${GITHUB_WORKSPACE}"', self.action)
        self.assertIn('script="${FLAGGEMS_DIR}/${GPU_CHECK_SCRIPT}"', self.action)
        self.assertIn('ln -s "${FLAGGEMS_VENV}" .venv', self.action)
        self.assertNotIn("cd .ci/flaggems", self.action)

    def test_backend_test_prefers_the_physical_vendor_venv(self):
        self.assertIn('VENV_PATH="${FLAGGEMS_VENV:-.venv}"', self.test_script)
        self.assertIn('source "${VENV_PATH}/bin/activate"', self.test_script)

    def test_uv_diagnostic_uses_the_selected_backend_python(self):
        self.assertIn('uv python find "${backend_python}"', self.action)
        self.assertIn("--managed-python --no-python-downloads", self.action)
        self.assertNotIn("uv python find 3.10", self.action)

    def test_each_composite_phase_emits_a_specific_failure_annotation(self):
        for title in (
            "Persistent runner cleanup failed",
            "Runner environment preparation failed",
            "FlagGems vendor setup failed",
            "FlagGems-vllm installation failed",
            "Accelerator availability check failed",
            "Portable backend preflight failed",
        ):
            with self.subTest(title=title):
                self.assertIn(f"::error title={title}::", self.action)


class CiWorkflowPolicyTest(unittest.TestCase):
    def test_all_backend_fanout_requires_an_explicit_request(self):
        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github/workflows/basic-ci.yml").read_text(
            encoding="utf-8"
        )
        all_enabled = workflow.split("ALL_ENABLED: >-", maxsplit=1)[1].split(
            "run: |", maxsplit=1
        )[0]

        self.assertNotIn("github.event_name == 'push'", all_enabled)
        self.assertIn("inputs.run_non_nvidia == true", all_enabled)
        self.assertIn("'ci/all-vendors'", all_enabled)

    def test_forks_are_allowed_only_on_the_non_nvidia_lane(self):
        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github/workflows/basic-ci.yml").read_text(
            encoding="utf-8"
        )
        author_guard = "github.event.pull_request.user.login != 'dependabot[bot]'"
        repository_guard = (
            "github.event.pull_request.head.repo.full_name == github.repository"
        )
        nvidia_job = workflow.split("  nvidia-tests:", maxsplit=1)[1].split(
            "  non-nvidia-tests:", maxsplit=1
        )[0]
        non_nvidia_job = workflow.split("  non-nvidia-tests:", maxsplit=1)[1].split(
            "  multi-backend-summary:", maxsplit=1
        )[0]

        self.assertIn(repository_guard, nvidia_job)
        self.assertNotIn(repository_guard, non_nvidia_job)
        self.assertIn(author_guard, nvidia_job)
        self.assertIn(author_guard, non_nvidia_job)
        self.assertNotIn("github.actor != 'dependabot[bot]'", workflow)


class CiSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        workflow = (repo_root / ".github/workflows/basic-ci.yml").read_text(
            encoding="utf-8"
        )
        summary = workflow.split("  multi-backend-summary:", maxsplit=1)[1]
        cls.script = textwrap.dedent(
            summary.split("        run: |\n", maxsplit=1)[1]
        ).strip()

    def run_summary(self, **overrides):
        values = {
            "CODE_STYLE_RESULT": "success",
            "SELECT_TARGETS_RESULT": "success",
            "NVIDIA_RESULT": "skipped",
            "NON_NVIDIA_RESULT": "skipped",
            "SHOULD_RUN": "false",
            "HAS_NON_NVIDIA_BACKENDS": "false",
            "EVENT_NAME": "push",
            "NVIDIA_RUN_ALLOWED": "true",
            "NON_NVIDIA_RUN_ALLOWED": "true",
        }
        values.update(overrides)
        return subprocess.run(
            ["bash", "-c", self.script],
            env={**os.environ, **values},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_expected_accelerator_results_succeed(self):
        cases = {
            "trusted_nvidia_only": {
                "SHOULD_RUN": "true",
                "NVIDIA_RESULT": "success",
            },
            "trusted_full_matrix": {
                "SHOULD_RUN": "true",
                "HAS_NON_NVIDIA_BACKENDS": "true",
                "NVIDIA_RESULT": "success",
                "NON_NVIDIA_RESULT": "success",
            },
            "pr_backend_preflight_only": {
                "HAS_NON_NVIDIA_BACKENDS": "true",
                "EVENT_NAME": "pull_request",
                "NON_NVIDIA_RESULT": "success",
            },
            "untrusted_fork": {
                "SHOULD_RUN": "true",
                "HAS_NON_NVIDIA_BACKENDS": "true",
                "EVENT_NAME": "pull_request",
                "NVIDIA_RUN_ALLOWED": "false",
                "NON_NVIDIA_RESULT": "success",
            },
            "dependabot_fork": {
                "SHOULD_RUN": "true",
                "HAS_NON_NVIDIA_BACKENDS": "true",
                "EVENT_NAME": "pull_request",
                "NVIDIA_RUN_ALLOWED": "false",
                "NON_NVIDIA_RUN_ALLOWED": "false",
            },
            "no_targets": {},
        }

        for name, values in cases.items():
            with self.subTest(name=name):
                result = self.run_summary(**values)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_unexpected_results_fail(self):
        cases = {
            "preparation_failed": {"CODE_STYLE_RESULT": "failure"},
            "expected_nvidia_skipped": {
                "SHOULD_RUN": "true",
                "NVIDIA_RESULT": "skipped",
            },
            "expected_backend_failed": {
                "SHOULD_RUN": "true",
                "HAS_NON_NVIDIA_BACKENDS": "true",
                "NVIDIA_RESULT": "success",
                "NON_NVIDIA_RESULT": "failure",
            },
            "unexpected_backend_run": {"NON_NVIDIA_RESULT": "success"},
        }

        for name, values in cases.items():
            with self.subTest(name=name):
                result = self.run_summary(**values)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
