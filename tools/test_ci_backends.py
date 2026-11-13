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

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import select_ci_backends

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".github/configs/ci-backends.yaml"
CAPABILITIES_PATH = REPO_ROOT / ".github/backend-capabilities.json"


class SelectCiBackendsTest(unittest.TestCase):
    def write_config(self, body: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "ci-backends.yaml"
        path.write_text(textwrap.dedent(body), encoding="utf-8")
        return path

    def test_disabled_platform_is_skipped(self):
        path = self.write_config(
            """
            platforms:
              ready:
                backend: ready-backend
                runner_labels: ready-1
                container_image: example.invalid/ready:1
                gpu_check_script: tools/gpu_check_nvidia.sh
              pending:
                enabled: false
                backend: pending-backend
                runner_labels: pending-1
                container_image: example.invalid/pending:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

        selected = select_ci_backends.select_platforms(
            select_ci_backends.load_platforms(path), set(), all_enabled=True
        )

        self.assertEqual([entry["platform"] for entry in selected], ["ready"])

    def test_explicit_request_overrides_disabled(self):
        path = self.write_config(
            """
            platforms:
              pending:
                enabled: false
                backend: pending-backend
                runner_labels: pending-1
                container_image: example.invalid/pending:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

        selected = select_ci_backends.select_platforms(
            select_ci_backends.load_platforms(path), {"pending"}
        )

        self.assertEqual([entry["platform"] for entry in selected], ["pending"])

    def test_unknown_requested_platform_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              ready:
                backend: ready-backend
                runner_labels: ready-1
                container_image: example.invalid/ready:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.select_platforms(
                select_ci_backends.load_platforms(path), {"typo"}
            )

    def test_non_string_test_env_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              broken:
                backend: broken-backend
                runner_labels: broken-1
                container_image: example.invalid/broken:1
                gpu_check_script: tools/gpu_check_nvidia.sh
                test_env:
                  SOME_FLAG: 1
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.load_platforms(path)

    def labelled_config(self) -> Path:
        return self.write_config(
            """
            platforms:
              hygon:
                label: vendor/Hygon
                backend: hygon
                runner_labels: hygon-1
                container_image: example.invalid/hygon:1
                gpu_check_script: tools/gpu_check_nvidia.sh
              metax:
                label: vendor/MetaX
                backend: metax-maca3810
                runner_labels: metax-1
                container_image: example.invalid/metax:1
                gpu_check_script: tools/gpu_check_nvidia.sh
              nvidia:
                label: vendor/NVIDIA
                backend: nvidia-cuda133
                runner_labels: a100-1
                container_image: example.invalid/nvidia:1
                gpu_check_script: tools/gpu_check_nvidia.sh
              pending:
                enabled: false
                label: vendor/Pending
                backend: pending-backend
                runner_labels: pending-1
                container_image: example.invalid/pending:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

    def platforms_for(self, labels=(), all_enabled=False):
        platforms = select_ci_backends.load_platforms(self.labelled_config())
        selected = select_ci_backends.select_platforms(
            platforms, set(), set(labels), all_enabled
        )
        return [entry["platform"] for entry in selected]

    def test_nothing_is_selected_without_a_request(self):
        self.assertEqual(self.platforms_for(), [])
        self.assertEqual(self.platforms_for(["documentation", "size/Small"]), [])

    def test_a_pr_label_selects_only_its_platform(self):
        self.assertEqual(self.platforms_for(["vendor/Hygon"]), ["hygon"])
        self.assertEqual(
            self.platforms_for(["vendor/MetaX", "vendor/Hygon"]), ["hygon", "metax"]
        )
        self.assertEqual(self.platforms_for(["vendor/NVIDIA"]), ["nvidia"])

    def test_all_enabled_excludes_nvidia_and_disabled_platforms(self):
        self.assertEqual(self.platforms_for(all_enabled=True), ["hygon", "metax"])

    def test_a_change_under_the_backend_directory_selects_the_platform(self):
        platforms = select_ci_backends.load_platforms(self.labelled_config())
        base = "src/flaggems_vllm/runtime/backend"

        def run(*paths):
            selected = select_ci_backends.select_platforms(
                platforms, set(), set(), False, set(paths)
            )
            return [entry["platform"] for entry in selected]

        self.assertEqual(run(f"{base}/_hygon/ops/x.py"), ["hygon"])
        self.assertEqual(
            run(f"{base}/_metax/ops/x.py", f"{base}/_hygon/y"), ["hygon", "metax"]
        )
        self.assertEqual(run("src/flaggems_vllm/ops/add.py", "README.md"), [])
        # NVIDIA is label-only, and a disabled platform never runs.
        self.assertEqual(run(f"{base}/_nvidia/ops/x.py"), [])
        self.assertEqual(run(f"{base}/_pending/ops/x.py"), [])

    def test_path_and_label_requests_are_combined(self):
        platforms = select_ci_backends.load_platforms(self.labelled_config())
        selected = select_ci_backends.select_platforms(
            platforms,
            set(),
            {"vendor/NVIDIA"},
            False,
            {"src/flaggems_vllm/runtime/backend/_hygon/ops/x.py"},
        )
        self.assertEqual([e["platform"] for e in selected], ["hygon", "nvidia"])

    def test_default_platforms_run_only_when_defaults_are_included(self):
        path = self.write_config(
            """
            platforms:
              a100:
                backend: nvidia-cuda133
                runner_labels: a100-1
                container_image: example.invalid/a100:1
                run_by_default: true
                gpu_check_script: tools/gpu_check_nvidia.sh
              hygon:
                backend: hygon
                runner_labels: hygon-1
                container_image: example.invalid/hygon:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )
        platforms = select_ci_backends.load_platforms(path)

        def run(include_defaults, **kwargs):
            selected = select_ci_backends.select_platforms(
                platforms, set(), include_defaults=include_defaults, **kwargs
            )
            return [entry["platform"] for entry in selected]

        self.assertEqual(run(False), [])
        self.assertEqual(run(True), ["a100"])
        # "all" still never picks NVIDIA; defaults add it back.
        self.assertEqual(run(False, all_enabled=True), ["hygon"])
        self.assertEqual(run(True, all_enabled=True), ["a100", "hygon"])

    def test_non_boolean_run_by_default_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              broken:
                backend: broken-backend
                runner_labels: broken-1
                container_image: example.invalid/broken:1
                gpu_check_script: tools/gpu_check_nvidia.sh
                run_by_default: "yes"
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.load_platforms(path)

    def test_a_label_cannot_enable_a_disabled_platform(self):
        self.assertEqual(self.platforms_for(["vendor/Pending"]), [])

    def test_non_string_label_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              broken:
                label: 1
                backend: broken-backend
                runner_labels: broken-1
                container_image: example.invalid/broken:1
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.load_platforms(path)

    def test_label_json_requires_string_array(self):
        self.assertEqual(select_ci_backends.parse_labels("null"), set())
        with self.assertRaises(ValueError):
            select_ci_backends.parse_labels('{"a": 1}')

    def test_missing_required_field_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              broken:
                backend: broken-backend
                runner_labels: broken-1
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.load_platforms(path)

    def test_folded_image_is_collapsed_to_one_line(self):
        # A YAML folded scalar (`>-`) keeps the value on one logical line but
        # inserts a space at each fold, which would corrupt an image ref.
        path = self.write_config(
            """
            platforms:
              folded:
                backend: folded-backend
                runner_labels: folded-1
                container_image: >-
                  example.invalid/some/rather/long/path/vllm:1.2.3-tag
                container_options: >-
                  --ipc=host
                  --privileged
                gpu_check_script: tools/gpu_check_nvidia.sh
            """
        )

        entry = select_ci_backends.select_platforms(
            select_ci_backends.load_platforms(path), set(), all_enabled=True
        )[0]

        self.assertEqual(
            entry["container_image"],
            "example.invalid/some/rather/long/path/vllm:1.2.3-tag",
        )
        self.assertEqual(entry["container_options"], "--ipc=host --privileged")

    def test_non_boolean_install_with_setup_sh_is_rejected(self):
        path = self.write_config(
            """
            platforms:
              broken:
                backend: broken-backend
                runner_labels: broken-1
                container_image: example.invalid/broken:1
                gpu_check_script: tools/gpu_check_nvidia.sh
                install_with_setup_sh: "yes"
            """
        )

        with self.assertRaises(ValueError):
            select_ci_backends.load_platforms(path)


class RepositoryCiBackendsConfigTest(unittest.TestCase):
    """Guard the checked-in config, not just the loader."""

    def setUp(self):
        self.platforms = select_ci_backends.load_platforms(CONFIG_PATH)
        # Name every platform so the checks below cover the whole config.
        self.selected = select_ci_backends.select_platforms(
            self.platforms, set(self.platforms)
        )

    def test_config_loads_and_selects_the_enabled_platforms(self):
        self.assertEqual(
            [entry["platform"] for entry in self.selected],
            ["hygon", "ascend", "iluvatar", "metax", "mthreads", "nvidia"],
        )

    def test_test_env_round_trips_as_json(self):
        # The workflow consumes this with fromJSON(matrix.test_env), which
        # fails the whole job if the value is not valid JSON.
        by_platform = {entry["platform"]: entry for entry in self.selected}
        for entry in self.selected:
            with self.subTest(platform=entry["platform"]):
                self.assertIsInstance(json.loads(entry["test_env"]), dict)
        self.assertEqual(
            json.loads(by_platform["mthreads"]["test_env"]),
            {"FLAGTREE_BACKEND": "mthreads"},
        )
        self.assertEqual(json.loads(by_platform["hygon"]["test_env"]), {})

    def test_every_platform_has_a_vendor_label(self):
        # Without a label a platform could only run through `ci/all-vendors`.
        for name, entry in self.platforms.items():
            with self.subTest(platform=name):
                self.assertTrue(entry.get("label", "").startswith("vendor/"))

    def test_each_platform_maps_to_an_existing_backend_directory(self):
        # Path routing keys on the vendor prefix of `backend`; a typo would
        # make the platform unreachable by path.
        for name, entry in self.platforms.items():
            if entry["backend"].startswith("nvidia"):
                continue
            vendor = entry["backend"].split("-", maxsplit=1)[0]
            directory = REPO_ROOT / "src/flaggems_vllm/runtime/backend" / f"_{vendor}"
            with self.subTest(platform=name):
                self.assertTrue(directory.is_dir(), directory)

    def test_only_the_a100_platform_runs_by_default(self):
        defaults = [
            name
            for name, entry in self.platforms.items()
            if entry.get("run_by_default")
        ]
        self.assertEqual(defaults, ["nvidia"])

    def test_container_images_have_no_embedded_whitespace(self):
        for entry in self.selected:
            with self.subTest(platform=entry["platform"]):
                self.assertNotIn(" ", entry["container_image"])

    def test_no_platform_declares_container_volumes(self):
        # The ARC `k8s-novolume` hook backing these runners cannot bind-mount
        # host paths; it mkdirs each mount point and fails when the target
        # exists as a file. Run 42145278734 failed on exactly the platforms
        # that declared a volume.
        for name, entry in self.platforms.items():
            with self.subTest(platform=name):
                self.assertEqual(entry.get("container_volumes") or [], [])

    def test_gpu_check_scripts_exist(self):
        for name, entry in self.platforms.items():
            with self.subTest(platform=name):
                self.assertTrue((REPO_ROOT / entry["gpu_check_script"]).is_file())

    def test_backends_are_known_to_the_capability_policy(self):
        # run_ci_targets.py falls back to the fail-closed defaults for an
        # unknown backend, which would silently drop every test. Keep the two
        # files in sync instead.
        capabilities = json.loads(CAPABILITIES_PATH.read_text(encoding="utf-8"))
        for name, entry in self.platforms.items():
            with self.subTest(platform=name):
                self.assertIn(entry["backend"], capabilities["backends"])

    def test_nvidia_platform_installs_through_setup_sh(self):
        by_platform = {entry["platform"]: entry for entry in self.selected}
        nvidia = by_platform["nvidia"]
        self.assertEqual(nvidia["runner_labels"], "nvidia-a100-1")
        self.assertEqual(nvidia["container_image"], "nvcr.io/nvidia/vllm:26.10-py3")
        self.assertTrue(nvidia["install_with_setup_sh"])
        # Vendor images already ship flag-gems; only the NGC image needs setup.sh.
        for name, entry in by_platform.items():
            if name != "nvidia":
                with self.subTest(platform=name):
                    self.assertFalse(entry["install_with_setup_sh"])


if __name__ == "__main__":
    unittest.main()
