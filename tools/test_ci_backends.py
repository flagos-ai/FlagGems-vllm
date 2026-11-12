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
            select_ci_backends.load_platforms(path), set()
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
            select_ci_backends.load_platforms(path), set()
        )[0]

        self.assertEqual(
            entry["container_image"],
            "example.invalid/some/rather/long/path/vllm:1.2.3-tag",
        )
        self.assertEqual(entry["container_options"], "--ipc=host --privileged")


class RepositoryCiBackendsConfigTest(unittest.TestCase):
    """Guard the checked-in config, not just the loader."""

    def setUp(self):
        self.platforms = select_ci_backends.load_platforms(CONFIG_PATH)
        self.selected = select_ci_backends.select_platforms(self.platforms, set())

    def test_config_loads_and_selects_the_enabled_platforms(self):
        self.assertEqual(
            [entry["platform"] for entry in self.selected],
            ["hygon", "ascend", "iluvatar", "metax", "mthreads"],
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


if __name__ == "__main__":
    unittest.main()
