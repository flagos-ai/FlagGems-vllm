# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise setup's smoke flow with real capability probes and isolated imports."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SetupSmokeTest(unittest.TestCase):
    def run_smoke(self, *, tle=False, broken_compiler=False, collect_status=0):
        with tempfile.TemporaryDirectory(prefix="setup smoke ") as directory:
            root = Path(directory)
            package = root / "flaggems_vllm"
            (package / "utils").mkdir(parents=True)
            (root / "tests").mkdir()
            (package / "__init__.py").write_text(
                "import triton\n__version__='test'\nvendor_name='metax'\ndevice='cuda'\n"
            )
            (package / "utils/__init__.py").touch()
            (package / "utils/triton_version_utils.py").write_text(
                (ROOT / "src/flaggems_vllm/utils/triton_version_utils.py").read_text()
            )
            triton = root / "triton"
            triton.mkdir()
            (triton / "__init__.py").write_text(
                "raise ImportError('broken compiler')\n"
                if broken_compiler
                else f"__version__={'3.6.0' if tle else '3.1.0'!r}\n"
            )
            if tle:
                (triton / "experimental/tle").mkdir(parents=True)
                for name in (
                    "experimental/__init__.py",
                    "experimental/tle/__init__.py",
                    "experimental/tle/language.py",
                ):
                    (triton / name).touch()
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\n"
                "__version__='test'\nversion=SimpleNamespace(cuda=None)\n"
            )
            for name in ("flag_gems.py", "vllm.py"):
                (root / name).write_text("__version__='test'\n")
            binary = root / "bin"
            binary.mkdir()
            wrapper = binary / "python"
            wrapper.write_text(
                "#!/usr/bin/env bash\n"
                'if [[ "${1-}" == -m && "${2-}" == pytest ]]; then\n'
                '  touch "${SMOKE_MARKERS}/collected"\n'
                '  echo "collection fixture"\n'
                '  exit "${COLLECT_STATUS}"\n'
                "fi\n"
                f'exec {shlex.quote(sys.executable)} "$@"\n'
            )
            wrapper.chmod(0o755)
            uv = binary / "uv"
            uv.write_text(
                '#!/usr/bin/env bash\ntouch "${SMOKE_MARKERS}/dependency-checked"\n'
            )
            uv.chmod(0o755)
            # Execute the real shell function without installing any packages.
            source = (ROOT / "tools/setup.sh").read_text()
            body = source.split("run_smoke_checks() {", 1)[1].split("\n}\n", 1)[0]
            script = (
                "set -euo pipefail\n"
                "ok() { :; }\n"
                'die() { echo "$1" >&2; exit 1; }\n'
                'warn() { echo "$1"; }\n'
                "SKIP_SMOKE=0\nDRY_RUN=0\nCOMPILER=flagtree\nBACKEND_VENDOR=metax\n"
                "NO_VENV=0\nSTRICT_PIP_CHECK=1\n"
                f"REPO_ROOT={shlex.quote(str(root))}\n"
                f"PYTHON_BIN={shlex.quote(str(wrapper))}\n"
                "run_smoke_checks() {" + body + "\n}\nrun_smoke_checks\n"
            )
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(root),
                PATH=str(binary) + os.pathsep + env["PATH"],
                SMOKE_MARKERS=str(root),
                COLLECT_STATUS=str(collect_status),
            )
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return (
                result,
                (root / "collected").exists(),
                (root / "dependency-checked").exists(),
            )

    def test_legacy_flagtree_without_optional_tle_runs_all_smoke_phases(self):
        result, collected, checked = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("has_triton_tle(): False", result.stdout)
        self.assertTrue(collected)
        self.assertTrue(checked)

    def test_tle_capability_is_reported_when_available(self):
        result, collected, checked = self.run_smoke(tle=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("has_triton_tle(): True", result.stdout)
        self.assertTrue(collected)
        self.assertTrue(checked)

    def test_broken_compiler_is_still_a_hard_failure(self):
        result, collected, checked = self.run_smoke(broken_compiler=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("import verification failed", result.stderr)
        self.assertFalse(collected)
        self.assertFalse(checked)

    def test_collection_error_is_still_a_hard_failure(self):
        result, collected, checked = self.run_smoke(collect_status=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pytest collection failed", result.stderr)
        self.assertTrue(collected)
        self.assertFalse(checked)


if __name__ == "__main__":
    unittest.main()
