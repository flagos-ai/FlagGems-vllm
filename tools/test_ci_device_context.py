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

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class DeviceContextFailure(Exception):
    pass


class DeviceContextGuardTest(unittest.TestCase):
    def setUp(self):
        # Execute the real hooks without importing accelerator-only dependencies.
        source = Path(__file__).resolve().parents[1] / "tests/conftest.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {
            "_probe_device_context",
            "pytest_runtest_setup",
            "pytest_runtest_teardown",
            "get_reason",
            "pytest_runtest_logreport",
        }
        hooks = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(len(hooks), len(names))
        self.backend = Mock()
        self.backend.is_available.return_value = True
        self.pytest = SimpleNamespace(
            fail=Mock(side_effect=DeviceContextFailure),
            skip=Mock(side_effect=unittest.SkipTest),
            hookimpl=lambda **kwargs: lambda function: function,
        )
        self.namespace = {
            "_DEVICE_CONTEXT_DEAD": False,
            "RECORD_LOG": False,
            "TEST_RESULTS": {},
            "flaggems_vllm": SimpleNamespace(
                runtime=SimpleNamespace(torch_device_fn=self.backend)
            ),
            "logging": Mock(),
            "pytest": self.pytest,
        }
        module = ast.Module(body=hooks, type_ignores=[])
        exec(compile(module, str(source), "exec"), self.namespace)
        self.item = SimpleNamespace(nodeid="test_launch.py::test_async_launch")

    def test_async_fault_fails_current_test_and_skips_later_tests(self):
        self.backend.synchronize.side_effect = RuntimeError("device context lost")

        with self.assertRaises(DeviceContextFailure):
            self.namespace["pytest_runtest_teardown"](self.item, None)

        self.assertTrue(self.namespace["_DEVICE_CONTEXT_DEAD"])
        self.pytest.fail.assert_called_once_with(
            f"device context poisoned after {self.item.nodeid}"
        )
        with self.assertRaises(unittest.SkipTest):
            self.namespace["pytest_runtest_setup"](self.item)
        self.namespace["pytest_runtest_teardown"](self.item, None)
        self.backend.synchronize.assert_called_once_with()

    def test_healthy_context_keeps_scheduling_tests(self):
        self.namespace["pytest_runtest_teardown"](self.item, None)
        self.namespace["pytest_runtest_setup"](self.item)

        self.assertFalse(self.namespace["_DEVICE_CONTEXT_DEAD"])
        self.backend.synchronize.assert_called_once_with()
        self.pytest.fail.assert_not_called()
        self.pytest.skip.assert_not_called()

    def test_teardown_fault_updates_a_successful_call_report(self):
        report = self.namespace["pytest_runtest_logreport"]
        report(
            SimpleNamespace(
                nodeid=self.item.nodeid, when="call", outcome="passed", longrepr=None
            )
        )
        report(
            SimpleNamespace(
                nodeid=self.item.nodeid,
                when="teardown",
                outcome="failed",
                longrepr="asynchronous device fault",
            )
        )
        result = self.namespace["TEST_RESULTS"][self.item.nodeid]
        self.assertEqual(result["result"], "failed")
        self.assertEqual(result["reason"], "asynchronous device fault")

    def test_teardown_fault_preserves_the_original_call_failure(self):
        report = self.namespace["pytest_runtest_logreport"]
        for when, reason in [
            ("call", "original kernel fault"),
            ("teardown", "device context lost"),
        ]:
            report(
                SimpleNamespace(
                    nodeid=self.item.nodeid,
                    when=when,
                    outcome="failed",
                    longrepr=reason,
                )
            )
        result = self.namespace["TEST_RESULTS"][self.item.nodeid]
        self.assertEqual(result["result"], "failed")
        self.assertEqual(result["reason"], "original kernel fault")

    def test_healthy_teardown_preserves_the_call_outcome(self):
        report = self.namespace["pytest_runtest_logreport"]
        for outcome in ["passed", "failed", "skipped"]:
            with self.subTest(outcome=outcome):
                report(
                    SimpleNamespace(
                        nodeid=self.item.nodeid,
                        when="call",
                        outcome=outcome,
                        longrepr="original outcome",
                    )
                )
                expected = dict(self.namespace["TEST_RESULTS"][self.item.nodeid])
                report(
                    SimpleNamespace(
                        nodeid=self.item.nodeid,
                        when="teardown",
                        outcome="passed",
                        longrepr=None,
                    )
                )
                self.assertEqual(
                    self.namespace["TEST_RESULTS"][self.item.nodeid], expected
                )

    def test_unavailable_backend_does_not_require_synchronization(self):
        self.backend.is_available.return_value = False

        self.namespace["pytest_runtest_teardown"](self.item, None)
        self.namespace["pytest_runtest_setup"](self.item)

        self.assertFalse(self.namespace["_DEVICE_CONTEXT_DEAD"])
        self.backend.synchronize.assert_not_called()
        self.pytest.fail.assert_not_called()
        self.pytest.skip.assert_not_called()
