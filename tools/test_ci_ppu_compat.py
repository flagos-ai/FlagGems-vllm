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

"""The required compatibility gate must reject missing and skipped tests."""

import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ppu_compat", Path(__file__).with_name("run_ppu_compat_tests.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReportGateTests(unittest.TestCase):
    def _report(self, names, skipped=(), failed=()):
        root = ET.Element("testsuite")
        for name in names:
            case = ET.SubElement(root, "testcase", name=name + "[case]")
            if name in skipped:
                ET.SubElement(case, "skipped")
            if name in failed:
                ET.SubElement(case, "failure")
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "junit.xml"
        ET.ElementTree(root).write(path)
        return path

    def test_complete_report_passes(self):
        module.validate_report(self._report(module.REQUIRED_FAMILIES))

    def test_empty_or_unrelated_report_fails(self):
        for names in ([], ["unrelated"]):
            with self.subTest(names=names), self.assertRaises(RuntimeError):
                module.validate_report(self._report(names))

    def test_each_required_family_must_execute(self):
        for name in module.REQUIRED_FAMILIES:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                module.validate_report(
                    self._report(module.REQUIRED_FAMILIES, skipped=[name])
                )

    def test_failure_is_not_a_pass(self):
        with self.assertRaises(RuntimeError):
            module.validate_report(
                self._report(
                    module.REQUIRED_FAMILIES, failed=[module.REQUIRED_FAMILIES[0]]
                )
            )
