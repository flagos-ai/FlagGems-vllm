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

"""Compatibility shim for old/new GCUIR formats (shared version, 2026-11-09).

Background (identified during a full review on 220 hosts): in-process pass libraries built after
2026-11-07 20:22 emit three new GCUIR formats that older external gcu-compiler-opt rejects
``invalid properties {operandSegmentSizes ...} for op memref_ext.slice_pad_start``
(about 70 ops across both repos are affected; see workspace compiler_issues/vllm_memref_slice_pad_start.md):

1. ``memref_ext.slice_*`` / ``gcu.matmul`` has one extra trailing empty
   variadic segment (trailing zero);
2. ``gcu.matmul`` carries an inputPrecision attribute;
3. ``memref_ext.memset_start`` carries a properties dict.

This shim wraps ``triton.backends.enflame.toolkit.gcu_compiler_opt``: on the first failure it
downgrades the new format back to the old one and retries once; valid old-format IR is unaffected and
the original error is propagated unchanged. This shared version consolidates six earlier per-op copies
(vllm FLA/parallel_nsa.py, sglang hash_topk.py, etc.) with identical behavior but applying to all kernels.

This module can be removed once the toolchain (external gcu-compiler-opt) understands the new formats.
"""

from __future__ import annotations

import re


def downgrade_gcuir_for_external_tool(text: str) -> str:
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def install_gcuir_text_compat() -> None:
    """Wrap enflame toolkit.gcu_compiler_opt with a global downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_gcuir_text_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            try:
                return original(downgrade_gcuir_for_external_tool(text), *args)
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._gcuir_text_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat
