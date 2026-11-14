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

"""GCUIR 新旧格式兼容 shim(共享版,2026-11-09)。

背景(220 单机全量复核定位):2026-11-07 20:22 后的 in-process pass 库会在
GCUIR 里产出三种新格式;较老的外部 gcu-compiler-opt 拒收并报
``invalid properties {operandSegmentSizes ...} for op memref_ext.slice_pad_start``
(两仓 ~70 op 受影响,详见 workspace compiler_issues/vllm_memref_slice_pad_start.md):

1. ``memref_ext.slice_*`` / ``gcu.matmul`` 的 operandSegmentSizes 多一个尾部空
   variadic 段(尾零);
2. ``gcu.matmul`` 带 inputPrecision 属性;
3. ``memref_ext.memset_start`` 带 properties dict。

本 shim 包装 ``triton.backends.enflame.toolkit.gcu_compiler_opt``:首次失败时把
新格式降级回旧格式重试一次;合法旧格式 IR 不受影响、原错误原样透传。
单 op 私有版先例(vllm FLA/parallel_nsa.py、sglang hash_topk.py 等 6 处)由此
共享版收敛——行为一致、作用域为全部 kernel。

工具链升级(外部 gcu-compiler-opt 认识新格式)后本模块整体可移除。
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
