# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Importing the MetaX backend must not replace compiler entry points."""
import subprocess
import sys
from pathlib import Path

import pytest

import flaggems_vllm as ops


@pytest.mark.skipif(ops.vendor_name != "metax", reason="MetaX compiler contract")
def test_backend_import_preserves_compiler_entrypoint():
    source_root = str(Path(ops.__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {source_root!r})\n"
        + """import triton.backends.metax.compiler as compiler
before = compiler.metax.translate_llvmir_to_mcfatbin
import flaggems_vllm
assert flaggems_vllm.__file__.startswith(sys.path[0])
assert compiler.metax.translate_llvmir_to_mcfatbin is before
"""
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
