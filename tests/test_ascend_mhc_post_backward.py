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

"""Accuracy tests for the Ascend MHC (Manifold-constrained Hyper-Connection)
operator family living in
``flaggems_vllm.runtime.backend._ascend.mhc``:

    mhc_post_backward

Each Triton implementation is compared against its pure-PyTorch reference
(suffixed ``_ref``) which lives in the same module and is test-only.
"""

import gc

import pytest
import torch

try:
    import torch_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False

pytestmark = pytest.mark.skipif(not HAS_NPU, reason="requires Ascend NPU")


@pytest.fixture(autouse=True)
def _npu_cleanup():
    """Free device memory between parametrized cases; the mHC refs hold large
    fp32 intermediates and 64G HBM is exhausted after a few big shapes."""
    yield
    gc.collect()
    torch.npu.empty_cache()


import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

_MHC_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/flaggems_vllm/runtime/backend/_ascend/mhc"
)

# Register stub parent packages so the backend op modules can be loaded
# directly by path (with their relative imports intact) without pulling in
# the full flaggems_vllm package, which requires vLLM.
for _pkg in [
    "flaggems_vllm",
    "flaggems_vllm.runtime",
    "flaggems_vllm.runtime.backend",
    "flaggems_vllm.runtime.backend._ascend",
]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

# The leaf package must be a real package (with __path__) for relative
# imports inside the op modules to resolve.
_mhc_pkg = types.ModuleType("flaggems_vllm.runtime.backend._ascend.mhc")
_mhc_pkg.__path__ = [str(_MHC_DIR)]
sys.modules["flaggems_vllm.runtime.backend._ascend.mhc"] = _mhc_pkg


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(
        f"flaggems_vllm.runtime.backend._ascend.mhc.{mod_name}",
        _MHC_DIR / file_name,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_post_bwd_mod = _load("ascend_mhc_post_backward", "mhc_post_backward.py")

mhc_post_backward = _post_bwd_mod.mhc_post_backward
mhc_post_backward_ref = _post_bwd_mod.mhc_post_backward_ref

DEVICE = "npu"
HC_MULT = 4
HC_MIX = HC_MULT * (HC_MULT + 2)  # 24


def _assert_close(actual, expected, dtype, name="", rtol=None, atol=None):
    if rtol is None or atol is None:
        if dtype == torch.float32:
            rtol, atol = 1e-3, 1e-4
        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-3
        else:  # bfloat16
            rtol, atol = 2e-2, 2e-3
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        msg=lambda m: f"{name} mismatch:\n{m}",
    )


# ---------------------------------------------------------------------------
# input generators
# ---------------------------------------------------------------------------
def gen_post_inputs(t, d, dtype=torch.bfloat16, seed=42, dim3=True):
    torch.manual_seed(seed)
    shape_x = (t, HC_MULT, d) if dim3 else (2, t // 2, HC_MULT, d)
    x = torch.randn(shape_x, dtype=dtype, device=DEVICE)
    h_res = torch.randn((t, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE)
    h_out = torch.randn((t, d), dtype=dtype, device=DEVICE)
    h_post = torch.randn((t, HC_MULT), dtype=torch.float32, device=DEVICE)
    return x, h_res, h_out, h_post


# ---------------------------------------------------------------------------
# mhc_post_backward
# ---------------------------------------------------------------------------
@pytest.mark.mhc_post_backward
@pytest.mark.parametrize("t", [1, 64, 512, 2048])
@pytest.mark.parametrize("d", [1280, 3584])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mhc_post_backward(t, d, dtype):
    x, h_res, h_out, h_post = gen_post_inputs(t, d, dtype)
    torch.manual_seed(7)
    grad_y = torch.randn((t, HC_MULT, d), dtype=dtype, device=DEVICE)

    grads = mhc_post_backward(grad_y, x, h_res, h_out, h_post)
    refs = mhc_post_backward_ref(grad_y, x, h_res, h_out, h_post)
    names = ["grad_x", "grad_hres", "grad_hout", "grad_hpost"]
    for g, r, n in zip(grads, refs, names):
        assert g.shape == r.shape, n
        _assert_close(g.float(), r.float(), dtype, n)
