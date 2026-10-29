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

    mhc_pre_clamp_sinkhorn (forward, aclnn semantics)

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


_pre_mod = _load("ascend_mhc_pre_clamp_sinkhorn", "mhc_pre_clamp_sinkhorn.py")

mhc_pre_clamp_sinkhorn = _pre_mod.mhc_pre_clamp_sinkhorn
mhc_pre_clamp_sinkhorn_ref = _pre_mod.mhc_pre_clamp_sinkhorn_ref

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
def gen_pre_inputs(t, d, dtype=torch.bfloat16, seed=42, dim3=True):
    torch.manual_seed(seed)
    shape_x = (t, HC_MULT, d) if dim3 else (2, t // 2, HC_MULT, d)
    x = torch.randn(shape_x, dtype=dtype, device=DEVICE)
    phi = torch.randn((HC_MIX, HC_MULT * d), dtype=torch.float32, device=DEVICE) * 0.02
    alpha = torch.randn((3,), dtype=torch.float32, device=DEVICE) * 0.1
    base = torch.randn((HC_MIX,), dtype=torch.float32, device=DEVICE) * 0.1
    return x, phi, alpha, base


# ---------------------------------------------------------------------------
# mhc_pre_clamp_sinkhorn (forward)
# ---------------------------------------------------------------------------
@pytest.mark.mhc_pre_clamp_sinkhorn
@pytest.mark.parametrize("t", [1, 64, 512, 4096])
@pytest.mark.parametrize("d", [1280, 3584])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim3", [True, False], ids=["3d", "4d"])
def test_mhc_pre_clamp_sinkhorn(t, d, dtype, dim3):
    if not dim3 and t % 2:
        pytest.skip("4d layout needs even token count")
    x, phi, alpha, base = gen_pre_inputs(t, d, dtype, dim3=dim3)

    out = mhc_pre_clamp_sinkhorn(x, phi, alpha, base)
    ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base)

    assert out["y"].shape == ref["y"].shape and out["y"].dtype == x.dtype
    _assert_close(out["y"].float(), ref["y"].float(), dtype, "pre y")
    _assert_close(out["post_out"], ref["post_out"], torch.float32, "pre post_out")
    # comb_frag is a 20-iteration Sinkhorn-normalized doubly-stochastic
    # matrix; the Triton kernel and the torch ref iterate in different
    # orders, so fp32 rounding accumulates differently (~0.08 max abs on
    # entries in [0, 1]).
    _assert_close(
        out["comb_frag"],
        ref["comb_frag"],
        torch.float32,
        "pre comb_frag",
        rtol=5e-1,
        atol=1e-1,
    )


@pytest.mark.mhc_pre_clamp_sinkhorn
def test_mhc_pre_clamp_sinkhorn_with_clamp():
    x, phi, alpha, base = gen_pre_inputs(256, 1280, torch.bfloat16)
    kwargs = dict(clamp_min=-5.0, clamp_max=5.0, iter_times=20)
    out = mhc_pre_clamp_sinkhorn(x, phi, alpha, base, **kwargs)
    ref = mhc_pre_clamp_sinkhorn_ref(x, phi, alpha, base, **kwargs)
    _assert_close(out["y"].float(), ref["y"].float(), torch.bfloat16, "clamp y")
    _assert_close(
        out["comb_frag"],
        ref["comb_frag"],
        torch.float32,
        "clamp comb_frag",
        rtol=5e-1,
        atol=1e-1,
    )
