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

    mhc_pre_clamp_sinkhorn_backward

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


_pre_bwd_mod = _load(
    "ascend_mhc_pre_clamp_sinkhorn_backward", "mhc_pre_clamp_sinkhorn_backward.py"
)

mhc_pre_clamp_sinkhorn_backward = _pre_bwd_mod.mhc_pre_clamp_sinkhorn_backward
mhc_pre_clamp_sinkhorn_backward_ref = _pre_bwd_mod.mhc_pre_clamp_sinkhorn_backward_ref

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


def _torch_pre_forward(x, phi, alpha, base, norm_eps=1e-6, hc_eps=1e-6):
    """Test-only torch forward producing the saved intermediates that
    mhc_pre_clamp_sinkhorn_backward consumes (avoids depending on the
    forward Triton op in this PR)."""
    T, N, D = x.shape
    x_flat = x.reshape(T, N * D).float()
    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(ms + norm_eps).squeeze(-1)
    x_scaled = x_flat * inv_rms.unsqueeze(-1)
    mixes = x_scaled @ phi.float().t()
    a = alpha.float()
    b = base.float()
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    logits = (mixes[:, 2 * N :] * a[2] + b[2 * N :]).reshape(T, N, N)
    return {
        "inv_rms": inv_rms,
        "x_scaled": x_scaled,
        "mixes": mixes,
        "h_res_logits": logits,
        "pre": pre,
    }


# ---------------------------------------------------------------------------
# mhc_pre_clamp_sinkhorn_backward
# ---------------------------------------------------------------------------
@pytest.mark.mhc_pre_clamp_sinkhorn_backward
@pytest.mark.parametrize("t", [64, 512])
@pytest.mark.parametrize("d", [1280, 3584])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mhc_pre_clamp_sinkhorn_backward(t, d, dtype):
    x, phi, alpha, base = gen_pre_inputs(t, d, dtype)
    torch.manual_seed(7)
    grad_y = torch.randn((t, d), dtype=dtype, device=DEVICE)
    grad_post = torch.randn((t, HC_MULT), dtype=torch.float32, device=DEVICE)
    grad_comb = torch.randn((t, HC_MULT, HC_MULT), dtype=torch.float32, device=DEVICE)

    fwd = _torch_pre_forward(x, phi, alpha, base)
    grads = mhc_pre_clamp_sinkhorn_backward(
        x,
        phi,
        alpha,
        base,
        fwd["inv_rms"],
        fwd["x_scaled"],
        fwd["mixes"],
        fwd["h_res_logits"],
        fwd["pre"],
        grad_y,
        grad_post,
        grad_comb,
    )
    refs = mhc_pre_clamp_sinkhorn_backward_ref(
        x, phi, alpha, base, grad_y, grad_post, grad_comb
    )
    names = ["grad_x", "grad_phi", "grad_alpha", "grad_base"]
    for g, r, n in zip(grads, refs, names):
        assert g.shape == r.shape, n
        _assert_close(g.float(), r.float(), dtype, n)
