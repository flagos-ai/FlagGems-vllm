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

import math

import pytest
import torch

import flaggems_vllm

from .conftest import QUICK_MODE

torch_npu = pytest.importorskip("torch_npu")

CHUNK = 64
# H=4 is the production per-rank head count ([1,T,4,128] with A_log[4] /
# dt_bias[512] in the kernel_details records, TP16-sharded); 32 is a stress
# variant beyond the production shape.
HEAD_COUNTS = (4, 32)
D = 128
LOWER_BOUND = -5.0

# Production context: GLM-5.3-Flash-W8A8 serving profile (random 16k-in /
# 1k-out @ 4-way concurrency, TP16 + EP16, MTP spec=3).  Shape source:
# kernel_details.csv of the per-rank baseline profiles (aggregated in
# analysis/op_catalog.csv).
FULL_CASES = (
    # lens: per-sequence token counts (list)
    # ---- production shapes (kernel_details) ----
    [8192],  # full prefill block (8192 tok, 1 seq)
    [8188],  # spec-boundary block (8192-4 draft tokens)
    [8184],  # spec-boundary block (8192-8)
    [8180],  # spec-boundary block (8192-12)
    [40],  # chunked-prefill tail step
    [36],  # chunked-prefill tail step
    [24],  # chunked-prefill tail step
    # ---- stress shapes ----
    [4096, 4096],  # stress: full block split into 2 requests
    [4096, 2044, 2044],  # stress: ragged 3-request split of a full block
    [
        2048,
        2048,
        2048,
        2048,
    ],  # stress: even 4-way split of a full block (constructed; the profile records step totals only)
)
CASES = (FULL_CASES[0], FULL_CASES[1], FULL_CASES[4]) if QUICK_MODE else FULL_CASES

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="the optimized KDA gate cumsum targets Ascend",
)


def _chunk_indices_host(cu_host, chunk):
    import triton

    idx = []
    for s in range(len(cu_host) - 1):
        n = triton.cdiv(cu_host[s + 1] - cu_host[s], chunk)
        idx.extend((s, c) for c in range(n))
    if not idx:
        return torch.zeros((0, 2), dtype=torch.int64)
    return torch.tensor(idx, dtype=torch.int64)


def _official(g, A_log, dt_bias, cu_host, lower_bound, chunk):
    """Torch reference of the fused safe-gate + chunk-local cumsum semantics."""
    _, T, Hh, Dd = g.shape
    gate = lower_bound * torch.sigmoid(
        (g.float() + dt_bias.view(1, 1, Hh, Dd)) * torch.exp(A_log).view(1, 1, Hh, 1)
    )
    out = torch.empty_like(gate)
    for s in range(len(cu_host) - 1):
        bos, eos = cu_host[s], cu_host[s + 1]
        for cs in range(bos, eos, chunk):
            ce = min(cs + chunk, eos)
            out[0, cs:ce] = gate[0, cs:ce].cumsum(dim=0)
    return out * (1.0 / math.log(2.0))


def _make_inputs(lens, dtype=torch.bfloat16, H=4):
    torch.manual_seed(hash(tuple(lens)) % 100000)
    T = sum(lens)
    cu_host = [0]
    for n in lens:
        cu_host.append(cu_host[-1] + n)
    cu_dev = torch.tensor(cu_host, dtype=torch.int64, device="npu")
    chunk_indices = _chunk_indices_host(cu_host, CHUNK).to("npu")
    g = (torch.rand(1, T, H, D, dtype=torch.float32, device="npu") * 6 - 3).to(dtype)
    A_log = torch.rand(H, dtype=torch.float32, device="npu") - 2.0
    dt_bias = torch.rand(H * D, dtype=torch.float32, device="npu") * 0.4 - 0.2
    return g, A_log, dt_bias, cu_dev, chunk_indices, cu_host


def _load_ascendc_op() -> bool:
    """Load the replaced AscendC operator when the vllm-ascend extension is
    available (CI/dev boxes with vllm_ascend installed); plain flaggems
    environments fall back to the torch-chain reference instead."""
    try:
        import os

        import vllm_ascend

        so = os.path.join(
            os.path.dirname(vllm_ascend.__file__),
            "vllm_ascend_C.cpython-312-aarch64-linux-gnu.so",
        )
        if not os.path.exists(so):
            return False
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env(include_vendor_lib=True)
        torch.ops.load_library(so)
        return hasattr(torch.ops._C_ascend, "kda_gate_cumsum")
    except Exception:
        return False


HAS_ASCENDC = _load_ascendc_op()


def _ascendc(g, A_log, dt_bias, cu_dev, chunk_indices, lower_bound, chunk):
    """The exact replaced operator (host contract included: host cu_seqlens)."""
    return torch.ops._C_ascend.kda_gate_cumsum(
        g.contiguous(),
        chunk,
        A_log=A_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        cu_seqlens=tuple(cu_dev.tolist()),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        layout="BSND",
    )


# Shape provenance: the T values (8192/8180/8184/8188 series and the small
# chunked-prefill tails 56/24/40) are the real per-step token counts recorded
# in the production torch profiler (kernel_details of the GLM-5.3-Flash baseline
# run); the ragged tuples are edge cases beyond the profiled set.


@pytest.mark.kda_gate_cumsum
@pytest.mark.parametrize("lens", CASES)
@pytest.mark.parametrize("H", HEAD_COUNTS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_kda_gate_cumsum_accuracy(lens, H, dtype):
    g, A_log, dt_bias, cu_dev, chunk_indices, cu_host = _make_inputs(lens, dtype, H)
    expected = _official(g, A_log, dt_bias, cu_host, LOWER_BOUND, CHUNK)
    actual = flaggems_vllm.kda_gate_cumsum_triton(
        g, A_log, dt_bias, cu_dev, chunk_indices, LOWER_BOUND, CHUNK
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.kda_gate_cumsum
def test_kda_gate_cumsum_empty_batch():
    g, A_log, dt_bias, cu_dev, chunk_indices, cu_host = _make_inputs([0])
    out = flaggems_vllm.kda_gate_cumsum_triton(
        g, A_log, dt_bias, cu_dev, chunk_indices, LOWER_BOUND, CHUNK
    )
    assert out.shape == g.shape


@pytest.mark.kda_gate_cumsum
@pytest.mark.skipif(not HAS_ASCENDC, reason="vllm_ascend extension not available")
@pytest.mark.parametrize("lens", CASES)
@pytest.mark.parametrize("H", HEAD_COUNTS)
def test_kda_gate_cumsum_matches_ascendc(lens, H):
    """Parity against the exact replaced AscendC operator."""
    g, A_log, dt_bias, cu_dev, chunk_indices, cu_host = _make_inputs(lens, H=H)
    ref = _ascendc(g, A_log, dt_bias, cu_dev, chunk_indices, LOWER_BOUND, CHUNK)
    actual = flaggems_vllm.kda_gate_cumsum_triton(
        g, A_log, dt_bias, cu_dev, chunk_indices, LOWER_BOUND, CHUNK
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-5)
