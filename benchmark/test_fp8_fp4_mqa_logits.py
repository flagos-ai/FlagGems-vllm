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

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops import fp8_fp4_mqa_logits
from tests.fp8_fp4_quant import quantize_to_mxfp4
from tests.test_fp8_fp4_mqa_logits import _ref_dense_mqa

from . import base

_vendor = flaggems_vllm.vendor_name
_skip_arch = False
_skip_ref = False

if _vendor == "mthreads":
    from tests.fp8_fp4_quant import per_custom_dims_cast_to_fp8
elif _vendor == "nvidia":
    try:
        from vllm.platforms import current_platform
        from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8

        if not current_platform.has_device_capability(90):
            _skip_arch = "requires CUDA with Hopper architecture (SM90+)"
    except ImportError:
        _skip_ref = "requires vLLM with DeepGEMM and FP8 quantization support"
else:
    _skip_arch = f"unsupported vendor: {_vendor}"
    _skip_ref = f"unsupported vendor: {_vendor}"

# FP4 dense DeepGEMM requires SM120+; skip on unsupported architectures
_fp4_supported = True
if _vendor == "nvidia":
    try:
        cap = torch.cuda.get_device_capability()
        if cap[0] < 12:
            _fp4_supported = False
    except Exception:
        pass
elif _vendor == "mthreads":
    _fp4_supported = False


# DeepSeek V4 production config
H = 64
D = 128


def _build_case(M, N, dtype, device, use_fp4=False):
    """Build FP8 quantized inputs for benchmarking."""
    q_bf16 = torch.randn(M, H, D, device=device, dtype=dtype)
    k_bf16 = torch.randn(N, D, device=device, dtype=dtype)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()

    if use_fp4:
        q_packed, q_scale = quantize_to_mxfp4(q_bf16)
    else:
        q_packed = q_bf16.clamp(
            min=torch.finfo(torch.float8_e4m3fn).min,
            max=torch.finfo(torch.float8_e4m3fn).max,
        ).to(torch.float8_e4m3fn)
        q_scale = None

    k_fp8, k_scale = per_custom_dims_cast_to_fp8(k_bf16, (0,), False)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    return q_packed, q_scale, k_fp8, k_scale, weights, ks, ke


# Shapes: (M, N)
BENCH_SHAPES = [
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 2048),
    (4, 4096),
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 8192),
    (1024, 8192),
]


class Fp8Fp4MqaLogitsBenchmark(base.Benchmark):
    """Common benchmark class for fp8/fp4 mqa logits"""

    use_fp4: bool

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(BENCH_SHAPES)

    def get_input_iter(self, dtype):
        for M, N in self.shapes:
            q_packed, q_scale, k_fp8, k_scale, weights, ks, ke = _build_case(
                M, N, dtype, self.device, use_fp4=self.use_fp4
            )
            yield ((q_packed, q_scale), k_fp8, k_scale, weights, ks, ke)


def _vllm_wrapper(q, k_fp8, k_scale, weights, ks, ke):
    """Baseline: platform-specific DeepGEMM kernel (FP8 only currently)."""
    return _ref_dense_mqa(
        q=q,
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=True,
    )


def _gems_wrapper(q, k_fp8, k_scale, weights, ks, ke):
    return fp8_fp4_mqa_logits(
        q=q,
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
        clean_logits=True,
    )


@pytest.mark.skipif(_skip_arch, reason=_skip_arch or "")
@pytest.mark.skipif(_skip_ref, reason=_skip_ref or "")
@pytest.mark.fp8_fp4_mqa_logits
def test_fp8_mqa_logits():
    bench = Fp8Fp4MqaLogitsBenchmark(
        op_name="fp8_mqa_logits",
        torch_op=_vllm_wrapper,
        gems_op=_gems_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.use_fp4 = False
    bench.run()


@pytest.mark.skipif(_skip_arch, reason=_skip_arch or "")
@pytest.mark.skipif(_skip_ref, reason=_skip_ref or "")
@pytest.mark.skipif(
    not _fp4_supported,
    reason="FP4 dense DeepGEMM requires SM120+",
)
@pytest.mark.fp8_fp4_mqa_logits
def test_fp4_mqa_logits():
    bench = Fp8Fp4MqaLogitsBenchmark(
        op_name="fp4_mqa_logits",
        torch_op=_vllm_wrapper,
        gems_op=_gems_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.use_fp4 = True
    bench.run()
