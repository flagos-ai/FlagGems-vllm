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

from . import base

vendor = flaggems_vllm.vendor_name

try:
    if vendor == "mthreads":
        from vllm_musa import _custom_ops as vendor_ops

        reference_fused_q_kv_rmsnorm = vendor_ops.deepseek_v4_fused_q_kv_rmsnorm
    elif vendor == "nvidia":
        from vllm.v1.attention.ops.deepseek_v4_ops import (
            fused_q_kv_rmsnorm as reference_fused_q_kv_rmsnorm,
        )
    elif vendor == "ascend":
        import torchair
        from torchair.configs.compiler_config import CompilerConfig

        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        @torch.compile(backend=npu_backend, dynamic=False)
        def reference_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps=1e-6):
            q_var = qr.pow(2).mean(-1, keepdim=True)
            q_normed = qr * torch.rsqrt(q_var + eps) * q_weight

            kv_var = kv.pow(2).mean(-1, keepdim=True)
            kv_normed = kv * torch.rsqrt(kv_var + eps) * kv_weight

            return q_normed, kv_normed

    _HAS_REFERENCE_FUSED_Q_KV_RMSNORM = True
except Exception as e:
    print(e)
    reference_fused_q_kv_rmsnorm = None
    _HAS_REFERENCE_FUSED_Q_KV_RMSNORM = False


class FusedQKVRMSNormBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_q_kv_rmsnorm",
            reference_fused_q_kv_rmsnorm,
            [torch.bfloat16],
            # Use the top-level API so vendor-specific backend
            # overrides are respected.
            gems_op=flaggems_vllm.fused_q_kv_rmsnorm,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            (1, 1536, 512),
            (32, 1536, 512),
            (128, 1536, 512),
            (512, 1536, 512),
            (2048, 1536, 512),
        ] + ([(32, 64 * 576, 576), (128, 64 * 576, 576)] if vendor != "ascend" else [])

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        for tokens, qdim, kvdim in self.shapes:
            qr = torch.randn(
                (tokens, qdim),
                device=device,
                dtype=dtype,
            )
            kv = torch.randn(
                (tokens, kvdim),
                device=device,
                dtype=dtype,
            )
            q_weight = torch.randn(
                (qdim,),
                device=device,
                dtype=dtype,
            )
            kv_weight = torch.randn(
                (kvdim,),
                device=device,
                dtype=dtype,
            )

            yield (
                qr,
                kv,
                q_weight,
                kv_weight,
                1e-6,
            )


@pytest.mark.fused_q_kv_rmsnorm
@pytest.mark.skipif(
    not _HAS_REFERENCE_FUSED_Q_KV_RMSNORM,
    reason="requires fused_q_kv_rmsnorm reference implementation",
)
def test_fused_q_kv_rmsnorm_benchmark():
    FusedQKVRMSNormBenchmark().run()
