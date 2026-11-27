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

import os

# ruff: noqa: I001
os.environ["FLAGTREE_AABS"] = "0"

from itertools import product  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

import flaggems_vllm  # noqa: E402
from benchmark.base import Benchmark  # noqa: E402
from flaggems_vllm.ops.marlin_gemm import UINT4B8_ID  # noqa: E402

GROUP_SIZE = 128

try:
    from vllm._custom_ops import marlin_gemm as _vllm_marlin_gemm
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    def baseline_op(a, b_q_weight, b_scales, workspace, size_m, size_n, size_k):
        return _vllm_marlin_gemm(
            a=a,
            c=None,
            b_q_weight=b_q_weight,
            b_bias=None,
            b_scales=b_scales,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=scalar_types.uint4b8,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
        )

    def make_workspace(device):
        return marlin_make_workspace_new(torch.device(device), 4)

    HAS_BASELINE_OP = True
except ImportError:
    HAS_BASELINE_OP = False


class MarlinGemmBenchmark(Benchmark):
    _ms = [1, 16, 64, 128]
    _ns = [512, 1024, 2048]
    _ks = [512, 1024, 2048]
    _shapes = list(product(_ms, _ns, _ks))

    def set_shapes(self, shape_file_path=None):
        self.shapes = MarlinGemmBenchmark._shapes

    def get_input_iter(self, dtype):
        device = flaggems_vllm.runtime.device.name
        scale_perm = torch.tensor(
            [i + 8 * j for i in range(8) for j in range(8)], device=device
        )
        for m, n, k in self.shapes:
            w = torch.randn((n, k), dtype=dtype, device=device) * 0.1
            grouped = w.reshape(n, k // GROUP_SIZE, GROUP_SIZE).float()
            scales_nk = grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
            q = (
                (torch.round(grouped / scales_nk).clamp(-7, 7) + 8)
                .to(torch.int32)
                .reshape(n, k)
            )
            scales = (
                scales_nk.squeeze(-1)
                .t()
                .reshape(-1, 64)[:, scale_perm]
                .reshape(-1, n)
                .to(dtype)
                .contiguous()
            )

            words = torch.zeros(
                (k // 16) * (n // 64) * 128, dtype=torch.int32, device=device
            )
            kk = torch.arange(16, device=device)
            n_loc = torch.arange(64, device=device)
            waddr2 = 4 * (
                ((n_loc % 8) * 4).unsqueeze(0) + ((kk % 8) // 2).unsqueeze(1)
            ) + (n_loc // 16).unsqueeze(0)
            J4 = (
                4
                * (
                    (4 * (kk % 2) + 2 * (kk // 8)).unsqueeze(1)
                    + ((n_loc // 8) % 2).unsqueeze(0)
                )
            ).int()
            waddr1 = waddr2.reshape(-1).long()
            for kb in range(k // 16):
                for pid_n in range(n // 64):
                    base = (kb * (n // 64) + pid_n) * 128
                    tile = q[pid_n * 64 : (pid_n + 1) * 64, kb * 16 : (kb + 1) * 16].T
                    words.index_put_(
                        (base + waddr1,), (tile << J4).reshape(-1), accumulate=True
                    )
            words = words.view(k // 16, n * 2)

            a = torch.randn((m, k), dtype=dtype, device=device)
            workspace = make_workspace(device)
            yield a, words, scales, workspace, m, n, k


@pytest.mark.skipif(
    not HAS_BASELINE_OP, reason="Missing vLLM marlin_gemm on current platform"
)
@pytest.mark.marlin_gemm
def test_marlin_gemm():
    def gems_op(a, b_q_weight, b_scales, workspace, size_m, size_n, size_k):
        from flaggems_vllm.ops.marlin_gemm import marlin_gemm

        return marlin_gemm(
            a=a,
            c=None,
            b_q_weight=b_q_weight,
            b_bias=None,
            b_scales=b_scales,
            a_scales=None,
            global_scale=None,
            b_zeros=None,
            g_idx=None,
            perm=None,
            workspace=workspace,
            b_q_type=UINT4B8_ID,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
        )

    dtypes = [torch.float16]
    bench = MarlinGemmBenchmark(
        op_name="marlin_gemm",
        torch_op=baseline_op,
        dtypes=dtypes,
    )
    bench.set_gems(gems_op)
    bench.run()
