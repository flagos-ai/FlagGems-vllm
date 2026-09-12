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
import torch.nn.functional as F

from flaggems_vllm.ops.topk_softplus_sqrt import (
    HAS_TLE,
    topk_softplus_sqrt,
    topk_softplus_sqrt_baseline,
    topk_softplus_sqrt_tle,
)

from . import base

try:
    from vllm._custom_ops import topk_hash_softplus_sqrt as _vllm_topk_softplus_sqrt

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    _vllm_topk_softplus_sqrt = None


def _vllm_topk_softplus_sqrt_wrapper(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """vLLM CUDA kernel baseline."""
    _vllm_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        input_ids,
        tid2eid,
    )


def _torch_topk_softplus_sqrt_ref(
    topk_weights,
    topk_indices,
    token_expert_indices,
    gating_output,
    renormalize,
    routed_scaling_factor,
    correction_bias=None,
    input_ids=None,
    tid2eid=None,
):
    """Pure-PyTorch fallback reference (used when vLLM is not installed)."""
    num_tokens = gating_output.shape[0]
    topk = topk_weights.shape[1]

    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    if tid2eid is not None:
        assert input_ids is not None
        top_ids = tid2eid[input_ids.long()]
    else:
        top_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    top_weights = original_scores.gather(1, top_ids.long())
    if renormalize:
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        top_weights = top_weights * routed_scaling_factor

    topk_weights.copy_(top_weights.to(torch.float32))
    topk_indices.copy_(top_ids.to(torch.int32))
    tei = torch.arange(num_tokens, device=gating_output.device).unsqueeze(1) * topk
    tei = tei + torch.arange(topk, device=gating_output.device).unsqueeze(0)
    token_expert_indices.copy_(tei.to(torch.int32))


_baseline_op = (
    _vllm_topk_softplus_sqrt_wrapper if HAS_VLLM else _torch_topk_softplus_sqrt_ref
)


class TopkSoftplusSqrtBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 256, 6),
            (10, 256, 6),
            (16, 256, 6),
            (128, 256, 6),
            (512, 256, 6),
            (1024, 256, 6),
            (2048, 256, 6),
            (4096, 256, 6),
        ]

    def get_input_iter(self, dtype):
        for num_tokens, num_experts, topk in self.shapes:
            torch.manual_seed(0)
            gating_output = torch.randn(
                (num_tokens, num_experts), dtype=dtype, device=self.device
            )
            correction_bias = torch.randn(
                (num_experts,), dtype=torch.float32, device=self.device
            )
            topk_weights = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device=self.device
            )
            topk_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=self.device
            )
            token_expert_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=self.device
            )
            yield (
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                True,
                1.0,
                correction_bias,
            )


class TopkSoftplusSqrtLargeBenchmark(TopkSoftplusSqrtBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (32768, 256, 8),
            (65536, 256, 8),
            (131072, 256, 8),
            (4096, 1024, 8),
            (4096, 2048, 8),
            (4096, 4096, 8),
            (4096, 256, 16),
            (4096, 256, 32),
            (4096, 256, 64),
            (4096, 2048, 32),
            (16384, 2048, 32),
        ]


class TopkSoftplusSqrtHashBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, num_experts, topk"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (256, 256, 8),
            (1024, 256, 8),
            (4096, 256, 8),
            (16384, 256, 8),
            (4096, 128, 8),
            (4096, 512, 8),
            (32768, 256, 8),
            (65536, 256, 8),
            (131072, 256, 8),
            (4096, 1024, 8),
            (4096, 2048, 8),
            (4096, 4096, 8),
            (4096, 256, 16),
            (4096, 256, 32),
            (4096, 256, 64),
            (4096, 2048, 32),
            (16384, 2048, 32),
        ]

    def get_input_iter(self, dtype):
        for num_tokens, num_experts, topk in self.shapes:
            torch.manual_seed(1)
            gating_output = torch.randn(
                (num_tokens, num_experts), dtype=dtype, device=self.device
            )
            correction_bias = torch.randn(
                (num_experts,), dtype=torch.float32, device=self.device
            )
            topk_weights = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device=self.device
            )
            topk_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=self.device
            )
            token_expert_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=self.device
            )
            input_ids = torch.arange(
                num_tokens, device=self.device, dtype=torch.int64
            )
            tid2eid = torch.randint(
                0,
                num_experts,
                (num_tokens, topk),
                device=self.device,
                dtype=torch.int32,
            )
            yield (
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                True,
                1.0,
                correction_bias,
                input_ids,
                tid2eid,
            )


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt():
    bench = TopkSoftplusSqrtBenchmark(
        op_name="topk_softplus_sqrt",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_baseline_triton():
    bench = TopkSoftplusSqrtBenchmark(
        op_name="topk_softplus_sqrt_baseline_triton",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(
    not HAS_TLE,
    reason="Triton TLE (triton.experimental.tle) is not available in this environment",
)
def test_topk_softplus_sqrt_tle():
    bench = TopkSoftplusSqrtBenchmark(
        op_name="topk_softplus_sqrt_tle",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_tle,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_baseline_triton_large():
    bench = TopkSoftplusSqrtLargeBenchmark(
        op_name="topk_softplus_sqrt_baseline_triton_large",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(
    not HAS_TLE,
    reason="Triton TLE (triton.experimental.tle) is not available in this environment",
)
def test_topk_softplus_sqrt_tle_large():
    bench = TopkSoftplusSqrtLargeBenchmark(
        op_name="topk_softplus_sqrt_tle_large",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_tle,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_baseline_triton_large_fp32():
    bench = TopkSoftplusSqrtLargeBenchmark(
        op_name="topk_softplus_sqrt_baseline_triton_large_fp32",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_baseline,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(
    not HAS_TLE,
    reason="Triton TLE (triton.experimental.tle) is not available in this environment",
)
def test_topk_softplus_sqrt_tle_large_fp32():
    bench = TopkSoftplusSqrtLargeBenchmark(
        op_name="topk_softplus_sqrt_tle_large_fp32",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_tle,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_baseline_triton_hash():
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_baseline_triton_hash",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(
    not HAS_TLE,
    reason="Triton TLE (triton.experimental.tle) is not available in this environment",
)
def test_topk_softplus_sqrt_tle_hash():
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_tle_hash",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_tle,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_baseline_triton_hash_fp32():
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_baseline_triton_hash_fp32",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_baseline,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(
    not HAS_TLE,
    reason="Triton TLE (triton.experimental.tle) is not available in this environment",
)
def test_topk_softplus_sqrt_tle_hash_fp32():
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_tle_hash_fp32",
        torch_op=_baseline_op,
        gems_op=topk_softplus_sqrt_tle,
        dtypes=[torch.float32],
    )
    bench.run()
