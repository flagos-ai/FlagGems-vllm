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

import functools

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm

from . import base

vendor = flaggems_vllm.vendor_name
topk_softplus_sqrt = flaggems_vllm.topk_softplus_sqrt

# ---------------------------------------------------------------------------
# Per-vendor TLE / baseline entry points.
#
# HAS_TLE           : this vendor's TLE-optimized kernel is usable here.
# HAS_TLE_HASH      : the TLE-optimized hash-mode kernel is usable here
#                     (mthreads only, for now).
# _gems_tle_op      : callable forcing the TLE kernel.
# _gems_baseline_op : callable forcing the plain Triton kernel.
# ---------------------------------------------------------------------------
HAS_TLE = False
HAS_TLE_HASH = False
_gems_tle_op = None
_gems_baseline_op = topk_softplus_sqrt

if vendor == "ascend":
    try:
        from flaggems_vllm.runtime.backend._ascend.ops.topk_softplus_sqrt import HAS_TLE

        # `use_ascend_tle` is pinned via functools.partial so the callable
        # drops into the benchmark framework's `gems_op` slot unchanged.
        _gems_tle_op = functools.partial(topk_softplus_sqrt, use_ascend_tle=True)
        _gems_baseline_op = functools.partial(topk_softplus_sqrt, use_ascend_tle=False)
    except ImportError:
        pass
elif vendor == "mthreads":
    from flaggems_vllm.runtime.backend._mthreads.ops.topk_softplus_sqrt import HAS_TLE
    from flaggems_vllm.runtime.backend._mthreads.ops.topk_softplus_sqrt import (
        topk_softplus_sqrt_baseline as _gems_baseline_op,
    )
    from flaggems_vllm.runtime.backend._mthreads.ops.topk_softplus_sqrt import (
        topk_softplus_sqrt_tle as _gems_tle_op,
    )

    # mthreads has a dedicated shared-memory kernel for hash mode too
    # (ascend's hash path is TLE-agnostic, so this only applies here).
    HAS_TLE_HASH = HAS_TLE

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


class TopkSoftplusSqrtHashBenchmark(base.Benchmark):
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
            torch.manual_seed(1)
            gating_output = torch.randn(
                (num_tokens, num_experts), dtype=dtype, device=self.device
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
            input_ids = torch.arange(num_tokens, device=self.device, dtype=torch.int32)
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
                None,
                input_ids,
                tid2eid,
            )


_skip_no_tle = pytest.mark.skipif(
    not HAS_TLE,
    reason=f"TLE-optimized kernel is not available on {vendor}",
)
_skip_no_tle_hash = pytest.mark.skipif(
    not HAS_TLE_HASH,
    reason=f"TLE-optimized hash kernel is not available on {vendor}",
)


# ------------------------------- dense path --------------------------------
@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt():
    """Default entry point. Runs whether or not TLE is available."""
    bench = TopkSoftplusSqrtBenchmark(
        op_name="topk_softplus_sqrt",
        torch_op=_baseline_op,
        gems_op=_gems_baseline_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@_skip_no_tle
def test_topk_softplus_sqrt_tle():
    """TLE-optimized kernel (dense path)."""
    bench = TopkSoftplusSqrtBenchmark(
        op_name="topk_softplus_sqrt_tle",
        torch_op=_baseline_op,
        gems_op=_gems_tle_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()


# ------------------------------- hash path ---------------------------------
@pytest.mark.topk_softplus_sqrt
def test_topk_softplus_sqrt_hash():
    """Baseline kernel, hash mode."""
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_hash",
        torch_op=_baseline_op,
        gems_op=_gems_baseline_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.topk_softplus_sqrt
@_skip_no_tle_hash
def test_topk_softplus_sqrt_tle_hash():
    """TLE-optimized kernel, hash mode (mthreads only)."""
    bench = TopkSoftplusSqrtHashBenchmark(
        op_name="topk_softplus_sqrt_tle_hash",
        torch_op=_baseline_op,
        gems_op=_gems_tle_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()
