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
from tests.compressor_cases import PRODUCTION_CASES, make_inputs

from . import base

try:
    import vllm_ascend.utils as vllm_ascend_utils

    HAS_VLLM_ASCEND = True
except ImportError:
    vllm_ascend_utils = None
    HAS_VLLM_ASCEND = False


def _input_fn(case_name, dtype, device):
    inputs = make_inputs(case_name, device=device, seed=42)
    yield (inputs,)


class CompressorBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Each "shape" is a production case name; the concrete layout comes
        # from the shared case registry (see tests/compressor_cases.py).
        self.shapes = list(PRODUCTION_CASES)
        self.shape_desc = "case(B,S,T,H,D,R,C)"

    def set_more_shapes(self):
        return []


@pytest.mark.compressor
@pytest.mark.skipif(
    not HAS_VLLM_ASCEND or flaggems_vllm.vendor_name != "ascend",
    reason="compressor benchmark needs the ascend vendor and the vllm-ascend AscendC baseline",
)
def test_compressor():
    if not vllm_ascend_utils.enable_custom_op():
        pytest.skip("vllm-ascend custom operators are unavailable")
    baseline = torch.ops._C_ascend.compressor

    bench = CompressorBenchmark(
        op_name="compressor",
        input_fn=_input_fn,
        torch_op=baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.compressor)
    bench.run()


def _paged_public_call(*args, **overrides):
    x, wkv, wgate, ape, state, pos, slots, table, requests, ends, weight, rope = args[
        :12
    ]
    ratio = args[12] if len(args) > 12 else 4
    settings = dict(
        state_block_table=table,
        cmp_ratio=ratio,
        coff=2 if ratio == 4 else 1,
        rope_head_dim=rope.shape[1],
        positions=pos,
        slot_mapping=slots,
        output_request_ids=requests,
        output_positions=ends,
        cos_sin_cache=rope,
    )
    settings.update(overrides)
    return flaggems_vllm.compressor(x, wkv, wgate, state, ape, weight, **settings)


def _paged_make_inputs(tokens, hidden, dim, dtype, device):
    generator = torch.Generator().manual_seed(817)

    def sample(shape):
        return (torch.randn(shape, generator=generator) * 0.03).to(
            device=device, dtype=dtype
        )

    ratio, block = 4, 16
    pages = (tokens + block - 1) // block
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    ends = torch.arange(ratio - 1, tokens, ratio, dtype=torch.int64, device=device)
    angle = positions.float()[:, None] * torch.linspace(
        0.01, 0.07, min(32, dim // 2), device=device
    )
    rope = torch.cat([angle.cos(), angle.sin()], -1)
    return (
        sample((tokens, hidden)),
        sample((2 * dim, hidden)),
        sample((2 * dim, hidden)),
        sample((ratio, 2 * dim)).float(),
        torch.zeros((pages, block, 4 * dim), device=device),
        positions,
        positions.clone(),
        torch.arange(pages, dtype=torch.int64, device=device)[None, :],
        torch.zeros_like(ends),
        ends,
        torch.ones(dim, dtype=dtype, device=device),
        rope,
        ratio,
    )


def _paged_torch_reference(
    x,
    wkv,
    wgate,
    ape,
    state,
    positions,
    slots,
    table,
    requests,
    ends,
    weight,
    rope,
    ratio=4,
):
    # This benchmark uses valid unique slots, one request, and contiguous prefill.
    dim, block = weight.numel(), state.shape[1]
    kv = torch.nn.functional.linear(x.float(), wkv.float())
    gate = torch.nn.functional.linear(x.float(), wgate.float()) + ape[positions % ratio]
    state[slots // block, slots % block] = torch.cat([kv, gate], -1)
    history = (
        ends[:, None]
        - 2 * ratio
        + 1
        + torch.arange(2 * ratio, device=x.device)[None, :]
    )
    valid = history >= 0
    safe = history.clamp_min(0)
    pages = table[requests[:, None], safe // block]
    raw = state[pages, safe % block]
    half = (torch.arange(2 * ratio, device=x.device) // ratio)[None, :, None]
    columns = torch.arange(dim, device=x.device)[None, None, :]
    values = torch.gather(raw, 2, (half * dim + columns).expand(ends.numel(), -1, -1))
    scores = torch.gather(
        raw, 2, ((2 + half) * dim + columns).expand(ends.numel(), -1, -1)
    )
    scores = scores.masked_fill(~valid[..., None], -float("inf"))
    compressed = (values * torch.softmax(scores, dim=1)).sum(dim=1)
    normed = (
        compressed
        * torch.rsqrt(compressed.square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    )
    rd = rope.shape[1]
    if rd:
        cosine, sine = rope[ends - ends % ratio].chunk(2, -1)
        tail = normed[:, -rd:].clone()
        normed[:, -rd::2] = tail[:, ::2] * cosine - tail[:, 1::2] * sine
        normed[:, -rd + 1 :: 2] = tail[:, 1::2] * cosine + tail[:, ::2] * sine
    return normed.to(x.dtype)


def _paged_input_fn(shape, dtype, device):
    yield _paged_make_inputs(*shape, dtype, device)


class PagedCompressorBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(8, 512, 128), (128, 4096, 128), (257, 1024, 128)]
        self.shape_desc = "(tokens, hidden, head_dim)"

    def set_more_shapes(self):
        return []


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_compressor_paged():
    bench = PagedCompressorBenchmark(
        op_name="compressor",
        input_fn=_paged_input_fn,
        torch_op=_paged_torch_reference,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_paged_public_call)
    bench.run()
