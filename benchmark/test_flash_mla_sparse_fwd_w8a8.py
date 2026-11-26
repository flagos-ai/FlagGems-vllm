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

from pathlib import Path

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    assert_sparse_fixture_close,
    flashmla_reference,
    make_sparse_inputs,
    quantize_sparse_fp8_fixture,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin

PAGE_TOKENS = 64
NOPE = 448
ROPE = 64
GROUP = 64
SCALE_BYTES = 8
DATA_BYTES = NOPE + ROPE * 2
TOKEN_BYTES = DATA_BYTES + SCALE_BYTES
PAGE_PAYLOAD = PAGE_TOKENS * TOKEN_BYTES
PAGE_STRIDE = ((PAGE_PAYLOAD + DATA_BYTES - 1) // DATA_BYTES) * DATA_BYTES


def pack_v4_cache(kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native packed pages and the corresponding BF16 reference KV."""
    tokens = kv.shape[0]
    assert tokens > 0 and kv.shape == (tokens, 1, NOPE + ROPE)
    assert kv.dtype == torch.bfloat16

    pages = (tokens + PAGE_TOKENS - 1) // PAGE_TOKENS
    padded = torch.zeros(
        (pages * PAGE_TOKENS, NOPE + ROPE), dtype=kv.dtype, device=kv.device
    )
    padded[:tokens].copy_(kv[:, 0])
    padded = padded.view(pages, PAGE_TOKENS, NOPE + ROPE)
    groups = (
        padded[..., :NOPE].float().reshape(pages, PAGE_TOKENS, NOPE // GROUP, GROUP)
    )
    exponent = (groups.abs().amax(-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
    scales = torch.exp2(exponent)
    fp8 = (groups / scales[..., None]).to(torch.float8_e4m3fn)

    page_bytes = torch.zeros((pages, PAGE_STRIDE), dtype=torch.uint8, device=kv.device)
    data_end = PAGE_TOKENS * DATA_BYTES
    payload = page_bytes[:, :data_end].view(pages, PAGE_TOKENS, DATA_BYTES)
    payload[..., :NOPE] = fp8.view(torch.uint8).reshape(pages, PAGE_TOKENS, NOPE)
    payload[..., NOPE:] = padded[..., NOPE:].contiguous().view(torch.uint8)
    scale = page_bytes[:, data_end:PAGE_PAYLOAD].view(pages, PAGE_TOKENS, SCALE_BYTES)
    scale[..., : NOPE // GROUP] = (exponent + 127).to(torch.uint8)

    reference = padded.clone()
    reference[..., :NOPE] = (fp8.float() * scales[..., None]).reshape(
        pages, PAGE_TOKENS, NOPE
    )
    packed = page_bytes[:, :PAGE_PAYLOAD].view(pages, PAGE_TOKENS, 1, TOKEN_BYTES)
    return packed, reference.view(-1, 1, NOPE + ROPE)[:tokens]


def quantize_v4_query(q: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Prepare the native V4 448 FP8 Q and 64 BF16 RoPE."""
    batch, heads, width = q.shape
    assert width == NOPE + ROPE and q.dtype == torch.bfloat16
    blocks = q[..., :NOPE].float().reshape(batch, heads, NOPE // GROUP, GROUP)
    exponent = (blocks.abs().amax(-1) / 448.0).clamp_min(1.0e-4).log2().ceil()
    scales = torch.exp2(exponent)
    fp8 = (blocks / scales[..., None]).to(torch.float8_e4m3fn)
    rope = q[..., NOPE:].contiguous()
    reference = q.clone()
    reference[..., :NOPE] = (fp8.float() * scales[..., None]).reshape(
        batch, heads, NOPE
    )
    return fp8.reshape(batch, heads, NOPE), rope, scales, reference


def native_v4_inputs(inputs):
    query, kv, indices, scale, value_dim, sink, length = inputs
    if sink is not None or length is not None:
        raise ValueError("Native V4 benchmark requires fixed top-k without sinks")
    fp8, rope, scales, query_reference = quantize_v4_query(query)
    cache, kv_reference = pack_v4_cache(kv)
    baseline = query_reference, kv_reference, indices, scale, value_dim, sink, length
    packed = (
        fp8[:, None],
        rope[:, None],
        cache,
        None,
        scales[:, None],
        None,
        indices,
        scale,
    )
    return baseline, packed


class FlashmlaSparseBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            flashmla_reference().flash_mla_sparse_fwd,
            [torch.bfloat16],
        )
        self.set_gems(flaggems_vllm.flash_mla_sparse_fwd_w8a8)

    def set_shapes(self, shape_file_path):
        super().set_shapes(shape_file_path)
        if Path(shape_file_path).samefile(Path(__file__).with_name("core_shapes.yaml")):
            # Built-in D512 shapes use Model1 pages, including in parallel workers.
            self.shapes = [
                (*shape, "v4") if shape[2] == 512 and shape[-1] != "v4" else shape
                for shape in self.shapes
            ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            native_v4 = shape[-1] == "v4"
            parameters = shape[:-1] if native_v4 else shape
            if len(parameters) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            inputs = make_sparse_inputs(parameters, dtype=dtype)
            if native_v4:
                inputs, packed = native_v4_inputs(inputs)
            else:
                packed = quantize_sparse_fp8_fixture(*inputs)
            reference, _, reference_lse = self.torch_op(*inputs)
            output, lse = self.gems_op(*packed)
            channels = (slice(None), slice(448, None)) if native_v4 else (slice(None),)
            for columns in channels:
                assert_sparse_fixture_close(
                    output[:, 0, :, columns],
                    lse[:, :, 0],
                    reference[..., columns],
                    reference_lse,
                )
            yield inputs, packed, shape

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, packed, _ = args[0]
        selected = packed if op is self.gems_op else original
        return super().get_latency(op, *selected, **kwargs)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE")
def test_flash_mla_sparse_fwd_w8a8():
    FlashmlaSparseBenchmark().run()
