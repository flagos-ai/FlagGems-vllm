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

from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from packaging.version import Version

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8 import (
    HAS_FP8_MLA,
    MTHREADS_FP8_MLA,
    assert_mthreads_sparse_accuracy,
    assert_sparse_fixture_close,
    flashmla_reference,
    make_mthreads_sparse_inputs,
    make_sparse_inputs,
    mthreads_sparse_reference,
    prepare_flash_mla_sparse_fwd_w8a8,
    quantize_sparse_fp8_fixture,
)

from . import base


def prepare_mate_baseline(inputs, query, *, attn_sink=None, topk_length=None):
    """Prepare native MODEL1 D512 or V3.2 D576 MATE FP8 cache.

    Power-of-two fixture scales preserve the dequantized FP8 mathematical input
    in BF16 query and tail storage. MODEL1 pages place all 576-byte token
    payloads before the 8-byte exponent records. The seven 64-element groups
    use the original token scale. Packing and metadata are outside timing.
    """
    from mate.flashmla import flash_mla_with_kvcache, get_mla_metadata

    _, qrope, kv, krope, _, ks, indices = inputs
    batch, queries, heads, _ = query.shape
    pages = kv.shape[0]
    raw = kv.view(pages, 64, 1, 512)
    token_scale = ks.view(pages, 64, 1, 1)
    if qrope.shape[-1] == 0:
        packed = torch.empty((pages, 64, 1, 584), dtype=torch.uint8, device=kv.device)
        page = packed.view(pages, -1)
        payload = page[:, : 64 * 576].view(pages, 64, 1, 576)
        exponents = page[:, 64 * 576 :].view(pages, 64, 1, 8)
        for first in range(0, pages, 256):
            last = min(first + 256, pages)
            values = raw[first:last]
            scales = token_scale[first:last]
            payload[first:last, ..., :448].copy_(values[..., :448].view(torch.uint8))
            payload[first:last, ..., 448:].copy_(
                (values[..., 448:].float() * scales)
                .to(torch.bfloat16)
                .view(torch.uint8)
            )
            exponent = ((scales.view(torch.int32) >> 23) & 255).to(torch.uint8)
            exponents[first:last, ..., :7].copy_(
                exponent.expand(last - first, 64, 1, 7)
            )
            exponents[first:last, ..., 7:].zero_()
    else:
        rope = krope.view(pages, 64, 1, 64)
        packed = torch.empty((pages, 64, 1, 656), dtype=torch.uint8, device=kv.device)
        for first in range(0, pages, 256):
            last = min(first + 256, pages)
            scales = token_scale[first:last]
            packed[first:last, ..., :512].copy_(raw[first:last].view(torch.uint8))
            packed[first:last, ..., 512:528].copy_(
                scales.expand(last - first, 64, 1, 4).contiguous().view(torch.uint8)
            )
            packed[first:last, ..., 528:].copy_(
                (rope[first:last].float() * scales).to(torch.bfloat16).view(torch.uint8)
            )
    scale = (512 + qrope.shape[-1]) ** -0.5
    # muDNN FillOp fails for some real batch sizes. Prepare the equivalent
    # explicit lengths by host-to-device copy before either path is timed.
    if topk_length is None:
        topk_length = torch.full(
            (batch,), indices.shape[-1], dtype=torch.int32, device="cpu"
        ).to(query.device)
    legacy_mate = Version(version("mate")) < Version("0.2.7")
    if legacy_mate and qrope.shape[-1] == 0:
        extra_topk_length = torch.zeros((batch,), dtype=torch.int32, device="cpu").to(
            query.device
        )
        extra_cache, extra_indices = packed, indices[..., :0]
    else:
        extra_topk_length = None
        extra_cache, extra_indices = None, None
    plans = []
    # MATE 0.2.4 needs bounded native calls for large batches.
    batch_step = 8192 if legacy_mate else max(1, batch)
    for first in range(0, batch, batch_step):
        last = min(first + batch_step, batch)
        metadata, splits = get_mla_metadata(
            None,
            queries * heads,
            1,
            num_heads_q=heads,
            is_fp8_kvcache=True,
            topk=indices.shape[-1],
            q=query[first:last],
            bs=last - first,
            topk_length=topk_length[first:last],
            extra_topk_length=(
                None if extra_topk_length is None else extra_topk_length[first:last]
            ),
        )
        plans.append((first, last, metadata, splits))
    if len(plans) == 1:
        output, lse = None, None
    else:
        output = torch.empty(
            (batch, queries, heads, 512), dtype=query.dtype, device=query.device
        )
        lse = torch.empty(
            (batch, heads, queries), dtype=torch.float32, device=query.device
        )

    def run():
        for first, last, metadata, splits in plans:
            values, logsum = flash_mla_with_kvcache(
                query[first:last],
                packed,
                None,
                None,
                512,
                metadata,
                splits,
                softmax_scale=scale,
                attn_sink=attn_sink,
                is_fp8_kvcache=True,
                indices=indices[first:last],
                topk_length=topk_length[first:last],
                extra_topk_length=(
                    None if extra_topk_length is None else extra_topk_length[first:last]
                ),
                extra_k_cache=extra_cache,
                extra_indices_in_kvcache=(
                    None if extra_indices is None else extra_indices[first:last]
                ),
            )
            if len(plans) == 1:
                return values, logsum
            output[first:last].copy_(values)
            lse[first:last].copy_(logsum)
        return output, lse

    return run


class FlashmlaSparseBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_w8a8",
            (
                prepare_mate_baseline
                if MTHREADS_FP8_MLA
                else flashmla_reference().flash_mla_sparse_fwd
            ),
            [torch.bfloat16],
        )
        self.set_gems(
            prepare_flash_mla_sparse_fwd_w8a8
            if MTHREADS_FP8_MLA
            else flaggems_vllm.flash_mla_sparse_fwd_w8a8
        )

    def get_input_iter(self, dtype):
        if MTHREADS_FP8_MLA:
            yield from self._get_mthreads_inputs(dtype)
            return
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "Sparse MLA shapes need SQ,HQ,DQK,SKV,HKV,TOPK,DV, "
                    "optionally followed by sink,topk_length,seed"
                )
            inputs = make_sparse_inputs(shape, dtype=dtype)
            packed = quantize_sparse_fp8_fixture(*inputs)
            reference, _, reference_lse = self.torch_op(*inputs)
            output, lse = self.gems_op(*packed)
            assert_sparse_fixture_close(
                output[:, 0], lse[:, :, 0], reference, reference_lse
            )
            yield inputs, packed, shape

    def set_shapes(self, shape_file_path):
        if MTHREADS_FP8_MLA and Path(shape_file_path).resolve() == Path(
            self.DEFAULT_SHAPE_FILES
        ):
            shape_file_path = (
                Path(__file__).resolve().parents[1]
                / "src/flaggems_vllm/runtime/backend/_mthreads/core_shapes.yaml"
            )
        super().set_shapes(shape_file_path)

    def _get_mthreads_inputs(self, dtype):
        for shape in self.shapes:
            if len(shape) not in (7, 10):
                raise ValueError(
                    "MThreads sparse FP8 MLA requires 7 or 10 shape fields"
                )
            batch, heads, dims, tokens, h_kv, topk, value_dim = shape[:7]
            if h_kv != 1 or value_dim != 512:
                raise ValueError(
                    "MThreads sparse FP8 MLA requires one KV head and DV=512"
                )
            if dims not in (512, 576):
                raise ValueError("MThreads sparse FP8 MLA requires DQK=512 or 576")
            seed = shape[9] if len(shape) == 10 else 42
            packed, query = make_mthreads_sparse_inputs(
                batch, heads, topk, tokens, dtype, dims - 512, seed
            )
            sink, length = shape[7:9] if len(shape) == 10 else (False, False)
            attn_sink = (
                torch.randn(heads, device=query.device, dtype=torch.float32)
                if sink
                else None
            )
            topk_length = (
                torch.randint(
                    0, topk + 1, (batch,), device=query.device, dtype=torch.int32
                )
                if length
                else None
            )
            kwargs = dict(attn_sink=attn_sink, topk_length=topk_length)
            expected = mthreads_sparse_reference(packed, **kwargs)
            gems, actual = self.gems_op(*packed, **kwargs)
            assert_mthreads_sparse_accuracy(actual, expected, output_atol=2e-3)
            native = self.torch_op(packed, query, **kwargs)
            assert_mthreads_sparse_accuracy(native(), expected, output_atol=2e-3)
            del expected
            yield native, gems, shape

    def unpack_to_args_kwargs(self, input_tuple):
        return [input_tuple], {}

    def record_shapes(self, input_tuple):
        return input_tuple[2]

    def get_latency(self, op, *args, **kwargs):
        original, packed, _ = args[0]
        selected = packed if op is self.gems_op else original
        if MTHREADS_FP8_MLA:
            return super().get_latency(selected)
        return super().get_latency(op, *selected)


@pytest.mark.flash_mla_sparse_fwd_w8a8
@pytest.mark.skipif(not HAS_FP8_MLA, reason="Sparse FP8 MLA backend is unavailable")
def test_flash_mla_sparse_fwd_w8a8(request):
    if request.config.getoption("--parallel") > 0:
        pytest.skip("Use test_blas_perf_parallel.py for --parallel")
    FlashmlaSparseBenchmark().run()
