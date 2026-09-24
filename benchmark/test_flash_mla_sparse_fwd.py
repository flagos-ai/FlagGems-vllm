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

import dataclasses
import math
import random
from typing import List

import pytest
import torch
import triton

import flaggems_vllm

from . import base


def _get_vllm_flashmla_sparse_reference():
    """Return vLLM's sparse FlashMLA op only when its extension is usable."""
    try:
        from vllm.v1.attention.ops import flashmla
    except Exception as error:
        return None, f"vLLM FlashMLA could not be imported: {error}"

    support_check = getattr(flashmla, "is_flashmla_sparse_supported", None)
    if not callable(support_check):
        return None, "vLLM does not expose is_flashmla_sparse_supported()"

    try:
        support = support_check()
    except Exception as error:
        return None, f"vLLM FlashMLA capability check failed: {error}"

    if isinstance(support, tuple):
        supported, reason = support
    else:
        supported, reason = bool(support), None
    if not supported:
        return None, reason or "vLLM sparse FlashMLA is not supported"

    reference = getattr(flashmla, "flash_mla_sparse_fwd", None)
    if not callable(reference):
        return None, "vLLM sparse FlashMLA reference is not callable"
    return reference, None


vllm_flash_mla_sparse_fwd, VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON = (
    _get_vllm_flashmla_sparse_reference()
)
HAS_VLLM_FLASHMLA_SPARSE = vllm_flash_mla_sparse_fwd is not None


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flaggems_vllm.device


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd", vllm_flash_mla_sparse_fwd, [torch.bfloat16]
        )
        self.set_gems(flaggems_vllm.flash_mla_sparse_fwd)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for param in FlashmlaSparseBenchmark.get_performance_test_params_flashmla():
            yield from FlashmlaSparseBenchmark.make_input_flashmla(param)

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def get_performance_test_params_flashmla():
        cases = (
            [
                TestParam(4096, s_kv, 2048, h_q=128, d_qk=576, have_attn_sink=True)
                for s_kv in [8192, 32768, 65536, 98304, 131072]
            ]
            + [
                TestParam(4096, s_kv, 512, h_q=64, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
            + [
                TestParam(4096, s_kv, 1024, h_q=128, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
        )
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int,
        perm_range: torch.Tensor,
        perm_size: int,
        paddings: List[int],
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size].
        `0 <= res[i, :] < perm_range[i]` holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains
        values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
        """
        if perm_range.numel() != batch_size:
            raise ValueError("perm_range must contain one value per batch row")
        if perm_size < 0:
            raise ValueError("perm_size must be non-negative")
        if not paddings:
            raise ValueError("paddings must not be empty")

        ranges = perm_range.to(device="cpu", dtype=torch.int64).reshape(-1)
        if torch.any(ranges < 0):
            raise ValueError("perm_range values must be non-negative")

        # An affine permutation (offset + step * i) % range is unique whenever
        # step and range are coprime.  It needs only O(perm_size) temporary
        # storage instead of the previous O(batch_size * max(perm_range))
        # random score matrix (about 2 GiB for the largest benchmark case).
        positions = torch.arange(perm_size, dtype=torch.int64)
        padding_values = torch.tensor(paddings, dtype=torch.int32)
        res = torch.empty((batch_size, perm_size), dtype=torch.int32)
        for row, range_value in enumerate(ranges.tolist()):
            valid_size = min(range_value, perm_size)
            if valid_size:
                offset = random.randrange(range_value)
                step = 1
                if range_value > 1:
                    step = random.randrange(1, range_value)
                    while math.gcd(step, range_value) != 1:
                        step += 1
                        if step == range_value:
                            step = 1
                values = (offset + step * positions[:valid_size]) % range_value
                res[row, :valid_size] = values.to(torch.int32)

            if valid_size < perm_size:
                if len(paddings) == 1:
                    res[row, valid_size:].fill_(paddings[0])
                else:
                    filler_indices = torch.randint(
                        0, len(paddings), (perm_size - valid_size,)
                    )
                    res[row, valid_size:] = padding_values[filler_indices]
        return res

    @staticmethod
    def make_input_flashmla(param: TestParam):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseBenchmark._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]

        indices = FlashmlaSparseBenchmark._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)

        yield (q, kv, indices, 0.5, param.d_v, attn_sink, topk_length)


def _legacy_hq4_sparse_prefill_baseline(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
    **hq4_options,
):
    """Measure the current padded-head path with HQ4 options removed."""
    del hq4_options
    return flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )


class HQ4SparsePrefillBenchmark(base.Benchmark):
    """H20 active set; speedup is legacy padded path / gated HQ4 path."""

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_hq4",
            _legacy_hq4_sparse_prefill_baseline,
            [torch.bfloat16],
            gems_op=flaggems_vllm.flash_mla_sparse_fwd,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [64, 128, 512, 1024, 2048, 4096]

    def get_input_iter(self, dtype):
        for s_q in self.shapes:
            torch.manual_seed(6015 + s_q)
            source_topk = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(
                s_q, 1
            )
            query_start = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
            seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
            gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
            indices, lengths, pairs, quads = flaggems_vllm.combine_topk_swa_indices(
                source_topk,
                query_start,
                seq_lens,
                gather_lens,
                128,
                4,
                2048,
                34944,
                2048,
                enable_hq4_sparse_prefill=True,
                return_pair_metadata=True,
                return_quad_metadata=True,
            )
            # The baseline consumes all 64 physical heads. The optimized call
            # reads only the four logical heads and writes a strided view into
            # equally padded output storage.
            q = torch.randn((s_q, 64, 512), device="cuda", dtype=dtype) * 0.1
            kv = torch.randn((34944, 1, 512), device="cuda", dtype=dtype) * 0.1
            output_storage = torch.empty((s_q, 64, 512), device="cuda", dtype=dtype)
            yield (
                q,
                kv,
                indices[:, None],
                512**-0.5,
                512,
                None,
                lengths,
                {
                    "enable_hq4_sparse_prefill": True,
                    "logical_num_heads": 4,
                    "out": output_storage[:, :4],
                    "return_stats": False,
                    "pair_metadata": pairs,
                    "pair_window_size": 128,
                    "quad_metadata": quads,
                    "max_kv_length": 1280,
                },
            )


def _legacy_sparse_prefill_end_to_end(
    source_topk,
    query_start,
    seq_lens,
    gather_lens,
    q,
    kv,
    sm_scale,
    output_storage,
):
    del output_storage
    indices, lengths = flaggems_vllm.combine_topk_swa_indices(
        source_topk,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
    )
    return flaggems_vllm.flash_mla_sparse_fwd(
        q, kv, indices[:, None], sm_scale, 512, None, lengths
    )


def _gated_sparse_prefill_end_to_end(
    source_topk,
    query_start,
    seq_lens,
    gather_lens,
    q,
    kv,
    sm_scale,
    output_storage,
):
    indices, lengths, pairs, quads = flaggems_vllm.combine_topk_swa_indices(
        source_topk,
        query_start,
        seq_lens,
        gather_lens,
        128,
        4,
        2048,
        34944,
        2048,
        enable_hq4_sparse_prefill=True,
        return_pair_metadata=True,
        return_quad_metadata=True,
    )
    return flaggems_vllm.flash_mla_sparse_fwd(
        q,
        kv,
        indices[:, None],
        sm_scale,
        512,
        None,
        lengths,
        enable_hq4_sparse_prefill=True,
        logical_num_heads=4,
        out=output_storage[:, :4],
        return_stats=False,
        pair_metadata=pairs,
        pair_window_size=128,
        quad_metadata=quads,
        max_kv_length=1280,
    )


class HQ4SparsePrefillEndToEndCudaGraphBenchmark(base.Benchmark):
    """Capture producer and consumer together so metadata cost is included."""

    def __init__(self):
        super().__init__(
            "flash_mla_sparse_fwd_hq4_end_to_end_cudagraph",
            _legacy_sparse_prefill_end_to_end,
            [torch.bfloat16],
            gems_op=_gated_sparse_prefill_end_to_end,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [1024, 2048, 4096]

    def get_input_iter(self, dtype):
        for s_q in self.shapes:
            source_topk = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(
                s_q, 1
            )
            query_start = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
            seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
            gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
            q = torch.randn((s_q, 64, 512), device="cuda", dtype=dtype) * 0.1
            kv = torch.randn((34944, 1, 512), device="cuda", dtype=dtype) * 0.1
            output_storage = torch.empty((s_q, 64, 512), device="cuda", dtype=dtype)
            yield (
                source_topk,
                query_start,
                seq_lens,
                gather_lens,
                q,
                kv,
                512**-0.5,
                output_storage,
            )

    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        return triton.testing.do_bench_cudagraph(
            fn,
            rep=base.Config.repetition,
            return_mode="median",
        )


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(
    not HAS_VLLM_FLASHMLA_SPARSE,
    reason=(
        VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON or "vLLM sparse FlashMLA is unavailable"
    ),
)
def test_flash_mla_sparse_fwd():
    bench = FlashmlaSparseBenchmark()
    bench.run()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_fwd_hq4_benchmark():
    # Metric is median latency in ms; lower is better. Speedup is
    # legacy-padded latency / explicitly gated HQ4 latency.
    HQ4SparsePrefillBenchmark().run()


@pytest.mark.flash_mla_sparse_fwd
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_fwd_hq4_end_to_end_cudagraph_benchmark():
    # Includes combine metadata generation and attention in the same captured
    # graph. Speedup is legacy combine+HQ64 / gated metadata+HQ4.
    HQ4SparsePrefillEndToEndCudaGraphBenchmark().run()
