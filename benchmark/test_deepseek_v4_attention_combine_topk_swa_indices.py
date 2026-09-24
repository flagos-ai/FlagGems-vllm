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

# vLLM >= 0.23 relocated this op to vllm.models.deepseek_v4.common.ops (the
# definition lives in its .cache_utils submodule); older releases exposed it as
# vllm.v1.attention.ops.deepseek_v4_ops.
try:
    from vllm.models.deepseek_v4.common.ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )

    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = True
except Exception:
    vllm_combine_topk_swa_indices = None
    _HAS_VLLM_COMBINE_TOPK_SWA_INDICES = False


# Bind through the top-level entry rather than the ops submodule: when a vendor
# ships a specialized implementation, the runtime rebinds it on the package at
# import time, so importing from flaggems_vllm.ops.<module> would bypass vendor
# dispatch (same reasoning as persistent_topk).
combine_topk_swa_indices = flaggems_vllm.combine_topk_swa_indices

# Device-agnostic entry point: "cuda" on NVIDIA / MetaX / Hygon / T-Head,
# "musa" on MThreads, "npu" on Ascend. Never hard-code "cuda" below.
device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()

# Keep in sync with _SPARSE_PREFILL_TOPK_ALIGNMENT in the operator module.
_TOPK_ALIGNMENT = 128


def _torch_combine_topk_swa_indices(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
):
    """Vectorized PyTorch baseline.

    The vLLM op is CUDA-only, so it is unavailable on every non-NVIDIA backend.
    This reference keeps the benchmark runnable there; it is built from plain
    PyTorch ops so it lowers on any vendor backend. Note that when this
    fallback is in use the reported speedup is against PyTorch, not vLLM.
    """
    dev = topk_indices.device
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _TOPK_ALIGNMENT - 1) // _TOPK_ALIGNMENT * _TOPK_ALIGNMENT
    )

    base_off = query_start_loc[0].to(torch.int64)
    starts = query_start_loc[:-1].to(torch.int64) - base_off
    ends = query_start_loc[1:].to(torch.int64) - base_off
    query_lens = ends - starts

    batch_id = torch.repeat_interleave(
        torch.arange(num_reqs, device=dev, dtype=torch.int64), query_lens
    )
    seq = seq_lens.to(torch.int64)[batch_id]
    gat = gather_lens.to(torch.int64)[batch_id]
    token_idx = torch.arange(num_tokens, device=dev, dtype=torch.int64)
    pos = (seq - query_lens[batch_id]) + (token_idx - starts[batch_id])
    gather_start = seq - gat

    topk_len = torch.clamp((pos + 1) // compress_ratio, max=topk)
    swa_len = torch.clamp(pos + 1, max=window_size)
    row_off = M * batch_id

    combined = torch.full(
        (num_tokens, combined_topk), -1, device=dev, dtype=torch.int32
    )
    cols_k = torch.arange(topk, device=dev, dtype=torch.int64).unsqueeze(0)
    combined[:, :topk] = torch.where(
        cols_k < topk_len.unsqueeze(1),
        (topk_indices.to(torch.int64) + row_off.unsqueeze(1)).to(torch.int32),
        combined[:, :topk],
    )
    cols = torch.arange(combined_topk, device=dev, dtype=torch.int64).unsqueeze(0)
    off = cols - topk_len.unsqueeze(1)
    swa_base = row_off + N + pos - swa_len + 1 - gather_start
    combined = torch.where(
        (off >= 0) & (off < swa_len.unsqueeze(1)),
        (swa_base.unsqueeze(1) + off).to(torch.int32),
        combined,
    )
    return combined, (topk_len + swa_len).to(torch.int32)


class CombineTopkSwaIndicesBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices",
            (
                vllm_combine_topk_swa_indices
                if _HAS_VLLM_COMBINE_TOPK_SWA_INDICES
                else _torch_combine_topk_swa_indices
            ),
            [torch.int32],
            gems_op=combine_topk_swa_indices,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [
            ([3, 2], [6, 4], [4, 3], 4, 4, 2, 20, 8),
            ([128], [512], [256], 32, 128, 4, 42240, 40960),
            ([512, 256], [2048, 1024], [1024, 512], 64, 256, 4, 45056, 40960),
            ([4096], [4096], [4096], 128, 256, 4, 45056, 40960),
            (
                [1024, 1024],
                [8192, 4096],
                [2048, 1024],
                128,
                256,
                4,
                45056,
                40960,
            ),
            ([128], [4096], [512], 32, 256, 128, 5632, 1280),
            ([4096], [4096], [4096], 128, 256, 128, 8448, 1280),
        ]

    def get_input_iter(self, dtype):
        _ = dtype
        for (
            query_lens,
            seq_lens_values,
            gather_lens_values,
            topk,
            window_size,
            compress_ratio,
            M,
            N,
        ) in self.shapes:
            num_tokens = sum(query_lens)
            topk_indices = torch.randint(
                -1,
                max(N, 1),
                (num_tokens, topk),
                device=device,
                dtype=torch.int32,
            )
            query_start_values = [0]
            for query_len in query_lens:
                query_start_values.append(query_start_values[-1] + query_len)
            query_start_loc = torch.tensor(
                query_start_values, device=device, dtype=torch.int32
            )
            seq_lens = torch.tensor(seq_lens_values, device=device, dtype=torch.int32)
            gather_lens = torch.tensor(
                gather_lens_values, device=device, dtype=torch.int32
            )
            yield (
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                window_size,
                compress_ratio,
                topk,
                M,
                N,
            )


def _legacy_hq4_metadata_baseline(
    topk_indices,
    query_start_loc,
    seq_lens,
    gather_lens,
    window_size,
    compress_ratio,
    topk,
    M,
    N,
    **metadata_options,
):
    """Measure the unchanged two-output producer without metadata work."""
    del metadata_options
    return combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )


class HQ4MetadataProducerBenchmark(base.Benchmark):
    """Exact C4/W128/K2048 producer active set used by the HQ4 consumer."""

    def __init__(self):
        super().__init__(
            "combine_topk_swa_indices_hq4_metadata",
            _legacy_hq4_metadata_baseline,
            [torch.int32],
            gems_op=combine_topk_swa_indices,
        )

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = [64, 128, 512, 1024, 2048, 4096]

    def get_input_iter(self, dtype):
        _ = dtype
        for s_q in self.shapes:
            topk_indices = torch.arange(2048, device="cuda", dtype=torch.int32).repeat(
                s_q, 1
            )
            query_start_loc = torch.tensor([0, s_q], device="cuda", dtype=torch.int32)
            seq_lens = torch.tensor([4096 + s_q], device="cuda", dtype=torch.int32)
            gather_lens = torch.tensor([s_q + 127], device="cuda", dtype=torch.int32)
            yield (
                topk_indices,
                query_start_loc,
                seq_lens,
                gather_lens,
                128,
                4,
                2048,
                34944,
                2048,
                {
                    "enable_hq4_sparse_prefill": True,
                    "return_pair_metadata": True,
                    "return_quad_metadata": True,
                },
            )


def _run_benchmark_with_baseline_label(bench, label):
    """Run ``bench`` with the baseline column renamed to ``label``.

    ``base.BenchmarkResult.__str__`` hard-codes the header "Torch Latency (ms)"
    for whatever was passed as the baseline op, which would mislabel the vLLM
    baseline. Same monkey-patch idiom as run_vllm_benchmark() in
    test_indexer_k_quant_and_cache.py; the replacement keeps the original
    18-character width so columns stay aligned.
    """
    original_str = base.BenchmarkResult.__str__

    def labelled_str(result):
        return original_str(result).replace("Torch Latency (ms)", label)

    base.BenchmarkResult.__str__ = labelled_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


# "Torch Latency (ms)" is 18 characters; keep the replacement the same width so
# the header, the rule line and the data rows stay aligned.
_VLLM_BASELINE_LABEL = "vLLM  Latency (ms)"


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not _HAS_DEVICE, reason=f"requires an available {device} device")
def test_combine_topk_swa_indices_benchmark():
    if _HAS_VLLM_COMBINE_TOPK_SWA_INDICES:
        print(
            "\n[baseline] vLLM's own combine_topk_swa_indices from "
            "vllm.models.deepseek_v4.common.ops; the speedup column is vs vLLM."
        )
        _run_benchmark_with_baseline_label(
            CombineTopkSwaIndicesBenchmark(), _VLLM_BASELINE_LABEL
        )
        return

    # No vLLM op: fall back to the PyTorch reference, and say so -- the table
    # then keeps its "Torch Latency (ms)" header, which is accurate.
    print(
        "\n[baseline] WARNING: vLLM's combine_topk_swa_indices is unavailable; "
        "the speedup column is vs the PyTorch reference, NOT vs vLLM."
    )
    CombineTopkSwaIndicesBenchmark().run()


@pytest.mark.combine_topk_swa_indices
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_combine_topk_swa_indices_hq4_metadata_benchmark():
    # Median latency in ms, lower is better. The speedup column compares the
    # metadata producer with the unchanged two-output producer.
    HQ4MetadataProducerBenchmark().run()
