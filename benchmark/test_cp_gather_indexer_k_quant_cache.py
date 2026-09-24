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

import math
import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

import flaggems_vllm

from . import base

# Bind the top-level entry: on vendor backends with a specialized
# implementation (runtime/backend/_<vendor>/ops), this is the vendor version
# (replaced at import time); elsewhere it is the generic one.
cp_gather_indexer_k_quant_cache = flaggems_vllm.cp_gather_indexer_k_quant_cache

# "cuda" on NVIDIA / MetaX / Hygon / T-Head, "musa" on MThreads, "npu" on Ascend.
device = flaggems_vllm.device
_device_module = getattr(torch, device, None)
_HAS_DEVICE = _device_module is not None and _device_module.is_available()

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")


def run_vllm_benchmark(bench):
    original_str = base.BenchmarkResult.__str__

    def vllm_str(result):
        return (
            original_str(result)
            .replace("Torch Latency (ms)", "vLLM CUDA Latency (ms)")
            .replace("Torch GBPS ", "vLLM CUDA GBPS ")
        )

    base.BenchmarkResult.__str__ = vllm_str
    try:
        bench.run()
    finally:
        base.BenchmarkResult.__str__ = original_str


def _default_fp8_dtype():
    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for cp_gather_indexer_k_quant_cache")


def load_vllm_cuda_op_and_fp8_dtype():
    """Return (vllm_op, fp8_dtype), or (None, fp8_dtype) when the vLLM CUDA
    custom op is unavailable (e.g. on non-NVIDIA vendor backends)."""
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if device != "cuda" or getattr(torch.version, "cuda", None) is None:
        return None, _default_fp8_dtype()
    try:
        import vllm
        import vllm._custom_ops as ops
        from vllm.platforms import current_platform
    except Exception:
        return None, _default_fp8_dtype()

    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            return None, _default_fp8_dtype()
    except InvalidVersion:
        pass

    if not hasattr(ops, "cp_gather_indexer_k_quant_cache"):
        return None, _default_fp8_dtype()

    def vllm_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k,
            dst_scale,
            block_table,
            cu_seq_lens,
        )

    return vllm_gather, current_platform.fp8_dtype()


def torch_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
    """Torch baseline for backends without the vLLM CUDA op.

    Assumes every row of dst_k is a valid token (true for the inputs built
    below), so no device-to-host sync is needed to size the gather.
    """
    num_blocks, block_size, _ = kv_cache.shape
    num_tokens, head_dim = dst_k.shape
    dst_k_bytes = dst_k.view(torch.uint8)
    dst_scale_bytes = dst_scale.view(torch.uint8)

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(
        num_blocks, block_size, head_dim
    )
    cache_scales = flat_cache[:, block_size * head_dim :].view(
        num_blocks, block_size, dst_scale_bytes.size(1)
    )

    cu_seq_lens = cu_seq_lens.long()
    seq_lens = cu_seq_lens[1:] - cu_seq_lens[:-1]
    batch_ids = torch.repeat_interleave(
        torch.arange(seq_lens.numel(), device=kv_cache.device),
        seq_lens,
        output_size=num_tokens,
    )
    token_offsets = (
        torch.arange(num_tokens, device=kv_cache.device) - cu_seq_lens[batch_ids]
    )
    block_ids = block_table[batch_ids, token_offsets // block_size].long()
    block_offsets = token_offsets % block_size

    dst_k_bytes.copy_(cache_values[block_ids, block_offsets])
    dst_scale_bytes.copy_(cache_scales[block_ids, block_offsets])


def fill_cache(k_cache, head_dim, quant_block_size):
    # The op is a byte-exact gather, so value bytes need not be valid fp8;
    # raw uint8 avoids an fp8 cast kernel some vendor devices lack.
    num_blocks, block_size, _ = k_cache.shape
    num_quant_blocks = head_dim // quant_block_size
    k_cache.copy_(
        torch.randint(0, 256, k_cache.shape, dtype=torch.uint8, device=k_cache.device)
    )
    flat_cache = k_cache.view(num_blocks, -1)
    scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    scales.copy_(
        torch.rand(
            num_blocks,
            block_size * num_quant_blocks,
            dtype=torch.float32,
            device=k_cache.device,
        )
        + 0.01
    )


def make_gather_metadata(batch_size, seq_len, block_size, device):
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_lens, dim=0)

    blocks_per_seq = math.ceil(seq_len / block_size)
    block_table = torch.arange(
        batch_size * blocks_per_seq,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_seq)

    return block_table, cu_seqlen


class CpGatherIndexerKQuantCacheBenchmark(base.Benchmark):
    def __init__(self, baseline_op, fp8_dtype):
        super().__init__(
            op_name="cp_gather_indexer_k_quant_cache",
            torch_op=baseline_op,
            dtypes=[torch.float16],
        )
        self.set_gems(cp_gather_indexer_k_quant_cache)
        self.fp8_dtype = fp8_dtype
        self.shape_desc = "batch_size, seq_len, block_size, head_dim, quant_block_size"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 256, 16, 128, 128),
            (8, 512, 16, 128, 128),
            (16, 1024, 16, 512, 128),
            (32, 1024, 16, 512, 128),
            # DeepSeek-V3.2-style indexer cache: head_dim 128, block_size 64.
            (1, 8192, 64, 128, 128),
            (8, 4096, 64, 128, 128),
            (64, 1024, 64, 128, 128),
        ]

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency=None):
        _, k_fp8, k_fp8_scale, block_table, cu_seqlen = args
        gathered = sum(t.numel() * t.element_size() for t in (k_fp8, k_fp8_scale))
        metadata = sum(t.numel() * t.element_size() for t in (block_table, cu_seqlen))
        # Gathered bytes are read from the cache once and written once.
        return (2 * gathered + metadata) * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, dtype):
        del dtype
        for (
            batch_size,
            seq_len,
            block_size,
            head_dim,
            quant_block_size,
        ) in self.shapes:
            block_table, cu_seqlen = make_gather_metadata(
                batch_size,
                seq_len,
                block_size,
                self.device,
            )
            num_blocks = block_table.numel()
            num_tokens = batch_size * seq_len
            cache_stride = head_dim + head_dim * 4 // quant_block_size
            k_cache = torch.empty(
                num_blocks,
                block_size,
                cache_stride,
                dtype=torch.uint8,
                device=self.device,
            )
            fill_cache(k_cache, head_dim, quant_block_size)
            k_fp8 = torch.empty(
                num_tokens,
                head_dim,
                dtype=self.fp8_dtype,
                device=self.device,
            )
            k_fp8_scale = torch.empty(
                num_tokens,
                head_dim * 4 // quant_block_size,
                dtype=torch.uint8,
                device=self.device,
            )
            yield k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen


@pytest.mark.skipif(not _HAS_DEVICE, reason=f"requires an available {device} device")
@pytest.mark.cp_gather_indexer_k_quant_cache
def test_cp_gather_indexer_k_quant_cache_benchmark():
    vllm_op, fp8_dtype = load_vllm_cuda_op_and_fp8_dtype()
    if vllm_op is not None:
        run_vllm_benchmark(CpGatherIndexerKQuantCacheBenchmark(vllm_op, fp8_dtype))
    else:
        # No vLLM CUDA op on this backend: compare against the torch baseline.
        CpGatherIndexerKQuantCacheBenchmark(torch_gather, fp8_dtype).run()
