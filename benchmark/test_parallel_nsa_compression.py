import pytest
import torch
import triton

import flaggems_vllm
from benchmark.base import Benchmark


class ParallelNSACompressionBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    # 形状: (B, T, H, HQ, D) — 参考 FLA benchmarks/ops/registry.py _nsa_default_shapes
    DEFAULT_SHAPES = [
        # Small H regime
        (1, 16384, 4, 64, 64),  # H4_S16K
        # Main NSA workload
        (1, 8192, 16, 256, 64),  # H16_S8K
        (1, 16384, 16, 256, 64),  # H16_S16K
        (1, 65536, 16, 256, 64),  # H16_S64K
        # Large H
        (1, 16384, 32, 512, 64),  # H32_S16K
        # Head dimension
        (1, 16384, 16, 256, 128),  # H16_D128
        # 多序列
        (4, 8192, 16, 256, 64),  # B4_H16_S8K
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, HQ, D"

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for B, T, H, HQ, D in self.shapes:
            yield self._build_inputs(B, T, H, HQ, D, cur_dtype)

    def _build_inputs(
        self, B: int, T: int, H: int, HQ: int, D: int, dtype: torch.dtype
    ):
        device = flaggems_vllm.device
        block_size = 64
        # 压缩后的 token 数
        TC = triton.cdiv(T, block_size)

        q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
        k = torch.randn(B, TC, H, D, device=device, dtype=dtype)
        v = torch.randn(B, TC, H, D, device=device, dtype=dtype)
        scale = D**-0.5

        return (
            q,
            k,
            v,
            block_size,
            scale,
            None,  # cu_seqlens
        )


@pytest.mark.parallel_nsa_compression
def test_perf_parallel_nsa_compression():
    bench = ParallelNSACompressionBenchmark(
        op_name="parallel_nsa_compression",
        torch_op=flaggems_vllm.parallel_nsa_compression,
    )
    bench.set_gems(flaggems_vllm.parallel_nsa_compression)
    bench.run()
