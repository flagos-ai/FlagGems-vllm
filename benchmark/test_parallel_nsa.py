import pytest
import torch
import triton

import flaggems_vllm
from benchmark.base import Benchmark


class ParallelNSABenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    # 形状: (B, T, H, HQ, D) — 参考 FLA benchmarks/ops/registry.py _nsa_default_shapes
    DEFAULT_SHAPES = [
        # Group 1: Small H regime (低并行度)
        (1, 16384, 4, 64, 64),  # H4_S16K, G=16
        # Group 2: Main NSA workload (长上下文, memory throughput)
        (1, 8192, 16, 256, 64),  # H16_S8K,  G=16
        (1, 16384, 16, 256, 64),  # H16_S16K, G=16
        (1, 65536, 16, 256, 64),  # H16_S64K, G=16
        # Group 3: Large H (接近真实模型)
        (1, 16384, 32, 512, 64),  # H32_S16K, G=16
        # Group 4: Head dimension 对比
        (1, 16384, 16, 256, 128),  # H16_D128, G=16
        # Group 6: 多序列训练场景
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
        S = 16
        n_blocks = triton.cdiv(T, block_size)

        q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)
        scale = D**-0.5

        # Random block indices for benchmarking
        block_indices = torch.randint(
            0, max(1, n_blocks), (B, T, H, S), device=device, dtype=torch.int32
        )

        # Return positional args + kwargs dict as last element
        return (
            q,
            k,
            v,
            {
                "block_indices": block_indices,
                "block_counts": S,
                "block_size": block_size,
                "scale": scale,
                "cu_seqlens": None,
            },
        )


@pytest.mark.parallel_nsa
def test_perf_parallel_nsa():
    bench = ParallelNSABenchmark(
        op_name="parallel_nsa",
        torch_op=flaggems_vllm.parallel_nsa,
    )
    bench.set_gems(flaggems_vllm.parallel_nsa)
    bench.run()
