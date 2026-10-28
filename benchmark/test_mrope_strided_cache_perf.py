# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from flaggems_vllm.ops.mrope_strided_cache import mrope_strided_cache

from . import base


def reference(q, k, cache, positions):
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

    values = cache[positions]
    a, b = triton_mrope(
        q.reshape(q.shape[0], -1),
        k.reshape(k.shape[0], -1),
        values[..., :32],
        values[..., 32:],
        [12, 10, 10],
        64,
        64,
        True,
        False,
    )
    return a.view(q.shape), b.view(k.shape)


class MRoPECacheBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(m, 4, 64) for m in (4, 64, 1024)]

    def get_input_iter(self, dtype):
        for rows, heads, width in self.shapes:
            yield (
                torch.randn(rows, heads, 96, dtype=dtype, device=self.device)[
                    ..., :width
                ],
                torch.randn(rows, 1, 96, dtype=dtype, device=self.device)[..., :width],
                torch.randn(2048, width, dtype=dtype, device=self.device) / 4,
                torch.stack(tuple(torch.arange(rows) + a for a in range(3))).to(
                    self.device
                ),
            )


@pytest.mark.mrope_strided_cache
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)
@torch.inference_mode()
def test_mrope_strided_cache_perf():
    pytest.importorskip("vllm.model_executor.layers.rotary_embedding.mrope")
    bench = MRoPECacheBenchmark(
        op_name="mrope_strided_cache",
        torch_op=reference,
        gems_op=mrope_strided_cache,
        dtypes=[torch.bfloat16],
    )
    bench.run()
