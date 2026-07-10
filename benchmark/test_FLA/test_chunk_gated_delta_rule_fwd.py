import os
from contextlib import contextmanager

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

RECOMPUTE_TLE_ENV = "FLAGGEMS_CHUNK_GDR_RECOMPUTE_TLE"
FULL_TLE_ENV = "FLAGGEMS_CHUNK_GATED_DELTA_RULE_TLE"


@contextmanager
def _set_gdn_tle(*, full_tle: bool, recompute_tle: bool):
    old_full = os.environ.get(FULL_TLE_ENV)
    old_recompute = os.environ.get(RECOMPUTE_TLE_ENV)
    os.environ[FULL_TLE_ENV] = "1" if full_tle else "0"
    os.environ[RECOMPUTE_TLE_ENV] = "1" if recompute_tle else "0"
    try:
        yield
    finally:
        if old_full is None:
            os.environ.pop(FULL_TLE_ENV, None)
        else:
            os.environ[FULL_TLE_ENV] = old_full
        if old_recompute is None:
            os.environ.pop(RECOMPUTE_TLE_ENV, None)
        else:
            os.environ[RECOMPUTE_TLE_ENV] = old_recompute


def _gdn_fwd_with_tle(full_tle: bool, recompute_tle: bool, *args):
    with _set_gdn_tle(full_tle=full_tle, recompute_tle=recompute_tle):
        return flaggems_vllm.chunk_gated_delta_rule_fwd(*args)


def _native_gdn_fwd(*args):
    return _gdn_fwd_with_tle(False, False, *args)


def _optimized_gdn_fwd(*args):
    return _gdn_fwd_with_tle(True, True, *args)


class ChunkGatedDeltaRuleFwdBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (2, 16384, 16, 128, 128),
        (4, 2048, 16, 128, 128),
        (4, 4096, 64, 128, 128),
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, K, V"

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for B, T, H, K, V in self.shapes:
            yield self._build_inputs(B, T, H, K, V, cur_dtype)

    def _build_inputs(self, B: int, T: int, H: int, K: int, V: int, dtype: torch.dtype):
        device = flaggems_vllm.device

        q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
        scale = K**-0.5

        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            None,
            True,
            None,
        )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    not has_triton_tle(3, 6, 0),
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue",
)
def test_perf_chunk_gated_delta_rule_fwd():
    bench = ChunkGatedDeltaRuleFwdBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=_native_gdn_fwd,
    )
    bench.set_gems(_optimized_gdn_fwd)
    bench.run()
