"""Fast smoke coverage for the standalone Qwen4 benchmark entry point."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from benchmark.qwen4_self_kernels import _run


@pytest.mark.qwen4_self_kernels
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen4_self_kernels_benchmark_smoke():
    results = _run(1, torch.device("cuda"), warmup=1, iters=1)
    assert len(results) == 8
    assert all(item["correctness"] for item in results)
