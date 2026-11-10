# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent reference and changed-input graph tests for MetaX."""
import pytest
import torch

import flaggems_vllm as ops

pytestmark = pytest.mark.skipif(
    ops.vendor_name != "metax", reason="MetaX specialization"
)


def check_replays(call, update, check, outputs=()):
    # No compilation, allocation growth or tuning is permitted during capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        update(value)
        for output in outputs:
            output.fill_(float("nan") if output.is_floating_point() else -999)
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "rows,hidden",
    [(1, 4096), (32, 4096), (64, 4096), (65, 4096), (128, 4096), (129, 4096), (3, 128)],
)
def test_public_mhc_pre_reference_and_new_input_replay(rows, hidden):
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre_ref

    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(rows, 4, hidden)
    result = []

    def call():
        result[:] = ops.mhc_pre(**data)

    def check():
        cpu = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        expected = mhc_pre_ref(**cpu)
        for actual, reference in zip(result, expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=1e-2, atol=1e-2)

    call()
    check()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for value in (0.5, -0.75):
        data["residual"].fill_(value)
        for output in result:
            output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


@pytest.mark.mhc_pre
def test_mhc_pre_empty_batch_preserves_shape():
    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(0, 4, 4096)
    post, comb, layer = ops.mhc_pre(**data)
    assert post.shape == (0, 4, 1) and comb.shape == (0, 4, 4)
    assert layer.shape == (0, 4096) and layer.dtype == torch.bfloat16


@pytest.mark.mhc_pre
def test_mhc_pre_mixed_device_rejected_before_mutation():
    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(1, 4, 4096)
    data["fn"] = data["fn"].cpu()
    before = {
        key: value.clone()
        for key, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    with pytest.raises(ValueError, match="share an accelerator"):
        ops.mhc_pre(**data)
    for key, value in before.items():
        assert torch.equal(data[key], value)


@pytest.mark.mhc_pre
def test_mhc_pre_nondefault_splits_preserves_common_warning():
    from flaggems_vllm.ops.mhc.mhc_pre import mhc_pre_ref

    from .test_mhc_ops import generate_mhc_pre_data

    data = generate_mhc_pre_data(1, 4, 4096)
    data["n_splits"] = 2
    with pytest.warns(RuntimeWarning, match="n_splits"):
        actual = ops.mhc_pre(**data)
    reference = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
        if key != "n_splits"
    }
    for result, expected in zip(actual, mhc_pre_ref(**reference)):
        torch.testing.assert_close(result.cpu(), expected, rtol=1e-2, atol=1e-2)


@pytest.fixture(autouse=True)
def restore_mhc_test_rng_state():
    # The upstream data helper deliberately seeds globally; contain that effect
    # within each test rather than leaking it into another operator's tests.
    cpu, gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
    yield
    torch.set_rng_state(cpu)
    torch.cuda.set_rng_state(gpu)
