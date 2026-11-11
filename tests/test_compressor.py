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

from .compressor_cases import (
    ALL_CASES,
    clone_inputs,
    compressor_torch,
    make_inputs,
    split_operator_inputs,
    valid_output_rows,
)
from .conftest import QUICK_MODE

try:
    import vllm_ascend.utils as vllm_ascend_utils

    HAS_VLLM_ASCEND = True
except ImportError:
    vllm_ascend_utils = None
    HAS_VLLM_ASCEND = False

# The case registry and comparison semantics are migrated from the source
# operator repository. Keep a small subset under --quick.
CASES = ALL_CASES[:4] if QUICK_MODE else ALL_CASES

# Triton kernel vs reference implementations: fp32 accumulation order differs,
# so compare with the same tolerance the source repository used against the
# official AscendC operator.
RTOL = 0.05
ATOL = 0.05


def _synchronize():
    if flaggems_vllm.vendor_name == "ascend":
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


@pytest.fixture(scope="module", autouse=True)
def _register_ascendc_baseline():
    if HAS_VLLM_ASCEND:
        vllm_ascend_utils.enable_custom_op()


def _assert_outputs_close(triton_output, ref_output, inputs):
    valid_rows = valid_output_rows(inputs)
    torch.testing.assert_close(
        triton_output.reshape(-1, triton_output.shape[-1])[:valid_rows].float(),
        ref_output.reshape(-1, ref_output.shape[-1])[:valid_rows].float(),
        rtol=RTOL,
        atol=ATOL,
    )
    padding = triton_output.reshape(-1, triton_output.shape[-1])[valid_rows:]
    if padding.numel():
        torch.testing.assert_close(padding, torch.zeros_like(padding), rtol=0, atol=0)


@pytest.mark.compressor
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend",
    reason="compressor is only implemented for the ascend vendor",
)
@pytest.mark.parametrize("case_name", CASES)
def test_compressor_matches_torch_golden(case_name):
    inputs = make_inputs(case_name, device=flaggems_vllm.device, seed=42)
    tensor_inputs, config = split_operator_inputs(clone_inputs(inputs))
    golden_inputs = {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in clone_inputs(inputs).items()
    }

    with torch.no_grad():
        triton_output = flaggems_vllm.compressor(**tensor_inputs, **config)
        golden_output = compressor_torch(**golden_inputs)
    _synchronize()

    _assert_outputs_close(triton_output.cpu(), golden_output, inputs)
    # The reference also updates state_cache in place with the same projections.
    torch.testing.assert_close(
        tensor_inputs["state_cache"].cpu(),
        golden_inputs["state_cache"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.compressor
@pytest.mark.skipif(
    not HAS_VLLM_ASCEND or flaggems_vllm.vendor_name != "ascend",
    reason="vllm-ascend custom operators are unavailable",
)
@pytest.mark.parametrize("case_name", CASES)
def test_compressor_matches_ascendc(case_name):
    if not vllm_ascend_utils.enable_custom_op():
        pytest.skip("vllm-ascend custom operators are unavailable")

    inputs = make_inputs(case_name, device=flaggems_vllm.device, seed=42)
    official_inputs = clone_inputs(inputs)
    tensor_inputs, config = split_operator_inputs(clone_inputs(inputs))

    with torch.no_grad():
        official_output = torch.ops._C_ascend.compressor(**official_inputs)
        triton_output = flaggems_vllm.compressor(**tensor_inputs, **config)
    _synchronize()

    _assert_outputs_close(triton_output.cpu(), official_output.cpu(), inputs)
    torch.testing.assert_close(
        tensor_inputs["state_cache"].float(),
        official_inputs["state_cache"].float(),
        rtol=RTOL,
        atol=ATOL,
    )


def _small_strided_inputs(dtype, layout, cmp_ratio=4, coff=2, rotary_mode=2):
    batch, length, hidden, head, block = (
        2 if layout == "batch_strided" else 1,
        5,
        512,
        128,
        8,
    )
    if cmp_ratio == 128:
        length, block = 129, 256
    generator = torch.Generator().manual_seed(127)
    rank3 = layout == "batch_strided"
    shape = (batch, length, hidden) if rank3 else (length, hidden)
    source = torch.randn(shape, generator=generator).to(dtype) * 0.1
    if rank3:
        storage = torch.empty(
            (length, batch, hidden * 2), device=flaggems_vllm.device, dtype=dtype
        )
        x = storage.permute(1, 0, 2)[..., ::2]
    elif layout == "column_strided":
        storage = torch.empty(
            (length, hidden * 2), device=flaggems_vllm.device, dtype=dtype
        )
        x = storage[:, ::2]
    else:
        x = torch.empty(shape, device=flaggems_vllm.device, dtype=dtype)
    x.copy_(source)
    per_request = (length + cmp_ratio - 1) // cmp_ratio
    rows = batch * per_request if rank3 else length // cmp_ratio
    rope_shape = (batch, per_request, 64) if rank3 else (rows, 64)
    width = coff * head
    table = torch.arange(1, batch + 1, dtype=torch.int32).reshape(batch, 1)
    inputs = dict(
        x=x,
        wkv=(torch.randn((width, hidden), generator=generator).to(dtype) * 0.1).to(
            x.device
        ),
        wgate=(torch.randn((width, hidden), generator=generator).to(dtype) * 0.1).to(
            x.device
        ),
        state_cache=(
            torch.randn((batch + 1, block, 2 * width), generator=generator) * 0.01
        ).to(x.device),
        ape=(torch.randn((cmp_ratio, width), generator=generator) * 0.01).to(x.device),
        norm_weight=torch.ones((head,), dtype=dtype, device=x.device),
        rope_sin=torch.full(rope_shape, 0.1, dtype=dtype, device=x.device),
        rope_cos=torch.full(rope_shape, 0.9, dtype=dtype, device=x.device),
        state_block_table=table.to(x.device),
        cu_seqlens=(
            None
            if rank3
            else torch.tensor([0, length], dtype=torch.int32, device=x.device)
        ),
        seqused=None,
        start_pos=torch.zeros((batch,), dtype=torch.int32, device=x.device),
        rope_head_dim=64,
        cmp_ratio=cmp_ratio,
        coff=coff,
        norm_eps=1e-6,
        rotary_mode=rotary_mode,
        cache_mode=1,
    )
    return inputs


def _cpu_compressor_inputs(inputs):
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend workspace")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "column_strided", "batch_strided"])
@pytest.mark.parametrize(
    "cmp_ratio,coff,rotary_mode", [(4, 2, 2), (4, 1, 1), (128, 1, 2)]
)
def test_compressor_prepared_reads_live_input(
    dtype, layout, cmp_ratio, coff, rotary_mode
):
    import importlib

    implementation = importlib.import_module(flaggems_vllm.compressor.__module__)
    inputs = _small_strided_inputs(dtype, layout, cmp_ratio, coff, rotary_mode)
    workspace = implementation.prepare_compressor_workspace(**inputs)
    initial_state = inputs["state_cache"].clone()
    source = inputs["x"].cpu()
    for scale in [0.5, -0.25]:
        inputs["x"].copy_(source * scale + 0.0625)
        inputs["state_cache"].copy_(initial_state)
        reference_inputs = _cpu_compressor_inputs(inputs)
        expected = compressor_torch(**reference_inputs)
        actual = implementation.compressor_prepared(workspace)
        _synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            inputs["state_cache"].cpu(),
            reference_inputs["state_cache"],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.compressor
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="Ascend workspace graph"
)
@pytest.mark.parametrize("layout", ["column_strided", "batch_strided"])
def test_compressor_prepared_strided_graph(layout):
    import importlib

    implementation = importlib.import_module(flaggems_vllm.compressor.__module__)
    inputs = _small_strided_inputs(torch.bfloat16, layout)
    workspace = implementation.prepare_compressor_workspace(**inputs)
    initial_state = inputs["state_cache"].clone()
    implementation.compressor_prepared(workspace)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = implementation.compressor_prepared(workspace)
    source = inputs["x"].cpu()
    for scale in [0.5, -0.25]:
        inputs["x"].copy_(source * scale + 0.0625)
        inputs["state_cache"].copy_(initial_state)
        reference_inputs = _cpu_compressor_inputs(inputs)
        expected = compressor_torch(**reference_inputs)
        actual.fill_(float("nan"))
        graph.replay()
        _synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            inputs["state_cache"].cpu(),
            reference_inputs["state_cache"],
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend cache bounds")
@pytest.mark.parametrize("page", [-1, 0, 3])
@pytest.mark.parametrize("tokens,start", [(1, 0), (1, 3), (5, 0)])
def test_compressor_invalid_page_preserves_backing_storage(page, tokens, start):
    inputs = _small_strided_inputs(torch.bfloat16, "contiguous")
    inputs["x"] = inputs["x"][:tokens]
    inputs["cu_seqlens"] = torch.tensor(
        [0, tokens], dtype=torch.int32, device=inputs["x"].device
    )
    inputs["start_pos"].fill_(start)
    # A negative physical page lands in this owned prefix, so a regression is
    # observable without intentionally accessing outside the allocation.
    backing = torch.full((5, 8, 512), 0.125, device=inputs["x"].device)
    inputs["state_cache"] = backing[1:4]
    inputs["state_block_table"].fill_(page)
    before = backing.cpu()
    reference_inputs = _cpu_compressor_inputs(inputs)
    expected = compressor_torch(**reference_inputs)
    actual = flaggems_vllm.compressor(**inputs)
    _synchronize()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(backing.cpu(), before, rtol=0, atol=0)
