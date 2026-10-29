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

"""Paged MLA: decode/prefill, sparse distributions and address safety."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from tests.sparse_flash_attention_utils import (
    CASES,
    GLM_CASES,
    GLM_PHASES,
    MODES,
    contract_cpu_reference,
    cpu_reference,
    load_vllm_sfa,
    make_contract_inputs,
    make_glm_inputs,
    make_inputs,
    vllm_reference,
)

_IS_ASCEND = flaggems_vllm.vendor_name == "ascend" and hasattr(torch, "npu")
pytestmark = [
    pytest.mark.sparse_flash_attention,
    pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU"),
]


def module():
    return importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.sparse_flash_attention"
    )


@pytest.mark.sparse_flash_attention
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
@torch.inference_mode()
def test_sparse_flash_attention_accuracy(case, mode):
    inputs = make_inputs(case, mode, flaggems_vllm.device)
    actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
    expected = vllm_reference(inputs)
    assert (
        actual.dtype == inputs["query"].dtype and actual.shape == inputs["query"].shape
    )
    torch.testing.assert_close(
        actual.to(expected.dtype), expected, atol=0.01, rtol=0.01, equal_nan=False
    )
    expected_cpu = cpu_reference(inputs, 128)
    torch.testing.assert_close(
        actual.float().cpu(),
        expected_cpu,
        atol=0.01,
        rtol=0.01,
        equal_nan=False,
    )
    torch.testing.assert_close(
        expected.float().cpu(),
        expected_cpu,
        atol=0.01,
        rtol=0.01,
        equal_nan=False,
    )


@pytest.mark.parametrize("batch", [1, 3, 8, 25])
@torch.inference_mode()
def test_sparse_flash_attention_mutated_inputs_and_workspace_reuse(batch):
    inputs = make_inputs((batch, 2560, 2560), "duplicates", flaggems_vllm.device)
    for step in range(3):
        if step:
            inputs["query"].add_(0.01)
            inputs["sparse_indices"].copy_(inputs["sparse_indices"].roll(step, 2))
            inputs["block_table"].copy_(inputs["block_table"].roll(1, 1))
        actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
        torch.testing.assert_close(
            actual.float().cpu(),
            cpu_reference(inputs, 128),
            atol=0.01,
            rtol=0.01,
            equal_nan=False,
        )


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize(
    "heads", [pytest.param(4, marks=pytest.mark.glm), 128], ids=["h4", "h128"]
)
@torch.inference_mode()
def test_sparse_flash_attention_large_physical_offset(batch, heads):
    inputs = make_inputs((batch, 2560, 2560), "duplicates", flaggems_vllm.device)
    inputs["query"] = inputs["query"][:, :heads].contiguous()
    inputs["query_rope"] = inputs["query_rope"][:, :heads].contiguous()
    # The last page starts at element 2**31; only its 128 rows are accessed.
    inputs["key"] = torch.empty(
        (32769, 128, 1, 512), dtype=torch.float16, device=flaggems_vllm.device
    )
    inputs["key_rope"] = torch.empty(
        (32769, 128, 1, 64), dtype=torch.float16, device=flaggems_vllm.device
    )
    inputs["value"] = inputs["key"]
    inputs["key"][-1].normal_(0, 0.5)
    inputs["key_rope"][-1].normal_(0, 0.5)
    inputs["block_table"].fill_(32768)
    actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
    torch.testing.assert_close(
        actual[:, :2].float().cpu(),
        cpu_reference(inputs),
        atol=0.01,
        rtol=0.01,
        equal_nan=False,
    )


@pytest.mark.parametrize("scale", [0.0, -0.25, 0.125])
@torch.inference_mode()
def test_sparse_flash_attention_scale(scale):
    inputs = make_inputs((3, 2560, 2560), "random", flaggems_vllm.device)
    inputs["scale_value"] = scale
    actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
    torch.testing.assert_close(
        actual.float().cpu(),
        cpu_reference(inputs, 128),
        atol=0.01,
        rtol=0.01,
        equal_nan=False,
    )


def test_sparse_flash_attention_empty():
    inputs = make_inputs((1, 2560, 2560), "random", flaggems_vllm.device)
    for name in ("query", "query_rope", "sparse_indices", "block_table"):
        inputs[name] = inputs[name][:0]
    with pytest.raises(ValueError, match="nonempty"):
        flaggems_vllm.sparse_flash_attention(**inputs)


@pytest.mark.parametrize(
    "kind",
    [
        "cpu",
        "bf16",
        "fp32",
        "heads",
        "independent_value",
        "pre_tokens",
        "no_lengths",
        "mode",
        "indices_dtype",
        "rope",
        "table",
        "scale",
        "grad",
    ],
)
def test_sparse_flash_attention_unsupported(kind):
    inputs = make_inputs((1, 2560, 2560), "random", flaggems_vllm.device)
    error = NotImplementedError
    if kind == "cpu":
        inputs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }
    elif kind in ("bf16", "fp32"):
        inputs["query"] = inputs["query"].to(
            torch.bfloat16 if kind == "bf16" else torch.float32
        )
        if kind == "bf16":
            error = ValueError
    elif kind == "heads":
        inputs["query"] = inputs["query"][:, :65]
    elif kind == "independent_value":
        inputs["value"] = torch.randn_like(inputs["key"])
    elif kind == "pre_tokens":
        inputs["pre_tokens"] = 10
    elif kind == "no_lengths":
        inputs["actual_seq_lengths_query"] = None
        error = ValueError
    elif kind == "mode":
        inputs["attention_mode"] = 0
    elif kind == "indices_dtype":
        inputs["sparse_indices"] = inputs["sparse_indices"].long()
    elif kind == "rope":
        inputs["query_rope"] = inputs["query_rope"][:, :64]
        error = ValueError
    elif kind == "table":
        inputs["block_table"] = inputs["block_table"][:0]
        error = ValueError
    elif kind == "scale":
        inputs["scale_value"] = float("nan")
        error = ValueError
    elif kind == "grad":
        inputs["query"].requires_grad_()
    with pytest.raises(error):
        flaggems_vllm.sparse_flash_attention(**inputs)[0]


@pytest.mark.parametrize(
    "properties,expected",
    [
        ({"cube_core_num": 20, "vector_core_num": 40}, (20, 40)),
        ({"cube_core_num": 0, "vector_core_num": 0}, RuntimeError),
        ({}, RuntimeError),
        ({"cube_core_num": 24, "vector_core_num": 24}, NotImplementedError),
    ],
)
def test_sparse_flash_attention_core_query(monkeypatch, properties, expected):
    device = torch.device("npu:0")

    def device_properties(queried_device):
        assert queried_device == device
        return SimpleNamespace(**properties)

    monkeypatch.setattr(torch.npu, "get_device_properties", device_properties)
    if isinstance(expected, tuple):
        assert module()._get_core_num(device) == expected
    else:
        with pytest.raises(expected):
            module()._get_core_num(device)


def test_sparse_flash_attention_core_query_failure(monkeypatch):
    def unavailable(device):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(torch.npu, "get_device_properties", unavailable)
    with pytest.raises(RuntimeError, match="Cannot query physical NPU core counts"):
        module()._get_core_num(torch.device("npu:0"))


CONTRACT_CASES = (
    [
        dict(heads=h, return_softmax_stats=True, kv_layout="TND")
        for h in (1, 2, 4, 8, 16, 32, 64, 128)
    ]
    + [
        dict(
            query_layout="BSND",
            kv_layout="BSND",
            return_softmax_stats=True,
            mode=0,
            omitted_lengths=True,
            qlens=(3, 3),
            klens=(211, 211),
        ),
        dict(
            query_layout="BSND",
            kv_layout="BSND",
            return_softmax_stats=True,
            sparse_block=4,
        ),
        dict(
            query_layout="BSND",
            kv_layout="PA_BSND",
            page_size=16,
            return_softmax_stats=True,
            sparse_block=8,
        ),
        dict(qlens=(0, 2, 0, 3), klens=(128, 211, 0, 173), return_softmax_stats=True),
        dict(klens=(0, 173), return_softmax_stats=True),
        dict(all_padding=True, return_softmax_stats=True),
        dict(num_sparse_blocks=137, page_size=48, mode=0),
        dict(
            num_sparse_blocks=1024, heads=64, klens=(2560, 2111), qlens=(1, 1), mode=0
        ),
        dict(num_sparse_blocks=4096, heads=32, klens=(5001,), qlens=(1,), mode=0),
    ]
    + [
        dict(sparse_block=sb, page_size=256, return_softmax_stats=True)
        for sb in (1, 2, 4, 8, 16, 32, 64, 128)
    ]
    + [
        dict(
            heads=128,
            num_sparse_blocks=2048,
            klens=(2560,) * n,
            qlens=(1,) * n,
            dtype=torch.bfloat16,
            mode=0,
        )
        for n in (1, 4, 8, 16)
    ]
    + [
        dict(
            heads=128,
            num_sparse_blocks=2048,
            klens=(2560,) * n,
            qlens=(0, 2) * (n // 2),
            return_softmax_stats=True,
        )
        for n in (4, 8, 16)
    ]
    + [
        dict(dtype=torch.bfloat16, kv_layout="TND", return_softmax_stats=True),
        dict(
            query_layout="BSND",
            kv_layout="BSND",
            dtype=torch.bfloat16,
            return_softmax_stats=True,
        ),
    ]
    + [
        dict(all_padding=True, return_softmax_stats=True, sparse_block=sb)
        for sb in (2, 4)
    ]
    + [
        dict(
            heads=128,
            num_sparse_blocks=2048,
            klens=(2560,) * n,
            qlens=(1,) * n,
            all_padding=True,
            return_softmax_stats=True,
        )
        for n in (1, 4, 8, 16)
    ]
)


@pytest.mark.parametrize("config", CONTRACT_CASES)
@torch.inference_mode()
def test_sparse_flash_attention_vllm_contract(config):
    inputs = make_contract_inputs(flaggems_vllm.device, **config)
    _assert_sfa_contract(inputs)


def test_sparse_flash_attention_signature_matches_vllm():
    import inspect

    signature = inspect.signature(flaggems_vllm.sparse_flash_attention)
    schema = load_vllm_sfa()._schema
    assert list(signature.parameters) == [x.name for x in schema.arguments]
    for argument in schema.arguments:
        parameter = signature.parameters[argument.name]
        if argument.has_default_value():
            assert parameter.default == argument.default_value
        else:
            assert parameter.default is inspect.Parameter.empty
        assert (parameter.kind is inspect.Parameter.KEYWORD_ONLY) == argument.kwarg_only


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 25])
@torch.inference_mode()
def test_sparse_flash_attention_runtime_path_changes(batch):
    """Revalidate mutable inputs; exercise mixed paths and workspace reuse."""
    inputs = make_inputs((batch, 2560, 2560), "random", flaggems_vllm.device)
    inputs["sparse_block_size"] = 1
    saved = inputs["sparse_indices"].clone()
    for phase in range(4):
        inputs["sparse_indices"].copy_(saved)
        if phase in (1, 2):
            inputs["sparse_indices"][..., -17:] = -1
        if phase == 2:
            inputs["sparse_indices"][0].fill_(-1)
            if batch > 20:
                inputs["sparse_indices"][20].copy_(saved[20])
        actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
        expected = contract_cpu_reference(inputs)[0]
        reference = load_vllm_sfa()(**inputs)[0]
        torch.testing.assert_close(
            actual.float().cpu(), expected, atol=0.01, rtol=0.01, equal_nan=False
        )
        torch.testing.assert_close(
            actual, reference, atol=0.01, rtol=0.01, equal_nan=False
        )
    if batch == 4:
        inputs["actual_seq_lengths_query"].copy_(
            torch.tensor([0, 2, 2, 4], device=flaggems_vllm.device, dtype=torch.int32)
        )
        actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
        expected = contract_cpu_reference(inputs)[0]
        torch.testing.assert_close(
            actual.float().cpu(), expected, atol=0.01, rtol=0.01, equal_nan=False
        )
        torch.testing.assert_close(
            actual, load_vllm_sfa()(**inputs)[0], atol=0.01, rtol=0.01, equal_nan=False
        )


@pytest.mark.parametrize(
    "field",
    [
        "query",
        "key",
        "query_rope",
        "key_rope",
        "sparse_indices",
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
        "all",
    ],
)
@pytest.mark.parametrize("layout", ["slice", "transpose", "expand"])
@torch.inference_mode()
def test_sparse_flash_attention_strided(field, layout):
    inputs = make_inputs((4, 6400, 2560), "random", flaggems_vllm.device)
    fields = [
        "query",
        "key",
        "query_rope",
        "key_rope",
        "sparse_indices",
        "block_table",
        "actual_seq_lengths_query",
        "actual_seq_lengths_kv",
    ]
    for name in fields if field == "all" else [field]:
        x = inputs[name]
        if layout == "slice" or x.ndim == 1:
            shape = list(x.shape)
            shape[-1] = 2 * shape[-1] + 1
            storage = torch.zeros(shape, dtype=x.dtype, device=x.device)
            view = storage[..., 1::2]
            view.copy_(x)
        elif layout == "transpose":
            view = x.transpose(0, -1).contiguous().transpose(0, -1)
        else:
            # Broadcast the last dimension; preserve valid index/length ranges.
            view = x[..., :1].contiguous().expand_as(x)
        inputs[name] = view
    inputs["value"] = inputs["key"]
    reference_inputs = {
        k: v.contiguous() if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    reference_inputs["value"] = reference_inputs["key"]
    expected = load_vllm_sfa()(**reference_inputs)
    actual = flaggems_vllm.sparse_flash_attention(**inputs)
    torch.testing.assert_close(
        actual[0], expected[0], atol=0.01, rtol=0.01, equal_nan=False
    )


@torch.inference_mode()
def test_sparse_flash_attention_large_strided_copy():
    # The previous one-program-per-tile grid exceeded the device launch limit.
    storage = torch.empty(
        (65536, 2048), dtype=torch.float16, device=flaggems_vllm.device
    ).uniform_(-1, 1)
    view = storage[:, ::2]
    actual = module()._sfa_make_contiguous(view)
    torch.testing.assert_close(actual, view.contiguous(), atol=0, rtol=0)


@pytest.mark.parametrize("axis_pair", [(0, 0), (1, 1), (0, 1), (0, None)])
@torch.inference_mode()
def test_sparse_flash_attention_strided_native_rows(axis_pair):
    inputs = make_inputs((16, 2560, 2560), "random", flaggems_vllm.device)
    inputs["return_softmax_lse"] = True
    for name, axis in zip(("key", "key_rope"), axis_pair):
        if axis is None:
            continue
        x = inputs[name]
        shape = list(x.shape)
        shape[axis] = shape[axis] * 2 + 1
        storage = torch.empty(shape, dtype=x.dtype, device=x.device)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(1, None, 2)
        view = storage[tuple(slices)]
        view.copy_(x)
        inputs[name] = view
    inputs["value"] = inputs["key"]
    for phase in range(3):
        inputs["key"].add_(0.01)
        if phase == 1:
            inputs["sparse_indices"][..., -17:] = -1
            inputs["sparse_indices"][0].fill_(-1)
        if phase == 2:
            inputs["actual_seq_lengths_kv"].fill_(1024)
        _assert_sfa_contract(inputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_sparse_flash_attention_long_selection_updates(dtype):
    inputs = make_inputs((16, 48000, 48000), "random", flaggems_vllm.device)
    for name in ("query", "query_rope", "key", "key_rope"):
        inputs[name] = inputs[name].to(dtype)
    inputs["value"] = inputs["key"]
    inputs["return_softmax_lse"] = True
    saved = inputs["sparse_indices"].clone()
    for phase in range(4):
        inputs["query"].add_(0.01)
        if phase == 1:
            inputs["sparse_indices"][..., -17:] = -1
        elif phase == 2:
            inputs["sparse_indices"].copy_(saved.flip(-1))
        elif phase == 3:
            inputs["actual_seq_lengths_kv"].fill_(1024)
        _assert_sfa_contract(inputs)


@pytest.mark.glm
@pytest.mark.sparse_flash_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("phase", GLM_PHASES)
@pytest.mark.parametrize("case", GLM_CASES, ids=lambda case: case.name)
@torch.inference_mode()
def test_sparse_flash_attention_glm_accuracy(case, phase, dtype):
    inputs = make_glm_inputs(case, phase, flaggems_vllm.device, dtype)
    actual = flaggems_vllm.sparse_flash_attention(**inputs)[0]
    expected = vllm_reference(inputs)
    assert inputs["value"] is inputs["key"]
    assert actual.shape == inputs["query"].shape
    assert actual.dtype == dtype and actual.device == inputs["query"].device
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01, equal_nan=False)


def _assert_sfa_contract(inputs):
    actual = flaggems_vllm.sparse_flash_attention(**inputs)
    # Native paged statistics depend on which vendor implementation was loaded.
    # Compare its attention output and check our statistics against the
    # independent FP32 formula, including when paged native stats are unavailable.
    return_stats = inputs.get("return_softmax_lse", False)
    native_stats = return_stats and inputs["layout_kv"] != "PA_BSND"
    reference = load_vllm_sfa()(**dict(inputs, return_softmax_lse=native_stats))
    expected = contract_cpu_reference(
        dict(inputs, sparse_block_size=inputs.get("sparse_block_size", 1))
    )
    assert isinstance(actual, tuple) and len(actual) == 3
    assert actual[0].shape == inputs["query"].shape
    assert actual[0].dtype == inputs["query"].dtype
    torch.testing.assert_close(
        actual[0], reference[0], atol=0.01, rtol=0.01, equal_nan=False
    )
    torch.testing.assert_close(
        reference[0].float().cpu(), expected[0], atol=0.01, rtol=0.01, equal_nan=False
    )
    for index, (result, cpu) in enumerate(zip(actual, expected)):
        assert result.device == inputs["query"].device
        if index:
            assert result.dtype == torch.float32
            if not return_stats:
                assert result.shape == reference[index].shape == (0,)
                continue
        assert result.shape == cpu.shape
        torch.testing.assert_close(
            result.float().cpu(), cpu, atol=0.01, rtol=0.01, equal_nan=False
        )
        if index and native_stats:
            torch.testing.assert_close(
                reference[index].float().cpu(),
                cpu,
                atol=0.01,
                rtol=0.01,
                equal_nan=False,
            )


GLM_CONTRACT_CASES = [
    dict(heads=1, page_size=48, num_sparse_blocks=256, qlens=(1,), klens=(701,)),
    dict(heads=2, page_size=48, num_sparse_blocks=257, qlens=(2,), klens=(701,)),
    dict(heads=4, page_size=128, num_sparse_blocks=2048, qlens=(3,), klens=(4099,)),
    dict(
        heads=4,
        page_size=128,
        num_sparse_blocks=2048,
        qlens=(0, 2, 0, 3),
        klens=(4099, 4113, 0, 4201),
    ),
    dict(
        heads=4,
        page_size=128,
        num_sparse_blocks=2048,
        qlens=(7, 9),
        klens=(4099, 4107),
    ),
    dict(
        heads=4,
        page_size=128,
        num_sparse_blocks=2048,
        query_layout="BSND",
        qlens=(2, 3),
        klens=(4099, 4113),
    ),
    dict(heads=8, page_size=128, num_sparse_blocks=2048, qlens=(1,), klens=(4099,)),
    dict(heads=16, page_size=128, num_sparse_blocks=2048, qlens=(3,), klens=(4099,)),
    dict(heads=32, page_size=64, num_sparse_blocks=2051, qlens=(4,), klens=(4113,)),
    dict(heads=64, page_size=128, num_sparse_blocks=2048, qlens=(5,), klens=(4101,)),
]


@pytest.mark.glm
@pytest.mark.sparse_flash_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "config",
    GLM_CONTRACT_CASES,
    ids=lambda config: (
        f"h{config['heads']}-t{sum(config['qlens'])}-"
        f"{config.get('query_layout', 'TND').lower()}"
    ),
)
@torch.inference_mode()
def test_sparse_flash_attention_glm_causal_padding_and_stats(config, dtype):
    inputs = make_contract_inputs(
        flaggems_vllm.device, **config, dtype=dtype, mode=3, return_softmax_stats=True
    )
    inputs["scale_value"] = 1.0 / math.sqrt(576)
    _assert_sfa_contract(inputs)
    inputs["sparse_indices"][..., -17:] = -1
    inputs["sparse_indices"][-1].fill_(-1)
    _assert_sfa_contract(inputs)


@pytest.mark.glm
@pytest.mark.sparse_flash_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("qlens", [(8, 9), (22, 23)], ids=["t17", "t45"])
@torch.inference_mode()
def test_sparse_flash_attention_glm_large_query_updates(dtype, qlens):
    # T17 crosses the small-query boundary; T45 gives each of the 20 Cube
    # cores two or three queries, exercising pipeline state reuse and tails.
    # The requests have different page tables and KV lengths.
    num_tokens = sum(qlens)
    inputs = make_contract_inputs(
        flaggems_vllm.device,
        dtype=dtype,
        heads=64,
        page_size=128,
        num_sparse_blocks=2048,
        qlens=qlens,
        klens=(2560, 2573),
        mode=0,
        return_softmax_stats=True,
    )
    inputs["scale_value"] = 1.0 / math.sqrt(576)
    saved_indices = inputs["sparse_indices"].clone()
    _assert_sfa_contract(inputs)

    # Reuse every allocation while changing full selections to causal,
    # partially padded and entirely padded selections.
    inputs["sparse_mode"] = 3
    inputs["sparse_indices"][..., -17:] = -1
    inputs["sparse_indices"][0].fill_(-1)
    _assert_sfa_contract(inputs)

    inputs["query"].add_(0.015625)
    inputs["query_rope"].sub_(0.03125)
    inputs["key"].add_(0.0625)
    inputs["key_rope"].sub_(0.015625)
    inputs["block_table"].copy_(inputs["block_table"].roll(1, 1))
    inputs["actual_seq_lengths_query"].copy_(
        torch.tensor([3, num_tokens], dtype=torch.int32, device=flaggems_vllm.device)
    )
    inputs["actual_seq_lengths_kv"].copy_(
        torch.tensor([1024, 0], dtype=torch.int32, device=flaggems_vllm.device)
    )
    _assert_sfa_contract(inputs)

    # Return to full selections after masked execution, catching stale
    # prepared indices, query metadata, and unreinitialized pipeline state.
    inputs["sparse_mode"] = 0
    inputs["sparse_indices"].copy_(saved_indices)
    inputs["actual_seq_lengths_query"].copy_(
        torch.tensor(
            [qlens[0], num_tokens], dtype=torch.int32, device=flaggems_vllm.device
        )
    )
    inputs["actual_seq_lengths_kv"].copy_(
        torch.tensor([2560, 2573], dtype=torch.int32, device=flaggems_vllm.device)
    )
    _assert_sfa_contract(inputs)


@pytest.mark.glm
@pytest.mark.sparse_flash_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch.inference_mode()
def test_sparse_flash_attention_glm_selection_tail(dtype):
    inputs = make_contract_inputs(
        flaggems_vllm.device,
        dtype=dtype,
        heads=32,
        page_size=64,
        num_sparse_blocks=2051,
        qlens=(1,),
        klens=(4160,),
        mode=0,
        return_softmax_stats=True,
    )
    # All entries are valid and distinct. Make the irregular tail dominate
    # attention so truncating the last three entries cannot pass tolerance.
    inputs["sparse_indices"][..., :2048] = torch.arange(
        2048, dtype=torch.int32, device=flaggems_vllm.device
    )
    tail = torch.tensor(
        [2050, 3000, 4159], dtype=torch.int64, device=flaggems_vllm.device
    )
    inputs["sparse_indices"][..., -3:] = tail.int()
    inputs["query"].fill_(1)
    inputs["query_rope"].zero_()
    inputs["key"].zero_()
    inputs["key_rope"].zero_()
    physical = inputs["block_table"][0, tail // 64].long() * 64 + tail % 64
    inputs["key"].view(-1, 512)[physical] = 0.5
    inputs["scale_value"] = 1.0 / math.sqrt(576)
    _assert_sfa_contract(inputs)
