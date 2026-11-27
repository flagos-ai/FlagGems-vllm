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

import importlib
import itertools

import pytest
import torch

import flaggems_vllm
from tests import conftest as test_config
from tests import lightning_indexer_hygon_utils as hygon
from tests.lightning_indexer_utils import (
    CASES,
    HAS_VLLM_ASCEND_REFERENCE,
    QUICK_CASE_IDS,
    assert_index_multisets_equal,
    get_reference,
    is_torch_fallback_feasible,
    make_inputs,
)

_requires_hygon = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "hygon", reason="requires Hygon"
)
_IS_ASCEND = (
    flaggems_vllm.vendor_name == "ascend"
    and hasattr(torch, "npu")
    and torch.npu.is_available()
)
_LIGHTNING_INDEXER = getattr(flaggems_vllm, "lightning_indexer", None)
try:
    _LIGHTNING_INDEXER_MODULE = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.ops.lightning_indexer"
    )
except ImportError:
    # triton.language.extra.cann only exists in Ascend's Triton build, so the
    # module cannot be imported (and so cannot be collected) elsewhere.
    _LIGHTNING_INDEXER_MODULE = None


@pytest.mark.lightning_indexer
@pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
@torch.inference_mode()
def test_lightning_indexer_matches_reference(case):
    if test_config.QUICK_MODE and case.case_id not in QUICK_CASE_IDS:
        pytest.skip("excluded by --quick")
    if not HAS_VLLM_ASCEND_REFERENCE and not is_torch_fallback_feasible(case):
        pytest.skip("long prefill Torch fallback is intentionally bounded")
    assert callable(_LIGHTNING_INDEXER)

    inputs = make_inputs(case, flaggems_vllm.device)
    reference, _ = get_reference()
    expected = reference(**inputs)
    actual = _LIGHTNING_INDEXER(**inputs)
    torch.npu.synchronize()
    assert_index_multisets_equal(actual, expected)


@pytest.mark.lightning_indexer
@pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")
@torch.inference_mode()
def test_lightning_indexer_rejects_unsupported_layout():
    assert callable(_LIGHTNING_INDEXER)
    inputs = make_inputs(CASES[0], flaggems_vllm.device)
    inputs["layout_query"] = "BSND"
    with pytest.raises(NotImplementedError, match="TND query"):
        _LIGHTNING_INDEXER(**inputs)


@pytest.mark.lightning_indexer
@pytest.mark.skipif(not _IS_ASCEND, reason="requires an Ascend NPU")
def test_lightning_indexer_requires_pr1065_custom_ops(monkeypatch):
    monkeypatch.setattr(_LIGHTNING_INDEXER_MODULE, "tle", None)
    monkeypatch.setattr(
        _LIGHTNING_INDEXER_MODULE,
        "_PR1065_IMPORT_ERROR",
        ImportError("missing FlagTree PR #1065"),
    )
    with pytest.raises(RuntimeError, match="FlagTree PR #1065 CustomOps"):
        _LIGHTNING_INDEXER_MODULE.lightning_indexer(None, None, None)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
@pytest.mark.parametrize(
    "query_lengths,key_lengths,heads,strided,table_width",
    [
        ((0,), (0,), 64, False, 1),
        ((1,), (130,), 64, False, 4),
        ((1,), (513,), 64, False, 8),
        ((1, 0), (130, 2050), 64, False, 17),
        ((1, 2), (130, 1025), 32, True, 9),
        ((3, 3), (1024, 2050), 16, False, 17),
    ],
)
@torch.inference_mode()
def test_lightning_indexer_device_paths(
    monkeypatch, version, query_lengths, key_lengths, heads, strided, table_width
):
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    values = hygon._paged_inputs(
        query_lengths,
        key_lengths,
        heads,
        flaggems_vllm.device,
        strided=strided,
        table_width=table_width,
    )
    actual = flaggems_vllm.lightning_indexer(**values)
    expected = hygon.pytorch_indexer(**values)
    hygon.assert_index_multisets_equal(actual, expected, inputs_dict=values)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1"])
@pytest.mark.parametrize(
    "q_lens,k_lens,heads,page,q_dtype,w_dtype,sparse_count,sparse_mode",
    [
        ((1, 2), (12, 300), 64, 128, torch.bfloat16, torch.bfloat16, 1, 3),
        ((2,), (400,), 16, 128, torch.bfloat16, torch.float32, 129, 3),
        ((1, 3), (500, 2600), 64, 64, torch.bfloat16, torch.float32, 2048, 3),
        ((2,), (2500,), 24, 48, torch.float16, torch.float16, 1000, 0),
        ((1, 2), (300, 4500), 7, 128, torch.float16, torch.float32, 3072, 3),
        ((1,), (5000,), 1, 256, torch.bfloat16, torch.float32, 8192, 0),
    ],
)
@torch.inference_mode()
def test_lightning_indexer_extended_pa_bsnd(
    monkeypatch,
    version,
    q_lens,
    k_lens,
    heads,
    page,
    q_dtype,
    w_dtype,
    sparse_count,
    sparse_mode,
):
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    values = hygon._paged_inputs(
        q_lens,
        k_lens,
        heads,
        flaggems_vllm.device,
        page=page,
        q_dtype=q_dtype,
        w_dtype=w_dtype,
    )
    kwargs = dict(
        **values,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        return_value=False,
    )
    actual = flaggems_vllm.lightning_indexer(**kwargs)
    expected = hygon.pytorch_indexer(**kwargs)
    hygon.assert_index_multisets_equal(actual, expected, inputs_dict=kwargs)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1"])
@pytest.mark.parametrize("return_value", [False, True])
@pytest.mark.parametrize(
    "q_lens,k_lens,heads,q_dtype,w_dtype,sparse_count,sparse_mode",
    [
        ((0,), (0,), 16, torch.bfloat16, torch.bfloat16, 64, 3),
        ((2,), (0,), 16, torch.float16, torch.float32, 64, 3),
        ((2, 3), (15, 320), 32, torch.bfloat16, torch.float32, 64, 3),
        ((2, 2), (600, 2400), 19, torch.float16, torch.float16, 1024, 0),
        ((1, 2), (200, 3500), 8, torch.bfloat16, torch.bfloat16, 4096, 3),
    ],
)
@torch.inference_mode()
def test_lightning_indexer_tnd_key_and_return_value(
    monkeypatch,
    version,
    return_value,
    q_lens,
    k_lens,
    heads,
    q_dtype,
    w_dtype,
    sparse_count,
    sparse_mode,
):
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    device = flaggems_vllm.device
    torch.manual_seed(512)
    t1 = sum(q_lens)
    t2 = sum(k_lens)
    q = torch.randn((t1, heads, 128), dtype=torch.float32).to(q_dtype).to(device)
    k = (
        torch.randn((max(1, t2), 1, 128), dtype=torch.float32)
        .to(q_dtype)
        .to(device)[:t2]
    )
    w = torch.randn((t1, heads), dtype=torch.float32).to(w_dtype).to(device)
    q_ends = torch.tensor(
        list(itertools.accumulate(q_lens)), dtype=torch.int32, device=device
    )
    k_ends = torch.tensor(
        list(itertools.accumulate(k_lens)), dtype=torch.int32, device=device
    )
    kwargs = dict(
        query=q,
        key=k,
        weights=w,
        actual_seq_lengths_query=q_ends,
        actual_seq_lengths_key=k_ends,
        layout_query="TND",
        layout_key="TND",
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        return_value=return_value,
    )
    actual = flaggems_vllm.lightning_indexer(**kwargs)
    expected = hygon.pytorch_indexer(**kwargs)
    hygon.assert_index_multisets_equal(actual, expected, inputs_dict=kwargs)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1"])
@pytest.mark.parametrize("name,case", hygon.CASES_30.items(), ids=hygon.CASES_30.keys())
@torch.inference_mode()
def test_lightning_indexer_30cases(monkeypatch, version, name, case):
    if test_config.QUICK_MODE and name not in {
        "R1",
        "R4",
        "R11",
        "R17",
        "R25",
        "R28",
        "R30",
    }:
        pytest.skip("excluded by --quick")
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    values = hygon.build_case_inputs(case, flaggems_vllm.device)
    actual = flaggems_vllm.lightning_indexer(**values)
    expected = hygon.pytorch_indexer(**values)
    hygon.assert_index_multisets_equal(actual, expected, inputs_dict=values)


@pytest.mark.lightning_indexer
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in {"hygon", "nvidia"}, reason="requires a GPU"
)
@pytest.mark.parametrize("name", ["R1", "R6", "R28", "R29"])
@torch.inference_mode()
def test_lightning_indexer_generic(name):
    from flaggems_vllm.ops.lightning_indexer import lightning_indexer as generic

    values = hygon.build_case_inputs(hygon.CASES_30[name], flaggems_vllm.device)
    expected = hygon.pytorch_indexer(**values)
    hygon.assert_index_multisets_equal(generic(**values), expected, inputs_dict=values)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
def test_lightning_indexer_rejects_unsupported_arguments(monkeypatch, version):
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    values = hygon._paged_inputs((1,), (16,), 64, flaggems_vllm.device)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, sparse_count=2049)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, sparse_count=9216)
    with pytest.raises(ValueError):
        flaggems_vllm.lightning_indexer(**values, sparse_count=0)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, layout_query="BSND")
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, layout_key="BSND")
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, sparse_mode=2)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.lightning_indexer(**values, return_value=True)
    tnd_inputs = dict(
        query=values["query"],
        key=values["key"][0],
        weights=values["weights"],
        actual_seq_lengths_query=values["actual_seq_lengths_query"],
        actual_seq_lengths_key=values["actual_seq_lengths_key"],
        block_table=values["block_table"],
        layout_query="TND",
        layout_key="TND",
    )
    with pytest.raises(ValueError):
        flaggems_vllm.lightning_indexer(**tnd_inputs)
    values["actual_seq_lengths_key"] = values["actual_seq_lengths_key"].long()
    with pytest.raises(TypeError):
        flaggems_vllm.lightning_indexer(**values)


@pytest.mark.lightning_indexer
@_requires_hygon
@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
@pytest.mark.parametrize(
    "query_ends,key_lengths",
    [
        ((2, 1), (16, 16)),
        ((1, 1), (16, 16)),
        ((1, 2), (-1, 16)),
        ((1, 2), (16, 129)),
    ],
)
def test_lightning_indexer_rejects_invalid_length_contents(
    monkeypatch, version, query_ends, key_lengths
):
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", version)
    values = hygon._paged_inputs(
        (1, 1), (16, 16), 64, flaggems_vllm.device, table_width=1
    )
    device = values["query"].device
    values["actual_seq_lengths_query"] = torch.tensor(
        query_ends, dtype=torch.int32, device=device
    )
    values["actual_seq_lengths_key"] = torch.tensor(
        key_lengths, dtype=torch.int32, device=device
    )
    with pytest.raises(ValueError):
        flaggems_vllm.lightning_indexer(**values)


@pytest.mark.lightning_indexer
@_requires_hygon
def test_lightning_indexer_v0_does_not_initialize_operator_extensions(monkeypatch):
    backend = importlib.import_module(
        "flaggems_vllm.runtime.backend._hygon.ops.lightning_indexer"
    )
    monkeypatch.setenv("GET_FLAGGEMS_VERSION", "v0")

    def forbidden():
        raise AssertionError("v0 initialized an operator extension")

    monkeypatch.setattr(backend, "_init_hcu_tle", forbidden)
    values = hygon._paged_inputs((1,), (513,), 64, flaggems_vllm.device)
    actual = flaggems_vllm.lightning_indexer(**values)
    expected = hygon.pytorch_indexer(**values)
    hygon.assert_index_multisets_equal(actual, expected, inputs_dict=values)
