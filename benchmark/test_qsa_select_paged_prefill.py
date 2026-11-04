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

"""QSA prefill through the public entry point, with a Torch reference.

The ten inference layouts are defined below. NVIDIA uses
FP8; other backends start with BF16 versions of the same layouts. Native vLLM
full/score/TopK comparisons are separate NVIDIA SM90 tests. Torch-reference
ratios include host metadata reads and are not native-vLLM speedup results.
The isolated native score comparison uses a single full-row launch.
"""

from functools import partial
from importlib import import_module

import pytest
import torch

import flaggems_vllm
from tests.test_qsa_select_paged_prefill import (
    QsaPrefillCase,
    make_inputs,
    torch_select,
    vllm_reference,
)

from . import base

# Eight inference envelopes and two derived cases. Per-request lengths,
# contiguous strides, visibility values and shuffled page mappings are
# constructed assumptions. The first eight retain the observed cache/table
# capacities; compact and long100k use tightly sized caches.
# name, query lengths, maximum sequence length, physical pages, table columns
SHAPES = [
    ("first_chunk", [16384], 16384, 2712, 88),
    ("budget_exact", [16384], 32768, 2712, 88),
    ("dominant_long", [16331], 33842, 2712, 88),
    ("tail112", [16368], 32853, 2712, 88),
    ("tail113", [16369], 32851, 2712, 88),
    ("two_balanced", [10371, 5958], 32932, 2712, 88),
    ("two_skewed", [16330, 54], 32822, 2712, 88),
    ("short_tail", [2316], 33843, 2712, 88),
    ("compact", [3200] + [112] * 38 + [96] * 15, 4100, None, None),
    ("long100k", [16384], 118784, None, None),
]


def _benchmark_cases():
    result = []
    for name, queries, maximum, pages, columns in SHAPES:
        case = dict(
            name=f"fp8_{name}_float8_e4m3fn",
            query_lengths=queries,
            prefix_tokens=[maximum - q for q in queries],
            batch_max_seq_len=maximum,
            query_base=0,
            page_size=196,
            heads=4,
            dim=128,
            ratio=4,
            k=512,
            dtype="float8_e4m3fn",
        )
        if pages is not None:
            case.update(
                physical_pages=pages, table_columns=columns, cache_page_stride_factor=1
            )
        result.append(case)
    return result


def _benchmark_geometry(case):
    queries, prefixes = case["query_lengths"], case["prefix_tokens"]
    if len(queries) != len(prefixes) or not queries:
        raise ValueError("Each prefill request needs one query length and prefix")
    if any(q <= 0 for q in queries) or any(p < 0 for p in prefixes):
        raise ValueError("Query lengths must be positive and prefixes nonnegative")
    ratio, page = case["ratio"], case["page_size"]
    sequences = [q + p for q, p in zip(queries, prefixes)]
    compressed = [(s + ratio - 1) // ratio for s in sequences]
    pages = [(c + page - 1) // page for c in compressed]
    columns = case.get("table_columns", max(pages))
    physical = case.get("physical_pages", sum(pages))
    maximum = case.get("batch_max_seq_len", max(sequences))
    if columns < max(pages) or physical < sum(pages) or maximum < max(sequences):
        raise ValueError("Cache/table/context capacity cannot truncate prefill inputs")
    if case.get("cache_page_stride_factor", 1) < 1:
        raise ValueError("Cache page stride factor must be positive")
    return dict(
        query_lengths=queries,
        compressed_lengths=compressed,
        physical_pages=physical,
        page_table_shape=[len(queries), columns],
        batch_max_seq_len=maximum,
        logits_width=min(
            columns * page, ((maximum + ratio - 1) // ratio + 63) // 64 * 64
        ),
    )


def _benchmark_paged_inputs(case, *, device=None):
    device = flaggems_vllm.device if device is None else device
    info = _benchmark_geometry(case)
    visible = [
        (prefix + row + 1) // case["ratio"]
        for queries, prefix in zip(case["query_lengths"], case["prefix_tokens"])
        for row in range(queries)
    ]
    args = list(
        make_inputs(
            info["query_lengths"],
            info["compressed_lengths"],
            k=case["k"],
            ratio=case["ratio"],
            page_size=case["page_size"],
            query_base=case["query_base"],
            visible=visible,
            device=device,
            dtype=getattr(torch, case["dtype"]),
            num_heads=case["heads"],
            head_dim=case["dim"],
        )
    )
    cache, table = args[1:3]
    if "physical_pages" in case:
        # Model two adjacent layer pages in each physical block. Unused pages
        # are zeroed to keep unused storage deterministic.
        factor = case["cache_page_stride_factor"]
        storage = torch.zeros(
            (info["physical_pages"], factor, case["page_size"], 1, case["dim"]),
            dtype=cache.dtype,
            device=device,
        )
        expanded = storage[:, factor - 1]
        ids = torch.randperm(info["physical_pages"], device=device)[: cache.shape[0]]
        expanded[ids] = cache
        live = table >= 0
        table[live] = ids[table[live].long()].to(table.dtype)
        args[1] = expanded
    if table.shape[1] != info["page_table_shape"][1]:
        expanded = torch.full(
            info["page_table_shape"], -1, dtype=table.dtype, device=device
        )
        expanded[:, : table.shape[1]].copy_(table)
        table = expanded
    args[2] = table
    base = case["query_base"]
    if base:
        # Preserve the view offsets created when the preceding decode rows are
        # sliced off. Starts keep their absolute values, not rebased values.
        for index in (0, 4, 8):
            value = args[index]
            storage = torch.zeros(
                (base + value.shape[0], *value.shape[1:]),
                dtype=value.dtype,
                device=device,
            )
            storage[base:].copy_(value)
            args[index] = storage[base:]
        for index in (2, 3):
            value = args[index]
            storage = torch.zeros(
                (1 + value.shape[0], *value.shape[1:]), dtype=value.dtype, device=device
            )
            storage[1:].copy_(value)
            args[index] = storage[1:]
    args[9] = info["batch_max_seq_len"]
    return tuple(args)


def make_benchmark_inputs(case, *, device=None):
    device = flaggems_vllm.device if device is None else device
    if "physical_pages" in case:
        # Remap live pages into the captured full cache and table capacities.
        return _benchmark_paged_inputs(case, device=device)
    info = _benchmark_geometry(case)
    visible = [
        (prefix + row + 1) // case["ratio"]
        for length, prefix in zip(case["query_lengths"], case["prefix_tokens"])
        for row in range(length)
    ]
    args = make_inputs(
        info["query_lengths"],
        info["compressed_lengths"],
        k=case["k"],
        ratio=case["ratio"],
        page_size=case["page_size"],
        query_base=0,
        visible=visible,
        device=device,
        dtype=getattr(torch, case["dtype"]),
        num_heads=case["heads"],
        head_dim=case["dim"],
    )
    return (*args[:9], info["batch_max_seq_len"])


BENCHMARK_CASES = {(case["name"],): case for case in _benchmark_cases()}


class QSAPrefillBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(BENCHMARK_CASES)
        self.shape_desc = "batch case (H=4, D=128, page=196, block_k=512)"

    def set_more_shapes(self):
        return []


def _inputs(shape, dtype, device, *, stage):
    case = dict(BENCHMARK_CASES[shape], dtype=str(dtype).removeprefix("torch."))
    inputs = QsaPrefillCase(
        operands=make_benchmark_inputs(case, device=device), device=device
    )
    if stage == "full":
        yield inputs.launch_args()
        return
    op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    width = _benchmark_geometry(case)["logits_width"]
    workspace = torch.empty((inputs.q.shape[0] * width,), device=device)
    score_args = (
        *inputs.score_args(),
        workspace,
        inputs.max_query_len,
        width,
        0,
        inputs.q.shape[0],
    )
    if stage == "score":
        yield score_args
        return
    logits = op_module._prefill_logits(*score_args)
    topk_workspace = torch.empty((1024 * 1024,), dtype=torch.uint8, device=device)
    yield logits, inputs.visible_blocks, inputs.token_topk, inputs.compress_ratio, inputs.block_indices, topk_workspace


def _gems_topk(logits, visible, token_topk, ratio, output, workspace, *, topk):
    topk(
        logits,
        visible,
        output,
        workspace,
        k=token_topk // ratio,
        max_seq_len=logits.shape[1],
    )


@pytest.mark.qsa_select_paged_prefill
def test_qsa_select_paged_prefill():
    QSAPrefillBenchmark(
        input_fn=partial(_inputs, stage="full"),
        op_name="qsa_select_paged_prefill_vs_torch",
        torch_op=torch_select,
        gems_op=flaggems_vllm.qsa_select_paged_prefill,
        dtypes=(
            [torch.float8_e4m3fn]
            if flaggems_vllm.vendor_name == "nvidia"
            else [torch.bfloat16]
        ),
    ).run()


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native vLLM QSA comparison requires NVIDIA SM90",
)
@pytest.mark.parametrize("stage", ["full", "score", "topk"])
def test_qsa_select_paged_prefill_native_vllm(stage):
    reference = vllm_reference()
    op_module = import_module("flaggems_vllm.ops.qsa_select_paged_prefill")
    baseline, production = {
        "full": (
            reference.qsa_select_paged_prefill,
            flaggems_vllm.qsa_select_paged_prefill,
        ),
        "score": (reference._prefill_logits, op_module._prefill_logits),
        "topk": (
            reference._topk,
            partial(_gems_topk, topk=op_module.persistent_topk),
        ),
    }[stage]
    QSAPrefillBenchmark(
        input_fn=partial(_inputs, stage=stage),
        op_name=f"qsa_select_paged_prefill_{stage}_vs_native_vllm",
        torch_op=baseline,
        gems_op=production,
        dtypes=[torch.float8_e4m3fn],
    ).run()
