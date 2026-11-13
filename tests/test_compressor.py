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
from tests import conftest as _paged_cfg

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


@pytest.fixture(scope="module")
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
@pytest.mark.usefixtures("_register_ascendc_baseline")
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


def _paged_public_call(*args, **overrides):
    x, wkv, wgate, ape, state, pos, slots, table, requests, ends, weight, rope = args[
        :12
    ]
    ratio = args[12] if len(args) > 12 else 4
    settings = dict(
        state_block_table=table,
        cmp_ratio=ratio,
        coff=2 if ratio == 4 else 1,
        rope_head_dim=rope.shape[1],
        positions=pos,
        slot_mapping=slots,
        output_request_ids=requests,
        output_positions=ends,
        cos_sin_cache=rope,
    )
    settings.update(overrides)
    return flaggems_vllm.compressor(x, wkv, wgate, state, ape, weight, **settings)


def _paged_full_compressor_cpu_oracle(args):
    """CPU projection/state writes followed by independent history mathematics."""
    tensors = [x.detach().cpu() for x in args[:-1]]
    x, wkv, wgate, ape, state, pos, slots, table, req, ends, weight, rope = tensors
    ratio = args[-1]
    state = state.clone()
    kv = torch.nn.functional.linear(x.float(), wkv.float())
    gate = torch.nn.functional.linear(x.float(), wgate.float())
    block, d = state.shape[1], weight.numel()
    coff = 2 if ratio == 4 else 1
    for row, slot in enumerate(slots.tolist()):
        if 0 <= slot < state.shape[0] * block:
            state[slot // block, slot % block] = torch.cat(
                [kv[row], gate[row] + ape[int(pos[row]) % ratio].float()]
            )
    output = torch.zeros((ends.numel(), d), dtype=torch.float32)
    for row, (request, end) in enumerate(zip(req.tolist(), ends.tolist())):
        rope_pos = (end // ratio) * ratio
        if not (
            0 <= request < table.shape[0]
            and end >= 0
            and (end + 1) % ratio == 0
            and rope_pos < rope.shape[0]
        ):
            continue
        values, scores = [], []
        for index in range(coff * ratio):
            source = end - coff * ratio + 1 + index
            if not (source >= 0 and source // block < table.shape[1]):
                continue
            page = int(table[request, source // block])
            if not 0 <= page < state.shape[0]:
                continue
            half = index // ratio
            raw = state[page, source % block]
            values.append(raw[half * d : (half + 1) * d])
            scores.append(raw[(coff + half) * d : (coff + half + 1) * d])
        if not values:
            continue
        compressed = (torch.stack(values) * torch.softmax(torch.stack(scores), 0)).sum(
            0
        )
        normed = (
            compressed * torch.rsqrt(compressed.square().mean() + 1e-6) * weight.float()
        )
        rd = rope.shape[1]
        if rd:
            tail = normed[-rd:].clone()
            cos, sin = rope[rope_pos].float().chunk(2)
            normed[-rd::2] = tail[::2] * cos - tail[1::2] * sin
            normed[-rd + 1 :: 2] = tail[1::2] * cos + tail[::2] * sin
        output[row] = normed
    return output.to(x.dtype), state


def _paged_full_compressor_case(
    ratio=4, dim=512, dtype=torch.bfloat16, rope_dim=64, seed=931
):
    gen = torch.Generator().manual_seed(seed)
    block = 4 if ratio == 4 else 8
    nc = (2 * ratio + block - 1) // block + 1
    pages = torch.randperm(2 * nc, generator=gen)
    zero_index = int((pages == 0).nonzero()[0])
    pages[zero_index], pages[0] = pages[0].clone(), pages[zero_index].clone()
    table = pages.reshape(2, nc).long()
    logical = [(request, pos) for request in range(2) for pos in range(2 * ratio)]
    logical = [logical[i] for i in torch.randperm(len(logical), generator=gen).tolist()]
    slots = [
        int(table[request, pos // block]) * block + pos % block
        for request, pos in logical
    ]
    n, hidden = len(logical) + 2, 33
    coff = 2 if ratio == 4 else 1
    x = (torch.randn((n, hidden), generator=gen) * 0.03).to(dtype)
    x[-2:].fill_(torch.nan)
    wkv = (torch.randn((coff * dim, hidden), generator=gen) * 0.04).to(dtype)
    wgate = (torch.randn((coff * dim, hidden), generator=gen) * 0.04).to(dtype)
    ape = torch.randn((ratio, coff * dim), generator=gen) * 0.02
    state = torch.randn((2 * nc + 1, block, 2 * coff * dim), generator=gen) * 0.02
    state[-1].fill_(torch.nan)
    pos = torch.tensor([pos for _, pos in logical] + [-1, 2**63 - 1])
    slots = torch.tensor(slots + [-1, 2**40])
    req = torch.tensor([1, 0, -1, 0, 1, 0, 2**40])
    ends = torch.tensor(
        [
            ratio - 1,
            2 * ratio - 1,
            2 * ratio - 1,
            ratio - 2,
            2 * ratio - 1,
            ratio - 1,
            2**63 - 1,
        ]
    )
    weight = (torch.rand(dim, generator=gen) + 0.5).to(dtype)
    angle = (
        torch.arange(2 * ratio + 8).float()[:, None]
        * torch.linspace(0.01, 0.07, max(1, rope_dim // 2))[None, :]
    )
    rope = (
        torch.cat([angle.cos(), angle.sin()], -1)
        if rope_dim
        else torch.empty(2 * ratio + 8, 0)
    )
    return [
        x,
        wkv,
        wgate,
        ape,
        state,
        pos,
        slots,
        table,
        req,
        ends,
        weight,
        rope,
        ratio,
    ]


def _paged_full_to_device(args):
    return [x.npu() if isinstance(x, torch.Tensor) else x for x in args]


def _paged_check_full_compressor(args):
    expected, state = _paged_full_compressor_cpu_oracle(args)
    output = _paged_public_call(*args)
    torch.testing.assert_close(
        output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
    )
    torch.testing.assert_close(
        args[4].cpu(), state, rtol=1.3e-6, atol=1e-5, equal_nan=True
    )
    return output


@pytest.mark.parametrize("ratio,dim", [(4, 512), (4, 128), (128, 512)])
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_cpu_output_and_state(ratio, dim):
    _paged_check_full_compressor(
        _paged_full_to_device(_paged_full_compressor_case(ratio, dim))
    )


@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_fp32_strided_and_odd_rope(rope_dim):
    args = _paged_full_to_device(
        _paged_full_compressor_case(4, 5, torch.float32, rope_dim)
    )
    for index in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11):
        x = args[index]
        backing = torch.empty((*x.shape, 2), dtype=x.dtype, device=x.device)
        backing.fill_(torch.nan if x.is_floating_point() else -37)
        backing[..., 0].copy_(x)
        args[index] = backing[..., 0]
    _paged_check_full_compressor(args)


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_empty_history_and_valid_nan():
    args = _paged_full_to_device(_paged_full_compressor_case(4, 5, torch.float32, 4))
    args[0], args[5], args[6] = args[0][:0], args[5][:0], args[6][:0]
    _paged_check_full_compressor(args)
    page = int(args[7][0, 0].cpu())
    args[4][page, 0, 0] = torch.nan
    _paged_check_full_compressor(args)
    args[4][page, 0, 0] = 0.125
    args[4][page, 0, 2 * args[10].numel()] = torch.nan
    _paged_check_full_compressor(args)
    args[8], args[9] = args[8][:0], args[9][:0]
    assert _paged_check_full_compressor(args).shape == (0, 5)


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_mutating_metadata_graph():
    cpu = _paged_full_compressor_case(4, 5, torch.bfloat16, 4)
    table, block = cpu[7], cpu[4].shape[1]
    cpu[0] = cpu[0][:6]
    cpu[0][-2:].fill_(torch.nan)
    cpu[5] = torch.tensor([4, 5, 6, 7, -1, -1])
    cpu[6] = torch.tensor([int(table[0, 1]) * block + i for i in range(4)] + [-1, -1])
    cpu[8], cpu[9] = torch.tensor([0, 1, -1]), torch.tensor([7, 7, -1])
    args = _paged_full_to_device(cpu)
    _paged_public_call(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _paged_public_call(*args)
    for version in (1, 2):
        changed = [x.clone() if isinstance(x, torch.Tensor) else x for x in cpu]
        changed[0][:-2] *= 0.7 + version * 0.1
        changed[4][:-1] += version * 0.01
        if version == 2:
            changed[7] = changed[7].flip(1)
            changed[6][:4] = changed[7][0, 1] * block + torch.arange(4)
            changed[8], changed[9] = torch.tensor([1, -1, 0]), torch.tensor([3, 7, 7])
        expected, state = _paged_full_compressor_cpu_oracle(changed)
        for actual, value in zip(args, changed):
            if isinstance(actual, torch.Tensor):
                actual.copy_(value)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
        )
        torch.testing.assert_close(
            args[4].cpu(), state, rtol=1.3e-6, atol=1e-5, equal_nan=True
        )


@pytest.mark.skipif(
    _paged_cfg.QUICK_MODE, reason="39 GB address-boundary case excluded by --quick"
)
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_actual_shared_parent_high_pages():
    """Exercise all three real layouts over one 39 GB allocation, without copying it."""
    parent_bytes = 39_022_701_952
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < parent_bytes + 3 * 1024**3:
        pytest.skip("39 GB shared parent plus 3 GiB headroom is unavailable")
    parent = torch.empty(parent_bytes // 4, dtype=torch.float32, device="npu")
    for ratio, dim, block, width, offset in (
        (4, 512, 4, 2048, 0),
        (4, 128, 4, 512, 8192),
        (128, 512, 8, 1024, 0),
    ):
        cpu = _paged_full_compressor_case(ratio, dim)
        page_count = cpu[4].shape[0]
        physical = torch.tensor([0] + list(range(26079 - page_count, 26078)))
        assert physical.numel() == page_count and int(physical[-1]) == 26077
        # Put a known valid request-zero block on the highest physical page.
        active_compact_page = int(cpu[7][0, 1])
        physical[active_compact_page], physical[-1] = (
            physical[-1].clone(),
            physical[active_compact_page].clone(),
        )
        assert physical[0] == 0 and physical[active_compact_page] == 26077
        # Independent compact oracle uses identical page contents and logical ids.
        # Invalid table pages are kept invalid rather than clamped to a valid page.
        cpu[7][1, 0] = -1
        cpu[7][1, 1] = 2**40
        expected, expected_state = _paged_full_compressor_cpu_oracle(cpu)
        args = _paged_full_to_device([*cpu[:4], torch.empty(0), *cpu[5:]])
        wide = torch.as_strided(
            parent, (26078, block, width), (374096, width, 1), offset
        )
        for compact, page in enumerate(physical.tolist()):
            wide[page].copy_(cpu[4][compact])
        args[4] = wide
        valid_slots = (cpu[6] >= 0) & (cpu[6] < page_count * block)
        slots = cpu[6].clone()
        slots[valid_slots] = (
            physical[slots[valid_slots] // block] * block + slots[valid_slots] % block
        )
        table = cpu[7].clone()
        valid_pages = (table >= 0) & (table < page_count)
        table[valid_pages] = physical[table[valid_pages]]
        table_columns = 8192 if ratio == 4 else 4096
        actual_table = torch.full((2, table_columns), -1, dtype=torch.int64)
        actual_table[:, : table.shape[1]] = table
        assert actual_table[0, 1] == 26077
        assert (slots[valid_slots] // block == 26077).any()
        history_length = (2 if ratio == 4 else 1) * ratio
        assert any(
            request == 0
            and end >= 0
            and (end + 1) % ratio == 0
            and end - history_length + 1 <= block <= end
            for request, end in zip(cpu[8].tolist(), cpu[9].tolist())
        )
        requests = torch.full((512,), -1, dtype=torch.int64)
        positions = torch.full((512,), -1, dtype=torch.int64)
        requests[: cpu[8].numel()] = cpu[8]
        positions[: cpu[9].numel()] = cpu[9]
        args[6], args[7] = slots.npu(), actual_table.npu()
        args[8], args[9] = requests.npu(), positions.npu()
        expected = torch.cat(
            [
                expected,
                torch.zeros((512 - expected.shape[0], dim), dtype=expected.dtype),
            ]
        )
        guards = []
        for page in (0, int(physical[1]), 26077):
            begin = offset + page * 374096
            for guard_start in (begin - 16, begin + block * width):
                if guard_start >= 0:
                    guard = parent.narrow(0, guard_start, 16)
                    guard.fill_(19.125)
                    guards.append(guard)
        output = _paged_public_call(*args)
        for guard in guards:
            torch.testing.assert_close(
                guard.cpu(), torch.full((16,), 19.125), rtol=0, atol=0
            )
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
        )
        # Read only the initialized pages, including untouched rows/NaN sentinels.
        for compact, page in enumerate(physical.tolist()):
            torch.testing.assert_close(
                wide[page].cpu(),
                expected_state[compact],
                rtol=1.3e-6,
                atol=1e-5,
                equal_nan=True,
            )
        print(
            f"wide R={ratio} D={dim} parent_bytes={parent_bytes} "
            f"page=26077 element_offset={offset + 26077 * 374096}"
        )
        del args, wide, output


@pytest.mark.parametrize("n,hidden,ratio,dim", [(10, 4096, 128, 512), (17, 65, 4, 5)])
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_projection_tail_and_slot_masks(n, hidden, ratio, dim):
    # Place the activation at the end of its backing storage. A padded Cube
    # row must not be loaded merely because it is later masked before the dot.
    generator = torch.Generator().manual_seed(10030)
    coff = 2 if ratio == 4 else 1
    x = (torch.randn(n, hidden, generator=generator) * 0.03).bfloat16()
    x[3].fill_(torch.nan)
    wkv = (torch.randn(coff * dim, hidden, generator=generator) * 0.03).bfloat16()
    wgate = (torch.randn(coff * dim, hidden, generator=generator) * 0.03).bfloat16()
    ape = torch.randn(ratio, coff * dim, generator=generator) * 0.01
    positions = torch.arange(n)
    slots = positions.clone()
    slots[3], slots[5] = -1, 2**40
    empty = torch.empty(0, dtype=torch.int64)
    cpu = [
        x,
        wkv,
        wgate,
        ape,
        torch.full((4, 8, 2 * coff * dim), 7.125),
        positions,
        slots,
        torch.tensor([[0, 1, 2, 3]]),
        empty,
        empty,
        torch.ones(dim),
        torch.empty(32, 0),
        ratio,
    ]
    args = _paged_full_to_device(cpu)
    backing = torch.empty(16 * 1024 * 1024, dtype=x.dtype, device="npu")
    args[0] = backing[-n * hidden :].view(n, hidden)
    args[0].copy_(x)
    assert args[0].storage_offset() + args[0].numel() == backing.numel()
    assert _paged_check_full_compressor(args).shape == (0, dim)


def _paged_single_token_compressor_case(ratio, dim, dtype, hidden):
    args = _paged_full_compressor_case(ratio, dim, dtype, min(64, 2 * (dim // 2)))
    gen = torch.Generator().manual_seed(817 + hidden + dim)
    width = (2 if ratio == 4 else 1) * dim
    args[0] = (torch.randn(1, hidden, generator=gen) * 0.03).to(dtype)
    args[1] = (torch.randn(width, hidden, generator=gen) * 0.03).to(dtype)
    args[2] = (torch.randn(width, hidden, generator=gen) * 0.03).to(dtype)
    position = 2 * ratio - 1
    block = args[4].shape[1]
    args[5] = torch.tensor([position])
    args[6] = torch.tensor(
        [int(args[7][0, position // block]) * block + position % block]
    )
    args[8], args[9] = torch.tensor([0]), torch.tensor([position])
    return args


@pytest.mark.parametrize(
    "ratio,dim,dtype,hidden",
    [
        (4, 512, torch.bfloat16, 4096),
        (128, 512, torch.bfloat16, 4096),
        (4, 128, torch.bfloat16, 4096),
        (4, 5, torch.float16, 65),
        (128, 5, torch.bfloat16, 63),
        (4, 5, torch.float32, 33),
    ],
)
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_single_token_output_and_state(ratio, dim, dtype, hidden):
    for mode in (
        "valid",
        "negative_slot",
        "large_slot",
        "negative_position",
        "empty_output",
        "valid_nan",
        "strided_input",
    ):
        cpu = _paged_single_token_compressor_case(ratio, dim, dtype, hidden)
        if mode == "negative_slot":
            cpu[6].fill_(-1)
            cpu[0].fill_(torch.nan)
        elif mode == "large_slot":
            cpu[6].fill_(2**40)
            cpu[0].fill_(torch.nan)
        elif mode == "negative_position":
            cpu[5].fill_(-3)
        elif mode == "empty_output":
            cpu[8], cpu[9] = cpu[8][:0], cpu[9][:0]
        elif mode == "valid_nan":
            cpu[0][0, 0] = torch.nan
        args = _paged_full_to_device(cpu)
        backing = torch.full(
            (*args[4].shape, 2), 29.0, dtype=torch.float32, device=args[4].device
        )
        backing[..., 0].copy_(args[4])
        args[4] = backing[..., 0]
        for index, padding in ((1, 7), (2, 11)):
            weight = args[index]
            storage = torch.full(
                (weight.shape[0], hidden + padding),
                torch.nan,
                dtype=dtype,
                device=weight.device,
            )
            storage[:, :hidden].copy_(weight)
            args[index] = storage[:, :hidden]
        if mode == "strided_input":
            storage = torch.full(
                (*args[0].shape, 2), torch.nan, dtype=dtype, device=args[0].device
            )
            storage[..., 0].copy_(args[0])
            args[0] = storage[..., 0]
        _paged_check_full_compressor(args)
        torch.testing.assert_close(
            backing[..., 1], torch.full_like(backing[..., 1], 29)
        )


@pytest.mark.parametrize("ratio,dim", [(4, 512), (128, 512), (4, 128)])
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_full_compressor_single_token_mutating_graph(ratio, dim):
    cpu = _paged_single_token_compressor_case(ratio, dim, torch.bfloat16, 4096)
    args = _paged_full_to_device(cpu)
    _paged_public_call(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = _paged_public_call(*args)
    for version in (1, 2, 3):
        changed = [x.clone() if isinstance(x, torch.Tensor) else x for x in cpu]
        if version == 1:
            changed[0] *= 0.8
        elif version == 2:
            changed[0].fill_(torch.nan)
            changed[6].fill_(-1)
        else:
            changed[0] *= 1.1
            changed[5].fill_(0)
            changed[6].fill_(0)
            changed[9].fill_(ratio - 1)
        expected, state = _paged_full_compressor_cpu_oracle(changed)
        for actual, value in zip(args, changed):
            if isinstance(actual, torch.Tensor):
                actual.copy_(value)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            output.cpu(), expected, rtol=0.01, atol=0.01, equal_nan=True
        )
        torch.testing.assert_close(
            args[4].cpu(), state, rtol=1.3e-6, atol=1e-5, equal_nan=True
        )


@pytest.mark.parametrize("invalid", ["ratio", "metadata", "state", "weight", "rope"])
@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_compressor_rejects_invalid_contract_before_state_update(invalid):
    args = _paged_full_to_device(_paged_full_compressor_case(4, 5, torch.float32, 4))
    initial = args[4].clone()
    if invalid == "ratio":
        args[-1] = 8
    elif invalid == "metadata":
        args[5] = args[5].float()
    elif invalid == "state":
        args[4] = args[4].bfloat16()
    elif invalid == "weight":
        args[1] = args[1][:-1]
    else:
        args[11] = args[11][:, :3]
    with pytest.raises((TypeError, ValueError)):
        _paged_public_call(*args)
    torch.testing.assert_close(
        args[4].float(), initial.to(args[4].dtype).float(), equal_nan=True
    )


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
def test_paged_compressor_fake_output_and_mutation_schema():
    from torch._subclasses.fake_tensor import FakeTensorMode

    args = _paged_full_to_device(_paged_full_compressor_case(4, 5, torch.float32, 4))
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in args]
        out = _paged_public_call(*fake)
        assert out.shape == (args[9].numel(), args[10].numel())
        assert out.dtype == args[0].dtype
    schema = torch.ops.flaggems_vllm.compressor_paged.default._schema
    assert schema.arguments[4].alias_info.is_write


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
@pytest.mark.parametrize(
    "field",
    [
        "workspace",
        "out",
        "cu_seqlens",
        "seqused",
        "start_pos",
        "rope_sin",
        "rope_cos",
        "rotary_mode",
        "cache_mode",
        "coff",
        "rope_head_dim",
    ],
)
def test_paged_rejects_mixed_contract_before_state_update(field):
    args = _paged_full_to_device(_paged_full_compressor_case(4, 5, torch.float32, 4))
    before = args[4].clone()
    value = {"rotary_mode": 1, "cache_mode": 0, "coff": 1, "rope_head_dim": 2}.get(
        field, args[0]
    )
    with pytest.raises(ValueError, match="Paged compressor"):
        _paged_public_call(*args, **{field: value})
    # Compare storage bits so unchanged NaN sentinel rows remain equal.
    assert torch.equal(args[4].cpu().view(torch.uint8), before.cpu().view(torch.uint8))


@pytest.mark.compressor
@pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend")
@pytest.mark.parametrize(
    "field",
    [
        "positions",
        "slot_mapping",
        "output_request_ids",
        "output_positions",
        "cos_sin_cache",
        "state_block_table",
    ],
)
def test_paged_rejects_incomplete_metadata_before_state_update(field):
    args = _paged_full_to_device(_paged_full_compressor_case(4, 5, torch.float32, 4))
    before = args[4].clone()
    with pytest.raises(ValueError, match="complete explicit metadata"):
        _paged_public_call(*args, **{field: None})
    # Compare storage bits so unchanged NaN sentinel rows remain equal.
    assert torch.equal(args[4].cpu().view(torch.uint8), before.cpu().view(torch.uint8))
