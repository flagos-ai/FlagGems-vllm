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

"""Correctness tests and shared inference fixtures for QSA paged prefill."""

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

BF16_RTOL = 2e-4
BF16_ATOL = 2e-3
FP8_RTOL = 1e-3
FP8_ATOL_PER_SQRT_DIM = 1e-3


def make_inputs(
    query_lengths,
    compressed_lengths,
    *,
    k=512,
    ratio=4,
    page_size=16,
    query_base=0,
    visible=None,
    padded=False,
    device=None,
    seed=20261013,
    dtype=torch.bfloat16,
    head_dim=128,
    num_heads=4,
):
    """Build request-major packed queries and shuffled physical pages."""
    device = cfg.device if device is None else device
    torch.manual_seed(seed)
    rows = sum(query_lengths)
    pages_per_req = [
        (length + page_size - 1) // page_size for length in compressed_lengths
    ]
    width = max(pages_per_req)
    num_pages = sum(pages_per_req)
    q = torch.randn(
        (rows, num_heads, head_dim), dtype=torch.bfloat16, device=device
    ).to(dtype)
    cache = torch.randn(
        (num_pages, page_size, 1, head_dim), dtype=torch.bfloat16, device=device
    ).to(dtype)
    page_order = torch.randperm(num_pages, device=device).to(torch.int32)
    table = torch.full(
        (len(query_lengths), width), -1, dtype=torch.int32, device=device
    )
    cursor = 0
    for request, count in enumerate(pages_per_req):
        table[request, :count] = page_order[cursor : cursor + count]
        cursor += count
    offsets = [query_base]
    default_visible = []
    for length, context in zip(query_lengths, compressed_lengths):
        offsets.append(offsets[-1] + length)
        # Simulate the last `length` logical tokens of a causal prefill request.
        default_visible.extend(
            max(0, min(context, (context * ratio - length + row + 1) // ratio))
            for row in range(length)
        )
    starts = torch.tensor(offsets, dtype=torch.int32, device=device)
    visible = torch.tensor(
        default_visible if visible is None else visible,
        dtype=torch.int32,
        device=device,
    )
    if padded:
        q_storage = torch.empty(
            (rows, num_heads + 1, head_dim), dtype=q.dtype, device=device
        )
        q_storage[:, :num_heads].copy_(q)
        q = q_storage[:, :num_heads]
        cache_storage = torch.empty(
            (num_pages, page_size + 1, 1, head_dim), dtype=cache.dtype, device=device
        )
        cache_storage[:, :page_size].copy_(cache)
        cache = cache_storage[:, :page_size]
        table_storage = torch.full(
            (len(query_lengths), width * 2), -1, dtype=table.dtype, device=device
        )
        table_storage[:, ::2].copy_(table)
        table = table_storage[:, ::2]
    output = torch.full((rows, k), -17, dtype=torch.int32, device=device)
    return (
        q,
        cache,
        table,
        starts,
        visible,
        k * ratio,
        ratio,
        max(query_lengths),
        output,
        max(compressed_lengths) * ratio,
    )


def make_upstream_prefill_inputs(*, seq_len_slack=0, device=None, dtype=torch.bfloat16):
    """Exact BF16/FP8 inputs from upstream test_qsa_prefill_selection_correctness.

    Source: vLLM tests/models/qwen4_exp/test_qsa_reference.py.
    The oversized page table and page size 24 exercise clipped logits widths
    that need not coincide with the physical page geometry.
    """
    device = cfg.device if device is None else device
    torch.manual_seed(2)
    query_lengths = [3, 33]
    q = torch.randn((36, 4, 128), device=device, dtype=torch.bfloat16).to(dtype)
    cache = torch.randn((128, 24, 1, 128), device=device, dtype=torch.bfloat16).to(
        dtype
    )
    table = torch.randperm(128, device=device, dtype=torch.int32).reshape(2, 64)
    starts = torch.tensor([0, 3, 36], device=device, dtype=torch.int32)
    sequence_lengths = torch.tensor([5120, 4224], device=device, dtype=torch.int32)
    token_to_req = torch.repeat_interleave(
        torch.arange(2, device=device, dtype=torch.int32),
        torch.tensor(query_lengths, device=device),
    )
    positions = torch.cat(
        [
            torch.arange(length - queries, length, device=device, dtype=torch.int32)
            for queries, length in zip(query_lengths, [5120, 4224])
        ]
    )
    visible = torch.minimum(
        (positions + 1) // 4,
        sequence_lengths.index_select(0, token_to_req.long()) // 4,
    )
    return (
        q,
        cache,
        table,
        starts,
        visible,
        2048,
        4,
        33,
        torch.full((36, 512), -17, device=device, dtype=torch.int32),
        5120 + seq_len_slack,
    )


def make_upstream_split_prefill_inputs(decode_query_len, *, device=None):
    """Prefill slice from upstream test_qsa_split_selection_correctness.

    Preserve the preceding decode rows so query_start_loc has its original
    nonzero base. Decode and compressed-index expansion are separate APIs.
    """
    device = cfg.device if device is None else device
    torch.manual_seed(13)
    query_lengths = [decode_query_len, decode_query_len, 33]
    rows = sum(query_lengths)
    q = torch.randn((rows, 4, 128), device=device, dtype=torch.bfloat16)
    cache = torch.randn((120, 16, 1, 128), device=device, dtype=torch.bfloat16)
    table = torch.arange(120, device=device, dtype=torch.int32).view(3, 40)
    starts = torch.tensor(
        [0, decode_query_len, 2 * decode_query_len, rows],
        device=device,
        dtype=torch.int32,
    )
    positions = torch.cat(
        [torch.arange(2560 - queries, 2560, device=device) for queries in query_lengths]
    )
    visible = ((positions + 1) // 4).to(torch.int32)
    prefill = slice(2 * decode_query_len, rows)
    return (
        q[prefill],
        cache,
        table[2:],
        starts[2:],
        visible[prefill],
        2048,
        4,
        33,
        torch.full((33, 512), -17, device=device, dtype=torch.int32),
        2560,
    )


def make_upstream_no_complete_blocks_inputs(*, device=None):
    """Exact selection inputs from the upstream D=8 zero-visible fixture."""
    device = cfg.device if device is None else device
    return (
        torch.zeros((2, 4, 8), device=device, dtype=torch.bfloat16),
        torch.zeros((1, 16, 1, 8), device=device, dtype=torch.bfloat16),
        torch.zeros((1, 1), device=device, dtype=torch.int32),
        torch.tensor([0, 2], device=device, dtype=torch.int32),
        torch.zeros((2,), device=device, dtype=torch.int32),
        2048,
        4,
        2,
        torch.empty((2, 512), device=device, dtype=torch.int32),
        64,
    )


def torch_logits(q, cache, table, starts, visible, width=None):
    """Independent FP32 reference, with inactive columns set to -inf."""
    width = table.shape[1] * cache.shape[1] if width is None else width
    logits = torch.full((q.shape[0], width), -torch.inf, device=q.device)
    offsets = (starts - starts[0]).cpu().tolist()
    lengths = visible.cpu().tolist()
    for request in range(table.shape[0]):
        begin, end = offsets[request : request + 2]
        if begin == end:
            continue
        count = max(lengths[begin:end], default=0)
        if count == 0:
            continue
        columns = torch.arange(count, device=q.device)
        pages = table[request, columns // cache.shape[1]].long()
        keys = cache[pages, columns % cache.shape[1], 0].float()
        scores = torch.einsum("rhd,cd->rhc", q[begin:end].float(), keys)
        scores = scores.clamp_min(0).sum(1)
        mask = columns[None, :] < visible[begin:end, None]
        logits[begin:end, :count] = torch.where(mask, scores, -torch.inf)
    return logits


def torch_select(
    q,
    cache,
    table,
    starts,
    visible,
    token_topk,
    ratio,
    max_query_len,
    output,
    max_seq_len,
):
    """Select compressed groups with Torch FP32 scores and unsorted TopK.

    Request-local row chunks bound the temporary score tensors. This reference
    includes host metadata reads and is separate from native vLLM timings.
    """
    del max_query_len
    k = token_topk // ratio
    width = min(table.shape[1] * cache.shape[1], (max_seq_len + ratio - 1) // ratio)
    output.fill_(-1)
    offsets = (starts - starts[0]).cpu().tolist()
    for request, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        for first in range(begin, end, 256):
            last = min(first + 256, end)
            local_starts = torch.tensor(
                [0, last - first], dtype=starts.dtype, device=starts.device
            )
            scores = torch_logits(
                q[first:last],
                cache,
                table[request : request + 1],
                local_starts,
                visible[first:last],
                width,
            )
            selected = torch.topk(scores, min(k, width), dim=1, sorted=False).indices
            selected = torch.where(selected < visible[first:last, None], selected, -1)
            output[first:last, : selected.shape[1]].copy_(selected.to(torch.int32))


def upstream_torch_logits(q, cache, table, starts, visible):
    """Upstream fixture's exact FP32 einsum/reduction, before sqrt(D) scaling.

    Only use with the upstream fixtures whose entire page tables are valid.
    Other fixtures may contain sentinel page IDs and use ``torch_logits``.
    """
    lengths = starts[1:] - starts[:-1]
    token_to_req = torch.repeat_interleave(
        torch.arange(table.shape[0], device=q.device, dtype=torch.int32), lengths
    )
    pages = table.index_select(0, token_to_req.long()).long()
    keys = cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1)
    columns = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(columns >= visible.unsqueeze(1), -torch.inf)


def assert_selection(logits, visible, output, *, rtol=2e-4, atol=2e-3):
    """Check count, range, uniqueness, padding and the selected score multiset."""
    k = output.shape[1]
    lengths = visible.cpu().tolist()
    for row, length in enumerate(lengths):
        selected = output[row]
        valid = selected[selected >= 0].long()
        count = min(length, k)
        assert valid.numel() == count, (row, length, selected)
        assert (selected >= -1).all(), (row, selected)
        assert (valid < length).all(), (row, valid, length)
        assert valid.unique().numel() == count, (row, valid)
        assert (selected == -1).sum().item() == k - count
        if count:
            actual = logits[row, valid].sort().values
            expected = logits[row, :length].topk(count).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def assert_upstream_selection(logits, visible, output, *, dtype, head_dim=128):
    """Match upstream BF16 index equality and SM90 FP8 value tolerances.

    The upstream Torch reference divides scores by sqrt(D); production scores
    are unscaled. Scaling here reproduces its FP8 absolute tolerance and BF16
    reference Top-K without changing the production scoring contract.
    """
    fp8 = dtype == torch.float8_e4m3fn
    scale = head_dim**-0.5
    assert_selection(
        logits,
        visible,
        output,
        rtol=1e-3 if fp8 else 2e-4,
        atol=1e-3 / scale if fp8 else 2e-3,
    )
    for row, length in enumerate(visible.cpu().tolist()):
        count = min(length, output.shape[1])
        scores = logits[row, :length] / head_dim**0.5
        if fp8:
            if count:
                selected = output[row][output[row] >= 0].long()
                torch.testing.assert_close(
                    scores[selected].sort().values,
                    scores.topk(count).values.sort().values,
                    rtol=1e-3,
                    atol=1e-3,
                )
        else:
            expected = torch.full_like(output[row], -1)
            if count:
                expected[:count] = scores.topk(count).indices.to(torch.int32)
            torch.testing.assert_close(
                output[row].sort().values, expected.sort().values, rtol=0, atol=0
            )


def assert_native_selection(logits, visible, actual, native, *, dtype, head_dim):
    """Compare selected scores, counts and padding against native vLLM.

    Upstream BF16 fixtures separately check exact index sets.
    Synthetic ties may choose different equivalent indices; compare their
    selected score multisets directly, including valid counts and -1 padding.
    """
    for row, length in enumerate(visible.cpu().tolist()):
        count = min(length, actual.shape[1])
        a = actual[row][actual[row] >= 0].long()
        b = native[row][native[row] >= 0].long()
        assert a.numel() == b.numel() == count
        assert (actual[row] == -1).sum() == (native[row] == -1).sum()
        scores = logits[row]
        if dtype == torch.float8_e4m3fn:
            scores = scores / head_dim**0.5
        torch.testing.assert_close(
            scores[a].sort().values,
            scores[b].sort().values,
            rtol=1e-3 if dtype == torch.float8_e4m3fn else 2e-4,
            atol=1e-3 if dtype == torch.float8_e4m3fn else 2e-3,
        )


def vllm_reference():
    """Import vLLM's own QSA operator when it is available."""
    return pytest.importorskip(
        "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer",
        reason="vLLM QSA is required for the native cross-check",
    )


class QsaPrefillCase:
    """Build one packed prefill launch and its independent Torch reference."""

    def __init__(
        self,
        *,
        query_lengths=(7, 0, 13),
        compressed_lengths=(1536, 17, 2048),
        dtype=torch.bfloat16,
        device=None,
        metadata_dtype=torch.int32,
        fixture="packed",
        seq_len_slack=0,
        decode_query_len=1,
        zero_scores=False,
        operands=None,
        **settings,
    ):
        self.fixture = fixture
        self.device = cfg.device if device is None else device
        if operands is None:
            if fixture == "upstream_prefill":
                operands = make_upstream_prefill_inputs(
                    seq_len_slack=seq_len_slack, dtype=dtype, device=self.device
                )
            elif fixture == "upstream_split":
                operands = make_upstream_split_prefill_inputs(
                    decode_query_len, device=self.device
                )
            elif fixture == "no_complete_blocks":
                operands = make_upstream_no_complete_blocks_inputs(device=self.device)
            else:
                operands = make_inputs(
                    query_lengths,
                    compressed_lengths,
                    dtype=dtype,
                    device=self.device,
                    **settings,
                )
        (
            self.q,
            self.k_cache,
            self.page_table,
            self.query_start_loc,
            self.visible_blocks,
            self.token_topk,
            self.compress_ratio,
            self.max_query_len,
            self.block_indices,
            self.max_seq_len,
        ) = operands
        self.page_table = self.page_table.to(metadata_dtype)
        self.query_start_loc = self.query_start_loc.to(metadata_dtype)
        if zero_scores:
            self.q.zero_()

    def outputs(self):
        return torch.full_like(self.block_indices, -17)

    def score_args(self):
        return (
            self.q,
            self.k_cache,
            self.page_table,
            self.query_start_loc,
            self.visible_blocks,
        )

    def launch_args(self, output=None):
        return (
            *self.score_args(),
            self.token_topk,
            self.compress_ratio,
            self.max_query_len,
            self.block_indices if output is None else output,
            self.max_seq_len,
        )

    def run_kernel(self):
        output = self.outputs()
        pointer = output.data_ptr()
        result = flaggems_vllm.qsa_select_paged_prefill(*self.launch_args(output))
        assert result is None
        assert output.data_ptr() == pointer
        return output

    def run_reference(self):
        output = self.outputs()
        torch_select(*self.launch_args(output))
        return output

    def reference_scores(self):
        if self.fixture in ("upstream_prefill", "upstream_split", "no_complete_blocks"):
            return upstream_torch_logits(*self.score_args())
        return torch_logits(*self.score_args())


def _assert_matches(actual, expected, case) -> None:
    """Check selection values, valid counts, unique indices and -1 padding."""
    scores = case.reference_scores()
    dtype, dim = case.q.dtype, case.q.shape[-1]
    for output in (actual, expected):
        if case.fixture in ("upstream_prefill", "upstream_split", "no_complete_blocks"):
            assert_upstream_selection(
                scores, case.visible_blocks, output, dtype=dtype, head_dim=dim
            )
        else:
            fp8 = dtype == torch.float8_e4m3fn
            assert_selection(
                scores,
                case.visible_blocks,
                output,
                rtol=FP8_RTOL if fp8 else BF16_RTOL,
                atol=FP8_ATOL_PER_SQRT_DIM * dim**0.5 if fp8 else BF16_ATOL,
            )
    assert_native_selection(
        scores, case.visible_blocks, actual, expected, dtype=dtype, head_dim=dim
    )


# Each entry describes a batch layout or a public API boundary. The native
# comparison uses the same table; only unsupported native resource cases skip.
CASES = [
    pytest.param(dict(query_lengths=(1,), compressed_lengths=(17,)), id="single-short"),
    pytest.param(
        dict(
            query_lengths=(7, 0, 13),
            compressed_lengths=(1536, 17, 2048),
            query_base=19,
            padded=True,
        ),
        id="ragged-strided",
    ),
    pytest.param(
        dict(
            query_lengths=(33, 19),
            compressed_lengths=(1537, 4097),
            k=1024,
            page_size=64,
            query_base=61,
        ),
        id="topk1024",
    ),
    pytest.param(
        dict(query_lengths=(17,), compressed_lengths=(4097,), k=2048), id="topk2048"
    ),
    pytest.param(
        dict(
            query_lengths=(39, 34),
            compressed_lengths=(4096, 8192),
            query_base=113,
            budget_mb=1,
        ),
        id="cross-request-chunks",
    ),
    pytest.param(
        dict(query_lengths=(7,), compressed_lengths=(1025,), head_dim=8),
        id="dim8-scored",
    ),
    *[
        pytest.param(
            dict(
                query_lengths=(7,),
                compressed_lengths=(1537,),
                visible=[0, 1, 511, 512, 513, 1024, 1537],
                zero_scores=True,
                page_size=page,
            ),
            id=f"ties-page{page}",
        )
        for page in (16, 64)
    ],
    *[
        pytest.param(
            dict(
                query_lengths=lengths,
                compressed_lengths=(17,),
                visible=visible,
                num_heads=heads,
                head_dim=dim,
            ),
            id=f"{name}-h{heads}-d{dim}",
        )
        for name, lengths, visible in (
            ("empty", (0,), []),
            ("select-all", (5,), [0, 1, 3, 16, 17]),
        )
        for heads, dim in ((4, 128), (1, 1), (5, 96))
    ],
    *[
        pytest.param(
            dict(
                query_lengths=(4, 0, 5),
                compressed_lengths=(1025, 16, 1537),
                query_base=37,
                padded=True,
                visible=[0, 1, 511, 1025, 512, 513, 800, 1024, 1537],
                num_heads=heads,
                head_dim=dim,
                budget_mb=budget,
            ),
            id=f"h{heads}-d{dim}-budget{budget}",
        )
        for heads, dim in ((1, 7), (3, 24), (5, 96), (8, 256))
        for budget in (0, 512)
    ],
    *[
        pytest.param(
            dict(fixture="upstream_prefill", seq_len_slack=slack, budget_mb=budget),
            id=f"upstream-slack{slack}-budget{budget}",
        )
        for slack in (0, 1792)
        for budget in (0, 512)
    ],
    *[
        pytest.param(
            dict(fixture="upstream_split", decode_query_len=length),
            id=f"split-decode{length}",
        )
        for length in (1, 2, 3, 4)
    ],
    pytest.param(dict(fixture="no_complete_blocks"), id="no-complete-blocks"),
    *[
        pytest.param(
            dict(
                query_lengths=(rows,), compressed_lengths=(context,), k=k, query_base=53
            ),
            id=f"topk-r{rows}-k{k}",
        )
        for rows, context, k in (
            (1, 4096, 1024),
            (1, 8192, 2048),
            (65, 8192, 512),
            (65, 8192, 1024),
            (65, 8192, 2048),
        )
    ],
    *[
        pytest.param(
            dict(
                query_lengths=(3, 7),
                compressed_lengths=(1025, 1537),
                query_base=19,
                num_heads=heads,
                head_dim=dim,
            ),
            id=f"native-h{heads}-d{dim}",
        )
        for heads, dim in ((1, 32), (3, 64))
    ],
    pytest.param(dict(metadata_dtype=torch.int64), id="metadata-int64"),
]
if not cfg.QUICK_MODE:
    CASES += [
        pytest.param(
            dict(
                query_lengths=(129, 67, 31),
                compressed_lengths=(8193, 4097, 1025),
                query_base=97,
                padded=True,
                page_size=64,
            ),
            id="ragged-long",
        ),
        pytest.param(
            dict(query_lengths=(512,), compressed_lengths=(16385,), k=1024),
            id="long-prefill",
        ),
        *[
            pytest.param(
                dict(
                    query_lengths=(4, 0, 5),
                    compressed_lengths=(1025, 16, 1537),
                    query_base=37,
                    padded=True,
                    num_heads=heads,
                    head_dim=dim,
                ),
                id=f"h{heads}-d{dim}",
            )
            for heads, dim in ((1, 1), (2, 16), (5, 40), (12, 192), (17, 33), (3, 160))
        ],
    ]

DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(
        torch.float8_e4m3fn,
        id="fp8",
        marks=pytest.mark.skipif(
            flaggems_vllm.vendor_name != "nvidia",
            reason="FP8 selection is currently validated on NVIDIA",
        ),
    ),
]


@pytest.fixture
def accelerator():
    if torch.device(cfg.device).type == "cpu":
        pytest.skip("QSA prefill requires an accelerator")
    if flaggems_vllm.vendor_name == "nvidia":
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
    else:
        yield


def _case(settings, dtype, monkeypatch) -> QsaPrefillCase:
    settings = dict(settings)
    fixture = settings.get("fixture", "packed")
    if fixture in ("upstream_split", "no_complete_blocks") and dtype != torch.bfloat16:
        pytest.skip("The upstream fixture uses BF16 inputs")
    budget = settings.pop("budget_mb", 512)
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", str(budget))
    return QsaPrefillCase(dtype=dtype, **settings)


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.parametrize("settings", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_qsa_select_paged_prefill_matches_torch_reference(
    accelerator, monkeypatch, settings, dtype
) -> None:
    case = _case(settings, dtype, monkeypatch)
    _assert_matches(case.run_kernel(), case.run_reference(), case)


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0),
    reason="Native vLLM QSA comparison requires NVIDIA SM90",
)
@pytest.mark.parametrize("settings", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_qsa_select_paged_prefill_matches_vllm(
    accelerator, monkeypatch, settings, dtype
) -> None:
    reference = vllm_reference()
    case = _case(settings, dtype, monkeypatch)
    # Native vLLM scores every nonempty batch, including select-all inputs.
    # Restrict its cross-check to validated dot shapes; smaller dimensions
    # remain covered by the independent Torch test.
    if (
        case.fixture == "packed"
        and case.q.shape[0] > 0
        and case.q.shape[1:] not in ((4, 128), (1, 32), (3, 64), (5, 96), (8, 256))
    ):
        pytest.skip("Native cross-check is limited to validated head/dimension pairs")
    if dtype == torch.bfloat16 and case.q.shape[1:] == (8, 256):
        pytest.skip("Native BF16 H8/D256 exceeds H800 shared memory")
    monkeypatch.setattr(
        reference.envs,
        "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
        settings.get("budget_mb", 512),
    )
    actual = case.run_kernel()
    expected = case.outputs()
    # Native vLLM indexes page-table columns with unit stride. Keep the
    # production input strided while comparing the same logical page IDs.
    native_args = list(case.launch_args(expected))
    native_args[2] = case.page_table.contiguous()
    reference.qsa_select_paged_prefill(*native_args)
    _assert_matches(actual, expected, case)


@pytest.mark.qsa_select_paged_prefill
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia", reason="NVIDIA wrapper error contract"
)
@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "mixed_dtype",
        "heads",
        "head_dim",
        "cache_dim",
        "dim_stride",
        "output_stride",
        "visible_dtype",
        "ratio",
        "k",
        "grad",
        "overlap",
    ],
)
def test_qsa_select_paged_prefill_invalid_contract(accelerator, case) -> None:
    inputs = QsaPrefillCase(query_lengths=(8,), compressed_lengths=(1025,))
    error = NotImplementedError
    if case == "dtype":
        inputs.q = inputs.q.float()
        inputs.k_cache = inputs.k_cache.float()
    elif case == "mixed_dtype":
        inputs.k_cache = inputs.k_cache.to(torch.float8_e4m3fn)
        error = ValueError
    elif case == "heads":
        inputs.q = inputs.q[:, :0]
        error = ValueError
    elif case == "head_dim":
        inputs.q = inputs.q[..., :0]
        inputs.k_cache = inputs.k_cache[..., :0]
        error = ValueError
    elif case == "cache_dim":
        inputs.k_cache = inputs.k_cache[..., :64]
    elif case == "dim_stride":
        inputs.q = torch.empty((8, 4, 256), device=cfg.device, dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif case == "output_stride":
        inputs.block_indices = torch.empty(
            (8, 1024), device=cfg.device, dtype=torch.int32
        )[:, ::2]
    elif case == "visible_dtype":
        inputs.visible_blocks = inputs.visible_blocks.long()
        error = ValueError
    elif case == "ratio":
        inputs.compress_ratio = 0
        error = ValueError
    elif case == "k":
        inputs.token_topk = 256 * inputs.compress_ratio
    elif case == "grad":
        inputs.q.requires_grad_(True)
    elif case == "overlap":
        inputs.visible_blocks = inputs.block_indices.view(-1)[:8]
        error = ValueError
    with pytest.raises(error):
        flaggems_vllm.qsa_select_paged_prefill(*inputs.launch_args())
