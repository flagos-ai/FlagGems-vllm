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

"""Correctness tests for the QSA fused prepare operator."""

from __future__ import annotations

import pytest
import torch

from flaggems_vllm import qsa_prepare

# ------------------------------------------------------------------ constants
HEAD_DIM = 128
QUERY_HEADS = 4
COMPRESS_RATIO = 4
BLOCK_SIZE = 16
COMPRESSED_PAGE_SIZE = BLOCK_SIZE // COMPRESS_RATIO
MROPE_SECTION = (11, 11, 10)
NORM_EPS = 1e-6
ROTARY_DIM = HEAD_DIM // 2
COS_SIN_ROWS = 32768
# The torch reference rounds its intermediates at slightly different points
# than the kernel, so the comparison window is a few bf16 ulp wide.
RTOL = 1.0e-3
ATOL = 3.125e-2
# Tolerance for the fp8 attention path against vLLM, whose rope entry point
# differs from the fused kernel by up to one bf16 ulp before rounding.
FP8_RTOL = 1.6e-2
FP8_ATOL = 1.0e-2

# Main attention: the RoPE width follows the indexer's head dim, not MAIN_D.
MAIN_QUERY_HEADS = 6
MAIN_KV_HEADS = 2
MAIN_HEAD_DIM = 256
MAIN_PAGE_SIZE = 16
MAIN_EPS = 1e-6
MAIN_K_SCALE = 0.5
MAIN_V_SCALE = 2.0
MAIN_HALF = HEAD_DIM // 4
MAIN_ROTARY_DIM = 2 * MAIN_HALF

# Each entry in CASES is a batch scenario combined with a code-path setting, not a
# tensor size: _case() expands one entry into about two dozen operands whose shapes
# are unrelated to each other, from (128,) norm weights up to a (32768, 64) rope
# table. The settings vary the RoPE and position-cache layout, which selects
# different code paths inside the kernel, across eight batch layouts.

MIXED_BATCH = ([260, 259, 138], [1, 1, 37], [8, 8, 8])
FRESH_BATCH = ([37], [37], [0])
TILED_BATCH = ([4097], [4097], [0])
# Sixteen single-token requests behind long histories: the one layout where every
# request contributes one token, so the ring commit stores a single row for each
# of them and takes the scalar path instead of the [STATE_SIZE, D] tile.
DECODE_BATCH = ([512] * 16, [1] * 16, [8] * 16)

# The layouts below sit at the envelope vLLM actually allows rather than at the
# sizes a lone request produces. On this class of GPU (its >= 70 GiB, non-A100
# tier) the defaults are max_num_seqs = 1024 and max_num_batched_tokens = 8192 for
# an OpenAI server, 16384 for offline throughput.
#
# A loaded server decodes with every slot busy, which is also when the scalar ring
# commit and the main-attention tiling meet: 1024 tokens is exactly MAIN_TILE_T's
# threshold, so DECODE_MAX is the only layout combining the two.
DECODE_DEEP = ([512] * 512, [1] * 512, [8] * 512)
DECODE_MAX = ([512] * 1024, [1] * 1024, [8] * 1024)
# The ceilings the benchmark's last rows carry: 1010 + 14 = 1024 sequences, which
# is max_num_seqs, holding 8178 tokens — as close to the 8192 serving token budget
# as max_num_seqs allows.
MIXED_DEEP = (
    [512] * 1024,
    [1] * 1010 + [512] * 14,
    [8] * 1010 + [0] * 14,
)
# One request filling the offline token budget, which is twice the serving one.
LONG_BATCH = ([16384], [16384], [0])

CASES = [
    pytest.param(True, True, True, 4, MIXED_BATCH, id="mrope"),
    pytest.param(False, False, False, 4, MIXED_BATCH, id="text"),
    pytest.param(True, False, True, 8, MIXED_BATCH, id="mrope-model-1d"),
    pytest.param(True, True, False, 4, MIXED_BATCH, id="mrope-no-position-cache"),
    pytest.param(
        True, False, False, 8, MIXED_BATCH, id="mrope-model-1d-no-position-cache"
    ),
    pytest.param(False, False, True, 4, MIXED_BATCH, id="text-with-position-cache"),
    pytest.param(True, True, True, 4, FRESH_BATCH, id="fresh"),
    pytest.param(True, True, True, 4, TILED_BATCH, id="tiled"),
    pytest.param(True, True, True, 4, DECODE_BATCH, id="decode"),
    pytest.param(True, True, True, 8, DECODE_BATCH, id="decode-state-8"),
    pytest.param(True, True, False, 4, DECODE_BATCH, id="decode-no-position-cache"),
    pytest.param(
        False, False, False, 4, DECODE_BATCH, id="decode-text-no-position-cache"
    ),
    pytest.param(True, True, True, 4, DECODE_DEEP, id="decode-deep"),
    pytest.param(True, True, True, 8, DECODE_DEEP, id="decode-deep-state-8"),
    pytest.param(True, True, True, 4, DECODE_MAX, id="decode-max"),
    pytest.param(True, True, True, 4, MIXED_DEEP, id="mixed-deep"),
    pytest.param(True, True, True, 4, LONG_BATCH, id="long"),
]

DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(torch.float8_e4m3fn, id="fp8"),
]


# --------------------------------------------------------- torch reference
def _gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm in fp32.

    The kernel keeps the normalized rows in fp32 (``cos.dtype`` is fp32 because
    the cos/sin table is) and only rounds when it stores, so the reference must
    not round early either.
    """

    x = x.to(torch.float32)
    variance = x.square().mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * (weight.to(torch.float32) + 1.0)


def _apply_interleaved_mrope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope: bool,
) -> torch.Tensor:
    """NeoX rotary over the first head half with an interleaved MRoPE axis map.

    ``x`` is ``[rows, heads, head_dim]`` and ``positions`` is one ``[t, h, w]``
    row per row of ``x``.
    """

    lane = torch.arange(ROTARY_DIM // 2, device=x.device)
    axis = lane.remainder(3) if mrope else torch.zeros_like(lane)
    position = positions.index_select(1, axis)
    cache_row = cos_sin_cache[position]
    cos = cache_row.gather(2, lane.expand_as(position)[..., None]).squeeze(-1)
    sin = cache_row.gather(
        2, (lane + ROTARY_DIM // 2).expand_as(position)[..., None]
    ).squeeze(-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    first = x[..., : ROTARY_DIM // 2].to(torch.float32)
    second = x[..., ROTARY_DIM // 2 : ROTARY_DIM].to(torch.float32)
    rotated = torch.cat(
        (first * cos - second * sin, second * cos + first * sin), dim=-1
    )
    return torch.cat((rotated.to(x.dtype), x[..., ROTARY_DIM:]), dim=-1)


def _to_dst_dtype(x: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    """Round to BF16 like the unfused path, then scale for an FP8 destination."""

    x = x.to(torch.bfloat16)
    if dtype == torch.float8_e4m3fn:
        x = (x.to(torch.float32) / scale).to(dtype)
    else:
        x = x.to(dtype)
    return x


def _store_rotated(
    y: torch.Tensor,
    o1: torch.Tensor,
    o2: torch.Tensor,
    dtype: torch.dtype,
    scale: float | None,
) -> torch.Tensor:
    """Store a normalized head whose first ``2 * len(o1)`` dims are rotated."""

    half = o1.shape[-1]
    out = y.clone()
    out[..., :half] = o1
    out[..., half : 2 * half] = o2
    return _to_dst_dtype(out, dtype, scale)


def _main_norm_rope_gate(
    case: "QsaPrepareCase",
    q_out: torch.Tensor,
    gate_out: torch.Tensor,
    kv_cache: torch.Tensor,
) -> None:
    """Reference for the main-attention path: QK-norm + RoPE + gate + cache."""

    qkv = case.main["main_qkv"]
    num_tokens = qkv.shape[0]
    # The Q/gate outputs are always bf16; only the KV cache follows the
    # indexer dtype.
    q_dtype = q_out.dtype
    kv_dtype = kv_cache.dtype
    lane = torch.arange(MAIN_HALF, device=qkv.device)
    is_h = (lane % 3 == 1) & (lane < 3 * MROPE_SECTION[1])
    is_w = (lane % 3 == 2) & (lane < 3 * MROPE_SECTION[2])

    for token in range(num_tokens):
        for head in range(MAIN_QUERY_HEADS + MAIN_KV_HEADS):
            is_k = head >= MAIN_QUERY_HEADS
            kv_head = head - MAIN_QUERY_HEADS
            if is_k:
                src = (2 * MAIN_QUERY_HEADS + kv_head) * MAIN_HEAD_DIM
                weight = case.main["main_k_norm_weight"]
            else:
                src = 2 * head * MAIN_HEAD_DIM
                weight = case.main["main_q_norm_weight"]
            row = qkv[token, src : src + MAIN_HEAD_DIM].to(torch.float32)
            inv_rms = torch.rsqrt(row.square().sum() / MAIN_HEAD_DIM + MAIN_EPS)
            w = weight.to(torch.float32) + 1.0
            y = row * inv_rms * w

            x1 = qkv[token, src : src + MAIN_HALF].to(torch.float32)
            x2 = qkv[token, src + MAIN_HALF : src + 2 * MAIN_HALF].to(torch.float32)
            w1 = w[:MAIN_HALF]
            w2 = w[MAIN_HALF : 2 * MAIN_HALF]
            x1 = (x1 * inv_rms * w1).to(torch.bfloat16).to(torch.float32)
            x2 = (x2 * inv_rms * w2).to(torch.bfloat16).to(torch.float32)

            if case.is_2d_positions:
                pos = case.positions[:, token].to(torch.int64)
                axis_pos = torch.where(is_h, pos[1], torch.where(is_w, pos[2], pos[0]))
            else:
                # A 1D position table means the temporal position on all axes.
                axis_pos = case.positions[token].to(torch.int64).expand(MAIN_HALF)
            cos = case.cos_sin_cache[axis_pos, lane].to(torch.float32)
            sin = case.cos_sin_cache[axis_pos, MAIN_HALF + lane].to(torch.float32)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin

            if is_k:
                slot = int(case.main["main_slot_mapping"][token])
                if slot >= 0:
                    block, row_index = divmod(slot, MAIN_PAGE_SIZE)
                    fp8 = kv_dtype == torch.float8_e4m3fn
                    scale_k = MAIN_K_SCALE if fp8 else None
                    scale_v = MAIN_V_SCALE if fp8 else None
                    kv_cache[block, row_index, kv_head, :MAIN_HEAD_DIM].copy_(
                        _store_rotated(y, o1, o2, kv_dtype, scale_k)
                    )
                    value = qkv[
                        token,
                        src
                        + MAIN_KV_HEADS * MAIN_HEAD_DIM : src
                        + MAIN_KV_HEADS * MAIN_HEAD_DIM
                        + MAIN_HEAD_DIM,
                    ]
                    kv_cache[block, row_index, kv_head, MAIN_HEAD_DIM:].copy_(
                        _to_dst_dtype(value, kv_dtype, scale_v)
                    )
            else:
                q_out[token, head].copy_(_store_rotated(y, o1, o2, q_dtype, None))
                gate = qkv[token, src + MAIN_HEAD_DIM : src + 2 * MAIN_HEAD_DIM]
                gate_out[token, head].copy_(gate)


def _main_inputs(case: "QsaPrepareCase", fp8_cache: bool) -> dict:
    """Random main-attention arguments with a paged cache in vLLM's layout."""

    num_tokens = case.num_tokens
    num_blocks = num_tokens // MAIN_PAGE_SIZE + 2
    cache = torch.zeros(
        num_blocks,
        MAIN_KV_HEADS,
        MAIN_PAGE_SIZE,
        2 * MAIN_HEAD_DIM,
        dtype=torch.uint8 if fp8_cache else torch.bfloat16,
        device=case.device,
    ).transpose(1, 2)
    slots = torch.randperm(num_blocks * MAIN_PAGE_SIZE, device=case.device)[:num_tokens]
    slots[-1] = -1
    norm_weights = (
        torch.randn(2, MAIN_HEAD_DIM, dtype=torch.bfloat16, device=case.device) * 0.2
    )
    qkv_width = 2 * (MAIN_QUERY_HEADS + MAIN_KV_HEADS) * MAIN_HEAD_DIM
    qkv = torch.randn(
        num_tokens,
        qkv_width + (QUERY_HEADS + 1) * HEAD_DIM,
        dtype=torch.bfloat16,
        device=case.device,
    )[:, :qkv_width]
    return dict(
        main_qkv=qkv,
        main_q_norm_weight=norm_weights[0],
        main_k_norm_weight=norm_weights[1],
        main_eps=MAIN_EPS,
        main_kv_cache=cache.view(torch.float8_e4m3fn) if fp8_cache else cache,
        main_slot_mapping=slots,
        main_k_scale=MAIN_K_SCALE,
        main_v_scale=MAIN_V_SCALE,
    )


# ------------------------------------------------------------------ fixture
class QsaPrepareCase:
    """Builds one QSA prepare launch plus its unfused torch reference."""

    def __init__(
        self,
        *,
        seq_lens: list[int],
        query_lens: list[int],
        history_lens: list[int],
        state_size: int = 4,
        mrope: bool = True,
        is_2d_positions: bool = True,
        cache_rope_positions: bool = True,
        indexer_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.state_size = state_size
        self.mrope = mrope
        self.is_2d_positions = is_2d_positions
        self.cache_rope_positions = cache_rope_positions
        self.indexer_dtype = indexer_dtype
        self.seq_lens = list(seq_lens)
        self.query_lens = list(query_lens)
        self.history_lens = list(history_lens)
        self.num_requests = len(seq_lens)
        self._build_metadata()
        self._build_caches()
        self._build_inputs()
        self.main = _main_inputs(self, indexer_dtype == torch.float8_e4m3fn)

    # ------------------------------------------------------------------- setup
    def _build_metadata(self) -> None:
        device = self.device
        self.token_to_req = torch.cat(
            [
                torch.full((length,), request, dtype=torch.int32)
                for request, length in enumerate(self.query_lens)
            ]
        ).to(device)
        self.logical_positions = torch.cat(
            [
                torch.arange(seq_len - query_len, seq_len, dtype=torch.int64)
                for seq_len, query_len in zip(self.seq_lens, self.query_lens)
            ]
        ).to(device)
        self.num_tokens = self.logical_positions.numel()
        self.query_start_loc = torch.tensor(
            [0, *torch.tensor(self.query_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        self.positions = (
            torch.stack(
                [
                    self.logical_positions,
                    self.logical_positions // 7 + 3,
                    self.logical_positions // 13 + 11,
                ]
            )
            if self.is_2d_positions
            else self.logical_positions
        )
        compressed_counts = [
            (seq_len // COMPRESS_RATIO + COMPRESSED_PAGE_SIZE - 1)
            // COMPRESSED_PAGE_SIZE
            for seq_len in self.seq_lens
        ]
        self.raw_block_table = torch.arange(
            self.num_requests, dtype=torch.int32, device=device
        ).unsqueeze(1)
        self.compressed_block_table = torch.full(
            (self.num_requests, max(compressed_counts)),
            -1,
            dtype=torch.int32,
            device=device,
        )
        offset = 0
        for request, count in enumerate(compressed_counts):
            self.compressed_block_table[request, :count] = torch.arange(
                offset, offset + count, dtype=torch.int32, device=device
            )
            offset += count
        self.num_compressed_blocks = offset

        self.state_slots = torch.full(
            (self.num_tokens,), -1, dtype=torch.int64, device=device
        )
        self.compressed_slots = torch.full(
            (self.num_tokens,), -1, dtype=torch.int64, device=device
        )
        start = 0
        for request, (seq_len, query_len) in enumerate(
            zip(self.seq_lens, self.query_lens)
        ):
            for index in range(query_len):
                token = start + index
                position = seq_len - query_len + index
                if token + self.state_size >= start + query_len:
                    self.state_slots[token] = (
                        request * self.state_size + position % self.state_size
                    )
                if (position + 1) % COMPRESS_RATIO == 0:
                    compressed = position // COMPRESS_RATIO
                    page = int(
                        self.compressed_block_table[
                            request, compressed // COMPRESSED_PAGE_SIZE
                        ]
                    )
                    self.compressed_slots[token] = (
                        page * COMPRESSED_PAGE_SIZE + compressed % COMPRESSED_PAGE_SIZE
                    )
            start += query_len

        self.group_counts = [
            seq_len // COMPRESS_RATIO - (seq_len - query_len) // COMPRESS_RATIO
            for seq_len, query_len in zip(self.seq_lens, self.query_lens)
        ]
        work_counts = [max(1, count) for count in self.group_counts]
        requests: list[int] = []
        local: list[int] = []
        for request, count in enumerate(work_counts):
            requests.extend([request] * count)
            local.extend(range(count))
        max_k_work = (
            self.num_tokens + (COMPRESS_RATIO - 1) * self.num_requests
        ) // COMPRESS_RATIO
        self.key_work_metadata = torch.full(
            (max_k_work, 2), -1, dtype=torch.int32, device=device
        )
        self.key_work_metadata[: len(requests)] = torch.tensor(
            list(zip(requests, local)), dtype=torch.int32, device=device
        )

    def _token_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return the ``[len(tokens), 3]`` MRoPE row of each token."""

        if self.is_2d_positions:
            rows = self.positions[:, tokens]
        else:
            rows = self.logical_positions[tokens].expand(3, -1)
        if not self.mrope:
            rows = rows[:1].expand(3, -1)
        return rows.transpose(0, 1).to(torch.int64)

    def _build_caches(self) -> None:
        device = self.device
        raw_width = HEAD_DIM + 12 if self.cache_rope_positions else HEAD_DIM
        raw_page_elements = self.state_size * raw_width
        self.state_cache_storage = torch.zeros(
            self.num_requests * (raw_page_elements + 16),
            dtype=torch.bfloat16,
            device=device,
        )
        self.base_raw_cache = torch.as_strided(
            self.state_cache_storage,
            (self.num_requests, self.state_size, 1, raw_width),
            (raw_page_elements + 16, raw_width, raw_width, 1),
        )
        for request, history_len in enumerate(self.history_lens):
            history_end = self.seq_lens[request] - self.query_lens[request]
            for position in range(history_end - history_len, history_end):
                row = self.base_raw_cache[request, position % self.state_size, 0]
                row[:HEAD_DIM] = torch.randn(
                    HEAD_DIM, dtype=torch.bfloat16, device=device
                )
                if self.cache_rope_positions:
                    row[HEAD_DIM:].view(torch.int64)[:3].copy_(
                        torch.tensor(
                            [position, position // 7 + 3, position // 13 + 11],
                            dtype=torch.int64,
                            device=device,
                        )
                    )
        page_elements = COMPRESSED_PAGE_SIZE * HEAD_DIM
        self.compressed_cache_storage = torch.zeros(
            self.num_compressed_blocks * (page_elements + 16),
            dtype=self.indexer_dtype,
            device=device,
        )
        self.base_compressed_cache = torch.as_strided(
            self.compressed_cache_storage,
            (self.num_compressed_blocks, COMPRESSED_PAGE_SIZE, 1, HEAD_DIM),
            (page_elements + 16, HEAD_DIM, HEAD_DIM, 1),
        )
        self.query_norm_weight = (
            torch.randn(HEAD_DIM, dtype=torch.float32, device=device) * 0.2
        ).to(torch.bfloat16)
        self.key_norm_weight = (
            torch.randn(HEAD_DIM, dtype=torch.float32, device=device) * 0.2
        ).to(torch.bfloat16)

    def _build_inputs(self) -> None:
        device = self.device
        projected = torch.randn(
            self.num_tokens,
            (QUERY_HEADS + 1) * HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        self.query = projected[:, : QUERY_HEADS * HEAD_DIM].contiguous()
        self.key = projected[:, QUERY_HEADS * HEAD_DIM :].contiguous()
        frequencies = torch.arange(ROTARY_DIM // 2, dtype=torch.float32, device=device)
        angles = torch.arange(COS_SIN_ROWS, dtype=torch.float32, device=device)[
            :, None
        ] * (1.0 + frequencies[None, :] * 1e-3)
        self.cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=1).contiguous()

    # ------------------------------------------------------------------ launch
    def outputs(self):
        """Fresh output buffers for the fused launch."""

        query_out = torch.empty(
            self.num_tokens,
            QUERY_HEADS,
            HEAD_DIM,
            dtype=self.indexer_dtype,
            device=self.device,
        )
        return (
            query_out,
            self.base_raw_cache.clone(),
            self.base_compressed_cache.clone(),
        )

    def reference_outputs(self):
        """Buffers the torch reference fills; the main side is filled below."""

        query_out = torch.empty(
            self.num_tokens,
            QUERY_HEADS,
            HEAD_DIM,
            dtype=self.indexer_dtype,
            device=self.device,
        )
        main_q_out = torch.empty(
            self.num_tokens,
            MAIN_QUERY_HEADS,
            MAIN_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        main_gate_out = torch.empty_like(main_q_out)
        return (
            query_out,
            self.base_raw_cache.clone(),
            self.base_compressed_cache.clone(),
            main_q_out,
            main_gate_out,
            self.main["main_kv_cache"].clone(),
        )

    def run_kernel(self):
        outputs = self.outputs()
        main_q_out, main_gate_out = qsa_prepare(
            *self._launch_args(outputs),
            compress_ratio=COMPRESS_RATIO,
            mrope_section=MROPE_SECTION if self.mrope else None,
            rope_pos_offset=HEAD_DIM if self.cache_rope_positions else None,
            **self.main,
        )
        torch.cuda.synchronize()
        return (*outputs, main_q_out, main_gate_out, self.main["main_kv_cache"])

    def _launch_args(self, outputs):
        query_out, raw_cache, compressed_cache = outputs
        return (
            self.query,
            self.key,
            self.positions,
            self.cos_sin_cache,
            self.query_norm_weight,
            self.key_norm_weight,
            NORM_EPS,
            query_out,
            raw_cache,
            self.state_slots,
            self.raw_block_table,
            self.query_start_loc,
            self.logical_positions,
            compressed_cache,
            self.compressed_slots,
            self.key_work_metadata,
        )

    # --------------------------------------------------------------- reference
    def run_reference(self):
        (
            query_out,
            raw_cache,
            compressed_cache,
            main_q_out,
            main_gate_out,
            main_kv_cache,
        ) = self.reference_outputs()

        normalized = _gemma_rms_norm(
            self.query.reshape(self.num_tokens, QUERY_HEADS, HEAD_DIM),
            self.query_norm_weight,
            NORM_EPS,
        )
        query_out.copy_(
            _apply_interleaved_mrope(
                normalized,
                self._token_positions(
                    torch.arange(self.num_tokens, device=self.device)
                ),
                self.cos_sin_cache,
                self.is_2d_positions,
            ).to(self.indexer_dtype)
        )
        self._reference_compress(raw_cache, compressed_cache)
        self._reference_state_ring(raw_cache)
        _main_norm_rope_gate(self, main_q_out, main_gate_out, main_kv_cache)
        return (
            query_out,
            raw_cache,
            compressed_cache,
            main_q_out,
            main_gate_out,
            main_kv_cache,
        )

    def _reference_compress(self, raw_cache, compressed_cache) -> None:
        raw_keys = self.key.reshape(self.num_tokens, 1, HEAD_DIM)
        state_rows = raw_cache[..., :HEAD_DIM]
        pooled_rows: list[torch.Tensor] = []
        slot_list: list[int] = []
        first_tokens: list[int] = []
        first_from_ring: list[tuple[int, int, int]] = []
        for request, work_in_request in self.key_work_metadata.tolist():
            if request < 0 or work_in_request >= self.group_counts[request]:
                continue
            query_start = int(self.query_start_loc[request])
            query_end = int(self.query_start_loc[request + 1])
            chunk_end = int(self.logical_positions[query_end - 1])
            chunk_start = chunk_end - (query_end - query_start) + 1
            first_boundary = (
                (chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO
            ) * COMPRESS_RATIO - 1
            group_end = first_boundary + work_in_request * COMPRESS_RATIO
            token = query_start + group_end - chunk_start
            slot = int(self.compressed_slots[token])
            if slot < 0:
                continue
            rows = []
            for offset in range(COMPRESS_RATIO):
                position = group_end - (COMPRESS_RATIO - 1) + offset
                if position >= chunk_start:
                    rows.append(raw_keys[query_start + position - chunk_start])
                else:
                    rows.append(
                        state_rows[request, position % self.state_size, 0].unsqueeze(0)
                    )
            pooled_rows.append(
                (torch.stack(rows).to(torch.float32).sum(0) / COMPRESS_RATIO)
                .to(torch.bfloat16)
                .unsqueeze(0)
            )
            slot_list.append(slot)
            first_position = group_end - (COMPRESS_RATIO - 1)
            if first_position >= chunk_start:
                first_tokens.append(query_start + first_position - chunk_start)
                first_from_ring.append((-1, -1, first_position))
            else:
                first_tokens.append(-1)
                first_from_ring.append(
                    (request, first_position % self.state_size, first_position)
                )

        if not pooled_rows:
            return
        pooled = torch.cat(pooled_rows, dim=0)
        normalized = _gemma_rms_norm(pooled, self.key_norm_weight, NORM_EPS)
        positions = torch.zeros(
            len(pooled_rows), 3, dtype=torch.int64, device=self.device
        )
        for index, (token, ring) in enumerate(zip(first_tokens, first_from_ring)):
            if not self.cache_rope_positions:
                positions[index] = ring[2]
            elif token >= 0:
                positions[index] = self._token_positions(
                    torch.tensor([token], device=self.device)
                )[0]
            else:
                positions[index] = raw_cache[ring[0], ring[1], 0, HEAD_DIM:].view(
                    torch.int64
                )[:3]
        normalized = _apply_interleaved_mrope(
            normalized, positions, self.cos_sin_cache, self.mrope
        )
        for index, slot in enumerate(slot_list):
            block, offset = divmod(slot, COMPRESSED_PAGE_SIZE)
            compressed_cache[block, offset, 0].copy_(
                normalized[index, 0].to(self.indexer_dtype)
            )

    def _reference_state_ring(self, raw_cache) -> None:
        for request, query_len in enumerate(self.query_lens):
            query_end = int(self.query_start_loc[request + 1])
            rows = min(query_len, self.state_size)
            for offset in range(rows):
                token = query_end - rows + offset
                slot = int(self.state_slots[token])
                if slot < 0:
                    continue
                block, ring = divmod(slot, self.state_size)
                raw_cache[block, ring, 0, :HEAD_DIM].copy_(self.key[token])
                if self.cache_rope_positions:
                    raw_cache[block, ring, 0, HEAD_DIM:].view(torch.int64)[:3].copy_(
                        self._token_positions(
                            torch.tensor([token], device=self.device)
                        )[0]
                    )


def _assert_fp8_within_one_ulp(
    actual: torch.Tensor, expected: torch.Tensor, tag: str
) -> None:
    """e4m3 is sign-magnitude, so within a sign the code order matches the value
    order and one ulp is one code step."""

    code_diff = (
        actual.view(torch.uint8).int() - expected.view(torch.uint8).int()
    ).abs()
    abs_diff = (actual.float() - expected.float()).abs()
    ok = (code_diff <= 1) | (abs_diff <= 2**-8)
    assert bool(ok.all()), f"{tag}: fp8 differs by more than one ulp"


def _assert_matches(actual, expected) -> None:
    (
        query_out,
        raw_cache,
        compressed_cache,
        main_q_out,
        main_gate_out,
        main_kv_cache,
    ) = actual
    (
        expected_query,
        expected_raw,
        expected_compressed,
        expected_main_q,
        expected_main_gate,
        expected_main_cache,
    ) = expected
    fp8 = query_out.dtype == torch.float8_e4m3fn

    if fp8:
        # Both paths round the same bf16 intermediates to e4m3, so an exact
        # match is not expected; the unfused rope differs by up to one code step.
        _assert_fp8_within_one_ulp(query_out, expected_query, "query output")
    else:
        torch.testing.assert_close(query_out, expected_query, rtol=RTOL, atol=ATOL)
    # The ring commit is a pure copy, so it must agree bit for bit.
    assert torch.equal(
        raw_cache.view(torch.int16), expected_raw.view(torch.int16)
    ), "state ring cache mismatch"
    if fp8:
        _assert_fp8_within_one_ulp(
            compressed_cache, expected_compressed, "compressed key cache"
        )
    else:
        torch.testing.assert_close(
            compressed_cache, expected_compressed, rtol=RTOL, atol=ATOL
        )
    # The main Q/gate pair is always bf16, whatever the indexer dtype is.
    torch.testing.assert_close(main_q_out, expected_main_q, rtol=RTOL, atol=ATOL)
    # The gate is copied, not computed, so it must agree exactly.
    assert torch.equal(main_gate_out, expected_main_gate), "main gate mismatch"
    if main_kv_cache.dtype == torch.float8_e4m3fn:
        _assert_fp8_within_one_ulp(main_kv_cache, expected_main_cache, "main kv cache")
    else:
        torch.testing.assert_close(
            main_kv_cache, expected_main_cache, rtol=RTOL, atol=ATOL
        )


def _case(mrope, is_2d_positions, cache_rope_positions, state_size, batch, dtype):
    seq_lens, query_lens, history_lens = batch
    return QsaPrepareCase(
        seq_lens=seq_lens,
        query_lens=query_lens,
        history_lens=history_lens,
        state_size=state_size,
        mrope=mrope,
        is_2d_positions=is_2d_positions,
        cache_rope_positions=cache_rope_positions,
        indexer_dtype=dtype,
    )


# ------------------------------------------------------------------ tests
@pytest.mark.qsa_prepare
@pytest.mark.parametrize(
    "mrope,is_2d_positions,cache_rope_positions,state_size,batch", CASES
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_qsa_prepare_matches_torch_reference(
    mrope, is_2d_positions, cache_rope_positions, state_size, batch, dtype
) -> None:
    case = _case(mrope, is_2d_positions, cache_rope_positions, state_size, batch, dtype)
    _assert_matches(case.run_kernel(), case.run_reference())


@pytest.mark.qsa_prepare
@pytest.mark.parametrize(
    "mrope,is_2d_positions,cache_rope_positions,state_size,batch", CASES
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_qsa_prepare_matches_vllm(
    mrope, is_2d_positions, cache_rope_positions, state_size, batch, dtype
) -> None:
    """Cross-check the port against vLLM's own operator, when it is importable."""

    vllm_op = pytest.importorskip(
        "vllm.models.qwen4_exp.nvidia.ops.qsa_prepare",
        reason="vLLM is required for the cross-check",
    ).qsa_prepare

    case = _case(mrope, is_2d_positions, cache_rope_positions, state_size, batch, dtype)
    actual = case.run_kernel()

    outputs = case.outputs()
    main_q_out, main_gate_out = vllm_op(
        *case._launch_args(outputs),
        compress_ratio=COMPRESS_RATIO,
        mrope_section=MROPE_SECTION if case.mrope else None,
        rope_pos_offset=HEAD_DIM if case.cache_rope_positions else None,
        **case.main,
    )
    torch.cuda.synchronize()
    expected = (*outputs, main_q_out, main_gate_out, case.main["main_kv_cache"])
    _assert_matches(actual, expected)
