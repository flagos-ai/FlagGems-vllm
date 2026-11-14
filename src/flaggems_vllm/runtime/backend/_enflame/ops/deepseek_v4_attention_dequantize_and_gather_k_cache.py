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

from typing import Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner


def _default_scale_slots(nope_dim: int) -> int:
    return triton.cdiv(nope_dim, 64) + (1 if nope_dim % 64 == 0 else 0)


def _stat_total(gather_lens, num_reqs: int) -> int:
    """Total tokens to gather: the tuner key for the work size."""
    return num_reqs if gather_lens is None else int(gather_lens.shape[-1]) * num_reqs


def _stat_max(gather_lens) -> int:
    """Largest per-request gather length: the tuner key for the parallelism shape."""
    return 0 if gather_lens is None else int(gather_lens.shape[-1])


_DEFAULT_TUNE_CONFIGS = [triton.Config({}, num_warps=1)]


def _as_cache_2d(k_cache: torch.Tensor) -> torch.Tensor:
    if k_cache.ndim == 2:
        return k_cache
    if k_cache.ndim == 3:
        if k_cache.is_contiguous():
            return k_cache.view(k_cache.shape[0], -1)
        return k_cache.contiguous().view(k_cache.shape[0], -1)
    raise ValueError(f"k_cache must be 2D or 3D, got shape={tuple(k_cache.shape)}")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dequantize_and_gather_k_cache")
    or _DEFAULT_TUNE_CONFIGS,
    key=["num_tokens", "max_gather_len"],
    strategy=["log", "log"],
)
@triton.jit
def _dequantize_and_gather_k_cache_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    scale_slots: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    cache_block_stride: tl.constexpr,
    output_dim: tl.constexpr,
    n_quant_blocks: tl.constexpr,
    HAVE_GATHER_LENS: tl.constexpr,
    num_tokens: tl.constexpr,
    max_gather_len: tl.constexpr,
    num_workers: tl.constexpr,
    SCALE_ROWS: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
    NOPE_OVERREAD: tl.constexpr,
    SCALE_FULL: tl.constexpr,
    ROPE_EXACT: tl.constexpr,
    UNROLL: tl.constexpr,
):
    req_idx = tl.program_id(0)
    worker_idx = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    if HAVE_GATHER_LENS:
        gather_len = tl.load(gather_lens_ptr + req_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    rows = tl.arange(0, SCALE_ROWS)
    cols = tl.arange(0, quant_block)
    qoffs = rows[:, None] * quant_block + cols[None, :]
    qmask = qoffs < nope_dim
    soffs = tl.arange(0, SCALE_ROWS)
    smask = soffs < n_quant_blocks
    roffs = tl.arange(0, ROPE_BLOCK)
    rmask = roffs < rope_dim
    bt_row = block_table_ptr + req_idx * max_blocks_per_seq

    first_pos = start_pos + worker_idx
    pb = tl.load(
        bt_row + first_pos // cache_block_size,
        mask=worker_idx < gather_len,
        other=0,
    )

    if UNROLL == 2:
        pb_b = tl.load(
            bt_row + (first_pos + num_workers) // cache_block_size,
            mask=worker_idx + num_workers < gather_len,
            other=0,
        )
        last_pos = start_pos + gather_len - 1
        for local_i in range(worker_idx, gather_len, 2 * num_workers):
            pos_a = start_pos + local_i
            pib_a = pos_a % cache_block_size
            i_b = local_i + num_workers
            pos_b = tl.minimum(start_pos + i_b, last_pos)
            pib_b = pos_b % cache_block_size
            cb_a = k_cache_ptr + pb.to(tl.int64) * cache_block_stride
            td_a = cb_a + pib_a * token_data_size
            sb_a = cb_a + cache_block_size * token_data_size + pib_a * scale_slots
            cb_b = k_cache_ptr + pb_b.to(tl.int64) * cache_block_stride
            td_b = cb_b + pib_b * token_data_size
            sb_b = cb_b + cache_block_size * token_data_size + pib_b * scale_slots

            if NOPE_OVERREAD:
                x_a = tl.load(td_a + qoffs)
                x_b = tl.load(td_b + qoffs)
            else:
                x_a = tl.load(td_a + qoffs, mask=qmask, other=0)
                x_b = tl.load(td_b + qoffs, mask=qmask, other=0)
            if SCALE_FULL:
                s_a = tl.load(sb_a + soffs)
                s_b = tl.load(sb_b + soffs)
            else:
                s_a = tl.load(sb_a + soffs, mask=smask, other=0)
                s_b = tl.load(sb_b + soffs, mask=smask, other=0)
            rp_a = (td_a + nope_dim).to(tl.pointer_type(tl.bfloat16))
            rp_b = (td_b + nope_dim).to(tl.pointer_type(tl.bfloat16))
            if ROPE_EXACT:
                r_a = tl.load(rp_a + roffs)
                r_b = tl.load(rp_b + roffs)
            else:
                r_a = tl.load(rp_a + roffs, mask=rmask, other=0.0)
                r_b = tl.load(rp_b + roffs, mask=rmask, other=0.0)

            out_a = (
                out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1
            )
            xf_a = x_a.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_a = tl.exp2(s_a.to(tl.float32) - 127.0)
            tl.store(out_a + qoffs, (xf_a * sc_a[:, None]).to(tl.bfloat16), mask=qmask)
            if ROPE_EXACT:
                tl.store(out_a + nope_dim + roffs, r_a)
            else:
                tl.store(out_a + nope_dim + roffs, r_a, mask=rmask)

            if i_b < gather_len:
                out_b = (
                    out_ptr
                    + req_idx * out_stride0
                    + (offset + i_b) * out_stride1
                )
                xf_b = x_b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                sc_b = tl.exp2(s_b.to(tl.float32) - 127.0)
                tl.store(
                    out_b + qoffs,
                    (xf_b * sc_b[:, None]).to(tl.bfloat16),
                    mask=qmask,
                )
                if ROPE_EXACT:
                    tl.store(out_b + nope_dim + roffs, r_b)
                else:
                    tl.store(out_b + nope_dim + roffs, r_b, mask=rmask)

            step = 2 * num_workers
            pb = tl.load(
                bt_row + tl.minimum(pos_a + step, last_pos) // cache_block_size
            )
            pb_b = tl.load(
                bt_row
                + tl.minimum(pos_a + step + num_workers, last_pos) // cache_block_size
            )
    elif UNROLL == 4:
        pb_b = tl.load(
            bt_row + (first_pos + num_workers) // cache_block_size,
            mask=worker_idx + num_workers < gather_len,
            other=0,
        )
        pb_c = tl.load(
            bt_row + (first_pos + 2 * num_workers) // cache_block_size,
            mask=worker_idx + 2 * num_workers < gather_len,
            other=0,
        )
        pb_d = tl.load(
            bt_row + (first_pos + 3 * num_workers) // cache_block_size,
            mask=worker_idx + 3 * num_workers < gather_len,
            other=0,
        )
        last_pos = start_pos + gather_len - 1
        for local_i in range(worker_idx, gather_len, 4 * num_workers):
            pos_a = start_pos + local_i
            pib_a = pos_a % cache_block_size
            i_b = local_i + num_workers
            i_c = local_i + 2 * num_workers
            i_d = local_i + 3 * num_workers
            pos_b = tl.minimum(start_pos + i_b, last_pos)
            pos_c = tl.minimum(start_pos + i_c, last_pos)
            pos_d = tl.minimum(start_pos + i_d, last_pos)
            pib_b = pos_b % cache_block_size
            pib_c = pos_c % cache_block_size
            pib_d = pos_d % cache_block_size
            cb_a = k_cache_ptr + pb.to(tl.int64) * cache_block_stride
            td_a = cb_a + pib_a * token_data_size
            sb_a = cb_a + cache_block_size * token_data_size + pib_a * scale_slots
            cb_b = k_cache_ptr + pb_b.to(tl.int64) * cache_block_stride
            td_b = cb_b + pib_b * token_data_size
            sb_b = cb_b + cache_block_size * token_data_size + pib_b * scale_slots
            cb_c = k_cache_ptr + pb_c.to(tl.int64) * cache_block_stride
            td_c = cb_c + pib_c * token_data_size
            sb_c = cb_c + cache_block_size * token_data_size + pib_c * scale_slots
            cb_d = k_cache_ptr + pb_d.to(tl.int64) * cache_block_stride
            td_d = cb_d + pib_d * token_data_size
            sb_d = cb_d + cache_block_size * token_data_size + pib_d * scale_slots

            if NOPE_OVERREAD:
                x_a = tl.load(td_a + qoffs)
                x_b = tl.load(td_b + qoffs)
                x_c = tl.load(td_c + qoffs)
                x_d = tl.load(td_d + qoffs)
            else:
                x_a = tl.load(td_a + qoffs, mask=qmask, other=0)
                x_b = tl.load(td_b + qoffs, mask=qmask, other=0)
                x_c = tl.load(td_c + qoffs, mask=qmask, other=0)
                x_d = tl.load(td_d + qoffs, mask=qmask, other=0)
            if SCALE_FULL:
                s_a = tl.load(sb_a + soffs)
                s_b = tl.load(sb_b + soffs)
                s_c = tl.load(sb_c + soffs)
                s_d = tl.load(sb_d + soffs)
            else:
                s_a = tl.load(sb_a + soffs, mask=smask, other=0)
                s_b = tl.load(sb_b + soffs, mask=smask, other=0)
                s_c = tl.load(sb_c + soffs, mask=smask, other=0)
                s_d = tl.load(sb_d + soffs, mask=smask, other=0)
            rp_a = (td_a + nope_dim).to(tl.pointer_type(tl.bfloat16))
            rp_b = (td_b + nope_dim).to(tl.pointer_type(tl.bfloat16))
            rp_c = (td_c + nope_dim).to(tl.pointer_type(tl.bfloat16))
            rp_d = (td_d + nope_dim).to(tl.pointer_type(tl.bfloat16))
            if ROPE_EXACT:
                r_a = tl.load(rp_a + roffs)
                r_b = tl.load(rp_b + roffs)
                r_c = tl.load(rp_c + roffs)
                r_d = tl.load(rp_d + roffs)
            else:
                r_a = tl.load(rp_a + roffs, mask=rmask, other=0.0)
                r_b = tl.load(rp_b + roffs, mask=rmask, other=0.0)
                r_c = tl.load(rp_c + roffs, mask=rmask, other=0.0)
                r_d = tl.load(rp_d + roffs, mask=rmask, other=0.0)

            out_a = (
                out_ptr + req_idx * out_stride0 + (offset + local_i) * out_stride1
            )
            xf_a = x_a.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_a = tl.exp2(s_a.to(tl.float32) - 127.0)
            tl.store(out_a + qoffs, (xf_a * sc_a[:, None]).to(tl.bfloat16), mask=qmask)
            if ROPE_EXACT:
                tl.store(out_a + nope_dim + roffs, r_a)
            else:
                tl.store(out_a + nope_dim + roffs, r_a, mask=rmask)

            vb = i_b < gather_len
            out_b = out_ptr + req_idx * out_stride0 + (offset + i_b) * out_stride1
            xf_b = x_b.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_b = tl.exp2(s_b.to(tl.float32) - 127.0)
            tl.store(
                out_b + qoffs,
                (xf_b * sc_b[:, None]).to(tl.bfloat16),
                mask=qmask & vb,
            )
            if ROPE_EXACT:
                tl.store(out_b + nope_dim + roffs, r_b, mask=vb)
            else:
                tl.store(out_b + nope_dim + roffs, r_b, mask=rmask & vb)

            vc = i_c < gather_len
            out_c = out_ptr + req_idx * out_stride0 + (offset + i_c) * out_stride1
            xf_c = x_c.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_c = tl.exp2(s_c.to(tl.float32) - 127.0)
            tl.store(
                out_c + qoffs,
                (xf_c * sc_c[:, None]).to(tl.bfloat16),
                mask=qmask & vc,
            )
            if ROPE_EXACT:
                tl.store(out_c + nope_dim + roffs, r_c, mask=vc)
            else:
                tl.store(out_c + nope_dim + roffs, r_c, mask=rmask & vc)

            vd = i_d < gather_len
            out_d = out_ptr + req_idx * out_stride0 + (offset + i_d) * out_stride1
            xf_d = x_d.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            sc_d = tl.exp2(s_d.to(tl.float32) - 127.0)
            tl.store(
                out_d + qoffs,
                (xf_d * sc_d[:, None]).to(tl.bfloat16),
                mask=qmask & vd,
            )
            if ROPE_EXACT:
                tl.store(out_d + nope_dim + roffs, r_d, mask=vd)
            else:
                tl.store(out_d + nope_dim + roffs, r_d, mask=rmask & vd)

            step = 4 * num_workers
            pb = tl.load(
                bt_row + tl.minimum(pos_a + step, last_pos) // cache_block_size
            )
            pb_b = tl.load(
                bt_row
                + tl.minimum(pos_a + step + num_workers, last_pos) // cache_block_size
            )
            pb_c = tl.load(
                bt_row
                + tl.minimum(pos_a + step + 2 * num_workers, last_pos)
                // cache_block_size
            )
            pb_d = tl.load(
                bt_row
                + tl.minimum(pos_a + step + 3 * num_workers, last_pos)
                // cache_block_size
            )
    else:
        for local_i in range(worker_idx, gather_len, num_workers):
            pos = start_pos + local_i
            block_in_seq = pos // cache_block_size
            pos_in_block = pos - block_in_seq * cache_block_size
            physical_block = pb
            cache_block = (
                k_cache_ptr + physical_block.to(tl.int64) * cache_block_stride
            )
            token_data = cache_block + pos_in_block * token_data_size
            scale_base = (
                cache_block
                + cache_block_size * token_data_size
                + pos_in_block * scale_slots
            )
            out_row = (
                out_ptr
                + req_idx * out_stride0
                + (offset + local_i) * out_stride1
            )

            if NOPE_OVERREAD:
                x_u8 = tl.load(token_data + qoffs)
            else:
                x_u8 = tl.load(token_data + qoffs, mask=qmask, other=0)
            if SCALE_FULL:
                s_u8 = tl.load(scale_base + soffs)
            else:
                s_u8 = tl.load(scale_base + soffs, mask=smask, other=0)
            x_f32 = x_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale = tl.exp2(s_u8.to(tl.float32) - 127.0)
            x = x_f32 * scale[:, None]
            tl.store(out_row + qoffs, x.to(tl.bfloat16), mask=qmask)

            bf16_ptr = (token_data + nope_dim).to(tl.pointer_type(tl.bfloat16))
            if ROPE_EXACT:
                vals = tl.load(bf16_ptr + roffs)
                tl.store(out_row + nope_dim + roffs, vals)
            else:
                vals = tl.load(bf16_ptr + roffs, mask=rmask, other=0.0)
                tl.store(out_row + nope_dim + roffs, vals, mask=rmask)

            next_pos = pos + num_workers
            pb = tl.load(
                bt_row + next_pos // cache_block_size,
                mask=next_pos < start_pos + gather_len,
                other=0,
            )


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int = 0,
    rope_dim: int = 64,
    nope_dim: Optional[int] = None,
    scale_slots: Optional[int] = None,
) -> None:
    assert out.ndim == 3 and out.dtype == torch.bfloat16
    assert seq_lens.ndim == 1 and block_table.ndim == 2
    assert seq_lens.shape[0] == block_table.shape[0] <= out.shape[0]
    output_dim = out.shape[-1]
    if nope_dim is None:
        nope_dim = output_dim - rope_dim
    if scale_slots is None:
        scale_slots = _default_scale_slots(nope_dim)

    n_quant_blocks = triton.cdiv(nope_dim, 64)
    assert nope_dim + rope_dim <= output_dim
    k_cache_2d = _as_cache_2d(k_cache)
    token_data_size = nope_dim + rope_dim * 2
    num_reqs = seq_lens.shape[0]
    num_workers = max(1, -(-48 // num_reqs))
    est_tokens_per_worker = max(1, -(-(out.shape[1] - offset) // num_workers))
    unroll = 4 if est_tokens_per_worker >= 16 else 2
    scale_rows = triton.next_power_of_2(n_quant_blocks)
    rope_block = triton.next_power_of_2(rope_dim)
    with torch_device_fn.device(out.device):
        _dequantize_and_gather_k_cache_kernel[(num_reqs, num_workers)](
            out,
            out.stride(0),
            out.stride(1),
            k_cache_2d,
            seq_lens,
            block_table,
            offset,
            gather_lens,
            block_table.shape[-1],
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            scale_slots=scale_slots,
            quant_block=64,
            cache_block_size=block_size,
            token_data_size=token_data_size,
            cache_block_stride=k_cache_2d.stride(0),
            output_dim=output_dim,
            n_quant_blocks=n_quant_blocks,
            HAVE_GATHER_LENS=gather_lens is not None,
            num_tokens=_stat_total(gather_lens, num_reqs),
            max_gather_len=_stat_max(gather_lens),
            num_workers=num_workers,
            SCALE_ROWS=scale_rows,
            ROPE_BLOCK=rope_block,
            NOPE_OVERREAD=scale_rows * 64 <= token_data_size,
            SCALE_FULL=scale_slots >= scale_rows,
            ROPE_EXACT=rope_dim == rope_block,
            UNROLL=unroll,
        )


__all__ = ["dequantize_and_gather_k_cache"]
