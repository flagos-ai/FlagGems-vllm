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

"""Reference implementations for the QSA fused prepare cross-checks.

On CUDA the reference is vLLM's native fused operator, resolved through the
merged ``baseline.qsa_prepare`` module. Ascend has no single equivalent
operator, so the reference composes the vllm-ascend operators covering the same
work: the main
split/norm/rope kernel, the indexer norm+RoPE (fused on newer vendor builds,
``triton_qsa_rmsnorm_mrope``), the group compressor and
the row-store kernel for every cache write. Only metadata preparation and the
``.contiguous()`` the vendor kernels require stay on torch.

Both the functional cross-check test and the benchmark import this module so the
reference and the performance denominator stay identical.
"""

from __future__ import annotations

import os

import torch

import flaggems_vllm

# vLLM derives the token->request map once per step and shares it across layers,
# so the benchmark must not charge each layer for it.  Cache on the identity of
# ``query_start_loc``, which the benchmark reuses for a given shape.
_TOKEN_TO_REQ_CACHE: dict[tuple, torch.Tensor] = {}


def _cached_token_to_req(
    query_start_loc: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    key = (query_start_loc.data_ptr(), tuple(query_start_loc.shape), num_tokens)
    cached = _TOKEN_TO_REQ_CACHE.get(key)
    if cached is not None:
        return cached
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    built = torch.repeat_interleave(
        torch.arange(query_lens.numel(), device=query_start_loc.device),
        query_lens,
        output_size=num_tokens,
    ).to(torch.int32)
    if len(_TOKEN_TO_REQ_CACHE) > 16:
        _TOKEN_TO_REQ_CACHE.clear()
    _TOKEN_TO_REQ_CACHE[key] = built
    return built


FP8 = torch.float8_e4m3fn


def _import_indexer_norm_rope():
    """The fused indexer norm+RoPE kernel, or ``None`` if the build lacks it."""

    try:
        from vllm_ascend.ops.triton.linearnorm.qsa_rmsnorm_mrope import (
            triton_qsa_rmsnorm_mrope,
        )
    except ImportError:
        return None
    return triton_qsa_rmsnorm_mrope


def _import_common_vendor_ops():
    try:
        from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import (
            triton_split_qkv_rmsnorm_mrope,
        )
        from vllm_ascend.ops.triton.qwen4_exp.qsa import (
            qsa_compress_groups_with_ratio,
            qsa_store_cache_rows,
        )
    except ImportError:
        return None
    try:  # the vendor kernels read device properties filled in by this helper
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        init_device_properties_triton()
    except Exception:  # noqa: BLE001 - optional convenience on newer builds
        pass
    return (
        triton_split_qkv_rmsnorm_mrope,
        qsa_compress_groups_with_ratio,
        qsa_store_cache_rows,
    )


def build_ascend_baseline(vendor_io: bool = True):
    """Compose the vllm-ascend operators that cover the fused prepare work.

    Returns ``None`` when the vendor build lacks the required operators, so the
    caller can skip instead of failing.  ``vendor_io`` selects the vendor
    row-store kernel for the cache writes; otherwise torch indexing is used,
    which measures slower and is only kept for reference.
    """

    if flaggems_vllm.vendor_name != "ascend":
        return None
    common = _import_common_vendor_ops()
    if common is None:
        return None
    (
        triton_split_qkv_rmsnorm_mrope,
        qsa_compress_groups_with_ratio,
        qsa_store_cache_rows,
    ) = common
    norm_rope_impl = _import_indexer_norm_rope()
    if norm_rope_impl is None:
        return None

    def _token_positions(positions, logical_positions, mrope):
        """``[tokens, 3]`` MRoPE row of every token."""

        if positions.ndim == 2:
            rows = positions
        else:
            rows = logical_positions.expand(3, -1)
        if not mrope:
            rows = rows[:1].expand(3, -1)
        return rows.transpose(0, 1).to(torch.int64).contiguous()

    def _scatter_rows(dst, slot_mapping, rows, width, col_offset=0):
        """Write ``rows`` into ``dst[slot // page, slot % page, :, col:+width]``."""

        valid = slot_mapping >= 0
        if not bool(valid.any()):
            return
        slots = slot_mapping[valid]
        page = dst.shape[1]
        block = torch.div(slots, page, rounding_mode="floor")
        offset = slots - block * page
        dst[block, offset, :, col_offset : col_offset + width] = rows[valid]

    def _store_rows_vendor(dst, slot_mapping, rows, width, col_offset=0):
        """Same write through ``qsa_store_cache_rows``, one head at a time.

        The vendor kernel requires ``cache.shape[2] == 1``, so each head goes
        through its own ``[blocks, page, 1, width]`` view.
        """

        for head in range(rows.shape[1]):
            qsa_store_cache_rows(
                dst[:, :, head : head + 1, col_offset : col_offset + width],
                slot_mapping,
                rows[:, head : head + 1, :],
            )

    def _position_views(positions, n_rows):
        """Return the axis-major view of a position tensor.

        Vendor releases disagree on the layout of the compressed-group
        positions: older builds hand over ``[3, groups]`` while newer ones give
        ``[groups, 3]``.  The fused kernel gathers along the axis-major form
        (``cos_sin.shape[-2]`` must equal the row count).
        """

        if positions.ndim == 1:
            return positions
        if positions.shape[-1] == 3 and positions.shape[0] == n_rows:
            return positions.transpose(0, 1)
        if positions.shape[0] == 3 and positions.shape[1] == n_rows:
            return positions
        raise ValueError(
            f"position tensor shape {tuple(positions.shape)} does not match {n_rows} rows"
        )

    def _indexer_norm_rope(
        tensor,
        weight,
        cos_sin_cache,
        positions,
        eps,
        rope_dim,
        mrope_section,
    ):
        """Norm+RoPE of the indexer tensors through the available vendor path."""

        axis_major = _position_views(positions, tensor.shape[0])
        cos_sin = cos_sin_cache[axis_major.to(torch.int64)]
        sections = (
            list(mrope_section)
            if (mrope_section is not None and cos_sin.ndim == 3)
            else [0, 0, 0]
        )
        return norm_rope_impl(tensor, weight, cos_sin, eps, sections, rope_dim)

    def qsa_prepare_ascend_composed(
        q,
        k,
        positions,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        state_cache,
        state_slots,
        state_block_table,
        query_start_loc,
        logical_positions,
        compressed_cache,
        compressed_slots,
        k_work_metadata,
        *,
        compress_ratio,
        mrope_section,
        rope_pos_offset,
        main_qkv,
        main_q_norm_weight,
        main_k_norm_weight,
        main_eps,
        main_kv_cache,
        main_slot_mapping,
        main_k_scale,
        main_v_scale,
    ):
        _ = k_work_metadata, rope_pos_offset, main_k_scale, main_v_scale
        if q_out.dtype in (FP8, torch.float8_e5m2):
            raise NotImplementedError("composed baseline covers bf16 only")

        num_tokens = q.shape[0]
        head_dim = q.shape[-1] // q_out.shape[1]
        num_indexer_heads = q_out.shape[1]
        num_kv_heads = main_kv_cache.shape[2]
        main_dim = main_kv_cache.shape[-1] // 2
        num_q_heads = main_qkv.shape[1] // (2 * main_dim) - num_kv_heads
        mrope = mrope_section is not None
        rope_dim = 64 if head_dim >= 128 else head_dim // 2

        tpos = _token_positions(positions, logical_positions, mrope)
        # --- main attention -------------------------------------------------
        if positions.ndim == 2:
            main_cos_sin = cos_sin_cache[positions]
            main_sections = list(mrope_section)
        else:
            main_cos_sin = (
                cos_sin_cache[positions].unsqueeze(0).expand(3, -1, -1).contiguous()
            )
            main_sections = [head_dim // 4, 0, 0]
        # The vendor operator reads a contiguous [tokens, 2*(Q+KV)*head] buffer;
        # callers may hand over a projection slice, so normalise it first.
        if not main_qkv.is_contiguous():
            main_qkv = main_qkv.contiguous()
        q_main, k_main, v_main, gate_main = triton_split_qkv_rmsnorm_mrope(
            qkv=main_qkv,
            q_weight=1.0 + main_q_norm_weight,
            k_weight=1.0 + main_k_norm_weight,
            cos_sin=main_cos_sin,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=main_dim,
            eps=main_eps,
            mrope_section=main_sections,
            is_interleaved=True,
            rope_dim=rope_dim,
            has_gate=True,
        )
        main_q_out = q_main.reshape(num_tokens, num_q_heads, main_dim).contiguous()
        main_gate_out = gate_main.reshape(
            num_tokens, num_q_heads, main_dim
        ).contiguous()
        k_rows = k_main.reshape(num_tokens, num_kv_heads, main_dim)
        v_rows = v_main.reshape(num_tokens, num_kv_heads, main_dim)
        if vendor_io:
            _store_rows_vendor(main_kv_cache, main_slot_mapping, k_rows, main_dim)
            _store_rows_vendor(
                main_kv_cache, main_slot_mapping, v_rows, main_dim, main_dim
            )
        else:
            _scatter_rows(main_kv_cache, main_slot_mapping, k_rows, main_dim)
            _scatter_rows(main_kv_cache, main_slot_mapping, v_rows, main_dim, main_dim)

        # --- indexer Q ------------------------------------------------------
        q_roped = _indexer_norm_rope(
            q.reshape(num_tokens, num_indexer_heads, head_dim),
            q_norm_weight,
            cos_sin_cache,
            positions,
            eps,
            rope_dim,
            list(mrope_section) if mrope else None,
        )
        q_out.copy_(q_roped.reshape(q_out.shape).to(q_out.dtype))

        # --- indexer K ------------------------------------------------------
        state_keys = state_cache[..., :head_dim]
        rope_tail = None
        if state_cache.shape[-1] > head_dim:
            rope_tail = (
                state_cache[..., head_dim:].view(torch.int64)[..., :3].contiguous()
            )
        token_to_req = _cached_token_to_req(query_start_loc, num_tokens)
        pooled, first_positions = qsa_compress_groups_with_ratio(
            k.reshape(num_tokens, 1, head_dim),
            tpos.view(num_tokens, 1, 3),
            state_keys,
            state_block_table,
            token_to_req,
            query_start_loc.to(torch.int32),
            logical_positions,
            compressed_slots,
            compress_ratio,
            rope_tail,
        )
        compressed = _indexer_norm_rope(
            pooled.reshape(-1, 1, head_dim),
            k_norm_weight,
            cos_sin_cache,
            first_positions,
            eps,
            rope_dim,
            list(mrope_section) if mrope else None,
        )
        qsa_store_cache_rows(
            compressed_cache,
            compressed_slots,
            compressed.reshape(-1, 1, head_dim).to(compressed_cache.dtype),
        )

        # --- raw ring -------------------------------------------------------
        qsa_store_cache_rows(
            state_keys, state_slots, k.reshape(num_tokens, 1, head_dim)
        )
        if rope_tail is not None:
            if vendor_io:
                qsa_store_cache_rows(
                    state_cache[..., head_dim:].view(torch.int64)[..., :3],
                    state_slots,
                    tpos,
                )
            else:
                valid = state_slots >= 0
                if bool(valid.any()):
                    slots = state_slots[valid]
                    page = state_cache.shape[1]
                    block = torch.div(slots, page, rounding_mode="floor")
                    offset = slots - block * page
                    bits = tpos[valid].view(torch.int16).view(torch.bfloat16)
                    state_cache[block, offset, 0, head_dim:] = bits

        return main_q_out, main_gate_out

    return qsa_prepare_ascend_composed


def vendor_reference():
    """Platform-appropriate vendor reference, or ``None`` when unavailable.

    CUDA builds compare against vLLM's native fused operator. Ascend builds
    compare against the composed vendor pipeline. Other backends return ``None``
    so the cross-check test can skip.
    """

    if flaggems_vllm.vendor_name == "ascend":
        io = os.environ.get("QSA_ASCEND_BASELINE_IO", "vendor").strip().lower()
        return build_ascend_baseline(io != "torch")
    # The merged `baseline/qsa_prepare.py` already resolves the vLLM native
    # operator; reuse it instead of duplicating the import here.
    try:
        from baseline.qsa_prepare import native as native_baseline
    except Exception:  # noqa: BLE001 - import failure means "not available"
        return None
    return native_baseline()
