"""Reproducible Torch-reference vs Triton benchmark for Qwen4 self kernels.

The script is checkpoint-free and model-wiring-free. Correctness is checked
before any timing. Results are JSON with CUDA-event latency, GB/s where useful,
and Torch-over-Triton speedup.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

from flaggems_vllm.ops.qwen4.hyperconnection import (
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
)
from flaggems_vllm.ops.qwen4.ple_state import ple_state_gather, ple_state_scatter_
from flaggems_vllm.ops.qwen4.qsa import (
    qwen4_compress_norm_mrope_store_groups,
    qwen4_qsa_mqa_paged_dot,
    qwen4_store_qsa_kv_rows,
)

HC = 4
HIDDEN = 2560
EPS = 1.0e-6
QSA_DIM = 128
QSA_HEADS = 4
QSA_RATIO = 4
PAGE_SIZE = 16


def _event_us(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    return float(statistics.median(samples))


def _hc_inputs(rows: int, device: torch.device):
    torch.manual_seed(1000 + rows)
    dtype = torch.bfloat16
    x = torch.randn((rows, HC * HIDDEN), device=device, dtype=dtype)
    weight = torch.randn((HC * HIDDEN,), device=device, dtype=dtype) * 0.01
    logits = torch.randn_like(x)
    normed = torch.randn_like(x)
    injection = torch.randn((rows, HC), device=device, dtype=dtype)
    block = torch.randn((rows, HIDDEN), device=device, dtype=dtype)
    residual = torch.randn_like(x)
    return x, weight, logits, normed, injection, block, residual


def _hc_refs(inputs):
    x, weight, logits, normed, injection, block, residual = inputs
    x3 = x.reshape(-1, HC, HIDDEN).float()
    w2 = weight.reshape(HC, HIDDEN).float()
    norm = x3 * torch.rsqrt(x3.square().mean(-1, keepdim=True) + EPS) * (1.0 + w2)
    gate = (
        torch.sigmoid(logits.float().reshape(-1, HC, HIDDEN))
        * normed.float().reshape(-1, HC, HIDDEN)
    ).mean(-2)
    alpha = 2.0 * torch.sigmoid(injection.float() / HC)
    combine = residual.float().reshape(-1, HC, HIDDEN) + block.float().unsqueeze(
        -2
    ) * alpha.unsqueeze(-1)
    return (
        norm.to(x.dtype).reshape_as(x),
        gate.to(normed.dtype),
        combine.to(residual.dtype).reshape_as(residual),
    )


def _qsa_inputs(rows: int, device: torch.device):
    torch.manual_seed(2000 + rows)
    cache = torch.randn(
        (rows * 4, PAGE_SIZE, 1, QSA_DIM), device=device, dtype=torch.bfloat16
    )
    q = torch.randn((rows, QSA_HEADS, QSA_DIM), device=device, dtype=torch.bfloat16)
    page_table = torch.arange(rows * 4, device=device, dtype=torch.int32).reshape(
        rows, 4
    )
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    positions = torch.full((rows,), 63, device=device, dtype=torch.int64)
    lengths = torch.full((rows,), 64, device=device, dtype=torch.int64)
    return q, cache, page_table, requests, positions, lengths


def _qsa_ref(inputs):
    q, cache, table, requests, positions, lengths = inputs
    columns = table.shape[1] * cache.shape[1]
    req = requests.to(torch.int64)
    safe_req = req.clamp(0, table.shape[0] - 1)
    visible = torch.minimum(
        (positions + 1) // QSA_RATIO, lengths.index_select(0, safe_req) // QSA_RATIO
    )
    col = torch.arange(columns, device=q.device, dtype=torch.int64)
    page = col // cache.shape[1]
    token = col % cache.shape[1]
    physical = table[safe_req[:, None], page[None, :]].to(torch.int64)
    valid = (
        (col[None, :] < visible[:, None])
        & (physical >= 0)
        & (physical < cache.shape[0])
    )
    keys = cache[physical.clamp(0, cache.shape[0] - 1), token[None, :], 0].float()
    logits = torch.relu(torch.matmul(q.float(), keys.transpose(1, 2))).sum(
        1
    ) / math.sqrt(QSA_DIM)
    return logits.masked_fill(~valid, -float("inf")), visible.to(torch.int32)


def _kv_inputs(rows: int, device: torch.device):
    torch.manual_seed(3000 + rows)
    blocks = (rows + PAGE_SIZE - 1) // PAGE_SIZE + 2
    cache = torch.randn(
        (blocks, PAGE_SIZE, 1, QSA_DIM), device=device, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(cache)
    key = torch.randn((rows, 1, QSA_DIM), device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    return cache, value_cache, slots, key, value


def _kv_ref(inputs):
    k_cache, v_cache, slots, key, value = inputs
    block = slots // k_cache.shape[1]
    token = slots % k_cache.shape[1]
    k_cache.index_put_((block, token), key, accumulate=False)
    v_cache.index_put_((block, token), value, accumulate=False)
    return k_cache, v_cache


def _compress_inputs(rows: int, device: torch.device):
    torch.manual_seed(4000 + rows)
    page_size = 8
    raw = torch.randn((2, page_size, 1, QSA_DIM), device=device, dtype=torch.bfloat16)
    table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    requests = torch.zeros((rows,), device=device, dtype=torch.int32)
    positions = torch.arange(rows, device=device, dtype=torch.int64) % 8
    positions = positions + 3
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    compressed_blocks = (rows + page_size - 1) // page_size + 2
    compressed = torch.full(
        (compressed_blocks, page_size, 1, QSA_DIM),
        7,
        device=device,
        dtype=torch.bfloat16,
    )
    weight = torch.randn((QSA_DIM,), device=device, dtype=torch.bfloat16) * 0.01
    rope = torch.empty((2, page_size, 1, 3), device=device, dtype=torch.int64)
    base = torch.arange(page_size, device=device).view(1, page_size, 1)
    rope[..., 0] = base % 16
    rope[..., 1] = (base + 1) % 16
    rope[..., 2] = (base + 2) % 16
    freq = torch.arange(QSA_DIM // 4, device=device, dtype=torch.float32)
    t = torch.arange(16, device=device, dtype=torch.float32)
    angles = t[:, None] * (1.0 / (10000.0 ** (freq[None, :] / 32.0)))
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)
    return raw, table, requests, positions, slots, compressed, weight, cos_sin, rope


def _compress_ref(inputs):
    raw, table, requests, positions, slots, out, weight, cos_sin, rope = inputs
    result = out.clone()
    for row in range(requests.numel()):
        end = int(positions[row])
        pieces = []
        for offset in range(QSA_RATIO):
            pos = end - (QSA_RATIO - 1 - offset)
            page = int(table[0, pos // raw.shape[1]])
            pieces.append(raw[page, pos % raw.shape[1], 0].float())
        pooled = (torch.stack(pieces).sum(0) / QSA_RATIO).to(torch.bfloat16)
        pooled_fp32 = pooled.float()
        norm = (
            pooled_fp32
            * torch.rsqrt(pooled_fp32.square().mean() + EPS)
            * (weight.float() + 1.0)
        ).to(torch.bfloat16)
        first_pos = end - QSA_RATIO + 1
        rope_page = int(table[0, first_pos // raw.shape[1]])
        axes = rope[rope_page, first_pos % raw.shape[1], 0, :3]
        freq = torch.arange(QSA_DIM // 4, device=raw.device)
        use_h = (freq % 3 == 1) & (freq < 33)
        use_w = (freq % 3 == 2) & (freq < 30)
        axis = torch.where(use_h, axes[1], torch.where(use_w, axes[2], axes[0]))
        cos = cos_sin[axis, freq].float()
        sin = cos_sin[axis, 32 + freq].float()
        first, second = norm[:32].float(), norm[32:64].float()
        stored = torch.cat(
            (
                (first * cos - second * sin).to(torch.bfloat16),
                (second * cos + first * sin).to(torch.bfloat16),
                norm[64:],
            )
        )
        block, token = int(slots[row]) // out.shape[1], int(slots[row]) % out.shape[1]
        result[block, token, 0].copy_(stored)
    return result


def _ple_inputs(rows: int, device: torch.device):
    torch.manual_seed(5000 + rows)
    cache_rows, hidden, width = 8192, 3, 65
    storage = torch.randn(
        (cache_rows, width, hidden), device=device, dtype=torch.bfloat16
    )
    state = storage.transpose(1, 2)
    indices = torch.arange(1, rows + 1, device=device, dtype=torch.int64)
    write_mask = torch.ones((rows,), device=device, dtype=torch.bool)
    return state, indices, write_mask


def _ple_gather_ref(state, indices):
    valid = (indices >= 0) & (indices < state.shape[0])
    bounded = indices.clamp(0, state.shape[0] - 1)
    values = torch.ops.aten.index_select.default(state, 0, bounded)
    return torch.where(valid[:, None, None], values, torch.zeros_like(values))


def _ple_scatter_ref(state, indices, rows, write_mask):
    result = state.clone()
    for row in range(indices.numel()):
        index = int(indices[row])
        if bool(write_mask[row]) and 0 <= index < state.shape[0]:
            result[index].copy_(rows[row])
    return result


def _record(name, ref_fn, candidate_fn, compare_fn, warmup, iters, bytes_moved=0):
    ref = ref_fn()
    candidate = candidate_fn()
    compare_fn(ref, candidate)
    torch_us = _event_us(ref_fn, warmup, iters)
    triton_us = _event_us(candidate_fn, warmup, iters)
    result = {
        "kernel": name,
        "correctness": True,
        "torch_us": torch_us,
        "triton_us": triton_us,
        "speedup_torch_over_triton": torch_us / triton_us,
    }
    if bytes_moved:
        result["triton_gbps"] = bytes_moved / (triton_us * 1e-6) / 1e9
    return result


def _run(rows: int, device: torch.device, warmup: int, iters: int):
    results = []
    hc = _hc_inputs(rows, device)
    results.append(
        _record(
            "_grouped_gemma_rmsnorm_kernel",
            lambda: _hc_refs(hc)[0],
            lambda: qwen4_grouped_gemma_rmsnorm(hc[0], hc[1], HC, EPS),
            lambda r, c: torch.testing.assert_close(r, c, atol=0.03, rtol=0.02),
            warmup,
            iters,
        )
    )
    results.append(
        _record(
            "_hc_gate_reduce_kernel",
            lambda: _hc_refs(hc)[1],
            lambda: qwen4_hc_gate_reduce(hc[2], hc[3], HC),
            lambda r, c: torch.testing.assert_close(r, c, atol=0.03, rtol=0.02),
            warmup,
            iters,
        )
    )
    results.append(
        _record(
            "_hc_inject_combine_kernel",
            lambda: _hc_refs(hc)[2],
            lambda: qwen4_hc_inject_combine(hc[4], hc[5], hc[6], HC),
            lambda r, c: torch.testing.assert_close(r, c, atol=0.03, rtol=0.02),
            warmup,
            iters,
        )
    )
    qsa = _qsa_inputs(rows, device)
    results.append(
        _record(
            "_qsa_mqa_paged_dot_kernel",
            lambda: _qsa_ref(qsa),
            lambda: qwen4_qsa_mqa_paged_dot(*qsa, compress_ratio=QSA_RATIO),
            lambda r, c: (
                torch.testing.assert_close(r[0], c[0], atol=0.2, rtol=0.02),
                torch.testing.assert_close(r[1], c[1], atol=0, rtol=0),
            ),
            warmup,
            iters,
        )
    )
    kv = _kv_inputs(rows, device)
    bytes_kv = (
        2 * rows * QSA_DIM * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    results.append(
        _record(
            "_store_qsa_kv_rows_kernel",
            lambda: _kv_ref((kv[0].clone(), kv[1].clone(), kv[2], kv[3], kv[4])),
            lambda: (
                qwen4_store_qsa_kv_rows(*kv),
                kv[0],
                kv[1],
            )[1:],
            lambda r, c: (
                torch.testing.assert_close(r[0], c[0], atol=0, rtol=0),
                torch.testing.assert_close(r[1], c[1], atol=0, rtol=0),
            ),
            warmup,
            iters,
            bytes_kv,
        )
    )
    comp = _compress_inputs(rows, device)
    results.append(
        _record(
            "_compress_norm_mrope_store_qsa_groups_kernel",
            lambda: _compress_ref(comp),
            lambda: (
                qwen4_compress_norm_mrope_store_groups(
                    comp[0],
                    comp[1],
                    comp[2],
                    comp[3],
                    comp[4],
                    comp[5],
                    comp[6],
                    comp[7],
                    compress_ratio=QSA_RATIO,
                    norm_eps=EPS,
                    rotary_dim=64,
                    mrope_section=(11, 11, 10),
                    mrope_interleaved=True,
                    rope_cache=comp[8],
                ),
                comp[5],
            )[1],
            lambda r, c: torch.testing.assert_close(r, c, atol=0.03, rtol=0.02),
            warmup,
            iters,
        )
    )
    ple = _ple_inputs(rows, device)
    results.append(
        _record(
            "_ple_state_gather_kernel_3d",
            lambda: _ple_gather_ref(ple[0], ple[1]),
            lambda: ple_state_gather(ple[0], ple[1]),
            lambda r, c: torch.testing.assert_close(r, c, atol=0, rtol=0),
            warmup,
            iters,
        )
    )
    scatter_rows = torch.randn((rows, 3, 65), device=device, dtype=torch.bfloat16)
    results.append(
        _record(
            "_ple_state_scatter_kernel_3d",
            lambda: _ple_scatter_ref(ple[0], ple[1], scatter_rows, ple[2]),
            lambda: (
                ple_state_scatter_(ple[0], ple[1], scatter_rows, write_mask=ple[2]),
                ple[0],
            )[1],
            lambda r, c: torch.testing.assert_close(r, c, atol=0, rtol=0),
            warmup,
            iters,
        )
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows", default="1,8,64")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen4 kernel benchmark")
    device = torch.device(args.device)
    rows = [int(value) for value in args.rows.split(",") if value]
    payload = {
        "status": "pass",
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "results": [],
    }
    for row_count in rows:
        payload["results"].append(
            {
                "rows": row_count,
                "kernels": _run(row_count, device, args.warmup, args.iters),
            }
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
