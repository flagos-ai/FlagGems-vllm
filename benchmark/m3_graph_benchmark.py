# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Synthetic two-round CUDA Graph benchmark; invoke as a standalone script."""

import argparse
import importlib
import json
import pathlib
import statistics
import time

import torch

import flaggems_vllm as fg

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    type=pathlib.Path,
    default=pathlib.Path("m3-operator-graph-results.json"),
)
OUTPUT = parser.parse_args().output
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(59)
torch.backends.cuda.matmul.allow_tf32 = False


moe = importlib.import_module("flaggems_vllm.ops.fused_moe")


def timing(fn):
    for _ in range(25):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(100):
            fn()
    samples = []
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def compare(op, shape, baseline, candidate, reference):
    rows = []
    for number in (1, 2):
        order = (
            [("baseline", baseline), ("candidate", candidate)]
            if number == 1
            else [("candidate", candidate), ("baseline", baseline)]
        )
        row = {"operator": op, "shape": shape, "round": number, "reference": reference}
        for name, fn in order:
            row[name] = timing(fn)
        row["speedup"] = row["baseline"]["median_us"] / row["candidate"]["median_us"]
        rows.append(row)
        results["measurements"].append(row)
        OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(op, shape, "speedup", [round(row["speedup"], 3) for row in rows], flush=True)


results = {
    "gpu": torch.cuda.get_device_name(),
    "torch_version": torch.__version__,
    "method": "CUDA Graph; 25 warmup calls; capture 100 calls; 5 event samples; two rounds with reversed order",
    "measurements": [],
}
for m in (1, 4, 8, 16, 32):
    x = torch.randn((m, 6144), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((128, 6144), device="cuda") * 0.01
    expected = torch.nn.functional.linear(x.float(), w)
    actual = fg.fp32_router_gemm(x, w)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
    compare(
        "fp32_router_gemm",
        [m, 6144, 128],
        lambda: torch.nn.functional.linear(x.float(), w),
        lambda: fg.fp32_router_gemm(x, w),
        "literal FP32 F.linear, TF32 disabled",
    )
for m, k in ((1, 6144), (64, 6144), (4096, 6144), (5089, 768), (8192, 768)):
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)

    def baseline():
        s = x.abs().amax(-1, keepdim=True).clamp(min=1e-10).float() / 127
        return (x.float() / s).round().clamp(-128, 127).to(torch.int8), s

    actual = moe._int8_quantize_per_token_triton(x)
    expected = baseline()
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    compare(
        "per_token_int8_quant",
        [m, k],
        baseline,
        lambda: moe._int8_quantize_per_token_triton(x),
        "complete plain Torch quantization chain, BF16",
    )
for b, s, h, d in ((1, 128, 16, 64), (1, 512, 16, 80), (2, 256, 16, 80)):
    q = torch.randn((b, s, h, d), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def baseline():
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def candidate():
        return fg.vit_flash_attn_wrapper(q, k, v, b)

    torch.testing.assert_close(candidate(), baseline(), rtol=0.016, atol=0.016)
    compare(
        "vit_attention",
        [b, s, h, d],
        baseline,
        candidate,
        "Torch optimized SDPA; adapter uses existing FlagGems-vllm varlen attention",
    )
importlib.import_module("vllm._C_stable_libtorch")

native = torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert
for m in (1, 4, 64, 4096, 8192):
    qkv = torch.randn((m, 11 * 128), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((128,), device="cuda", dtype=torch.bfloat16) * 0.1
    cs = torch.randn((max(m, 1), 64), device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(m, device="cuda", dtype=torch.int64)
    slots = torch.arange(m, device="cuda", dtype=torch.int64)
    cache = torch.empty(
        (max(1, (m + 15) // 16), 2, 16, 2, 128), device="cuda", dtype=torch.uint8
    )
    idx = torch.empty((max(m, 1), 128), device="cuda", dtype=torch.bfloat16)
    qo = torch.empty((m, 4 * 128), device="cuda", dtype=torch.bfloat16)
    iqo = torch.empty((m, 2 * 128), device="cuda", dtype=torch.bfloat16)
    args = [
        qkv,
        w,
        w,
        cs,
        pos,
        4,
        2,
        64,
        1e-6,
        w,
        w,
        2,
        slots,
        slots,
        cache,
        idx,
        16,
        qo,
        iqo,
        "fp8_e4m3",
    ]
    original = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
    native(*original)
    fg.fused_minimax_m3_qknorm_rope_kv_insert(*args)
    for i in (0, 14, 15, 17, 18):
        assert torch.equal(args[i], original[i])

    def baseline():
        local = args.copy()
        local[0] = qkv.clone()
        native(*local)

    def candidate():
        local = args.copy()
        local[0] = qkv.clone()
        fg.fused_minimax_m3_qknorm_rope_kv_insert(*local)

    compare(
        "fused_minimax_m3_qknorm_rope_kv_insert",
        [m, 4, 2, 2, 128],
        baseline,
        candidate,
        "native vLLM M3 CUDA fused cache writer; identical clone/reset cost included",
    )

results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
