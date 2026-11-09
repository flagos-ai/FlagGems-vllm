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

"""Standalone benchmark: BNSD FlashAttention router vs npu_fusion_attention.

A CLI sweep over the clc.yaml grid, filtered to the kernel's hard constraints
(fp16, D == 64, symmetric Sq == Skv, S % 128 == 0, Hq % Hkv == 0). For each
compatible shape it reports the shipping router's wall-clock latency, speedup
against the vendor fused op (`torch_npu.npu_fusion_attention` ->
aclnnFlashAttentionScoreV4), TFLOPS, bandwidth, and an fp32 accuracy check,
and writes a CSV with a fixed schema.

This is a measurement tool, not a pytest accuracy gate (that lives in
tests/test_flash_attention_bnsd.py). Run it on an idle card:

    ASCEND_RT_VISIBLE_DEVICES=7 python benchmark/bench_flash_attention_bnsd.py \
        --csv bench_fa_bnsd.csv

    # a quick subset while iterating:
    ASCEND_RT_VISIBLE_DEVICES=7 python benchmark/bench_flash_attention_bnsd.py \
        --filter-batch 1,4 --filter-seqlen 2048,4096

The causal vendor baseline uses sparse_mode=2 (leftUpCausal) with the
compressed 2048x2048 mask, not sparse_mode=0 with a full S x S mask: mode 0 is
1.5-2.2x slower for identical output and would overstate the speedup.
"""

import argparse
import csv
import math
import os
import time

import torch

import torch_npu  # noqa: F401  registers the npu device + npu_fusion_attention

from flaggems_vllm.runtime.backend._ascend.ops import flash_attention_fwd

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# clc.yaml sweep grid
# ---------------------------------------------------------------------------
#   batches:      [1, 4, 8, 16, 32]
#   seqlen_pairs: [[32, 8192], [256, 256], [1024, 1024], [2048, 2048],
#                  [4096, 4096], [8192, 8192], [16384, 16384]]
#   head_dims:    [64, 96, 128, [192, 128]]
#   head_pairs:   [[16, 16], [16, 8], [16, 4], [16, 1]]
#   causal:       [true, false]
# Filtered to DIM == 64 and symmetric Sq == Skv -> 240 compatible cases.
_CLC_BATCHES = [1, 4, 8, 16, 32]
_CLC_SEQLEN_PAIRS = [
    (32, 8192),       # asymmetric: decode-like (skipped: Sq != Skv)
    (256, 256),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
_CLC_HEAD_DIMS = [
    (64, 64),         # standard (the only one the kernel supports)
    (96, 96),         # skipped: DIM must be 64
    (128, 128),       # skipped
    (192, 128),       # skipped
]
_CLC_HEAD_PAIRS = [
    (16, 16),   # MHA
    (16, 8),    # GQA 2:1
    (16, 4),    # GQA 4:1
    (16, 1),    # MQA
]
_CLC_CAUSAL = [True, False]

# The shipping router's hard constraints. Unlike the retired hand-written CV
# kernel there is no ring/sub-vec/combine-batch state to validate -- the router
# picks its own tile and pipeline config internally.
_KERNEL_DIM = 64
_BLOCK = 128      # both BLOCK_M and BLOCK_N on 950DT
_NUM_CORES = 32
_CONFIG_LABEL = "M128/N128 (router auto-tunes dense M64/N256 at S>=4096)"


def _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv):
    reasons = []
    if Dq != _KERNEL_DIM or Dkv != _KERNEL_DIM:
        reasons.append(f"DIM must be {_KERNEL_DIM}, got Dq={Dq} Dkv={Dkv}")
    if Sq % _BLOCK != 0:
        reasons.append(f"Sq={Sq} not divisible by BLOCK={_BLOCK}")
    if Skv % _BLOCK != 0:
        reasons.append(f"Skv={Skv} not divisible by BLOCK={_BLOCK}")
    if Hq % Hkv != 0:
        reasons.append(f"Hq={Hq} not divisible by Hkv={Hkv}")
    if Sq != Skv:
        reasons.append(f"asymmetric Sq={Sq} Skv={Skv} not supported")
    return reasons


def _build_configs(filter_batch=None, filter_seqlen=None):
    compatible, skipped = [], []
    for B in _CLC_BATCHES:
        if filter_batch and B not in filter_batch:
            continue
        for (Sq, Skv) in _CLC_SEQLEN_PAIRS:
            if filter_seqlen and Sq not in filter_seqlen and Skv not in filter_seqlen:
                continue
            for (Dq, Dkv) in _CLC_HEAD_DIMS:
                for (Hq, Hkv) in _CLC_HEAD_PAIRS:
                    for causal in _CLC_CAUSAL:
                        reasons = _is_compatible(B, Sq, Skv, Dq, Dkv, Hq, Hkv)
                        cfg = dict(B=B, Sq=Sq, Skv=Skv, Dq=Dq, Dkv=Dkv,
                                   Hq=Hq, Hkv=Hkv, causal=causal)
                        (skipped.append((cfg, reasons)) if reasons
                         else compatible.append(cfg))
    return compatible, skipped


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _device():
    return "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"


def _sync(device):
    if device == "npu":
        torch.npu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tflops(B, Hq, Sq, Skv, D, latency_ms):
    """QK^T: B*Hq*Sq*Skv*D MADs, PV: same, each = 2 flops."""
    return 2 * 2.0 * B * Hq * Sq * Skv * D / (latency_ms * 1e-3) / 1e12


def _bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, latency_ms):
    """Read Q + K + V, write O. All fp16 = 2 bytes."""
    elem = 2
    bytes_io = elem * (B * Hq * Sq * D +       # Q
                       B * Hkv * Skv * D +      # K
                       B * Hkv * Skv * D +      # V
                       B * Hq * Sq * D)         # O
    return bytes_io / (latency_ms * 1e-3) / 1e9


def _wall_stats(latencies):
    s = sorted(latencies)
    return s[len(s) // 2], sum(s) / len(s), s[0], s[-1]


def _bench_wall(fn, device, warmup, rep):
    for _ in range(warmup):
        fn()
    _sync(device)
    latencies = []
    for _ in range(rep):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        latencies.append((time.perf_counter() - t0) * 1e3)
    return _wall_stats(latencies)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

_MASK_CACHE: dict = {}


def _causal_mask_2048(device):
    """The compressed 2048x2048 causal mask sparse_mode=2 expects."""
    key = str(device)
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = torch.triu(
            torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1)
    return _MASK_CACHE[key]


def _ref_npu_fa(q, k, v, is_causal):
    """aclnnFlashAttentionScoreV4. GQA is native through head_num."""
    B, Hq, S, D = q.shape
    scale = (1.0 / D) ** 0.5
    if not is_causal:
        return torch_npu.npu_fusion_attention(
            q, k, v, head_num=Hq, input_layout="BNSD", scale=scale)[0]
    return torch_npu.npu_fusion_attention(
        q, k, v, head_num=Hq, input_layout="BNSD", scale=scale,
        atten_mask=_causal_mask_2048(q.device), sparse_mode=2)[0]


def _ref_fp32(q, k, v, is_causal):
    """fp32 ground truth, GQA expanded, streamed over query blocks.

    The full S x S score matrix is never materialized: at B=32/S=16384 it would
    be hundreds of GB. Each query block does an exact (not online) softmax over
    all visible keys. qblk shrinks with B*Hq*S to bound peak memory to ~2 GB.
    """
    B, Hq, S, Dh = q.shape
    Hkv = k.shape[1]
    scale = 1.0 / math.sqrt(Dh)
    qf, kf, vf = (t.float() for t in (q, k, v))
    if Hq != Hkv:
        rep = Hq // Hkv
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    out = torch.empty_like(qf)
    budget = 512 * 1024 * 1024  # ~2 GB of fp32 scores
    qblk = max(1, min(1024, budget // (B * Hq * S)))
    qblk = (qblk // 128) * 128 or 128
    row_idx = torch.arange(S, device=q.device)
    for s0 in range(0, S, qblk):
        s1 = min(s0 + qblk, S)
        scores = torch.matmul(qf[:, :, s0:s1], kf.transpose(-1, -2)) * scale
        if is_causal:
            scores = scores.masked_fill(
                row_idx[None, :] > row_idx[s0:s1][:, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, s0:s1] = torch.matmul(probs, vf)
    return out.to(q.dtype)


def _ref_sdpa(q, k, v, is_causal):
    """torch's scaled_dot_product_attention, a third baseline (GQA expanded)."""
    Hq, Hkv = q.shape[1], k.shape[1]
    if Hq != Hkv:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal)


# ---------------------------------------------------------------------------
# One shape
# ---------------------------------------------------------------------------

def run_one(cfg, warmup, rep, no_check, device):
    B, Sq, Skv = cfg["B"], cfg["Sq"], cfg["Skv"]
    Hq, Hkv, D = cfg["Hq"], cfg["Hkv"], cfg["Dq"]
    causal = cfg["causal"]

    torch.manual_seed(0)
    q = (torch.randn(B, Hq, Sq, D, device=device) * 0.1).to(torch.float16)
    k = (torch.randn(B, Hkv, Skv, D, device=device) * 0.1).to(torch.float16)
    v = (torch.randn(B, Hkv, Skv, D, device=device) * 0.1).to(torch.float16)

    max_err = float("nan")
    check = "SKIP"
    if not no_check:
        ref = _ref_fp32(q, k, v, causal)
        out = flash_attention_fwd(q, k, v, is_causal=causal)
        max_err = (out.float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(ref.float(), out.float(),
                                       rtol=2e-2, atol=2e-2)
            check = "PASS"
        except AssertionError:
            check = "FAIL"

    fa_fn = lambda: flash_attention_fwd(q, k, v, is_causal=causal)
    npufa_fn = lambda: _ref_npu_fa(q, k, v, causal)
    fa_med, fa_mean, fa_min, fa_max = _bench_wall(fa_fn, device, warmup, rep)
    npufa_med = _bench_wall(npufa_fn, device, warmup, rep)[0]

    sdpa_med = float("nan")
    try:
        sdpa_fn = lambda: _ref_sdpa(q, k, v, causal)
        sdpa_fn()
        sdpa_med = _bench_wall(sdpa_fn, device, warmup, rep)[0]
    except Exception:
        pass

    speedup = npufa_med / fa_med if fa_med > 0 else float("inf")
    return dict(
        B=B, Hq=Hq, Hkv=Hkv, Sq=Sq, Skv=Skv, D=D, causal=causal, cb=1,
        config=_CONFIG_LABEL,
        fa_ms=fa_med, fa_mean=fa_mean, fa_min=fa_min, fa_max=fa_max,
        npufa_ms=npufa_med, speedup=speedup,
        tflops=_tflops(B, Hq, Sq, Skv, D, fa_med),
        bw_gbs=_bandwidth_gbs(B, Hq, Hkv, Sq, Skv, D, fa_med),
        sdpa_ms=sdpa_med,
        sdpa_tflops=(_tflops(B, Hq, Sq, Skv, D, sdpa_med)
                     if sdpa_med == sdpa_med else float("nan")),
        max_err=max_err, check=check,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark the BNSD FA router vs npu_fusion_attention")
    # 30/60 rather than thin defaults: at S=256 the vendor column alone swings
    # ~1.5x between runs with few reps, which moves the speedup denominator.
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--rep", type=int, default=60)
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--csv", type=str, default="bench_fa_bnsd.csv")
    p.add_argument("--filter-batch", type=str, default=None,
                   help="Comma-separated batch sizes, e.g. '1,4'")
    p.add_argument("--filter-seqlen", type=str, default=None,
                   help="Comma-separated seqlens, e.g. '2048,4096'")
    return p.parse_args()


def main():
    args = _parse_args()
    device = _device()

    filter_batch = ([int(x) for x in args.filter_batch.split(",")]
                    if args.filter_batch else None)
    filter_seqlen = ([int(x) for x in args.filter_seqlen.split(",")]
                     if args.filter_seqlen else None)
    compatible, skipped = _build_configs(filter_batch, filter_seqlen)

    print(f"Device : {device}")
    print(f"Warmup : {args.warmup}   Rep: {args.rep}")
    print(f"Kernel : DIM={_KERNEL_DIM}  NUM_CORES={_NUM_CORES}  config={_CONFIG_LABEL}")
    print()

    if skipped:
        print(f"=== Skipped configs: {len(skipped)} (kernel incompatible) ===")
        for cfg, reasons in skipped[:10]:
            print(f"  SKIP B={cfg['B']} Hq={cfg['Hq']} Hkv={cfg['Hkv']} "
                  f"Sq={cfg['Sq']} Skv={cfg['Skv']} Dq={cfg['Dq']} Dkv={cfg['Dkv']}"
                  f"  — {'; '.join(reasons)}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
        print()

    print(f"=== Running {len(compatible)} compatible configs ===\n")
    hdr = (f"{'#':>3} {'B':>3} {'Hq':>3} {'Hkv':>4} {'Sq':>6} {'Skv':>6} {'D':>3} "
           f"{'csl':>4} {'FA(ms)':>9} {'npu_fa(ms)':>11} {'sdpa(ms)':>10} "
           f"{'spd':>6} {'TFLOPS':>8} {'BW(GB/s)':>10} {'max_err':>9} {'status':>8}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    results = []
    for i, cfg in enumerate(compatible, 1):
        try:
            r = run_one(cfg, args.warmup, args.rep, args.no_check, device)
            direction = "↑" if r["speedup"] > 1 else "↓"
            print(f"{i:>3} {r['B']:>3} {r['Hq']:>3} {r['Hkv']:>4} {r['Sq']:>6} "
                  f"{r['Skv']:>6} {r['D']:>3} {'Y' if r['causal'] else 'N':>4} "
                  f"{r['fa_ms']:>9.3f} {r['npufa_ms']:>11.3f} {r['sdpa_ms']:>10.3f} "
                  f"{r['speedup']:>5.3f}{direction} {r['tflops']:>8.3f} "
                  f"{r['bw_gbs']:>10.1f} {r['max_err']:>9.2e} {r['check']:>8}")
            results.append(r)
        except Exception as exc:
            print(f"{i:>3} {cfg['B']:>3} {cfg['Hq']:>3} {cfg['Hkv']:>4} {cfg['Sq']:>6} "
                  f"{cfg['Skv']:>6} {cfg['Dq']:>3} {'Y' if cfg['causal'] else 'N':>4} "
                  f"{'ERROR':>66}")
            print(f"    ERROR: {str(exc).splitlines()[0][:60]}")
            results.append(dict(
                B=cfg["B"], Hq=cfg["Hq"], Hkv=cfg["Hkv"], Sq=cfg["Sq"],
                Skv=cfg["Skv"], D=cfg["Dq"], causal=cfg["causal"], cb=1,
                config=_CONFIG_LABEL, fa_ms=float("nan"), fa_mean=float("nan"),
                fa_min=float("nan"), fa_max=float("nan"), npufa_ms=float("nan"),
                speedup=float("nan"), tflops=float("nan"), bw_gbs=float("nan"),
                sdpa_ms=float("nan"), sdpa_tflops=float("nan"),
                max_err=float("nan"), check="ERROR"))

    print(sep)

    passed = [r for r in results if r["fa_ms"] == r["fa_ms"]]
    failed = len(results) - len(passed)
    if passed:
        avg = sum(r["speedup"] for r in passed) / len(passed)
        best = max(passed, key=lambda r: r["speedup"])
        worst = min(passed, key=lambda r: r["speedup"])
        print(f"\nSummary: {len(passed)} passed, {failed} failed, {len(skipped)} skipped")
        print(f"  Average speedup: {avg:.3f}x")
        print(f"  Best:  B={best['B']} Hq={best['Hq']} Hkv={best['Hkv']} "
              f"Sq={best['Sq']} -> {best['speedup']:.3f}x "
              f"({best['fa_ms']:.3f} vs {best['npufa_ms']:.3f} ms)")
        print(f"  Worst: B={worst['B']} Hq={worst['Hq']} Hkv={worst['Hkv']} "
              f"Sq={worst['Sq']} -> {worst['speedup']:.3f}x "
              f"({worst['fa_ms']:.3f} vs {worst['npufa_ms']:.3f} ms)")
        acc = [r for r in passed if r["check"] in ("PASS", "FAIL")]
        if acc:
            n_fail = sum(1 for r in acc if r["check"] == "FAIL")
            print(f"  Accuracy: {len(acc) - n_fail}/{len(acc)} PASS, "
                  f"max_err {max(r['max_err'] for r in acc):.2e}")

    if args.csv and results:
        csv_path = os.path.join(_HERE, args.csv)
        fields = ["B", "Hq", "Hkv", "Sq", "Skv", "D", "causal", "cb", "config",
                  "fa_ms", "fa_mean", "fa_min", "fa_max",
                  "sdpa_ms", "speedup", "tflops", "sdpa_tflops", "bw_gbs",
                  "max_err", "check", "npufa_ms"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"\nCSV saved to {csv_path}")


if __name__ == "__main__":
    main()
