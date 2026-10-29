# MiniMax TP8 W8A8 MoE: guarded H100 tuning and quant/OAI fusion

This is a draft port of a measured serving implementation into the existing
`flaggems_vllm.ops.fused_moe` implementation. It does not add another kernel copy.
The target-repository GPU port has **not been executed**: the eight H100s are
reserved for formal end-to-end ablations. All performance below is from the
frozen precursor/production implementation; CPU AST equivalence does not turn
those results into a GPU validation of this checkout.

## Relationship to earlier work

- Continues the INT8 configuration work in
  https://github.com/flagos-ai/FlagGems-vllm/pull/957. This patch retains that
  PR's INT8 dtype key and forwarding: selecting the plain-half path can omit
  INT8 scales. It does not claim the original PR's YAML alone produced the
  current gains, or copy its broad device tables.
- Builds on the quant/OAI implementation already merged here in #902. The
  correct OAI formula and boundary protections from
  https://github.com/flagos-ai/FlagGems/pull/6916 are synchronized into that
  existing copy: FP32 scalar range, int64 addressing, and NaN propagation.
  This does not open another FlagGems PR for the same activation changes.
- The measured production namespace is `flag_gems.fused.fused_moe`, SHA256
  `c770bc11d5de800c6d52cf36518e5e9834df3ceb5c7d3ad19545f96f45e9e6dd`.
  Its function/class/statement AST matches this port after only import namespace
  and debug-log normalization. The proof, exact hashes, and limitations are in
  [minimax_moe_port_provenance.json](data/minimax_moe_port_provenance.json).

**Integration with #957 needs a new run.** Its proposed H100 YAML contains the
`128,384,int8_w8a8,0,0` key, while the measured image and this PR's main base do
not. This port intentionally retains the measured rule that embedded tuning
configs take priority. If #957 is merged first, that table can suppress the
opt-in profile. Do not present the current measurements as a combined validation
of both PRs; coordinate the profile/table policy and rerun before promotion.

## Contract and dispatch

| Item | Contract |
| --- | --- |
| API | Existing `fused_experts_impl`; forward-only inference; no new export/registration |
| Target | Exact NVIDIA H100 80GB HBM3 / SM90, BF16 activations, W8A8 INT8 weights and FP32 per-channel scales |
| Rank-local shape | E128, H6144, I384, top-k4; W1 `[128,768,6144]`, W2 `[128,6144,384]` |
| OAI | `gate * sigmoid(alpha * gate) * (up + beta)`, gate upper clamp, up symmetric clamp; retain NaNs |
| Rounding | FP32 OAI computation, explicit BF16 activation store before quant2, fixed-order FP32 MoE sum |
| Scalars | Real, finite FP32 alpha/beta/clamp; clamp nonnegative; validate before tensor work |
| Unsupported fast scope | Other platforms/products/layouts/dtypes/shapes, EP/bias/input-router weighting retain existing library dispatch |
| New compute fallback | None. No new ATen/cuBLAS compute fallback or precision relaxation; existing unfused quant chain is retained |
| FP8 | No FP8 computation is introduced by this MoE path; it remains INT8 |

Opt in before importing the module:

```bash
export FLAG_GEMS_W8A8_MOE_TUNING=tp8_h100
export FLAG_GEMS_W8A8_QUANT_FUSION=quant_oai
```

Both flags remain off by default. Keep the deployment blacklist
`topk,_scaled_dot_product_attention_math,maximum,true_divide,scatter_`.
The native-div recipe recognizes the live registrar keys and the exact native
CUDA div kernel identities for Torch `2.13.0+cu129`, git `cf30153c4c131c8164ee7798e5022d810682e2cb`.
Partial registration, foreign div overrides, other Torch builds and unknown
interfaces retain the existing unfused chain. No decision is cached by registrar
identity. FP16 is excluded from this mixed recipe because its source-dtype
1e-10 epsilon underflows. Plain/all-FG recipes remain separate.

| Path | Trigger | Configuration |
| --- | --- | --- |
| Decode | Exact guard, 16<=M<=64 | GEMM1 SWAP_AB, original K128 FP32 partial-sum order; GEMM2 unchanged |
| Prefill | Exact guard, 1024<=M<=8192 | Shared BM64, K64; stage-specific N/group/stages; GEMM2 INT32 down for K384 |
| Quant | Opt-in, SM90, 2D BF16/FP16, widths384/6144, rows<=32768 | Existing fused quant and OAI+quant2 kernels, separate reduction/NaN and precise-div policies |
| General | Flag off / guard fails / embedded config wins | Existing kernel/config/unfused chain |

The INT32 down guard additionally requires both A/B INT8, no block quant,
K<=1024 and no fused SiLU. The worst possible integer absolute sum is <=2^24,
so conversion to FP32 is exact. No extrapolation to long reductions.

**Autotune exemption:** these are explicit opt-in, previously measured profiles
inside the existing config dispatcher, with no new runtime search. Preserving
GEMM1 K reduction order and graph serving behavior is part of the contract.
Existing embedded tuning remains authoritative. No generic backend candidate
space is expanded.

## Measured precursor microbenchmarks

H100 80GB HBM3, Torch2.13+cu129, frozen FlagTree3.6; BF16 input, INT8 W1/W2,
FP32 scales, TP8 rank-local shape above. Each cell has 30 warmup calls and 100
paired alternating samples. CUDA event eager elapsed, CPU enqueue-call wall
latency, and graph device replay were recorded separately. Never subtract host
and event timings to estimate CPU cost.

The chain includes quant1, alignment, GEMM1, OAI/quant2, GEMM2 and final sum;
it excludes router, shared expert and collectives. Routes are synthetic uniform
and 80%-hot16, not a claimed model-routing distribution. Individual cells are
reported without a fabricated weighted aggregate. These were separate rounds,
so their ratios must not be multiplied.

| Change measured | M / route | Before graph us | After graph us | Speedup |
| --- | --- | ---: | ---: | ---: |
| Prefill tile/INT32 profile, fusion off | 8192 / uniform | 2423.78 | 2048.56 | 1.183x |
| Prefill tile/INT32 profile, fusion off | 8192 / hot16 | 2640.19 | 2209.95 | 1.195x |
| Production mixed-policy fusion, same tile profile both sides | 64 / uniform | 352.98 | 328.32 | 1.075x |
| Production mixed-policy fusion, same tile profile both sides | 64 / hot16 | 206.74 | 183.47 | 1.127x |
| Production mixed-policy fusion, same tile profile both sides | 8192 / uniform | 2066.14 | 1255.86 | 1.645x |
| Production mixed-policy fusion, same tile profile both sides | 8192 / hot16 | 2215.41 | 1400.61 | 1.582x |
| Subsequent GEMM1 SWAP_AB, fusion/profile on both sides | 24 / uniform | 271.33 | 249.52 | 1.087x |
| Subsequent GEMM1 SWAP_AB, fusion/profile on both sides | 64 / uniform | 338.75 | 332.19 | 1.020x |
| Subsequent GEMM1 SWAP_AB, fusion/profile on both sides | 64 / hot16 | 183.54 | 178.19 | 1.030x |

P8192 fusion reduced eager enqueue medians 2434.56->1101.43 us (uniform) and
2446.10->1110.22 us (hot16). Those include dispatch/allocation and possible
blocking APIs, and are not pure CPU-time measurements. Decode sweep covered
M1/8/16/24/32/40/48/56/64/128/256/512/1024/2048/4096/8192; wider M controls did
not gain materially. Compact per-cell min/max/p90 evidence and source hashes:
[fusion](data/minimax_moe_fusion.json), [decode](data/minimax_moe_decode.json).

Precursor validation retained strict bitwise current-baseline comparisons,
raw quant/scale/activation bytes, zero/NaN/Inf/halfway/stride fixtures, graph
inputs refreshed four times, independent graph workspaces, and full INT8-range
stress. **Independent plain PyTorch reference is not all-pass:** the production
mixed-policy study had 23/24 passing independent literal cells. Both original
and fused implementations share an M8192/hot16 OAI BF16 midpoint discrepancy:
8 of 50,331,648 outputs exceed the original atol=.004/rtol=.016 threshold,
max absolute strict error .0078125. The known fixture is gate=-1.703125,
up=.1337890625. No tolerance was relaxed and no dispatcher oracle was relabeled
as an independent plain PyTorch oracle. This port preserves the existing
arithmetic; it does not resolve that prior discrepancy.

A later cache eviction-hint experiment failed with CUDA illegal instruction
before functional validation; it was rejected. Its source/hints are absent here.
NCU on the retained D64 uniform path measured G1/G2 HBM utilization90.98/86.75%
and L2 hit2.24/2.62%, supporting a streaming-weight bottleneck. These counters
are separate from ordinary benchmark timings.

## Formal production MoE-family E2E ablation

Attention stays enabled. Only the two flags above are set to `off` in the control;
correct INT8 dtype/scales/OAI remain. This measures the current MoE tuning plus
quant/OAI fusion family, including later modifications. It is shared follow-up
benefit related to #957, **not** #957's original isolated causal delta or an
E2E run of this `flaggems_vllm` port.

| Input / output | Family off output tok/s | Current output tok/s | Throughput gain |
| --- | ---: | ---: | ---: |
| 4K / 1K | 1857.57 | 2112.55 | +13.73% |
| 16K / 1K | 674.11 | 788.62 | +16.99% |
| 32K / 1K | 361.15 | 423.07 | +17.15% |

8xH100 80GB NVLink; TP8/DP1, EP off, no speculation; client C64,128 requests,
output1024, independent per-request seeds retained, temperature1/top_p.95,
ignore_eos. FP8 e4m3 main KV, BF16 index KV; 14GiB KV per rank,489088 token
slots; max-batched-tokens8192. R1 warmup excluded; table averages R2 and R3.
4K current reuses an unchanged source/config run. All requests succeeded,
prefix hits were zero and server ITL count was130944. Both 16K variants had
4 preemptions each round; 4K/32K had none. Long contexts are capacity constrained
(approximately28/14 full-length contexts at16K/32K), so C64 is not a claim of64
resident long-context decode requests. Raw round values and source hashes:
[E2E evidence](data/minimax_moe_e2e.json). Attention/joint ablations are still
running; conditional percentages must not be added.

## Reproduce and remaining validation

CPU metadata/policy tests avoid importing Torch or initializing CUDA:

```bash
python -m pytest --noconftest -c /dev/null -q tests/test_fused_moe_policy_cpu.py
```

Run only when an H100 is exclusively allocated and a compatible environment is
installed. First execute the target-repository GPU suite, then verify complete
chains and time the same verified source:

```bash
PYTHONPATH=src python -m pytest -q tests/test_fused_moe.py
PYTHONPATH=src python benchmark/benchmark_fused_moe_tp8_h100.py \
  --comparison family --stage verify --ms 16,24,64,1024,4096,8192 --output verify-family.jsonl
PYTHONPATH=src python benchmark/benchmark_fused_moe_tp8_h100.py \
  --comparison family --stage bench --ms 16,24,64,1024,4096,8192 \
  --verified-by verify-family.jsonl --warmup 30 --rep 100 --output bench-family.jsonl
```

Use separate verify/bench files with `--comparison fusion` to isolate fusion
(profile on both sides), and `--comparison small-decode --ms 16,24,32,48,64` to
isolate GEMM1 SWAP_AB (profile/fusion on both sides). Output files cannot be
overwritten; verification receipts are bound to source hash, comparison and
shape/route set. Generic PyTorch quant baseline remains in
`benchmark/test_fused_moe_quantfusion.py`.

Draft gates: 63 CPU policy checks passed; source compilation and lint checked.
GPU target-module numerics, dependency compatibility, benchmark reproduction,
and target-port E2E are **NOT_RUN**. Existing precursor data is evidence for
review, not permission to mark those gates passed. No new FlagGems dependency
PR is required because the implementation already lives in this repository.
