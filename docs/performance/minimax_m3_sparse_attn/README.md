# MiniMax M3 native packed-KV sparse attention (draft)

This adds an explicit operator API for native vLLM 0.28 packed pages. It keeps
selected-block attention and a device-controlled compact union schedule in the
operator library. Plugin installation, environment overrides, indexer changes,
cache allocation and small-batch native routing remain the caller's concern.

Related optimization work: [FlagAttention PR #85](https://github.com/flagos-ai/FlagAttention/pull/85).
That PR's prefetch/layout/scheduling work informed the broader optimization
campaign. This implementation was independently rewritten; it does not claim
that these kernels are a direct port of #85 or that its public BF16 benchmark
numbers measure the current FP8 cache path. Benefits of the optimization family
are shared with that work. This is also related to the existing broad
[FlagGems-vllm MSA PR #58](https://github.com/flagos-ai/FlagGems-vllm/pull/58);
this PR is restricted to the native packed-KV attention consumer, without its
indexer or a dependency on that PR's branch.

## API and numerical contract

```python
flaggems_vllm.minimax_m3_sparse_attn(
    q, kv_cache, topk_idx, block_table, cu_seqlens_q,
    seq_lens, prefix_lens, max_query_len, num_kv_heads,
    sm_scale, output, k_scale=None, v_scale=None,
    qk_dtype="bf16", schedule="auto", chunks=None,
)
```

- Q/output: contiguous BF16 `[tokens, heads, D]`, D96/128, GQA8. Output must not
  overlap inputs. Forward inference only.
- KV: typed BF16 or FP8-E4M3 `[pages, KV_heads, 128, 2*D]`. Arbitrary positive
  cache/top-k/block-table strides are passed to the kernels without copies.
- Metadata: int32, top-k `[KV_heads,tokens,TOPK]`, TOPK1..32. Length arrays are
  contiguous. CU, selected IDs, lengths and physical page contents must satisfy
  the documented device-data bounds; the host never reads them back.
- Prefix and sequence length enforce causal bounds. `-1` holes are ignored,
  repeated IDs retain list multiplicity, and entirely masked rows are zero.
- Optional K/V scales are paired scalar FP32 tensors. QK and PV accumulate in
  FP32; PV uses BF16 inputs and the output is BF16.

Precision and schedule are independent. A BF16 QK request stays BF16 on SM80,
SM86, SM89 and SM90. An explicit FP8 request first probes actual NVIDIA target
capability: on SM89+ it needs typed FP8 KV; on SM80/86 it uses INT8 QK only if
the caller's precision planner has already allocated BF16 KV. Passing FP8 KV
to an unsupported device raises before launch; bytes are never reinterpreted.
Explicit INT8 QK also needs BF16 KV. The quantization policy follows the
capability-first intent of [vllm-plugin-FL PR #622](https://github.com/flagos-ai/vllm-plugin-FL/pull/622).

INT8 uses symmetric per-row D-axis scales, ties-even rounding to [-127,127],
INT32 dot and FP32 dequantization. For D<=128 the absolute INT32 dot bound is
2,064,512. Nonfinite pairs take the frozen device-only BF16 exceptional branch.
This is a quantized computation request, not an equivalence claim to BF16.

Auto scheduling uses the measured H100 SM90 domain, TOPK16 and q>=384. Its
union tile publishes compact/scattered/duplicate flags on every invocation,
including graph replay. Selected traversal and optional merge consume exactly
those flags. Other validated shapes use selected traversal. FP8 QK always uses
selected traversal so one call cannot mix BF16 and FP8 QK according to index
layout. The serving policy that keeps native decode at B<=16 is intentionally
not replaced or hidden inside this standalone operator.

## Historical microbenchmarks

These are actual retained measurements from the prior integration source
`3e97e3ccd4068475e1a9d3314093343875e4b589cc76ad416161d71ca069d4fd`.
They are context for this draft, **not results from the new library wrapper**.
`historical_micro.json` contains the selected raw records, source digest,
metadata, errors and eager/graph timings, including the small-Q regression.

Hardware/software: NVIDIA H100 80GB HBM3, PyTorch 2.13.0+cu129, Triton/FlagTree
3.6.0. Q is BF16 with 8 heads, D128, one KV head; cache is typed FP8-E4M3,
context32768, selected16 pages of128 tokens. QK remains BF16 for both native
and candidate prefill. Samples use 30 warmup calls, 100 CUDA-event measurements,
interleaved candidate/native order, median latency. Graph and eager are separate.

| Q | Distribution | Eager native / auto (us) | Graph native / auto (us) |
| --- | --- | ---: | ---: |
| 512 | compact span16 | 156.448 / 230.464 | 127.648 / 75.168 |
| 512 | scattered | 129.664 / 125.120 | 129.248 / 123.296 |
| 2048 | compact span16 | 485.872 / 81.776 | 485.632 / 80.416 |
| 2048 | scattered | 489.440 / 440.752 | 489.248 / 439.360 |
| 4096 | compact span16 | 956.224 / 156.576 | 956.080 / 155.760 |
| 4096 | scattered | 961.856 / 854.240 | 961.664 / 853.232 |

The q512 compact eager case regresses; no all-shape or end-to-end speedup is
inferred from this table. All six cases checked finite output and max-absolute
error<0.02 before timing; retained dense-sample errors are in the JSON.

Original campaign command (requires the retained integration harness/fixtures):

```shell
python bench_auto.py --out results/policy-release.jsonl --samples 100
```

The new repo benchmark reproduces the important dimensions and timing method,
with independently generated fixtures and the new API. It does not pretend to
reproduce the original random fixture bits. It compares native vs candidate
BF16-QK under identical KV storage, checks sampled dense output before timing,
and reports all raw samples plus median/P10/P90:

```shell
PYTHONPATH=src pytest -q -s benchmark/test_minimax_m3_sparse_attn.py --level core
PYTHONPATH=src pytest -q -s benchmark/test_minimax_m3_sparse_attn.py --level comprehensive --record json
```

## Integration validation boundary

The five retained JIT functions, including signatures and decorators, are
AST-identical to frozen production `attn_tuned.py` SHA256
`c44d602c25417a9e5281d55a8cdccb0f917e03d7760c9f638d05da79c458a314`;
`kernel_equivalence.json` records their AST hashes. The production INT8 component
had 19 full-forward cases /57 graph stages, 90 family stages and 25 SM80 offline
compile cases. The FP8-capable original path had 105 parity/graph stages against
its preceding source. Those are historical component checks on H100; SM80
compilation is **not A100 device execution**.

New code in this draft: explicit per-call precision, metadata errors, standalone
launch wrappers, stream-isolated workspaces, uninitialized auto-flag allocation,
package exports, tests and benchmark. Kernel equality does not validate these
wrappers. CPU-only gates passed: import with CUDA uninitialized,47 host contract
checks, AST equivalence, no-Torch-compute audit, Black/isort/flake8 and compileall.
**Package GPU correctness, graph replay and timing are pending**. GPUs are
reserved for the separate serving attribution experiment; this PR remains draft.

```shell
python -m pytest -q --noconftest tests/test_minimax_m3_sparse_attn_host.py
PYTHONPATH=src pytest -q tests/test_minimax_m3_sparse_attn.py --quick
PYTHONPATH=src pytest -q tests/test_minimax_m3_sparse_attn.py
```

No Torch compute fallback is installed: production Torch calls are metadata,
CUDA capability/stream queries and `torch.empty` allocations; contiguous views
are no-copy. All Torch compute/cast/copy/reference operations are test-only.
Non-NVIDIA, SM<80, unsupported shapes/layouts/scales and logical/workspace offsets
outside the preserved int32 range raise explicitly.

Autotune exemption for this draft: keep the measured fixed schedules. The union
flag producer and selected/merge consumers require matching launch geometry;
adding an unverified libtuner search would change the frozen graph workspace
contract. No new tuning space or tune-config entry is claimed. New platform
schedules need a separate correctness-gated experiment. There is no new generic
FlagGems change in this attention PR.

Exports are in `ops.__all__` and the top-level star import. This API has no
ATen schema, so it is explicitly excluded from the automatic `_FULL_CONFIG`
ATen-override list. No Torch dispatcher schema or override is invented.

## End-to-end attribution: partial (4K complete)

The serving four-arm experiment compares current / MoE-off / attention-off /
both-off at4K/1K,16K/1K and32K/1K, TP8, EPoff, C64,128 requests,1024 output
and independent per-request seeds. Main KV stays FP8-E4M3 and index KV stays
BF16, KV budget14GiB per GPU /489088 usable tokens. R1 is warmup; R2/R3 are
retained. An extra BF16-QK attention arm at4K/16K separates structural attention
benefit from the existing FP8-QK decode path. Longer requests are KV-capacity
limited; client concurrency64 does not mean64 resident long requests.

At4K/1K, current output throughput R2/R3 mean is2112.5498255103557 tok/s;
attention-off is1945.312365341078 tok/s (R2=1934.5183494181642,
R3=1956.106381263992), a conditional +8.5969463387% with MoE kept on.
All128 requests succeed in each scored round; prefix cache hits and preemptions
are zero. This compares the frozen production precursor optimization family,
including its existing FP8-QK decode path. The new package port has not run on
GPU. At16K/1K, current R2/R3 mean is788.6247006357867 tok/s; attention-off is
748.6739107854564 tok/s (R2=747.1117427303548,R3=750.236078840558),
a conditional +5.3362070288%. Both arms have4 preemptions per round,
zero prefix cache hits and128 successes. BF16-QK alignment,32K and combined
ablation remain pending. Neither number measures this new API's default BF16
parameters: production keeps native decode at B<=16 and uses the preexisting
FP8-QK tuned path at larger eligible batches.

The remaining numbers will be appended to this same draft PR body when the
experiment finishes. These numbers include later rewrites and shared ideas;
they are not an isolated attribution to original PR #85 or to this new wrapper.
