# GDN W8A16 FP8 Migration

## Project Contract

| Field | Contract |
| --- | --- |
| API | `flaggems_vllm.gdn_w8a16`; recurrent GDN with FP8 state, not a PyTorch ATen operator |
| Source | FlagGems PR 4957, commit `71eea34b` |
| Editable files | New operator, op exports, new accuracy and benchmark files, this document |
| Frozen reference | Canonical FP32 recurrence in tests; installed vLLM BF16 GDN in benchmark |
| Build | Python source import through `PYTHONPATH=src`; no compiled extension needed |
| Validation | `pytest tests/test_gdn_w8a16.py`; output and reconstructed state against FP32 |
| Benchmark | `pytest benchmark/test_gdn_w8a16.py --mode cudagraph`; milliseconds, baseline / FP8 |
| Active set | Decode accuracy N=1..16384, paired benchmark N=1..8192; 15 prefill (N,T) pairs; H/HV=2/4 and 2/6, K=V=128 |
| Aggregation | Per-shape speedup plus arithmetic and geometric means; key-shape target >=0.9 |
| Timeout | Four-hour single-GPU allocation; individual suites expected below 30 minutes |
| Profiler | H20 Nsight tools available on nodes 020/038 if performance requires investigation |
| Autotune exemption | Preserve validated shape-based BV, warps and stages from the source PR. In-place state and scale make repeated autotune trials stateful. No new online tuner is added in this migration. |
| Fallback | No torch compute fallback; existing repository Triton chunk kernels are reused |

## Input and Output Contract

| Field | Contract |
| --- | --- |
| Inputs | q/k `[B,T,H,K]`, v `[B,T,HV,V]`, g/beta `[B,T,HV]`; HV divisible by H |
| Dtypes | q/k/v BF16 or FP16, state FP8 E4M3FN, scale FP32 |
| State layout | Contiguous `[S,HV,K,V]` and matching per-V scale `[S,HV,V]` |
| Packed layout | B=1 with GPU cu_seqlens; one state index per sequence or 2D candidate slots |
| Output | Activation dtype `[B,T,HV,V]`; state and scale updated in place and returned |
| Arithmetic | Shape-dependent BF16/FP16 or FP32 products, FP32 reductions and state accumulation |
| Scope | Finite inference inputs, forward only; training/autograd and nonfinite stability are not promised |
| Unsupported | Other activation/state dtypes and malformed indices raise errors; valid non-overlapping slot indices are the caller's responsibility |
| Reference | Test-only torch FP32 recurrence; vLLM BF16 is a performance reference, not the precision oracle |

## Execution Paths

| Path | Trigger | Implementation | Validation |
| --- | --- | --- | --- |
| Empty | Zero tokens/sequences | Empty output, unchanged state | Boundary test |
| Decode | One token per sequence | Original tiled/Grouped/TMA shape dispatch | Canonical FP32 decode, original grouped shape |
| Short prefill | Below chunk crossover or strided inputs | Pipelined recurrent Triton kernel | BF16/FP16, packed and nonpacked |
| Long prefill | Eligible contiguous K=V=128 shapes | Existing chunk GDN, indexed state conversion | Canonical FP32 recurrence |
| Speculative | 2D slots and num_accepted_tokens | Accepted-slot load, paired state/scale snapshots per candidate | Distinct scales and slots |

The module marker and benchmark `op_name` are both `gdn_w8a16`, so the
repository's native accuracy and benchmark JSON writers use the same operator
identifier. Initial state quantization is outside the timed call; dynamic
requantization and scale updates are inside it. BF16 and FP8 use distinct state
buffers, identical inputs and unique slots per sequence. CUDA Graph mode is
selected by the standard benchmark option, not hardcoded in the test.
