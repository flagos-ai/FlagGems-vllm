# Research packaging contract

| Project fact | Decision |
|---|---|
| Current operator | Existing MiniMax packed-KV attention in PR #979, pinned at `fead47fdc6d4b81cc73b0d6488834dce75d95bf1` |
| Editable scope | This experiment directory only; library code, exports, registration, tuning tables and existing PR remain unchanged |
| Build | Pure Python research sources and immutable measured-source archive; no package build or serving installation |
| Validation | Verify artifact hashes, source AST equivalence and raw-sample statistics locally; retained H100 receipts describe the earlier experiment |
| Benchmark | Complete serving-precursor `attn_auto` chain, paired eager/Graph CUDA events; median microseconds, lower is better |
| Active set | BF16 Q/output, FP8 E4M3 KV storage, BF16 QK/PV, FP32 accumulation; H100/GQA8/Hkv1/D128/page128/TopK16; q8192 and q8153 representative cases |
| Aggregation | Per-cell raw samples and three-round ratios; no fabricated production-distribution weighted score |
| Autotune | Research exemption: six explicitly selected pipeline variants, no runtime search or library autotune expansion |
| Profiler | PTX/TTGIR/cubin SASS/resource evidence retained; NCU and compute-sanitizer were unavailable and were not run |
| Timeout | Preparation is CPU-only; GPU reproduction requires a fresh exclusive lease and an external process timeout |

| Numerical contract | Decision |
|---|---|
| Reference | Frozen serving precursor SHA256 `1ab42fc298abf8a950d02ac5204869ed0f2eeade5185b17b58416e1be845a789`, native vLLM and an independent FP32 literal oracle |
| Math | Preserve per-128-token-page softmax and page order, BF16 probability rounding, scalar scale order, and scaled accumulator as PV WGMMA C |
| Storage vs compute | FP8 storage is converted to BF16 before QK/PV; no FP8 dot instructions |
| Outputs | BF16, with redzones and exact-write checks; representative accepted cases have zero numerical difference from the frozen baseline |
| Scope | Forward-only inference research; this is not an installed library operator or a full supported-domain validation |
| Torch use | Input generation, references and test bookkeeping only; experimental kernels use Triton/TLE |

| Path | Implementation | Decision |
|---|---|---|
| Existing PR #979 API | Unchanged | No new exports or registration required |
| Experimental WS | Raw 2/3-slot rings, consumer warp alternatives, producer-side BF16 conversion | Correctness checked on representative cases; performance rejected |
| Experimental non-WS | Full-row K/V TMA with 2/3 slots and existing compute warp counts | Correctness checked on representative cases; performance rejected |
| General/unsupported | Frozen serving implementation inside the measured-source workspace | No new Torch compute fallback |

G4 registration/config integration and G6 production promotion are deliberately not attempted:
all measured variants regressed. This draft records the negative result rather than installing a
regressed operator. The source archive preserves the exact tested files; readable kernel copies
are formatting-only views with function AST equivalence recorded in the manifest.
