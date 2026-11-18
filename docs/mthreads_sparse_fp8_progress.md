<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

## 2026-11-18 K640 preparation and short-query exponent

For K640 decode with at least 128 query rows, the preparation kernel gathers selected KV scales, writes their scales and byte offsets, and computes a probability bound for each query. The conservative fast certificate requires non-negative query scales, bounds from zero through 2.0, selected KV scales at most 2^-9, an observed exponent no greater than 8, and normal probability scales. Other inputs use the existing full qualification formula. The repair launcher scans disjoint tiles of row flags and calls the existing FP32 routine for rejected rows. The global-bound fallback keeps its compact list.

In the native exponent branches for B4 and B16 K640, four logits already assigned to one thread are grouped for the MUSA four-element exp2 instruction. B16 uses the existing conservative margin for grouped FP16 query-norm additions. Both cases retain FP8 QK, two FP8 PV terms and the original FP32 repair.

**The count-weighted speedup is 1.30083584× over 414 shapes and 13,920 original calls.** Individual speedups range from 0.166057× to 1.789978×. The previous published measurement was 1.29591498× on its own complete run.

The independent raw audit reconstructed the measured 1,148-path runtime tree, checked both 195-test suites and all 44 dynamic accuracy checks, and reproduced the entire CSV. The CSV contains the exact per-shape latencies and original call weights. [mthreads_sparse_fp8_performance_20261118_k640.csv](mthreads_sparse_fp8_performance_20261118_k640.csv)

The upstream merge `154a22fc` retained the same target source and tuning entries and passed both 195-test suites, both core sets and all 44 dynamic checks on S5000. The complete timing is tied to the measured runtime snapshot before the merge.

## 2026-11-18: lossless current-input metadata compression

For long K8320 queries, the preparation checks current scales and uses a
parity bitmap for adjacent positive normal power-of-two scale bands.
Selected metadata packs the exact exponent and aligned KV byte offset
into one int32 word. Other scale formats retain original FP32 bits and
raw indices. Both FP8 PV terms and FP32 repair remain.

**Final count-weighted speedup: 1.29591498× over 414 shapes / 13,920 original calls.** This remains below the 1.3× target. The per-shape range is 0.238219×–1.788203×. Large K8320 contains 133 shapes / 4,045 calls and measures 1.738277×. The preceding published full run measured 1.26749697×; these are independent complete runs.

Current metadata change, seven alternating complete-call pairs against the published dfa12444 on the same S5000, with identical inputs within each pair:

| Batch | Output | Eager latency change | Graph latency change |
| ---: | --- | ---: | ---: |
| 512 | BF16 | -3.88% | -3.79% |
| 512 | FP16 | -3.54% | -4.65% |
| 4112 | BF16 | -5.08% | -4.24% |

B512 uses the archived seed42 pilot fixture. B4112 uses the original maximum-batch fixture sliced to 4112. These local comparisons are separate from the full-matrix result.

Both compilers pass 195 functional cases and four core records. All 414
original shapes pass the unchanged precision tolerances. The complete
JSON/precision/source audit is in
`reports/mthreads-pr17-packed-global-meta-20261118/packed-final`.
The operator code remains under 2,000 lines per file after moving the
preparation functions to a private metadata module.

## 2026-11-17: specialize exponent modes outside the long K8320 loop

The long K8320 QK producer selects the paired or native exponent mode
before entering the page loop. Both loops expand from one source body and
retain existing precision gates, FP32 repair and the same barrier events.

**Final count-weighted speedup: 1.26749697× over 414 shapes / 13,920 original calls.** This does not meet the 1.3× target. The per-shape range is 0.632500×–1.846682×. Long K8320 contains 133 shapes / 4,045 calls and measures 1.641154× against native MATE (1.626178× for the same group in the preceding run). The preceding full run was 1.26523108×; these are independent full runs.

Exponent-mode specialization for long K8320, seven alternating complete-call pairs on the same S5000 against b54ac5c9, with identical inputs within each pair:

| Batch | Output | Eager latency change | Graph latency change |
| ---: | --- | ---: | ---: |
| 512 | BF16 | -1.75% | -2.13% |
| 512 | FP16 | -2.86% | -2.06% |
| 4112 | BF16 | -2.11% | -1.66% |
| 4112 | FP16 | -0.42% | -1.52% |

Both exponent modes run in the same mutating BF16 and FP16 eager and
graph tests. All 414 original shapes pass the same precision thresholds;
private and stock compilers each pass 193 functional tests and yield
four complete core records. The full evidence is in
`reports/mthreads-pr17-phase-period-20261117/exp-accept`.

## 2026-11-17: schedule long K8320 PV in groups of four pages

The long K8320 double-buffered consumer processes the first 64 pages in
four-page groups using constant slot and phase indices, then consumes the
last page without prefetching. The math and the per-page barrier sequence
stay the same. Other K values retain per-page scheduling.

**Final count-weighted speedup: 1.26523108× over 414 shapes / 13,920 original calls.** This does not meet the 1.3× target. The per-shape range is 0.221120×–1.671772×. The large-K8320 group covers 133 shapes / 4,045 calls and measures 1.626178× against native MATE (1.600033× for the same group in the preceding run). The preceding full run was 1.25584662×; these are independent full runs.

Long K8320, seven alternating complete-call pairs on the same S5000 against d39d3f06, using identical inputs within each pair:

| Batch | Output | Eager latency change | Graph latency change |
| ---: | --- | ---: | ---: |
| 512 | BF16 | -2.85% | -1.63% |
| 512 | FP16 | -2.58% | -1.29% |
| 4112 | BF16 | -1.57% | -1.38% |
| 4112 | FP16 | -1.45% | -1.28% |

Dynamic BF16 and FP16 tests cover changing the final page, graph replay,
compilation fallback, and raw-index fallback. All 414 original shapes pass
the same precision thresholds. Private and stock compilers each pass
193 functional tests and yield four complete core benchmark records.
The full results are recorded in
`reports/mthreads-pr17-long-metadata-pipeline-20261117/group-accept`.

## 2026-11-17: extend K640 scale preparation to pointer Q

The existing fused scale preparation and packed Q norm now apply to the
existing K640 pointer path, starting at 128 query rows. Compiler rejection
falls back to the existing global-bound preparation. The main kernel,
whole-row FP32 repair, and public API stay unchanged.

**Final count-weighted speedup: 1.25584662× over 414 shapes / 13,920 original calls.** This does not meet the 1.3× target. The per-shape range is 0.240651×–1.646639×. The new K640 pointer coverage (128–4095 rows) contains 65 shapes / 1,789 calls and measures 1.192309× against native MATE (1.150709× for the same group in the preceding run). The preceding complete run was 1.24943226×; these are independent full runs.

B512/K640 eager latency fell by 3.53% (BF16) and 2.69% (FP16) in seven
same-card complete-call pairs. Graph latency fell by 2.38% and 2.33%.
The full matrix uses the private compiler; both private and stock compilers
pass 193 functional cases and produce four core results each. Mixed 32/16
byte copy variants were measured and rejected. The full evidence is in
`reports/mthreads-pr17-g2s-width-20261117/pointer-accept` on the development host.

## 2026-11-17: fuse selected scale preparation for large K640

For the existing K640 Q TME route (at least 4096 query rows), gather selected KV
scales and byte offsets and compute the safety bound in one preparation kernel.
The maximum selected scale is sufficient for the existing certificate.
The FP8 QK/PV main kernel and FP32 repair are retained. A compiler rejection
returns to the previous preparation. All inputs and metadata are refreshed
when invoking a prepared call or replaying a graph.

Seven same-card complete-call pairs at B4112 against c2458933:

| Output | Eager | Graph |
| --- | ---: | ---: |
| BF16 | -2.86% | -0.82% |
| FP16 | -2.32% | -1.03% |

The original 414 shapes and 13,920 calls measure **1.24943226×**
against MATE 0.2.4 on S5000. This does not meet the 1.3× target. All shapes
pass the original output and LSE tolerances. Private and stock compilers each
pass 193 functional cases and produce four core benchmark records.
The full run uses the private compiler and vLLM 0.24.0+empty. Full results are in
[mthreads_sparse_fp8_performance_20261117.csv](mthreads_sparse_fp8_performance_20261117.csv).

Two B16/K8320 changes to the order of KV-copy completion and PV issue pass
precision but regress graph latency. They are not included.

The development report and validation logs are in
`reports/mthreads-pr17-row-scale-20261117`. The validated runtime module
SHA256 is `656ed25437b6775aefb910d1c658c05bfb2c2a878651c6275c90acf992f8cb35`.

# MThreads sparse FP8 MLA optimization progress

Optimization is **active**. The latest validated checkpoint on
worker33026 measures **1.24959722×** over the original **414 shapes / 13,920
calls**. The **1.3× target remains unmet**. This new-host result must not be
attributed solely to the source change by subtracting an older machine's score.
The preceding published checkpoint and its evidence remain below.

## K128 persistent query scheduling, 2026-11-17

The supported large-query K128 path now assigns complete query ranges to a
fixed CTA grid (60 on the measured S5000). Q/KV transfer can overlap the prior
query's consumer work. A producer-only end-of-query rendezvous completes all
producer LSE scratch reads before the next query reuses that storage. The
existing preparation, guards, two-term FP8 PV and FP32 repair are retained.
Compilation, shared-capacity and optional-offset workspace fallbacks are checked.
The new implementation is backend-internal; the legacy JIT implementations
are unchanged. Existing host combine preparation is grouped with scheduling.

Eight alternating complete-call pairs with the final isolated implementation:

| Case | Eager latency change | Graph latency change |
| --- | ---: | ---: |
| B4112 / BF16 | −14.28% | −19.77% |
| B4112 / FP16 | −19.48% | −19.34% |
| B16384 / BF16 | −21.76% | −21.66% |

The final full-matrix count-weighted native-MATE/gems speedup is
**1.24959722×**. The large K128 group has 133 shapes / 415 calls and measures
**0.95871314×**, so it is still slightly slower than native on average.
All 414 shapes pass original output atol 0.002, LSE atol 0.0002 and infinity
masks; maximum absolute errors are 0.0002781897783 and 1.907348633e-6.
Private and stock compilers each pass 193 functional cases and four core
benchmark records. Six forced fallback checks pass. Final formatting and
compatibility-layer checks pass 46 focused cases per compiler. Six existing
kernel executable-section comparisons are byte-identical, and all 546
non-doc source files match the final validation snapshot after the full run.

The first new-host full run measured 1.14882894× with an anomalous B16/K8320
eager latency. Paired diagnostics reproduced eager variability in both the
prior and updated implementations while graph latency remained near 0.124 ms.
After restoring the recorded TorchADA compatibility layer, the complete frozen
matrix was rerun with the same weights and measurement settings. Both runs are
retained; their difference is not claimed as an isolated operator improvement.
The existing large-scale FP32-repair stress limitation is documented separately.

Private compiler e01d6731 is unchanged. The new image's stock compiler is
5a61c68b; no stock full-matrix or NVIDIA hardware result is claimed. Source,
full results, manifests and rejected trials are in
`reports/mthreads-pr17-k128-continue-20261117` in the development workspace.
The complete per-shape records are available in
[mthreads_sparse_fp8_performance_20261117.csv](mthreads_sparse_fp8_performance_20261117.csv).
The initial and final new-host measurements remain separately identified.

## Publication synchronization, 2026-11-17

The validated optimization was saved as `b2f2757b` before synchronizing with
the current fork `main` (`6f7b4be5`), which includes upstream `af184e57`.
The prior merge/revert history required restoring the target branch's
unrelated files while preserving this PR's changes. The 11 operator-specific
implementation/test/benchmark/shape files remain byte-identical to the measured
checkpoint. All fixture-generation and FP32-reference methods used by this
operator retain their AST. The target branch's exports, tuning entries and CI
infrastructure are preserved. The benchmark result-recording support now comes
from the target branch directly.

The 1.24959722× full-matrix result belongs to the pre-synchronization validated
checkpoint; post-merge functional/core checks verify integration separately.
No new full-matrix measurement is inferred from the merge.

## Fused short-query preparation and pipeline, 2026-11-14

For B16/K8320 on the supported grouped PH1 path, the CTA prepares fresh
fragment offsets, FP32 normalized scales, guard modes and factors before
entering two 16-warp roles. The producer computes QK, probabilities and
denominators; the consumer performs KV prefetch and the two FP8 PV terms.
Two KV slots and two high/low probability slots occupy 192 KiB. Page phases
continue across active fragments, and skipped fragments do not advance them.
The existing combine performs whole-query FP32 repair when a fragment is unsafe.
Unavailable compilation, resource capacity or optional workspace retains the
ordinary path. Other shapes retain their existing kernels.

The implementation lives in a backend-internal module. The existing native
binding cache accepts the additional query-norm LLVM text without changing
its default bindings. Existing exponential-boundary and partition tests now
include B16/K8320 for both output dtypes, and the native/short helper files
select the corresponding public CI tests and benchmark.

| Full matrix | Shapes / calls | Count-weighted speedup |
| --- | --- | ---: |
| Previous implementation | 414 / 13,920 | 1.22943718× |
| Fused short pipeline | 414 / 13,920 | **1.23703661×** |
| Selected B16/K8320 | 1 / 1,919 | 0.96250364× → 1.00999107× |

The selected shape's latency decreased 5.47% versus the preceding full run;
its native latency changed by −0.81%. A final-source same-card check with
seven alternating complete-call pairs measured 3.80% lower eager latency
and 20.12% lower graph latency. These checks include the fused preparation
and combine, rather than reporting only the faster matrix stage. Earlier
prototype checks on two cards are kept separately in the development report.
B4 variants and the paired-exponent follow-up are not selected.

All six original shards completed successfully and cover exactly 414 shapes.
Every shape passes output atol 0.002 and 0.004, LSE atol 0.0002 and the original
infinity-mask check. Maximum absolute errors are 0.000268153846 for output and
0.000001907349 for finite LSE. Private and stock compilers each pass **187
functional cases** and a core benchmark with **four complete records**, with
no skips or failures. The stock compiler retains its existing fallback when
the specialized pipeline is unavailable.

The final short module passes **74** additional BF16/FP16 comparisons with
changed content, scales, invalid/duplicate indices, unsafe fragments, tiny
scales and empty selections. Eager and graph replay outputs are checked
independently after poisoning output and workspace buffers. All 545 tracked files
outside `docs/` match the remote validation
snapshot before and after the full run. Full-matrix performance uses the
unchanged private compiler; no stock full-matrix or NVIDIA performance claim
is made. Inputs, weights, native MATE 0.2.4 baseline and tolerances are unchanged.

Evidence: `reports/mthreads-pr17-fused-delivery-20261114` in the development
workspace contains final-source manifests, six complete records, precision
ledgers, paired timings and compiler checks.

## Subsequent experiments, 2026-11-14

The following rejected experiments used runtime commit `3f9d6b77`, whose
full-matrix speedup was **1.22943718×**. They are not part of the retained runtime.
Measured pairs and validation metadata are recorded in
[mthreads_sparse_fp8_experiments.json](mthreads_sparse_fp8_experiments.json).

### Quartic short-query exponent

A lower-degree bounded polynomial was compared on three S5000 devices
(0, 1, 5), B=16 and K=640/8320. All six jobs passed the original
output atol=0.002 / LSE atol=0.0002 and checked actual fast-path eligibility.
Each case used seven alternating complete prepared-call pairs.

| TopK | Eager median change across the three cards | Graph median change |
| --- | ---: | ---: |
| 640 | −0.67% to +0.51% | −1.58% to −0.92% |
| 8320 | −0.34% to −0.28% | +0.10% to +0.34% |

Negative values mean lower latency. The K640 eager direction was inconsistent,
and every K8320 graph comparison regressed. This version was not retained.
No full-matrix result is attributed to it.

### Two-split last-CTA combine

The B16/K640 diagnostic retained two KV prefetch slots, split each 32-head
group into two CTAs, and let the last finisher combine partial numerators
and denominators or invoke the original repair. A counter initialized at
prepare time was reset by the finisher on each call.

The first run stopped because the device-side reference produced NaNs for
a fully invalid row. Debugging found zero output, +inf LSE, valid partials
and reset counters in the candidate; the existing CPU reference confirmed
those results. Only the experimental runner's reference-device argument
was changed to CPU. Kernel code and precision thresholds were unchanged.

A subsequent run without a debugger completed on GPU0. Both BF16 and FP16
passed output atol=0.002 / LSE atol=0.0002, NaN-poisoned graph replays,
counter-reset checks, input mutations, and forced-unsafe rows. There were
seven alternating complete-call timing pairs in each mode.

| Output dtype | Eager control → candidate (µs) | Graph control → candidate (µs) |
| --- | ---: | ---: |
| BF16 | 43.64 → 130.44 | 41.75 → 133.89 |
| FP16 | 43.68 → 131.48 | 41.80 → 133.69 |

Every timing pair regressed. This diagnostic is rejected and does not alter
that 1.22943718× full-matrix checkpoint. The raw failed run and
the debugger-assisted run are excluded from the table above. Neither
experiment constitutes a new 414-shape acceptance result.

## TME for contiguous Q at large batch, 2026-11-14

For K=640 and at least 4096 query rows, the supported large-batch path
moves the contiguous 64×512 FP8 Q tile using two hardware TME loads. The
producer waits for their shared 32 KiB transaction before QK. Irregular KV
indices still use bounded G2S. When the TME specialization cannot compile,
the existing pointer path is tried. The compiler cache stores compiled code;
a prepared handle owns its Q descriptor. Outlining the existing native LLVM
bindings into a private module keeps the pipeline below 2000 lines.

The selection covers 68 shapes / 2464 calls. Their weighted mean of
per-shape speedups was 1.33530× versus 1.32604× previously; their
count-weighted kernel latency change was −0.73%. The all-shape gain is
small. Separate same-card control/candidate checks on three selected shapes
in opposite execution order on two cards measured −0.02% and −0.20% latency.

Private and stock compilers each passed 183 functional cases and four core
benchmark records. All 414 frozen cases passed the old output atol 0.002
as well as the repository output atol 0.004; LSE atol stayed 0.0002.
Maximum absolute errors against the FP32 reference were 0.000268154
(output) and 0.00000190735 (LSE). The native MATE baseline and weights did
not change. The frozen private compiler's executable sections match the
prior pointer kernels and the device-tested TME kernel for the checked
K640 and K8320 cases. The stock compiler retains its original fallback.

## Paired native exponent, 2026-11-13

The large-query producer now evaluates two register-local FP32 exponent inputs
with the PH1 two-wide instruction used in
[FlagGems-vllm #843](https://github.com/flagos-ai/FlagGems-vllm/pull/843).
The existing half-score scaling, certified input range and conservative error
budget are unchanged. Denominator partials use the existing full-buffer store
API, preserving their layout and avoiding 1 KiB of conversion scratch. Both
probability buffers still fit in 192 KiB. Unused layout constants and the old
scalar exponent helper are removed; no host dispatch, workspace allocation,
barrier protocol or other kernel body changes.

| Full matrix | Shapes / calls | Count-weighted speedup |
| --- | --- | ---: |
| Previous runtime `90e1d1b4` | 414 / 13,920 | 1.19381429× |
| Paired exponent | 414 / 13,920 | **1.22897363×** |
| Large-query group, previous → current | 399 / 8,713 | 1.32247899× → 1.37406185× |

Two independent cards repeated seven alternating complete prepared-call pairs
on the same frozen B512 inputs. K640 eager latency improves by approximately
5%, and K8320 by approximately 3%; graph measurements improve on both cards.
Registers, spill slots and shared bytes remain unchanged for these cases.
A 195,590-value primitive check was bitwise equal to the original scalar
exponent, with maximum relative FP64 error 1.93e-7. Five executable-section
comparisons show that the cleaned source matches the measured prototype,
including K8193 tails and raw offsets beyond 4 GiB.

All 414 final-source cases pass the original all-row FP32 comparison, output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors remain
0.000268153846 and 0.000001907349. Private and stock FlagTree each pass 183
functional cases and a core benchmark with four complete JSON records.
The existing guard test now covers B128/K640 and K8193 for BF16 and FP16.
Additional checks pass 96 eager/graph comparisons with changed content,
indices, scales, unsafe rows and empty selections, plus 16 single-P-buffer
comparisons exercising bounded and ordinary exponent modes.

The 841 tracked runtime/test/other files match the validation snapshot;
this progress document is maintained separately. Full-file pre-commit,
54 CI helper tests, pin checks and diff checks pass. vLLM remains
0.24.0+empty, MATE 0.2.4, and private/stock compiler SHA256 values remain
`e01d6731...` / `c3b5d830...`. Full-matrix performance uses the private compiler;
no stock full-matrix or NVIDIA hardware performance claim is made.

The Dense two-consumer design from #945 and dedicated-copy design from #843
were also investigated. Tested adaptations did not improve complete calls and
are excluded. Stage profiling changed register allocation and added substantial
overhead, so its interval proportions are diagnostic only. The existing
isolated #1450 compiler rejected both the control and a subview prototype at
explicit SQMMA wait-layout verification; it was not installed or used for GPU
acceptance. No compiler change is included.

Evidence: `reports/mthreads-pr17-paired-exp-delivery-20261113` contains the
accepted source manifest, six complete full-matrix records, precision ledger,
environment and validation results. Rejected prototypes and diagnostic data
remain in `reports/mthreads-pr17-dense-followup-20261113`. The approved fork
base remains `6e951850a50ad176d644eb17e81ca9d3a5fb9c2a`.

## Saved validated checkpoint, 2026-11-12

The implementation is saved in the fork PR branch and in
`reports/mthreads-pr17-best-20261112/best-source-90e1d1b4.tar.gz`.
`BEST_RESULT.json` in that directory records the runtime commit, frozen PR base,
archive checksum, 841-file validation manifest, original weights, complete
matrix results and precision records, environments and check logs.

| Checkpoint item | Value |
| --- | --- |
| Runtime code commit | `90e1d1b40dc608f232892a4042431e454d685598` |
| Runtime SHA256 | `82b5936673f77fc0c9f9c18590d78f1e9994b3c36503b343db872d7ddc1a738e` |
| Scheduling SHA256 | `86a9bbce5fe01b955ce24509ed5d323e6afbfb25d711f8ace8890a5769649d46` |
| Full measured weighted speedup | **1.19381429×** |
| Frozen shape/count coverage | **414 / 13,920** |
| Original output/LSE tolerances | **0.002 / 0.0002** |

The code retains consumer-owned long-TopK prefetch and the validated B16/K640
single-split prefetch. Later wide-tile, page-grouping, head-size and packed-offset
candidates have not provided an acceptable improvement. The common-high
numerical prototype has only passed targeted output checks; its bound audit,
performance and full acceptance are incomplete. It is excluded from this
checkpoint. The experimental report directories are preserved for later work.

The documentation checkpoint can advance the PR head while the runtime stays
byte-identical to `90e1d1b4`. The approved base remains
`6e951850a50ad176d644eb17e81ca9d3a5fb9c2a`.

## Current single-split KV prefetch, 2026-11-12

The existing B16/K640 single-split route now uses two KV buffers. It copies the
first page before the loop and issues the next page's bounded 16-byte copies
before computing current QK and both current PV terms. The end-of-page copy
wait, local-memory fence and CTA barrier establish next-page readiness and
old-slot release. The probability-store barrier and original FP32 repair are
preserved. Thread count, page schedule, arithmetic, error thresholds, global
workspace and kernel launch count are unchanged.

Host dispatch requires the existing single-split route, at least 192 KiB shared
memory and valid uint32 token offsets. Resource or offset limitations retain
the original single-buffer path through a default-false constexpr. The
independent host page-grouping helper moved into a backend-local scheduling
module; its AST is identical. Shared views stay inside the kernel because the
frozen compiler cannot lower their transformed types across helper calls.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup | 1.19331369× |
| Current full weighted speedup, 414 shapes / 13,920 calls | **1.19381429×** |
| Full speedup with previous unaffected timings | 1.19587549× |
| Changed group | One shape / 1,982 calls |
| Changed-group speedup, before → after | 1.00216462× → 1.02015672× |

Two-card prototype pairs show approximately 4% complete-call gains for B16/K640.
A final-source seven-pair repeat measures eager 36.840 → 35.520 µs and graph
43.243 → 41.703 µs. Registers remain 240 and spill slots 418, including the
existing cold repair path; shared storage grows from 98,816 to 164,352 bytes.
Other probed B4 routes have no stable complete-call gain and retain their old
executable code.

All 414 final-source cases pass the original all-row FP32 comparison, output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors remain
0.000268153846 and 0.000001907349. Private and stock FlagTree each pass 179
functional cases and a core benchmark with four complete JSON records.
Additional final-source checks pass 144 eager/graph mutation comparisons
covering BF16/FP16, unsafe rows, invalid indices, tiny scales and a simulated
128-KiB resource limit, plus two compile-fallback comparisons. Eleven
unaffected executable-section comparisons match the previous runtime,
including raw offsets beyond 4 GiB and large-path buffer fallbacks.

The 841 tracked runtime/test/other files match the immutable validation
snapshot; only this progress document is maintained separately. Guard arithmetic
is byte-identical, scheduling-helper AST is identical and production torch
calls are unchanged. vLLM remains 0.24.0+empty, MATE 0.2.4, and private/stock
compiler SHA256 values remain `e01d6731...` / `c3b5d830...`.

Separate B16/K8320 metadata experiments preserve hoisted scales but reload IDs
and KS per page. Without prefetch, eager regresses 139.940 → 157.100 µs despite
removing three spills. With same-CTA prefetch, eager regresses 140.040 →
173.840 µs and introduces 12 spills. Both pass original strict fixtures and
actual fragment-flag checks, but are rejected. Their results are not mixed
with the accepted full-matrix measurement.

Evidence: `reports/mthreads-pr17-single-prefetch-20261112`,
`reports/mthreads-pr17-single-prefetch-delivery-20261112` and
`reports/mthreads-pr17-short-uncached-20261112`.

## Rejected short-query follow-ups, 2026-11-12

Five report-only candidates were measured against runtime `40e276ac`, whose
SHA256 remains `4d72068c85289983e2f6d74ca743328faaf490125286822fb071dae44a2ac35a`.
Each candidate passed the original frozen all-row output `atol=0.002` and LSE
`atol=0.0002` checks for B16/B4 at its listed TopK. Seven alternating complete
prepared-call pairs were measured separately in eager and graph mode for each
case. All five candidates are rejected; no runtime change is retained.

| Candidate | TopK | B16 eager before → candidate | B4 eager before → candidate |
| --- | ---: | ---: | ---: |
| Two-role, 64-token pages | 8320 | 140.520 → 350.520 µs | 84.440 → 156.500 µs |
| Outlined scalar guard | 8320 | 140.240 → 149.940 µs | 86.160 → 88.500 µs |
| Outlined guard and explicit matrix layouts | 8320 | 140.080 → 145.520 µs | 85.600 → 87.640 µs |
| 16 heads, four warps | 640 | 37.560 → 72.720 µs | 36.320 → 82.640 µs |
| 16 heads, eight warps | 640 | 36.640 → 59.600 µs | 35.720 → 48.600 µs |

The two-role design keeps 60 CTAs, carries buffer phases across query
fragments and uses consumer-owned KV prefetch. Its 128-token-page implicit
variant requires 229,376 shared bytes, above the 196,608-byte limit, and was
never launched. The 64-token-page version fits in 147,456 bytes and passes an
integer schedule audit over 208 configurations and 136,816 query-pages, but
raises main-kernel spill slots to 163/128 and roughly doubles graph latency.
Final LLVM has CTA synchronization only before the role branch; internal
reduction synchronization uses a named barrier with 16 arrivals.

The outlined guard retains all 27 original guard/staging statements with
identical AST after excluding the relocated Q store and accumulator
initializations. Three scale fields use a power-of-two-padded shared buffer.
Explicit probability and accumulator layouts recover some outlining cost but
still regress B16 complete calls. Final candidates compile; intermediate
layout, call-type and allocation failures were corrected before GPU testing.

The smaller K640 head tiles increase B16's launch from 32 to 64 CTAs and
reduce per-CTA shared memory to 88,576 bytes. Four warps introduce substantial
register pressure. Eight warps remove the B4 hot-path spills, but its main
kernel still regresses from 19.256 to 27.829 µs in graph mode. More CTAs and
lower shared-memory use do not establish higher throughput. B16 retains the
same inline FP32 repair and one timed kernel launch. Since that path leaves
its unused flags unwritten, eligibility is checked by a separate untimed
invocation of the identical kernel with inline repair disabled. Those are
diagnostic guard-probe flags; B4 uses its actual partial flags.

Raw logs preserve occupancy rejections and an initial diagnostic assertion
about unused inline-repair flags. They are not performance results. All five
final GPU jobs end with exit zero. Compiler SHA256 values remain unchanged:
private `e01d6731...` and stock `c3b5d830...`. No broader device validation or
new full-matrix performance result is claimed for rejected candidates. The
accepted full result remains **1.19331369× over 414 shapes / 13,920 calls**,
below the **1.3×** target.

Evidence: `reports/mthreads-pr17-small-ws2-20261112`,
`reports/mthreads-pr17-ordinary-layout-20261112` and
`reports/mthreads-pr17-small-head16-20261112`.

## Consumer-owned long-TopK prefetch, 2026-11-12

For TopK>1024 with double probability buffering and the existing certified
compiler capability, the PV consumer now copies the next KV page while the
producer computes current QK and probabilities. The consumer acquires the
existing empty slot, completes the original bounded copy, then publishes a
copy-ready barrier. The producer waits that barrier before QK. Empty-slot
release still follows both PV waits, protecting KV and probability storage.
K640, short-query kernels and the single-buffer fallback retain their previous
copy schedule. No workspace, numerical arithmetic, threshold or compiler
binary changes are required.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup | 1.11197191× |
| Current full weighted speedup, 414 shapes / 13,920 calls | **1.19331369×** |
| Full speedup with previous unaffected timings | 1.19238015× |
| Changed group | 133 shapes / 4,045 calls |
| Changed-group weighted speedup, before → after | 1.24047471× → 1.51718242× |

Two independent cards repeated seven alternating complete prepared-call pairs
on frozen B512/K8320 inputs:

| GPU | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| 0 | 2.75810 → 2.30252 ms | 2.75952 → 2.26469 ms |
| 2 | 2.70678 → 2.25238 ms | 2.73669 → 2.25092 ms |

The main kernel retains 120 registers, 32 total warps and 196,608 shared bytes;
spills increase from zero to 18. Retention is based on complete-call gains.
A split-copy variant reduces spills only to 17 without a stable additional
gain, and is excluded. K640 regressed under consumer prefetch, so the retained
scope preserves its original executable code.

All 414 final-source cases pass the original all-row FP32 comparison, output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors remain
0.000268153846 and 0.000001907349. Private and stock FlagTree each pass 179
functional cases and the core benchmark, with four complete core JSON records
per environment. The actual imported vLLM remains 0.24.0+empty and MATE 0.2.4.
Full-matrix performance is measured on the existing private compiler.

Final-source checks also pass 72 repeated tail/content/scale/index eager/graph
comparisons, eight double/single-buffer comparisons, 26 even-page comparisons
and four optional-workspace fallbacks. These cover K8193 tails, K1153/K1536
even page counts, selected-only and full-cache metadata, mixed unsafe rows,
tiny scales, empty selections and raw-index fallback.

Seven executable-section comparisons pass: five unaffected configurations
match the previous runtime, and changed K8320/K8193 configurations match the
measured prototype. The 840 tracked runtime/test/other files match the tested
snapshot; the progress document is maintained separately. Stock public-path
validation passes; forced unsupported stock TLE configurations are not claimed
as executable-equality successes.

Installed MATE source and generated code were captured without changing the
baseline. The actual MODEL1 scheduled path uses BM64/BN64, 640 main threads,
BF16 QK/PV operands, FP32 accumulation and a separate combine kernel. This
evidence guided the copy-role experiment. Rejected separate-repair testing
also showed that eliminating cold-path spills alone did not improve latency.

Evidence: `reports/mthreads-pr17-consumer-delivery-20261112`,
`reports/mthreads-pr17-consumer-copy-20261112`,
`reports/mthreads-pr17-baseline-study-20261112` and
`reports/mthreads-pr17-unfused-single-20261112`.

## K=640 short-query schedule, 2026-11-12

For B16/K640, two 32-head CTAs per query now process all five pages in one
split, using 16 warps and the existing in-CTA FP32 repair. This removes the
combine launch. For B1–4/K640, the existing page splits remain and each CTA
uses 16 rather than eight warps. The frozen matrix contains B4 and B16 in
these changed routes: two shapes and 2,402 calls.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`d1406fde`) | 1.09026451× |
| Final-source full weighted speedup, 414 shapes / 13,920 calls | **1.11197191×** |
| Full speedup with previous unaffected timings | 1.11049765× |
| Changed K640 shapes, before → after | 0.86164332× → 0.97889783× |
| First candidate full measurement, before host cleanup | 1.11438940× |

Seven alternating complete prepared-call pairs on frozen inputs measured:

| Shape | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| B16/K640 | 42.32 → 37.20 µs | 46.80 → 42.70 µs |
| B4/K640 | 35.36 → 31.52 µs | 33.23 → 30.83 µs |

The B16 main kernel uses 240 registers, 418 spills and 98,816 shared-memory
bytes. The complete call improves despite the spills. The B4 main kernel
uses 89 registers, zero spills and 91,648 shared-memory bytes. Neither result
is inferred from an isolated kernel timing.

All JIT kernels, FP8 QK, two-term FP8 PV, numerical certificates, thresholds
and repair arithmetic are unchanged. The single-split repair route is gated
by persistent-path support; disabling that compiler capability retains the
existing flagged repair. BF16 and FP16 checks exercise this route with actual
ineligible rows. Host cleanup preserved the scheduling plan on 1,260 metadata
combinations and every JIT AST.

The final source passed all 414 original all-row FP32 comparisons, with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349. Private and stock FlagTree each passed 179
functional cases and the core benchmark; each core JSON has four complete
latency/speedup records. The full performance matrix uses the existing private
compiler; no stock full-matrix speedup is claimed. The vLLM empty environment,
MATE baseline, compiler binaries, fixtures, weights and tolerances are unchanged.

The first final-source repeat reached the runner's eight-minute deadline with
partial records and is excluded from aggregation. Six new disjoint shards
completed successfully with the same benchmark parameters and final source.
Their source hashes and complete shape coverage were checked before aggregation.
The runtime and tests match 840 tracked files in the tested snapshot; the
progress document is maintained separately.

Further candidates were rejected:

| Candidate | Representative complete-call eager timing, before → candidate | Result |
| --- | --- | --- |
| Force inline repair | B16/K640: 37.40 → 38.52 µs | Spills increase from 418 to 460 |
| Ordinary masked loads and Triton dot | B16/K640: 37.60 → 51.36 µs; B16/K8320: 166.94 → 212.40 µs | Slower on all four tested shapes |
| Two-stage ordinary-load loop | B16/K640: 39.50 → 51.56 µs; B16/K8320: 168.36 → 224.00 µs | Slower on all four tested shapes |
| 16 heads with 16 warps | No device launch | Compiler cannot select a supported TLE SQMMA configuration |

The ordinary-load candidates passed the frozen comparisons and actual
fast-path flag checks. These diagnostics are excluded from production.

## Follow-up deep optimization, 2026-11-12

Eleven further executable variants completed 23 paired scenarios on the
existing private runtime. Each scenario used seven alternating complete-call
eager and graph pairs, frozen inputs, the original strict output/LSE checks
and actual fast-path flags. None provided a stable improvement for adoption.
The retained runtime and its 1.11197191× full-matrix result are unchanged.

| Candidate | Representative eager timing, before → candidate | Result |
| --- | --- | --- |
| Native exponential in short persistent queries | B16/K8320: 167.56 → 169.34 µs | Mixed results across three shapes |
| More token lanes with column warp groups | B512/K8320: 2757.34 → 2793.22 µs and 2707.42 → 3058.80 µs | Both layouts regress; spills increase to two |
| BN256 with one KV slot and matched producer layout | B512/K640: 349.52 → 638.88 µs | 69 spills |
| BN256 with compact denominator partials | B512/K640: 351.38 → 617.04 µs | 58 spills; still slower |
| Per-query large-batch persistent dispatch | B512/K8320: 2739.62 → 4320.12 µs | Fewer launches do not offset the main-kernel cost |
| Two 32-head CTAs per query with uncached metadata | B512/K8320: 2788.68 → 6457.92 µs | Lower resource use does not improve complete calls |
| Ordinary async copy with a zero buffer | B512/K8320: 2740.04 → 3049.70 µs | 12 spills |
| Precomputed 64-bit source pointers | B512/K8320: 2756.18 → 2971.00 µs | Eight spills; a higher global register cap does not remove them |
| Native masked copy with shared zero stores | B512/K8320: 2710.36 → 2797.50 µs | Zero spills, but about 3% slower |

The wide-tile investigation identified an extra 64 KiB QK layout conversion.
Matching the producer layout removed it and restored the original 192 KiB
resource limit. Lowered role code retained full-CTA barriers only in the
prologue. No compiler or hardware limit was changed.

Zero-buffer, pointer-staging and native masked-copy checks covered 60 tiles
and 3,932,160 bytes exactly, including all byte encodings, mixed valid/invalid
indices and both index representations. The first two also checked their
per-call zero initialization and metadata paths. These checks establish copy
correctness; the variants were rejected on complete-call performance.

A uniform-numerator diagnostic passed no head under its conservative guard:
the combined bounds were 0.00885–0.00987, above the unchanged 0.0038 limit.
A high-level masked async-load/store candidate compiled but generated no
direct async-copy operations in the installed compiler, and was excluded
before device testing. Neither result permits dropping numerical checks.

Raw sources, exit records, IR and timings are retained under
`reports/mthreads-pr17-deepopt-20261112` and its referenced report directories.
The subset trials do not establish stock-runtime or full-matrix acceptance
for any rejected candidate. Hardware counters were not collected in this
batch; compiled resources, lowered code and device event timings supplied
the profiling evidence. The 1.3× target remains unmet.

## Earlier native-exponential implementation

The baseline is native MATE 0.2.4 MODEL1 D512 on MTT S5000, with BF16 QK/PV
computation and FP8 KV-cache storage. The environment uses vLLM 0.24.0 empty.
The frozen matrix contains 414 shapes and 13,920 original calls. Speedups are
weighted by each shape's call count; the baseline, inputs and weights remain
unchanged.

| Measurement | Result |
| --- | ---: |
| Previous full weighted speedup (`45f58771`) | 1.07757864× |
| Native-exponential full weighted speedup (`d1406fde`) | 1.09026451× |
| Previous native-long result with previous unaffected timings | 1.08606611× |
| Previous long-TopK group | 133 shapes / 4,045 calls |
| Previous long-TopK speedup, before → after | 1.20944285× → 1.23763440× |
| New large K=640 group | 133 shapes / 4,253 calls |
| New K=640 speedup, before → after | 1.17237381× → 1.18580684× |

The retained copy-layout change uses eight token lanes per warp for every TopK greater than
128, extending the existing medium-TopK copy layout to long TopK. The producer
row-warp count stays at 16. At the 128-token tile size, these cover 128 distinct
token rows without duplicate copies. The long-TopK main kernel's spills drop
from 13 to zero; its 32 total warps and 196,608 shared-memory bytes are unchanged.

The previous producer-barrier removal and double probability buffering remain.
That update uses the bounded native exponential in the large producer
for K=640 and TopK>1024. Short queries and the medium short path retain their
existing polynomial. The existing native path already uses the same FP32 LLVM exp2 intrinsic. Valid
contributing half-scores lie in [-1, 1], so its base-2 arguments stay within
approximately ±2.8854 and never need expf's subnormal-output workaround.

A primitive check covered 195,590 uniform, finite raw-bit and boundary inputs;
results were bitwise equal to the existing native exponential and had maximum
relative error 1.93e-7 against CPU FP64. The original conservative polynomial
budget is retained. FP8 QK, two-term FP8 PV, all eligibility checks, numerical
thresholds, scale handling and whole-row repair are unchanged. No fast-math
flag or compiler change is required.

The earlier seven executable-section comparisons passed: three unaffected configurations
(short TopK 640/8320 and large TopK 128) match the previous runtime, and
three scoped long-TopK configurations match the measured prototype, including
single buffering and the TopK 8193 tail. The current main kernel still uses 120 registers, zero
spills, 32 total warps and 196,608 shared-memory bytes.

All 414 shapes passed the original all-row FP32 comparison with output
`atol=0.002` and LSE `atol=0.0002`. Maximum absolute errors were
0.000268153846 and 0.000001907349, respectively. Private and stock FlagTree each
passed 179 functional cases and the core benchmark, with four complete latency
and speedup records per core JSON. The current candidate passed 432 additional
eager/graph comparisons, 12 optional workspace fallbacks and eight compile/fallback
comparisons. Full pre-commit,
54 CI helper tests, CI pins and diff checks passed.

Two independent cards repeated seven alternating complete-call timing pairs
on the frozen B512/K8320 input:

| GPU | Eager before → candidate | Graph before → candidate |
| --- | --- | --- |
| 4 | 2.78416 → 2.71810 ms | 2.82393 → 2.75449 ms |
| 7 | 2.77632 → 2.71030 ms | 2.79833 → 2.72337 ms |

## Subsequent experiments

These are diagnostics on subsets, not new full-matrix acceptance results.
Timings below are median complete prepared calls. Each pair used the same
process, device and frozen input. Unless stated otherwise, the original strict
FP32 output comparison passed. Rejected variants are excluded from runtime code.

| Experiment | Representative eager timing, before → candidate | Decision |
| --- | --- | --- |
| Centered PV: M16 common sum plus M64 residual, shared result transfer | B512/K640: 0.35844 → 0.45376 ms; B512/K8320: 2.94458 → 3.27676 ms | Rejected; more spills and slower calls |
| Separate FP32 vector common-sum pass | B512/K640: 0.35878 → 0.72072 ms; B512/K8320: 2.94608 → 5.69728 ms | Rejected; additional KV pass dominates |
| Eight consumer warps, split M64 layout | B512/K640: 0.35100 → 0.45844 ms | Rejected; additional spills |
| Eight producer warps | B512/K640: 0.35824 → 0.56670 ms; B512/K8320: 2.93982 → 5.57360 ms | Rejected |
| Paired FP8 conversion option | B512/K640: 0.35788 → 0.40704 ms; B512/K8320: 2.93938 → 3.57004 ms | Rejected despite bitwise conversion equivalence |
| FP16 producer residual | B512/K8320: 2.93708 → 2.91568 ms | Not adopted; weak gain, no stable medium-TopK gain, additional rounding budget required |
| Shared PV scale across heads | B4112/K640: 2.59114 → 2.71448 ms; B512/K8320: 2.94180 → 3.02924 ms | Rejected; early weight scheduling also regressed |
| Short-query split warp-specialized pipeline | B16/K8320: 141.36 → 261.80 µs | Rejected |
| Reordered short-query split tasks | B16/K8320: 141.72 → 214.38 µs | Rejected |
| Fused metadata with reordered split tasks | B16/K8320: 141.80 → 224.84 µs | Rejected |
| Two raw KV buffers, uncached metadata | B16/K8320: 141.72 → 175.08 µs | Rejected |
| Move next-page copy after QK issue | B16/K8320: 141.54 → 171.46 µs | Rejected |
| Aligned packed query reduction | B16/K640: 42.40 → 41.60 µs, graph 45.88 → 46.18 µs; B16/K8320: 166.60 → 181.88 µs | Not adopted; mixed results and extra layout conversion |

The native-M32 eight-consumer variant exceeded shared-memory limits and was
not launched. The explicit M64N128 QK layout also exceeded the limit. A cached
suffix variant of the two-buffer kernel required 204,800 bytes, exceeding the
196,608-byte device limit, and was not device-tested.

Several explicit-layout variants failed compilation with the scalar-reduction
layout verifier. These initial experiments did not modify the compiler or
system environment; the later isolated compiler diagnostic is described below.
An initial reordered split trial silently used the legacy preparation fallback;
its timing was discarded. The corrected trial checked the actual fused-kernel
name before timing. New short-path verification also inspects the fragment
flags when repair is fused into combine, so output correctness alone cannot
be mistaken for evidence that the approximate fast path ran.

## Packed-query reduction work

The remaining candidates reuse the existing native four-value FP8-to-FP16
conversion to compute four absolute query values per word. FP16 subtotals are
converted to FP32, reduced and multiplied by 1.00154. Attention QK/PV arithmetic
is unchanged. The aligned short variant above is not retained.

Completed evidence:

- A device primitive check covered all 254 finite FP8 codes and 8,192 four-term
  groups. Native subtotals matched the CPU FP16 reference exactly.
- Shared query storage round-tripped its bytes exactly under an actual SQMMA
  consumer. The computed query L1 bound passed the upper-bound check.
- Exact rational analysis using the actual FP32 inflation gives a lower ratio
  of 1.00006306386 to the exact L1 sum, above the original FP32 summation upper
  ratio of 1.00003045890. Positive finite query-scale multiplication preserves
  that ordering. Four FP8 magnitudes sum to at most 1792, avoiding FP16 overflow.
- Unaligned packed preparation and a small shared-memory layout bridge compiled
  without a GPU, retaining 148,480 bytes of shared memory. This does not prove
  their device behavior or performance.

The previously pending unaligned short candidate, shared-memory bridge and
large-batch query-bound candidate completed the subset device checks below.
None was adopted. Any promising future candidate still requires dynamic
contents/scales/indices, tails, actual repair and fallback, private/stock
FlagTree, and the complete 414-shape matrix with original weights and tolerances.

On 2026-11-09, device launches were rejected by the occupancy guard while other
tasks held device allocations. Rejected launches are not completed tests. On
2026-11-10, testing resumed after fresh checks found available cards. The guard
was not relaxed and other processes were not stopped.

## Resumed device experiments, 2026-11-10

Eight candidates completed 27 fixed-input paired scenarios on the existing
private runtime. The complete prepared calls were measured in alternating
eager and graph pairs. All scenarios passed the original frozen comparisons
and checked actual fast-path flags. These are BF16 subset checks, not new
414-shape or stock-runtime acceptance results. No candidate had a sufficient
stable complete-call benefit for adoption.

| Candidate | Representative eager timing, before → candidate | Result |
| --- | --- | --- |
| Unaligned packed query L1 | B16/K8320: 166.08 → 216.96 µs | Rejected; spills increase from 3 to 72 |
| Packed shared-memory bridge | B16/K8320: 168.48 → 177.76 µs | Rejected |
| Packed large query-bound prepass | B512/K640: 350.18 → 347.40 µs | Query stage improves, but complete graph timing regresses slightly |
| Outlined fast combine | B16/K640: 42.16 → 41.52 µs; B16/K8320: 166.38 → 167.44 µs | Mixed benefit; not adopted |
| Normalized FP16 split buffers | B4/K8320: 84.52 → 89.42 µs | Rejected |
| Unnormalized FP16 split buffers with added rounding/overflow guards | B16/K8320: 167.54 → 167.72 µs | No stable benefit |
| Existing separate combine/repair fallback | B16/K640: 42.70 → 54.24 µs | Rejected |
| Shared query-L1 prepass and async Q copy | B16/K8320: 140.68 → 141.82 µs | Rejected; extra launch offsets preparation savings |

Private and stock FlagTree each compiled the control and packed large-query
bound kernels for TopK 128, 640 and 8320 without exposing any GPU: 12 successful
compilation checks. These checks establish compilation only.

A separate whole-query FP32 diagnostic examined single-term FP8 PV on 32
frozen rows. Maximum observed absolute errors were approximately 0.00096 for
TopK 640 and 0.00073 for TopK 8320. However, the quantization-only range bound
was 0.00954–0.01080, exceeding the existing 0.0038 guard. Small observed errors
therefore do not justify removing the second PV term. A Cauchy–Schwarz bound
was 0.00262–0.00303, but excludes QK, accumulation and denominator errors and
requires KV column norms. Its complete numerical certificate and the cost of
computing those norms remain unverified. No single-term kernel is adopted.

## Additional diagnostics, 2026-11-11

The following results are excluded from the retained runtime change:

- Conditional second-PV execution passed frozen comparisons but raised spills
  to 116/172 and substantially slowed both tested TopK values.
- Alternative QK layouts required 229,376 shared-memory bytes, exceeding the
  196,608-byte limit. A 64-token variant fit but slowed complete calls.
- Moving a shared common PV into the producer, or storing it between pages,
  still increased spills and slowed complete calls. Removing residual PV
  entirely gave no medium-TopK benefit and only about 4% at long TopK before
  the missing omission certificate was added; it was not adopted.
- Early next-page KV copy, polynomial-only compilation, four-token-lane copy
  and normalized selected key scales had no stable benefit across their
  intended cases. The polynomial-only variant would also lose the native-exp
  fast path for some inputs. These variants were not adopted.

A report-only per-element snapshot matched every KV square on all rows/pages
of TopK 640 and 8320 over three replays each. The original cross-warp column
norm reduction still produced occasional wrong values. Moving its token-axis
reduction within a warp passed the sampled FP64 norm and inflated upper-bound
checks. Combining four adjacent columns reduced overhead, but it remained
slower than the original operator. The cross-warp root cause is unproven, and
none of this norm work is used by the retained kernel.

A separate scalar-reduction compiler reproducer identified layout inference
being applied to scalar results. A fix in an owned experimental source tree
compiled and passed four device patterns; three control executable sections
were byte-identical before and after it. The actual FlagTree checkout, private
runtime and stock installation were not changed. That isolated fix is not a
dependency or part of this PR, and establishes no operator speedup.

A mixed 3/4-fragment short-query plan passed 79,518 coverage/capacity checks
and the frozen B16/K640 oracle. It used 60 rather than 48 CTAs with the same
maximum two pages per fragment, but did not speed up the main kernel and
increased combine time. It was rejected.

Removing the explicit end-of-page copy barrier was also rejected. The private
compiler retained an implicit input barrier only in double-buffered kernels;
the single-buffered versions lacked that edge and were not device-tested.
The verified double-buffer diagnostic passed frozen comparisons but showed no
speedup. The accepted runtime keeps the original explicit copy synchronization
in both buffering modes.

## Evidence and reproduction

The development workspace retains raw logs, source prototypes, before/after IR,
exit records and result JSON files under:

- `reports/mthreads-pr17-headsplit-delivery-20261112`: final source manifest,
  six complete matrix JSON files, strict precision records, before/after
  aggregation, private/stock validation and rejected prototype logs.
- `reports/mthreads-pr17-headsplit-w16-20261111`,
  `reports/mthreads-pr17-b4h32w16-20261111`: retained scheduling pairs and
  BF16/FP16 mutation, repair and compile-fallback checks.
- `reports/mthreads-pr17-regular-pages-20261112`,
  `reports/mthreads-pr17-head16-w16-20261112`: rejected ordinary-load and
  head-count diagnostics.
- `reports/mthreads-pr17-native-delivery-20261111`: previous validated source
  manifest, four full-matrix JSON files, precision summary, private/stock core
  JSON files, mutation/tail/fallback logs and delivery comparison.
- `reports/mthreads-pr17-native-exp-20261111`: primitive equivalence, executable
  scope checks and two-card prepared-call pairs for the native exponential.
- `reports/mthreads-pr17-copy8-delivery-20261111`: previous validated copy-layout
  runtime and full matrix used for the current comparison.
- `reports/mthreads-pr17-publish-20261109`: previous validated runtime and full
  matrix used for the before/after comparison.
- `reports/mthreads-pr17-centered-20261109`,
  `reports/mthreads-pr17-consumer8-20261109`,
  `reports/mthreads-pr17-burst-20261109`,
  `reports/mthreads-pr17-fixed-scale-20261109`: rejected data-flow experiments.
- `reports/mthreads-pr17-short-ws-20261109`,
  `reports/mthreads-pr17-short-prefetch-20261109`: rejected short-query pipelines.
- `reports/mthreads-pr17-packed-qnorm-20261109`: primitive checks, rounding proof
  and pending packed-reduction candidates.
- `reports/mthreads-pr17-resume-20261110`: resumed paired trials, CPU compilation
  matrix, single-term PV bound diagnostics and rejected source variants.
- `reports/mthreads-pr17-large-prefetch-20261111`: retained copy-layout paired
  evidence and rejected copy/exponent schedules.
- `reports/mthreads-pr17-norm-barrier-20261111`,
  `reports/mthreads-pr17-producer-common-20261111`,
  `reports/mthreads-pr17-common-only-20261111`,
  `reports/mthreads-pr17-keyscale-20261111`: additional rejected diagnostics.
- `reports/mthreads-pr17-scalar-layout-20261111`: isolated compiler reproducer,
  patch, binary manifest and before/after controls.
- `reports/mthreads-pr17-balanced-20261111`,
  `reports/mthreads-pr17-copy-sync-20261111`: rejected fragment/synchronization
  diagnostics; neither is included in the current runtime.

Validated operator commands:

```bash
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 --quick \
  tests/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 \
  benchmark/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py \
  --level core --warmup 1 --iter 1 --record json --output benchmark_sparse_fp8.json
python -m pytest -q -m flash_mla_with_kvcache_sparse_w8a8_fp8 \
  benchmark/test_flash_mla_with_kvcache_sparse_w8a8_fp8.py \
  --level comprehensive --warmup 25 --iter 100 --record json \
  --output benchmark_sparse_fp8_full.json
```

The fork PR base remains `6e951850a50ad176d644eb17e81ca9d3a5fb9c2a`.
