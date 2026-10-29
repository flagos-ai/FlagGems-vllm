# NCU follow-up and reuse boundaries, 2026-10-26

The follow-up collected **20 NCU reports / 45 kernel actions** and separately
repeated paired eager/Graph microbenchmarks. Synchronization, local-memory
traffic and a shared-memory occupancy threshold support rejecting the tested
WS/deeper-ring implementations. Exact FlagAttention #85 can run through its
original public entrypoint in the isolated harness, but the result supports
**selective reuse**, not a blanket claim that it is incompatible or slower.

The original six-variant result remains `REJECT_NO_INTEGRATION`. This follow-up
revisited WS consumer4/ring2, non-WS ring2/ring3 and exact #85; it did not repeat
all six variants or promote any production implementation.

- [Full report in Chinese](REPORT.zh-CN.md)
- [Copy-ready reuse explanation in Chinese](REUSE_EXPLANATION.zh-CN.md)
- [Derived metrics and timing summaries](data/metrics.json)
- [Independent FP32 oracle checks](data/oracle-control.json)

## Matched comparisons and their limits

All comparisons use one exclusive H100 80GB HBM3 / SM90 / 132 SM, q8192,
seq16384, GQA8, Hkv1, D128, page128 and TopK16, with identical input values,
indices and page tables within each A/B pair. Torch is `2.13.0+cu129`,
FlagTree `0.7.0` / Triton `3.6.0`, using the same frozen minimal FlagTree #1415
load-mask backport as the original trial; [toolchain hashes](../data/toolchain.json)
remain unchanged. NCU is `2025.2.1`.

| Comparison | KV storage / QK / PV | What it establishes |
|---|---|---|
| Frozen baseline vs WS/non-WS pipeline | Both FP8 E4M3 / BF16 / BF16 | Matched deployment-precision pipeline comparison |
| Frozen baseline vs original #85 control | Both BF16 / BF16 / BF16 | Matched storage/compute comparison; not an FP8 deployment ranking |
| Frozen baseline vs original #85 FP8 entrypoint | Both store FP8, but baseline QK is BF16 and #85 QK is FP8; PV is BF16 | Different compute contracts: numerical check only; no accepted timing score or NCU range |

The BF16 control converts the *same FP8 fixture values* exactly into BF16;
thus input values are preserved while storage width and available dispatch
change. The frozen serving precursor has SHA256
`1ab42fc298abf8a950d02ac5204869ed0f2eeade5185b17b58416e1be845a789`;
these are not GPU measurements of #979's packaged wrapper.

Exact #85 is pinned at `027ebdc9eb23e37aa3238d4a8b09296bc64e0dd8`, source SHA256
`6c7bb5b5768b6c7c925da9457c20d812195a5b7f66c2fc9608d071d86ae64a60`.
Its kernels and launch dispatch are unmodified. The isolated harness supplies
only the `flag_attn.utils.current_platform` import/query stub; prefill does not
use its PDL query. This does not constitute full serving-framework integration.

All latency values below are **ordinary full-wrapper Graph median microseconds**.
Timing is separate from NCU: 30 warmups, 3 rounds, 100 samples per arm per round,
alternating A/B, in both eager and Graph modes. Twelve accepted pairs contribute
14,400 timed calls. NCU replay durations in the exports are diagnostic only.

## Pipeline decomposition

| Matched FP8-storage comparison | Baseline / candidate Graph us | NCU evidence in the corresponding main kernels |
|---|---:|---|
| Compact-union vs WS consumer4/ring2 | 318.72 / 645.26 | Barrier stall per active warp-cycle 20.06% -> 46.30%; local request traffic 2 MiB -> 1.286 GiB; Tensor Core active elapsed 24.22% -> 11.51% |
| Non-WS ring2 vs ring3, independent scattered | 1555.54 / 2475.87 | Shared bytes/block 116312 -> 149100; shared-memory resident-block limit 2 -> 1; occupancy 11.82% -> 6.25%; local request traffic 0 -> 1.184 GiB |

The second row compares two candidates directly, not either candidate with
the frozen baseline. Ring2 still regressed against its paired frozen baseline
(1493.76 -> 1555.54 us). Candidate outputs passed exact element equality to
the frozen baseline before timing.

Local traffic is L1TEX load/store sectors multiplied by 32 bytes, **not DRAM
bytes**. Barrier percentages describe active warp-cycles, **not a fraction of
total kernel wall time**. In the compact comparison, achieved occupancy stays
near 12.5% while DRAM active elapsed stays near 3.3-3.5%; the evidence points to
synchronization/local traffic, not saturated HBM bandwidth, in this warm-input
trial. WS's largest sampled synchronization hotspot maps to the kernel entry;
it cannot be assigned to one exact Python barrier statement from that mapping.

## Exact #85 BF16 control

Speedup below is frozen-baseline median / #85 median; >1 favors #85.

| Index distribution | Frozen baseline us | #85 us | Speedup |
|---|---:|---:|---:|
| Compact/recent | 249.66 | 619.55 | 0.403x |
| Correlated8 | 1570.16 | 926.43 | 1.695x |
| Independent scattered | 1419.44 | 981.39 | 1.446x |

For compact inputs, #85 takes 2.48x as long. NCU shows 8192 CTAs vs 512,
141468736 vs 15074870 L2 TEX-source read sectors (**9.38x**), and Tensor Core
active elapsed 11.25% vs 30.44%, despite higher occupancy (36.37% vs 12.48%).
This supports the importance of cross-query KV-page reuse; the sector ratio is
not an HBM byte ratio or a universal theoretical factor.

At this exact BF16 shape #85 actually dispatches its retained
`_gqa_sparse_fwd_tle_upstream_kernel` / half-KV loop. The result is **not** a
measurement of #85's new prefetch kernel in isolation. It measures the public
entrypoint's real dispatch choice.

The scattered results are deliberately retained: **#85 wins both BF16
scattered controls**. The frozen baseline disables its FP8-only sparse-wide arm
for BF16 storage, so this is not a same-storage comparison against the deployed
FP8 sparse-wide path. It does not justify a universal ranking of either operator.

BF16 KV with scattered TopK is a potential deployment mode when FP8 KV is
disabled for precision or platform capability. These are synthetic controls,
not evidence that this storage/index combination is active in the current
deployment. The indexer's separate BF16 KV cache must not be confused with the
main attention KV cache measured here.

## Prefill FP8 entrypoint precision check

The existing prefill contract stores FP8 KV and computes BF16 QK/PV. Original
#85 instead quantizes Q and uses FP8 QK when its FP8-storage path is selected.
Against an independent FP32 literal oracle with
`abs(error) <= 0.004 + 0.04 * abs(reference)`, all three frozen-baseline fixtures
pass; #85 has respectively 635 / 632 / 525 exceeding elements out of 8388608,
with max absolute errors 0.015481 / 0.012768 / 0.012350.

The BF16 compact control passes the literal oracle for both implementations.
The FP8 results only reject #85 under **this trial's BF16-QK numerical contract**;
they do not establish a violation of upstream's FP8 accuracy requirements or a
model-quality regression. The three rejected FP8 arms were not timed and exit
before NCU capture starts. No speedup or counter result is claimed for them.

## Decode is a separate decision

The BF16 scattered results above are **q8192 prefill**, not decode measurements.
This follow-up did not benchmark #85 decode against the current native/tuned
split-K decode policy. The prefill union-reuse argument does not establish a
decode ranking.

In exact #85, the general decode path converts FP8 KV to Q's dtype before QK;
it does not inherit the prefill path's Q-to-FP8 quantization. Its specialized
BF16 fused-decode eligibility requires parallelism >=512 and an exactly filled
head tile of at least 16; GQA8 does not qualify. Any reuse decision therefore
needs matched KV storage, QK precision, active batch/query length, split-K and
Graph timing, plus numerical/replay checks. No extra decode gain is claimed.

## Reuse explanation

The packed-KV interface can be adapted. The current choice is to keep the
existing main path and assess #85 per input/precision domain: cross-query union
reuse wins the matched compact large-q control, whereas the FP8 entrypoint
changes the compute contract. Adding explicit TLE/TMA rings to the current
algorithm has also been tested; the additional synchronization, local traffic
and occupancy costs erased the hoped-for gain. This is a decision against a
wholesale replacement, not against TLE/TMA or all reuse of #85.

## Published evidence and reproduction

- `data/ncu-summary.json.gz` retains all action summaries.
- `data/ncu-exports/*.json.gz` retain the exact API-exported metric JSON and
  source-stall text, embedded with their SHA256 identities. They were generated
  using NCU's Python API on the profiling environment, not parsed from screenshots.
- `data/timings/*.json.gz` retain byte-identical raw timing/rejection receipts.
- `measured_sources.tar.gz` contains exact GPU-execution sources, including
  original #85 and the fixture/gate/benchmark harness. Receipt inventories also
  mention analysis-only helper files, which are not GPU dependencies.
- Raw `.ncu-rep` binaries are not committed; their hashes are recorded in each
  export and action summary. CPU verification rechecks the published exports,
  not unavailable binary contents. The binaries were hash-checked locally.
- `MANIFEST.json` records publication, source and original timing identities.

From the repository root, CPU-only verification and optional preparation:

```bash
python3 benchmark/experiments/minimax_m3_tle_pipeline/ncu-20261026/verify.py
python3 benchmark/experiments/minimax_m3_tle_pipeline/ncu-20261026/verify.py \
  --prepare /tmp/minimax-ncu-followup
```

In an independently provisioned exclusive H100 environment matching the pinned
compiler, make FlagTree importable, use a separate Triton cache, then run:

```bash
cd /tmp/minimax-ncu-followup
python3 scripts/profile_case.py --arm nows2 --mode scattered --bench
ncu --set detailed --section SourceCounters --profile-from-start off \
  --cache-control none --clock-control none -o reports/new-nows2-scattered \
  python3 scripts/profile_case.py --arm nows2 --mode scattered
```

Use a new output name. NCU metrics must be enumerated on the actual device.
The captured detailed/source runs covered all kernels in one warmed wrapper
call; the six targeted runs selected the relevant main kernel. Neither clocks
nor driver-wide counter permissions were changed during measurement.

## Remaining boundaries

No small-q/decode sweep, full context matrix, cold/rotating KV, multilayer or
model end-to-end comparison was run in this follow-up. Graph timing replays
fixed inputs; this round did not repeat the earlier A->B->A mutation/stress gate
and does not extend it to #85. Compute-sanitizer and PM-sampling causal timelines
remain NOT_RUN. The counters support the diagnoses but do not isolate the
independently removable wall-time share of each cost.

Production code, runtime registration, tuning tables, deployment images and
#979 are unchanged. The GPU capture task was stopped and released. Publication
of this follow-up starts no new GPU work.
