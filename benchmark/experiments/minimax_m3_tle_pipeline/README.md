# MiniMax M3 BF16 TLE/TMA pipeline experiment

**Decision: `REJECT_NO_INTEGRATION`.** All six measured implementations were
slower than the frozen serving baseline in every timed eager and CUDA Graph
cell, in all three rounds. This draft preserves the implementations and evidence
for review. It does not enable an operator or change production dispatch.

## NCU follow-up, 2026-10-26

The [NCU report and matched #85 controls](ncu-20261026/README.md) add 20 profiler
reports / 45 kernel actions, raw API exports, 14,400 separately timed wrapper
calls and a [copy-ready reuse explanation](ncu-20261026/REUSE_EXPLANATION.zh-CN.md).
Synchronization/local traffic and a shared-memory occupancy threshold now have
measured counter support. The follow-up distinguishes matched FP8-storage BF16
pipeline tests, matched BF16-storage #85 controls, and unscored #85 FP8-QK
precision checks. BF16 scattered controls favor #85; the frozen baseline's
FP8-only sparse-wide arm is disabled there. Those controls are not the current
deployment configuration and do not rank the deployed FP8 sparse-wide path.

## Relationship to the existing PR

This branch starts at attention draft
[FlagGems-vllm #979](https://github.com/flagos-ai/FlagGems-vllm/pull/979), commit
`fead47fdc6d4b81cc73b0d6488834dce75d95bf1`. All incremental files are inside this
experiment directory; the existing operator, exports, configs and tests are
unchanged. The PR description contains a pinned commit comparison that isolates
this directory from the unmerged parent PR.

The experiment revisits TLE producer/consumer scheduling and TMA, following
[FlagAttention #85](https://github.com/flagos-ai/FlagAttention/pull/85), pinned at
`027ebdc9eb23e37aa3238d4a8b09296bc64e0dd8`. It preserves BF16 QK and PV rather
than transplanting that PR's FP8 compute path.

The **measured baseline is a frozen serving precursor with later sparse-wide
support**, SHA256
`1ab42fc298abf8a950d02ac5204869ed0f2eeade5185b17b58416e1be845a789`.
It is not byte-identical to #979's packaged API. The five common JIT functions
are AST-identical; the serving version also has `_wide_union_sparse_kernel`,
`_span_split_kernel` and `_bitmap_or`. See the
[comparison guide](comparisons/README.md), complete source diffs and
[machine-readable AST comparison](comparisons/pr979-baseline.json).
These timings are not a GPU validation or benchmark of the #979 package wrapper.

## Original six-variant results, 2026-10-23

H100 80GB HBM3, SM90; Torch `2.13.0+cu129`; FlagTree `0.7.0` / Triton `3.6.0`
with the frozen minimal load-mask dependency backport from FlagTree #1415.
[Toolchain hashes](data/toolchain.json) identify the actual compiler and TLE API.

The following values are **baseline median / candidate median CUDA Graph GPU
time**, so greater than 1 would be faster. Query length is 8192, sequence length
16384, GQA8, one KV head, D128, page128 and TopK16. Q/output are BF16; KV storage
is FP8 E4M3, converted to BF16 before QK/PV; accumulation is FP32.

| Measured implementation | Compact/recent | Correlated scattered (groups of 8) | Independent scattered |
|---|---:|---:|---:|
| Warp-specialized, 4 consumer warps, raw 2-slot ring | 0.4927x | 0.5632x | 0.6537x |
| Warp-specialized, 4 consumer warps, raw 3-slot ring | 0.4775x | 0.5536x | 0.6426x |
| Warp-specialized, compact 8 / wide 4 consumer warps, raw 2-slot ring | 0.5693x | 0.6125x | 0.6532x |
| No warp specialization, full-row 2-slot ring | 0.6150x | 0.7538x | 0.9621x |
| No warp specialization, full-row 3-slot ring | 0.5966x | 0.5963x | 0.6046x |
| Warp-specialized, producer BF16 conversion, 2-slot ring | 0.6241x | 0.6824x | 0.7453x |

The best Graph cell is still **3.93% slower** (`0.96214658x`); the best eager
cell is `0.96095234x`. The 2-slot WS and non-WS variants were also timed at
q8153 for all three distributions, with the same negative conclusion.
[summary.json](data/summary.json) contains absolute microseconds, p10/p90,
per-round ratios, eager host enqueue statistics and receipt hashes.

Protocol: same inputs/device/compiler, alternating A/B order, 3 rounds, 30
warmups and 100 samples per arm per round, separately in eager and Graph modes.
Each timed arm is the complete `attn_auto` call including dispatch flags,
compact-union, sparse-wide and selected fallback/merge work. There are 24
variant/shape cells and **28,800 timed wrapper invocations**, excluding warmups.
Host enqueue and drained paired-round wall time were measured independently;
CPU cost was not inferred by subtracting CUDA events. These are repeated-input
microbenchmarks, not cold-cache or production-weighted throughput estimates.

## Correctness and rejected attempts

Accepted representative eager checks cover q8192 compact/correlated and q8153
independent scattered for each variant. Each has zero element differences
against the frozen baseline, finite outputs, output redzones, ten-repeat
determinism, and the unchanged native-vLLM / independent FP32 literal oracle
thresholds (`atol=0.004`, `rtol=0.04`). The literal oracle is a tolerance check;
the frozen baseline is the exact-parity reference.

Graph checks mutate Q, KV, TopK, block table, CU lengths, sequence/prefix lengths
and K/V scales in place at fixed addresses, through A -> B -> A and empty
states. Two streams use separate graph flag/workspace allocations. Redzones,
restored outputs and repeated replays pass. In total, **22 graphs and 2,200
stress replays** were checked. WS consumer4/ring2 used sequence length 32768;
the other variants used 16384 and 32768. These are representative gates, not a
proof for the whole supported serving domain.

Two failed attempts remain visible:

- [Initial q8192 attempt](data/correctness/smoke-s2.json): 682 differing elements
  out of 8,388,608, maximum absolute error `0.0009765625`. A fresh zero-C PV WGMMA
  followed by an add changed the frozen compiler's operation order. Passing the
  scaled prior accumulator as PV WGMMA C restored large-q parity.
- [q512 compact, four consumer warps](data/correctness/correctness-s2.json):
  three differing elements, maximum absolute error `0.00048828125`. Its reduction
  layout differs from the baseline's eight-warps compact path. This small-q
  configuration was rejected and contributes no accepted timing result.

## Implementation and compiler evidence

Readable implementations are [tle_pipeline.py](kernels/tle_pipeline.py) and
[tle_nows.py](kernels/tle_nows.py). Their function ASTs match the final measured
source revision. The archived experimental INT8 branch is outside this BF16
trial and is not a supported mode of these kernels.

- Real TMA copies, BF16 WGMMA and explicit wait/mbarrier protocols are used.
  The rings contain 2 or 3 slots; launch `num_stages=1` does not describe ring
  depth. The three-slot version allocates four barrier records to meet the API's
  power-of-two constraint and uses slots 0..2.
- The non-WS variant copies an entire packed K/V row in one TMA transaction and
  uses one ready barrier. Slot reuse waits for consuming WGMMA completion.
- Producer conversion moves FP8-storage-to-BF16 conversion into the producer;
  QK/PV operands and page-wise softmax/rounding semantics remain BF16.
- The frozen baseline TTGIR already issues next-page async copies between PV
  commit and wait. PV waits before loop yield. There is no evidence here for
  overlapping PV_i with QK_i+1. The
  [IR audit](data/baseline-ir-audit.json) records the inspected full-file hash
  and line numbers; the full original compiler cache is not included.
- The [resource audit](data/compiler-resources.json) summarizes 48 compiled
  cache entries. WS4 compact has 528-632 B stack frames and LDL/STL instructions;
  non-WS ring3 wide has 96-104 B stack frames; non-WS ring2 wide has no LDL/STL.
  The audit records TMA/BF16 WGMMA and zero FP8 WGMMA instructions. Larger shared
  footprints, stack traffic and WS resource costs are consistent with the
  regressions. No NCU counters were collected in that initial trial; the
  [2026-10-26 follow-up](ncu-20261026/README.md) now supplies counter evidence.

## Evidence and reproduction

The selected synthetic raw timing and correctness JSON files are gzip-compressed
without changing their decompressed bytes. Readable correctness JSON copies add
only a final newline. No model
weights, genuine routing dumps, private host inventory, GPU lease records or
binary compiler caches are included.

[MANIFEST.json](MANIFEST.json) records publication file hashes, exact source
hashes and each receipt's recorded source hashes. `measured_sources.tar.gz`
contains the final consolidated source workspace. `source_history.tar.gz`
preserves the ten earlier source revisions needed by the retained receipts:
the first two variants were collected before the final harness consolidation.
The final sources implement all six variants; they must not be described as the
byte-identical source revision of every earlier run.

Run these CPU-only checks from the repository root:

```bash
python3 benchmark/experiments/minimax_m3_tle_pipeline/verify_artifacts.py
python3 benchmark/experiments/minimax_m3_tle_pipeline/reproduce.py \
  --prepare /tmp/minimax-tle-trial
```

For exact historical source inspection, choose a bundled receipt from the
manifest, for example:

```bash
python3 benchmark/experiments/minimax_m3_tle_pipeline/reproduce.py \
  --receipt data/benchmarks/quick-benchmark-s2.json.gz \
  --prepare /tmp/minimax-tle-historical
```

Preparation starts no GPU work. In an independently provisioned, exclusive
single-H100 environment matching the compiler hashes, the final consolidated
workspace can run the following commands. Use a separate `TRITON_CACHE_DIR` and
make the matching FlagTree importable; the archived harness imports candidate
sources by relative workspace paths.

```bash
cd /tmp/minimax-tle-trial
python3 scripts/gate.py --quick --no-ws --compact-warps 8 \
  --out smoke-nows-s2.json
python3 scripts/bench.py --no-ws --compact-warps 8 \
  --q 8192 8153 --seq 16384 --modes contig correlated8 scattered \
  --gate smoke-nows-s2.json --out quick-benchmark-nows-s2.json
```

Other measured settings (gate first, then bench with the same flags) are:

| Variant | Flags | Timed q |
|---|---|---|
| WS4/ring2 | `--stages 2 --compact-warps 4` | 8192, 8153 |
| WS4/ring3 | `--stages 3 --compact-warps 4` | 8192 |
| WS8/ring2 | `--stages 2 --compact-warps 8` | 8192 |
| non-WS/ring2 | `--no-ws --stages 2 --compact-warps 8` | 8192, 8153 |
| non-WS/ring3 | `--no-ws --stages 3 --compact-warps 8` | 8192 |
| Producer conversion | `--producer-convert --stages 2 --compact-warps 4` | 8192 |

## Initial-trial validation limits and follow-up scope

After all six representative variants regressed consistently, production
promotion stopped. The full q512..8192 x seq16k/32k/64k/long-context matrix,
BF16-storage GPU control, D96, multi-KV-head and stride domains, standalone
duplicate/unsorted cases, genuine TopK GPU replay, rotating/cold KV, multilayer
and end-to-end runs are **NOT_RUN** for this experiment. Some metadata/empty
states were exercised inside Graph checks; that does not complete those domains.
The archive's optional real-routing replay mode requires dumps not distributed
here. NCU and compute-sanitizer memcheck/racecheck were **NOT_RUN in the initial
trial** because its container lacked the tools. The 2026-10-26 follow-up adds
NCU and limited BF16-storage q8192 controls; it does not complete the full
domain matrix. Compute-sanitizer remains NOT_RUN.

No fresh GPU benchmark was run for the original 2026-10-23 PR packaging or the
publication of the follow-up. The follow-up GPU measurements were collected
separately on 2026-10-26. Offline verification
rehashes the published artifacts and sources, verifies readable-kernel AST
parity, checks #979's operator is unchanged, and recalculates all timing ratios
from raw samples, including the follow-up exports. See
[PROJECT_CONTRACT.md](PROJECT_CONTRACT.md) for the explicit
research-only integration and autotune exemptions.
