# MThreads paged FlashAttention

Inference FP16/BF16 attention on MUSA. The public frontend validates supported
arguments, selects a legal kernel family, and tunes the complete pipeline during
eager warmup. There is no native attention or PyTorch compute fallback.

## Configuration and capture

`flash_attn_varlen_mthreads` in the backend `tune_configs.yaml` contains search
templates. Each exact shape has at most 32 candidates, including its validated
heuristic seed and each legal family seed. Generic paths search query/KV tiles,
warps/stages, splits and head packing. Cached paths search gather tiles/warps,
packing and compact scheduling. All routes use the installed Triton backend
without operator-side compiler overrides or generated IR rewriting. Single-KV
decode searches the original MMA path and vector tiles of 1/4/8/16 rows, with
4 or 8 warps where supported. Short-KV decode also searches a public-TLE
32-by-32 tile with 4 warps; larger warp counts are not supported by this backend.
Prefill searches joint gather-tile, gather-warp and GQA-packing configurations.

The cache key includes tensor shape, stride, dtype and storage offset; host
length bounds and API options; device properties; and compiler/source/library
fingerprints, including the native compiler binary and MUSA LLVM environment
switches. It excludes tensor values and addresses. Persistent configurations
use the existing FlagGems `libcache`; no second database implementation is used.
Persistent tables are separated by Triton version and serialized Config schema,
because vendor Config fields can differ even when row keys identify the compiler.
Single- and multi-kernel paths use the same whole-pipeline tuner so preparation
cost is included. The MUSA Graph adapter uses MUSA streams, events and graphs
without modifying `torch.cuda` or importing test plugins.

After the initial search, the seed and the three fastest measured candidates
are checked in rotating order over five rounds. Selection uses the lower
quartile and requires a gain above 1%; two fast observations are needed, so one
low outlier cannot win. This limits inflated timing samples from shared devices
while keeping the outer benchmark's median statistic unchanged.

Warm up every exact shape before graph capture. Capture only reads the in-memory
configuration. A wholly cold shape raises an explicit warmup error. For checking
heuristic routing, `MTHREADS_FA_AUTOTUNE=0` uses the default seed. A graph captured
after this warmup retains that seed. Shape-specific tensors, page tables, Q/K/V
values, lengths, bounds and classifications are recomputed on every invocation.

## Workspaces and synchronization

Paged KV staging uses the smaller of the caller's `max_seqlen_k` and page-table
capacity, rounded to a consumer tile. Fully unused tiles are skipped; the last
tile read by MMA/TME is zero padded, including V, to avoid `0 * NaN` propagation.
All norm entries are initialized on every replay. KV staging is capped at 1 GiB;
Split-KV partial O/LSE is capped at 64 MiB by the candidate policy.

TLE primitives must be both explicitly registered for MThreads and present in
the installed frontend. Missing features select a complete ordinary Triton
implementation. Shared-memory buffers and synchronization are expressed through
TLE allocation, copy and wait primitives. There are no custom compiler backends,
IR text substitutions, shared allocation offset patches or barrier phase patches.

The former M192 and paired warp-specialized implementations required compiler
fixes that this package cannot provide. They are excluded from the search space;
the same inputs use bounded/online TLE or ordinary Triton attention instead.

Vector single-KV and TLE short-KV decode report exceptional rows to the same
Triton MMA repair kernel. This preserves the original NaN/Inf and overflow
behavior while keeping the ordinary vector path free of MMA register pressure.
Only flagged heads are repaired. Both stages are captured and timed together;
there is no native-C attention fallback or host inspection of tensor values.

## SDK helpers

`math_packed4.mu` and `math_affine4.mu` wrap MUSA SDK float4 FMA/exp2 intrinsics.
Their bitcode is packaged beside the Python code. Rebuild with the installed
MUSA SDK and its default C++ include search paths:

```sh
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_31 --cuda-device-only \
  -fgpu-rdc -c -emit-llvm -O2 math_packed4.mu -o math_packed4.bc
/usr/local/musa/bin/mcc -x musa --offload-arch=mp_31 --cuda-device-only \
  -fgpu-rdc -c -emit-llvm -O2 math_affine4.mu -o math_affine4.bc
```

Do not copy a different machine's `CPLUS_INCLUDE_PATH` into this build. A new SDK
or compiler ABI requires revalidation; a matching helper filename alone does not
establish binary compatibility.

The GPU suites check FP32 references, empty/all-masked rows, noncontiguous KV,
updated Graph inputs, candidate resource limits and cache behavior. Full-shape
tests are opt-in through `MTHREADS_FA_SHAPE_SET=full` (1602 BF16 cases). Frozen
native denominators and validation artifacts remain outside the package.
