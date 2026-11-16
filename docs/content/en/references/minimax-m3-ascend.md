---
title: MiniMax M3 Ascend Operators
---

These explicit `*_ascend` APIs implement the MiniMax M3 BF16 D128 path.
They do not enable themselves through an environment switch. Callers select
the provider before cache allocation/capture and retain that provider for all
stateful execution; failures must propagate without retrying another backend.

The runtime owns cache/workspace lifetime, slot/sequence metadata and graph
replay updates. The operator package owns scoring, top-k selection, normalization,
partial rotary, cache writes and the independent native sparse attention payload.

Build only the M3 native payload in an isolated Ascend environment:

```sh
# Source the matching CANN environment before building.
bash tools/build_minimax_m3_ascend.sh ascend910_93 /tmp/m3-private-build
PYTHONPATH="$PWD/src:$PYTHONPATH" python -m pytest tests/test_minimax_m3_ascend.py --quick
PYTHONPATH="$PWD/src:$PYTHONPATH" python -m pytest benchmark/test_minimax_m3_ascend.py --level core --iter 10 --warmup 5
```

The package must be built against the active torch/torch_npu/CANN ABI. The bridge
registers `_flaggems_m3_ascend::npu_sparse_attention_score`; CANN registers the
distinct `FlagGemsM3SparseAttentionScore` operator. Both are independent of the
framework plugin. `load_minimax_m3_ascend_native()` performs a strict idempotent
preflight, restores its environment changes if loading fails, and never uses the
plugin's `_C_ascend` operator as a fallback. Source packaging includes only the
needed sparse attention numerical tree and common CANN build infrastructure;
the original Huawei/CANN/BSD/Apache headers remain intact.

The fused QKV API accepts contiguous BF16 packed rows and BF16 [128] weights,
positive even rotary prefixes up to 128, and matching contiguous position/cache
metadata. Dedicated Q outputs suppress writes to the corresponding original
QKV lanes. V lanes are unchanged. Main cache layout must be explicit for the
standalone writer; the upstream fused signature defaults to block-major and
accepts `plane_major=True` for runtime-owned plane-major storage.

Negative slots skip all K/V/index writes. Positive destinations must be unique
and in bounds; positions/block-table data must also be valid. These are caller
preconditions so graph capture never reads tensor contents back to the host.
FP8 caches, autograd and unlisted device/dtype/layout contracts are unsupported.
Partial vision rotary additionally supports FP16/FP32 and the D80/R78 prefix,
preserving the tail and the distinct NeoX/GPT-J rounding contracts.

M3 score and top-k APIs retain the causal block visibility and init/local forcing
of the imported vendor chain. Selection uses the FlagGems top-k operator; ordering
among exactly tied scores is unspecified, as in the previous Torch top-k path.
Indices are int32, invalid selections are -1, and caller outputs are validated
before numerical execution. The index backend has no vLLM/model imports.

All new entrypoints are Ascend-only: CUDA/NVIDIA autotuning is intentionally not
enabled. Index kernels preserve their existing request-count split-K search and
fixed Ascend launches. Other kernels preserve row ownership and use fixed small
tiles; no performance or numerical equivalence is claimed without target-device
results. Benchmark compares the full explicit Torch reference with the candidate,
including identical clone/reset policy; it is not an end-to-end model speedup.

The public upstream base is `8af2cc05dc97789ab55312ad6f0670a78c723c26`.
These additional APIs require the accompanying local source changes until an
operator-library commit containing them is delivered. Installing the unmodified
upstream SHA does not provide these APIs; do not pin a nonexistent remote commit.
