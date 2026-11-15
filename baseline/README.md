# Recurrent GDN native baselines

The two GDN operators share one change set and retain separate implementation,
test, benchmark and baseline modules.

| Operator | Native vLLM function | Grid |
| --- | --- | --- |
| `fused_recurrent_gated_delta_rule` | `fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule` | 8 bindings × 3 dtypes |
| `fused_recurrent_gated_delta_rule_packed_decode` | `fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode` | 8 bindings × 3 dtypes |

Native paths above are relative to `vllm.model_executor.layers`. Resolvers return
the exact callable or `None` with `NATIVE_UNAVAILABLE_REASON`; no candidate or
mathematical fallback is used. The FlagOS vLLM image needs its bundled `fl`
platform plugin enabled to initialize native CUDA imports.

Each module exports `OP_NAME`, `DTYPES`, `IS_INPLACE`, `native()`, `grid()`,
`build_inputs()`, `key_shape()` and `config()`. Configs retain shapes, dtypes,
strides, scalar parameters, state-pool layout `[pool, HV, V, K]`, aliases and
mutation semantics. Benchmarks reuse baseline input constructors; generic and
top-level entries share the same native workload rather than duplicate it.

`MUTATED_INPUTS` identifies the full state pool, and additionally `out` for packed
decode. The native collector restores these tensors outside every warmup/event
interval, uses five warmups and 25 samples, and records the median and samples.
Graph is disabled. NCU subprocesses rebuild inputs between warmups and before
the sampled invocation.

The ordinary `ragged` workload retains a token-stride-2 state-index table. The
inspected native vLLM indexes it as stride 1 and previously caused an illegal
device access. The three dtype rows remain in the grid and are recorded as
unsupported with a reason. Re-audit native indexing before removing this guard.

`tools/collect_baseline_nvidia.py --check --ops <names>` probes only the first
binding/dtype of each operator; it does not establish complete precision or
performance coverage. Shape keys identify these fixed workloads. Consumers must
also compare the complete config before reusing a native baseline record.

## Gumbel sampling

`gumbel_sample.py` covers eight shapes at temperatures 0 and 0.8, FP32,
with immutable explicit seed/position inputs and no cache or bias. It preserves
the benchmark's expanded logits layout, including zero row stride when N > 1.
The native function is pinned to vLLM commit
`0ef2a95cc861adfae9a92ba104bdfd40ede8cbaa`; source/dependency hashes are declared
in `NATIVE_SOURCE`. An incompatible installed vLLM is reported as unavailable.

To use a separate checkout without replacing installed packages, set
`GUMBEL_NATIVE_SOURCE_ROOT` to that vLLM source root. The loader verifies all
three declared files, executes the original function with its pinned draft
helper and the active Triton runtime, and records this as a source-tree native
baseline. Qualification checks independent CPU winners, native Murmur known
answers, replay, unchanged inputs and unchanged RNG states before collection.
This qualification covers the 16 FP32 cases on NVIDIA only.

```bash
export GUMBEL_NATIVE_SOURCE_ROOT=/absolute/path/to/pinned-vllm
python tools/collect_baseline_nvidia.py --ops gumbel_sample --check --device 0
python tools/collect_baseline_nvidia.py --ops gumbel_sample --device 0 \
  --reference-chip H800 --output /absolute/path/gumbel-nv.json
pytest benchmark/test_gumbel_sample.py --level core --record json \
  --output /absolute/path/gumbel-result.json \
  --base-data=/absolute/path/gumbel-nv.json
```

Select the actual reference chip; H800 above is an example, not a substitute
for H20/H20Z. Both sides use no-Graph `triton.testing.do_bench` with 25 ms warmup,
100 ms repetition and median, with standard latencies in milliseconds. The
benchmark emits one dtype group with 16 unique case keys. It keeps on-device
`speedup` and normalized `compared_speedup` separate; the latter requires exact
case/config/protocol, measured reference GPU and target SKU/resource factors.
Missing NCU classification, mismatched source/config or absent factors produce
null with a reason. Raw cross-device ratios never stand in for normalization.
Native execution/qualification errors stop the process even without `-x`;
native ratios below 0.9 remain ordinary test failures. Private `GUMBEL_BENCH`
and optional `GUMBEL_BENCH_OUTPUT` JSONL are retained for manual inspection.

The adapter's CPU-only checks run with
`python -m unittest tools/test_gumbel_baseline.py`; they include synthetic
comparison data and do not claim hardware performance.
