# Recurrent GDN native baselines

The two GDN operators share one change set and retain separate implementation,
test, benchmark and baseline modules.

| Operator | Native vLLM function | Grid |
| --- | --- | --- |
| `fused_recurrent_gated_delta_rule_fwd` | `fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule` | 8 bindings × 3 dtypes |
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
