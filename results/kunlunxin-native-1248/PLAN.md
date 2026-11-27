# Pure-native P800 preflight

Proposed only; no native GPU invocation or timing yet. Current candidate7991a6c7 has clean11/11 original provider UTs; this does not validate native performance.

Same immutable image93caedbf/local562f5a, p800_4 physical4/PCI132, source files read from never-started runc metadata container:

- installed `vllm/v1/worker/gpu/structured_outputs.py`: SHA256 `9e212ac51c603e9b95bbc8dceec4510d96b764e49eafa7c10a581ac1e6535297`;
- installed `vllm/v1/worker/gpu/sample/penalties.py`: `a5e99c71e6ac50ef025b8a7b0b903a1d2679738d8319af9c98bb200389388b29`.

`preflight.py` is complete and ready for parent review. It imports actual vLLM native callables, checks those source hashes and original module identity, sets plugins off before imports, and asserts no FlagGems-vllm/plugin import before/after. The candidate is not installed or imported into the native process. SDK Torch/XPU integration remains the qualified vendor stack.

Three **sequential independent containers/processes**, each new own namespace/cache/results, same device/locks/image/stack, original target compiler0.7.0. Each mode is invoked once:

```sh
python -B /task/preflight.py grammar
python -B /task/preflight.py penalties
python -B /task/preflight.py bincount
```

Grammar uses the prior fixed first native benchmark preflight1x167/seed873 and original block8192/exact CPU mask oracle. Penalties uses1x1031/seed873/original block8192 wrapper and prior rtol=atol1e-5 oracle. These are compile/oracle prechecks, not the complete3+3 performance shape matrices. Bincount executes the original four PR1248 UT fixtures, including unchanged2048 repeated-ID contract, with exact rtol0/atol0. A compiler error or first bad oracle stops that mode, with no retry. Other modes may run only in a fresh process after exact owned cleanup/device idle succeeds. Unknown cleanup/idle stops everything.

Runtime is runc, no network, private IPC, physical4 only plus original control devices, capdropALL, own results/cache, no source/runtime package replacement. Use known `/etc/bash_env.sh` SDK path, clear old plugin restriction variables, `VLLM_PLUGINS=''`, `USE_FLAGGEMS=0` only for this explicitly native reference. No operator whitelist, no Torch reference substituted for the native callable, no timing. CPU oracle calculations here validate native outputs and are never timed as a baseline.

Freeze300s per-mode administrative limit and original kernel arguments before launch. Named tmux, fresh cooperative4/6 admission, actualCID+tasklabel cleanup/absence/postidle and raw stdout/JSON preserved. Native failure leaves G6 blocked and does not invalidate the provider's independently passed correctness tests. Native success only permits the subsequent original paired-performance matrix; it is not itself a>=0.9 performance pass.
