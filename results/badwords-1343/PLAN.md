# Current-main #1343 native-0.32 revalidation

Candidate: `2d21db6765f9ea022f086ff5f6030bd5f8cf9e94`, tree `3dd88c7242e55e012c03ff73f8718e759bff4160`, exact main `12210ab628e0b8f5beed55791958e3bd57accb2b` plus original PR head `e655b413a15e41b5135b63842237c575283199fd`. No production kernel changes. The existing frozen canonical LF archive verifies every raw Git blob.

Use physical GPU4, cached image `sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d`, same reviewed read-only FlagTree0.7/Triton3.6 and FlagGems5.4/FlagCX0.14.1 mounts as public KDA. Torch2.13.0+cu130 and vLLM0.32.0 are asserted. Separate candidate namespace, CID and phase caches; no shared environment mutation, no test selection or retry. Timing only after the NVIDIA agent explicitly releases the G6 window.

The native reference is the original import in `benchmark/test_bad_words.py`: `vllm.v1.worker.gpu.sample.bad_words.apply_bad_words`. Before timing, a new process will assert that the benchmark's actual `apply_bad_words_reference` is the same object and non-null; record its installed module path and source SHA256, and require the path under the installed vLLM0.32 tree. No Torch/Python substitute, fake namespace, reference implementation, or benchmark monkeypatch. The original `_reference_needs_draft_shift` probe then runs inside the unmodified benchmark. It rejects any behavior outside the two recognized native contracts. Native parity and independently derived expected masks are checked before each original timing workload.

Accuracy (original full module, six cases):

```sh
/usr/bin/python3 -m pytest /candidate/tests/test_bad_words.py -q -rA \
  -p pytest_context_observer -c /candidate/pytest.ini \
  -o cache_dir=/results/cache/accuracy/pytest \
  --junitxml=/results/accuracy.xml
```

Benchmark (original full module, one pytest case, 15 workloads x 3 match patterns = 45 native reference measurements):

```sh
/usr/bin/python3 -m pytest /candidate/benchmark/test_bad_words.py -q -s \
  -c /candidate/pytest.ini -o cache_dir=/results/cache/performance/pytest \
  --level core --warmup 25 --iter 100 \
  --record json --output /results/performance.json \
  --junitxml=/results/performance.xml
```

The accuracy process records the actual tests/conftest `_DEVICE_CONTEXT_DEAD` flag and requires false. The benchmark loads its own original `benchmark/conftest.py`, which has no such flag; do not manufacture a false context receipt or claim that the accuracy hook ran there. Preserve original benchmark failures, raw logs and all JSON row/native latency values; require 45 complete rows, positive finite native/candidate latencies, no error rows and pytest exit0. The benchmark's equality checks and original synchronized timing remain unchanged. Source hashes are verified before/after, and exact CID natural exit/removal plus GPU4 idle/no selected PID are mandatory.

## Legacy profile necessity and limit

Current main still explicitly supports the manually selectable `nvidia-cuda128` setup profile: vLLM0.20.2, Torch2.10.0+cu128, FlagTree0.6.1. The corresponding `.github/backends.json` bare-runner lane is disabled. The enabled/default H20 profile is `nvidia-cuda133`, vLLM0.32.0. The PR description's claim that current NVIDIA CI uses0.20.2 must be updated. No current compatible cached image has yet been verified for a new0.20 run, and this0.32 scope cannot validate the legacy branch. Keep the old-head0.20 results labeled historical; do not construct an unnecessary second stack during this scope.

The patch remains independently useful on0.32: main's45-case benchmark only compares FlagGems to native, whereas this PR also derives exact visible-history masks and rejects unexpected native behavior. Main's six separate small accuracy tests do not duplicate that45-workload independent reference contract.
