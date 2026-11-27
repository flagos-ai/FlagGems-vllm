# #1346 dual-device concat coverage completion

The original source is unchanged: integration `40941ef9145b2749714d4358c9e14e0394d3c4d7`, tree `4aae57a750c926de27e24d6c73d6677463232bfb`. The concat test module is byte-identical to main12210, Git blob `9143f1cfb8e1925fd2476fcb7ae4d1621ff9d97d`. All1808 canonical-LF raw Git blobs match the same frozen source used for the earlier four-module run.

After the NVIDIA agent explicitly released the completed Marlin G6 timing window, the full original70-case concat module ran once with physicalGPU4 andGPU5, a new namespace/CID and independent fresh caches. No case filtering, source changes, tolerance changes or skip changes were made. The command was:

```sh
/usr/bin/python3 -m pytest /candidate/tests/test_concat_mla_q.py -q -rA \
  -p pytest_context_observer -c /candidate/pytest.ini \
  -o cache_dir=/results/cache/concat/pytest --junitxml=/results/concat.xml
```

**70 passed, zero skipped, zero failures**,10.03seconds. All12 `test_concat_mla_q_uses_input_device_and_restores_current` cases executed and passed; they were resource skips in the earlier single-device scope. The earlier225 original passes,9 inherited namespace-regression passes and12 resource skips remain recorded unchanged. This70-case supplementary run includes58 previously passed cases and the12 newly exercised cases; do not add70 to225 as distinct coverage. Across the unchanged source's four original modules, all237 distinct original cases now have passing execution evidence, plus9 namespace-regression cases. This is still an explicitly dependent#1345/#1346 integration result, not independently green#1346 on unchanged main.

Device admission required both fixed UUIDs, memory<=16MiB/utilization0/no compute PID and both cooperative lock files. Docker granted exactly DeviceIDs4,5, and the actual Torch device_count was2. The two actual UUIDs matched physical4/5 exactly after normalizing only the optional textual `GPU-` prefix. Both `can_device_access_peer(0,1)` and reverse returnedTrue. Host topology/P2P read/write reports are retained. GPU6 was used only for the separately approved Marlin numerical scope; no acceptance timing overlapped this run.

All source-before/after hashes passed. The actual candidate tests/conftest observer reported device context false and pytest exit0. CID `48e09442aa43af2e4861628b3c13e170b0c1d653f423307c210bfa9bde0c119f` naturally exited0 without OOM or forced stop, was removed and exact CID absence was positively checked. Both physical4/5 retired at0MiB/0util/no selected PID. Their leases are released and no hardware jobs remain owned by this worker.

An earlier G1-only attempt is retained in concat-dual-v2. The harness had compared Torch's bare UUID string to nvidia-smi's `GPU-` prefixed representation. It failed before pytest ran, then naturally exited1 and cleaned up with both cards idle. The new harness preserves original raw strings and only normalizes the prefix for exact identity comparison; no test failure was retried. The failed preflight raw SHA256 is `d87d151af9cd5a0924ba9547890c4cd09309f2436207abbd794ef09e08b05773`.

Evidence: summary.json and raw.tar.gz. Successful raw SHA256 `05805da957c046d3b4943eb89576149e7be95118f69c2d2d4c28742b8bb73a8d`. No source or GitHub push was performed for this supplementary scope.
