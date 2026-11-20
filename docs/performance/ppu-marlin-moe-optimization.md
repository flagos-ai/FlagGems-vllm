# Quantized Marlin MoE on PPU-ZW810E

## Part 1 — Implementation

The T-Head backend reuses packed W4 and W8 weight tiles and decodes them as BF16 for matrix multiplication. The sparse paths specialize small routing groups and bound their temporary storage. The dense path prepares decoded weight banks for grouped GEMM, reuses projection storage across the two MLP projections, and caps temporary storage at 2 GiB.

For the wide FP8 decode workload with M=16384, E=256, K=4096, N=256 and top-k=6, GPU histograms and prefix sums define the two expert windows. Route scattering and weight decoding prepare the next window on an auxiliary stream while the first gate GEMM runs. The first gate GEMM waits only for its routes and decoded weights. The second gate GEMM waits for the corresponding auxiliary work. Scale and route changes remain visible on CUDA Graph replay.

TLE provides shared-memory allocation, local pointers and asynchronous tile transfers in the raw and grouped matrix paths. PPU inline assembly is used in the packed decoders and matrix fragments. PRMT, LOP3, integer bit operations and packed BF16 arithmetic decode the original quantized values directly into BF16 and apply the supplied scales.

## Part 2 — Performance setup

Measurements use PPU-ZW810E with driver `1.3.2-d7f5a2` and the `sdk2.2.1-pytorch2.13.0-ubuntu24.04-cuda13.0-py312-20261116` image from `pkg.flytiger-eco.com/docker_release/llm` at digest `sha256:79a65c0cda61d530668d6d64aa3d845102f500caa24adf65c6ffc4ffc4f1213d`. The image supplies PyTorch 2.13.0, HGGC SDK 2.2.1-184ee6 and Runtime API v3. The vendor process uses its Triton 3.7.0 and the candidate process uses FlagTree `0.7.0+ppu.git1a96a36a`. Both load SAIL 0.33.0rc1, the bundled vLLM 0.33.1.dev0 at `4ba787ec`, and DeepGEMM `0f9756f`. [SAIL release](https://github.com/vllm-project/vllm-sail/releases/tag/v0.33.0rc1) · [DeepGEMM revision](https://github.com/t-head/DeepGEMM-for-sail/commit/0f9756f27305e97c0f4794f27ac2f2f19155b059)

The native timing entry is `vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel.apply` with the SAIL `PPUDeepGemmExperts` class. W8 measurements use the unquantized BF16 path with the same decoded weights. The native SAIL W8 MoE uses quantized activations. INT4 and MXFP4 comparisons use the corresponding native W4A16 paths. Native INT4 repeats each group-128 scale over four group-32 blocks with identical weight values. Decoding and repacking the native baseline weights precede timing.

```python
output = kernel.apply(
    hidden_states, gate_up_weight, down_weight, routing_weights, expert_ids,
    MoEActivation.SILU, 256, None, False,
)
```

The 53 `(M, E, K, N, topk)` shapes and their call counts come from the DeepSeek-V4-Flash `p32768d1024` workload. The comparison uses deterministic synthetic activation and routing values with seed `7 + M`, W8 weights with seed 7, and W4 weights with seed 0. Each baseline and candidate pair alternates on the same physical PPU and uses the median of three 100 ms CUDA Graph measurements. Speedup is native latency divided by candidate latency. The weighted mean uses the captured call counts. [Shape provenance](https://github.com/July-h5kf3/FlagGems-vllm/blob/b07646fcb7c46e6ced7f9efa178bb1c977a748f8/docs/performance/ppu-marlin-moe-vendor031/shape_provenance.json)

| Precision | Native comparison | Precision passes | Call count | Weighted speedup | Per-shape range |
| --- | --- | ---: | ---: | ---: | ---: |
| FP8 W8A16 | SAIL BF16 MoE | 53/53 | 10,105 | 1.30361× | 0.95152×–2.19562× |
| INT8 W8A16 | SAIL BF16 MoE | 53/53 | 10,105 | 1.32044× | 0.61759×–1.51364× |
| INT4 W4A16 | SAIL INT4 W4A16 | 53/53 | 10,105 | 0.97541× | 0.68597×–1.34650× |
| MXFP4 W4A16 | SAIL MXFP4 W4A16 | 53/53 | 10,105 | 0.97382× | 0.61179×–1.24994× |

The minimum FP8 result is M=16384 at 7.521870 ms against 7.157176 ms native. The independent result after syncing upstream is 0.95206×. M=2048 is the lowest-speedup case for INT8, INT4 and MXFP4. [All shape timings and errors](ppu-marlin-moe-optimization.csv)

## Part 3 — Validation commands

The native output is checked against a FP32 accumulation reference, and the candidate output is checked against the native output. The gates are relative RMS error below 1% and peak relative error below 2%.

The test host retains the benchmark harness, shape list, exact candidate snapshot and device ownership logs under `/tmp/pr752-vendor-baseline-20261116/pr23-vllm033-20261119`. The launchers below select the SDK 2.2.1 test container and its isolated vendor and FlagTree environments. The vendor environment resolves ACExt's libraries and LUT directory from the image. DeepGEMM uses a cache dedicated to SDK 2.2.1. `--gpu` is the host device index and `--device` is the container ordinal. The performance launcher validates device identity and rejects external device contexts during a run.

```bash
cd /tmp/pr752-vendor-baseline-20261116/pr23-vllm033-20261119
repo=repo-pr23-formal
operator=src/flaggems_vllm/runtime/backend/_thead/fused/fused_marlin_moe.py
source_sha=$(sha256sum "$repo/$operator" | cut -d' ' -f1)
export PR23_TEST_REPO="/work/pr23-vllm033-20261119/$repo"

bash run_trial_formal.sh 1 verify-fused --program pytest -- \
  -q --quick tests/test_fused_marlin_moe.py
bash run_trial_formal.sh 3 verify-align --program pytest -- \
  -q --quick tests/test_moe_align_block_size.py
bash run_trial_formal.sh 3 verify-graph --program pytest -- \
  -q tests/test_fused_marlin_moe.py -k test_fused_marlin_moe_w8a16_fp8_graph_scale_updates_ppu

python3 run_pair_formal.py --gpu 0 --device 0 --job verify-native-0 \
  --precision fp8,int8,int4,mxfp4 --deepgemm-release --native-w4 \
  --shard-i 0 --shard-n 4 --candidate-repo "$repo" --candidate-sha256 "$source_sha"
# Run shards 1, 2 and 3 on the other idle cards. The measured host/container
# pairs were 11/2, 13/4 and 14/5.
```

The Marlin MoE quick suite passed 551 tests and skipped 178. The alignment suite passed all 128 tests. All 16 FP8 CUDA Graph cases passed again after merging upstream. For M=16384, the replay checks concentrate all 98,304 routes on a single expert in each window and update the down scale. Both input-side router weighting modes match the previous implementation bit for bit. Peak temporary allocation was 1,930,109,952 bytes in eager execution and 1,930,110,976 bytes during CUDA Graph capture.

Performance records are `formal-sdk221-native-{0,1,2,3}.json` and `formal-sdk221-merged-fp8-m16384.json` on the test host. Precision logs are in `trial/formal-sdk221-test-*.log` and `trial/formal-sdk221-merged-graph.log`. The operator SHA-256 is `7742c2699fdefdb30cba43b639e7811a117d40aaae35bc347d9bc37e692a8fe2`, and the alignment module SHA-256 is `54ee615d3301ce9afd5a11c79121a5c7be4f1bebb2736fbd4942d5b563a601ab`.
