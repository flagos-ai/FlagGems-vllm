# GDN W8A16 FP8 H20 Validation

Date: 2026-10-23. Device: NVIDIA H20, node 033. PyTorch 2.11.0+cu128, Triton 3.6.0, vLLM 0.20.2. Upstream base: `d3214ce`.

- Accuracy: **68 passed** against canonical FP32 recurrence and boundary/quantization checks.
- Performance: **58 shapes completed**, standard `--mode=cudagraph --iter=20`.
- Arithmetic mean speedup: **2.625x**; geometric mean: **1.646x**.
- 12 shapes are below 0.9x; the per-shape results are reported below without filtering.
- Accuracy JSON `opname` and performance JSON key/`op_name`: **gdn_w8a16**.

The baseline is vLLM's **BF16 fused recurrent GDN**, including prefill; it is not vLLM's chunk-prefill implementation. Large prefill gains therefore include the algorithmic benefit of the repository's chunk path and should not be attributed solely to FP8 storage. Both paths receive identical q/k/v/g/beta, with separate state buffers and unique slots starting at 1 because vLLM reserves slot 0. A closed-form first-token reference validates the baseline before timing. Initial FP8 quantization is excluded; in-call dynamic quantization is included. Diagnostic runs and experimental tuning trials are excluded from these results.

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_gdn_w8a16.py -v -s --tb=short \
  --record=json --output=gdn_w8a16_accuracy.json
PYTHONPATH=src python -m pytest benchmark/test_gdn_w8a16.py -v -s --tb=short \
  --level=comprehensive --mode=cudagraph --warmup=100 --iter=20 \
  --record=json --output=gdn_w8a16_benchmark.json
```

FP32 output reference tolerances: atol=0.002, rtol=0.05. Reconstructed-state tolerances: atol=0.02, rtol=0.25. Tests cover zero and nonzero state, BF16/FP16 activations, strided inputs, variable lengths, empty/padded sequences, speculative state slots and dynamic per-channel range. Accuracy includes decode N=16384. Paired performance stops at N=8192 to stay within the native vLLM grid limit.

## H=2, HV=4: Decode

Mean 0.955x; geometric mean 0.929x. K=V=128.

| N | T | vLLM BF16 ms | FP8 W8A16 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.003117 | 0.002839 | 1.098x |
| 2 | 1 | 0.003116 | 0.003861 | 0.807x |
| 4 | 1 | 0.003174 | 0.005002 | 0.634x |
| 8 | 1 | 0.003296 | 0.004998 | 0.660x |
| 16 | 1 | 0.003738 | 0.005135 | 0.728x |
| 32 | 1 | 0.004676 | 0.006105 | 0.766x |
| 64 | 1 | 0.008121 | 0.009987 | 0.813x |
| 128 | 1 | 0.014223 | 0.016287 | 0.873x |
| 256 | 1 | 0.025880 | 0.028387 | 0.912x |
| 512 | 1 | 0.067269 | 0.053838 | 1.249x |
| 1024 | 1 | 0.128995 | 0.108482 | 1.189x |
| 2048 | 1 | 0.251736 | 0.209111 | 1.204x |
| 4096 | 1 | 0.498822 | 0.411118 | 1.213x |
| 8192 | 1 | 0.992042 | 0.812947 | 1.220x |

## H=2, HV=4: Prefill

Mean 3.280x; geometric mean 2.137x. K=V=128.

| N | T | vLLM BF16 ms | FP8 W8A16 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.004373 | 0.004709 | 0.929x |
| 1 | 8 | 0.011931 | 0.010765 | 1.108x |
| 1 | 32 | 0.042516 | 0.026874 | 1.582x |
| 1 | 256 | 0.328503 | 0.044491 | 7.384x |
| 1 | 512 | 0.654627 | 0.051172 | 12.793x |
| 2 | 192 | 0.247129 | 0.043442 | 5.689x |
| 4 | 4 | 0.006910 | 0.008548 | 0.808x |
| 4 | 16 | 0.022299 | 0.015194 | 1.468x |
| 4 | 256 | 0.329282 | 0.048612 | 6.774x |
| 8 | 128 | 0.182024 | 0.050999 | 3.569x |
| 16 | 4 | 0.008561 | 0.008719 | 0.982x |
| 16 | 128 | 0.188706 | 0.071505 | 2.639x |
| 32 | 64 | 0.131290 | 0.080479 | 1.631x |
| 64 | 32 | 0.139003 | 0.133697 | 1.040x |
| 128 | 24 | 0.199925 | 0.249906 | 0.800x |

## H=2, HV=6: Decode

Mean 1.171x; geometric mean 1.149x. K=V=128.

| N | T | vLLM BF16 ms | FP8 W8A16 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.003941 | 0.003094 | 1.274x |
| 2 | 1 | 0.003986 | 0.003927 | 1.015x |
| 4 | 1 | 0.004108 | 0.005025 | 0.817x |
| 8 | 1 | 0.004346 | 0.005103 | 0.852x |
| 16 | 1 | 0.005091 | 0.005930 | 0.859x |
| 32 | 1 | 0.008883 | 0.007776 | 1.142x |
| 64 | 1 | 0.013374 | 0.012337 | 1.084x |
| 128 | 1 | 0.022353 | 0.021894 | 1.021x |
| 256 | 1 | 0.058387 | 0.040747 | 1.433x |
| 512 | 1 | 0.110700 | 0.082796 | 1.337x |
| 1024 | 1 | 0.214724 | 0.156199 | 1.375x |
| 2048 | 1 | 0.423554 | 0.304207 | 1.392x |
| 4096 | 1 | 0.839494 | 0.599373 | 1.401x |
| 8192 | 1 | 1.668820 | 1.192906 | 1.399x |

## H=2, HV=6: Prefill

Mean 4.887x; geometric mean 3.028x. K=V=128.

| N | T | vLLM BF16 ms | FP8 W8A16 ms | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.006023 | 0.004702 | 1.281x |
| 1 | 8 | 0.018654 | 0.010793 | 1.728x |
| 1 | 32 | 0.069591 | 0.027097 | 2.568x |
| 1 | 256 | 0.544988 | 0.044975 | 12.118x |
| 1 | 512 | 1.087583 | 0.053560 | 20.306x |
| 2 | 192 | 0.410073 | 0.044103 | 9.298x |
| 4 | 4 | 0.010751 | 0.008583 | 1.253x |
| 4 | 16 | 0.037610 | 0.017869 | 2.105x |
| 4 | 256 | 0.550103 | 0.061746 | 8.909x |
| 8 | 128 | 0.287607 | 0.061701 | 4.661x |
| 16 | 4 | 0.012588 | 0.010638 | 1.183x |
| 16 | 128 | 0.318670 | 0.106132 | 3.003x |
| 32 | 64 | 0.331369 | 0.125210 | 2.647x |
| 64 | 32 | 0.253711 | 0.194533 | 1.304x |
| 128 | 24 | 0.332530 | 0.352207 | 0.944x |
