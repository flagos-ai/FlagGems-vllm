---
title: MetaX einsum baseline
weight: 40
---

# MetaX einsum baseline

The MetaX INT8 einsum benchmark compares against the vendor-provided BF16
`deep_gemm.einsum` entry point used by vLLM-MetaX. A FlagTree development image
may contain PyTorch and Triton without that vendor package on its Python path.
The tested Torch 2.8 / Triton 3.6 combination also needs the old `triton_key`
import path used by Inductor.

Run the benchmark through `tools/run_metax_einsum.sh` inside the existing MetaX
environment. The launcher adds the repository and optional vendor package to
the child process's Python path, checks and reports the BF16 source, and enables
the compatibility alias for Torch 2.8 / Triton 3.6. An existing `triton_key` is
preserved. It does not install dependencies or change the vendor implementation.

## Requirements

- A working MetaX PyTorch/FlagTree environment and an available GPU.
- The vendor `deep_gemm` package from a compatible vLLM-MetaX environment.
  `METAX_DEEP_GEMM_ROOT` is its parent directory, containing `deep_gemm/`.
  Leave it unset when the vendor package is already installed.
- The interpreter belonging to that environment. Configure
  `METAX_EINSUM_PYTHON` if it is not the default `python3`.

The vendor BF16 entry point itself uses `torch.compile` and `torch.einsum`.
Keep that implementation and its compilation enabled when reproducing this
baseline. The upstream NVIDIA DeepGEMM CUDA kernels are a different backend.

## Run

Select currently available physical GPU indices before running:

```bash
export METAX_EINSUM_PYTHON=/opt/conda/bin/python
export METAX_DEEP_GEMM_ROOT=/path/to/vendor-packages
export METAX_EINSUM_GPU=0
bash tools/run_metax_einsum.sh -m pytest -q tests/test_int8_einsum.py --quick
```

`METAX_EINSUM_GPU` sets `CUDA_VISIBLE_DEVICES`, `MACA_VISIBLE_DEVICES`, and
`METAX_VISIBLE_DEVICES` consistently. When it is unset, existing device masks
are inherited. The launcher executes Python arguments from the repository root.

To pin a particular reference, set `METAX_EINSUM_EXPECTED_SHA256` to the hash
of the vendor `deep_gemm/einsum.py`. A mismatch stops before the supplied command.
Every invocation reports the source path, hash, and actual Torch/Triton versions.

```bash
bash tools/run_metax_einsum.sh -m pytest -q -s \
  'benchmark/test_int8_einsum.py::test_perf_int8_einsum[int8]' \
  --shape_file /path/to/accepted-shapes.yaml \
  --level comprehensive --mode kernel --parallel 0 --warmup 30 --iter 100
```

Use the same shape set, accuracy thresholds, and timing method on both sides.
Report `BF16 latency / INT8 latency`, including quantization error in accuracy
checks. Keep weighted and unweighted means distinct when using trace counts.

## Troubleshooting

- `No module named deep_gemm`: select the vendor environment or set the parent
  package directory through `METAX_DEEP_GEMM_ROOT`.
- `cannot import name triton_key`: use the launcher for the affected Torch 2.8 /
  Triton 3.6 stack, and verify the interpreter and actual Triton source path.
- A source hash mismatch: inspect the reported baseline before changing the pin.

The launcher has CPU-only startup regression checks:

```bash
python3 tools/test_metax_einsum_env.py
```

Those checks use isolated package fixtures. Validate the real vendor operator
and accuracy on C550 separately before publishing benchmark measurements.
