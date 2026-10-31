# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import HAS_TLE
from tests.accuracy_utils import gems_assert_equal, to_reference
from tests.flash_mla_with_kvcache_w8a8_utils import (
    CapturedCall,
    DenseW8A8Inputs,
    PreparedCall,
    assert_dense_fp8_replay,
    quantize_dense_fp8,
)

OUTPUT_ATOL = 2e-3


def supported() -> bool:
    return (
        HAS_TLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


def quantize(
    x: torch.Tensor, *, benchmark: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    return quantize_dense_fp8(x, dequantize=not benchmark)


@lru_cache(maxsize=1)
def dense_cuda_reference():
    source_path = os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH")
    if not source_path:
        try:
            return importlib.import_module(
                "vllm.third_party.flashmla.flash_mla_interface"
            )
        except ImportError as error:
            raise RuntimeError(
                "Official vLLM CUDA FlashMLA is required; set "
                "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
            ) from error
    source = Path(source_path).resolve()
    package_root = source.parents[2]
    created_parent = "vllm" not in sys.modules
    if created_parent:
        parent = ModuleType("vllm")
        parent.__path__ = [str(package_root)]
        sys.modules["vllm"] = parent
    for module_name in ("_flashmla_C", "_flashmla_extension_C"):
        qualified = f"vllm.{module_name}"
        if qualified in sys.modules:
            continue
        candidates = sorted(package_root.glob(f"{module_name}*.so"))
        if not candidates:
            raise RuntimeError(f"Missing CUDA reference extension: {qualified}")
        spec = importlib.util.spec_from_file_location(qualified, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    name = "flaggems_vllm_test_dense_flashmla_reference"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if created_parent:
        del sys.modules["vllm"]
    return module


def _dense_cuda_reference_or_skip():
    try:
        return dense_cuda_reference()
    except (ImportError, RuntimeError):
        pytest.skip(
            "vLLM CUDA FlashMLA reference is unavailable; set "
            "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
        )


def run_cuda(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    reference = _dense_cuda_reference_or_skip()
    metadata, _ = reference.get_mla_metadata()
    return reference.flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, **kwargs
    )


def functional_reference(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    return run_cuda(
        inputs.native_query,
        inputs.native_cache,
        inputs.block_table,
        inputs.lengths,
        inputs.value_dim,
        causal=True,
    )


def prepare_baseline(inputs: DenseW8A8Inputs) -> DenseW8A8Inputs:
    from flash_mla_fp8 import get_mla_metadata

    metadata, splits = get_mla_metadata(
        inputs.lengths, inputs.q_nope.shape[1] * inputs.q_nope.shape[2], 1
    )
    return inputs._replace(native_metadata=metadata, native_splits=splits)


def run_baseline(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token

    return flash_mla_ckv_fp8_per_token(*inputs[:11], causal=True)


def benchmark_reference(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    return run_baseline(inputs)


def prepare_gems(inputs: DenseW8A8Inputs) -> PreparedCall:
    handle, (output, lse) = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        *inputs[:9],
        causal=True,
        initial_cache_seqlens=inputs.host_lengths,
        max_cache_seqlens=inputs.host_lengths,
    )
    return PreparedCall(lambda: handle(out=output, lse=lse), output, lse)


def capture(run) -> CapturedCall:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = run()
    torch.cuda.current_stream().wait_stream(stream)
    return CapturedCall(graph.replay, output, lse)


def validate_functional(
    inputs: DenseW8A8Inputs,
    actual: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    metadata, splits = flaggems_vllm.get_mla_fp8_metadata(
        inputs.lengths, inputs.q_nope.shape[1] * inputs.q_nope.shape[2]
    )
    planned = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        *inputs[:9], metadata, splits, causal=True
    )
    gems_assert_equal(planned[0], to_reference(actual[0]))
    gems_assert_equal(planned[1], to_reference(actual[1]))
    prepared = prepare_gems(inputs)
    assert_dense_fp8_replay(
        prepared.run, reference, output_atol=OUTPUT_ATOL, capture=capture
    )
