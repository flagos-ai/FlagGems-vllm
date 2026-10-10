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

import dataclasses
import importlib
import importlib.util
import math
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import List

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_sparse_fwd_w8a8_fp8 import HAS_TLE
from tests.accuracy_utils import gems_assert_close, gems_assert_equal

from . import base

HAS_FP8_MLA = (
    HAS_TLE and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9
)


@lru_cache(maxsize=1)
def flashmla_reference() -> ModuleType:
    """Load the explicitly selected, unmodified vLLM CUDA reference for tests only."""
    reference_path = os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH")
    if reference_path:
        # Initialize Torch shared libraries before loading the CUDA extension.
        importlib.import_module("torch")
        source = Path(reference_path).resolve()
        package_root = source.parents[2]
        parent = sys.modules.get("vllm")
        if parent is None:
            # The approved CUDA source only imports its extension; do not initialize serving.
            parent = ModuleType("vllm")
            parent.__path__ = [str(package_root)]
            sys.modules["vllm"] = parent
            created_parent = True
        else:
            created_parent = False

        for module_name in ("_flashmla_C", "_flashmla_extension_C"):
            qualified = f"vllm.{module_name}"
            if qualified in sys.modules:
                continue
            candidates = sorted(package_root.glob(f"{module_name}*.so"))
            if not candidates:
                raise RuntimeError(
                    f"missing CUDA reference extension: {package_root}/{module_name}"
                )
            spec = importlib.util.spec_from_file_location(qualified, candidates[0])
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        name = "flaggems_vllm_test_flashmla_reference"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if created_parent:
            del sys.modules["vllm"]
        return module
    module = importlib.import_module("vllm.v1.attention.ops.flashmla")
    supported, reason = module.is_flashmla_sparse_supported()
    if not supported:
        raise RuntimeError(
            f"vLLM SparseMLA reference unavailable: {reason}. "
            "Set FLAGGEMS_FLASHMLA_REFERENCE_PATH to the approved CUDA reference "
            "interface; required benchmarks must not be skipped."
        )
    return module


def _get_vllm_flashmla_sparse_reference():
    """Return the official vLLM CUDA BF16 sparse MLA reference."""
    if os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH"):
        return flashmla_reference().flash_mla_sparse_fwd, None
    try:
        from vllm.v1.attention.ops import flashmla
    except Exception as error:
        return None, f"vLLM FlashMLA could not be imported: {error}"

    support_check = getattr(flashmla, "is_flashmla_sparse_supported", None)
    if not callable(support_check):
        return None, "vLLM does not expose is_flashmla_sparse_supported()"

    try:
        support = support_check()
    except Exception as error:
        return None, f"vLLM FlashMLA capability check failed: {error}"

    if isinstance(support, tuple):
        supported, reason = support
    else:
        supported, reason = bool(support), None
    if not supported:
        return None, reason or "vLLM sparse FlashMLA is not supported"

    reference = getattr(flashmla, "flash_mla_sparse_fwd", None)
    if not callable(reference):
        return None, "vLLM sparse FlashMLA reference is not callable"
    return reference, None


vllm_flash_mla_sparse_fwd, VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON = (
    _get_vllm_flashmla_sparse_reference()
)
HAS_VLLM_FLASHMLA_SPARSE = vllm_flash_mla_sparse_fwd is not None


def quantize_sparse_fp8_fixture(
    q, kv, indices, sm_scale, value_dim, attn_sink=None, topk_length=None
):
    # Preserve the upstream BF16 fixture and its valid physical token IDs.
    query_content = q[..., :512]
    cache_content = kv[:, 0, :512]
    query_scale = query_content.float().abs().amax(-1, keepdim=True) / 448.0
    cache_scale = cache_content.float().abs().amax(-1, keepdim=True) / 448.0
    query_scale = torch.where(query_scale > 0, query_scale, 1.0)
    cache_scale = torch.where(cache_scale > 0, cache_scale, 1.0)
    q_nope = (query_content.float() / query_scale).to(torch.float8_e4m3fn)[:, None]
    if q.shape[-1] == 576:
        q_rope = (q[..., 512:].float() / query_scale).to(torch.bfloat16)[:, None]
    else:
        q_rope = torch.empty(
            (*q_nope.shape[:-1], 0), device=q.device, dtype=torch.bfloat16
        )
    pages = math.ceil(kv.shape[0] / 64)
    cache_nope = torch.zeros(
        (pages * 64, 512), device=kv.device, dtype=torch.float8_e4m3fn
    )
    rope_dim = q_rope.shape[-1]
    cache_rope = torch.zeros(
        (pages * 64, rope_dim), device=kv.device, dtype=torch.bfloat16
    )
    scales = torch.ones((pages * 64, 1), device=kv.device, dtype=torch.float32)
    cache_nope[: kv.shape[0]].copy_(
        (cache_content.float() / cache_scale).to(torch.float8_e4m3fn)
    )
    scales[: kv.shape[0]].copy_(cache_scale)
    if kv.shape[-1] == 576:
        cache_rope[: kv.shape[0]].copy_(
            (kv[:, 0, 512:].float() / cache_scale).to(torch.bfloat16)
        )
    valid_indices = torch.where((indices >= 0) & (indices < kv.shape[0]), indices, -1)
    return (
        q_nope,
        q_rope,
        cache_nope.view(pages, 64, 512),
        cache_rope.view(pages, 64, rope_dim),
        query_scale[:, None],
        scales.view(pages, 64, 1),
        valid_indices,
        sm_scale,
        attn_sink,
        topk_length,
    )


def run_sparse_fp8_fixture(*inputs):
    return flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs)


def assert_sparse_fixture_close(output, lse, reference, reference_lse):
    relative = torch.linalg.vector_norm(
        output.float() - reference.float()
    ) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    gems_assert_close(relative, torch.zeros_like(relative), torch.float32, atol=0.05)
    gems_assert_equal(torch.isposinf(lse), torch.isposinf(reference_lse))
    gems_assert_equal(torch.isneginf(lse), torch.isneginf(reference_lse))
    finite = torch.isfinite(reference_lse)
    scaled_error = (lse[finite].float() - reference_lse[finite].float()) / (
        0.025 + 0.002 * reference_lse[finite].float().abs()
    )
    gems_assert_close(
        scaled_error, torch.zeros_like(scaled_error), torch.float32, atol=1.0
    )


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flaggems_vllm.device


# used by make_input_flashmla
_flashmla_sparse_counter = 0


class FlashmlaSparseBenchmark(base.Benchmark):
    def __init__(self, operator="flash_mla_sparse_fwd"):
        self.is_fp8_query = operator == "flash_mla_sparse_fwd_w8a8_fp8"
        super().__init__(operator, vllm_flash_mla_sparse_fwd, [torch.bfloat16])
        if self.is_fp8_query:
            self.set_gems(run_sparse_fp8_fixture)
        else:
            self.set_gems(flaggems_vllm.flash_mla_sparse_fwd)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for param in FlashmlaSparseBenchmark.get_performance_test_params_flashmla():
            for inputs in FlashmlaSparseBenchmark.make_input_flashmla(param):
                if self.is_fp8_query:
                    reference, _, reference_lse = vllm_flash_mla_sparse_fwd(*inputs)
                    packed = quantize_sparse_fp8_fixture(*inputs)
                    output, lse = run_sparse_fp8_fixture(*packed)
                    assert_sparse_fixture_close(
                        output[:, 0], lse[:, :, 0], reference, reference_lse
                    )
                    # Baseline remains official BF16 CUDA, with the original upstream inputs.
                    yield (inputs, packed)
                else:
                    yield inputs

    def unpack_to_args_kwargs(self, input_tuple):
        if self.is_fp8_query:
            return [input_tuple], {}
        return super().unpack_to_args_kwargs(input_tuple)

    def get_latency(self, op, *args, **kwargs):
        if self.is_fp8_query:
            original, packed = args[0]
            selected = packed if op is self.gems_op else original
            return super().get_latency(op, *selected)
        return super().get_latency(op, *args, **kwargs)

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def get_performance_test_params_flashmla():
        cases = (
            [
                TestParam(4096, s_kv, 2048, h_q=128, d_qk=576, have_attn_sink=True)
                for s_kv in [8192, 32768, 65536, 98304, 131072]
            ]
            + [
                TestParam(4096, s_kv, 512, h_q=64, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
            + [
                TestParam(4096, s_kv, 1024, h_q=128, d_qk=512, have_attn_sink=True)
                for s_kv in [8192, 32768, 49152, 65536]
            ]
        )
        return cases

    @staticmethod
    def _randperm_batch(
        batch_size: int,
        perm_range: torch.Tensor,
        perm_size: int,
        paddings: List[int],
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size].
        `0 <= res[i, :] < perm_range[i]` holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains
        values in `[0, perm_range[i])` as many as possible, and the rest are filled with `padding`.
        """
        if perm_range.numel() != batch_size:
            raise ValueError("perm_range must contain one value per batch row")
        if perm_size < 0:
            raise ValueError("perm_size must be non-negative")
        if not paddings:
            raise ValueError("paddings must not be empty")

        ranges = perm_range.to(device="cpu", dtype=torch.int64).reshape(-1)
        if torch.any(ranges < 0):
            raise ValueError("perm_range values must be non-negative")

        # An affine permutation (offset + step * i) % range is unique whenever
        # step and range are coprime.  It needs only O(perm_size) temporary
        # storage instead of the previous O(batch_size * max(perm_range))
        # random score matrix (about 2 GiB for the largest benchmark case).
        positions = torch.arange(perm_size, dtype=torch.int64)
        padding_values = torch.tensor(paddings, dtype=torch.int32)
        res = torch.empty((batch_size, perm_size), dtype=torch.int32)
        for row, range_value in enumerate(ranges.tolist()):
            valid_size = min(range_value, perm_size)
            if valid_size:
                offset = random.randrange(range_value)
                step = 1
                if range_value > 1:
                    step = random.randrange(1, range_value)
                    while math.gcd(step, range_value) != 1:
                        step += 1
                        if step == range_value:
                            step = 1
                values = (offset + step * positions[:valid_size]) % range_value
                res[row, :valid_size] = values.to(torch.int32)

            if valid_size < perm_size:
                if len(paddings) == 1:
                    res[row, valid_size:].fill_(paddings[0])
                else:
                    filler_indices = torch.randint(
                        0, len(paddings), (perm_size - valid_size,)
                    )
                    res[row, valid_size:] = padding_values[filler_indices]
        return res

    @staticmethod
    def make_input_flashmla(param: TestParam):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseBenchmark._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]

        indices = FlashmlaSparseBenchmark._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)

        yield (q, kv, indices, 0.5, param.d_v, attn_sink, topk_length)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param("flash_mla_sparse_fwd", marks=pytest.mark.flash_mla_sparse_fwd),
        pytest.param(
            "flash_mla_sparse_fwd_w8a8_fp8",
            marks=pytest.mark.flash_mla_sparse_fwd_w8a8_fp8,
        ),
    ],
)
def test_flash_mla_sparse_fwd(operator):
    if operator == "flash_mla_sparse_fwd_w8a8_fp8":
        if not HAS_FP8_MLA:
            pytest.skip("FP8 MLA requires NVIDIA Hopper and Triton TLE")
        if not HAS_VLLM_FLASHMLA_SPARSE:
            raise RuntimeError(VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON)
    elif not HAS_VLLM_FLASHMLA_SPARSE:
        pytest.skip(VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON)
    FlashmlaSparseBenchmark(operator).run()
