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
import random

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils.device_info import get_device_capability

from . import base

torch_device_fn = flaggems_vllm.runtime.torch_device_fn

SUPPORTED_VENDORS = frozenset({"nvidia", "mthreads", "hygon", "metax", "thead"})

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
VENDOR = flaggems_vllm.vendor_name


UINT8_CONTAINER_VENDORS = frozenset({"hygon", "metax", "thead"})
FP8_STORE_DTYPE = torch.float8_e4m3fn
HEAD_DIM = 512
ROPE_DIM = 64


def is_support_fp8e4nv():
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if VENDOR in SUPPORTED_VENDORS - {"nvidia"}:
        return True
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


# -----------------------------------------------------------------------------
# Optional vLLM native baseline
# -----------------------------------------------------------------------------

HAS_VLLM = False
if VENDOR == "nvidia":
    try:
        import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

        HAS_VLLM = True
    except (ImportError, AttributeError, RuntimeError):
        # No NVIDIA native baseline: fall back to FlagGems-only latency.
        pass


def _skip_if_unrunnable(ref, op_name):
    """Skip a registered NVIDIA baseline whose kernel fails on first launch."""
    checked = False

    def wrapper(*args, **kwargs):
        nonlocal checked
        if checked:
            return ref(*args, **kwargs)
        try:
            out = ref(*args, **kwargs)
            # Surface an asynchronous native launch failure before timing.
            torch.zeros(1, device=flaggems_vllm.device).sum()
            torch_device_fn.synchronize()
        except Exception as exc:
            reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            pytest.skip(f"{op_name} is registered but fails to run: {reason}")
        checked = True
        return out

    return wrapper


VLLM_REF_AVAILABLE = (
    VENDOR == "nvidia"
    and HAS_VLLM
    and hasattr(torch.ops, "_C")
    and hasattr(torch.ops._C, OP_NAME)
)
_VENDOR_REF = (
    _skip_if_unrunnable(getattr(torch.ops._C, OP_NAME), OP_NAME)
    if VLLM_REF_AVAILABLE
    else None
)


def _get_baseline_op():
    """vLLM is the NVIDIA native implementation, not a cross-vendor baseline."""
    if VENDOR == "nvidia":
        return _VENDOR_REF
    return None


# -----------------------------------------------------------------------------
# FlagGems-vllm candidate
# -----------------------------------------------------------------------------

_GEMS_OP_ERROR = None


def _resolve_gems_op():
    """Load the active vendor's port, falling back to the NV/generic module."""
    global _GEMS_OP_ERROR
    try:
        module = importlib.import_module(
            f"flaggems_vllm.runtime.backend._{VENDOR}.fused.{OP_NAME}"
        )
    except ModuleNotFoundError:
        try:
            module = importlib.import_module(f"flaggems_vllm.ops.{OP_NAME}")
        except ImportError as exc:
            _GEMS_OP_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    op = getattr(module, OP_NAME, None)
    if op is None:
        _GEMS_OP_ERROR = f"{module.__name__} exports no {OP_NAME}"
    return op


_GEMS_OP = _resolve_gems_op()


def _run_gems(*args):
    """Adapt only the output container dtype; the allocation stays FP8."""
    if VENDOR not in UINT8_CONTAINER_VENDORS:
        return _GEMS_OP(*args)
    args = list(args)
    args[2] = args[2].view(torch.uint8)  # q_fp8
    args[3] = args[3].view(torch.uint8)  # k_cache
    return _GEMS_OP(*args)


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class TestParam:
    __test__ = False

    num_tokens: int
    num_heads: int
    block_size: int
    max_pos: int
    eps: float
    apply_q_norm: bool = True
    dtype: torch.dtype = torch.bfloat16
    device: str = flaggems_vllm.device


_random_counter = 0


def _make_cos_sin_cache(max_pos, rope_dim, dtype, device):
    if max_pos <= 8192:
        base_value = 10000.0
    elif max_pos <= 32768:
        base_value = 20000.0
    elif max_pos <= 65536:
        base_value = 40000.0
    elif max_pos <= 98304:
        base_value = 60000.0
    else:
        base_value = 100000.0
    inv_freq = 1.0 / (
        base_value
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


class FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark(base.Benchmark):
    BASELINE_METRICS = ("latency_base", "speedup")

    def __init__(self):
        super().__init__(OP_NAME, _get_baseline_op(), [torch.bfloat16, torch.float16])
        if self.torch_op is None:
            self.metrics = [
                metric for metric in self.metrics if metric not in self.BASELINE_METRICS
            ]
            self.to_bench_metrics = self.metrics
        self.set_gems(_run_gems)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        for param in self.get_performance_test_params():
            yield from self.make_input(dataclasses.replace(param, dtype=dtype))

    @staticmethod
    def get_performance_test_params():
        return [
            TestParam(
                num_tokens,
                num_heads,
                block_size=64,
                max_pos=4096,
                eps=1e-6,
            )
            for num_tokens in [
                1,
                4,
                17,
                64,
                1024,
                2048,
                8192,
                32768,
                65536,
                98304,
                131072,
            ]
            for num_heads in [64, 128]
        ]

    @staticmethod
    def init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_input(param: TestParam):
        num_tokens = param.num_tokens
        num_heads = param.num_heads
        block_size = param.block_size
        max_pos = max(param.max_pos, num_tokens)
        device = param.device

        global _random_counter
        FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark.init_seed(
            _random_counter
        )
        _random_counter += 1

        q = torch.randn(
            num_tokens, num_heads, HEAD_DIM, dtype=param.dtype, device=device
        )
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=param.dtype, device=device)
        # FP8 allocations serve both the native op and uint8-view Triton ports.
        q_fp8 = torch.empty(
            num_tokens, num_heads, HEAD_DIM, dtype=FP8_STORE_DTYPE, device=device
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        k_cache = torch.zeros(
            num_blocks,
            block_size,
            HEAD_DIM,
            dtype=FP8_STORE_DTYPE,
            device=device,
        )
        fp8_scale = torch.ones(1, dtype=torch.float32, device=device)
        q_fp8_scale_inv = torch.ones(1, dtype=torch.float32, device=device)

        yield (
            q,
            kv,
            q_fp8,
            k_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            fp8_scale,
            q_fp8_scale_inv,
            param.eps,
            block_size,
            param.apply_q_norm,
        )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
@pytest.mark.skipif(VENDOR not in SUPPORTED_VENDORS, reason="Unsupported vendor")
@pytest.mark.skipif(
    not is_support_fp8e4nv(),
    reason="E4M3 dtype/storage is unavailable on this device",
)
def test_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert():
    bench = FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark()
    bench.run()
