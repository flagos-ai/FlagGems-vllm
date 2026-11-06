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
from packaging.version import InvalidVersion, Version

import flaggems_vllm
from flaggems_vllm.utils.device_info import get_device_capability

from . import base

torch_device_fn = flaggems_vllm.runtime.torch_device_fn

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
VENDOR = flaggems_vllm.vendor_name
SUPPORTED_VENDORS = frozenset({"nvidia"})

HEAD_DIM = 512
ROPE_DIM = 64

_FP8E4NV_CAPABLE_VENDORS = frozenset({"ascend", "metax", "mthreads"})


def is_support_fp8e4nv():
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if VENDOR in _FP8E4NV_CAPABLE_VENDORS:
        return True
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


# -----------------------------------------------------------------------------
# vLLM native baseline
# -----------------------------------------------------------------------------

_NATIVE_BASELINE_REQUIRED = VENDOR == "nvidia"

# The operator first appears in vLLM 0.32.0. Only the skip message is derived
# from the version: whether it can run is decided by hasattr below, so a nightly
# or a backport that carries the op is still benchmarked.
MIN_VLLM_VERSION = Version("0.32.0")


def _vllm_version():
    """vLLM's parsed version, or None when it cannot be determined."""
    try:
        from vllm import __version__ as raw  # noqa: F401 - optional dependency
    except Exception:
        return None
    try:
        return Version(str(raw).split("+", 1)[0])
    except InvalidVersion:
        return None


HAS_VLLM = False
_VLLM_VERSION = None
if _NATIVE_BASELINE_REQUIRED:
    try:
        import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

        HAS_VLLM = True
    except (ImportError, AttributeError, RuntimeError):
        pass
    _VLLM_VERSION = _vllm_version()


def _skip_if_unrunnable(ref, op_name):
    """Wrap the baseline so a registered-but-unlaunchable kernel skips.

    A vendor build can register the op and still fail every launch. Skip with the
    launch error as the reason instead of failing the benchmark.

    A failed launch is reported asynchronously, so the first call forces it out
    with a throwaway kernel launch; synchronize() alone is silent there.
    ``pytest.skip`` raises ``Skipped``, a BaseException, which passes the
    harness's ``except Exception``.
    """
    checked = False

    def wrapper(*args, **kwargs):
        nonlocal checked
        if checked:
            return ref(*args, **kwargs)
        try:
            out = ref(*args, **kwargs)
            torch.zeros(1, device=flaggems_vllm.device).sum()
            torch_device_fn.synchronize()
        except Exception as e:
            reason = str(e).splitlines()[0] if str(e) else type(e).__name__
            pytest.skip(
                f"{op_name} is registered but its kernel fails to run: {reason}"
            )
        checked = True
        return out

    return wrapper


def _missing_baseline_reason():
    if not HAS_VLLM:
        return f"vLLM >= {MIN_VLLM_VERSION} is required but vLLM is not importable"
    if _VLLM_VERSION is not None and _VLLM_VERSION < MIN_VLLM_VERSION:
        return (
            f"vLLM >= {MIN_VLLM_VERSION} is required for the native"
            f" implementation (found {_VLLM_VERSION})"
        )
    found = f"vLLM {_VLLM_VERSION}" if _VLLM_VERSION else "vLLM of an unknown version"
    return (
        f"{found} is installed but registers no {OP_NAME};"
        " the build may be CPU-only or stripped"
    )


VLLM_REF_AVAILABLE = (
    HAS_VLLM and hasattr(torch.ops, "_C") and hasattr(torch.ops._C, OP_NAME)
)

_BASELINE_SKIP_REASON = _missing_baseline_reason()
_VENDOR_REF = (
    _skip_if_unrunnable(getattr(torch.ops._C, OP_NAME), OP_NAME)
    if VLLM_REF_AVAILABLE
    else None
)


# -----------------------------------------------------------------------------
# FlagGems-vllm candidate
# -----------------------------------------------------------------------------


_GEMS_OP_ERROR = None


def _resolve_gems_op():
    """Return the FlagGems implementation of this operator.

    The port is its own module under ``flaggems_vllm.ops``; importing it by path
    also resolves before it is listed in ``ops/__init__.py``, so the benchmark
    can be run against the port and the merged library alike.
    """
    global _GEMS_OP_ERROR
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


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
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


def _make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    """cos||sin cache in the DeepseekV4ScalingRotaryEmbedding layout.

    cos_sin_cache[pos, :rope_dim/2] = cos(theta), [rope_dim/2:] = sin(theta).
    The RoPE base grows with max_pos, matching the long-context variant.
    """
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
    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
    return cache.to(dtype)


class FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            OP_NAME,
            _VENDOR_REF,
            [torch.bfloat16, torch.float16],
        )
        self.set_gems(_GEMS_OP)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        # TestParam carries a default dtype only; the runner walks self.dtypes
        # and wants this dtype, so stamp it onto every case. Without this the
        # fp16 pass would silently measure bf16 tensors.
        for param in self.get_performance_test_params():
            yield from self.make_input(dataclasses.replace(param, dtype=dtype))

    @staticmethod
    def get_performance_test_params():
        # num_heads are the local heads after TP sharding. The largest case
        # (131072 tokens, 128 heads) holds ~17 GB of bf16 q plus ~9 GB of fp8
        # q_fp8, so trim the tail of this list on a smaller card.
        cases = [
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
        return cases

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
        eps = param.eps
        dtype = param.dtype
        device = param.device

        global _random_counter
        FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark.init_seed(
            _random_counter
        )
        _random_counter = _random_counter + 1

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
        q_fp8 = torch.empty(
            num_tokens, num_heads, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        )
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = _make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)

        # slot_mapping covers every token: the Q side always runs, and the KV
        # store runs for every non-negative slot.
        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        k_cache = torch.zeros(
            num_blocks,
            block_size,
            HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        )

        # Production scales are per-tensor; both are 1.0 on the current
        # FlashInfer path. They stay tensors so the kernel reads them from device
        # memory, exactly as the call site does.
        fp8_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        q_fp8_scale_inv = torch.tensor([1.0], dtype=torch.float32, device=device)

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
            eps,
            block_size,
            param.apply_q_norm,
        )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
@pytest.mark.skipif(
    VENDOR not in SUPPORTED_VENDORS,
    reason="This benchmark only covers the CUDA (nvidia) backend for now",
)
@pytest.mark.skipif(
    _NATIVE_BASELINE_REQUIRED and not VLLM_REF_AVAILABLE,
    reason=_BASELINE_SKIP_REASON,
)
@pytest.mark.skipif(
    _GEMS_OP is None,
    reason=(
        "No FlagGems implementation found for this operator"
        f" ({_GEMS_OP_ERROR or 'unknown import failure'})"
    ),
)
@pytest.mark.skipif(
    not is_support_fp8e4nv(),
    reason="Do not support fp8e4nv when capability < 89",
)
def test_fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert():
    # Guarded by the skipif above; kept as a hard stop for a direct call.
    if _NATIVE_BASELINE_REQUIRED:
        assert (
            VLLM_REF_AVAILABLE
        ), "vLLM native fused_deepseek_v4 full-cache FP8 op is not available"

    bench = FusedDeepseekV4QnormRopeKVRopeFullCacheFp8InsertBenchmark()
    bench.run()
