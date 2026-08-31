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
import random

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.utils.device_info import get_device_capability

from . import base

torch_device_fn = flaggems_vllm.runtime.torch_device_fn

_FP8E4NV_CAPABLE_VENDORS = frozenset({"ascend", "metax", "mthreads"})


def is_support_fp8e4nv():
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    if flaggems_vllm.vendor_name in _FP8E4NV_CAPABLE_VENDORS:
        return True
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401 - loads torch.ops._C

    HAS_VLLM = True
except (ImportError, AttributeError, RuntimeError):
    # RuntimeError because a misconfigured vLLM should mean "no baseline", not a
    # collection error: with two platform plugins registered it raises
    # "Only one platform plugin can be activated, but got: ['fl', 'musa']" at
    # import, which aborted collection of this whole file on an MTT box.
    pass


def _skip_if_unrunnable(ref, op_name):
    """Wrap the reference so a registered-but-unlaunchable kernel skips.

    A vendor can register the op and still not be able to run it: MetaX's build
    returns `mcErrorInvalidValue` from every launch on C550. Failing the
    benchmark there blames FlagGems for someone else's defect, while the old
    `hasattr` gate hid the situation entirely by skipping as "not installed".
    Skip, but with the launch error as the reason.

    The first call forces the error to surface. A failed launch is reported
    asynchronously, so left alone it lands on whatever call comes next -- for
    MetaX that is `do_bench`'s 256 MB L2-flush allocation, which makes the
    failure both unattributable and too late to convert. Surfacing it takes a
    *new kernel launch*: on that backend `synchronize()` on its own is silent,
    and so is a device-to-host copy, while any launch (or an allocation large
    enough to reach the driver) raises. Hence the throwaway reduction below.
    Only the first call pays for it.

    `pytest.skip` raises `Skipped`, which derives from `BaseException` and so
    passes through the harness's `except (RuntimeError, Exception)` intact.

    This is the one deliberate deviation from how the sibling `torch.ops._C`
    benchmarks are written (`top_k_per_row_decode`, `persistent_topk`,
    `cutlass_scaled_mm`): they gate on the import alone, which is enough because
    none of them has met a vendor build that registers the op but cannot launch
    it.

    KEEP THIS WRAPPER. MetaX has fixed the defect in source -- mcoplib 0.4.9
    drops the `cudaLaunchKernelEx` path that 0.4.6 calls with an uninitialised
    `cudaLaunchConfig_t` -- but that does not retire the wrapper, for two
    reasons:

      * No wheel carrying the fix is published anywhere reachable. MetaX ships
        no wheels on GitHub (every release has zero assets) and the C550 image
        installs mcoplib from a local file, not an index. Whoever runs this
        still has 0.4.6.
      * 0.4.9 is a different operator. Upstream vLLM changed this op's schema at
        v0.22.0 -- `q` became read-only, a `q_head_padded` argument appeared and
        the result is returned rather than written in place -- and 0.4.9 follows
        it. This file targets the v0.21.0 contract, matching the vLLM version
        the repo pins, so a 0.4.9 baseline would not be comparable even if a
        wheel existed.

    The kernel itself is fine: rebuilt from MetaX's own 0.4.6 source with their
    own 0.4.9 launch fix, it runs on C550 and reaches 96.3% of the card's copy
    ceiling, matching what it scored when forced to run under an LD_PRELOAD
    shim. Only the published binary is unusable.
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


VLLM_REF_AVAILABLE = HAS_VLLM and hasattr(torch.ops._C, OP_NAME)
_VENDOR_REF = (
    _skip_if_unrunnable(getattr(torch.ops._C, OP_NAME), OP_NAME)
    if VLLM_REF_AVAILABLE
    else None
)
HEAD_DIM = 512
ROPE_DIM = 64
HEAD_BYTES = 584


@dataclasses.dataclass
class TestParam:
    # Instruct pytest to ignore this class
    __test__ = False

    num_tokens: int
    num_heads: int
    num_tokens_insert: int
    block_size: int
    max_pos: int
    eps: float
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = flaggems_vllm.device


_random_counter = 0


class FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            _VENDOR_REF,
            [torch.bfloat16],
        )
        self.set_gems(flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert)

    def set_shapes(self, shape_file_path=None):
        self.shapes = []

    def get_input_iter(self, dtype):
        _ = dtype
        for (
            param
        ) in (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.get_performance_test_params()
        ):
            yield from FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input(
                param
            )

    @staticmethod
    def get_performance_test_params():
        cases = [
            TestParam(
                num_tokens,
                num_heads,
                num_tokens_insert=num_tokens,
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
    def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
        if max_pos <= 8192:
            base = 10000.0
        elif max_pos <= 32768:
            base = 20000.0
        elif max_pos <= 65536:
            base = 40000.0
        elif max_pos <= 98304:
            base = 60000.0
        else:
            base = 100000.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
                / rope_dim
            )
        )
        t = torch.arange(max_pos, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
        return cache.to(dtype)

    @staticmethod
    def make_input(param: TestParam):
        num_tokens = param.num_tokens
        num_heads = param.num_heads
        num_tokens_insert = param.num_tokens_insert
        block_size = param.block_size
        max_pos = max(param.max_pos, num_tokens)
        eps = param.eps
        dtype = param.dtype
        device = param.device

        global _random_counter
        FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.init_seed(_random_counter)
        _random_counter = _random_counter + 1

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device)
        kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        cos_sin_cache = (
            FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_cos_sin_cache(
                max_pos, ROPE_DIM, torch.float32, device
            )
        )

        num_blocks = (num_tokens + block_size - 1) // block_size + 1
        slot_mapping = torch.arange(num_tokens_insert, dtype=torch.int64, device=device)
        k_cache = torch.zeros(
            num_blocks,
            block_size * HEAD_BYTES,
            dtype=torch.uint8,
            device=device,
        )
        yield (
            q,
            kv,
            k_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            eps,
            block_size,
        )


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not VLLM_REF_AVAILABLE,
    reason="The referenced vLLM implementation is not installed",
)
@pytest.mark.skipif(
    not is_support_fp8e4nv(),
    reason="Do not support fp8e4nv when capability < 89",
)
def test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert():
    bench = FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark()
    bench.run()
