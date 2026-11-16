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

"""hygon SiLU/mul FP8 quantization with default/expand CUDA Graph tuning.

Default schedules are searched for each input key; explicit FlagTune enables
the larger expand space. Tuned launches are cached without retaining tensors;
allocation alignment and HCU buffer-pointer specialization remain in the key.
"""

from contextlib import nullcontext
from threading import Lock

import torch
import triton
import triton.language as tl
from triton.backends.hcu.compiler import HIPBackend

from flaggems_vllm import runtime
from flaggems_vllm.ops.silu_mul_per_token_group_quant_fp8_colmajor import (
    _launch_platform_fp8,
    _needs_general_quantization,
    _prune_silu_tiles,
    _silu_mul_quant_tile,
    _validate_and_allocate,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libtuner
from flaggems_vllm.utils.libentry import LibTuner


class _SiluConfigTuner(LibTuner.get("default")):
    """Reuse tuned launches without retaining tensors or fixing tile sizes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._launches = {}
        self._key_positions = tuple(self.arg_names.index(name) for name in self.keys)
        self._config_lock = Lock()

    def run(self, *args, **kwargs):
        if self.apply_flagtune():
            self._launches.clear()
        key = tuple(args[i] for i in self._key_positions) + tuple(
            tensor.dtype for tensor in args[:3]
        )
        if getattr(self, "shared_config_pre_hook", None) is not None:
            return super().run(*args, **kwargs)
        # HCU's buffer operations specialize on the entire backing allocation,
        # not just tensor.numel(). Preserve that distinction for sliced inputs.
        key += (
            args[0].device.index,
            args[5],
            type(args[6]),
            tuple(
                HIPBackend.get_tensor_specialization(t, align=True) for t in args[:3]
            ),
        )
        cached = self._launches.get(key)
        if cached is None:
            with self._config_lock:
                cached = self._launches.get(key)
                if cached is None:
                    result = super().run(*args, **kwargs)
                    config = self.best_config
                    if config.pre_hook is None:
                        grid = kwargs["grid"]
                        if callable(grid):
                            grid = grid(
                                {**dict(zip(self.arg_names, args)), **config.kwargs}
                            )
                        grid = (*grid, 1, 1)[:3]
                        self._launches[key] = (result, config, result[grid])
                    return result
        kernel, config, launch = cached
        launch(*args, config.kwargs["BR"], config.kwargs["BG"])
        return kernel


@libtuner(
    configs=runtime.get_tuned_config(
        "silu_mul_per_token_group_quant_fp8_tile_hygon_default"
    ),
    flagtune_op_name="silu_mul_per_token_group_quant_fp8_colmajor",
    flagtune_expand_op_name="silu_mul_per_token_group_quant_fp8_tile_hygon",
    key=["M", "N", "DTYPE", "UE", "HAS_CLAMP", "DEVICE_NAME", "INDEX64"],
    prune_configs_by={"early_config_prune": _prune_silu_tiles},
    use_cuda_graph=True,
    rep=30,
    policy=_SiluConfigTuner,
)
@triton.jit(do_not_specialize=["CLAMP"])
def _silu_mul_quant_autotuned(
    X,
    Q,
    S,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP,
    HAS_CLAMP: tl.constexpr,
    UE: tl.constexpr,
    DTYPE: tl.constexpr,
    DEVICE_NAME: tl.constexpr,
    INDEX64: tl.constexpr,
    BR: tl.constexpr,
    BG: tl.constexpr,
):
    _silu_mul_quant_tile(
        X, Q, S, M, N, EPS, CLAMP, HAS_CLAMP, UE, BR, BG, "hygon", INDEX64
    )


def silu_mul_per_token_group_quant_fp8_colmajor(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
    eps: float = 1e-10,
    clamp_limit: float | None = None,
):
    """Return quantized activations and column-major FP32 group-128 scales."""
    output, scales, use_ue8m0 = _validate_and_allocate(
        input, output, use_ue8m0, eps, clamp_limit
    )
    m, n = input.shape
    if m == 0 or n == 0:
        return output, scales

    guard = (
        nullcontext()
        if torch_device_fn.current_device() == input.device.index
        else torch_device_fn.device(input.device)
    )
    with guard:
        if _needs_general_quantization(output, use_ue8m0, eps, clamp_limit):
            _launch_platform_fp8(input, output, scales, use_ue8m0, eps, clamp_limit)
            return output, scales
        _silu_mul_quant_autotuned[
            lambda meta: (m // meta["BR"], n // (256 * meta["BG"]))
        ](
            input,
            output,
            scales,
            m,
            n,
            eps,
            clamp_limit if clamp_limit is not None else 0.0,
            clamp_limit is not None,
            use_ue8m0,
            str(input.dtype),
            torch_device_fn.get_device_name(input.device),
            input.numel() >= 2**31,
            enable_fp_fusion=False,
        )
    return output, scales
