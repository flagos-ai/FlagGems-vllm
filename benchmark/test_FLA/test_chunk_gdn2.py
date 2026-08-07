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
import math
import os

import pytest
import torch

import flaggems_vllm
from benchmark.base import Benchmark
from flaggems_vllm.ops.FLA import chunk_gdn2
from flaggems_vllm.ops.FLA.gdn2_native.chunk_fwd import chunk_gdn2_fwd

FORCE_NATIVE_ENV = "FLAGGEMS_VLLM_GDN2_FORCE_NATIVE"


def _set_public_gdn2_native_for_call(force_native: bool):
    module = importlib.import_module("flaggems_vllm.ops.FLA.chunk_gdn2")
    old = module.HAS_TLE_GDN2
    if force_native:
        module.HAS_TLE_GDN2 = False
    return module, old


def _native_gdn2_op(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_size=64,
    **kwargs,
):
    del kwargs
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.inference_mode():
        (
            o,
            final_state,
            _g,
            _Aqk,
            _Akk,
            _w_wy,
            _u_wy,
            _qg,
            _kg,
            _v_new,
            _h,
            _initial_state,
        ) = chunk_gdn2_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w_gate=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=True,
            state_v_first=state_v_first,
        )
    return o, final_state


def _public_gdn2_op(*args, **kwargs):
    force_native = os.environ.get(FORCE_NATIVE_ENV, "0") == "1"
    call_kwargs = dict(kwargs)
    if not force_native:
        # Public TLE inference currently requires BT=16. The baseline remains
        # the original native Triton path with BT=64.
        call_kwargs["chunk_size"] = 16
    module, old = _set_public_gdn2_native_for_call(force_native)
    try:
        with torch.inference_mode():
            return chunk_gdn2(*args, **call_kwargs)
    finally:
        module.HAS_TLE_GDN2 = old


class ChunkGDN2Benchmark(Benchmark):
    DEFAULT_SHAPE_FILES = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "core_shapes.yaml"
    )
    DEFAULT_DTYPES = [torch.float16, torch.bfloat16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (2, 512, 8, 64, 64),
        (4, 1024, 8, 64, 64),
        (1, 2048, 8, 64, 64),
        (1, 4096, 16, 64, 64),
        (1, 8192, 96, 128, 128),
        (2, 2048, 16, 256, 512),
        (2, 16384, 16, 128, 128),
        (4, 1024, 8, 256, 512),
        (4, 2048, 16, 128, 128),
        (4, 4096, 64, 128, 128),
        (8, 1024, 8, 64, 64),
        (8, 2048, 32, 256, 256),
    ]
    DEFAULT_SHAPE_DESC = "B, T, H, K, V"

    def init_user_config(self):
        super().init_user_config()
        if any(len(shape) != 5 for shape in self.shapes):
            self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for B, T, H, K, V in self.shapes:
            yield self._build_inputs(B, T, H, K, V, cur_dtype)

    def _build_inputs(self, B, T, H, K, V, dtype):
        device = flaggems_vllm.device
        scale = K**-0.5

        q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.1).to(
            dtype
        )
        b = torch.rand(B, T, H, K, device=device, dtype=dtype)
        w = torch.rand(B, T, H, V, device=device, dtype=dtype)
        initial_state = (
            torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01
        )

        return (
            q,
            k,
            v,
            g,
            b,
            w,
            {
                "scale": scale,
                "initial_state": initial_state,
                "output_final_state": True,
                "use_gate_in_kernel": False,
                "safe_gate": False,
                "state_v_first": False,
                "cu_seqlens": None,
                "cu_seqlens_cpu": None,
                # Native Triton baseline uses BT=64. The TLE wrapper maps this
                # to BT=16 internally because the public TLE path requires it.
                "chunk_size": 64,
            },
        )


@pytest.mark.skipif(
    flaggems_vllm.device != "cuda", reason="chunk_gdn2 benchmark requires CUDA"
)
@pytest.mark.chunk_gdn2
@pytest.mark.gdn2
def test_chunk_gdn2():
    bench = ChunkGDN2Benchmark(
        op_name="chunk_gdn2",
        torch_op=_native_gdn2_op,
    )
    bench.set_gems(_public_gdn2_op)
    bench.run()
