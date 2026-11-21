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

# vLLM's mhc layer module crashes at import time on every non-CUDA-alike
# platform (tilelang=None + module-level @tilelang.jit), which aborts pytest
# collection of tests/test_mhc_ops.py, tests/test_mhc_pre_npu.py and
# benchmark/test_mhc.py on GCU. install_vllm_mhc_import_guard() below converts
# exactly that crash into a chained ImportError (the outcome the tests'
# ``except ImportError`` probes already expect) and is a pure passthrough
# otherwise. Nothing in this package imports the guarded module, so calling
# it after the public imports is sufficient: the guard is active before any
# external test or benchmark probes ``vllm...mhc``.
from flaggems_vllm.runtime.backend._enflame.ops.mhc._vllm_mhc_import_compat import (
    install_vllm_mhc_import_guard,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn,
    mhc_split_sinkhorn_torch_ref,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc.mhc_bwd import (
    mhc_bwd,
    mhc_bwd_ref,
    sinkhorn_forward,
)
from flaggems_vllm.runtime.backend._enflame.ops.mhc.mhc_post import MixLayout, mhc_post
from flaggems_vllm.runtime.backend._enflame.ops.mhc.mhc_pre import mhc_pre

install_vllm_mhc_import_guard()

__all__ = [
    "hc_head_fused_kernel",
    "hc_head_fused_kernel_ref",
    "hc_split_sinkhorn",
    "mhc_bwd",
    "mhc_bwd_ref",
    "mhc_post",
    "mhc_post_ref",
    "mhc_pre",
    "mhc_pre_ref",
    "mhc_split_sinkhorn_torch_ref",
    "MixLayout",
    "sinkhorn_forward",
]
