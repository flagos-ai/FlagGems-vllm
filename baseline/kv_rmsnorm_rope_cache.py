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

"""kv_rmsnorm_rope_cache baseline（方案 B）——昇腾专属，H800 无 NV 原生基准，native() 返回 None。

算子说明：KV RMSNorm + RoPE + cache 融合。benchmark 仅昇腾 NPU 跑。

为何 native() 返回 None（如实留痕，非编造跳过）：
    对应 benchmark FlagGems-vllm/benchmark/test_kv_rmsnorm_rope_cache_perf.py 带
    `pytest.mark.skipif(vendor_name != "ascend" / not _IS_ASCEND)`，**仅在昇腾
    NPU 上运行**；其 torch_op 为纯 torch 参考实现，benchmark 内 **没有任何 vllm /
    NVIDIA 原生 kernel 调用**。即该算子在 NVIDIA 侧无对应原生基准可采。
    xlsx《算子总揽》中本算子的“有数据”来自昇腾等国产芯片验收列，不代表 H800 可采。
    按 _FORK_CONTRACT.md 第 4/8 条：解析不到真实 NV callable 时 native() 必须返回
    None（禁止编造不存在的符号），采集器会打印
        「跳过算子 kv_rmsnorm_rope_cache: ops 模块 native() 解析不到 callable」并整算子跳过。

grid/build_inputs/key_shape 仅为契约占位（native() 为 None，采集器不会真正调用）。
"""

import torch

OP_NAME = "kv_rmsnorm_rope_cache"
DTYPES = [torch.bfloat16]
IS_INPLACE = False


def native():
    """昇腾专属，NVIDIA 侧无原生基准：如实返回 None（见模块 docstring）。"""
    return None


def grid():
    return [{"placeholder": 1}]


def build_inputs(binding, dtype, device):
    x = torch.zeros(1, dtype=dtype, device=device)
    return (x,), {}


def key_shape(binding):
    return [binding["placeholder"]]


def config(binding, dtype):
    return {
        "dims": {
            "placeholder": binding["placeholder"],
            "resolvable": False,
            "skip_reason": "昇腾专属算子，H800 无 NV 原生 kernel；benchmark 仅昇腾跑",
            "shape_source": "占位（NV 侧无基准）",
            "benchmark": "test_kv_rmsnorm_rope_cache_perf.py",
        },
    }
