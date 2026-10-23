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

"""instance_norm baseline（方案 B）——占位模块（待落地 / 无 vLLM 原生接口）。

本模块为对齐 gems-vllm 算子总揽（ops_interface 表）而建，**不落地基准**：
vLLM 无 instance_norm 接口；gems 对标的是 torch.instance_norm（PyTorch 原生）。
因此 native() 返回 None，采集器会打印
    「跳过算子 instance_norm: ops 模块 native() 解析不到 callable」
并整算子跳过，不产出基准数据（仿照 ops/gather_conv_state.py 等跳过算子）。

当前 grid/build_inputs/key_shape 仅为契约占位，使 tools/_smoke_ops.py 能跑通字段校验。
"""

import torch

OP_NAME = "instance_norm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False


def native():
    """占位：无 vLLM 原生接口，返回 None 让采集器跳过本算子。"""
    return None


def grid():
    # 占位采集点（native() 为 None，采集器不会真正调用 build_inputs）。
    return [{"placeholder": 1}]


def build_inputs(binding, dtype, device):
    # 占位实参：仅供契约冒烟（native() 为 None，不会真正喂给 kernel）。
    x = torch.zeros(1, dtype=dtype, device=device)
    return (x,), {}


def key_shape(binding):
    return [binding["placeholder"]]


def config(binding, dtype):
    return {
        "dims": {
            "placeholder": binding["placeholder"],
            "resolvable": False,
            "skip_reason": "vLLM 无 instance_norm 接口；gems 对标的是 torch.instance_norm（PyTorch 原生）。",
            "shape_source": "占位（无 vLLM 接口，跳过）",
        },
    }
