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

"""qsa_sparse_paged_attention baseline（方案 B）——占位模块（待落地）。

本模块为对齐 gems-vllm 算子总揽（关键算子表）而建，目前 **尚未落地**：
native() 返回 None，采集器会打印
    「跳过算子 qsa_sparse_paged_attention: ops 模块 native() 解析不到 callable」
并整算子跳过，不产出基准数据。

待落地事项（后续按契约补全）：
    - 回 vllm 源码核对 native 调用坐标（import 路径 / 符号 / 签名）；
    - 复刻对应 FlagGems-vllm benchmark 的 input_fn 与 shape（无 benchmark 则按
      源码调用点推断，并在 docstring 注明「shape 为 vllm 源码推断」）；
    - 把 grid()/build_inputs()/key_shape()/config() 换成真实配置。

当前 grid/build_inputs/key_shape 仅为契约占位，使 _smoke_ops.py 能跑通字段校验。
"""

import torch

OP_NAME = "qsa_sparse_paged_attention"
DTYPES = [torch.bfloat16]
IS_INPLACE = False


def native():
    """占位：尚未落地，返回 None 让采集器跳过本算子。"""
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
            "skip_reason": "占位模块，native() 未落地（待回源码核对后补全）",
            "shape_source": "占位（未落地）",
        },
    }
