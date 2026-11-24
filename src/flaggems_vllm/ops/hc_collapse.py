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

"""Inference-only BF16 weighted HC reduction; Ascend implementation."""
import torch


def validate_inputs(x, pre_mix):
    if x.ndim != 3 or pre_mix.ndim != 2:
        raise ValueError("expected X[T,HC,H] and pre_mix[T,HC]")
    if tuple(pre_mix.shape) != tuple(x.shape[:2]):
        raise ValueError("pre_mix shape must equal X.shape[:2]")
    if x.dtype != torch.bfloat16 or pre_mix.dtype != torch.float32:
        raise NotImplementedError("supported dtypes: X BF16, pre_mix FP32")
    if x.device != pre_mix.device:
        raise ValueError("X and pre_mix must share a device")
    if x.layout != torch.strided or pre_mix.layout != torch.strided:
        raise NotImplementedError("only strided tensors are supported")
    if any(s < 0 for s in (*x.stride(), *pre_mix.stride())):
        raise NotImplementedError("negative strides are unsupported")
    if x.requires_grad or pre_mix.requires_grad:
        raise NotImplementedError("hc_collapse is inference-only")


def hc_collapse(x, pre_mix):
    validate_inputs(x, pre_mix)
    if x.device.type != "npu":
        raise NotImplementedError("this contribution implements Ascend only")
    from flaggems_vllm.runtime.backend._ascend.ops.hc_collapse import (
        hc_collapse as ascend_hc_collapse,
    )

    return ascend_hc_collapse(x, pre_mix)
