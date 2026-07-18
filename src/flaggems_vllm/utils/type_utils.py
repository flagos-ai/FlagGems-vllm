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

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes


def type_promotion(*args, type_promotion: ELEMENTWISE_TYPE_PROMOTION_KIND):
    computation_dtype, result_dtype = elementwise_dtypes(
        *args,
        type_promotion_kind=type_promotion,
    )
    return computation_dtype, result_dtype


_accumulator_dtype_map = {
    torch.bfloat16: torch.float32,
    torch.float16: torch.float32,
    torch.complex32: torch.complex64,
}

_INTEGRAL_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def get_accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    return _accumulator_dtype_map.get(dtype, dtype)


def is_integral_dtype(dtype: torch.dtype) -> bool:
    return dtype in _INTEGRAL_DTYPES


def is_bool_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.bool


def is_python_bool(value) -> bool:
    return isinstance(value, bool)


def is_python_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_python_float(value) -> bool:
    return isinstance(value, float)
