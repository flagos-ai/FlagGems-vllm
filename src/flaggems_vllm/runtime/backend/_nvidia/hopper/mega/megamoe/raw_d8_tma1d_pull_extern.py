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

"""Triton extern ABI for the nvcc-compiled live D8 TMA1D helper."""

import triton.language as tl
import triton.language.core as core


@core.extern
def TleD8Tma1d(
    token_stage,
    token_state,
    global_token,
    num_bytes,
    active,
    op,
    _semantic=None,
):
    i32 = core.dtype("int32")
    i64 = core.dtype("int64")
    return core.extern_call(
        "",
        "",
        [
            token_stage,
            token_state,
            global_token,
            tl.cast(num_bytes, tl.int32, _semantic=_semantic),
            tl.cast(active, tl.int32, _semantic=_semantic),
            tl.cast(op, tl.int32, _semantic=_semantic),
        ],
        {
            (
                i64,
                i64,
                i64,
                i32,
                i32,
                i32,
            ): ("TleD8Tma1d", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
