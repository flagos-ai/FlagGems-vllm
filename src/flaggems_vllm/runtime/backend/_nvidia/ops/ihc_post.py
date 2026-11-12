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

"""Cache the tuned NVIDIA launch after full public argument normalization.

Only compiler handles and scalar metadata are retained, never input/output
buffers. Alignment is part of the key, matching libentry's specialization.
"""

import triton

from flaggems_vllm.ops.ihc.ihc_post import _ihc_post_kernel

_LAUNCHES = {}


def _launch_ihc_post(x, residual, post, output, n, hc, d, layout_tag):
    key = (
        x.device.index,
        x.dtype,
        n,
        hc,
        d,
        layout_tag,
        x.data_ptr() % 16 == 0,
        residual.data_ptr() % 16 == 0,
        post.data_ptr() % 16 == 0,
        output.data_ptr() % 16 == 0,
    )
    cached = _LAUNCHES.get(key)
    if cached is not None:
        launch, constants = cached
        launch(x, residual, post, output, *constants)
        return
    hc_pad = triton.next_power_of_2(hc)
    compiled, meta = _ihc_post_kernel[
        lambda cfg: (n * triton.cdiv(d, cfg["BLOCK_D"]),)
    ](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=d,
        HC_MULT=hc,
        HC_POW2=hc_pad,
        NUM_TOKENS=n,
        DTYPE_TAG=str(x.dtype),
        LAYOUT_TAG=layout_tag,
        enable_fp_fusion=False,
    )
    constants = (
        d,
        hc,
        hc_pad,
        meta["BLOCK_D"],
        n,
        str(x.dtype),
        layout_tag,
        meta["TOKEN_MAJOR"],
    )
    grid = (n * triton.cdiv(d, meta["BLOCK_D"]), 1, 1)
    _LAUNCHES[key] = (compiled[grid], constants)
