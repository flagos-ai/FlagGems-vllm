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

"""Small-row GLM5Next mHC Pre + RMSNorm, preserving the BF16 premix boundary."""
import logging

from flaggems_vllm import runtime

logger = logging.getLogger(__name__)


def mhc_pre_norm_small(
    residual,
    fn,
    hc_scale,
    hc_base,
    norm_weight,
    *,
    rms_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    norm_eps=1e-5,
    hc_post_mult_value=2.0,
    sinkhorn_repeat=20,
):
    """Return FP32 post/comb and BF16 normalized layer input for N<=4.

    Residual: BF16 [N,4,4096], fn: FP32 [24,16384], hc_scale/base: FP32
    [3]/[24], norm_weight: BF16 [4096]. Same-device contiguous Ascend tensors
    only. The projection remains FP32; weighted residual rounds to BF16
    before RMSNorm. Inputs are read-only. See docs/ascend_mhc_pre_norm_small.md.
    """
    if runtime.device.vendor_name != "ascend":
        raise NotImplementedError("mhc_pre_norm_small requires Ascend")
    from flaggems_vllm.runtime.backend._ascend.ops.mhc_pre_norm_small import (
        mhc_pre_norm_small as impl,
    )

    logger.debug("GEMS MHC_PRE_NORM_SMALL")
    return impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        norm_eps=norm_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )
