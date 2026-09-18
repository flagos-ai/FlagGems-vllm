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

"""One CTA per request for page16/GQA4 Decode, with direct final output."""
from .splitkv import flash_varlen_splitkv_kernel


def launch_page16_decode(params, *, batch_size):
    """Reuse the established BM4/N16 MMA loop without partial buffers or merge."""
    args = tuple(getattr(params, key) for key in params.__slots__)
    cfg = {
        "BLOCK_M": 4,
        "BLOCK_N": 16,
        "BLOCK_K": 256,
        "num_warps": 1,
        "num_stages": 1,
        "NUM_SPLITS": 1,
        "Q_TILES": 1,
    }
    return flash_varlen_splitkv_kernel[(1, batch_size, 1)](*args, **cfg)
