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

import pytest
import torch

import flaggems_vllm
from baseline import indexer_k_quant_and_cache

from . import base


def _vendor_native():
    if flaggems_vllm.vendor_name == "ascend":
        return None

    try:
        import vllm._custom_ops  # noqa: F401 - registers the cache operators

        op = torch.ops._C_cache_ops.indexer_k_quant_and_cache
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def native(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
        return op(k, kv_cache, slot_mapping, quant_block_size, scale_fmt)

    return native


class IndexerKQuantAndCacheBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            tuple(indexer_k_quant_and_cache.key_shape(binding))
            for binding in indexer_k_quant_and_cache.grid()
        ]
        self.shape_desc = ", ".join(indexer_k_quant_and_cache.SHAPE_FIELDS)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            binding = dict(zip(indexer_k_quant_and_cache.SHAPE_FIELDS, shape))
            args, _ = indexer_k_quant_and_cache.build_inputs(
                binding, dtype, self.device
            )
            # Benchmark.unpack_to_args_kwargs accepts string arguments by keyword.
            k, cache, slots, quant_block_size, scale_fmt = args
            yield k, cache, slots, quant_block_size, {"scale_fmt": scale_fmt}


@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "enflame",
    reason="aten and cpu dosenot support",
)
@pytest.mark.indexer_k_quant_and_cache
def test_indexer_k_quant_and_cache_benchmark():
    bench = IndexerKQuantAndCacheBenchmark(
        op_name=indexer_k_quant_and_cache.OP_NAME,
        torch_op=_vendor_native(),
        gems_op=flaggems_vllm.indexer_k_quant_and_cache,
        dtypes=indexer_k_quant_and_cache.DTYPES,
        is_inplace=indexer_k_quant_and_cache.IS_INPLACE,
    )
    bench.run()
