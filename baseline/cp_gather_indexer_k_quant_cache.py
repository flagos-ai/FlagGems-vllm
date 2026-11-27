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

OP_NAME = "cp_gather_indexer_k_quant_cache"
# FP16 is the benchmark's JSON lookup key. build_inputs sets the tensor dtypes.
DTYPES = [torch.float16]
IS_INPLACE = True
MUTATED_INPUTS = (1, 2)
NATIVE_UNAVAILABLE_REASON = None
FP8_DTYPE = torch.float8_e4m3fn
INPUT_SEED = 1
SHAPE_FIELDS = (
    "batch_size",
    "seq_len",
    "block_size",
    "head_dim",
    "quant_block_size",
)


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        import vllm._custom_ops as ops

        op = getattr(ops, OP_NAME)
        getattr(torch.ops._C_cache_ops, OP_NAME)
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    if not callable(op):
        NATIVE_UNAVAILABLE_REASON = "native symbol is not callable"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op


def grid():
    return [
        dict(
            batch_size=batch_size,
            seq_len=seq_len,
            block_size=block_size,
            head_dim=head_dim,
            quant_block_size=quant_block_size,
        )
        for (
            batch_size,
            seq_len,
            block_size,
            head_dim,
            quant_block_size,
        ) in (
            (4, 256, 16, 128, 128),
            (8, 512, 16, 128, 128),
            (16, 1024, 16, 512, 128),
            (32, 1024, 16, 512, 128),
            (1, 8192, 64, 128, 128),
            (8, 4096, 64, 128, 128),
            (64, 1024, 64, 128, 128),
        )
    ]


def key_shape(binding):
    return [binding[name] for name in SHAPE_FIELDS]


def _dimensions(binding):
    dims = dict(binding)
    dims["blocks_per_seq"] = (dims["seq_len"] + dims["block_size"] - 1) // dims[
        "block_size"
    ]
    dims["num_blocks"] = dims["batch_size"] * dims["blocks_per_seq"]
    dims["num_tokens"] = dims["batch_size"] * dims["seq_len"]
    dims["num_quant_blocks"] = dims["head_dim"] // dims["quant_block_size"]
    dims["num_scale_bytes"] = dims["num_quant_blocks"] * 4
    dims["cache_stride"] = dims["head_dim"] + dims["num_scale_bytes"]
    return dims


def build_inputs(binding, dtype, device):
    del dtype
    dims = _dimensions(binding)
    generator = torch.Generator(device=device)
    generator.manual_seed(INPUT_SEED)

    k_cache = torch.randint(
        0,
        256,
        (
            dims["num_blocks"],
            dims["block_size"],
            dims["cache_stride"],
        ),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    flat_cache = k_cache.view(dims["num_blocks"], -1)
    scales = flat_cache[:, dims["block_size"] * dims["head_dim"] :].view(torch.float32)
    scales.copy_(
        torch.rand(
            dims["num_blocks"],
            dims["block_size"] * dims["num_quant_blocks"],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        + 0.01
    )

    block_table = torch.arange(
        dims["num_blocks"], dtype=torch.int32, device=device
    ).view(dims["batch_size"], dims["blocks_per_seq"])
    seq_lens = torch.full(
        (dims["batch_size"],),
        dims["seq_len"],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlen = torch.zeros(dims["batch_size"] + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_lens, dim=0)

    dst_k = torch.empty(
        (dims["num_tokens"], dims["head_dim"]), dtype=FP8_DTYPE, device=device
    )
    dst_scale = torch.empty(
        (dims["num_tokens"], dims["num_scale_bytes"]),
        dtype=torch.uint8,
        device=device,
    )
    return (k_cache, dst_k, dst_scale, block_table, cu_seqlen), {}


def config(binding, dtype):
    del dtype
    dims = _dimensions(binding)
    dst_k_shape = [dims["num_tokens"], dims["head_dim"]]
    dst_scale_shape = [dims["num_tokens"], dims["num_scale_bytes"]]
    return {
        "inputs": {
            "kv_cache": {
                "shape": [
                    dims["num_blocks"],
                    dims["block_size"],
                    dims["cache_stride"],
                ],
                "dtype": "torch.uint8",
                "note": "paged FP8 K + scale 源",
            },
            "dst_k": {
                "shape": dst_k_shape,
                "dtype": str(FP8_DTYPE),
                "note": "原地写回收集的 K",
            },
            "dst_scale": {
                "shape": dst_scale_shape,
                "dtype": "torch.uint8",
                "note": "原地写回收集的 scale",
            },
            "block_table": {
                "shape": [dims["batch_size"], dims["blocks_per_seq"]],
                "dtype": "torch.int32",
            },
            "cu_seq_lens": {
                "shape": [dims["batch_size"] + 1],
                "dtype": "torch.int32",
            },
        },
        "outputs": {
            "dst_k": {"shape": dst_k_shape, "dtype": str(FP8_DTYPE)},
            "dst_scale": {"shape": dst_scale_shape, "dtype": "torch.uint8"},
        },
        "dims": {
            **binding,
            "num_blocks": dims["num_blocks"],
            "num_tokens": dims["num_tokens"],
            "cache_stride": dims["cache_stride"],
        },
    }
