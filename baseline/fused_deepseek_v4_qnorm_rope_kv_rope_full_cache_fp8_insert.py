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

OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert"
DTYPES = [torch.bfloat16, torch.float16]
# The native op writes q_fp8 and k_cache in place; q and kv stay read-only.
IS_INPLACE = True
MUTATED_INPUTS = (2, 3)
NATIVE_UNAVAILABLE_REASON = None
INPUT_SEED = 20261105

HEAD_DIM = 512
ROPE_DIM = 64
BLOCK_SIZE = 64
MAX_POS = 4096
EPS = 1e-6
FP8_DTYPE = torch.float8_e4m3fn
NUM_TOKENS = [1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072]
NUM_HEADS = [64, 128]


def native():
    global NATIVE_UNAVAILABLE_REASON
    try:
        import vllm._custom_ops  # noqa: F401

        op = getattr(torch.ops._C, OP_NAME)
    except (ImportError, AttributeError) as exc:
        NATIVE_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None
    NATIVE_UNAVAILABLE_REASON = None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": num_tokens, "num_heads": num_heads}
        for num_tokens in NUM_TOKENS
        for num_heads in NUM_HEADS
    ]


def key_shape(binding):
    # The benchmark's first tensor is q, shaped [num_tokens, num_heads, 512].
    return [binding["num_tokens"], binding["num_heads"], HEAD_DIM]


def build_inputs(binding, dtype, device):
    num_tokens = binding["num_tokens"]
    num_heads = binding["num_heads"]
    block_size = BLOCK_SIZE
    max_pos = max(MAX_POS, num_tokens)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    gen = torch.Generator(device=device)
    gen.manual_seed(INPUT_SEED)

    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=dtype, device=device, generator=gen
    )
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device, generator=gen)
    q_fp8 = torch.empty(num_tokens, num_heads, HEAD_DIM, dtype=FP8_DTYPE, device=device)
    k_cache = torch.zeros(
        num_blocks, block_size, HEAD_DIM, dtype=FP8_DTYPE, device=device
    )

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = _cos_sin_cache(max_pos, device)

    # Production scales are per-tensor and both are 1.0 on the current FlashInfer
    # path; they stay tensors so the kernel reads them from device memory.
    fp8_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    q_fp8_scale_inv = torch.tensor([1.0], dtype=torch.float32, device=device)

    args = (
        q,
        kv,
        q_fp8,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        fp8_scale,
        q_fp8_scale_inv,
        EPS,
        block_size,
        True,
    )
    return args, {}


def _cos_sin_cache(max_pos, device):
    """cos || sin cache in the DeepseekV4ScalingRotaryEmbedding layout."""
    if max_pos <= 8192:
        base_value = 10000.0
    elif max_pos <= 32768:
        base_value = 20000.0
    elif max_pos <= 65536:
        base_value = 40000.0
    elif max_pos <= 98304:
        base_value = 60000.0
    else:
        base_value = 100000.0

    arange = torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base_value ** (arange / ROPE_DIM))
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _dense(shape, dtype_name):
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return {"shape": list(shape), "dtype": dtype_name, "stride": stride}


def config(binding, dtype):
    num_tokens = binding["num_tokens"]
    num_heads = binding["num_heads"]
    block_size = BLOCK_SIZE
    max_pos = max(MAX_POS, num_tokens)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    dtype_name = str(dtype)
    fp8_name = str(FP8_DTYPE)

    return {
        "inputs": {
            "q": _dense([num_tokens, num_heads, HEAD_DIM], dtype_name),
            "kv": _dense([num_tokens, HEAD_DIM], dtype_name),
            "q_fp8": _dense([num_tokens, num_heads, HEAD_DIM], fp8_name),
            "k_cache": _dense([num_blocks, block_size, HEAD_DIM], fp8_name),
            "slot_mapping": _dense([num_tokens], "torch.int64"),
            "position_ids": _dense([num_tokens], "torch.int64"),
            "cos_sin_cache": _dense([max_pos, ROPE_DIM], "torch.float32"),
            "fp8_scale": _dense([1], "torch.float32"),
            "q_fp8_scale_inv": _dense([1], "torch.float32"),
            "eps": {"scalar": EPS},
            "cache_block_size": {"scalar": block_size},
            "apply_q_norm": {"scalar": True},
        },
        "outputs": {
            "q_fp8": {
                **_dense([num_tokens, num_heads, HEAD_DIM], fp8_name),
                "aliases": "q_fp8",
            },
            "k_cache": {
                **_dense([num_blocks, block_size, HEAD_DIM], fp8_name),
                "aliases": "k_cache",
                "written_slots": num_tokens,
            },
        },
        "dims": {
            "N": num_tokens,
            "H": num_heads,
            "D": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "nope_dim": HEAD_DIM - ROPE_DIM,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "max_pos": max_pos,
            "num_heads_padded": num_heads,
        },
        "notes": {
            "slot_layout": "slot_mapping[i] = i, every row inserts one cache slot",
            "cache_layout": "[num_blocks, block_size, 512], stride(2) == 1",
            "kv_quant": "per-tensor e4m3, saturation at +-448",
            "q_quant": "per-tensor e4m3 via q_fp8_scale_inv (multiplied, not inverted)",
            "rope": "GPT-J interleaved on the trailing 64 dims, first 448 sealed",
        },
        "input_seed": INPUT_SEED,
        "mutation": ["q_fp8 (fully rewritten every call)", "k_cache (slots rewritten)"],
    }
