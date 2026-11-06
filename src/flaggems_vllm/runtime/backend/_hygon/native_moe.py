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

"""Supply the native dependencies of the Hygon vLLM BF16 MoE baseline."""

import ctypes

import torch

native_libraries = []
hip_runtime = None


def align_block_size(
    ids, num_experts, block_size, sorted_ids, expert_ids, total, expert_map=None
):
    import aiter

    if expert_map is not None:
        raise NotImplementedError("This Hygon native provider has no expert_map API")
    with torch.cuda.device(ids.device):
        # Aiter writes valid routes; vLLM also needs invalid-route padding.
        status = hip_runtime.hipMemsetD32Async(
            sorted_ids.data_ptr(),
            ids.numel(),
            sorted_ids.numel(),
            torch.cuda.current_stream(ids.device).cuda_stream,
        )
        if status:
            raise RuntimeError(f"hipMemsetD32Async failed with HIP error {status}")
        aiter.moe_align_block_size(
            ids, num_experts, block_size, sorted_ids, expert_ids, total
        )


def register_native_moe_baseline() -> None:
    """Register missing baseline primitives before starting benchmark workers."""
    global hip_runtime
    from vllm import _custom_ops  # noqa: F401

    operations = (
        ("_C", "silu_and_mul", "silu_and_mul(Tensor! result, Tensor input) -> ()"),
        ("_moe_C", "moe_sum", "moe_sum(Tensor input, Tensor! output) -> ()"),
        (
            "_moe_C",
            "moe_align_block_size",
            "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
            "Tensor! sorted_token_ids, Tensor! experts_ids, Tensor! num_tokens_post_pad, "
            "Tensor? expert_map=None) -> ()",
        ),
    )
    missing = []
    for namespace, name, schema in operations:
        try:
            torch._C._dispatch_find_schema_or_throw(f"{namespace}::{name}", "")
        except RuntimeError:
            missing.append((namespace, name, schema, False))
            continue
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{namespace}::{name}", "CUDA"
        ):
            missing.append((namespace, name, schema, True))
    if not missing:
        return

    import aiter

    if any(name == "moe_align_block_size" for _, name, _, _ in missing):
        hip_runtime = ctypes.CDLL("libamdhip64.so")
        hip_runtime.hipMemsetD32Async.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        hip_runtime.hipMemsetD32Async.restype = ctypes.c_int
    implementations = {
        "silu_and_mul": aiter.silu_and_mul,
        "moe_sum": aiter.moe_sum,
        "moe_align_block_size": align_block_size,
    }
    for namespace, name, schema, has_schema in missing:
        if not has_schema:
            definition = torch.library.Library(namespace, "FRAGMENT")
            definition.define(schema)
            native_libraries.append(definition)
        library = torch.library.Library(namespace, "IMPL", "CUDA")
        library.impl(name, implementations[name])
        native_libraries.append(library)
