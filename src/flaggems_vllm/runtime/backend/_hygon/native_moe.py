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
import hashlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import torch

native_modules = {}
native_libraries = []
hip_runtime = None


def align_block_size(
    ids, num_experts, block_size, sorted_ids, expert_ids, total, expert_map=None
):
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
        native_modules["module_moe_utils"].moe_align_block_size(
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
    if (
        sys.version_info[:2] != (3, 10)
        or not importlib.metadata.version("torch").startswith("2.9.0+das.opt1.dtk2604")
        or torch.version.hip is None
        or importlib.metadata.version("vllm").split("+")[0] != "0.24.0"
    ):
        raise RuntimeError(
            "These Hygon binaries require Python 3.10, DAS torch 2.9.0 / "
            "DTK26.04 and vLLM 0.24.0"
        )
    directory = Path(sys.prefix) / "share/flaggems-vllm/hygon-native"
    artifacts = {
        "module_activation": "13cf939a8fb42c2a1b81375c308b15696189af3efd006d7f15a869d7f59d881d",
        "module_moe_utils": "3bde39f5a66ddb45d71be361b104b674735c5815666cb94f6d49a86eeb34f906",
    }
    for name, expected in artifacts.items():
        path = directory / f"{name}.so"
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"Install the verified Hygon native artifact at {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native_modules[name] = module
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
        "silu_and_mul": native_modules["module_activation"].silu_and_mul,
        "moe_sum": native_modules["module_moe_utils"].moe_sum,
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
