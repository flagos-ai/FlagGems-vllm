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

"""vLLM fused-MoE reference compatibility for the enflame GCU backend.

``tests/test_fused_experts_impl.py::test_fused_moe_vs_vllm`` compares the
FlagGems fused MoE against vLLM's own ``fused_experts_impl``.  On a GCU host
that reference cannot run as shipped, for two reasons this module works
around (both verified end-to-end on L600 / vllm 0.20.0 / torch_gcu 2.11.0):

1. vLLM detects no platform (pynvml sees no NVIDIA GPU, torch_gcu is not a
   known vLLM platform), so ``current_platform`` is ``UnspecifiedPlatform``
   and ``get_config_file_name`` raises ``NotImplementedError`` from
   ``UnspecifiedPlatform.get_device_name``.  The wrapper installed here
   falls back to the GCU device name; the config lookup then misses (no
   tuned JSON exists for it), vLLM logs "Using default MoE config" and its
   plain Triton ``fused_moe_kernel`` compiles and runs fine via FlagTree.
   Only the ``get_config_file_name`` attribute inside vLLM's fused_moe
   module is wrapped, so no other vLLM code path changes behaviour.

2. ``vllm._C`` cannot be dlopen'ed on this CUDA-less image (missing
   libcuda.so.1), so ``torch.ops._C.silu_and_mul`` and
   ``torch.ops._moe_C.moe_sum`` are missing and vLLM's
   ``apply_moe_activation`` / ``moe_sum`` call sites raise AttributeError.
   When the ops are absent, register them in the ``_C`` / ``_moe_C``
   namespaces and delegate to the FlagGems Triton implementations (no torch
   compute).  If vLLM's real extension ever loads, its registrations win and
   this shim stays dormant (the eager define is skipped when the op exists).

The patches only touch a vLLM that would otherwise raise; on hosts where
vLLM resolves a real platform and loads its extensions, everything is a
no-op.  Remove this shim once vLLM ships a GCU platform.
"""

import sys

_VLLM_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe"


def _gcu_device_name() -> str:
    try:
        from flaggems_vllm.runtime import torch_device_fn

        name = torch_device_fn.get_device_name()
    except Exception:
        name = ""
    if not name:
        try:
            from flaggems_vllm.runtime import device

            name = device.name
        except Exception:
            name = "gcu"
    return name or "gcu"


def _silu_and_mul_impl(out, input):
    from flaggems_vllm.runtime.backend._enflame.ops.fused_moe import (
        MoEActivation,
        apply_moe_activation,
    )

    apply_moe_activation(MoEActivation.SILU, out, input)


def _moe_sum_impl(input, output):
    from flaggems_vllm.runtime.backend._enflame.ops.fused_moe import moe_sum

    moe_sum(input, output)


def _moe_align_block_size_impl(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    expert_map=None,
):
    if expert_map is not None:
        raise NotImplementedError(
            "flaggems_vllm _moe_C::moe_align_block_size shim does not "
            "support expert_map (expert-parallel ranks); only the plain "
            "fused_experts_impl reference path is covered"
        )
    from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_triton

    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )


def _op_exists(namespace, name):
    try:
        getattr(getattr(torch_ops(namespace), name), "default")
        return True
    except (AttributeError, RuntimeError):
        return False


def torch_ops(namespace):
    import torch

    return getattr(torch.ops, namespace)


def _install_missing_torch_ops():
    # silu_and_mul(out!, input) -> ()
    if not _op_exists("_C", "silu_and_mul"):
        import torch

        torch.library.define("_C::silu_and_mul", "(Tensor(a!) out, Tensor input) -> ()")
        torch.library.impl(
            "_C::silu_and_mul", "CompositeExplicitAutograd", _silu_and_mul_impl
        )
    # moe_sum(input, output!) -> ()
    if not _op_exists("_moe_C", "moe_sum"):
        import torch

        torch.library.define(
            "_moe_C::moe_sum", "(Tensor input, Tensor(a!) output) -> ()"
        )
        torch.library.impl(
            "_moe_C::moe_sum", "CompositeExplicitAutograd", _moe_sum_impl
        )
    # moe_align_block_size(topk_ids, num_experts, block_size,
    #                      sorted_ids!, experts_ids!, num_tokens_post_pad!) -> ()
    if not _op_exists("_moe_C", "moe_align_block_size"):
        import torch

        torch.library.define(
            "_moe_C::moe_align_block_size",
            "(Tensor topk_ids, int num_experts, int block_size, "
            "Tensor(a!) sorted_token_ids, Tensor(a!) experts_ids, "
            "Tensor(a!) num_tokens_post_pad, Tensor? expert_map) -> ()",
        )
        torch.library.impl(
            "_moe_C::moe_align_block_size",
            "CompositeExplicitAutograd",
            _moe_align_block_size_impl,
        )


def _normalize_topk_ids_payload(topk_ids):
    """Re-expose int64 routing ids as the compact int32 payload words.

    torch_gcu (without TORCH_GCU_ENABLE_INT64_AND_UINT64=1) stores int64
    tensors as compact int32 words; a scalar i64 tl.load reads
    neighbour-joined garbage and aborts the device (see
    compiler_issues/int64_scalar_load_addressing.md). vLLM's naive
    expert-assignment path feeds ``topk_ids.view(-1)`` straight into such a
    load, so the reference needs int32 ids just like the FlagGems side.
    """
    import torch

    if not isinstance(topk_ids, torch.Tensor) or topk_ids.dtype != torch.int64:
        return topk_ids
    import os

    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return topk_ids
    flat = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    flat = flat.view(-1)
    words = flat.view(torch.int32)
    return words[: flat.numel()].view(topk_ids.shape)


def _wrap_fused_experts_impl(orig):
    def fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids, *args, **kwargs
    ):
        topk_ids = _normalize_topk_ids_payload(topk_ids)
        return orig(hidden_states, w1, w2, topk_weights, topk_ids, *args, **kwargs)

    fused_experts_impl.__wrapped__ = orig
    fused_experts_impl.__doc__ = orig.__doc__
    return fused_experts_impl


def _patch_fused_moe_module(module):
    if getattr(module, "_flaggems_gcu_moe_ref_compat", False):
        return
    module._flaggems_gcu_moe_ref_compat = True
    orig = module.get_config_file_name

    def get_config_file_name(E, N, dtype, block_shape=None):
        try:
            return orig(E, N, dtype, block_shape)
        except NotImplementedError:
            # UnspecifiedPlatform.get_device_name: vLLM cannot see the GCU.
            # Mirror the original name format with the GCU device name; the
            # file does not exist, so vLLM falls back to get_default_config.
            device_name = _gcu_device_name().replace(" ", "_")
            dtype_selector = "" if not dtype else f",dtype={dtype}"
            block_shape_selector = (
                ""
                if not block_shape or not all(block_shape)
                else f",block_shape={block_shape}"
            ).replace(" ", "")
            return (
                f"E={E},N={N},device_name={device_name}"
                f"{dtype_selector}{block_shape_selector}.json"
            )

    module.get_config_file_name = get_config_file_name

    impl = getattr(module, "fused_experts_impl", None)
    if impl is not None and not hasattr(impl, "__wrapped__"):
        module.fused_experts_impl = _wrap_fused_experts_impl(impl)


class _PatchingLoader:
    def __init__(self, real_loader):
        self._real_loader = real_loader

    def create_module(self, spec):
        return self._real_loader.create_module(spec)

    def exec_module(self, module):
        self._real_loader.exec_module(module)
        try:
            _patch_fused_moe_module(module)
        except Exception:
            pass


class _VllmMoeRefCompatFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _VLLM_FUSED_MOE_MODULE or fullname in sys.modules:
            return None
        try:
            start = sys.meta_path.index(self) + 1
        except ValueError:
            return None
        for finder in sys.meta_path[start:]:
            spec = finder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                # Wrap the loader in place: rebuilding a ModuleSpec would drop
                # has_location and leave the module without __file__, which
                # vLLM's get_moe_configs uses to find its configs directory.
                spec.loader = _PatchingLoader(spec.loader)
                return spec
        return None


def install_vllm_moe_ref_compat():
    """Install the vLLM fused-MoE reference compatibility (idempotent)."""
    _install_missing_torch_ops()
    module = sys.modules.get(_VLLM_FUSED_MOE_MODULE)
    if module is not None:
        _patch_fused_moe_module(module)
        return
    for finder in sys.meta_path:
        if isinstance(finder, _VllmMoeRefCompatFinder):
            return
    sys.meta_path.insert(0, _VllmMoeRefCompatFinder())
