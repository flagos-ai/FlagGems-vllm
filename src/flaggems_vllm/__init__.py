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

"""
flaggems_vllm - DNN operations implemented with Triton
"""

import torch

import flaggems_vllm.ops as _ops_module
from flaggems_vllm import testing  # noqa: F401
from flaggems_vllm import runtime
from flaggems_vllm.config import aten_patch_list, resolve_user_setting
from flaggems_vllm.ops import *  # noqa: F401,F403
from flaggems_vllm.runtime.register import Register

device = runtime.device.name
vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())

__version__ = "0.1.0"

_COMPAT_OP_ALIASES = {
    "ops_add_rms_norm": "fused_add_rms_norm",
    "ops_deepseek_v4_qnorm_rope_kv_rope_quant_insert": "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
    "ops_experts_impl": "fused_experts_impl",
    "ops_inv_rope_fp8_quant": "fused_inv_rope_fp8_quant",
    "ops_recurrent_gated_delta_rule_fwd": "fused_recurrent_gated_delta_rule_fwd",
}
for _alias, _target in _COMPAT_OP_ALIASES.items():
    if _target in globals():
        globals()[_alias] = globals()[_target]

_FULL_CONFIG = tuple(
    (op_name, globals()[op_name])
    for op_name in getattr(_ops_module, "__all__", [])
    if op_name in globals() and callable(globals()[op_name])
)

if "weight_norm_interface" in globals():
    _FULL_CONFIG += (("_weight_norm_interface", weight_norm_interface),)
if "weight_norm_interface_backward" in globals():
    _FULL_CONFIG += (
        ("_weight_norm_interface_backward", weight_norm_interface_backward),
    )

FULL_CONFIG_BY_FUNC: dict = {}
for _item in _FULL_CONFIG:
    if not _item or len(_item) < 2:
        continue
    fn = _item[1]
    func_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
    FULL_CONFIG_BY_FUNC.setdefault(func_name, []).append(_item)


def enable(
    lib=aten_lib,
    unused=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register all FlagGems-vllm ops except those explicitly excluded.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        unused: Which ops to skip. Supported forms:
            - list/tuple/set of function names (e.g., ["masked_fill", "mul"]).
            - str path to a YAML file ending with .yml/.yaml containing an
              `exclude:` list.
            - "default" or None: auto-load
              vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]
              enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable FlagGems-vllm logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Notes:
        - If the exclude list/YAML resolves to empty, all ops are registered.
    """
    global current_work_registrar
    exclude_ops = resolve_user_setting(unused, "exclude")
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=[],
        user_exclude_ops=exclude_ops,
        cpp_patched_ops=list(set(aten_patch_list)),
        lib=lib,
    )


def only_enable(
    lib=aten_lib,
    include=None,
    registrar=registrar,
    record=False,
    once=False,
    path=None,
):
    """Register only the specified FlagGems-vllm ops and skip the rest.

    Args:
        lib: torch.library.Library instance to register into. Defaults to the
            global `aten_lib` (IMPL mode).
        include: Which ops to register. Supported forms:
            - list/tuple/set of function names (e.g., ["rms_norm", "softmax"]).
            - str path to a YAML file ending with .yml/.yaml (expects a list or
              an `include:` key).
            - "default" or None: auto-load
              vendor/arch-specific
              runtime/backend/_<vendor>/[<arch>/]
              only_enable_configs.yaml if present.
        registrar: Registrar class; defaults to `Register`.
        record: Whether to enable Flag logging.
        once: When True, log only once.
        path: Optional log output path when recording.

    Classic usage:
        - Only register a few ops:
            only_enable(include=["rms_norm", "softmax"])
        - Use vendor default YAML:
            only_enable(include="default")  # or include=None
        - Use a custom YAML:
            only_enable(include="/path/to/only_enable.yaml")

    Notes:
        - If the include list/YAML resolves to empty or none of the names match
          known ops, the function warns and returns without registering.
    """
    import warnings

    include_ops = resolve_user_setting(include, "include")
    if not include_ops:
        warnings.warn(
            "only_enable failed: No include entries" " resolved from list or yaml."
        )
        return

    global current_work_registrar
    current_work_registrar = registrar(
        _FULL_CONFIG,
        user_include_ops=include_ops,
        user_exclude_ops=[],
        cpp_patched_ops=list(set(aten_patch_list)),
        full_config_by_func=FULL_CONFIG_BY_FUNC,
        lib=lib,
    )


class use_dnn:
    """
    The 'include' parameter has higher priority than 'exclude'.
    When 'include' is not None, use_dnn will not process 'exclude'.
    """

    def __init__(self, exclude=None, include=None, record=False, once=False, path=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.exclude = exclude if isinstance(exclude, (list, tuple, set, str)) else []
        self.include = include if isinstance(include, (list, tuple, set, str)) else []
        self.registrar = Register
        self.record = record
        self.once = once
        self.path = path

    def __enter__(self):
        if self.include:
            only_enable(
                lib=self.lib,
                include=self.include,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )
        else:
            enable(
                lib=self.lib,
                unused=self.exclude,
                registrar=self.registrar,
                record=self.record,
                once=self.once,
                path=self.path,
            )

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        if torch.__version__ >= "2.5":
            self.lib._destroy()
        del self.lib
        del self.exclude
        del self.include
        del self.registrar
        current_work_registrar = None


use_gems = use_dnn


def all_registered_ops():
    if current_work_registrar is None:
        return []
    return current_work_registrar.get_all_ops()


def all_registered_keys():
    if current_work_registrar is None:
        return []
    return current_work_registrar.get_all_keys()


__all__ = [
    "SUPPORTED_FP8_DTYPE",
    "enable",
    "only_enable",
    "use_dnn",
    "use_gems",
    "all_registered_ops",
    "all_registered_keys",
]
