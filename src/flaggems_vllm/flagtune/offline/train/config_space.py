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

"""Resolve runtime-owned FlagTune candidate spaces for training collection."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def _runtime_tuner(op_id: str, variant: str, platform: Optional[str] = None):
    """Resolve a trusted code-side binding instead of copying operator spaces."""
    import flaggems_vllm
    from flaggems_vllm import runtime
    from flaggems_vllm.flagtune.offline.contracts.operator_contract import (
        OperatorConfigError,
        resolve_public_operator,
    )
    from flaggems_vllm.flagtune.offline.runtime.tuner import (
        find_flagtune_benchmark_target,
    )
    from flaggems_vllm.flagtune.offline.train.route.common import _normalized_platform

    if platform is not None:
        requested = _normalized_platform(str(platform)) or str(platform).lower()
        active = str(runtime.device.vendor_name).lower()
        active = _normalized_platform(active) or active
        if requested != active:
            raise ValueError(
                f"runtime candidate lookup requested {requested!r}, active backend is {active!r}"
            )
    try:
        operator = resolve_public_operator(flaggems_vllm, op_id)
    except OperatorConfigError:
        return None
    # Missing or ambiguous variant bindings must fail, never choose an arbitrary
    # tuner just because its JIT body or candidate names look similar.
    _, tuner = find_flagtune_benchmark_target(operator, op_id, variant)
    return tuner


def runtime_op_name_for_variant(
    op_id: str, variant: str, platform: Optional[str] = None
) -> Optional[str]:
    """Return the Expand name from the actual registered kernel binding."""
    tuner = _runtime_tuner(op_id, variant, platform)
    return None if tuner is None else tuner._flagtune_expand_op_name


def runtime_configs_for_variant(
    op_id: str,
    variant: str,
    *,
    platform: Optional[str] = None,
    yaml_path: Optional[str] = None,
    pre_hook: Any = None,
) -> Optional[list[Any]]:
    """Resolve the same candidate space as the bound runtime tuner.

    No W8A8 or ordinary-MM domain is inferred from a variant string. Default
    fallback for a missing Expand YAML remains the tuner's own behavior. An
    explicit path/hook override retains the generic offline API without
    modifying the live tuner's state.
    """
    from flaggems_vllm import runtime

    tuner = _runtime_tuner(op_id, variant, platform)
    if tuner is None:
        return None
    if yaml_path is None and pre_hook is None:
        configs, _ = tuner._flagtune_configs_for_mode(
            tuner._flagtune_op_name or op_id, runtime.TuningMode.EXPANDED
        )
        return list(configs)
    op_name = tuner._flagtune_expand_op_name
    if op_name is None:
        return list(tuner._flagtune_default_configs)
    configs = runtime.ops_get_configs(
        op_name,
        yaml_path=yaml_path or tuner._flagtune_yaml_path,
        pre_hook=pre_hook if pre_hook is not None else tuner._flagtune_pre_hook,
    )
    return list(configs) if configs else list(tuner._flagtune_default_configs)


def runtime_configs_hash(configs: list[Any]) -> str:
    """Return a stable digest for a materialized runtime candidate domain."""
    normalized = []
    for config in configs:
        normalized.append(
            {
                "kwargs": dict(sorted(getattr(config, "kwargs", {}).items())),
                "launch": {
                    name: getattr(config, name, None)
                    for name in (
                        "num_warps",
                        "num_stages",
                        "num_ctas",
                        "num_buffers_warp_spec",
                        "num_consumer_groups",
                        "reg_dec_producer",
                        "reg_inc_consumer",
                        "maxnreg",
                    )
                },
                # The hook itself is executable and not serializable. Its
                # presence is nevertheless part of the candidate semantics;
                # the source YAML/config hash binds the concrete hook code.
                "has_pre_hook": getattr(config, "pre_hook", None) is not None,
            }
        )
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
