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

import functools
import re

import triton
from packaging.version import InvalidVersion, Version


def _coerce_triton_version(version: str) -> Version:
    try:
        return Version(version)
    except InvalidVersion:
        release = []
        for part in version.split("+", 1)[0].split(".")[:3]:
            match = re.match(r"\d+", part)
            release.append(match.group(0) if match else "0")
        while len(release) < 3:
            release.append("0")
        return Version(".".join(release))


def _triton_version_at_least(major: int, minor: int, patch: int = 0) -> bool:
    version = str(getattr(triton, "__version__", "0.0.0"))
    return _coerce_triton_version(version) >= Version(f"{major}.{minor}.{patch}")


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    if not _triton_version_at_least(major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language as _tle  # noqa: F401

        return True
    except ImportError:
        return False


def has_triton_tle_attrs(
    attrs: tuple[str, ...],
    major: int = 0,
    minor: int = 0,
    patch: int = 0,
) -> bool:
    if not _triton_version_at_least(major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language as tle
    except ImportError:
        return False

    for attr in attrs:
        obj = tle
        for part in attr.split("."):
            if not hasattr(obj, part):
                return False
            obj = getattr(obj, part)
    return True


HAS_TLE = has_triton_tle()


def has_tle_device_mesh() -> bool:
    if not HAS_TLE:
        return False
    try:
        import triton.experimental.tle.language as tle_exp

        return hasattr(tle_exp, "device_mesh")
    except ImportError:
        return False


HAS_TLE_DEVICE_MESH = has_tle_device_mesh()


@functools.lru_cache(maxsize=None)
def _compiled_pipeline_options() -> frozenset:
    """The FlagTree software-pipelining switches this build accepts.

    FlagTree's NVIDIA backend exposes them as ``pipeline_*`` compile options
    (``CUDAOptions``); upstream Triton and the other backends do not have them
    and reject unknown launch keywords, so they are empty there.
    """
    try:
        from triton.runtime.driver import driver

        if driver.active.get_current_target().backend != "cuda":
            return frozenset()
        from triton.backends.nvidia.compiler import CUDAOptions
    except Exception:
        return frozenset()
    return frozenset(
        name
        for name in CUDAOptions.__dataclass_fields__
        if name.startswith("pipeline_")
    )


def pipeline_enhance_enabled() -> bool:
    """Whether the FlagTree pipelining enhancements are on at run time.

    ``FLAGTREE_PIPELINE_ENHANCE=0`` (``knobs.nvidia.pipeline_enhance``) keeps the
    compile options but leaves the passes on their upstream path, so the
    per-shape choices measured with the enhancements must not be applied. A
    backend without the knob is not FlagTree and answers True, which the
    capability check then narrows down.

    Not cached: the knob can be changed between compilations (``knobs.scope()``)
    and every caller here runs once per launch, not per iteration.
    """
    try:
        from triton import knobs

        return bool(knobs.nvidia.pipeline_enhance)
    except Exception:
        return True


def has_pipeline_options() -> bool:
    """Whether the active backend takes FlagTree's ``pipeline_*`` compile options.

    This is the capability question only: the build has the options and the
    enhancements that give them meaning are switched on.
    """
    return bool(_compiled_pipeline_options()) and pipeline_enhance_enabled()


def pipeline_options(**options) -> dict:
    """Keep the FlagTree pipeline options the active backend supports.

    Use it to pass per-kernel pipelining switches as launch keywords or in a
    ``triton.Config``, e.g. ``kernel[grid](..., **pipeline_options(
    pipeline_merge_loads="any"))``; on a backend without them, or with
    ``FLAGTREE_PIPELINE_ENHANCE=0``, it returns ``{}``.
    """
    if not has_pipeline_options():
        return {}
    supported = _compiled_pipeline_options()
    return {name: value for name, value in options.items() if name in supported}


@functools.lru_cache(maxsize=None)
def _is_sm90(device_index) -> bool:
    import torch

    try:
        return torch.cuda.get_device_capability(device_index) == (9, 0)
    except (AssertionError, RuntimeError):
        return False


def pipeline_tuning_applies(device) -> bool:
    """Whether the fixed per-shape pipeline choices of the wrappers apply.

    ``has_pipeline_options()`` and SM90: the wrappers pick a ``num_stages`` or a
    ``pipeline_*`` value measured on H20 for the wgmma pipeliner, which is what
    SM90 runs. Other GPUs keep their own configuration even when the options
    exist (SM100 pipelines tcgen05 through a different pass), and so do other
    backends and upstream Triton.
    """
    if getattr(device, "type", None) != "cuda":
        return False
    if not has_pipeline_options():
        return False
    index = device.index
    if index is None:
        import torch

        index = torch.cuda.current_device()
    return _is_sm90(index)
