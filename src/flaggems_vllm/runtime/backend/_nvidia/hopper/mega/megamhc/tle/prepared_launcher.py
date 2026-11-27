"""Explicit, launch-free preparation for one compiled MegaMHC kernel.

This module is the production-oriented counterpart of ``compiled_launcher``.
It does not intercept ``JITFunction.run`` and it does not execute the kernel
during setup.  Instead, each admitted Plan must describe its launch explicitly
with a :class:`LaunchSpec`.  Setup calls ``JITFunction.warmup`` once, resolves
the bound ABI once, and retains the resulting ``CompiledKernel`` runner.

The current scope is intentionally narrow: one CUDA device, non-lite FlagTree,
one host owner, one ordered CUDA stream, and kernels without TLE node-buffer
bindings.  Unsupported runtime features fail closed instead of silently
bypassing their launch semantics.
"""

from __future__ import annotations

import operator
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from types import MappingProxyType

_TRUE_ENV_VALUES = frozenset(("1", "ON", "TRUE"))
_MISSING = object()


class PreparedLauncherError(RuntimeError):
    """Base error for explicit prepared launchers."""


class UnsupportedPreparedRuntimeError(PreparedLauncherError):
    """Raised when direct launch would bypass required runtime semantics."""


class PreparedLaunchContractError(PreparedLauncherError):
    """Raised when a Plan's explicit LaunchSpec is internally inconsistent."""


@dataclass(frozen=True)
class CooperativeGridAdmission:
    """Post-compilation proof that a cooperative grid can be resident."""

    grid_ctas: int
    active_blocks_per_sm: int
    multiprocessor_count: int
    resident_cta_capacity: int


@dataclass(frozen=True)
class ClusterLaunchAdmission:
    """Post-compilation proof that the requested CTA cluster is supported."""

    cluster_dims: tuple[int, int, int]
    required_cluster_size: int
    maximum_cluster_size: int


@dataclass(frozen=True)
class StaticTensorBindingSnapshot:
    """Preparation-time identity, storage, pointer, and layout signature."""

    object_id: int
    data_ptr: int
    storage_base: int
    storage_nbytes: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: object
    device: object
    requires_grad: bool

    @classmethod
    def capture(cls, value: object) -> "StaticTensorBindingSnapshot":
        storage_getter = getattr(value, "untyped_storage", None)
        storage_offset_getter = getattr(value, "storage_offset", None)
        stride_getter = getattr(value, "stride", None)
        if not callable(storage_getter):
            raise PreparedLaunchContractError(
                "static bound value must expose untyped_storage()"
            )
        if not callable(storage_offset_getter):
            raise PreparedLaunchContractError(
                "static bound value must expose storage_offset()"
            )
        if not callable(stride_getter):
            raise PreparedLaunchContractError("static bound value must expose stride()")
        storage = storage_getter()
        storage_data_ptr = getattr(storage, "data_ptr", None)
        storage_nbytes = getattr(storage, "nbytes", None)
        if not callable(storage_data_ptr) or not callable(storage_nbytes):
            raise PreparedLaunchContractError(
                "static bound storage must expose data_ptr() and nbytes()"
            )
        shape = getattr(value, "shape", None)
        if shape is None:
            raise PreparedLaunchContractError("static bound value must expose a shape")
        if not hasattr(value, "dtype") or not hasattr(value, "device"):
            raise PreparedLaunchContractError(
                "static bound value must expose dtype and device"
            )
        return cls(
            object_id=id(value),
            data_ptr=_data_pointer("static bound value", value),
            storage_base=int(storage_data_ptr()),
            storage_nbytes=int(storage_nbytes()),
            storage_offset=int(storage_offset_getter()),
            shape=tuple(int(dimension) for dimension in shape),
            stride=tuple(int(step) for step in stride_getter()),
            dtype=value.dtype,
            device=value.device,
            requires_grad=bool(getattr(value, "requires_grad", False)),
        )


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise PreparedLaunchContractError(f"{name} must be a positive integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as exc:
        raise PreparedLaunchContractError(f"{name} must be a positive integer") from exc
    if normalized <= 0:
        raise PreparedLaunchContractError(f"{name} must be a positive integer")
    return normalized


def _alignment_tuple(
    name: str,
    values: Sequence[object],
    expected_count: int,
) -> tuple[int, ...]:
    normalized = tuple(_positive_integer(name, value) for value in values)
    if len(normalized) != expected_count:
        raise PreparedLaunchContractError(
            f"{name} must contain exactly {expected_count} entries"
        )
    if any(value & (value - 1) for value in normalized):
        raise PreparedLaunchContractError(f"{name} entries must be powers of two")
    return normalized


def _data_pointer(name: str, value: object) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise PreparedLaunchContractError(f"{name} must expose data_ptr()")
    try:
        return int(data_ptr())
    except (TypeError, ValueError) as exc:
        raise PreparedLaunchContractError(
            f"{name}.data_ptr() must return an integer"
        ) from exc


def _validate_pointer_alignment(name: str, value: object, alignment: int) -> None:
    if _data_pointer(name, value) % alignment:
        raise PreparedLaunchContractError(
            f"{name} pointer must be {alignment}-byte aligned"
        )


@dataclass(frozen=True)
class LaunchSpec:
    """Immutable description of one exact JIT launch specialization.

    ``dynamic_bound_indices`` are indices in the JIT function's *bound*
    argument order, not necessarily indices in ``args``.  MegaMHC Plans must
    declare the eight activation/output slots explicitly.  ``dynamic_values``
    contains the corresponding tensors from the Plan used for preparation and
    is checked by identity after FlagTree binds the signature.  The four
    ``static_bound_values`` are ordered as fn/scale/base/norm.  Token count and
    both alignment tuples are immutable factory inputs rather than duplicated
    caller declarations.
    """

    jit_function: object
    grid: object
    args: tuple[object, ...]
    kwargs: Mapping[str, object]
    token_count: int
    dynamic_bound_indices: tuple[int, ...]
    dynamic_values: tuple[object, ...]
    dynamic_alignments: tuple[int, ...]
    static_bound_indices: tuple[int, ...]
    static_bound_values: tuple[object, ...]
    static_bound_alignments: tuple[int, ...]
    label: str = "megamhc"
    requires_full_grid_residency: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))
        cooperative_launch = self.kwargs.get("launch_cooperative_grid", False)
        if not isinstance(cooperative_launch, bool):
            raise PreparedLaunchContractError(
                "launch_cooperative_grid must be a boolean"
            )
        if self.requires_full_grid_residency is None:
            object.__setattr__(self, "requires_full_grid_residency", cooperative_launch)
        elif bool(self.requires_full_grid_residency) != cooperative_launch:
            raise PreparedLaunchContractError(
                "requires_full_grid_residency must match launch_cooperative_grid"
            )
        object.__setattr__(
            self, "token_count", _positive_integer("token_count", self.token_count)
        )
        object.__setattr__(
            self,
            "dynamic_bound_indices",
            tuple(int(index) for index in self.dynamic_bound_indices),
        )
        object.__setattr__(self, "dynamic_values", tuple(self.dynamic_values))
        object.__setattr__(
            self,
            "dynamic_alignments",
            _alignment_tuple("dynamic_alignments", self.dynamic_alignments, 8),
        )
        object.__setattr__(
            self,
            "static_bound_indices",
            tuple(int(index) for index in self.static_bound_indices),
        )
        object.__setattr__(self, "static_bound_values", tuple(self.static_bound_values))
        object.__setattr__(
            self,
            "static_bound_alignments",
            _alignment_tuple(
                "static_bound_alignments", self.static_bound_alignments, 4
            ),
        )
        if len(self.dynamic_bound_indices) != 8:
            raise PreparedLaunchContractError(
                "LaunchSpec must declare exactly eight dynamic arguments"
            )
        if len(self.dynamic_bound_indices) != len(self.dynamic_values):
            raise PreparedLaunchContractError(
                "dynamic_bound_indices and dynamic_values must have equal length"
            )
        if len(set(self.dynamic_bound_indices)) != len(self.dynamic_bound_indices):
            raise PreparedLaunchContractError("dynamic_bound_indices must be unique")
        if any(index < 0 for index in self.dynamic_bound_indices):
            raise PreparedLaunchContractError(
                "dynamic_bound_indices must be non-negative"
            )
        if len(self.static_bound_indices) != len(self.static_bound_values):
            raise PreparedLaunchContractError(
                "static_bound_indices and static_bound_values must have equal length"
            )
        if len(self.static_bound_indices) != 4:
            raise PreparedLaunchContractError(
                "LaunchSpec must declare exactly four static bound arguments"
            )
        if len(set(self.static_bound_indices)) != len(self.static_bound_indices):
            raise PreparedLaunchContractError("static_bound_indices must be unique")
        if any(index < 0 for index in self.static_bound_indices):
            raise PreparedLaunchContractError(
                "static_bound_indices must be non-negative"
            )
        if set(self.static_bound_indices) & set(self.dynamic_bound_indices):
            raise PreparedLaunchContractError(
                "static and dynamic bound indices must be disjoint"
            )
        x_shape = getattr(self.dynamic_values[0], "shape", None)
        if x_shape is None or not x_shape:
            raise PreparedLaunchContractError(
                "the first dynamic value must expose a non-empty shape"
            )
        if int(x_shape[0]) != self.token_count:
            raise PreparedLaunchContractError(
                "token_count does not match the first dynamic value"
            )


@dataclass(frozen=True)
class _RuntimeAdapters:
    current_device: Callable[[], object]
    current_stream: Callable[[object], object]
    validate_node_launch_bindings: Callable[[object, object, object], None]
    device_properties: Callable[[object], Mapping[str, object]]
    cooperative_launch_supported: Callable[[object], bool]
    active_blocks_per_sm: Callable[[object, int, int], int]
    maximum_cluster_size: Callable[
        [
            object,
            int,
            int,
            tuple[int, int, int],
            tuple[int, int, int],
        ],
        int,
    ]


def _cuda_success(error: object, operation: str) -> None:
    try:
        status = int(error)
    except (TypeError, ValueError) as exc:
        raise UnsupportedPreparedRuntimeError(
            f"{operation} returned an invalid CUDA status"
        ) from exc
    if status:
        raise UnsupportedPreparedRuntimeError(
            f"{operation} failed with CUDA status {status}"
        )


def _cuda_occupancy_count(error: object, value: object, operation: str) -> int:
    _cuda_success(error, operation)
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise UnsupportedPreparedRuntimeError(
            f"{operation} returned invalid occupancy"
        ) from exc
    if count < 0:
        raise UnsupportedPreparedRuntimeError(
            f"{operation} returned negative occupancy {count}"
        )
    return count


def _load_occupancy_function(compiled: object) -> object:
    from triton.runtime.errors import OutOfResources

    try:
        getattr(compiled, "_init_handles")()
    except OutOfResources as exc:
        raise UnsupportedPreparedRuntimeError(str(exc)) from exc
    return getattr(compiled, "function")


def _cooperative_launch_supported(device: object) -> bool:
    try:
        from cuda.bindings import driver as cuda_driver

        error, value = cuda_driver.cuDeviceGetAttribute(
            cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
            int(operator.index(device)),
        )
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "CUDA driver bindings do not expose cooperative-launch capability"
        ) from exc
    _cuda_success(error, "cuDeviceGetAttribute(COOPERATIVE_LAUNCH)")
    return bool(value)


def _active_blocks_per_sm(compiled: object, block_threads: int, shared: int) -> int:
    try:
        from cuda.bindings import driver as cuda_driver

        function = _load_occupancy_function(compiled)
        error, active = cuda_driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            function,
            block_threads,
            shared,
        )
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel occupancy cannot be queried through CUDA"
        ) from exc
    return _cuda_occupancy_count(
        error, active, "cuOccupancyMaxActiveBlocksPerMultiprocessor"
    )


def _maximum_cluster_size(
    compiled: object,
    block_threads: int,
    shared: int,
    grid3: tuple[int, int, int],
    cluster_dims: tuple[int, int, int],
) -> int:
    """Query the compiled kernel's supported cluster size on this device."""
    try:
        from cuda.bindings import driver as cuda_driver

        function = _load_occupancy_function(compiled)
        required_cluster_size = prod(cluster_dims)
        if required_cluster_size > 8:
            error = cuda_driver.cuFuncSetAttribute(
                function,
                cuda_driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )[0]
            _cuda_success(error, "cuFuncSetAttribute(NON_PORTABLE_CLUSTER_SIZE)")

        config = cuda_driver.CUlaunchConfig()
        config.gridDimX = grid3[0] * cluster_dims[0]
        config.gridDimY = grid3[1] * cluster_dims[1]
        config.gridDimZ = grid3[2] * cluster_dims[2]
        config.blockDimX = block_threads
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = shared
        config.hStream = 0
        cluster_attribute = cuda_driver.CUlaunchAttribute()
        cluster_attribute.id = (
            cuda_driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        )
        cluster_attribute.value.clusterDim.x = cluster_dims[0]
        cluster_attribute.value.clusterDim.y = cluster_dims[1]
        cluster_attribute.value.clusterDim.z = cluster_dims[2]
        config.attrs = [cluster_attribute]
        config.numAttrs = 1
        error, maximum = cuda_driver.cuOccupancyMaxPotentialClusterSize(
            function,
            config,
        )
    except (AttributeError, ImportError, OverflowError, TypeError, ValueError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel cluster capability cannot be queried through CUDA"
        ) from exc
    return _cuda_occupancy_count(error, maximum, "cuOccupancyMaxPotentialClusterSize")


def _default_runtime_adapters() -> _RuntimeAdapters:
    try:
        from triton.runtime._distributed import validate_node_launch_bindings
    except ImportError as exc:
        raise UnsupportedPreparedRuntimeError(
            "the installed FlagTree runtime does not expose direct node-launch "
            "binding validation; use the tested FlagTree main runtime"
        ) from exc
    from triton.runtime.driver import driver

    return _RuntimeAdapters(
        current_device=driver.active.get_current_device,
        current_stream=driver.active.get_current_stream,
        validate_node_launch_bindings=validate_node_launch_bindings,
        device_properties=driver.active.utils.get_device_properties,
        cooperative_launch_supported=_cooperative_launch_supported,
        active_blocks_per_sm=_active_blocks_per_sm,
        maximum_cluster_size=_maximum_cluster_size,
    )


def _lite_dist_requested() -> bool:
    return os.environ.get("FLAGTREE_LITE_DIST", "").strip().upper() in (
        _TRUE_ENV_VALUES
    )


def _resolve_compiled(value: object) -> object:
    result = getattr(value, "result", None)
    return result() if callable(result) else value


def _canonical_grid(
    grid: object, bound_args: Mapping[str, object]
) -> tuple[int, int, int]:
    resolved = grid(bound_args) if callable(grid) else grid
    if isinstance(resolved, int):
        values = (resolved,)
    else:
        try:
            values = tuple(resolved)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PreparedLaunchContractError(
                "grid must be an int, a 1-3 element sequence, or a callable"
            ) from exc
    if not 1 <= len(values) <= 3:
        raise PreparedLaunchContractError("grid must contain 1-3 dimensions")
    try:
        grid3 = tuple(int(value) for value in values) + (1,) * (3 - len(values))
    except (TypeError, ValueError) as exc:
        raise PreparedLaunchContractError("grid values must be integers") from exc
    if any(value <= 0 for value in grid3):
        raise PreparedLaunchContractError("grid values must be positive")
    return grid3  # type: ignore[return-value]


def _compiled_launch_resources(compiled: object) -> tuple[int, int]:
    """Return the physical CTA size and dynamic shared memory after JIT."""
    metadata = getattr(compiled, "metadata", None)
    try:
        block_threads = (
            _positive_integer("compiled num_warps", getattr(metadata, "num_warps")) * 32
        )
        shared = operator.index(getattr(metadata, "shared"))
    except (AttributeError, TypeError, PreparedLaunchContractError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel does not expose num_warps/shared occupancy metadata"
        ) from exc
    if shared < 0:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel reports a negative shared-memory size"
        )
    return block_threads, shared


def _canonical_cluster_dims(value: object) -> tuple[int, int, int]:
    try:
        dimensions = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise PreparedLaunchContractError(
            "cluster_dims must contain exactly three positive integers"
        ) from exc
    if len(dimensions) != 3:
        raise PreparedLaunchContractError(
            "cluster_dims must contain exactly three positive integers"
        )
    return tuple(
        _positive_integer("cluster_dims", dimension) for dimension in dimensions
    )  # type: ignore[return-value]


def _admit_cluster_launch(
    spec: LaunchSpec,
    compiled: object,
    grid3: tuple[int, int, int],
    runtime: _RuntimeAdapters,
) -> ClusterLaunchAdmission | None:
    """Fail before first launch unless the compiled CTA cluster is supported."""
    declared_dims = spec.kwargs.get("cluster_dims")
    if declared_dims is None:
        return None
    cluster_dims = _canonical_cluster_dims(declared_dims)
    num_ctas = _positive_integer("num_ctas", spec.kwargs.get("num_ctas", 1))
    metadata = getattr(compiled, "metadata", None)
    if num_ctas != 1 or getattr(metadata, "num_ctas", None) != 1:
        raise UnsupportedPreparedRuntimeError(
            "explicit cluster_dims admission requires one CTA per program"
        )
    try:
        compiled_dims = _canonical_cluster_dims(getattr(metadata, "cluster_dims", ()))
    except PreparedLaunchContractError as exc:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel does not expose three-dimensional cluster metadata"
        ) from exc
    if compiled_dims != cluster_dims:
        raise UnsupportedPreparedRuntimeError(
            f"compiled cluster dimensions {compiled_dims} do not match "
            f"LaunchSpec {cluster_dims}"
        )
    block_threads, shared = _compiled_launch_resources(compiled)
    maximum = runtime.maximum_cluster_size(
        compiled,
        block_threads,
        shared,
        grid3,
        cluster_dims,
    )
    required = prod(cluster_dims)
    if required > maximum:
        raise UnsupportedPreparedRuntimeError(
            f"{spec.label} requires a {required}-CTA cluster, but the compiled "
            f"kernel and active device support at most {maximum}; select a "
            "smaller-cluster specialization"
        )
    return ClusterLaunchAdmission(
        cluster_dims=cluster_dims,
        required_cluster_size=required,
        maximum_cluster_size=maximum,
    )


def _admit_cooperative_grid(
    spec: LaunchSpec,
    compiled: object,
    grid3: tuple[int, int, int],
    runtime: _RuntimeAdapters,
    device: object,
) -> CooperativeGridAdmission | None:
    """Fail before first launch unless the complete cooperative grid fits."""
    if not spec.requires_full_grid_residency:
        return None
    metadata = getattr(compiled, "metadata", None)
    if getattr(metadata, "launch_cooperative_grid", None) is not True:
        raise UnsupportedPreparedRuntimeError(
            f"{spec.label} requires cooperative launch, but the compiled "
            "kernel does not enable it"
        )
    if not runtime.cooperative_launch_supported(device):
        raise UnsupportedPreparedRuntimeError(
            f"{spec.label} requires cooperative launch, but the device does not "
            "support it"
        )
    num_ctas = _positive_integer("num_ctas", spec.kwargs.get("num_ctas", 1))
    if num_ctas != 1 or getattr(metadata, "num_ctas", None) != 1:
        raise UnsupportedPreparedRuntimeError(
            "cooperative full-grid admission currently supports one CTA per "
            "program only"
        )
    if "cluster_dims" in spec.kwargs:
        raise UnsupportedPreparedRuntimeError(
            "cooperative full-grid admission does not accept clustered launches"
        )

    block_threads, shared = _compiled_launch_resources(compiled)

    properties = runtime.device_properties(device)
    try:
        multiprocessor_count = _positive_integer(
            "multiprocessor_count", properties["multiprocessor_count"]
        )
    except (KeyError, TypeError, PreparedLaunchContractError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "device properties do not expose multiprocessor_count"
        ) from exc
    active_blocks = runtime.active_blocks_per_sm(
        compiled,
        block_threads,
        shared,
    )
    resident_capacity = active_blocks * multiprocessor_count
    grid_ctas = prod(grid3)
    if grid_ctas > resident_capacity:
        raise UnsupportedPreparedRuntimeError(
            f"{spec.label} cooperative grid has {grid_ctas} CTAs, but the "
            f"compiled kernel admits only {active_blocks} CTA/SM across "
            f"{multiprocessor_count} SMs ({resident_capacity} resident CTAs); "
            "select a lower-grid specialization"
        )
    return CooperativeGridAdmission(
        grid_ctas=grid_ctas,
        active_blocks_per_sm=active_blocks,
        multiprocessor_count=multiprocessor_count,
        resident_cta_capacity=resident_capacity,
    )


def _global_bindings(jit_function: object) -> tuple[tuple[object, ...], ...]:
    bindings = []
    used = getattr(jit_function, "used_global_vals", {})
    for key, value_and_namespace in used.items():
        try:
            name = key[0]
            value, namespace = value_and_namespace
        except (IndexError, TypeError, ValueError) as exc:
            raise PreparedLaunchContractError(
                "unexpected JITFunction.used_global_vals layout"
            ) from exc
        bindings.append((name, value, namespace))
    return tuple(bindings)


@dataclass
class PreparedCompiledLauncher:
    """One compiled runner owned by one host thread and one CUDA stream."""

    _runner: Callable[..., object]
    _arguments: list[object]
    _dynamic_bound_indices: tuple[int, ...]
    _dynamic_alignments: tuple[int, ...]
    _static_bound_indices: tuple[int, ...]
    _static_bound_values: tuple[object, ...]
    _static_bound_alignments: tuple[int, ...]
    _static_bound_snapshots: tuple[StaticTensorBindingSnapshot, ...]
    _token_count: int
    _device: object
    _stream: object
    _current_device: Callable[[], object]
    _current_stream: Callable[[object], object]
    _owner_thread: int
    _global_bindings: tuple[tuple[object, ...], ...]
    cooperative_grid_admission: CooperativeGridAdmission | None
    cluster_launch_admission: ClusterLaunchAdmission | None
    kernel_name: str
    label: str

    @property
    def device(self) -> object:
        return self._device

    @property
    def stream(self) -> object:
        return self._stream

    @property
    def stream_handle(self) -> int:
        """Raw ordered-stream handle captured during preparation."""
        for candidate in (
            self._stream,
            getattr(self._stream, "cuda_stream", None),
            getattr(self._stream, "handle", None),
        ):
            if candidate is None or isinstance(candidate, bool):
                continue
            try:
                return int(operator.index(candidate))
            except TypeError:
                continue
        raise PreparedLaunchContractError(
            "prepared stream does not expose an integer handle"
        )

    @property
    def owner_thread_id(self) -> int:
        return self._owner_thread

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def dynamic_alignments(self) -> tuple[int, ...]:
        return self._dynamic_alignments

    @property
    def static_bound_indices(self) -> tuple[int, ...]:
        return self._static_bound_indices

    @property
    def static_bound_values(self) -> tuple[object, ...]:
        return self._static_bound_values

    @property
    def static_bound_alignments(self) -> tuple[int, ...]:
        return self._static_bound_alignments

    @property
    def static_alignments(self) -> tuple[int, ...]:
        """Alignment contract for fn/scale/base/norm static tensors."""
        return self._static_bound_alignments

    @property
    def static_bound_snapshots(self) -> tuple[StaticTensorBindingSnapshot, ...]:
        return self._static_bound_snapshots

    @property
    def dynamic_argument_count(self) -> int:
        return len(self._dynamic_bound_indices)

    def validate_environment(self) -> None:
        """Check owner, device, and globals outside a latency-critical region."""
        if threading.get_ident() != self._owner_thread:
            raise PreparedLaunchContractError(
                "prepared launcher is owned by a different host thread"
            )
        if self._current_device() != self._device:
            raise PreparedLaunchContractError(
                "current CUDA device differs from the prepared device"
            )
        if self._current_stream(self._device) != self._stream:
            raise PreparedLaunchContractError(
                "current CUDA stream differs from the prepared stream"
            )
        for name, expected, namespace in self._global_bindings:
            actual = namespace.get(name, _MISSING)
            if actual != expected:
                raise PreparedLaunchContractError(
                    f"JIT global {name!r} changed after preparation"
                )

    def validate_static_bindings(self, values: Sequence[object]) -> None:
        """Reject any static object/storage/pointer/layout rebind after prepare."""
        if len(values) != len(self._static_bound_snapshots):
            raise PreparedLaunchContractError(
                f"expected {len(self._static_bound_snapshots)} static values, "
                f"got {len(values)}"
            )
        for position, (value, expected) in enumerate(
            zip(values, self._static_bound_snapshots)
        ):
            actual = StaticTensorBindingSnapshot.capture(value)
            if actual != expected:
                raise PreparedLaunchContractError(
                    "static bound value changed after launcher preparation at "
                    f"position {position}; rebuild the prepared launcher"
                )

    def _replace_dynamic(self, values: Sequence[object]) -> None:
        if len(values) != len(self._dynamic_bound_indices):
            raise PreparedLaunchContractError(
                f"expected {len(self._dynamic_bound_indices)} dynamic values, "
                f"got {len(values)}"
            )
        for index, value in zip(self._dynamic_bound_indices, values):
            self._arguments[index] = value

    def _validate_dynamic_alignment(self, values: Sequence[object]) -> None:
        for position, (value, alignment) in enumerate(
            zip(values, self._dynamic_alignments)
        ):
            _validate_pointer_alignment(f"dynamic_values[{position}]", value, alignment)

    def run(self, dynamic_values: Sequence[object]) -> None:
        """Validate ownership/device, rebind pointers, and enqueue one kernel."""
        self.validate_environment()
        self.validate_static_bindings(self._static_bound_values)
        self._validate_dynamic_alignment(dynamic_values)
        self._replace_dynamic(dynamic_values)
        self._runner(*self._arguments, stream=self._stream)

    def run_trusted(self, dynamic_values: Sequence[object]) -> None:
        """Lowest-overhead path for an already-validated owning context.

        The caller must guarantee the same host owner, device, stream contract,
        immutable JIT globals, tensor ABI, and pointer-alignment specialization.
        """
        try:
            self._replace_dynamic(dynamic_values)
            self._runner(*self._arguments, stream=self._stream)
        finally:
            # The direct runner has already marshalled the launch arguments.
            # Do not make a long-lived context retain the previous invocation's
            # activations and outputs (which can exceed 1 GiB for large T).
            # Clearing in ``finally`` also releases partially rebound values if
            # argument replacement or launch raises.
            for index in self._dynamic_bound_indices:
                self._arguments[index] = None


def prepare_launch_spec(
    spec: LaunchSpec,
    *,
    _runtime: _RuntimeAdapters | None = None,
) -> PreparedCompiledLauncher:
    """Compile/load ``spec`` without launching and retain its direct runner."""
    if not isinstance(spec, LaunchSpec):
        raise TypeError("spec must be a LaunchSpec")
    if _lite_dist_requested():
        raise UnsupportedPreparedRuntimeError(
            "direct prepared launch is unsupported with FLAGTREE_LITE_DIST"
        )
    hooks = tuple(getattr(spec.jit_function, "pre_run_hooks", ()))
    if hooks:
        raise UnsupportedPreparedRuntimeError(
            "direct prepared launch would bypass JIT pre-run hooks"
        )

    runtime = _runtime or _default_runtime_adapters()
    device = runtime.current_device()
    stream = runtime.current_stream(device)

    # KernelInterface.warmup calls JITFunction.run(warmup=True): compilation,
    # loading, hooks, and global validation happen, but no kernel is launched.
    compiled = spec.jit_function.warmup(
        *spec.args,
        grid=spec.grid,
        **dict(spec.kwargs),
    )
    compiled = _resolve_compiled(compiled)

    try:
        binder = spec.jit_function.device_caches[device][4]
        bound_args, specialization, _ = binder(*spec.args, **dict(spec.kwargs))
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "FlagTree JIT binder ABI is not compatible with prepared launch"
        ) from exc
    if not isinstance(bound_args, Mapping):
        raise PreparedLaunchContractError("JIT binder did not return a mapping")
    bound_values = tuple(bound_args.values())
    if len(specialization) != len(bound_values):
        raise PreparedLaunchContractError(
            "JIT specialization length does not match bound arguments"
        )

    for index, expected in zip(spec.dynamic_bound_indices, spec.dynamic_values):
        if index >= len(bound_values):
            raise PreparedLaunchContractError(
                f"dynamic bound index {index} exceeds ABI length "
                f"{len(bound_values)}"
            )
        if bound_values[index] is not expected:
            raise PreparedLaunchContractError(
                f"dynamic bound index {index} does not identify its declared value"
            )
        if specialization[index][0] == "constexpr":
            raise PreparedLaunchContractError(
                f"dynamic bound index {index} names a constexpr argument"
            )
    for position, (value, alignment) in enumerate(
        zip(spec.dynamic_values, spec.dynamic_alignments)
    ):
        _validate_pointer_alignment(f"dynamic_values[{position}]", value, alignment)

    for position, (index, expected, alignment) in enumerate(
        zip(
            spec.static_bound_indices,
            spec.static_bound_values,
            spec.static_bound_alignments,
        )
    ):
        if index >= len(bound_values):
            raise PreparedLaunchContractError(
                f"static bound index {index} exceeds ABI length {len(bound_values)}"
            )
        if bound_values[index] is not expected:
            raise PreparedLaunchContractError(
                f"static bound index {index} does not identify its declared value"
            )
        if specialization[index][0] == "constexpr":
            raise PreparedLaunchContractError(
                f"static bound index {index} names a constexpr argument"
            )
        _validate_pointer_alignment(
            f"static_bound_values[{position}]", expected, alignment
        )

    metadata = getattr(compiled, "metadata", None)
    if getattr(metadata, "tle_node_buffer_bindings", ""):
        raise UnsupportedPreparedRuntimeError(
            "direct prepared launch is unsupported for TLE node bindings"
        )
    runtime.validate_node_launch_bindings(compiled, bound_args, specialization)
    grid3 = _canonical_grid(spec.grid, bound_args)
    cooperative_grid_admission = _admit_cooperative_grid(
        spec,
        compiled,
        grid3,
        runtime,
        device,
    )
    cluster_launch_admission = _admit_cluster_launch(
        spec,
        compiled,
        grid3,
        runtime,
    )
    try:
        runner = compiled[grid3]
    except (AttributeError, TypeError) as exc:
        raise UnsupportedPreparedRuntimeError(
            "compiled kernel does not expose a direct grid runner"
        ) from exc

    static_bound_snapshots = tuple(
        StaticTensorBindingSnapshot.capture(value) for value in spec.static_bound_values
    )
    arguments = list(bound_values)
    for index in spec.dynamic_bound_indices:
        arguments[index] = None
    return PreparedCompiledLauncher(
        _runner=runner,
        _arguments=arguments,
        _dynamic_bound_indices=spec.dynamic_bound_indices,
        _dynamic_alignments=spec.dynamic_alignments,
        _static_bound_indices=spec.static_bound_indices,
        _static_bound_values=spec.static_bound_values,
        _static_bound_alignments=spec.static_bound_alignments,
        _static_bound_snapshots=static_bound_snapshots,
        _token_count=spec.token_count,
        _device=device,
        _stream=stream,
        _current_device=runtime.current_device,
        _current_stream=runtime.current_stream,
        _owner_thread=threading.get_ident(),
        _global_bindings=_global_bindings(spec.jit_function),
        cooperative_grid_admission=cooperative_grid_admission,
        cluster_launch_admission=cluster_launch_admission,
        kernel_name=str(getattr(compiled, "name", type(compiled).__name__)),
        label=spec.label,
    )


__all__ = [
    "ClusterLaunchAdmission",
    "CooperativeGridAdmission",
    "LaunchSpec",
    "PreparedCompiledLauncher",
    "PreparedLaunchContractError",
    "PreparedLauncherError",
    "StaticTensorBindingSnapshot",
    "UnsupportedPreparedRuntimeError",
    "prepare_launch_spec",
]
