"""Explicit LaunchSpec adapters for admitted Hopper MegaMHC Plans.

The kernel families predate :mod:`prepared_launcher` and expose dataclass
``Plan`` objects whose ``run`` methods spell out the JIT launch.  Repeating
that launch ABI in every kernel file makes review difficult, while inferring
it by monkeypatching ``JITFunction.run`` is not production safe.  This module
therefore keeps one exact, fail-closed registry for the Plan classes selected
by :mod:`dispatch`.

Only exact class/module pairs in ``ADMITTED_PLAN_ADAPTERS`` are accepted.
Each adapter names the JIT function, grid, compile options and argument owner
explicitly.  The eight address-rebindable activation/output slots are also
verified against the Plan structure before a :class:`LaunchSpec` is returned.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Mapping

from .prepared_launcher import LaunchSpec, PreparedLaunchContractError

_DYNAMIC_BOUND_INDICES = (0, 1, 2, 3, 8, 9, 10, 11)
_STATIC_BOUND_INDICES = (4, 5, 6, 7)
_DIRECT_POINTER_ALIGNMENT = 16
_DYNAMIC_ALIGNMENTS = (_DIRECT_POINTER_ALIGNMENT,) * 8
_STATIC_ALIGNMENTS = (_DIRECT_POINTER_ALIGNMENT,) * 4
_DYNAMIC_FIELD_NAMES = (
    "x",
    "residual",
    "post_mix",
    "comb_res_mix",
    "new_residual",
    "new_post_mix",
    "new_comb_res_mix",
    "y_bf16",
)
_KERNEL_TENSOR_FIELD_NAMES = (
    "x",
    "residual",
    "post_mix",
    "comb_res_mix",
    "fn",
    "mix_scales",
    "mix_bases",
    "rmsnorm_weight",
    "new_residual",
    "new_post_mix",
    "new_comb_res_mix",
    "y_bf16",
)
_NESTED_FIELDS = ("implementation", "base")


@dataclass(frozen=True)
class PlanAdapter:
    """Exact host ABI for one admitted Plan class."""

    kind: str
    kernel_attr: str | None = None
    grid_attr: str = "GRID_CTAS"
    warps_attr: str = "NUM_WARPS"
    argument_path: tuple[str, ...] = ()
    constants_path: tuple[str, ...] = ()
    dynamic_alignments: tuple[int, ...] = _DYNAMIC_ALIGNMENTS
    static_alignments: tuple[int, ...] = _STATIC_ALIGNMENTS


_PREFIX = __package__


ADMITTED_PLAN_ADAPTERS: Mapping[str, PlanAdapter] = {
    # Direct-field Graph Plans.
    f"{_PREFIX}.kernels.sm90_small_family.TLESM90MegaMHCPlan": PlanAdapter("small"),
    f"{_PREFIX}.kernels.sm90_padded_m64_wgmma.PaddedM64WGMMAPlan": PlanAdapter(
        "persistent_wgmma",
        "_t32_wgmma_bm64_split128_tf32_kernel",
    ),
    f"{_PREFIX}.fused_post_gemm_t64.T64BM64DotPlan": PlanAdapter(
        "persistent_wgmma",
        "_t64_bm64_split64_tf32_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_padded_m128_tile_split64_wgmma.PaddedM128TileSplit64Plan": PlanAdapter(
        "cooperative",
        "_sm90_padded_m128_tile_split64_wgmma_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_mid_grid_pipe4_role2_ptx_sinkhorn_wgmma.MidGridPipe4Role2Plan": PlanAdapter(
        "cooperative",
        "_sm90_mid_grid_pipe4_role2_wgmma_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_large_graph_role4_k32_wgmma.LargeGraphRole4Plan": PlanAdapter(
        "cooperative",
        "_sm90_large_graph_role4_k32_wgmma_kernel",
        warps_attr="ROOT_WARPS",
    ),
    # ``base`` is a direct-field Plan; this class owns the selected kernel.
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_wgmma.LargeGraphSplit16RSPlan": PlanAdapter(
        "cooperative",
        "_sm90_large_graph_split16_rs_wgmma_kernel",
        warps_attr="ROOT_WARPS",
    ),
    # Shape admission wrappers which deliberately delegate the exact launch.
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t416_nonstriped_wgmma.LargeGraphSplit16RST416Plan": PlanAdapter(
        "delegate",
        argument_path=("implementation",),
    ),
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t448_wgmma.LargeGraphSplit16RST448Plan": PlanAdapter(
        "delegate",
        argument_path=("implementation",),
    ),
    # Nested wrappers which own a different exact JIT kernel.
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t400_nonstriped_wgmma.LargeGraphSplit16RST400Plan": PlanAdapter(
        "cooperative",
        "_t400_allproducer_kernel",
        warps_attr="ROOT_WARPS",
        argument_path=("implementation",),
        constants_path=("_implementation",),
    ),
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t464_nonstriped_wgmma.LargeGraphSplit16RST464Plan": PlanAdapter(
        "cooperative",
        "_t464_allproducer_kernel",
        warps_attr="ROOT_WARPS",
        argument_path=("implementation",),
        constants_path=("_implementation",),
    ),
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t496_nonstriped_wgmma.LargeGraphSplit16RST496Plan": PlanAdapter(
        "cooperative",
        "_t496_allproducer_kernel",
        warps_attr="ROOT_WARPS",
        argument_path=("implementation",),
        constants_path=("_implementation",),
    ),
    f"{_PREFIX}.kernels.sm90_large_graph_split16_rs_t512_wgmma.LargeGraphSplit16RST512Plan": PlanAdapter(
        "cooperative",
        "_t512_allproducer_kernel",
        warps_attr="ROOT_WARPS",
        argument_path=("implementation",),
        constants_path=("_implementation",),
    ),
    # Tuple/workspace eager Plans.
    f"{_PREFIX}.kernels.sm90_eager_t1028_p256_manual512_cpasync.EagerT1028P256Manual512ProjectionPlan": PlanAdapter(
        "eager",
        "_sm90_eager_t1028_p256_manual512_cpasync_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_eager_t2048_p256_manual512_cpasync.EagerT2048P256Manual512ProjectionPlan": PlanAdapter(
        "eager",
        "_sm90_eager_t2048_p256_manual512_cpasync_kernel",
    ),
    # T=3805/4100 alias this class after mutating the pinned fresh-process
    # constants, so one exact adapter covers all three admitted buckets.
    f"{_PREFIX}.kernels.sm90_eager_t3076_p256_cluster16_manual512_cpasync.EagerT3076P256Cluster16Manual512ProjectionPlan": PlanAdapter(
        "cluster_eager",
        "_sm90_eager_t3076_p256_cluster16_manual512_cpasync_kernel",
    ),
    # Tuple + Fn TensorMap storage eager Plans.
    f"{_PREFIX}.kernels.sm90_eager_t12991_p384_rs_wgmma.EagerT12991P384Plan": PlanAdapter(
        "p384_tma",
        "_sm90_eager_t12991_p384_rs_wgmma_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_eager_t13005_p384_rs_wgmma.EagerT13005P384Plan": PlanAdapter(
        "p384_tma",
        "_sm90_eager_t13005_p384_rs_wgmma_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_eager_t13021_p384_rs_wgmma.EagerT13021P384Plan": PlanAdapter(
        "p384_tma",
        "_sm90_eager_t13021_p384_rs_wgmma_kernel",
    ),
    f"{_PREFIX}.kernels.sm90_eager_t16384_p384_rs_wgmma.EagerT16384P384Plan": PlanAdapter(
        "p384_tma",
        "_sm90_eager_t16384_p384_rs_wgmma_kernel",
    ),
}


def _qualified_type(plan: object) -> str:
    plan_type = type(plan)
    return f"{plan_type.__module__}.{plan_type.__qualname__}"


def _resolve_path(root: object, path: tuple[str, ...], label: str) -> object:
    value = root
    for name in path:
        try:
            value = getattr(value, name)
        except AttributeError as exc:
            raise PreparedLaunchContractError(
                f"{label} has no declared {name!r} component"
            ) from exc
    return value


def _module_for(plan: object) -> ModuleType:
    try:
        return importlib.import_module(type(plan).__module__)
    except (ImportError, TypeError) as exc:
        raise PreparedLaunchContractError(
            f"cannot import Plan module {type(plan).__module__!r}"
        ) from exc


def _leaf_plan(plan: object) -> object:
    for name in _NESTED_FIELDS:
        nested = getattr(plan, name, None)
        if nested is not None:
            return _leaf_plan(nested)
    return plan


def _declared_dynamic_values(plan: object) -> tuple[object, ...]:
    leaf = _leaf_plan(plan)
    tensors = getattr(leaf, "tensors", None)
    if tensors is not None:
        values = tuple(tensors)
        if len(values) < 12:
            raise PreparedLaunchContractError(
                f"{type(leaf).__name__}.tensors exposes fewer than 12 ABI slots"
            )
        return (*values[:4], *values[8:12])
    if all(hasattr(leaf, name) for name in _DYNAMIC_FIELD_NAMES):
        return tuple(getattr(leaf, name) for name in _DYNAMIC_FIELD_NAMES)
    raise PreparedLaunchContractError(
        f"cannot resolve explicit dynamic values from {type(leaf).__name__}"
    )


def _declared_static_values(plan: object) -> tuple[object, ...]:
    leaf = _leaf_plan(plan)
    tensors = getattr(leaf, "tensors", None)
    if tensors is not None:
        values = tuple(tensors)
        if len(values) < 8:
            raise PreparedLaunchContractError(
                f"{type(leaf).__name__}.tensors exposes fewer than 8 ABI slots"
            )
        return values[4:8]
    if all(hasattr(leaf, name) for name in _KERNEL_TENSOR_FIELD_NAMES[4:8]):
        return tuple(getattr(leaf, name) for name in _KERNEL_TENSOR_FIELD_NAMES[4:8])
    raise PreparedLaunchContractError(
        f"cannot resolve explicit static values from {type(leaf).__name__}"
    )


def _token_count(dynamic_values: tuple[object, ...]) -> int:
    shape = getattr(dynamic_values[0], "shape", None)
    if shape is None or not shape:
        raise PreparedLaunchContractError("Plan x tensor must expose a non-empty shape")
    try:
        tokens = int(shape[0])
    except (TypeError, ValueError) as exc:
        raise PreparedLaunchContractError(
            "Plan token count must be an integer"
        ) from exc
    if tokens <= 0:
        raise PreparedLaunchContractError("Plan token count must be positive")
    return tokens


def _kernel_args(owner: object) -> tuple[object, ...]:
    method = getattr(owner, "_kernel_args", None)
    if not callable(method):
        raise PreparedLaunchContractError(
            f"{type(owner).__name__} does not expose _kernel_args()"
        )
    arguments = tuple(method())
    if len(arguments) <= max(_DYNAMIC_BOUND_INDICES):
        raise PreparedLaunchContractError("kernel ABI exposes fewer than 12 slots")
    return arguments


def _check_dynamic_abi(
    plan: object, arguments: tuple[object, ...]
) -> tuple[object, ...]:
    declared = _declared_dynamic_values(plan)
    actual = tuple(arguments[index] for index in _DYNAMIC_BOUND_INDICES)
    if any(value is not expected for value, expected in zip(actual, declared)):
        raise PreparedLaunchContractError(
            "Plan kernel ABI does not place dynamic tensors in slots " "0:4 and 8:12"
        )
    return declared


def _check_static_abi(
    plan: object, arguments: tuple[object, ...]
) -> tuple[object, ...]:
    declared = _declared_static_values(plan)
    actual = tuple(arguments[index] for index in _STATIC_BOUND_INDICES)
    if any(value is not expected for value, expected in zip(actual, declared)):
        raise PreparedLaunchContractError(
            "Plan kernel ABI does not place static tensors in slots 4:8"
        )
    return declared


def _make_launch_spec(
    plan: object,
    adapter: PlanAdapter,
    *,
    kernel: object,
    grid: object,
    arguments: tuple[object, ...],
    kwargs: Mapping[str, object],
    label: str,
) -> LaunchSpec:
    """Build identical explicit ABI metadata for every admitted Plan family."""
    dynamic = _check_dynamic_abi(plan, arguments)
    static = _check_static_abi(plan, arguments)
    return LaunchSpec(
        jit_function=kernel,
        grid=grid,
        args=arguments,
        kwargs=kwargs,
        token_count=_token_count(dynamic),
        dynamic_bound_indices=_DYNAMIC_BOUND_INDICES,
        dynamic_values=dynamic,
        dynamic_alignments=adapter.dynamic_alignments,
        static_bound_indices=_STATIC_BOUND_INDICES,
        static_bound_values=static,
        static_bound_alignments=adapter.static_alignments,
        label=label,
    )


def _base_kwargs(warps: int) -> dict[str, object]:
    return {"num_warps": warps, "num_stages": 1, "num_ctas": 1}


def _persistent_wgmma_kwargs(plan: object, warps: int) -> dict[str, object]:
    """Return the single admitted M64 production specialization."""
    kwargs = _base_kwargs(warps)
    kwargs["launch_cooperative_grid"] = True
    return kwargs


def _small_spec(
    plan: object, module: ModuleType, adapter: PlanAdapter, label: str
) -> LaunchSpec:
    scalars = (
        plan.hc_norm_eps,
        plan.hc_pre_eps,
        plan.hc_post_scale,
        plan.sinkhorn_eps,
        plan.rmsnorm_eps,
        plan.rmsnorm_scale,
    )
    direct = tuple(getattr(plan, name) for name in _KERNEL_TENSOR_FIELD_NAMES)
    if plan.schedule.startswith("split") and "_tile" in plan.schedule:
        kernel = getattr(module, "_tle_sm90_mega_mhc_t1_split_kernel")
        args = (
            *direct,
            plan.projection_partials,
            plan.sqsum_partials,
            plan.arrival_counter,
            *scalars,
        )
        kwargs = _base_kwargs(plan.split_num_warps)
        kwargs.update(
            {
                "OUTPUTS_PER_TILE": plan.outputs_per_tile,
                "OUTPUT_TILE_BLOCK": plan.output_tile_block,
                "NUM_OUTPUT_TILES": plan.num_output_tiles,
                "TOKENS": plan.tokens,
                "NUM_SPLITS": plan.num_splits,
                "HIDDEN_PER_SPLIT": plan.hidden_per_split,
                "CTAS_PER_TOKEN": plan.ctas_per_token,
                "TOTAL_CTAS": plan.grid_ctas,
                "COUNTER_STRIDE_ELEMENTS": plan.counter_stride_elements,
                "CTA_THREADS_PER_BLOCK": plan.split_num_warps * 32,
                "USE_HIERARCHICAL_REDUCTION": plan.use_hierarchical_reduction,
                "USE_TID4_REDUCTION": plan.use_tid4_reduction,
                "USE_TID5_REDUCTION": plan.use_tid5_reduction,
                "USE_TID7_REDUCTION": plan.use_tid7_reduction,
                "USE_WARP_OVERLAP": plan.use_warp_overlap,
                "USE_CONTIGUOUS_PARTIALS": plan.use_contiguous_partials,
                "USE_T2_PROJECTION_ILP": plan.use_t2_projection_ilp,
            }
        )
        grid = (plan.grid_ctas,)
    else:
        raise PreparedLaunchContractError(
            f"{label} is not an admitted small-family split-tile schedule"
        )
    return _make_launch_spec(
        plan,
        adapter,
        kernel=kernel,
        grid=grid,
        arguments=args,
        kwargs=kwargs,
        label=label,
    )


def launch_spec_for_plan(plan: object) -> LaunchSpec:
    """Return the exact launch contract for one dispatch-admitted Plan.

    Unknown classes and unsupported configurations raise before JIT
    compilation. No Plan ``run`` method is invoked.
    """
    key = _qualified_type(plan)
    try:
        adapter = ADMITTED_PLAN_ADAPTERS[key]
    except KeyError as exc:
        raise PreparedLaunchContractError(
            f"no explicit LaunchSpec adapter for {key}"
        ) from exc
    if adapter.kind == "delegate":
        delegated = _resolve_path(plan, adapter.argument_path, key)
        return launch_spec_for_plan(delegated)

    module = _module_for(plan)
    if adapter.kind == "small":
        return _small_spec(plan, module, adapter, key)
    if adapter.kernel_attr is None:
        raise PreparedLaunchContractError(f"{key} does not declare a kernel")
    kernel = _resolve_path(module, (adapter.kernel_attr,), key)
    owner = _resolve_path(plan, adapter.argument_path, key)
    constants = _resolve_path(module, adapter.constants_path, key)
    arguments = _kernel_args(owner)
    try:
        grid_value = int(getattr(constants, adapter.grid_attr))
        warps = int(getattr(constants, adapter.warps_attr))
    except (AttributeError, TypeError, ValueError) as exc:
        raise PreparedLaunchContractError(
            f"{key} does not expose its declared grid/warp constants"
        ) from exc
    kwargs = _base_kwargs(warps)
    grid = (grid_value,)

    if adapter.kind == "cooperative":
        kwargs["launch_cooperative_grid"] = True
    elif adapter.kind == "persistent_wgmma":
        kwargs = _persistent_wgmma_kwargs(plan, warps)
    elif adapter.kind in ("eager", "p384_tma"):
        kwargs["maxnreg"] = int(plan.maxnreg)
        if adapter.kind == "p384_tma" and (
            module.ACTIVE_K64_BLOCKS != module.NUM_K64_BLOCKS
            or module.TAIL_BATCHES != 22
            or module.TAIL_WORK_PHASE != 3
        ):
            raise PreparedLaunchContractError(
                f"{key} does not match the admitted P384 production configuration"
            )
    elif adapter.kind == "cluster_eager":
        kwargs["maxnreg"] = int(plan.maxnreg)
        if not module.USE_GLOBAL_TILE_BARRIER:
            grid = (int(module.NUM_M_TILES),)
            kwargs["cluster_dims"] = (int(module.CLUSTER_CTAS), 1, 1)
    else:
        raise PreparedLaunchContractError(
            f"unknown LaunchSpec adapter kind {adapter.kind!r}"
        )

    return _make_launch_spec(
        plan,
        adapter,
        kernel=kernel,
        grid=grid,
        arguments=arguments,
        kwargs=kwargs,
        label=key,
    )


__all__ = [
    "ADMITTED_PLAN_ADAPTERS",
    "PlanAdapter",
    "launch_spec_for_plan",
]
