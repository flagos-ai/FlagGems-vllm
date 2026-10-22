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

import itertools
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple

import torch

import flaggems_vllm

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32]
BOOL_DTYPES = [torch.bool]
COMPLEX_DTYPES = [torch.complex64]
EXTRA_INT_DTYPES = [torch.int8, torch.uint8, torch.int64]


def get_fp8_dtype():
    # MThreads exposes FP8 dtypes through the torch namespace, but does not
    # implement CUDA's device-capability query.
    if flaggems_vllm.device == "musa":
        for name in ("float8_e4m3fn", "float8_e5m2"):
            if hasattr(torch, name):
                return getattr(torch, name)
        return None

    if flaggems_vllm.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()

    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn

    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2

    return None


_fp8_dtype = get_fp8_dtype()
FP8_DTYPES = [_fp8_dtype] if _fp8_dtype is not None else []

DEFAULT_WARMUP_TIME = 1000
DEFAULT_ITER_TIME = 100

# LEGACY_SHAPES are maintained for legacy benchmark SIZE settings and may be removed in the future.
# Do not reference this elsewhere.
LEGACY_SHAPES = [i * 64 for i in range(1, 22, 5)]
LEGACY_NON_BLAS_SHAPES = [(1024, shape) for shape in LEGACY_SHAPES]
LEGACY_BLAS_SHAPES = [(16, shape, shape, shape) for shape in LEGACY_SHAPES]

# Default shapes settings
DEFAULT_SHAPES = [
    (1024 * 1024 * 1024,),  # from perf
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),  # from perf
]


def model_shapes():
    # batch sizes * seq lengths
    BS = [1, 2, 3, 4, 8, 98, 256, 8192]
    # attn: wqkv, wo; ffn: w13, w2
    NK = [
        # extract from llama3-8b
        (1024, 4096),
        (128256, 4096),
        (14336, 4096),
        (4096, 14336),
        (4096, 4096),
        (6144, 4096),
        (28672, 4096),
        # extract from qwen2.5-7b
        (3584, 3584),
        (18944, 3584),
        (3584, 18944),
        (152064, 3584),
        (37888, 3584),
        (512, 3584),
        (4608, 3584),
    ]

    return [(4, bs, n, k) for bs, (n, k) in itertools.product(BS, NK)]


@dataclass
class BenchmarkMetrics:
    # Legacy shape information for backward compatibility
    # This field corresponds to the 'size' field in the previous version's benchmark.
    legacy_shape: Optional[int] = None
    # Detailed size info
    shape_detail: Optional[Tuple[int, ...]] = None
    # Latency base in ms
    latency_base: Optional[float] = None
    # Latency in ms
    latency: Optional[float] = None
    gbps_base: Optional[float] = None
    gbps: Optional[float] = None
    # Speedup over baseline
    speedup: Optional[float] = None
    # Accuracy over baseline (not implemented yet)
    accuracy: Optional[float] = None
    # TFLOPS (not implemented yet)
    tflops: Optional[float] = None
    # Utilization (not implemented yet)
    utilization: Optional[float] = None
    # Speedup compared to base data
    compared_speedup: Optional[float] = None
    # Why the baseline comparison was skipped (op/shape/dtype missing).
    # None means "matched" or "--base-data not enabled". Excluded from metrics.
    base_miss_reason: Optional[str] = None
    # Error message
    error_msg: Optional[str] = None


ALL_AVAILABLE_METRICS = set(map(lambda x: x.name, fields(BenchmarkMetrics))) - {
    "legacy_shape",
    "shape_detail",
    "base_miss_reason",
}

DEFAULT_METRICS = [
    metric
    for metric in ["latency_base", "latency", "speedup"]
    if metric in ALL_AVAILABLE_METRICS
]


def check_metric_dependencies(
    requested_metrics: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Checks if the requested metrics satisfy their dependencies.
    Returns True if the dependencies are satisfied, otherwise False.
    """
    # Predefined dependencies between metrics
    buildin_dependencies = {
        "speedup": ["latency", "latency_base"],
        "utilization": ["latency", "tflops"],
    }
    unsatisfied_metrics = []
    if requested_metrics is None:
        return unsatisfied_metrics

    satisfied_metrics = set()
    for metric in requested_metrics:
        if metric not in buildin_dependencies:
            # If the metric has no dependencies, it's automatically satisfied
            satisfied_metrics.add(metric)
        else:
            required_metrics = buildin_dependencies[metric]
            # Check if all dependencies are in the satisfied metrics list
            if not all(req in satisfied_metrics for req in required_metrics):
                unsatisfied_metrics.append(metric)
            else:
                satisfied_metrics.add(metric)
    return unsatisfied_metrics


def get_recommended_shapes(
    op_name: str, op_specified_shapes: Optional[List[Tuple[int, ...]]]
):
    def _shapes_sort(shapes):
        shapes = [shape if isinstance(shape, tuple) else (shape,) for shape in shapes]
        return sorted(shapes, key=lambda x: torch.tensor(x).prod().item())

    if op_specified_shapes:
        # TODO: handle situation that list as the basic element in shape.
        return _shapes_sort(op_specified_shapes)
    return _shapes_sort(DEFAULT_SHAPES)


class BenchMode(Enum):
    KERNEL = "kernel"
    OPERATOR = "operator"
    WRAPPER = "wrapper"
    CUDAGRAPH = "cudagraph"


class BenchLevel(Enum):
    COMPREHENSIVE = "comprehensive"
    CORE = "core"


@dataclass
class OperationAttribute:
    op_name: str
    # Recommended core benchmark shapes for the given operation
    recommended_core_shapes: List[Tuple[int, ...]]
    shape_desc: str

    def __str__(self) -> str:
        return (
            f"{'Operator name':<40} |  {self.op_name}\n"
            f"{'Recommended Core Shapes[' + self.shape_desc + ']':<40} |  {self.recommended_core_shapes}\n"
        )

    def to_dict(self) -> dict:
        return self.__dict__


def custom_json_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ---- NVIDIA baseline lookup ("vs Base" column) --------------------------------
# Shared by both benchmark stacks (base.py and performance_utils.py). Lives here
# because consts.py is the lowest layer both import, so there's no import cycle.

# torch dtype string -> scaling-factor dtype key (see _scaling_factors block).
_DTYPE_TO_SF_KEY = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float64": "fp64",
    "torch.int8": "int8",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
}

# baseline bottleneck unit -> scaling-factor resource group. mem is
# dtype-independent (memory bandwidth); tensor/cuda pick a per-dtype compute
# factor (Tensor Core vs CUDA Core/vector).
_BOTTLENECK_TO_SF = {"mem": "bandwidth_gbps", "tensor": "tensor", "cuda": "vector"}


def _first_tensor_shape(shape_detail):
    """Return the first tensor shape found in a record_shapes() result as a
    plain list of ints, or None if there is no tensor shape.

    record_shapes() encodes tensors as torch.Size and may nest them inside
    lists/tuples/dicts (and wrap args+kwargs in a top-level tuple), so walk the
    structure depth-first and return the first torch.Size we hit.
    """
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            return list(item)
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return None


def _all_tensor_shapes_from_record(shape_detail):
    """从 record_shapes() 结果里收集**所有**张量 shape，返回排序后的元组。

    与 _first_tensor_shape 不同，这里不只取第一个张量，而是深度遍历整个结构
    （args+kwargs 嵌套的 list/tuple/dict），把每个 torch.Size 收进来。张量
    shape 的「多重集合」在同一算子内几乎总能唯一区分一个采集点，用它做签名
    就能绕开「baseline 的 key_shape 是语义复合键、与第一个张量 shape 不一致」
    导致的匹配失败（见下方 lookup_base_record_ex 的签名回退）。
    """
    shapes = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            shapes.append(tuple(item))
        elif isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
    return tuple(sorted(shapes))


def _all_scalars_from_record(shape_detail):
    """收集 record_shapes() 结果里的标量（int/float/str/bool），排序后返回。

    仅用于「张量 shape 相同、靠标量区分」的少数算子（如 per_token_group_quant_fp8
    的 use_ue8m0、persistent_topk 的 max_seq_len）做二次判别。torch.Size 已在
    上面单独处理，这里只收散落的标量。bool 要先于 int 判断（isinstance(True,int)）。
    """
    scalars = []
    stack = [shape_detail]
    while stack:
        item = stack.pop(0)
        if isinstance(item, torch.Size):
            continue
        if isinstance(item, dict):
            stack[:0] = list(item.values())
        elif isinstance(item, (list, tuple)):
            stack[:0] = list(item)
        elif isinstance(item, bool):
            scalars.append(repr(item))
        elif isinstance(item, (int, float, str)):
            scalars.append(repr(item))
    return tuple(sorted(scalars))


def _base_shapes_from_config(cfg):
    """从 baseline 记录的 config.inputs 里收集所有张量 shape（排序元组）。

    与 _all_tensor_shapes_from_record 对称：采集器把每个输入张量写成
    {"shape": [...]}，标量写成 {"scalar": v}。缺 config/inputs 时返回 None，
    表示该采集点无法用签名匹配（只能靠原始 key）。
    """
    if not isinstance(cfg, dict):
        return None
    inputs = cfg.get("inputs")
    if not isinstance(inputs, dict):
        return None
    shapes = []
    for v in inputs.values():
        if isinstance(v, dict) and isinstance(v.get("shape"), list):
            shapes.append(tuple(v["shape"]))
    return tuple(sorted(shapes))


def _base_scalars_from_config(cfg):
    """从 baseline config.inputs 里收集标量值（排序元组），与记录侧对称。"""
    scalars = []
    inputs = (cfg or {}).get("inputs", {})
    if isinstance(inputs, dict):
        for v in inputs.values():
            if isinstance(v, dict) and "scalar" in v:
                scalars.append(repr(v["scalar"]))
    return tuple(sorted(scalars))


# 每个 (base_data, op_name) 的「张量签名 -> shape key」索引，按 base_data 的 id
# 缓存，避免每个采集点都重建。值为 {dtype: {tensor_sig: [shape_key,...]}}。
_SIG_INDEX_CACHE = {}


def _resolve_op_entry(base_data, op_name, variant=None):
    """Return the baseline entry for ``op_name``.

    A kanban operator may merge several weight precisions under one name (e.g.
    fused_marlin_moe_w4a16 = INT4 + MXFP4). Their shapes and tensor signatures
    are identical, so the entry keeps one sub-entry per precision under
    ``variants`` and the benchmark selects it with ``variant``.
    """
    op_entry = base_data.get(op_name)
    if variant is not None and isinstance(op_entry, dict) and "variants" in op_entry:
        return op_entry["variants"].get(variant)
    return op_entry


def _build_signature_index(base_data, op_name, variant=None):
    """为某算子构建 {dtype: {tensor_sig: [shape_key,...]}} 索引（带缓存）。"""
    cache_key = (id(base_data), op_name, variant)
    cached = _SIG_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    index = {}
    op_entry = _resolve_op_entry(base_data, op_name, variant) or {}
    for shape_key, per_dtype in op_entry.get("shapes", {}).items():
        if not isinstance(per_dtype, dict):
            continue
        for dtype_str, rec in per_dtype.items():
            if not isinstance(rec, dict):
                continue
            sig = _base_shapes_from_config(rec.get("config"))
            if not sig:
                continue
            index.setdefault(dtype_str, {}).setdefault(sig, []).append(shape_key)
    _SIG_INDEX_CACHE[cache_key] = index
    return index


def _match_by_signature(base_data, op_name, dtype, shape_detail, variant=None):
    """张量签名回退匹配：返回 (record, matched_shape_key) 或 (None, None)。

    直接用 str(first_shape) 匹配失败时调用。按「所有张量 shape 的多重集合」在
    该 (op, dtype) 下查找；若多个 shape key 的张量签名相同（极少数算子），再用
    标量集合做二次判别。仍无法唯一定位则放弃（返回 None），由调用方按原逻辑报
    未匹配，绝不臆测。
    """
    index = _build_signature_index(base_data, op_name, variant)
    per_dtype = index.get(str(dtype))
    if not per_dtype:
        return None, None
    sig = _all_tensor_shapes_from_record(shape_detail)
    keys = per_dtype.get(sig)
    if not keys:
        return None, None
    op_shapes = _resolve_op_entry(base_data, op_name, variant)["shapes"]
    if len(keys) == 1:
        return op_shapes[keys[0]].get(str(dtype)), keys[0]
    # 张量签名碰撞：用标量集合二次判别。
    rec_scalars = _all_scalars_from_record(shape_detail)
    for k in keys:
        rec = op_shapes[k].get(str(dtype))
        if rec is None:
            continue
        base_scalars = _base_scalars_from_config(rec.get("config"))
        # baseline 标量应为记录侧标量的子集（记录侧常含 stride 等额外散落 int）。
        if base_scalars and set(base_scalars) <= set(rec_scalars):
            return rec, k
    return None, None


def lookup_base_record_ex(base_data, op_name, dtype, shape_detail, variant=None):
    """Look up the NVIDIA baseline record, three levels deep:
    op_name -> shape -> dtype. Returns a (record, miss_reason) tuple:

    - (record_dict, None)  when every level matched;
    - (None, reason_str)   when any level is missing, where reason_str names
                           exactly which level failed so callers can print a
                           precise diagnostic and persist it.

    Never raises: an unmatched entry just yields a reason. Baseline layout
    (see tools/collect_baseline_nvidia.py):
        base_data[op_name]["shapes"][str([d0, d1, ...])][str(dtype)]
    The shape key is the string of the first tensor's dimensions, which matches
    the collector's key_shape for ops like grouped_topk / fused_add_rms_norm.
    Ops whose shape cannot be expressed this way simply won't match.
    """
    try:
        op_entry = _resolve_op_entry(base_data, op_name, variant)
        if not op_entry:
            return None, f"baseline 中没有算子 '{op_name}'"
        shapes = op_entry.get("shapes", {})
        first_shape = _first_tensor_shape(shape_detail)
        if first_shape is None:
            return None, (
                f"算子 '{op_name}' 的 shape 无法表达为张量维度 "
                f"(shape_detail={shape_detail})"
            )
        # 优先按原始 key（str(第一个张量 shape)）直查——简单算子的 key_shape 恰为
        # 第一个张量 shape，命中率高且无歧义。
        per_dtype = shapes.get(str(first_shape))
        if per_dtype:
            rec = per_dtype.get(str(dtype))
            if rec is not None:
                return rec, None
        # 直查未命中：多数复杂算子的 baseline key_shape 是语义复合键（如
        # flash_mla 的 [B,S_q,H_q,D,seqlen]）或字符串键（如 t1_e8_h4096...），
        # 与第一个张量 shape 不一致。改用「全部张量 shape 的多重集合」签名回退
        # 匹配——与采集器 config.inputs 对称、两边口径一致。
        rec, matched_key = _match_by_signature(
            base_data, op_name, dtype, shape_detail, variant
        )
        if rec is not None:
            return rec, None
        # 仍未命中：区分「该 shape 签名在 baseline 不存在」与「shape 在但 dtype
        # 缺」两种原因，给出精确诊断。
        if (
            str(first_shape) in shapes
            and shapes[str(first_shape)].get(str(dtype)) is None
        ):
            return None, (
                f"算子 '{op_name}' shape {first_shape} 在 baseline 中没有 "
                f"对应的 dtype {dtype}"
            )
        return None, (
            f"算子 '{op_name}' 在 baseline 中没有匹配的 shape"
            f"（第一个张量={first_shape}，已尝试张量签名回退）"
        )
    except (AttributeError, TypeError) as e:
        return None, f"baseline 查找 '{op_name}' 时出错: {e}"


def lookup_base_record(base_data, op_name, dtype, shape_detail, variant=None):
    """Backward-compatible wrapper: returns just the record dict, or None.

    Prefer lookup_base_record_ex when the miss reason is needed.
    """
    rec, _ = lookup_base_record_ex(base_data, op_name, dtype, shape_detail, variant)
    return rec


def _detect_self_chip():
    """Best-effort name of the chip this benchmark runs on, e.g. "NVIDIA H800".

    Used to pick the right scaling factor when one vendor has several chips
    (H800 vs H100) and to recognize when we're running on the baseline's own
    reference chip. Returns "" when the device name can't be read; callers then
    fall back to vendor-only matching.
    """
    try:
        return torch.cuda.get_device_name()
    except Exception:
        return ""


# Cache the device name once; it doesn't change within a run.
_SELF_CHIP = _detect_self_chip()


def _chip_matches(chip_name, device_name):
    """True if a spec's chip name (e.g. "H800") identifies this device
    (e.g. "NVIDIA H800 80GB"). Case-insensitive substring match, which tolerates
    the vendor prefix / memory suffix that get_device_name() adds."""
    if not chip_name or not device_name:
        return False
    return chip_name.lower() in device_name.lower()


def lookup_scaling_factor(base_data, vendor, bottle_neck_unit, dtype, self_chip=None):
    """Return this chip's hardware scaling factor (chip_peak / reference_peak)
    for the resource that bottlenecks the baseline record, or None.

    Factor comes from base_data["_scaling_factors"] (written by the collector
    into the same file as the latencies). Matching is by vendor AND concrete
    chip model, so that e.g. nvidia H800 and H100 don't collide. Prefers
    measured over nominal. Returns None when the vendor / chip / resource /
    dtype isn't covered, so the caller falls back to the raw latency ratio.

    The reference chip (e.g. the H800 the baseline was collected on) is itself
    listed under "chips" with all factors equal to 1.0, so it matches by model
    like any other chip.
    """
    try:
        sf = base_data.get("_scaling_factors")
        resource = _BOTTLENECK_TO_SF.get(bottle_neck_unit)
        if not sf or resource is None:
            return None
        if self_chip is None:
            self_chip = _SELF_CHIP
        candidates = [c for c in sf.get("chips", []) if c.get("vendor") == vendor]
        if not candidates:
            return None
        # Prefer the candidate whose chip model matches this device; only when a
        # single vendor chip exists do we accept it without a model match.
        chip = next(
            (c for c in candidates if _chip_matches(c.get("chip"), self_chip)), None
        )
        if chip is None:
            chip = candidates[0] if len(candidates) == 1 else None
        if chip is None:
            return None
        for prefer in ("measured", "nominal"):
            block = chip.get(prefer)
            if not block:
                continue
            if resource == "bandwidth_gbps":
                val = block.get("bandwidth_gbps")
            else:
                val = block.get(resource, {}).get(_DTYPE_TO_SF_KEY.get(str(dtype)))
            if val:
                return val
        return None
    except (AttributeError, TypeError):
        return None


@dataclass
class BenchmarkResult:
    """Record the benchmark result for each operator."""

    # Unique name of the operator
    op_name: str
    dtype: str
    mode: str
    level: str
    # Benchmark results
    result: List[BenchmarkMetrics]

    def __str__(self) -> str:
        header_title = (
            f"\nOperator: {self.op_name}  Performance Test (dtype={self.dtype}, mode={self.mode},"
            f"level={self.level})\n"
        )
        col_names = [
            f"{'Status':<10}",
            f"{'Torch Latency (ms)':>20}",
            f"{'Gems Latency (ms)':>20}",
            f"{'Gems Speedup':>20}",
        ]
        # Only surface the baseline column when baseline data was actually
        # matched for this op; otherwise the output stays identical to before.
        show_vs_base = any(m.compared_speedup is not None for m in self.result)
        if show_vs_base:
            col_names.append(f"{'vs Base':>20}")
        if self.result[0].tflops and self.result[0].tflops != 0.0:
            col_names.append(f"{'TFLOPS':>20}")
        if self.result[0].gbps is not None:
            col_names.append(f"{'Torch GBPS ':>20}")
            col_names.append(f"{'Gems GBPS ':>20}")
        col_names.append(f"{'Size Detail':>20}\n")
        header_col_names = " ".join(col_names)
        header_break = "-" * len(header_col_names) + "\n"
        header = header_title + header_col_names + header_break

        metrics_lines = "".join(
            self._format_metrics(ele, show_vs_base) for ele in self.result
        )
        return header + metrics_lines

    def _format_metrics(
        self, metrics: BenchmarkMetrics, show_vs_base: bool = False
    ) -> str:
        # self.gen_legacy_shape(metrics)
        # legacy_shape_str = (
        #     metrics.legacy_shape if metrics.legacy_shape is not None else "N/A"
        # )
        latency_base_str = (
            f"{metrics.latency_base:.6f}" if metrics.latency_base is not None else "N/A"
        )
        latency_str = f"{metrics.latency:.6f}" if metrics.latency is not None else "N/A"
        speedup_str = f"{metrics.speedup:.3f}" if metrics.speedup is not None else "N/A"
        compared_speedup_str = (
            f"{metrics.compared_speedup:.3f}"
            if metrics.compared_speedup is not None
            else "N/A"
        )
        torch_gbps_str = (
            f"{metrics.gbps_base:.3f}" if metrics.gbps_base is not None else "N/A"
        )
        gems_gbps_str = f"{metrics.gbps:.3f}" if metrics.gbps is not None else "N/A"
        if metrics.tflops and metrics.tflops != 0.0:
            tflops_str = (
                f"{metrics.tflops:.3f}" if metrics.tflops is not None else "N/A"
            )
        shape_detail_str = (
            metrics.shape_detail if metrics.shape_detail is not None else "N/A"
        )
        status = "SUCCESS" if metrics.error_msg is None else "FAILED"
        data_line = (
            f"{status:<10}"
            f"{latency_base_str:>20}"
            f"{latency_str:>20}"
            f"{speedup_str:>20}"
        )
        if show_vs_base:
            data_line += f"{compared_speedup_str:>20}"
        if metrics.tflops and metrics.tflops != 0.0:
            data_line += f"{tflops_str:>20}"
        if metrics.gbps is not None:
            data_line += f"{torch_gbps_str:>20}{gems_gbps_str:>20}"
        data_line += " " * 10
        data_line += f"{shape_detail_str}\n"
        return data_line

    def gen_legacy_shape(self, metrics: BenchmarkMetrics) -> Optional[int]:
        first_shape = (
            metrics.shape_detail[0] if isinstance(metrics.shape_detail, list) else None
        )
        to_record_shape = (
            tuple(first_shape) if isinstance(first_shape, torch.Size) else None
        )

        if to_record_shape in LEGACY_NON_BLAS_SHAPES:
            metrics.legacy_shape = to_record_shape[-1]
        elif (
            isinstance(to_record_shape, tuple)
            and len(to_record_shape) == 2
            and to_record_shape[0] == 1024
        ):
            metrics.legacy_shape = to_record_shape[-1]
        else:
            metrics.legacy_shape = None

    def to_json(self) -> str:
        import json

        # Convert to dict and handle tuple serialization for shape_detail
        result_dict = asdict(self)
        return json.dumps(result_dict, default=custom_json_encoder)

    def to_dict(self) -> dict:
        return self.__dict__


# Subset dtypes for specific operators
FP16_BF16_DTYPES = [torch.float16, torch.bfloat16]
