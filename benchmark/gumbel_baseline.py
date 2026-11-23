# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Strict, device-independent lookup for the Gumbel performance workload.

This module belongs to the benchmark, never the production operator. A missing
reference is a latency-only result, not an unnormalized speedup or a zero score.
"""

import hashlib
import json
import math
import re


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def canonical_chip(name):
    """Normalize spelling, preserving capacity, interface and model suffixes."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"^(?:nvidia|huawei|metax|iluvatar|moore\s+threads|ppu)\s+", "", name)
    return re.sub(r"[\s_-]+", "", name)


def normalize_case(
    base_data,
    case_key,
    dtype,
    config,
    candidate_ms,
    vendor,
    chip,
    base_data_source=None,
    timing_protocol=None,
):
    """Return a qualified normalized ratio, or explicit missing-data metadata.

    Native speedup is handled separately by the benchmark. No shape fallback,
    vendor-only factor fallback, or nominal hardware factor is accepted here.
    """
    if not _positive(candidate_ms):
        raise ValueError("candidate latency must be finite and positive (ms)")
    result = {
        "compared_speedup": None,
        "base_raw_speedup": None,
        "base_latency_ms": None,
        "base_scaling_factor": None,
        "base_shape_key": case_key,
        "base_data_source": base_data_source,
        "base_miss_reason": None,
        "base_native_identity": None,
        "base_native_qualification": None,
    }

    def missing(reason):
        result["base_miss_reason"] = reason
        return result

    if not isinstance(base_data, dict):
        return missing("NVIDIA baseline data not supplied or unreadable")
    op = base_data.get("gumbel_sample")
    if not isinstance(op, dict):
        return missing("gumbel_sample absent from NVIDIA baseline")
    shapes = op.get("shapes")
    row = shapes.get(case_key) if isinstance(shapes, dict) else None
    record = row.get(str(dtype)) if isinstance(row, dict) else None
    if not isinstance(record, dict):
        return missing("exact case key/dtype absent from NVIDIA baseline")
    if record.get("status") not in (None, "ok", "passed"):
        return missing("NVIDIA baseline case is not a successful measurement")
    if not isinstance(config, dict) or record.get("config") != config:
        return missing("NVIDIA workload config differs (layout/scalars/RNG/source)")
    identity = record.get("native_identity")
    receipt = record.get("native_qualification")
    result["base_native_identity"] = identity
    result["base_native_qualification"] = receipt
    if config.get("schema_version") != 2:
        return missing("gumbel baseline requires schema 2; recollect NVIDIA records")
    if not isinstance(identity, dict) or not isinstance(receipt, dict):
        return missing("NVIDIA native source identity or qualification receipt absent")
    files = identity.get("files")
    required_files = (
        "vllm/v1/worker/gpu/sample/gumbel.py",
        "vllm/v1/worker/gpu/spec_decode/draft_logits.py",
        "vllm/triton_utils/__init__.py",
    )
    if not isinstance(files, dict) or any(
        not isinstance(files.get(name), dict)
        or not isinstance(files[name].get("path"), str)
        or not files[name]["path"]
        or not isinstance(files[name].get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", files[name]["sha256"])
        for name in required_files
    ):
        return missing(
            "NVIDIA native actual source file paths/hashes absent or invalid"
        )
    device_identity = identity.get("device_identity")
    if (
        identity.get("source_policy") != "compatible-vllm-murmur3"
        or not isinstance(device_identity, dict)
        or device_identity.get("vendor") != "nvidia"
        or receipt.get("status") != "passed"
        or receipt.get("case_key") != case_key
        or receipt.get("dtype") != str(dtype)
        or receipt.get("native_identity") != identity
        or receipt.get("config_sha256")
        != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    ):
        return missing("NVIDIA native qualification does not match case/config/source")
    environment = base_data.get("_environment") or {}
    hardware = environment.get("hardware") if isinstance(environment, dict) else None
    gpu = hardware.get("gpu") if isinstance(hardware, dict) else None
    measured_chip = canonical_chip(gpu.get("name")) if isinstance(gpu, dict) else ""
    if (
        not measured_chip
        or canonical_chip(device_identity.get("device_name")) != measured_chip
    ):
        return missing("NVIDIA native qualification device differs from measured GPU")
    expected_protocol = config.get("timing_protocol")
    actual_protocol = timing_protocol or expected_protocol
    if (
        not expected_protocol
        or actual_protocol != expected_protocol
        or record.get("timing_protocol") != expected_protocol
    ):
        return missing("NVIDIA and candidate timing protocols differ or are missing")
    base_ms = record.get("latency_ms")
    if not _positive(base_ms):
        return missing("NVIDIA latency_ms must be finite and positive")
    result["base_latency_ms"] = base_ms
    raw = base_ms / candidate_ms
    if not _positive(raw):
        return missing("raw cross-device ratio is not finite and positive")
    result["base_raw_speedup"] = raw

    factors = base_data.get("_scaling_factors")
    if not isinstance(factors, dict):
        return missing("hardware scaling factors absent")
    reference_chip = canonical_chip(factors.get("reference_chip"))
    if not reference_chip or reference_chip != measured_chip:
        return missing("factor reference chip does not match measured NVIDIA GPU")
    actual_chip = canonical_chip(chip)
    if not actual_chip:
        return missing("actual target chip identity unavailable")
    chips = factors.get("chips")
    if not isinstance(chips, list):
        return missing("hardware scaling factor chip list absent")
    matches = [
        item
        for item in chips
        if isinstance(item, dict)
        and item.get("vendor") == vendor
        and canonical_chip(item.get("chip")) == actual_chip
    ]
    if len(matches) != 1:
        return missing("exact vendor/chip factor absent or ambiguous")
    resources = matches[0].get("measured")
    if not isinstance(resources, dict):
        return missing("measured hardware factors absent (nominal is not accepted)")
    bottleneck = record.get("bottle_neck_unit")
    if bottleneck == "mem":
        resource, factor = "bandwidth_gbps", resources.get("bandwidth_gbps")
    elif bottleneck in ("cuda", "tensor"):
        resource = "vector" if bottleneck == "cuda" else "tensor"
        # This workload fixes FP32 scores and use_fp64=False. Input precision
        # must never silently stand in for a different computation precision.
        if str(dtype) != "torch.float32":
            return missing("unqualified computation dtype for hardware factor")
        block = resources.get(resource)
        factor = block.get("fp32") if isinstance(block, dict) else None
    else:
        return missing("NVIDIA NCU bottleneck classification absent or unsupported")
    if not _positive(factor):
        return missing("measured bottleneck factor must be finite and positive")
    normalized = raw / factor
    if not _positive(normalized):
        return missing("normalized ratio is not finite and positive")
    result.update(
        compared_speedup=normalized,
        base_scaling_factor=factor,
        factor_chip=matches[0]["chip"],
        factor_resource=resource,
        factor_reference_chip=factors["reference_chip"],
    )
    return result
