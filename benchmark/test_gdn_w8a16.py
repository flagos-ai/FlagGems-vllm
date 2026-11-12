# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GDN benchmarks for FP8, INT8 and MetaX FP32 state.

MetaX: GDN_SHAPE_FILE=/external/shapes.json python benchmark/test_gdn_w8a16.py
The external manifest retains all 110 shapes and call counts. MetaX timing uses
CUDA Graph medians against installed native vLLM-MetaX and PyTorch; prefill
explicitly enables mixed_precision=True. Input creation, state reset, metadata
preparation, compilation and capture stay outside the measured scope.
"""

import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
import yaml

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Select the vendor CUDA runtime before FlagGems/TileLang loads its stub.
    from vllm.plugins import load_general_plugins

    load_general_plugins()

import flaggems_vllm  # noqa: E402
from benchmark.base import Benchmark  # noqa: E402
from benchmark.conftest import Config  # noqa: E402
from benchmark.consts import BenchLevel  # noqa: E402
from flaggems_vllm.ops.gdn_w8a16 import quantize_gdn_state_fp8  # noqa: E402
from tests.test_gdn_w8a16 import (  # noqa: E402
    build_int8_gdn_decode_inputs,
    build_int8_gdn_prefill_inputs,
)


def _vllm_bf16(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

    return fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=state_bf16,
        inplace_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )


def _fp8(q, k, v, g, beta, state_bf16, state_fp8, state_scale, cu, indices, t):
    return flaggems_vllm.gdn_w8a16(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        state_fp8,
        state_scale,
        cu_seqlens=cu,
        ssm_state_indices=indices[:, 0],
        use_qk_l2norm_in_kernel=True,
        max_sequence_length=t,
    )


class GdnW8A16Benchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "N, T, H, HV, K, V"
    DEFAULT_SHAPES = [
        (n, t, 2, hv, 128, 128)
        for hv in (4, 6)
        for n, t in (
            [
                (n, 1)
                for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
            ]
            + [
                (1, 2),
                (1, 8),
                (1, 32),
                (4, 4),
                (4, 16),
                (16, 4),
                (1, 256),
                (1, 512),
                (2, 192),
                (4, 256),
                (8, 128),
                (16, 128),
                (32, 64),
                (64, 32),
                (128, 24),
            ]
        )
    ]

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def get_input_iter(self, cur_dtype):
        for n, t, h, hv, key_dim, value_dim in self.shapes:
            if key_dim != 128 or value_dim != 128:
                raise ValueError("This benchmark covers K=V=128")
            torch.manual_seed(42)
            device = flaggems_vllm.device
            q = torch.randn(1, n * t, h, key_dim, device=device, dtype=cur_dtype)
            k = torch.randn_like(q)
            v = 0.125 * torch.randn(
                1, n * t, hv, value_dim, device=device, dtype=cur_dtype
            )
            g = torch.empty(1, n * t, hv, device=device, dtype=cur_dtype).uniform_(
                math.log(0.98), math.log(0.995)
            )
            beta = torch.rand(1, n * t, hv, device=device, dtype=cur_dtype)
            state = torch.zeros(
                n + 1, hv, key_dim, value_dim, device=device, dtype=cur_dtype
            )
            state_fp8, state_scale = quantize_gdn_state_fp8(state)
            cu = torch.arange(0, n * t + 1, t, device=device, dtype=torch.int32)
            indices = (
                torch.arange(1, n + 1, device=device, dtype=torch.int64)[:, None]
                .expand(n, t)
                .contiguous()
            )
            inputs = q, k, v, g, beta, state, state_fp8, state_scale, cu, indices, t
            baseline, _ = _vllm_bf16(*inputs)
            torch.cuda.synchronize()
            # A zero initial state gives a closed-form first-token output.
            query = q.reshape(n, t, h, key_dim)[:, 0].float()
            key = k.reshape(n, t, h, key_dim)[:, 0].float()
            query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            correlation = (query * key).sum(-1).repeat_interleave(hv // h, -1)
            expected = (
                correlation[..., None]
                * key_dim**-0.5
                * v.reshape(n, t, hv, value_dim)[:, 0].float()
                * beta.reshape(n, t, hv)[:, 0].float()[..., None]
            )
            first_output = baseline.reshape(n, t, hv, value_dim)[:, 0]
            torch.testing.assert_close(
                first_output.float(), expected, atol=2e-3, rtol=5e-2
            )
            state.zero_()
            yield inputs


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "nvidia"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 GDN requires NVIDIA SM90 or newer",
)
def test_gdn_w8a16():
    bench = GdnW8A16Benchmark(op_name="gdn_w8a16", torch_op=_vllm_bf16)
    bench.set_gems(_fp8)
    bench.run()


HYGON_INT8_STATE = (
    "gdn_w8a16_int8" in vars(flaggems_vllm)
    and flaggems_vllm.gdn_w8a16_int8.__module__
    == "flaggems_vllm.runtime.backend._hygon.ops.gdn_w8a16"
)


class Int8StateGdnBenchmark(Benchmark):
    DEFAULT_DTYPES = (
        [torch.bfloat16, torch.float16] if HYGON_INT8_STATE else [torch.bfloat16]
    )
    DEFAULT_SHAPES = [
        (1, 8192, 32, 128),
        (3, 8192, 32, 128),
        ("update", 1, 3, 16, 128, 32, 128, 3),
        (1, 17, 16, 128, 32, 128, 1),
        (1, 1826, 16, 128, 32, 128, 2),
    ]
    DEFAULT_SHAPE_DESC = (
        "decode: T,packed,HV,V; prefill: B,T,H,K,HV,V,N; update: update,B,T,H,K,HV,V,N"
    )

    def set_shapes(self, shape_file_path: str):
        # The shared operator ID also covers FP8, whose shape layout differs.
        with open(shape_file_path) as shape_file:
            shape_config = yaml.safe_load(shape_file).get("gdn_w8a16_int8")
        if shape_config is None:
            super().set_shapes(shape_file_path)
        else:
            self.shapes = [
                tuple(shape)
                for shape in shape_config.get("shapes", self.DEFAULT_SHAPES)
            ]
            self.shape_desc = shape_config.get("shape_desc", self.DEFAULT_SHAPE_DESC)
        if (
            Config.bench_level == BenchLevel.CORE
            and Path(shape_file_path).resolve()
            == Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            # Keep the default core run small; explicit shape files run in full.
            self.shapes = [
                shape for shape in self.shapes if shape in self.DEFAULT_SHAPES
            ]

    def get_input_iter(self, dtype):
        if HYGON_INT8_STATE:
            from vllm.model_executor.layers.fla.ops.chunk import (
                chunk_gated_delta_rule as prefill,
            )
            from vllm.model_executor.layers.fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode as packed_decode,
            )
            from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
                fused_sigmoid_gating_delta_rule_update as update,
            )
        else:
            # Resolve the compiled backends selected by vLLM-SAIL 0.31.0. Importing
            # upstream FLA functions alone would select its Triton implementation.
            from vllm_sail.attention.pla_decode import (
                get_sail_cuda_pla_k_last,
                get_sail_cuda_pla_k_last_packed,
            )
            from vllm_sail.attention.pla_prefill import get_sail_cuda_pla_prefill_fwd

            packed_decode = get_sail_cuda_pla_k_last_packed()
            update = get_sail_cuda_pla_k_last()
            prefill = get_sail_cuda_pla_prefill_fwd()
            if packed_decode is None or update is None or prefill is None:
                raise RuntimeError(
                    "GDN benchmarks require the vLLM-SAIL PLA native backends"
                )

        for shape in self.shapes:
            torch.manual_seed(42)
            if len(shape) == 7:
                _, reference, args = build_int8_gdn_prefill_inputs(shape, dtype)
                yield ("prefill", prefill, reference), args
                continue

            separate_qkv = shape[0] == "update"
            if separate_qkv:
                _, batch, tokens, query_heads, reduction, heads, columns, sequences = (
                    shape
                )
                if batch != 1 or sequences != tokens:
                    raise ValueError(
                        "update shapes require B=1 and one token per sequence"
                    )
                packed = 2 * query_heads * reduction + heads * columns
            else:
                tokens, packed, heads, columns = shape
                reduction = 128
            _, reference, args = build_int8_gdn_decode_inputs(
                (tokens, packed, heads, columns), dtype, reduction
            )
            mixed, a, b, a_log, bias, scale, state, out, indices, normalize = reference
            native_bias = bias.to(dtype)
            native_indices = indices.to(torch.int32)
            cu_seqlens = torch.arange(
                tokens + 1, device=mixed.device, dtype=torch.int32
            )
            if separate_qkv:
                q, k, v = torch.split(
                    mixed,
                    (query_heads * reduction, query_heads * reduction, heads * columns),
                    dim=-1,
                )
                qkv = (
                    q.reshape(1, tokens, query_heads, reduction).contiguous(),
                    k.reshape(1, tokens, query_heads, reduction).contiguous(),
                    v.reshape(1, tokens, heads, columns).contiguous(),
                )
                if HYGON_INT8_STATE:
                    native_call = partial(
                        update,
                        A_log=a_log,
                        a=a,
                        b=b,
                        dt_bias=bias,
                        q=qkv[0],
                        k=qkv[1],
                        v=qkv[2],
                        scale=scale,
                        initial_state=state,
                        inplace_final_state=True,
                        cu_seqlens=cu_seqlens,
                        ssm_state_indices=indices,
                        use_qk_l2norm_in_kernel=normalize,
                    )
                else:
                    native_call = partial(
                        update,
                        a_log,
                        a,
                        native_bias,
                        1.0,
                        20.0,
                        *qkv,
                        b,
                        state,
                        native_indices,
                        scale,
                        normalize,
                        cu_seqlens,
                        False,
                        False,
                        None,
                        None,
                        None,
                        None,
                        None,
                        False,
                    )
                yield ("update", native_call, state), (qkv, *args[1:])
            else:
                if HYGON_INT8_STATE:
                    native_call = partial(packed_decode, *reference)
                else:
                    native_call = partial(
                        packed_decode,
                        mixed,
                        a,
                        b,
                        a_log,
                        native_bias,
                        1.0,
                        20.0,
                        scale,
                        state,
                        native_indices,
                        out,
                        normalize,
                        cu_seqlens,
                        False,
                        False,
                    )
                yield ("decode", native_call, state), args


def native_int8_state_gdn_baseline(reference, args):
    kind, native_call, data = reference
    if kind == "prefill":
        if HYGON_INT8_STATE:
            return native_call(
                q=data["q"],
                k=data["k"],
                v=data["v"],
                g=data["g"],
                beta=data["beta"],
                initial_state=data["initial_state"],
                cu_seqlens=data["cu_seqlens"],
                chunk_indices=args[8],
                chunk_offsets=args[9],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        # Include normalization, beta conversion and the native state-layout
        # bridge in timing. Host sequence metadata is prepared for both routes.
        result = native_call(
            data["q"],
            data["k"],
            data["v"],
            data["g"],
            data["beta"].float(),
            initial_state=data["initial_state"].transpose(-1, -2).contiguous(),
            cu_seqlens=data["cu_seqlens"],
            output_final_state=True,
            normalize_qk=True,
            raw_cu_seqlens=data["raw_cu_seqlens"],
        )
        return result[2], result[4].transpose(-1, -2)
    elif HYGON_INT8_STATE:
        return native_call()
    else:
        return native_call(), data


def run_int8_state_gdn(reference, args):
    if reference[0] == "prefill":
        return flaggems_vllm.chunk_gated_delta_rule_w8a16_int8(*args)
    elif reference[0] == "update" and not HYGON_INT8_STATE:
        import flag_gems

        tokens = args[0][0].shape[1]
        mixed = flag_gems.cat(
            tuple(tensor.view(tokens, -1) for tensor in args[0]), dim=1
        )
        return flaggems_vllm.gdn_w8a16_int8(mixed, *args[1:])
    else:
        return flaggems_vllm.gdn_w8a16_int8(*args)


@pytest.mark.gdn_w8a16
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flaggems_vllm),
    reason="backend does not export INT8-state GDN",
)
def test_perf_gdn_w8a16_int8():
    bench = Int8StateGdnBenchmark(
        op_name="gdn_w8a16", torch_op=native_int8_state_gdn_baseline
    )
    bench.base_variant = "int8"
    bench.set_gems(run_int8_state_gdn)
    bench.run()


SHAPES = Path(
    os.getenv(
        "GDN_SHAPE_FILE",
        "/data/fzy/real_shapes_20261012/gdn_w8a16_fp8__int8/shapes.json",
    )
)


def summarize_results(results):
    """Keep frequency-weighted ratios distinct from total workload time savings."""
    groups = {"all": results}
    for row in results:
        groups.setdefault(row["op"], []).append(row)
    summary = {}
    for scope, rows in groups.items():
        calls = sum(row["count"] for row in rows)
        if calls <= 0 or any(row["count"] <= 0 for row in rows):
            raise ValueError("Captured shape counts must be positive")
        operator_time = sum(row["count"] * row["us"]["optimized"] for row in rows)
        baselines = {}
        for name in ("vllm_metax_default", "pytorch"):
            if not all(name in row["us"] for row in rows):
                continue
            baseline_time = sum(row["count"] * row["us"][name] for row in rows)
            baselines[name] = {
                "call_weighted_speedup": sum(
                    row["count"] * row["us"][name] / row["us"]["optimized"]
                    for row in rows
                )
                / calls,
                "weighted_total_time_speedup": baseline_time / operator_time,
                "weighted_baseline_us": baseline_time / calls,
            }
        summary[scope] = {
            "shapes": len(rows),
            "calls": calls,
            "weighted_operator_us": operator_time / calls,
            "baselines": baselines,
        }
    return summary


def make_inputs(row):
    shape = row["shape"]
    if len(shape) == 4:
        tokens, width, hv, dv = shape
        dk, n = 128, tokens
        heads, remainder = divmod(width - hv * dv, 2 * dk)
        if heads <= 0 or remainder or hv % heads:
            raise ValueError(f"Invalid packed GDN dimensions: {shape}")
        packed = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
        q = packed[:, : heads * dk].view(1, tokens, heads, dk)
        k = packed[:, heads * dk : 2 * heads * dk].view(1, tokens, heads, dk)
        v = packed[:, 2 * heads * dk :].view(1, tokens, hv, dv)
    else:
        _, tokens, heads, dk, hv, dv, n = shape
        q = torch.randn(1, tokens, heads, dk, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, tokens, hv, dv, device="cuda", dtype=torch.bfloat16)
    # Raw normal draws; no per-vector normalization. Small beta keeps the
    # recurrence stable with unit-variance keys (expected squared norm = K).
    g = -torch.rand(1, tokens, hv, device="cuda") * 0.1
    beta = torch.rand(1, tokens, hv, device="cuda") / dk
    state = torch.randn(n, hv, dv, dk, device="cuda") * 0.1
    bounds = [i * tokens // n for i in range(n + 1)]
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return (q, k, v, g, beta, state, cu), bounds


def torch_baseline(args, bounds):
    q, k, v, g, beta, state, _ = args
    if len(bounds) - 1 == q.shape[1]:
        group = v.shape[2] // q.shape[2]
        qq = q[0].float().repeat_interleave(group, 1)
        kk = k[0].float().repeat_interleave(group, 1)
        vv = v[0].float()
        s = state * g[0].exp()[..., None, None]
        delta = (vv - torch.matmul(s, kk.unsqueeze(-1)).squeeze(-1)) * beta[
            0, :, :, None
        ]
        s = s + delta.unsqueeze(-1) * kk.unsqueeze(-2)
        o = torch.matmul(s, qq.unsqueeze(-1)).squeeze(-1) * (q.shape[-1] ** -0.5)
        return o.unsqueeze(0).to(q.dtype), s
    from tests.test_gdn_w8a16 import torch_gdn_prefill

    return torch_gdn_prefill(q, k, v, g, beta, state, bounds, normalize=False)


def main(*, rows=None, public_entry=False):
    import gc
    import hashlib
    import importlib.util
    import inspect
    import json
    import os
    import statistics
    import sys
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import triton
    import vllm
    import vllm_metax
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fla.ops import chunk_delta_h
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )
    from vllm.platforms import current_platform

    ROOT = Path(os.environ.get("GDN_RESULT_DIR", "gdn_w8a16_vllm_results"))
    shape_file = SHAPES
    captured_counts = rows is None
    ROOT.mkdir(parents=True, exist_ok=True)
    REPO = Path(os.environ.get("GDN_REPO", Path(__file__).resolve().parents[1]))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    def source_path(fn):
        while hasattr(fn, "fn"):
            fn = fn.fn
        return inspect.getsourcefile(fn)

    if public_entry:
        from flaggems_vllm import gdn_w8a16 as candidate

        entry_path = Path(source_path(candidate))
        candidate_import = "flaggems_vllm.gdn_w8a16"
    else:
        # The standalone command can benchmark an unpacked checkout in a vendor image.
        scratch = tempfile.TemporaryDirectory(prefix="gdn-isolated-")
        tmp = Path(scratch.name)
        entry_path = REPO / "src/flaggems_vllm/runtime/backend/_metax/ops/gdn_w8a16.py"
        entry = entry_path.read_text().replace(
            "from flaggems_vllm import runtime",
            'from types import SimpleNamespace\nruntime = SimpleNamespace(device=SimpleNamespace(vendor_name="metax"))',
        )
        (tmp / "isolated_entry.py").write_text(entry)
        candidate = load("isolated_entry", tmp / "isolated_entry.py").gdn_w8a16
        candidate_import = "standalone source with MetaX runtime constant"

    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_key_head_dim=128)
    )
    with set_current_vllm_config(cfg):
        prefill = ChunkGatedDeltaRule()
    assert prefill.gdn_prefill_backend == "triton"
    assert os.getenv("MACA_VLLM_USE_EXTERNAL_FLA", "0") == "0"
    paths = [
        Path(__file__),
        entry_path,
        Path(source_path(prefill._forward_method)),
        Path(source_path(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)),
        Path(source_path(fused_recurrent_gated_delta_rule_packed_decode)),
        Path(source_path(fused_sigmoid_gating_delta_rule_update)),
    ]
    samples_per_case = int(os.environ.get("GDN_SAMPLES", "31"))
    metadata = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        vllm=vllm.__version__,
        vllm_metax=vllm_metax.__version__,
        platform=type(current_platform).__name__,
        gpu=torch.cuda.get_device_name(),
        candidate_revision=os.getenv("GDN_IMPL_COMMIT", "see source hashes"),
        correctness={"atol": 0.002, "rtol": 0.02},
        prefill_backend=prefill.gdn_prefill_backend,
        prefill_method=prefill._forward_method.__qualname__,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        timing=(
            f"CUDA Graph median of {samples_per_case} samples; mean of 3 individually timed "
            "replays/sample; state reset outside events"
        ),
        seed=20261019,
        experimental_config={
            k: v for k, v in os.environ.items() if k.startswith(("GDN_", "DECODE_"))
        },
        shape_count_source="manifest" if captured_counts else "one per YAML shape",
        raw_shape_file_sha256=hashlib.sha256(shape_file.read_bytes()).hexdigest(),
        candidate_import_adaptation=candidate_import,
        prefill_scope="precomputed gates, no QK normalization, immutable FP32 state, metadata prebuilt",
        decode_scope=(
            "gate generation + QK L2 normalization + recurrence + "
            "in-place FP32 state cache update; identity slots 1..N"
        ),
        excluded="projection, convolution, model scheduling, metadata construction, input creation, reset, capture/JIT",
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_PLUGINS", "MACA_VLLM_"))
        },
    )
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False

    def build_case(row):
        args, bounds = make_inputs(row)
        q, k, v, g, beta, state, cu = args
        if row["op"] == "chunk_gated_delta_rule":
            indices = prepare_chunk_indices(cu, 64)
            offsets = prepare_chunk_offsets(cu, 64)
            native = lambda: prefill(
                q, k, v, g, beta, state, True, cu, indices, offsets, False
            )
            return (
                {
                    "optimized": lambda: candidate(*args, mixed_precision=True),
                    "vllm_metax_default": native,
                    "pytorch": lambda: torch_baseline(args, bounds),
                },
                {},
                torch_baseline(args, bounds),
            )
        n, h, dk, hv, dv = (
            state.shape[0],
            q.shape[2],
            q.shape[3],
            v.shape[2],
            v.shape[3],
        )
        # Packed/raw QKV remain shared across paths. Gate values match historical stable range.
        mixed = torch.cat((q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)), dim=-1)
        q = mixed[:, : h * dk].view(1, n, h, dk)
        k = mixed[:, h * dk : 2 * h * dk].view(1, n, h, dk)
        v = mixed[:, 2 * h * dk :].view(1, n, hv, dv)
        a = torch.log(torch.expm1(-g[0])).to(torch.bfloat16)
        b = torch.logit(beta[0].clamp(min=1e-7, max=1 - 1e-7)).to(torch.bfloat16)
        alog = torch.zeros(hv, device=q.device, dtype=torch.float32)
        bias = torch.zeros_like(alog)
        ids = torch.arange(1, n + 1, device=q.device, dtype=torch.int32)
        original = torch.cat((torch.zeros_like(state[:1]), state), dim=0)
        caches = {
            name: original.clone()
            for name in ["optimized", "vllm_metax_default", "pytorch"]
        }

        def core_args(cache):
            qq = q.float()
            kk = k.float()
            qq = (qq / torch.sqrt(qq.square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
            kk = (kk / torch.sqrt(kk.square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
            gg = (
                -torch.exp(alog) * torch.nn.functional.softplus(a.float() + bias)
            ).unsqueeze(0)
            bb = b.float().sigmoid().to(b.dtype).float().unsqueeze(0)
            return qq, kk, v, gg, bb, cache[1:], cu

        optimized_out = torch.empty((1, n, hv, dv), device=q.device, dtype=q.dtype)

        def optimized():
            cache = caches["optimized"]
            packed = row["op"] == "fused_recurrent_gated_delta_rule_packed_decode"
            candidate(
                mixed if packed else q,
                None if packed else k,
                None if packed else v,
                initial_state=cache,
                decode=True,
                a=a,
                b=b,
                A_log=alog,
                dt_bias=bias,
                state_indices=ids,
                out=optimized_out,
            )
            return optimized_out, cache[1:]

        def pytorch():
            cache = caches["pytorch"]
            o, f = torch_baseline(core_args(cache), bounds)
            cache[1:].copy_(f)
            return o, cache[1:]

        # FP32 reference normalizes without BF16 rounding, matching native fused math.
        ref_args = list(core_args(original))
        ref_args[0] = q.float() / torch.sqrt(
            q.float().square().sum(-1, keepdim=True) + 1e-6
        )
        ref_args[1] = k.float() / torch.sqrt(
            k.float().square().sum(-1, keepdim=True) + 1e-6
        )
        expected = torch_baseline(tuple(ref_args), bounds)
        expected = (expected[0].to(v.dtype), expected[1])
        if row["op"] == "fused_recurrent_gated_delta_rule_packed_decode":
            out = torch.empty((n, 1, hv, dv), device=q.device, dtype=v.dtype)

            def native():
                o, f = fused_recurrent_gated_delta_rule_packed_decode(
                    mixed,
                    a,
                    b,
                    alog,
                    bias,
                    dk**-0.5,
                    caches["vllm_metax_default"],
                    out,
                    ids,
                    True,
                )
                return o.reshape(1, n, hv, dv), f[1:]

        else:

            def native():
                o, f = fused_sigmoid_gating_delta_rule_update(
                    alog,
                    a,
                    b,
                    bias,
                    q,
                    k,
                    v,
                    initial_state=caches["vllm_metax_default"],
                    inplace_final_state=True,
                    cu_seqlens=cu,
                    ssm_state_indices=ids,
                    use_qk_l2norm_in_kernel=True,
                )
                return o, f[1:]

        resets = {
            name: (lambda cache=cache: cache.copy_(original))
            for name, cache in caches.items()
        }
        return (
            {"optimized": optimized, "vllm_metax_default": native, "pytorch": pytorch},
            resets,
            expected,
        )

    def run_case(row):
        fns, resets, expected = build_case(row)
        if os.getenv("GDN_NO_TORCH") == "1":
            fns.pop("pytorch")
        graphs = {}
        outputs = {}
        valid = {}
        errors = {}
        memory = {}
        strict = {}
        relative_rms = {}
        noop = lambda: None
        for name, fn in fns.items():
            reset = resets.get(name, noop)
            for _ in range(3):
                reset()
                fn()
            reset()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            actual = fn()
            torch.cuda.synchronize()
            memory[name] = (torch.cuda.max_memory_allocated() - before) / 1024**2
            strict[name] = torch.allclose(
                actual[1].float(), expected[1].float(), atol=1e-4, rtol=0.002
            )
            relative_rms[name] = float(
                (actual[1].float() - expected[1].float()).square().mean().sqrt()
                / expected[1].float().square().mean().sqrt().clamp_min(1e-20)
            )
            valid[name] = all(
                torch.allclose(a.float(), b.float(), atol=0.002, rtol=0.02)
                for a, b in zip(actual, expected)
            )
            errors[name] = [
                float((a.float() - b.float()).abs().max())
                for a, b in zip(actual, expected)
            ]
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            reset()
            graph.replay()
            torch.cuda.synchronize()
            valid[name] &= all(
                torch.allclose(a.float(), b.float(), atol=0.002, rtol=0.02)
                for a, b in zip(out, expected)
            )
            graphs[name] = graph
            outputs[name] = out
        if not all(valid.values()):
            return dict(
                valid=valid,
                errors=errors,
                strict=strict,
                relative_rms=relative_rms,
                peak_mib=memory,
                us={},
                samples_us={},
            )
        samples = {name: [] for name in fns}
        names = list(fns)
        for i in range(samples_per_case):
            order = names[i % len(names) :] + names[: i % len(names)]
            if i % 2:
                order = order[::-1]
            for name in order:
                events = []
                for _ in range(3):
                    resets.get(name, noop)()
                    b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    b.record()
                    graphs[name].replay()
                    e.record()
                    events.append((b, e))
                events[-1][1].synchronize()
                samples[name].append(
                    statistics.mean(b.elapsed_time(e) * 1000 for b, e in events)
                )
        return dict(
            valid=valid,
            errors=errors,
            strict=strict,
            relative_rms=relative_rms,
            peak_mib=memory,
            us={k: statistics.median(v) for k, v in samples.items()},
            samples_us=samples,
        )

    if captured_counts:
        rows = json.loads(shape_file.read_text())
    results = []
    selected = os.getenv("GDN_CASES")
    selected = set(map(int, selected.split(","))) if selected else None
    for index, row in enumerate(rows, 1):
        if selected and index not in selected:
            continue
        print("START", index, row["op"], row["shape"], flush=True)
        try:
            result = dict(index=index, **row, **run_case(row))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            result = dict(index=index, **row, error=repr(exc))
        results.append(result)
        (ROOT / "results.json").write_text(
            json.dumps(dict(metadata=metadata, results=results), indent=2)
        )
        print({k: v for k, v in result.items() if k != "samples_us"}, flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    if not results:
        raise ValueError("No GDN shapes selected")
    print("DONE", flush=True)
    if any("error" in r or not all(r["valid"].values()) for r in results):
        raise SystemExit(1)
    summary = summarize_results(results)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("WEIGHTED SUMMARY", json.dumps(summary), flush=True)


def _metax_shape_row(shape):
    # Shared YAML tags distinguish independent QKV updates from chunk prefill.
    if len(shape) == 8 and shape[0] == "update":
        op = "fused_sigmoid_gating_delta_rule_update"
        shape = shape[1:]
        if shape[0] != 1 or shape[1] != shape[-1]:
            raise ValueError(f"Expected one token per update request: {shape}")
    elif len(shape) == 4:
        op = "fused_recurrent_gated_delta_rule_packed_decode"
    elif len(shape) == 7:
        op = "chunk_gated_delta_rule"
    else:
        raise ValueError(f"Unsupported GDN shape: {shape}")
    return {"op": op, "shape": list(shape), "count": 1}


@pytest.mark.gdn_w8a16
def test_perf_gdn_w8a16_metax(tmp_path):
    """Use shared shapes with the paired CUDA Graph benchmark and FP32 state."""
    from flaggems_vllm import vendor_name

    if vendor_name != "metax":
        pytest.skip("MetaX FP32 state")
    from benchmark.base import Benchmark
    from benchmark.conftest import Config, emit_record_logger, update_result
    from benchmark.consts import BenchLevel

    if Config.user_desired_dtypes and torch.bfloat16 not in Config.user_desired_dtypes:
        pytest.skip("MetaX GDN benchmarks use BF16 activations and FP32 state")
    shape_file = Path(os.environ.get("GDN_SHAPE_FILE") or Config.shape_file)
    result_dir = Path(os.environ.get("GDN_RESULT_DIR", tmp_path)).resolve()
    rows = None
    if shape_file.suffix.lower() != ".json":
        # Only reuse the compatible packed/update/prefill dimensions from YAML.
        bench = Benchmark(op_name="gdn_w8a16_int8", torch_op=None)
        bench.DEFAULT_SHAPES = ()
        bench.set_shapes(str(shape_file))
        if not bench.shapes:
            bench.op_name = "gdn_w8a16"
            bench.set_shapes(str(shape_file))
        shapes = bench.shapes
        if (
            Config.bench_level == BenchLevel.CORE
            and shape_file.resolve()
            == Path(__file__).with_name("core_shapes.yaml").resolve()
        ):
            shapes = shapes[:2]
        rows = [_metax_shape_row(shape) for shape in shapes]
    # Load vLLM's device plugin before FlagGems/TileLang in a fresh process.
    # In pytest the CUDA stub can already be loaded and mask the MACA runtime.
    subprocess.run(
        [sys.executable, "-m", "benchmark.test_gdn_w8a16", "--shared-benchmark"],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "GDN_SHAPE_FILE": str(shape_file.resolve()),
            "GDN_RESULT_DIR": str(result_dir),
        },
        input=json.dumps(rows),
        text=True,
        check=True,
    )
    results = json.loads((result_dir / "results.json").read_text())["results"]
    for baseline in ("vllm_metax_default", "pytorch"):
        metrics = [
            dict(
                shape_detail=row["shape"],
                path=row["op"],
                count=row["count"],
                latency=row["us"]["optimized"] / 1000,
                latency_base=row["us"][baseline] / 1000,
                speedup=row["us"][baseline] / row["us"]["optimized"],
            )
            for row in results
            if baseline in row["us"]
        ]
        if metrics:
            record = dict(
                op_name="gdn_w8a16",
                dtype=str(torch.bfloat16),
                mode="cudagraph",
                level=Config.bench_level.value,
                baseline=baseline,
                result=metrics,
            )
            update_result("gdn_w8a16", record)
            emit_record_logger(json.dumps(record))


if __name__ == "__main__":
    if sys.argv[1:] == ["--shared-benchmark"]:
        main(rows=json.load(sys.stdin), public_entry=True)
    else:
        main()
