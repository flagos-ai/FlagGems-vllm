# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX gdn_w8a16 versus installed vLLM default and PyTorch reference.

GDN_SHAPE_FILE selects an external captured-shape manifest. GDN_REPO may
select a checkout; GDN_RESULT_DIR selects output. CUDA Graph median timing.
Only package imports are isolated for vendor-container compatibility.

Run: GDN_SHAPE_FILE=/external/shapes.json python benchmark/test_gdn_w8a16_metax.py
Real-shape data remains external. Prefill explicitly opts into mixed_precision=True;
core default precision remains FP32. Decode includes fused raw gates, QK
normalization and FP32 state-cache writeback. CPU preprocessing, cache reset,
projection, convolution and model scheduling are outside the measured scope.

"""


import os
import sys
from pathlib import Path

import torch

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
    from tests.gdn_reference import torch_gdn_prefill

    return torch_gdn_prefill(*args, bounds, normalize=False)


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


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if sys.argv[1:] == ["--shared-benchmark"]:
        import json

        main(rows=json.load(sys.stdin), public_entry=True)
    else:
        main()
