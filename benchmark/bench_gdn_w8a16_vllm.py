# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the installed vLLM-MetaX default GDN path against this PR.

Run in an existing vLLM-MetaX environment, without replacing its compiler:
    GDN_SHAPE_FILE=/external/shapes.json python benchmark/bench_gdn_w8a16_vllm.py
Optional: GDN_REPO selects a frozen source checkout; GDN_CASES selects comma-
separated 1-based case indices; GDN_RESULT_DIR selects the output directory.

Prefill measures the default selected Triton/FLA core with precomputed gates,
matching the layer's prefill call (normalization already handled upstream).
Decode includes gate generation, QK normalization and state cache writeback;
state reset is outside timing. The migration adapter uses PyTorch preprocessing
and cache copy, so decode results describe this adapter, not a fused GDN kernel.
The primitive named by each external shape record is used for decode.

Only the candidate's two package imports are isolated to allow execution in
vendor containers without FlagGems dependencies. Kernel source and launch
configuration are unchanged; source hashes and installed versions are recorded.
This is an operator-level comparison, not full-model inference throughput.
"""


def main():
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

    # Load exactly the PR kernels without importing unrelated operators/dependencies.
    # Only two import statements are adapted; computation and launch configs unchanged.
    scratch = tempfile.TemporaryDirectory(prefix="gdn-isolated-")
    tmp = Path(scratch.name)
    sys.path.insert(0, str(tmp))
    entry_path = REPO / "src/flaggems_vllm/runtime/backend/_metax/ops/gdn_w8a16.py"
    chunk_path = REPO / "src/flaggems_vllm/runtime/backend/_metax/gdn_w8a16_chunk.py"
    entry = entry_path.read_text()
    assert entry.count("from flaggems_vllm import runtime") == 1
    assert (
        entry.count("from flaggems_vllm.runtime.backend._metax.gdn_w8a16_chunk import")
        == 1
    )
    entry = entry.replace(
        "from flaggems_vllm import runtime",
        'from types import SimpleNamespace\nruntime = SimpleNamespace(device=SimpleNamespace(vendor_name="metax"))',
    )
    entry = entry.replace(
        "from flaggems_vllm.runtime.backend._metax.gdn_w8a16_chunk import",
        "from isolated_chunk import",
    )
    (tmp / "isolated_chunk.py").write_bytes(chunk_path.read_bytes())
    (tmp / "isolated_entry.py").write_text(entry)
    candidate = load("isolated_entry", tmp / "isolated_entry.py").gdn_w8a16
    bench_path = REPO / "benchmark/test_gdn_w8a16_metax.py"
    (tmp / "isolated_bench.py").write_text(
        bench_path.read_text().replace(
            "from flaggems_vllm import gdn_w8a16",
            "from isolated_entry import gdn_w8a16",
        )
    )
    bench = load("isolated_bench", tmp / "isolated_bench.py")

    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(linear_key_head_dim=128)
    )
    with set_current_vllm_config(cfg):
        prefill = ChunkGatedDeltaRule()
    assert prefill.gdn_prefill_backend == "triton"
    assert os.getenv("MACA_VLLM_USE_EXTERNAL_FLA", "0") == "0"
    paths = [
        entry_path,
        chunk_path,
        bench_path,
        Path(source_path(prefill._forward_method)),
        Path(source_path(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)),
        Path(source_path(fused_recurrent_gated_delta_rule_packed_decode)),
        Path(source_path(fused_sigmoid_gating_delta_rule_update)),
    ]
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
            "CUDA Graph median of 31 samples; mean of 3 individually timed "
            "replays/sample; state reset outside events"
        ),
        seed=20261019,
        raw_shape_file_sha256=hashlib.sha256(bench.SHAPES.read_bytes()).hexdigest(),
        candidate_import_adaptation="runtime vendor constant + local chunk import only; no kernel edits",
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
        args, bounds = bench.make_inputs(row)
        q, k, v, g, beta, state, cu = args
        if row["op"] == "chunk_gated_delta_rule":
            indices = prepare_chunk_indices(cu, 64)
            offsets = prepare_chunk_offsets(cu, 64)
            native = lambda: prefill(
                q, k, v, g, beta, state, True, cu, indices, offsets, False
            )
            return (
                {
                    "migrated": lambda: candidate(*args),
                    "vllm_metax_default": native,
                    "pytorch": lambda: bench.torch_baseline(args, bounds),
                },
                {},
                bench.torch_baseline(args, bounds),
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
            for name in ["migrated", "vllm_metax_default", "pytorch"]
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

        def migrated():
            cache = caches["migrated"]
            o, f = candidate(*core_args(cache))
            cache[1:].copy_(f)
            return o, cache[1:]

        def pytorch():
            cache = caches["pytorch"]
            o, f = bench.torch_baseline(core_args(cache), bounds)
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
        expected = bench.torch_baseline(tuple(ref_args), bounds)
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
            {"migrated": migrated, "vllm_metax_default": native, "pytorch": pytorch},
            resets,
            expected,
        )

    def run_case(row):
        fns, resets, expected = build_case(row)
        graphs = {}
        outputs = {}
        valid = {}
        errors = {}
        noop = lambda: None
        for name, fn in fns.items():
            reset = resets.get(name, noop)
            for _ in range(3):
                reset()
                fn()
            reset()
            actual = fn()
            torch.cuda.synchronize()
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
            return dict(valid=valid, errors=errors, us={}, samples_us={})
        samples = {name: [] for name in fns}
        names = list(fns)
        for i in range(31):
            order = names[i % 3 :] + names[: i % 3]
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
            us={k: statistics.median(v) for k, v in samples.items()},
            samples_us=samples,
        )

    rows = json.loads(bench.SHAPES.read_text())
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
    print("DONE", flush=True)
    if any("error" in r or not all(r["valid"].values()) for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
