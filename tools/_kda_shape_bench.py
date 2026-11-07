"""Focused device-time bench for the Ascend chunk_kda / conv_prefill shapes.

Isolates the two slow operators from tools/test_kda_vs_vllm_ascend.py so block
and chunk parameters can be swept without re-running the whole 15-case harness.
Reports per-kernel device time via torch_npu.profiler, matching the harness.
"""

import glob
import json
import os
import shutil
import sys
import tempfile

import torch
import torch_npu  # noqa: F401

RCP_LN2 = 1.4426950408889634
HEAD_DIM = 128
ASCENDC_CHUNK_SIZE = 64
GATE_LOWER_BOUND = -5.0


def device_us(fn, warmup=5, active=5):
    """Mean per-iteration device kernel time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    out = tempfile.mkdtemp(prefix="kda_bench_")
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=active, repeat=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        ),
    )
    prof.start()
    for _ in range(active):
        fn()
        prof.step()
    torch.npu.synchronize()
    prof.stop()

    total, nk = 0.0, 0
    per_name = {}
    for path in glob.glob(
        os.path.join(out, "**", "kernel_details.csv"), recursive=True
    ):
        with open(path) as fh:
            head = fh.readline().strip().split(",")
            try:
                col = head.index("Duration(us)")
            except ValueError:
                continue
            ncol = head.index("Name") if "Name" in head else 0
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) > col:
                    try:
                        d = float(parts[col])
                    except ValueError:
                        continue
                    total += d
                    nk += 1
                    per_name[parts[ncol]] = per_name.get(parts[ncol], 0.0) + d
    shutil.rmtree(out, ignore_errors=True)
    if os.environ.get("BENCH_PERKERNEL"):
        for name, d in sorted(per_name.items(), key=lambda kv: -kv[1]):
            print(f"      {d / active:10.3f} us  {name}")
    return total / active, nk // max(active, 1)


def kda_inputs(heads, tokens, device, dtype, seed=1234):
    # Mirrors _kda_inputs in tools/test_kda_vs_vllm_ascend.py so the gate
    # magnitudes (and therefore the exp2 dynamic range) match the harness.
    torch.manual_seed(seed)
    q = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device)
    raw_g = -torch.rand(1, tokens, heads, HEAD_DIM, dtype=dtype, device=device) * 0.05
    raw_beta = torch.rand(1, tokens, heads, dtype=torch.float32, device=device)
    a_log = torch.randn(heads, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(heads * HEAD_DIM, dtype=torch.float32, device=device) * 0.1
    return q, k, v, raw_g, raw_beta, a_log, dt_bias


CHUNK_CASES = [
    (8192, 24, 2),
    (16384, 12, 4),
    (24576, 6, 8),
    (32768, 24, 2),
]


def bench_chunk(only=None):
    from flaggems_vllm.ops.FLA.index import prepare_chunk_indices
    from flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda import chunk_kda

    bt_probe = int(os.environ.get("FLAGGEMS_CHUNK_SIZE", "16"))
    if bt_probe != 16:
        # The shared validator pins chunk_size=16. Relax it here only, so BT
        # can be measured without editing the cross-backend 公版 file.
        # `flaggems_vllm.ops.FLA.chunk_kda` the attribute is the re-exported
        # function, so reach the module through sys.modules.
        _shared = sys.modules["flaggems_vllm.ops.FLA.chunk_kda"]
        _asc = sys.modules["flaggems_vllm.runtime.backend._ascend.FLA.chunk_kda"]

        _orig = _shared._validate_chunk_kda_inputs

        def _patched(*a, **kw):
            kw["chunk_size"] = 16
            return _orig(*a, **kw)

        _shared._validate_chunk_kda_inputs = _patched
        _asc._validate_chunk_kda_inputs = _patched

    dev = torch.device("npu:0")
    dtype = torch.bfloat16
    rows = []
    for idx, (tokens, heads, nseq) in enumerate(CHUNK_CASES):
        if only is not None and idx not in only:
            continue
        per = (tokens // nseq // ASCENDC_CHUNK_SIZE) * ASCENDC_CHUNK_SIZE
        cu_list = [j * per for j in range(nseq)] + [tokens]
        cu = torch.tensor(cu_list, dtype=torch.int64, device=dev)
        q, k, v, raw_g, raw_beta, a_log, dt_bias = kda_inputs(heads, tokens, dev, dtype)
        init_vk = torch.randn(
            nseq, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=dev
        )
        bt = int(os.environ.get("FLAGGEMS_CHUNK_SIZE", "16"))
        ci = prepare_chunk_indices(cu, bt)

        beta_c = raw_beta.to(q.dtype)

        def step():
            return chunk_kda(
                q,
                k,
                v,
                raw_g,
                beta_c,
                scale=HEAD_DIM**-0.5,
                initial_state=init_vk.clone(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=GATE_LOWER_BOUND,
                state_v_first=True,
                cu_seqlens=cu,
                chunk_indices=ci,
                chunk_size=bt,
                A_log=a_log,
                dt_bias=dt_bias,
            )

        us, nk = device_us(step)
        rows.append((f"T={tokens} H={heads} seq={nseq}", us, nk))
    return rows


CONV_CASES = [
    (8192, 24, 2),
    (16384, 12, 4),
    (24576, 6, 8),
    (32768, 24, 2),
]


def bench_conv(only=None):
    from flaggems_vllm.runtime.backend._ascend.ops.causal_conv1d_fn import (
        causal_conv1d_fn,
    )

    dev = torch.device("npu:0")
    dtype = torch.bfloat16
    width = 4
    rows = []
    for idx, (tokens, heads, nseq) in enumerate(CONV_CASES):
        if only is not None and idx not in only:
            continue
        # Mirror the harness layouts: AscendC-shaped tensors with transposed
        # views handed to FlagGems, so no copy and identical memory traffic.
        dim = 3 * heads * HEAD_DIM
        per = (tokens // nseq // ASCENDC_CHUNK_SIZE) * ASCENDC_CHUNK_SIZE
        cu_list = [j * per for j in range(nseq)] + [tokens]
        torch.manual_seed(77)
        x = torch.randn(tokens, dim, dtype=dtype, device=dev)
        qsl = torch.tensor(cu_list, dtype=torch.int32, device=dev)
        weight_t = torch.randn(width, dim, dtype=dtype, device=dev) * 0.1
        pool = max(2 * nseq + 2, 64)
        state = torch.randn(pool, width - 1, dim, dtype=dtype, device=dev)
        cidx = torch.arange(1, nseq + 1, dtype=torch.int32, device=dev)
        has_init = torch.ones(nseq, dtype=torch.bool, device=dev)

        gx = x.transpose(0, 1)
        gs = state.transpose(1, 2)
        gw = weight_t.transpose(0, 1)

        def step():
            return causal_conv1d_fn(
                gx,
                gw,
                None,
                conv_states=gs,
                query_start_loc=qsl,
                cache_indices=cidx,
                has_initial_state=has_init,
                activation="silu",
                pad_slot_id=-1,
            )

        us, nk = device_us(step)
        rows.append((f"T={tokens} dim={dim} seq={nseq}", us, nk))
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "chunk"
    only = None
    if len(sys.argv) > 2:
        only = {int(x) for x in sys.argv[2].split(",")}
    with torch.inference_mode():
        rows = bench_chunk(only) if which == "chunk" else bench_conv(only)
    tag = os.environ.get("BENCH_TAG", "")
    print(f"=== {which} {tag}")
    for name, us, nk in rows:
        print(f"{name:34s} {us:10.3f} us  k={nk}")
    print(
        json.dumps(
            {"which": which, "tag": tag, "rows": [[n, u, k] for n, u, k in rows]}
        )
    )
