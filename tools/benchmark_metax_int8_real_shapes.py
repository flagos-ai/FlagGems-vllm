# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay captured shapes and complete per-request sequence lengths.

Read every recorded CUQ and USED value. max_seqlen_q/max_seqlen_k remain
bounds; they must never replace a request's actual length. Incomplete sequence
metadata is rejected instead of inventing a workload. Physical page mappings,
INT8 values and FP32 descales are synthetic, because tensor contents were not
captured. Both implementations use the full recorded physical pool and the
same page table; FA2 receives the same INT8 values dequantized to BF16.
This is an operator benchmark with exact captured sequence metadata, not an
end-to-end model replay or a replay of original tensor values.

Example (configure PYTHONPATH and an idle MetaX GPU externally):
  python tools/benchmark_metax_int8_real_shapes.py --shapes /path/shapes.json \
      --output /path/results --indices 0,2,7,13,23,129
"""
import argparse
import hashlib
import importlib
import json
import math
import statistics
import time
import traceback
from collections import Counter
from pathlib import Path


def read_captured_shape(entry):
    d = dict(zip(entry["labels"].split(","), entry["shape"]))
    batch = d["cu_seqlens_q_dim0"] - 1
    if (
        d.get("cu_seqlens_q_value_count") != batch + 1
        or d.get("seqused_k_value_count") != batch
    ):
        raise ValueError("Complete per-request CUQ and USED values are required")
    try:
        cuq = [d[f"cu_seqlens_q_value_{i}"] for i in range(batch + 1)]
        klens = [d[f"seqused_k_value_{i}"] for i in range(batch)]
    except KeyError as error:
        raise ValueError("Incomplete captured sequence metadata") from error
    qlens = [end - start for start, end in zip(cuq, cuq[1:])]
    assert cuq[0] == 0 and cuq[-1] == d["q_dim0"]
    assert all(0 <= q <= d["max_seqlen_q"] for q in qlens)
    assert all(0 <= k <= d["max_seqlen_k"] for k in klens)
    pages = [math.ceil(k / d["k_dim1"]) for k in klens]
    assert max(pages) <= d["block_table_dim1"]
    assert sum(pages) <= d["k_dim0"]
    return d, qlens, klens, pages


def build(entry, index, torch):
    d, qlens, klens, pages = read_captured_shape(entry)
    torch.manual_seed(20261015 + index)
    dev = "cuda"
    batch, heads, kvheads = len(qlens), d["q_dim1"], d["k_dim2"]
    page, dim, pool = d["k_dim1"], d["q_dim2"], d["k_dim0"]
    assert tuple(d[f"k_dim{i}"] for i in range(4)) == tuple(
        d[f"v_dim{i}"] for i in range(4)
    )
    q = torch.randint(
        -127, 128, (d["q_dim0"], heads, dim), device=dev, dtype=torch.int8
    )
    # Head-wise scales broadcast across logical blocks; block-varying scales
    # are independently covered by the repository correctness tests.
    qs = torch.rand(heads, device=dev) * 0.015 + 0.002
    ks = torch.rand(kvheads, device=dev) * 0.015 + 0.002
    vs = torch.rand(kvheads, device=dev) * 0.015 + 0.002
    shape = (pool, page, kvheads, dim)
    k, v = [torch.empty(shape, device=dev, dtype=torch.int8) for _ in range(2)]
    kb, vb = [torch.empty(shape, device=dev, dtype=torch.bfloat16) for _ in range(2)]
    physical = torch.randperm(pool, device=dev)[: sum(pages)]
    table = torch.zeros(
        (d["block_table_dim0"], d["block_table_dim1"]), dtype=torch.int32, device=dev
    )
    start = 0
    for i, n in enumerate(pages):
        table[i, :n] = physical[start : start + n].to(torch.int32)
        start += n
    # Only referenced pages are initialized. Padding and unused physical cache
    # slots must be masked by both operators; no whole-pool quantization cost.
    for lo in range(0, physical.numel(), 2048):
        ids = physical[lo : lo + 2048]
        for target, bf16, scales in ((k, kb, ks), (v, vb, vs)):
            data = torch.randint(
                -127,
                128,
                (ids.numel(), page, kvheads, dim),
                device=dev,
                dtype=torch.int8,
            )
            target[ids] = data
            bf16[ids] = (data.float() * scales[:, None]).to(torch.bfloat16)
    qb = (q.float() * qs[:, None]).to(torch.bfloat16)
    cuq = torch.tensor([0, *accumulate(qlens)], device=dev, dtype=torch.int32)
    cuk = torch.tensor([0, *accumulate(klens)], device=dev, dtype=torch.int32)
    used = torch.tensor(klens, device=dev, dtype=torch.int32)
    descales = [
        scale[None, :, None].expand(batch, scale.numel(), math.ceil(bound / 128))
        for scale, bound in (
            (qs, d["max_seqlen_q"]),
            (ks, d["max_seqlen_k"]),
            (vs, d["max_seqlen_k"]),
        )
    ]
    metadata = torch.zeros(d["scheduler_metadata_dim0"], device=dev, dtype=torch.int32)
    return dict(
        d=d,
        qlens=qlens,
        klens=klens,
        qkv=(q, k, v),
        bf16=(qb, kb, vb),
        cuq=cuq,
        cuk=cuk,
        used=used,
        table=table,
        descales=descales,
        metadata=metadata,
        active_pages=sum(pages),
        max_page=int(physical.max().item()) if physical.numel() else -1,
    )


def accumulate(values):
    total = 0
    for value in values:
        total += value
        yield total


def run(entry, index, args, torch, gems, fa2):
    c = build(entry, index, torch)
    d = c["d"]
    kwargs = dict(
        max_seqlen_q=d["max_seqlen_q"],
        max_seqlen_k=d["max_seqlen_k"],
        cu_seqlens_q=c["cuq"],
        seqused_k=c["used"],
        block_table=c["table"],
        causal=bool(d["causal"]),
        q_descale=c["descales"][0],
        k_descale=c["descales"][1],
        v_descale=c["descales"][2],
        num_splits=d["num_splits"],
        scheduler_metadata=c["metadata"],
    )

    def optimized():
        return gems.flash_attn_varlen_func(*c["qkv"], **kwargs)

    def baseline():
        return fa2(
            *c["bf16"],
            c["cuq"],
            c["cuk"],
            d["max_seqlen_q"],
            d["max_seqlen_k"],
            causal=bool(d["causal"]),
            block_table=c["table"],
        )

    actual, reference = optimized(), baseline()
    torch.cuda.synchronize()
    delta = (actual.float() - reference.float()).abs()
    bad = ~(delta <= 0.03 + 0.03 * reference.float().abs())
    check = dict(
        mismatch=int(bad.sum().item()),
        numel=actual.numel(),
        max_abs=float(delta.max().item()),
    )
    row = dict(
        index=index,
        count=entry["count"],
        status="pass" if check["mismatch"] == 0 else "accuracy_fail",
        klass=(
            "batch1"
            if len(c["qlens"]) == 1
            else "decode" if max(c["qlens"]) <= 1 else "mixed"
        ),
        recorded=d,
        qlens=c["qlens"],
        klens=c["klens"],
        sequence_metadata_exact=True,
        active_pages=c["active_pages"],
        max_physical_page=c["max_page"],
        max_page_element_offset=c["max_page"] * c["qkv"][1].stride(0),
        check=check,
    )
    del actual, reference, delta, bad
    if row["status"] != "pass":
        return row
    for _ in range(args.warmup):
        optimized()
        baseline()
    torch.cuda.synchronize()
    samples = {"int8_us": [], "fa2_us": []}
    for round_id in range(args.batches):
        funcs = [("int8_us", optimized), ("fa2_us", baseline)]
        if round_id % 2:
            funcs.reverse()
        for name, fn in funcs:
            if args.timing == "graph":
                import triton.testing

                samples[name].append(
                    triton.testing.do_bench_cudagraph(
                        fn, rep=args.graph_ms, return_mode="mean"
                    )
                    * 1000
                )
                continue
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(args.repeat):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.repeat)
    row["samples_us"] = samples
    aggregate = statistics.mean if args.timing == "graph" else statistics.median
    row.update({key: aggregate(value) for key, value in samples.items()})
    row["speedup"] = row["fa2_us"] / row["int8_us"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timing", choices=("event", "graph"), default="event")
    parser.add_argument("--graph-ms", type=int, default=100)
    args = parser.parse_args()
    if min(args.repeat, args.batches, args.graph_ms) <= 0 or args.warmup < 0:
        parser.error("repeat, batches and graph-ms must be positive; warmup >= 0")
    import torch
    import triton
    from vllm_metax.v1.attention.backends.fa_utils import flash_attn_varlen_func as fa2

    import flaggems_vllm as gems

    assert gems.vendor_name == "metax"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    shapes = json.loads(args.shapes.read_text())
    indices = (
        [int(x) for x in args.indices.split(",")]
        if args.indices
        else list(range(len(shapes)))
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / "results.jsonl"
    if result_file.exists():
        raise FileExistsError(f"Refusing to overwrite {result_file}")
    launcher = importlib.import_module(
        "flaggems_vllm.runtime.backend._metax.ops.flash_attention.launcher"
    )
    metadata = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        triton_file=triton.__file__,
        gpu=torch.cuda.get_device_name(),
        source=gems.__file__,
        launcher_sha256=hashlib.sha256(
            Path(launcher.__file__).read_bytes()
        ).hexdigest(),
        shapes_sha256=hashlib.sha256(args.shapes.read_bytes()).hexdigest(),
        reference_module=fa2.__module__,
        timing=(
            "alternating CUDA Graph measurements, mean"
            if args.timing == "graph"
            else "alternating CUDA Event batches, median"
        ),
        graph_ms=args.graph_ms if args.timing == "graph" else None,
        repeat=args.repeat,
        batches=args.batches,
        warmup=args.warmup,
        reconstruction=__doc__,
        source_count=len(shapes),
        selected_indices=indices,
    )
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    started = time.time()
    with torch.inference_mode():
        for pos, index in enumerate(indices):
            try:
                row = run(shapes[index], index, args, torch, gems, fa2)
            except Exception:
                row = dict(
                    index=index,
                    count=shapes[index]["count"],
                    status="error",
                    error=traceback.format_exc(),
                )
            rows.append(row)
            with result_file.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(
                f"[{pos + 1}/{len(indices)}] index={index} status={row['status']} "
                f"speedup={row.get('speedup')} check={row.get('check')}",
                flush=True,
            )
            torch.cuda.empty_cache()
            if row["status"] != "pass":
                print(row.get("error", "numerical mismatch"), flush=True)
                break
    summary = dict(
        statuses=dict(Counter(r["status"] for r in rows)),
        elapsed_s=time.time() - started,
        requested=len(indices),
        completed=len(rows),
        remaining=len(indices) - len(rows),
        groups={},
    )
    for group in ("all", "batch1", "mixed", "decode"):
        subset = [
            r
            for r in rows
            if r["status"] == "pass" and (group == "all" or r["klass"] == group)
        ]
        if not subset:
            continue
        total = {
            key: sum(r["count"] * r[key] for r in subset) / 1e6
            for key in ("int8_us", "fa2_us")
        }
        summary["groups"][group] = dict(
            cases=len(subset),
            calls=sum(r["count"] for r in subset),
            weighted_seconds=total,
            total_time_speedup=total["fa2_us"] / total["int8_us"],
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(rows) != len(indices) or any(r["status"] != "pass" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
