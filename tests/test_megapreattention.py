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

"""Platform-specific MegaPreAttention validation and benchmark."""

import flaggems_vllm

if flaggems_vllm.vendor_name == "hygon":
    """Independent torch oracle and Hygon layer3 test adapter (test-only).

Oracle is derived from frozen NV PR #1007; it gathers original hidden shards
and never reads candidate communication buffers or intermediate projections.
"""
    import hashlib
    import json
    import os
    import statistics
    import tempfile
    import time
    from dataclasses import dataclass, replace
    from datetime import timedelta
    from functools import partial
    from pathlib import Path
    from types import SimpleNamespace

    import pytest
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    import flaggems_vllm

    if flaggems_vllm.vendor_name != "hygon":
        pytest.skip("MegaPreAttention tests require Hygon", allow_module_level=True)

    pytestmark = pytest.mark.megapreattention

    BF16_ATOLS = (0.001, 0.002, 0.003, 0.004, 0.005)
    BF16_RTOL = 0.016


    @dataclass(frozen=True)
    class Config:
        num_q_heads: int
        num_kv_heads: int
        head_dim: int = 256
        rotary_dim: int = 64
        mrope_section: tuple[int, int, int] = (11, 11, 10)
        eps: float = 1e-6
        global_kv_heads: int = 2

        @property
        def q_size(self):
            return self.num_q_heads * self.head_dim

        @property
        def kv_size(self):
            return self.num_kv_heads * self.head_dim

        @property
        def packed_size(self):
            return 2 * (self.q_size + self.kv_size)


    @dataclass
    class Inputs:
        hidden: torch.Tensor
        weight: torch.Tensor
        positions: torch.Tensor
        q_norm: torch.Tensor
        k_norm: torch.Tensor
        rope: torch.Tensor
        cache: torch.Tensor
        slots: torch.Tensor
        scale: torch.Tensor
        group: object
        config: Config


    @dataclass
    class Output:
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor
        gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.
        workspace: tuple = ()

        def tensors(self):
            return self.query, self.key, self.value, self.gate


    def file_sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def tensor_sha(tensor):
        return hashlib.sha256(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()


    def tensor_info(tensor):
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "sha256": tensor_sha(tensor),
        }


    def packed_weight(weights, dims, tp, rank):
        _, q_heads, kv_heads = dims
        qwidth = q_heads * 256 * 2 // tp
        kvwidth = max(1, kv_heads // tp) * 256
        kv_rank = rank // max(1, tp // kv_heads)
        kv_start = kv_rank * kvwidth
        return torch.cat(
            [
                weights["q_proj.weight"][rank * qwidth : (rank + 1) * qwidth],
                weights["k_proj.weight"][kv_start : kv_start + kvwidth],
                weights["v_proj.weight"][kv_start : kv_start + kvwidth],
            ]
        )


    class Dataset:
        """Portable replay loader. Load/allocate before warmup or graph capture."""

        def __init__(self, root):
            self.root = Path(root)
            self.manifest = json.loads((self.root / "manifest.json").read_text())
            if (
                file_sha(self.root / "extraction.json")
                != self.manifest["extraction_sha256"]
            ):
                raise ValueError("extraction.json hash mismatch")
            self.cases = {case["id"]: case for case in self.manifest["cases"]}
            self._checked = set()
            self._fixtures = {}
            self._shared = {}

        def _load(self, relative):
            path = self.root / relative
            info = self.manifest["files"][relative]
            if relative not in self._checked:
                if path.stat().st_size != info["bytes"] or file_sha(path) != info["sha256"]:
                    raise ValueError(f"File hash mismatch: {relative}")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if relative not in self._checked:
                key = "weights" if "weights" in value else "tensors"
                for name, tensor in value[key].items():
                    if tensor_info(tensor) != info[key][name]:
                        raise ValueError(f"Tensor mismatch: {relative}:{name}")
                self._checked.add(relative)
            return value

        def load_rank(self, case_id, rank, device="cpu", group=None):
            case = self.cases[case_id]
            if not 0 <= rank < case["tp"]:
                raise ValueError("Rank outside case TP group")
            model = case["model"]
            model_info = self.manifest["models"][model]
            if model not in self._fixtures:
                self._fixtures[model] = self._load(model_info["fixture"])
            weights = self._fixtures[model]["weights"]
            rank_info = case["ranks"][rank]
            hidden = self._load(rank_info["inputs"])["tensors"]["hidden"]
            weight = packed_weight(weights, model_info["model_dims"], case["tp"], rank)
            if tensor_info(weight) != rank_info["packed_weight"]:
                raise ValueError("Packed weight mismatch")
            for name in (case["shared_inputs"], self.manifest["config"]["rope_file"]):
                if name not in self._shared:
                    self._shared[name] = self._load(name)["tensors"]
            shared = self._shared[case["shared_inputs"]]
            config = Config(
                case["local_q_heads"],
                case["local_kv_heads"],
                global_kv_heads=model_info["model_dims"][2],
            )
            return SimpleNamespace(
                hidden=hidden.to(device),
                weight=weight.to(device),
                positions=shared["positions"].to(device),
                slots=shared["slots"].to(device),
                q_norm=weights["q_norm.weight"].to(device),
                k_norm=weights["k_norm.weight"].to(device),
                rope=self._shared[self.manifest["config"]["rope_file"]]["rope"].to(device),
                cache=torch.full(
                    case["cache_shape"],
                    case["cache_initial_value"],
                    dtype=torch.bfloat16,
                    device=device,
                ),
                scale=torch.ones(1, device=device),
                group=group,
                config=config,
            )


    def independent_reference(inputs):
        """Oracle starts at original shards; never reads candidate scratch/results."""
        c, t = inputs.config, inputs.positions.shape[1]
        hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
        dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
        qkv = hidden @ inputs.weight.t()
        qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
        query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
        frequency = torch.arange(c.rotary_dim // 2, device="cuda")
        axis = torch.zeros_like(frequency)
        axis[1 : 3 * c.mrope_section[1] : 3] = 1
        axis[2 : 3 * c.mrope_section[2] : 3] = 2
        position = inputs.positions[axis].t()
        cos = inputs.rope[position, frequency].float().unsqueeze(1)
        sin = inputs.rope[position, frequency + c.rotary_dim // 2].float().unsqueeze(1)

        def norm_rope(x, weight):
            x = x.float()
            normalized = (
                (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                    * (weight.float() + 1)
                )
                .to(torch.bfloat16)
                .float()
            )
            first, second = normalized[..., : c.rotary_dim].chunk(2, dim=-1)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
            return torch.cat((rotated, normalized[..., c.rotary_dim :]), dim=-1).to(
                torch.bfloat16
            )

        query = norm_rope(query, inputs.q_norm).reshape(t, c.q_size)
        key = norm_rope(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
        valid = inputs.slots >= 0
        slots = inputs.slots[valid]
        block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
        inputs.cache[block, :, offset, : c.head_dim] = key[valid]
        inputs.cache[block, :, offset, c.head_dim :] = value.view(
            t, c.num_kv_heads, c.head_dim
        )[valid]
        return Output(query, key.reshape(t, c.kv_size), value, gate)


    def changed(x, **kw):
        return SimpleNamespace(**{**vars(x), **kw})


    def implementation():
        from flaggems_vllm.runtime.backend._hygon.mega.megapreattention import api

        return api


    def candidate(x):
        c = x.config
        return Output(
            *implementation().megapreattention(
                x.hidden,
                x.weight,
                x.positions,
                x.q_norm,
                x.k_norm,
                x.rope,
                x.cache,
                x.slots,
                x.group,
                num_heads=c.num_q_heads * x.group.size(),
                num_kv_heads=c.global_kv_heads,
                head_dim=c.head_dim,
                rotary_dim=c.rotary_dim,
                mrope_section=c.mrope_section,
                eps=c.eps,
            )
        )


    def check_outputs(actual, expected, cache, reference_cache):
        errors = []
        for name, a, b in zip(
            ("query", "key", "value", "gate", "cache"),
            (*actual.tensors(), cache),
            (*expected.tensors(), reference_cache),
        ):
            if not torch.isfinite(a).all():
                raise AssertionError(f"{name}: nonfinite output")
            error = (a.float() - b.float()).abs()
            required_atol = (error - BF16_RTOL * b.float().abs()).amax().clamp_min(0).item()
            errors.append(
                {
                    "name": name,
                    "required_atol": required_atol,
                    "atol_passed": {
                        str(atol): required_atol <= atol for atol in BF16_ATOLS
                    },
                    "max_abs": error.max().item(),
                    "relative_rms": (
                        error.square().mean().sqrt()
                        / b.float().square().mean().sqrt().clamp_min(1e-12)
                    ).item(),
                }
            )
            torch.testing.assert_close(
                a, b, atol=BF16_ATOLS[-1], rtol=BF16_RTOL, msg=lambda msg: f"{name}: {msg}"
            )
        return errors


    def edge_inputs(x):
        t = x.positions.shape[1]
        pos = torch.stack(
            (
                torch.arange(t, device=x.hidden.device),
                torch.arange(t, device=x.hidden.device) * 2,
                torch.arange(t, device=x.hidden.device) * 3,
            )
        )
        backing = torch.empty((3, 2 * t), dtype=torch.int64, device=x.hidden.device)
        backing[:, ::2] = pos
        slots = torch.empty(2 * t, device=x.hidden.device, dtype=torch.int64)
        slots[::2] = torch.arange(t - 1, -1, -1, device=x.hidden.device)
        slots[::34] = -1
        b, h, n, d = x.cache.shape
        cache = torch.full(
            (n, h, b, d), 0.375, device=x.hidden.device, dtype=torch.bfloat16
        ).permute(2, 1, 0, 3)
        weight_storage = torch.empty(
            x.weight.numel() + 1, device=x.hidden.device, dtype=x.weight.dtype
        )
        weight = weight_storage[1:].view(x.weight.shape)
        weight.copy_(x.weight)
        hidden_storage = torch.empty(
            x.hidden.numel() + 1, device=x.hidden.device, dtype=x.hidden.dtype
        )
        hidden = hidden_storage[1:].view(x.hidden.shape)
        hidden.copy_(x.hidden)
        return changed(
            x,
            positions=backing[:, ::2],
            slots=slots[::2],
            cache=cache,
            hidden=hidden,
            weight=weight,
        )


    def check_contract(x):
        # These all fail before communicator creation/kernel launch.
        cases = {
            "fp32": changed(x, hidden=x.hidden.float()),
            "requires_grad": changed(x, weight=x.weight.clone().requires_grad_(True)),
            "strided_hidden": changed(
                x,
                hidden=torch.empty(
                    (x.hidden.shape[0], x.hidden.shape[1] * 2),
                    device=x.hidden.device,
                    dtype=x.hidden.dtype,
                )[:, ::2],
            ),
            "empty": changed(
                x, hidden=x.hidden[:0], positions=x.positions[:, :0], slots=x.slots[:0]
            ),
            "overlap_cache": changed(
                x, cache=x.cache.as_strided(x.cache.shape, (0, *x.cache.stride()[1:]))
            ),
            "bad_weight": changed(x, weight=x.weight[:-1]),
        }
        for label, bad in cases.items():
            try:
                candidate(bad)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError(f"accepted unsupported input: {label}")
        return list(cases)


    def distribution(samples):
        v = sorted(samples)

        def percentile(q):
            index = (len(v) - 1) * q
            i = int(index)
            return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (index - i)

        median = statistics.median(samples)
        span = (percentile(0.9) - percentile(0.1)) / median
        return dict(
            mean_ms=statistics.mean(samples),
            median_ms=median,
            span=span,
            stable=len(v) >= 10 and span <= 0.05,
        )


    def summarize_samples(ranks, names, iterations):
        """Average the slowest rank of each paired iteration."""
        metrics = {}
        for name in names:
            means = [statistics.mean(r["samples_ms"][name]) for r in ranks]
            rank_max = [
                max(r["samples_ms"][name][i] for r in ranks) for i in range(iterations)
            ]
            metrics[name] = dict(
                latency_ms=statistics.mean(rank_max),
                slowest_rank_mean_ms=max(means),
                rank_mean_ms=means,
                rank_max_distribution=distribution(rank_max),
            )
        return metrics


    def summarize_accuracy(reports):
        summary = {}
        for rows in reports:
            for row in rows:
                for rank in row["ranks"]:
                    for phase, errors in rank["errors"].items():
                        implementation = phase.split(":", 1)[0]
                        for error in errors:
                            key = f"{implementation}:{error['name']}"
                            value = summary.setdefault(
                                key,
                                {
                                    "max_abs": 0.0,
                                    "required_atol": 0.0,
                                    "atol_passed": {str(atol): True for atol in BF16_ATOLS},
                                },
                            )
                            for metric in ("max_abs", "required_atol"):
                                value[metric] = max(value[metric], error[metric])
                            for atol, passed in error["atol_passed"].items():
                                value["atol_passed"][atol] &= passed
        return summary


    def measure(calls, iterations, mode):
        from torch._dynamo.utils import counters
        from triton.runtime.jit import JITFunction

        original = JITFunction._do_compile
        compiled = []

        def tracked(*args, **kwargs):
            compiled.append(True)
            return original(*args, **kwargs)

        before = dict(counters["stats"])
        JITFunction._do_compile = tracked
        samples = {name: [] for name in calls}
        try:
            names = list(calls)
            for i in range(iterations):
                for name in names[i % len(names) :] + names[: i % len(names)]:
                    dist.barrier()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin = time.perf_counter_ns()
                    start.record()
                    calls[name]()
                    end.record()
                    end.synchronize()
                    elapsed = (
                        (time.perf_counter_ns() - begin) / 1e6
                        if mode == "operator"
                        else start.elapsed_time(end)
                    )
                    samples[name].append(elapsed)
        finally:
            JITFunction._do_compile = original
        assert not compiled, "measurement triggered Triton JIT"
        assert before == dict(counters["stats"]), "measurement triggered Dynamo compilation"
        return samples


    def _check_growth(x):
        impl = implementation()
        tp = x.group.size()
        small_t = 128 * tp
        small = changed(
            x,
            hidden=x.hidden[:128],
            positions=x.positions[:, :small_t],
            slots=x.slots[:small_t],
            cache=torch.full_like(x.cache, 0.375),
        )
        candidate(small)
        candidate(x)
        assert not impl._retired
        impl.clear_megapreattention_cache()
        candidate(small)
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = candidate(small)
        # Creating the larger generation must preserve the captured smaller one.
        candidate(x)
        dist.barrier()
        torch.cuda.synchronize()
        oracle_cache = torch.full_like(small.cache, 0.375)
        expected = independent_reference(changed(small, cache=oracle_cache))
        small.cache.fill_(0.375)
        graph.replay()
        torch.cuda.synchronize()
        check_outputs(out, expected, small.cache, oracle_cache)
        graph.reset()
        assert len(impl._retired) == 1
        irregular_t = 384 * tp
        irregular = changed(
            x,
            hidden=x.hidden[:384],
            positions=x.positions[:, :irregular_t],
            slots=x.slots[:irregular_t],
            cache=torch.full_like(x.cache, 0.375),
        )
        ref_cache = torch.full_like(x.cache, 0.375)
        expected = independent_reference(changed(irregular, cache=ref_cache))
        out = candidate(irregular)
        torch.cuda.synchronize()
        check_outputs(out, expected, irregular.cache, ref_cache)


    def _worker(
        rank,
        tp,
        directory,
        root,
        case_ids,
        mode,
        warmup,
        iterations,
        check_only,
        edge,
        rope_parameters,
        check_baseline=False,
    ):
        import faulthandler

        torch.set_num_threads(4)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=tp,
            init_method=f"file://{directory}/rdzv",
            device_id=torch.device("cuda", rank),
            timeout=timedelta(seconds=180),
        )
        data = Dataset(root)
        reports = []
        impl = implementation()
        try:
            for case_id in case_ids:
                faulthandler.dump_traceback_later(600, exit=True)
                x = data.load_rank(case_id, rank, f"cuda:{rank}", dist.group.WORLD)
                print(rank, case_id, "loaded", flush=True)
                if rope_parameters:
                    x = changed(x, config=replace(x.config, **rope_parameters))
                if edge:
                    x = edge_inputs(x)
                oracle_cache = torch.full_like(x.cache, 0.375)
                expected = independent_reference(changed(x, cache=oracle_cache))
                print(rank, "oracle ready", flush=True)
                if check_only:
                    with torch.cuda.device((rank + 1) % tp):
                        actual = candidate(x)
                        assert torch.cuda.current_device() == (rank + 1) % tp
                else:
                    actual = candidate(x)
                torch.cuda.synchronize()
                print(rank, "candidate ready", flush=True)
                errors = {
                    "megapreattention:initial": check_outputs(
                        actual, expected, x.cache, oracle_cache
                    )
                }
                functions = {"megapreattention": candidate}
                sources = {"megapreattention": x}
                baseline_meta = None
                native = None
                if not check_only or check_baseline:
                    from benchmark.test_megapreattention import NativeWindow

                    native = NativeWindow(x)
                    sources["native"] = changed(x, cache=torch.full_like(x.cache, 0.375))
                    functions["native"] = native
                    baseline_meta = native.evidence
                    result = native(sources["native"])
                    torch.cuda.synchronize()
                    errors["native:initial"] = check_outputs(
                        result,
                        expected,
                        native.cache_for_check(sources["native"]),
                        oracle_cache,
                    )
                dist.barrier()
                for _ in range(warmup):
                    for name, f in functions.items():
                        f(sources[name])
                dist.barrier()
                torch.cuda.synchronize()
                print(rank, "warmup ready", flush=True)
                graphs, outputs, calls = {}, {}, {}
                if mode == "cudagraph":
                    for name, f in functions.items():
                        dist.barrier()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs[name] = f(sources[name])
                        graphs[name] = graph
                        calls[name] = graph.replay
                else:
                    calls = {
                        name: partial(functions[name], sources[name]) for name in functions
                    }
                print(rank, "capture ready", flush=True)
                saved_hidden = x.hidden.clone()
                for factor in (-0.75, -1.25):
                    x.hidden.mul_(factor)
                    oracle_cache.fill_(0.375)
                    expected = independent_reference(changed(x, cache=oracle_cache))
                    for w in impl._windows.values():
                        w.buffer.fill_(float("nan"))
                    for name in functions:
                        sources[name].cache.fill_(0.375)
                        if name == "native":
                            native.reset_cache(sources[name])
                        if mode == "cudagraph":
                            calls[name]()
                            result = outputs[name]
                        else:
                            result = functions[name](sources[name])
                        torch.cuda.synchronize()
                        errors[f"{name}:hidden_scale={factor}"] = check_outputs(
                            result,
                            expected,
                            (
                                native.cache_for_check(sources[name])
                                if name == "native"
                                else sources[name].cache
                            ),
                            oracle_cache,
                        )
                x.hidden.copy_(saved_hidden)
                del saved_hidden
                print(rank, "changed input ready", flush=True)
                rejected = check_contract(x) if check_only else []
                samples = {} if check_only else measure(calls, iterations, mode)
                record = {
                    "rank": rank,
                    "errors": errors,
                    "rejected": rejected,
                    "samples_ms": samples,
                    "resources": [w.resources for w in impl._windows.values()],
                }
                ranks = [None] * tp
                dist.all_gather_object(ranks, record)
                metrics = (
                    {} if check_only else summarize_samples(ranks, functions, iterations)
                )
                reports.append(
                    dict(
                        case=case_id,
                        correctness_passed=True,
                        measurement_jit_free=None if check_only else True,
                        mode=mode,
                        timing="wall_ms" if mode == "operator" else "device_ms",
                        rank_aggregation="mean_iteration_rank_max",
                        accuracy=dict(
                            atol=BF16_ATOLS[-1], rtol=BF16_RTOL, profile_atols=BF16_ATOLS
                        ),
                        edge=edge,
                        baseline=baseline_meta,
                        metrics=metrics,
                        ranks=ranks,
                    )
                )
                for graph in graphs.values():
                    graph.reset()
                outputs.clear()
                graphs.clear()
                calls.clear()
                torch.cuda.synchronize()
                dist.barrier()
                if native is not None:
                    native.close()
                impl.clear_megapreattention_cache()
                if check_only and case_id.endswith("_t4096"):
                    _check_growth(x)
                    impl.clear_megapreattention_cache()
                if rank == 0:
                    print(
                        json.dumps(
                            {"case": case_id, "correctness": "pass", "metrics": metrics}
                        ),
                        flush=True,
                    )
                del x, sources, functions, expected, actual
            if not check_only or check_baseline:
                from benchmark.test_megapreattention import close_vendor_session

                close_vendor_session()
            if rank == 0:
                (Path(directory) / "report.json").write_text(json.dumps(reports, indent=2))
        finally:
            faulthandler.cancel_dump_traceback_later()
            dist.destroy_process_group()


    def run_distributed(
        tp,
        case_ids,
        *,
        root=None,
        mode="cudagraph",
        warmup=3,
        iterations=10,
        rounds=1,
        check_only=True,
        edge=False,
        rope_parameters=None,
        check_baseline=False,
    ):
        root = root or os.environ.get(
            "MEGAPREATTENTION_DATA",
            "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
        )
        if check_baseline and (not check_only or edge):
            raise ValueError(
                "baseline preflight requires check_only and the manifest layout"
            )
        if edge and not check_only:
            raise ValueError(
                "edge layouts are accuracy-only; benchmark uses the manifest layout"
            )
        reports = []
        for _ in range(rounds):
            with tempfile.TemporaryDirectory(prefix="megapreattention-hy-") as d:
                context = mp.spawn(
                    _worker,
                    args=(
                        tp,
                        d,
                        root,
                        list(case_ids),
                        mode,
                        warmup,
                        iterations,
                        check_only,
                        edge,
                        rope_parameters,
                        check_baseline,
                    ),
                    nprocs=tp,
                    join=False,
                )
                deadline = time.monotonic() + max(600, 600 * len(case_ids))
                try:
                    while not context.join(timeout=1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("MegaPreAttention workers exceeded timeout")
                    reports.append(json.loads((Path(d) / "report.json").read_text()))
                finally:
                    for p in context.processes:
                        if p.is_alive():
                            p.terminate()
                    for p in context.processes:
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()
                            p.join(timeout=5)
        summary = {}
        if not check_only:
            for i, case_id in enumerate(case_ids):
                values = {
                    name: statistics.mean(
                        r[i]["metrics"][name]["latency_ms"] for r in reports
                    )
                    for name in ("native", "megapreattention")
                }
                values["speedup"] = values["native"] / values["megapreattention"]
                values["stable_all_rounds"] = all(
                    r[i]["metrics"][n]["rank_max_distribution"]["stable"]
                    for r in reports
                    for n in ("native", "megapreattention")
                )
                summary[case_id] = values
        if check_only:
            print(
                json.dumps(
                    {
                        "accuracy_summary": summarize_accuracy(reports),
                        "tp": tp,
                        "edge": edge,
                    }
                ),
                flush=True,
            )
        return dict(
            correctness_passed=True,
            tp=tp,
            rounds=rounds,
            rank_aggregation="mean_iteration_rank_max",
            summary=summary,
            reports=reports,
        )


    def pytest_generate_tests(metafunc):
        if "tp" in metafunc.fixturenames:
            metafunc.parametrize(
                "tp", [2] if metafunc.config.getoption("--quick") else [2, 4]
            )


    def require_hygon(tp):
        if not torch.version.hip or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} Hygon GPUs")
        root = os.environ.get(
            "MEGAPREATTENTION_DATA",
            "/data/lixichen/megapreattention_data/layer3_nv_20261027_v1",
        )
        if not Path(root, "manifest.json").exists():
            pytest.skip("layer3 dataset unavailable")
        return root


    def test_megapreattention_exports():
        import flaggems_vllm
        from flaggems_vllm.ops import clear_megapreattention_cache, megapreattention

        assert callable(flaggems_vllm.megapreattention)
        assert callable(flaggems_vllm.clear_megapreattention_cache)
        assert callable(megapreattention) and callable(clear_megapreattention_cache)


    @pytest.mark.parametrize("edge", [False, True])
    def test_megapreattention(tp, edge, request):
        root = require_hygon(tp)
        tokens = (
            (4096,)
            if request.config.getoption("--quick") or edge
            else (4096, 8192, 16384, 32768)
        )
        cases = [
            f"{m}_tp{tp}_t{t}" for m in ("Qwen3.6-35B-A3B", "Qwen3.6-27B") for t in tokens
        ]
        result = run_distributed(
            tp,
            cases,
            root=root,
            edge=edge,
            check_baseline=not edge,
            mode="cudagraph",
            check_only=True,
            rounds=1,
            warmup=3,
        )
        assert result["correctness_passed"]


    @pytest.mark.parametrize("sections,eps", [((16, 8, 8), 1e-5), ((8, 12, 12), 1e-4)])
    def test_megapreattention_rope_parameters(sections, eps):
        root = require_hygon(2)
        result = run_distributed(
            2,
            ["Qwen3.6-35B-A3B_tp2_t4096"],
            root=root,
            edge=True,
            check_only=True,
            rounds=1,
            warmup=2,
            rope_parameters=dict(mrope_section=sections, eps=eps),
        )
        assert result["correctness_passed"]


    def test_megapreattention_rank_aggregation():
        ranks = [
            {"samples_ms": {"native": [1.0, 3.0], "megapreattention": [2.0, 2.0]}},
            {"samples_ms": {"native": [3.0, 1.0], "megapreattention": [2.0, 2.0]}},
        ]
        metrics = summarize_samples(ranks, ("native", "megapreattention"), 2)
        assert metrics["native"]["latency_ms"] == 3.0
        assert metrics["native"]["slowest_rank_mean_ms"] == 2.0
        assert metrics["megapreattention"]["latency_ms"] == 2.0


    def test_megapreattention_accuracy_tolerances():
        reference = torch.zeros(4, dtype=torch.bfloat16)
        small_error = torch.full_like(reference, 0.00390625)
        errors = check_outputs(
            Output(*([small_error] * 4)),
            Output(*([reference] * 4)),
            small_error,
            reference,
        )
        assert all(
            value["atol_passed"]
            == {
                "0.001": False,
                "0.002": False,
                "0.003": False,
                "0.004": True,
                "0.005": True,
            }
            for value in errors
        )
        large_error = torch.full_like(reference, 0.01)
        with pytest.raises(AssertionError):
            check_outputs(
                Output(*([large_error] * 4)),
                Output(*([reference] * 4)),
                large_error,
                reference,
            )


    def test_megapreattention_address_offsets():
        if not torch.version.hip or not torch.cuda.is_available():
            pytest.skip("requires a Hygon GPU")
        import triton
        import triton.language as tl

        from flaggems_vllm.runtime.backend._hygon.mega.megapreattention.megapreattention_kernel import (
            _row_offset as row_offset,
        )

        @triton.jit
        def offsets_kernel(rows, strides, columns, output, BLOCK: tl.constexpr):
            i = tl.arange(0, BLOCK)
            row = tl.load(rows + i)
            stride = tl.load(strides + i)
            column = tl.load(columns + i)
            tl.store(output + i, row_offset(row, stride, column))

        rows = torch.tensor(
            [0, 1, 8191, 8192, 32767, 1048575, 2147483647, -1],
            dtype=torch.int32,
            device="cuda",
        )
        strides = torch.tensor(
            [327680, 327680, 327680, 327680, 655360, 5120, 5120, 5120],
            dtype=torch.int32,
            device="cuda",
        )
        columns = torch.tensor(
            [0, 1023, 1023, 0, 511, 255, 1023, 7],
            dtype=torch.int32,
            device="cuda",
        )
        actual = torch.empty(8, dtype=torch.int64, device="cuda")
        offsets_kernel[(1,)](rows, strides, columns, actual, BLOCK=8)
        expected = rows.long() * strides.long() + columns.long()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

else:
    """Distributed accuracy, Graph replay and shared synthetic benchmark inputs."""

    import json
    import tempfile
    import time
    from dataclasses import dataclass, replace
    from datetime import timedelta
    from pathlib import Path

    import pytest
    import torch
    import torch.distributed as dist
    import triton
    import triton.language as tl

    from . import conftest as cfg
    from .accuracy_utils import gems_assert_close

    pytestmark = pytest.mark.megapreattention


    @dataclass(frozen=True)
    class Config:
        num_q_heads: int
        num_kv_heads: int
        head_dim: int = 256
        rotary_dim: int = 64
        mrope_section: tuple[int, int, int] = (11, 11, 10)
        eps: float = 1e-6
        global_kv_heads: int = 2

        @property
        def q_size(self):
            return self.num_q_heads * self.head_dim

        @property
        def kv_size(self):
            return self.num_kv_heads * self.head_dim

        @property
        def packed_size(self):
            return 2 * (self.q_size + self.kv_size)


    @dataclass
    class Inputs:
        hidden: torch.Tensor
        weight: torch.Tensor
        positions: torch.Tensor
        q_norm: torch.Tensor
        k_norm: torch.Tensor
        rope: torch.Tensor
        cache: torch.Tensor
        slots: torch.Tensor
        scale: torch.Tensor
        group: object
        config: Config


    @dataclass
    class Output:
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor
        gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.
        workspace: tuple = ()

        def tensors(self):
            return self.query, self.key, self.value, self.gate


    def make_synthetic_weights(model_dims, head_dim):
        hidden_size, q_heads, kv_heads = model_dims
        generator = torch.Generator().manual_seed(20261010)
        return {
            "q_proj.weight": torch.randn(
                2 * q_heads * head_dim,
                hidden_size,
                generator=generator,
                dtype=torch.bfloat16,
            )
            * 0.02,
            "k_proj.weight": torch.randn(
                kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
            )
            * 0.02,
            "v_proj.weight": torch.randn(
                kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
            )
            * 0.02,
            "q_norm.weight": torch.randn(
                head_dim, generator=generator, dtype=torch.bfloat16
            )
            * 0.1,
            "k_norm.weight": torch.randn(
                head_dim, generator=generator, dtype=torch.bfloat16
            )
            * 0.1,
        }


    def run_candidate(inputs):
        from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
            megapreattention,
        )

        c = inputs.config
        return Output(
            *megapreattention(
                inputs.hidden,
                inputs.weight,
                inputs.positions,
                inputs.q_norm,
                inputs.k_norm,
                inputs.rope,
                inputs.cache,
                inputs.slots,
                inputs.group,
                num_heads=c.num_q_heads * inputs.group.size(),
                num_kv_heads=c.global_kv_heads,
                head_dim=c.head_dim,
                rotary_dim=c.rotary_dim,
                mrope_section=c.mrope_section,
                eps=c.eps,
            )
        )


    def make_inputs(
        tokens,
        tp,
        rank,
        *,
        edge=False,
        model_dims=(2048, 16, 2),
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    ):
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

        hidden_size, q_heads, kv_heads = model_dims
        config = Config(
            q_heads // tp,
            max(1, kv_heads // tp),
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            mrope_section=tuple(mrope_section),
            eps=eps,
            global_kv_heads=kv_heads,
        )
        weights = make_synthetic_weights(model_dims, head_dim)
        qwidth = q_heads * head_dim * 2 // tp
        kv_rank = rank // max(1, tp // kv_heads)
        kv_start = kv_rank * config.kv_size
        local = torch.cat(
            (
                weights["q_proj.weight"][rank * qwidth : (rank + 1) * qwidth],
                weights["k_proj.weight"][kv_start : kv_start + config.kv_size],
                weights["v_proj.weight"][kv_start : kv_start + config.kv_size],
            )
        ).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(20261010 + rank + tokens)
        hidden = (
            torch.randn(
                (tokens // tp, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            * 0.05
        )
        position_base = torch.arange(tokens, device="cuda", dtype=torch.int64)
        positions = position_base.repeat(3, 1)
        slots = position_base.clone()
        if edge:
            storage = torch.empty((3, tokens + 7), device="cuda", dtype=torch.int64)
            positions = storage[:, :tokens]
            positions[0], positions[1], positions[2] = (
                position_base,
                position_base + 13,
                position_base + 29,
            )
            slot_storage = torch.empty(tokens * 2, device="cuda", dtype=torch.int64)
            slots = slot_storage[::2]
            slots.copy_(position_base * 2 + 15)
            slots[-1] = -1
        count = tokens * 2 + 128 if edge else tokens + 128
        blocks = (count + 127) // 128
        cache = torch.full(
            (blocks, config.num_kv_heads, 128, 2 * head_dim),
            0.375,
            device="cuda",
            dtype=torch.bfloat16,
        )
        if edge:
            cache = torch.full(
                (128, config.num_kv_heads, blocks, 2 * head_dim),
                0.375,
                device="cuda",
                dtype=torch.bfloat16,
            ).permute(2, 1, 0, 3)
        rope = MRotaryEmbedding(
            head_dim,
            rotary_dim,
            262144,
            1.0e7,
            True,
            torch.bfloat16,
            mrope_section=list(mrope_section),
            mrope_interleaved=True,
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to("cuda")
        inputs = Inputs(
            hidden,
            local,
            positions,
            weights["q_norm.weight"].to("cuda"),
            weights["k_norm.weight"].to("cuda"),
            rope.cos_sin_cache,
            cache,
            slots,
            torch.ones(1, device="cuda"),
            dist.group.WORLD,
            config,
        )
        return inputs, rope


    def independent_reference(inputs):
        """Oracle starts at original shards; never reads candidate scratch/results."""
        c, t = inputs.config, inputs.positions.shape[1]
        hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
        dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
        qkv = hidden @ inputs.weight.t()
        qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
        query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
        frequency = torch.arange(c.rotary_dim // 2, device="cuda")
        axis = torch.zeros_like(frequency)
        axis[1 : 3 * c.mrope_section[1] : 3] = 1
        axis[2 : 3 * c.mrope_section[2] : 3] = 2
        position = inputs.positions[axis].t()
        cos = inputs.rope[position, frequency].float().unsqueeze(1)
        sin = inputs.rope[position, frequency + c.rotary_dim // 2].float().unsqueeze(1)

        def norm_rope(x, weight):
            x = x.float()
            normalized = (
                (
                    x
                    * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                    * (weight.float() + 1)
                )
                .to(torch.bfloat16)
                .float()
            )
            first, second = normalized[..., : c.rotary_dim].chunk(2, dim=-1)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin), dim=-1
            )
            return torch.cat((rotated, normalized[..., c.rotary_dim :]), dim=-1).to(
                torch.bfloat16
            )

        query = norm_rope(query, inputs.q_norm).reshape(t, c.q_size)
        key = norm_rope(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
        valid = inputs.slots >= 0
        slots = inputs.slots[valid]
        block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
        inputs.cache[block, :, offset, : c.head_dim] = key[valid]
        inputs.cache[block, :, offset, c.head_dim :] = value.view(
            t, c.num_kv_heads, c.head_dim
        )[valid]
        return Output(query, key.reshape(t, c.kv_size), value, gate)


    # Keep the repository BF16 relative tolerance and bound near-zero differences.
    BF16_ATOL = 1e-2


    def check_outputs(actual, expected, actual_cache, expected_cache):
        for a, b in zip(actual.tensors(), expected.tensors()):
            gems_assert_close(a, b, torch.bfloat16, atol=BF16_ATOL)
            assert torch.isfinite(a).all()
        gems_assert_close(actual_cache, expected_cache, torch.bfloat16, atol=BF16_ATOL)


    def check_unaligned_inputs(inputs):
        """Exercise cached TMA inputs and Graph replay with only rank zero offset."""
        for field in ("hidden", "weight"):
            value = getattr(inputs, field)
            offset = int(dist.get_rank(inputs.group) == 0)
            storage = torch.empty(
                value.numel() + offset, device=value.device, dtype=value.dtype
            )
            changed = storage[offset:].view(value.shape)
            changed.copy_(value)
            candidate = replace(
                inputs, **{field: changed}, cache=torch.full_like(inputs.cache, 0.375)
            )
            assert changed.is_contiguous() and changed.data_ptr() % 16 == offset * 2
            # An aligned invocation has already populated the runtime cache.
            run_candidate(candidate)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run_candidate(candidate)
            try:
                for scale in (-0.75, -1.25):
                    changed.mul_(scale)
                    candidate.cache.fill_(0.375)
                    reference_cache = torch.full_like(candidate.cache, 0.375)
                    # Keep the oracle GEMM aligned while exercising candidate staging.
                    reference_weight = candidate.weight
                    if field == "weight" and offset:
                        reference_weight = reference_weight.clone()
                    expected = independent_reference(
                        replace(candidate, weight=reference_weight, cache=reference_cache)
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    check_outputs(actual, expected, candidate.cache, reference_cache)
            finally:
                graph.reset()


    def _worker(
        rank,
        tp,
        tokens,
        mode,
        iterations,
        warmup,
        edge,
        check_only,
        directory,
        model_dims,
        kernel_config,
    ):
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=tp,
            init_method=f"file://{directory}/rdzv",
            device_id=torch.device("cuda", rank),
            timeout=timedelta(seconds=180),
        )
        from vllm.config import VllmConfig
        from vllm.config.vllm import set_current_vllm_config

        native_context = set_current_vllm_config(VllmConfig())
        native_context.__enter__()
        source, rope = make_inputs(
            tokens, tp, rank, edge=edge, model_dims=model_dims, **kernel_config
        )
        from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
            megapreattention as impl,
        )

        names = ["megapreattention"] if check_only else ["native", "megapreattention"]
        inputs = {
            name: replace(source, cache=torch.full_like(source.cache, 0.375))
            for name in names
        }

        def candidate_with_other_current_device(value):
            # Exercise creation, cached calls and Graph capture with a different
            # current device. The API must restore the caller's device on return.
            other = (rank + 1) % tp
            with torch.cuda.device(other):
                result = run_candidate(value)
                assert torch.cuda.current_device() == other
            assert all(tensor.device == value.hidden.device for tensor in result.tensors())
            assert all(
                w.communication.device == value.hidden.device
                for w in impl._windows.values()
            )
            return result

        functions = {
            "megapreattention": (
                candidate_with_other_current_device if check_only else run_candidate
            ),
        }
        if not check_only:
            from benchmark.test_megapreattention import (
                NativeWindow,
                measure_window,
                summarize_samples,
            )

            native = NativeWindow(inputs["native"], rope, edge=edge)
            functions["native"] = native
        # Original-input oracle is independent of both native and candidate scratch.
        oracle_cache = torch.full_like(source.cache, 0.375)
        oracle = independent_reference(replace(source, cache=oracle_cache))
        for name in names:
            actual = functions[name](inputs[name])
            torch.cuda.synchronize()
            check_outputs(actual, oracle, inputs[name].cache, oracle_cache)
        for _ in range(warmup):
            for name in names:
                functions[name](inputs[name])
        torch.cuda.synchronize()
        graphs, outputs = {}, {}
        if mode == "cudagraph":
            for name in names:
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = functions[name](inputs[name])
                graphs[name] = graph

        calls = {
            name: (
                graphs[name].replay
                if mode == "cudagraph"
                else lambda name=name: functions[name](inputs[name])
            )
            for name in names
        }
        # Changed input and poisoned payload must work with the same registered runtime/Graph.
        for scale in (-0.75, -1.25):
            source.hidden.mul_(scale)
            oracle_cache.fill_(0.375)
            oracle = independent_reference(replace(source, cache=oracle_cache))
            for window in impl._windows.values():
                window.communication.buffer.fill_(
                    float("nan")
                )  # test-only, never production
            for name in names:
                inputs[name].cache.fill_(0.375)
                if mode == "cudagraph":
                    graphs[name].replay()
                    actual = outputs[name]
                else:
                    actual = functions[name](inputs[name])
                torch.cuda.synchronize()
                check_outputs(actual, oracle, inputs[name].cache, oracle_cache)
        # Unsupported metadata is rejected before any collective initialization.
        unsupported = {}
        check_metadata = (
            check_only
            and tp == 2
            and tokens == 1024
            and model_dims == (2048, 16, 2)
            and kernel_config["eps"] == 1e-6
        )
        if check_metadata:
            check_unaligned_inputs(inputs["megapreattention"])
        for name in ("megapreattention",) if check_metadata else ():
            value = inputs[name]
            bad_cases = {
                "fp32": replace(value, hidden=value.hidden.float()),
                "strided_hidden": replace(
                    value,
                    hidden=torch.empty(
                        (tokens // tp, value.hidden.shape[1] * 2),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )[:, ::2],
                ),
                "requires_grad": replace(
                    value, weight=value.weight.clone().requires_grad_(True)
                ),
                "wrong_heads": replace(
                    value,
                    config=replace(value.config, num_q_heads=value.config.num_q_heads + 1),
                ),
                "empty": replace(
                    value,
                    hidden=value.hidden[:0],
                    positions=value.positions[:, :0],
                    slots=value.slots[:0],
                ),
            }
            unsupported[name] = {}
            for label, bad in bad_cases.items():
                try:
                    run_candidate(bad)
                except AssertionError:
                    unsupported[name][label] = True
                else:
                    raise AssertionError(f"{name} silently accepted unsupported {label}")
            del bad_cases

        samples = {name: {"wall": [], "device": []} for name in names}
        if not check_only:
            samples = measure_window(calls, names, iterations)
        record = {
            "rank": rank,
            "samples": samples,
            "unsupported": unsupported,
            "jit_free": None if check_only else True,
            "native_source": None if check_only else native.source_evidence,
        }
        ranks = [None] * tp
        dist.all_gather_object(ranks, record)
        dist.barrier()
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        outputs.clear()
        graphs.clear()
        calls.clear()
        with torch.cuda.device((rank + 1) % tp):
            impl.clear_megapreattention_cache()
            assert torch.cuda.current_device() == (rank + 1) % tp
        if rank == 0:
            metrics = {} if check_only else summarize_samples(ranks, names, iterations)
            report = {
                "tp": tp,
                "tokens": tokens,
                "mode": mode,
                "edge": edge,
                "scope": "BF16 megapreattention window, not complete model inference",
                "correctness_passed": True,
                "measurement_jit_free": None if check_only else True,
                "iterations": iterations,
                "warmup": warmup,
                "metrics": metrics,
                "ranks": ranks,
                "fixture": {"input_origin": "seeded synthetic weights and hidden"},
            }
            (Path(directory) / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        dist.destroy_process_group()
        native_context.__exit__(None, None, None)


    def run_distributed(
        tp,
        tokens,
        mode,
        *,
        iterations=100,
        warmup=20,
        rounds=5,
        edge=False,
        check_only=False,
        model_dims=(2048, 16, 2),
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    ):
        """Run independent rounds with seeded synthetic inputs."""
        import torch.multiprocessing as mp

        model_dims = tuple(model_dims)
        kernel_config = dict(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            mrope_section=tuple(mrope_section),
            eps=eps,
        )
        reports = []
        for _ in range(rounds):
            with tempfile.TemporaryDirectory(prefix="megapreattention-") as directory:
                context = mp.spawn(
                    _worker,
                    args=(
                        tp,
                        tokens,
                        mode,
                        iterations,
                        warmup,
                        edge,
                        check_only,
                        directory,
                        model_dims,
                        kernel_config,
                    ),
                    nprocs=tp,
                    join=False,
                )
                deadline = time.monotonic() + 600
                try:
                    while not context.join(timeout=1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "megapreattention worker round exceeded 600s"
                            )
                    reports.append(
                        json.loads((Path(directory) / "report.json").read_text())
                    )
                finally:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in context.processes:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)
        summary = {}
        if not check_only:
            from benchmark.test_megapreattention import summarize_rounds

            summary = summarize_rounds(reports, mode)
        result = {
            "model_dims": model_dims,
            "kernel_config": kernel_config,
            "tp": tp,
            "tokens": tokens,
            "mode": mode,
            "correctness_passed": True,
            "measurement_jit_free": None if check_only else True,
            "summary": summary,
            "reports": reports,
        }
        return result


    def require_hopper(tp=1):
        if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} CUDA GPUs")
        if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(tp)):
            pytest.skip("requires SM90 GPUs")


    # The shared worker checks changed-input Graph replay and poisoned transport.
    # Cover each scheduling path without a second private-input test suite.
    CASES = [
        ((2048, 16, 2), 2, 1024, "cudagraph", True),
        ((5120, 24, 4), 2, 1024, "cudagraph", False),
        ((3072, 32, 8), 2, 1024, "cudagraph", True),
    ]
    if not cfg.QUICK_MODE:
        CASES += [
            ((2048, 16, 2), 2, 1280, "cudagraph", True),
            ((2048, 16, 2), 2, 32768, "cudagraph", True),
            ((2048, 16, 2), 4, 1536, "operator", True),
            ((2048, 16, 2), 4, 32768, "cudagraph", True),
            ((2048, 16, 2), 8, 2048, "cudagraph", True),
            ((2048, 16, 2), 8, 32768, "operator", False),
            ((5120, 24, 4), 4, 1024, "operator", True),
            ((5120, 24, 4), 8, 32768, "cudagraph", True),
            ((1024, 32, 16), 4, 1024, "cudagraph", True),
        ]


    @pytest.mark.parametrize("model_dims,tp,tokens,mode,edge", CASES)
    def test_megapreattention(model_dims, tp, tokens, mode, edge):
        require_hopper(tp)
        result = run_distributed(
            tp,
            tokens,
            mode,
            warmup=3,
            rounds=1,
            edge=edge,
            check_only=True,
            model_dims=model_dims,
        )
        assert result["correctness_passed"]
        assert all(len(r["ranks"]) == tp for r in result["reports"])


    def test_megapreattention_metadata_bounds():
        require_hopper()

        from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
            _initialize_metadata,
        )

        words, programs = 2050, 132
        flags = torch.full((words + 8,), 17, device="cuda", dtype=torch.uint64)
        counters = torch.full((programs + 8,), 17, device="cuda", dtype=torch.int64)
        _initialize_metadata[(triton.cdiv(words, 1024),)](
            flags[:words], counters[:programs], words, COUNTERS=programs, num_warps=4
        )
        torch.cuda.synchronize()
        assert flags.cpu().tolist() == [0] * words + [17] * 8
        assert counters.cpu().tolist() == [0] * programs + [17] * 8


    @triton.jit
    def _probe_packed_input_offsets(
        tiles, vectors, out, TILE_WORDS: tl.constexpr, offset_fn: tl.constexpr
    ):
        i = tl.arange(0, 4)
        offsets = offset_fn(tl.load(tiles + i), tl.load(vectors + i), TILE_WORDS)
        tl.store(out + i, offsets)


    def test_megapreattention_packed_input_offset_bounds():
        require_hopper()
        from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
            _packed_input_word_offset,
        )

        tile_words = 128 * 5120 // 2
        boundary = 2**31
        tile, remainder = divmod(boundary, tile_words)
        tiles = [tile, tile, tile + 1, 8191]
        vectors = [remainder // 4 - 1, remainder // 4, 0, tile_words // 4 - 1]
        actual = torch.empty(4, dtype=torch.int64, device="cuda")
        _probe_packed_input_offsets[(1,)](
            torch.tensor(tiles, dtype=torch.int32, device="cuda"),
            torch.tensor(vectors, dtype=torch.int32, device="cuda"),
            actual,
            TILE_WORDS=tile_words,
            offset_fn=_packed_input_word_offset,
            num_warps=1,
        )
        assert actual.tolist() == [
            boundary - 4,
            boundary,
            (tile + 1) * tile_words,
            1048576 * 5120 // 2 - 4,
        ]


    ROPE_CASES = [
        (2, (16, 8, 8), 1e-5),
        (4, (8, 12, 12), 1e-4),
    ]


    @pytest.mark.parametrize("tp,sections,eps", ROPE_CASES)
    def test_megapreattention_rope_parameters(tp, sections, eps):
        require_hopper(tp)
        result = run_distributed(
            tp,
            1024,
            "cudagraph",
            warmup=3,
            rounds=1,
            edge=True,
            check_only=True,
            mrope_section=sections,
            eps=eps,
        )
        assert result["correctness_passed"]
