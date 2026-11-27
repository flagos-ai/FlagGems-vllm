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

"""单用例调试脚本：GLM W8A8C8 档的 MXFP8 GroupedMatmulSwigluQuantV2。

按 remind.md 第 3 条，单用例用典型大模型 shape、用于迭代；多用例最终测试另行。
默认取 GLM W8A8C8，因为这个算子在那档占 7.37%，shape 比 Qwen 大、更能反映
带宽瓶颈。

  x  [64, 2048]        fp8_e4m3fn
  w  [32, 2048, 6144]  fp8_e4m3fn   gate|up 拼在一起，所以输出宽 3072
  y  [64, 3072]        fp8_e4m3fn
  ys [64, 48, 2]       e8m0

性能口径：只看 device kernel 时间（host 时间会放大假差距），用 profiler 的
kernel_details.csv 取 Duration 求和。比值必须取自同一次运行。

用法:
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src python \\
      src/flaggems_vllm/runtime/backend/_ascend/a5/tests/bench_grouped_matmul_swiglu_quant_v2.py
  ... --shape qwen
  ... --block-n 256      # 扫 tile
  ... --skip-ref         # 显存紧时跳过 torch 参考
"""

import argparse
import csv
import glob
import os
import shutil
import tempfile

# /data 已满，TMPDIR 默认指向那里，profiler 落不了盘会导致 kernel_details.csv
# 不生成、计时静默返回 None。必须在 import torch_npu 之前改。
if not os.environ.get("TMPDIR", "").startswith("/tmp"):
    os.environ["TMPDIR"] = "/tmp"

import torch
import torch_npu  # noqa: F401

from flaggems_vllm.runtime.backend._ascend.a5.ops.grouped_matmul_swiglu_quant_v2 import (
    MX_BLOCK,
    grouped_matmul_swiglu_quant_v2,
    grouped_matmul_swiglu_quant_v2_ref,
)

DEV = "npu"

# w13_weight 的真实档位（K/N 都是 process_weights_after_loading 转置之后的值，
# 即算子实际收到的）。N 是 gate|up 拼接，输出是 N/2。
SHAPES = {
    "glm_w8": dict(M=64, K=2048, N=6144, E=32, name="GLM W8A8C8 (7.37%)"),
    "qwen": dict(M=16, K=512, N=2048, E=128, name="Qwen3.6 (8.22%)"),
}


def build_inputs(M, K, N, E, seed=0):
    """造 MXFP8 输入。

    用 npu_dynamic_mx_quant 生成，保证 scale 的指数分布与真实推理一致——自己
    随便填 uint8 会得到不真实的指数范围，掩盖溢出和饱和问题。
    """
    torch.manual_seed(seed)
    x_bf = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w_bf = torch.randn(E, K, N, device=DEV, dtype=torch.bfloat16) * 0.1

    x, sa = torch_npu.npu_dynamic_mx_quant(x_bf, dst_type=torch.float8_e4m3fn)

    # weight 要按 K（倒数第二维）分块，而 npu_dynamic_mx_quant 按最后一维分，
    # 所以先转置再量化再转回。torch.stack 不支持 fp8（aclnnStack 的 dtype 列表
    # 里没有 DT_FLOAT8_E4M3FN），所以预分配后逐 expert 写。
    w = torch.empty(E, K, N, device=DEV, dtype=torch.float8_e4m3fn)
    sb_log = torch.empty(E, N, K // MX_BLOCK, device=DEV, dtype=torch.uint8)
    for e in range(E):
        wt = w_bf[e].t().contiguous()
        wq, sq = torch_npu.npu_dynamic_mx_quant(wt, dst_type=torch.float8_e4m3fn)
        w[e] = wq.t()
        sb_log[e] = sq.reshape(N, K // MX_BLOCK)
    # [E, N, K/32] → 物理 [E, K/64, N, 2]：逻辑 b 落在 (b//2, n, b%2)
    sb = sb_log.permute(0, 2, 1).contiguous()
    sb = sb.reshape(E, K // 64, 2, N).permute(0, 1, 3, 2).contiguous()

    # group_list：M 行均分到 E 个 expert。vllm 传 cumsum（type=0）。
    base, rem = M // E, M % E
    sizes = [base + (1 if i < rem else 0) for i in range(E)]
    gl_count = torch.tensor(sizes, device=DEV, dtype=torch.int64)
    gl = torch.cumsum(gl_count, dim=0)

    del w_bf, x_bf, sb_log
    torch.npu.empty_cache()
    return x, w, sb, sa.contiguous(), gl


def vendor(x, w, sb, sa, gl):
    """基线：vllm-ascend apply_gmm1_act_quant 的实际调用形态
    （w8a8_mxfp8.py:562）。"""
    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=x,
        weight=[w],
        group_list=gl,
        weight_scale=[sb],
        x_scale=sa,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        x_dtype=None,
        weight_dtype=None,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
    )


def as_u8(t):
    """vendor 的 scale 是 float8_e8m0fnu，不能 .int()（aclnnInplaceCopy 报
    561103）。它就是一个字节，view 成 uint8。"""
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def device_time_us(fn, warmup=5, iters=20):
    """用 profiler 取 device kernel 时间（us），返回 (总均值, kernel 名→均值)。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    out_dir = tempfile.mkdtemp(prefix="gmmsq_prof_")
    try:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
        ):
            for _ in range(iters):
                fn()
            torch.npu.synchronize()

        cands = glob.glob(
            os.path.join(out_dir, "**", "kernel_details.csv"), recursive=True
        )
        if not cands:
            return None, {}
        per_kernel = {}
        with open(cands[0]) as f:
            for row in csv.DictReader(f):
                name = row.get("Name") or row.get("Op Name") or "?"
                dur = (
                    row.get("Duration(us)")
                    or row.get("Duration (us)")
                    or row.get("Duration")
                )
                if dur is None:
                    continue
                try:
                    per_kernel[name] = per_kernel.get(name, 0.0) + float(dur)
                except ValueError:
                    continue
        total = sum(per_kernel.values()) / iters
        return total, {k: v / iters for k, v in per_kernel.items()}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def compare(tag, y, ys, ry, rys, valid, nh):
    """精度比对。判据是**逐位相等**：它同时证明走的是和厂商同一条硬件指令。"""
    yf = y[:valid].float()
    rf = ry[:valid].float()
    dy = (yf - rf).abs().max().item()
    nbad = (yf != rf).sum().item()
    a = as_u8(ys).reshape(ys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    b = as_u8(rys).reshape(rys.shape[0], -1)[:valid, : nh // MX_BLOCK].int()
    ds = (a - b).abs().max().item()
    sbad = (a != b).sum().item()
    verdict = "逐位相等" if (nbad == 0 and sbad == 0) else "有差异"
    print(
        f"  {tag:20s} y max_absdiff {dy:<10g} y 不同 {nbad}/{yf.numel()}  "
        f"scale max_absdiff {ds} 不同 {sbad}/{a.numel()}   {verdict}"
    )
    return nbad == 0 and sbad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="glm_w8", choices=sorted(SHAPES))
    ap.add_argument("--block-m", type=int, default=16)
    ap.add_argument(
        "--block-n",
        type=int,
        default=None,
        help="默认自动挑（两个累加器放得下 L0C 的最大值）",
    )
    ap.add_argument(
        "--block-k",
        type=int,
        default=64,
        help="必须是 64 的倍数（B 侧 scale 交错布局的约束）",
    )
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument(
        "--scale-alg", type=int, default=0, help="0=OCP（vendor 实测用这个），1=cuBLAS"
    )
    ap.add_argument(
        "--nd", action="store_true", help="强制走 ND（默认 NZ，那是默认部署布局）"
    )
    ap.add_argument(
        "--skip-ref",
        action="store_true",
        help="跳过 torch 参考（GLM 档 fp32 展开约 1.5GB）",
    )
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    cfg = SHAPES[args.shape]
    M, K, N, E = cfg["M"], cfg["K"], cfg["N"], cfg["E"]
    NH = N // 2
    print(f"== {cfg['name']}: x[{M},{K}] w[{E},{K},{N}] -> y[{M},{NH}] ==")
    print(
        f"   weight {E * K * N / 2**20:.0f}MB   " f"layout {'ND' if args.nd else 'NZ'}"
    )

    x, w, sb, sa, gl = build_inputs(M, K, N, E)
    w_nd = w  # torch 参考要 ND：NZ 张量 .to(fp32) 报
    # "copy_ operator with internal format tensors
    #  is not supported on Ascend950"
    if not args.nd:
        w = torch_npu.npu_format_cast(w, 29, customize_dtype=torch.float8_e4m3fn)
        assert torch_npu.get_npu_format(w) == 29

    kw = dict(
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_cores=args.cores,
        scale_alg=args.scale_alg,
    )

    # ---- 正确성 ----
    vy, vys = vendor(x, w, sb, sa, gl)
    valid = int(gl[-1])
    print(f"\n[精度] 有效行 [0,{valid})（grouplist 之外不更新也不初始化）")
    try:
        ty, tys = grouped_matmul_swiglu_quant_v2(
            x, w, sb, sa, gl, group_list_type=0, **kw
        )
    except Exception as err:
        print(f"  FAILED: {type(err).__name__}: {err}")
        raise
    ok = compare("triton vs vendor", ty, tys, vy, vys, valid, NH)

    if not args.skip_ref:
        ry, rys, rvalid = grouped_matmul_swiglu_quant_v2_ref(
            x, w_nd, sb, sa, gl, group_list_type=0, scale_alg=args.scale_alg
        )
        compare("vendor vs torch", vy, vys, ry, rys, min(valid, rvalid), NH)
        compare("triton vs torch", ty, tys, ry, rys, min(valid, rvalid), NH)
        del ry, rys
        torch.npu.empty_cache()

    # ---- 성능 ----
    tv, kv = device_time_us(lambda: vendor(x, w, sb, sa, gl), iters=args.iters)
    tt, kt = device_time_us(
        lambda: grouped_matmul_swiglu_quant_v2(
            x, w, sb, sa, gl, group_list_type=0, **kw
        ),
        iters=args.iters,
    )
    if tv is None or tt is None:
        print(f"\n[性能] profiler 没出 kernel_details.csv (v={tv}, t={tt})")
        return
    print(f"\n[性能] vendor {tv:.2f}us   triton {tt:.2f}us   " f"比值 {tv / tt:.3f}x")
    for tag, ks in (("vendor", kv), ("triton", kt)):
        for name, us in sorted(ks.items(), key=lambda kv2: -kv2[1])[:4]:
            print(f"   {tag:6s} {us:8.2f}us  {name}")
    print("\nPASS" if ok else "\nFAIL")


if __name__ == "__main__":
    main()
