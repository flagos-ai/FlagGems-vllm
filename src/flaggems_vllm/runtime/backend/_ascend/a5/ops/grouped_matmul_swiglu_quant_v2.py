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

"""GroupedMatmulSwigluQuantV2 (MX-FP8) — Triton on Ascend 950DT (A5).

对标 `torch_npu.npu_grouped_matmul_swiglu_quant_v2`，调用形态取自
`vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py:562` 的
`apply_gmm1_act_quant`。op-plugin 按 weight 的 format 分派到
`aclnnGroupedMatmulSwigluQuantV2`（ND）或 `...WeightNzV2`（NZ）。

一条 kernel 做三件事（这就是融合的意义——中间的 [M, N] fp32 不落 GM）：

    1. grouped matmul + MX 反量化      → fp32 [M, N]
    2. SwiGLU：沿 N 前后分半            → fp32 [M, N/2]
         S = silu(C[:, :N/2]) * C[:, N/2:]
    3. MX 动态量化（沿 N/2 每 32 个一组）→ fp8_e4m3 [M, N/2] + e8m0 scale

数值契约全部由 probe_contract.py 实测定死，不是从文档推的（文档与安装版已
分叉：torch_npu 的 docstring 还写着「quant_mode 仅支持 0」，而 vllm 实际传 2）：

  * SwiGLU 是 alpha=1 / bias=0 / 无 clamp。V2 不收这三个参数（V3 才暴露
    swigluMode/clampLimit/gluAlpha/gluBias），实测 silu(前半)*后半 的 scale
    字节与 vendor 6144/6144 全同。
  * 量化用 **scale_alg=0（OCP）**，不是 cuBLAS。实测 alg0 的 scale 字节
    6144/6144 相同、alg1 有 1129/6144 不同（18% 的 y 元素错）。V2 不收
    scaleAlg 参数，所以这一条只能实测。
  * 量化前先把 fp32 的 SwiGLU 结果按最近偶数舍入到 **BF16**，再按 32 分组。
  * 输出 scale 的 dtype 是 `float8_e8m0fnu`（不是 uint8），shape
    [M, nblk/2, 2]，物理就是连续的 [M, nblk]（平凡展平，与 DynamicMxQuant
    同一套布局）。注意它不能 `.int()`，aclnnInplaceCopy 报 561103，要先
    `view(torch.uint8)`。
  * grouplist 之外的行不被更新也不初始化，不能参与精度比对。
gmm 内核沿用 groupmatmul 那边已验证逐位等于 CANN 的 variant=2 路径
（`tl.dot_scaled` → `hfusion.matmul_mx` 硬件 MX-MAC）。那边踩出来、这里直接继承
的几条，改动时不要回退：

  * `//` 和 `%` 只能作用在**纯 arange** 上。NZ 索引写成
    `(tile_n*BLOCK_N + arange)//32` 会让编译器认不出规整访存、退化成逐元素
    gather，实测慢 17~37 倍。索引必须是「标量基址 + 编译期常量向量 + 仿射项」。
  * scale 的 load **绝不能加 mask**，实测慢 13 倍（653 → 8445us）。A 侧靠
    wrapper 按 M+BLOCK_M 补齐来免 mask，越界行靠 store 的 mask 丢掉。
  * B 侧 scale 的物理布局是 `[E,K/64,N,2]` 真交错，必须按
    `(kb0//2)*(N*2) + n*2 + (j//2)*(N*2) + j%2` 索引；按平凡 `[E,K/32,N]` 写，
    用均匀 scale 测精度能过，真实数据下错 20%。
  * `tl.dot_scaled` 的 rhs_scale 形状是 **[BLOCK_N, BLOCK_K/32]**（N 在前）。

本算子相对 groupmatmul 新增的结构问题：**SwiGLU 要 C[m,n] 和 C[m,n+N/2] 同时
在手**，所以一个 program 必须同时算两个半边，也就是两个 [BLOCK_M, BLOCK_N] 的
fp32 累加器同时活在 L0C 里。L0C 是 256KB，所以 BLOCK_N 的上限比 groupmatmul
那边（单累加器 1024）要减半，见 `_pick_block_n`。

用法见 bench_grouped_matmul_swiglu_quant_v2.py（单用例，迭代用）。
"""

import torch
import triton
import triton.language as tl

MX_BLOCK = 32  # 每个 e8m0 scale 覆盖的元素数，硬件定死
E4M3_EMAX = 8  # float8_e4m3fn 最大正则数的指数
BF16_MBITS = 7  # bf16 尾数位
L0C_BYTES = 256 * 1024

__all__ = [
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_v2_ref",
    "MX_BLOCK",
]


@triton.jit
def _tile_order(idx, num_m_tiles, num_n_tiles, THRESHOLD: tl.constexpr):
    """把线性 block 号映射成 (tile_m, tile_n)，让连续的 m-tile 共享 B 的 panel。

    沿用 groupmatmul 的实现：m-tile 少于 THRESHOLD 时按 n 优先（B 的 panel 留在
    L2），否则退化成行主序。
    """
    if num_m_tiles < THRESHOLD:
        return idx % num_m_tiles, idx // num_m_tiles
    return idx // num_n_tiles, idx % num_n_tiles


@triton.jit
def _gmm_swiglu_mxquant_kernel(
    A,  # fp8_e4m3fn [M+BLOCK_M, K]      已补齐行，故 load 不带 mask
    SA,  # uint8      [M+BLOCK_M, K/32]   A 侧 e8m0，同样补齐
    B,  # fp8_e4m3fn [E, K, N]           ND 或 FRACTAL_NZ
    SB,  # uint8      [E, K/64, N, 2]     真交错
    Y,  # fp8_e4m3fn [M, N/2]
    YS,  # uint8      [M, NSP]            平凡展平的 e8m0
    group_list,  # int32/64   [E]
    GROUPS: tl.constexpr,
    NH: tl.constexpr,  # N/2，输出宽度
    K: tl.constexpr,
    N: tl.constexpr,  # 2*NH，weight 的宽度
    NSP: tl.constexpr,  # YS 的行跨距 = NH/32
    GROUP_LIST_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # 输出 tile 宽（不是 weight tile 宽）
    BLOCK_K: tl.constexpr,
    TILE_THRESHOLD: tl.constexpr,
    MX: tl.constexpr,
    WEIGHT_NZ: tl.constexpr,
    CEIL: tl.constexpr,  # 量化 scale 算法：0=OCP（vendor 用这个），1=cuBLAS
    B_CACHE: tl.constexpr,
    SB_CACHE: tl.constexpr,
    EPI: tl.constexpr,  # 2=全量；1=只做 SwiGLU 不量化；0=都不做
    # （0/1 只用于计时拆解，数值无意义）
    DUMP: tl.constexpr,  # 诊断用：把两个累加器存出去（bf16，见下）
    DG,  # DUMP 时的 bf16 [M, NH]（gate 半边）
    DU,  # DUMP 时的 bf16 [M, NH]（up 半边）
):
    total_cores = tl.num_programs(0)
    core_idx = tl.program_id(0)
    num_n_tiles: tl.constexpr = NH // BLOCK_N  # wrapper 保证整除
    n_kblk: tl.constexpr = K // MX
    sb_stride_e: tl.constexpr = n_kblk * N
    NB: tl.constexpr = BLOCK_N // MX  # 一个 tile 里的量化组数

    # 量化用的位常量（在 bf16 的位域上做，见模块 docstring）。字面量写死而不是
    # 引模块常量：jit 函数读不到非 constexpr 的全局（NameError）。
    SGN: tl.constexpr = -32768  # 符号位，int16
    MAG: tl.constexpr = 32767  # 幅值位
    MANT: tl.constexpr = 127  # bf16 尾数 7 位
    FM: tl.constexpr = 96  # fmax(1.75) 的尾数 = 127-31

    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(GROUPS):
        gv = tl.load(group_list + group_idx).to(tl.int32)
        group_m = gv - group_end if GROUP_LIST_TYPE == 0 else gv
        group_end = group_start + group_m
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)

        current_count = last_count + num_m_tiles * num_n_tiles
        current_block = core_idx if core_idx >= last_count else core_idx + total_cores

        for block_idx in range(current_block, current_count, total_cores):
            tile_m, tile_n = _tile_order(
                block_idx - last_count, num_m_tiles, num_n_tiles, TILE_THRESHOLD
            )
            # int64：B 的元素数（E*K*N）可超 2^31
            offs_m = (group_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(
                tl.int64
            )
            offs_n0 = tl.arange(0, BLOCK_N).to(tl.int64)  # 纯 arange
            offs_n = (tile_n * BLOCK_N + offs_n0).to(tl.int64)
            offs_kk = tl.arange(0, BLOCK_K).to(tl.int64)
            offs_sk = tl.arange(0, BLOCK_K // MX).to(tl.int64)

            mask_m = offs_m < group_end
            g64 = group_idx.to(tl.int64)

            a_base = A + offs_m[:, None] * K
            sa_base = SA + offs_m[:, None] * n_kblk + offs_sk[None, :]
            # 两个半边的 B 基址。gate 在 [0, NH)，up 在 [NH, N)。
            # NH 是 BLOCK_N 的整数倍（wrapper 保证），所以 up 只是 gate 的列偏移
            # 加 NH——在 NZ 下也成立，因为 NH % 32 == 0 时 NH 跨的是整数个分形。
            if WEIGHT_NZ:
                # FRACTAL_NZ = [E, N/32, K/16, 16, 32]，展开
                #   offs = e*K*N + (n//32)*(K*32) + k*32 + (n%32)
                # （(k//16)*16 + k%16 ≡ k，k 轴在分形内就是 *32 的仿射）
                # `//32`、`%32` 只作用在 offs_n0 这个纯 arange 上，tile 偏移
                # 拆成标量项提到外面——这一条是 37 倍的差别，见 docstring。
                nz_g: tl.constexpr = BLOCK_N // 32
                nvec = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
                bg_base = (
                    B
                    + g64 * (K * N)
                    + tile_n.to(tl.int64) * (nz_g * K * 32)
                    + nvec[None, :]
                )
                bu_base = bg_base + (NH // 32) * (K * 32)
            else:
                bg_base = B + g64 * (K * N) + offs_n[None, :]
                bu_base = bg_base + NH
            sb_base = SB + g64 * sb_stride_e

            acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, n_kblk, BLOCK_K // MX):
                kb0 = k0.to(tl.int64)
                koff = kb0 * MX
                # A 的 load 不带 mask：wrapper 已按 M+BLOCK_M 补齐行，越界行对应
                # 的累加器行由 store 的 mask 丢掉。实测省 37.5us 且数值不变。
                a = tl.load(a_base + koff + offs_kk[None, :])
                sa = tl.load(sa_base + kb0)

                # B 侧 scale：物理 flat = (b//2)*(N*2) + n*2 + (b%2)，
                # b = kb0 + j。BLOCK_K 是 64 的倍数 ⇒ kb0 偶 ⇒
                #   (kb0+j)//2 = kb0//2 + j//2,  (kb0+j)%2 = j%2
                # 于是 `//`、`%` 只落在纯 arange 的 j 上。
                sb_off = (
                    (kb0 // 2) * (N * 2)
                    + (offs_sk[None, :] // 2) * (N * 2)
                    + (offs_sk[None, :] % 2)
                )
                sbg = tl.load(
                    sb_base + sb_off + offs_n[:, None] * 2, cache_modifier=SB_CACHE
                )
                sbu = tl.load(
                    sb_base + sb_off + (offs_n[:, None] + NH) * 2,
                    cache_modifier=SB_CACHE,
                )

                if WEIGHT_NZ:
                    kstep = (koff + offs_kk)[:, None] * 32
                else:
                    kstep = (koff + offs_kk)[:, None] * N
                bg = tl.load(bg_base + kstep, cache_modifier=B_CACHE)
                bu = tl.load(bu_base + kstep, cache_modifier=B_CACHE)

                # rhs_scale 的形状是 [BLOCK_N, BLOCK_K/32]（N 在前）
                acc_g = tl.dot_scaled(a, sa, "e4m3", bg, sbg, "e4m3", acc_g)
                acc_u = tl.dot_scaled(a, sa, "e4m3", bu, sbu, "e4m3", acc_u)

            if DUMP:
                # 诊断：把 gmm 的两个半边存出去，用来把「gmm 错了」和「epilogue
                # 错了」分开。正式路径不走这里。
                #
                # 必须先 .to(bf16)：把 fp32 累加器直接 store 到 fp32 张量会让
                # bishengir-compile SIGABRT，只给一句
                #   LLVM ERROR: Warning: Cannot calculate static size from
                #   mmadL1 shape
                # 不给 pass 名也不给 shape。groupmatmul 的 output_dtype=fp32 是
                # 同一条错。留档见
                # log/20261123_groupedMatmulSwigluQuant_fp32_store_mmadL1_abort.log
                d_off = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(DG + d_off, acc_g.to(tl.bfloat16), mask=mask_m[:, None])
                tl.store(DU + d_off, acc_u.to(tl.bfloat16), mask=mask_m[:, None])

            if EPI == 0:
                # 诊断：跳过 SwiGLU 和量化，直接把 gate 半边转 fp8 存出去。
                # 搬运量和 tile 都不变（两个 dot 照做、两份 B 照搬、store 的
                # 字节数相同），所以与正式路径的差值就是 epilogue 的净代价。
                # 数值无意义，只用来计时。
                y_off0 = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(
                    Y + y_off0, (acc_g + acc_u).to(tl.float8e4nv), mask=mask_m[:, None]
                )
                s_off0 = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off0,
                    tl.zeros((BLOCK_M, NB), dtype=tl.uint8),
                    mask=mask_m[:, None],
                )
            elif EPI == 3:
                # 拆开路径（正式）：只做到 SwiGLU，存 bf16，量化交给第二个
                # kernel 在稠密的 [M, NH] 上做。
                #
                # 为什么要拆：epilogue 的向量工作量是 BLOCK_M × BLOCK_N，而有效
                # 行只有 group_m。GLM 档 group_m=2、BLOCK_M=16（cube 的最小 M
                # 粒度），于是 7/8 的量化工作落在被 mask 掉的行上。实测
                # （probe_split.py）：量化融合在这里要 +32.51us，而同样的量化
                # 放在稠密 [64,3072] 上只要 3.84us——**8.5 倍**，和 8 倍的
                # padding 浪费对得上。
                #
                # 为什么精度不受影响：文档要求量化前把 SwiGLU 结果按最近偶数
                # 舍入到 BF16，所以这里存 bf16 正是该做的那一步，不是额外的
                # 精度损失。SwiGLU 本身仍然读 fp32 累加器。
                s3 = (acc_g * tl.sigmoid(acc_g)) * acc_u
                tl.store(
                    Y + offs_m[:, None] * NH + offs_n[None, :],
                    s3.to(tl.bfloat16),
                    mask=mask_m[:, None],
                )
            elif EPI == 1:
                # 诊断：只做 SwiGLU，不做量化的 reduce 链。用来把 epilogue 的
                # 35us（GLM 档）拆成「SwiGLU 的逐元素」和「量化的 reduce」两段。
                s1 = (acc_g * tl.sigmoid(acc_g)) * acc_u
                y_off1 = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(Y + y_off1, s1.to(tl.float8e4nv), mask=mask_m[:, None])
                s_off1 = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off1,
                    tl.zeros((BLOCK_M, NB), dtype=tl.uint8),
                    mask=mask_m[:, None],
                )
            else:
                # ---------------- SwiGLU ----------------
                # V2 的 MX 路径：alpha=1、bias=0、无 clamp。silu(前半) * 后半。
                s = (acc_g * tl.sigmoid(acc_g)) * acc_u

                # ---------------- MX 动态量化 ----------------
                # 先按最近偶数舍入到 bf16（文档明确要求），再在 bf16 的位域上做。
                xt = tl.reshape(s.to(tl.bfloat16), (BLOCK_M * NB, MX))
                xi = xt.to(tl.int16, bitcast=True)
                # 清掉符号位后整数位型对 |x| 单调，所以位型的 max 就是最大幅值的位型，
                # 省掉一次 fp32 abs。
                mg = (xi & MAG).to(tl.int16)
                mm = tl.max(mg, 1)
                # 先过 fp32 再读指数：denormal 的 amax 没有隐含前导 1，它的指数域
                # 不是它的指数。这个转换作用在 [BLOCK_M*NB] 的 amax 上，几乎免费。
                af = mm.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
                ab = af.to(tl.int32, bitcast=True)
                ef = (ab >> 23) & 0xFF
                eb = ef - 8  # float8_e4m3fn 的 emax，字面量同上
                if CEIL:
                    eb = eb + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
                nan = ef == 0xFF
                if CEIL:
                    flush = nan
                else:
                    flush = (eb <= 0) | nan
                ebs = tl.where(nan, 255, tl.maximum(eb, 0))
                # rc = 2^(127-ebs)，用位型拼而不是算 exp2；minimum 让 inf/nan 路保持
                # 有限，这样 0*rc 还是 0 而不是 nan。
                rc = ((254 - tl.minimum(ebs, 254)) << 23).to(tl.float32, bitcast=True)
                if CEIL:
                    cap = tl.where(nan, 0, MAG)
                else:
                    # cap = 1.75*2^e_amax：直接拿 amax 自己的指数位配 fmax 的尾数，
                    # 不做 eb 算术，这样 ebs 被钉到 0 时也仍然有效。
                    cap = tl.where(flush, 0, (mm.to(tl.int32) & ~MANT) | FM)
                sgn = (xi & SGN).to(tl.int16)
                # 饱和在整数幅值域做：fp32 的 min/max 对会让临时量挤 UB
                src = (tl.minimum(mg, cap[:, None].to(tl.int16)) | sgn).to(tl.int16)
                q = src.to(tl.bfloat16, bitcast=True).to(tl.float32) * rc[:, None]
                q8 = q.to(tl.float8e4nv)

                y_off = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(
                    Y + y_off, tl.reshape(q8, (BLOCK_M, BLOCK_N)), mask=mask_m[:, None]
                )
                s_off = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off,
                    tl.reshape(ebs.to(tl.uint8), (BLOCK_M, NB)),
                    mask=mask_m[:, None],
                )

        # `% total_cores` 不能省。last_count 是「下一个 group 的第一块落在哪个
        # 核上」，必须留在 [0, total_cores) 里，`core_idx >= last_count` 的判断
        # 和 `core_idx + total_cores` 的回绕才成立。写成 `= current_count` 时
        # last_count 单调增长，于是 current_block 恒取回绕分支、
        # `block_idx - last_count` 变成负数 → tile 索引错位，两个半边一起错到
        # max_rel≈1.5（看着像索引公式错，其实是分配错）。
        last_count = current_count % total_cores
        group_start = group_end


@triton.jit
def _mxquant_kernel(
    X,
    Y,
    S,
    M,
    NH: tl.constexpr,
    NSP: tl.constexpr,
    ROWS: tl.constexpr,
    NB: tl.constexpr,
    PROGS: tl.constexpr,
):
    """拆开路径的第二个 kernel：bf16 [M, NH] → fp8_e4m3 + e8m0 scale [M, NH/32]。

    位运算和上面融合 epilogue 的 scale_alg=0 分支完全一样（那条已对 vendor
    逐位验过），区别只在**数据是稠密的**：这里的 M 就是真实行数，没有
    BLOCK_M 补齐，所以不付 BLOCK_M/group_m 的那 8 倍浪费。

    一个 program 一次吃 ROWS 整行（W = NB*32 取整行宽）。不切列是故意的：
    部分行的 tile 会让 load 变成跨步 DMA，DynamicMxQuant 那边实测贵 14 倍。
    """
    pid = tl.program_id(0)
    MXC: tl.constexpr = 32  # 字面量：jit 读不到模块级全局
    W: tl.constexpr = NB * MXC
    SGN: tl.constexpr = -32768
    MAG: tl.constexpr = 32767
    MANT: tl.constexpr = 127
    FM: tl.constexpr = 96
    w = tl.arange(0, W)
    bl = tl.arange(0, NB)
    for r0 in range(pid * ROWS, M, PROGS * ROWS):
        r = r0 + tl.arange(0, ROWS)
        rm = (r < M)[:, None]
        xb = X + r[:, None] * NH
        yb = Y + r[:, None] * NH
        sb = S + r[:, None] * NSP
        for k0 in range(0, NH, W):
            flat = tl.load(xb + (k0 + w)[None, :], mask=rm, other=0)
            xt = tl.reshape(flat, (ROWS * NB, MXC))
            xi = xt.to(tl.int16, bitcast=True)
            mg = (xi & MAG).to(tl.int16)
            mm = tl.max(mg, 1)
            af = mm.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            ab = af.to(tl.int32, bitcast=True)
            ef = (ab >> 23) & 0xFF
            eb = ef - 8
            nan = ef == 0xFF
            flush = (eb <= 0) | nan
            ebs = tl.where(nan, 255, tl.maximum(eb, 0))
            rc = ((254 - tl.minimum(ebs, 254)) << 23).to(tl.float32, bitcast=True)
            cap = tl.where(flush, 0, (mm.to(tl.int32) & ~MANT) | FM)
            sgn = (xi & SGN).to(tl.int16)
            src = (tl.minimum(mg, cap[:, None].to(tl.int16)) | sgn).to(tl.int16)
            q = src.to(tl.bfloat16, bitcast=True).to(tl.float32) * rc[:, None]
            tl.store(
                yb + (k0 + w)[None, :],
                tl.reshape(q.to(tl.float8e4nv), (ROWS, W)),
                mask=rm,
            )
            tl.store(
                sb + (k0 // MXC + bl)[None, :],
                tl.reshape(ebs.to(tl.uint8), (ROWS, NB)),
                mask=rm,
            )


QUANT_TARGET_PROGS = 16  # 第二个 kernel 的目标 program 数，实测值，见下
QUANT_MAX_ROWS = 8
QUANT_MAX_TILE = 32 * 1024  # ROWS*NH 的上限，防止大 NH 下挤爆 UB


def _pick_quant_rows(M: int, cores: int, NH: int) -> int:
    """第二个 kernel 的 ROWS（一个 program 一次吃几整行）。

    实测（probe_quant_rows.py，真实两 kernel 流水内，kernel 2 的净时间）：

      | 档位           | ROWS=1 | ROWS=2 | ROWS=4 | ROWS=8 |
      | GLM [64,3072]  | 6.90 | 9.62 | **4.24** | 6.81 |
      | Qwen [16,1024] | **2.75** | 3.79 | 6.21 | 11.16 |

    两档的最优都落在 **progs=16**（GLM 64/4、Qwen 16/1），所以这里按
    progs=16 反推 ROWS。注意 progs=16 远小于 64 个 vector core——填满核
    反而更慢（GLM ROWS=1 是 progs=64，慢 1.6 倍），和 gmm 那边 BN 不能
    调小去填核是同一件事：per-program 的固定开销比空闲核贵。

    这个 16 是两个形状标出来的，其中 Qwen 那档 progs 被 M=16 顶死（ROWS
    最小是 1），真正自由选的只有 GLM。换形状要用 probe_quant_rows.py
    复标，别当成普适常数。
    """
    rows = 1
    while (
        rows * 2 <= QUANT_MAX_ROWS
        and rows * 2 * NH <= QUANT_MAX_TILE
        and -(-M // (rows * 2)) >= QUANT_TARGET_PROGS
    ):
        rows *= 2
    return rows


def _cube_core_num(device=None) -> int:
    props = torch.npu.get_device_properties(device)
    for attr in ("num_ai_cores", "multi_processor_count"):
        v = getattr(props, attr, None)
        if v:
            return int(v)
    return 32


def _is_nz(t: torch.Tensor) -> bool:
    """NZ 性只在 ACL format 描述符里——`npu_format_cast` 之后 torch 的 shape 和
    numel 都不变，所以 shape 没有判定力，只能问 format。"""
    try:
        import torch_npu

        return torch_npu.get_npu_format(t) == 29
    except Exception:
        return False


def _pick_block_n(NH: int, prefer: int | None = None, block_m: int = 16) -> int:
    """挑能整除 NH 的最大 BLOCK_N，且两个 fp32 累加器放得下 L0C。

    预算是**实测**出来的，不是推的（probe_tile.py，GLM-W8 档）：
        BM=16 BN=1024 → 编得过，181.4us（0.765x，最好）
        BM=16 BN=512  → 264.4us（0.525x）
        BM=32 BN=1024 → MLIRCompilationError
    即 `2 * BM * BN * 4 <= 256KB` 是能过的那条线（BM=16 BN=1024 恰好 128KB），
    双缓冲没有同时作用在两个累加器上——所以不要按
    `2(累加器) * 2(双缓冲) * BM * BN * 4` 算，那会把 BN 砍到 512、白丢 24%。

    BN 越大越好，因为 B 的 tile 越宽、搬运越连续，而且 epilogue 的 reduce
    摊到更多列上。
    """
    cap = L0C_BYTES // (2 * block_m * 4)
    cap = min(cap, 1024)
    if prefer is not None:
        if NH % prefer:
            raise ValueError(f"block_n={prefer} 不整除 NH={NH}")
        return prefer
    bn = 1
    while bn * 2 <= cap and NH % (bn * 2) == 0:
        bn *= 2
    if bn < MX_BLOCK:
        raise ValueError(f"NH={NH} 太窄或不是 32 的倍数，挑不出 BLOCK_N")
    return bn


def _check(x, weight, weight_scale, x_scale, group_list):
    if x.dtype != torch.float8_e4m3fn:
        raise ValueError(f"x 必须是 fp8_e4m3fn，实为 {x.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight 必须是 fp8_e4m3fn，实为 {weight.dtype}")
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError(f"x 要 2 维、weight 要 3 维，实为 {x.ndim}/{weight.ndim}")
    M, K = x.shape
    E, Kw, N = weight.shape
    if Kw != K:
        raise ValueError(f"K 不一致：x {K} vs weight {Kw}")
    if K % 64:
        raise ValueError(f"K 必须是 64 的倍数（e8m0 成对存储），实为 {K}")
    if N % 2:
        raise ValueError(f"N 必须是偶数（SwiGLU 分半），实为 {N}")
    if N % 64:
        # 文档「约束说明」：fp8 输入时 N 需 64 对齐
        raise ValueError(f"N 必须 64 对齐，实为 {N}")
    if group_list.numel() != E:
        raise ValueError(f"group_list 长度 {group_list.numel()} != E {E}")
    nkb = K // MX_BLOCK
    if x_scale.numel() < M * nkb:
        raise ValueError(f"x_scale 元素数 {x_scale.numel()} < M*K/32 {M * nkb}")
    if weight_scale.numel() < E * nkb * N:
        raise ValueError(
            f"weight_scale 元素数 {weight_scale.numel()} " f"< E*K/32*N {E * nkb * N}"
        )
    return M, K, E, N


def grouped_matmul_swiglu_quant_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    *,
    block_m: int = 16,
    block_n: int | None = None,
    block_k: int = 64,  # 别调大：BK=128 实测慢 9 倍，见下
    tile_threshold: int = 8,
    num_cores: int | None = None,
    weight_nz: bool | None = None,
    scale_alg: int = 0,
    b_cache: str = "",
    sb_cache: str = "",
    dump: bool = False,
    allow_slow_bk: bool = False,  # 只给探针用，见 block_k 的检查
    epi: int = 2,  # 2=融合；1/0 只给探针用（见 kernel 注释）
    split: bool = True,  # True=两 kernel（默认，两档都更快）
    quant_rows: int | None = None,
):
    """MX-FP8 GroupedMatmul + SwiGLU + MX 量化。

    对标 `torch_npu.npu_grouped_matmul_swiglu_quant_v2` 的 MXFP8 路径：

      x            fp8_e4m3fn [M, K]
      weight       fp8_e4m3fn [E, K, N]     N = 2*NH，gate|up 拼在一起
      weight_scale uint8，元素数 E*(K/32)*N  接受 4D [E,K/64,N,2] 或等价展平
      x_scale      uint8，元素数 M*(K/32)    接受 3D [M,K/64,2] 或等价展平
      group_list   int32/64 [E]
      group_list_type 0=cumsum（vllm 传这个），1=count

    返回 (y, y_scale)：
      y        fp8_e4m3fn [M, NH]
      y_scale  uint8      [M, nblk/2, 2]，nblk = NH/32

    `split=True`（默认）走两个 kernel：kernel 1 做 gmm + SwiGLU 存 bf16，
    kernel 2 在稠密的 [M, NH] 上量化。两档都更快（GLM 0.765→0.909x、
    Qwen 0.598→0.633x），原因见 kernel 里 EPI==3 分支的注释：融合时量化
    的向量工作量按 BLOCK_M 算而有效行只有 group_m。`split=False` 保留
    单 kernel 融合路径，两条路输出逐位相同（bf16 中转正是文档要求的
    量化前舍入）。

    注意 vendor 返回的 y_scale dtype 是 `float8_e8m0fnu`，这里返回 uint8——
    两者是同一个字节，比对时把 vendor 的 `view(torch.uint8)`（它不能 `.int()`，
    aclnnInplaceCopy 报 561103）。

    group_list 之外的行不写（与 vendor 一致：那部分不更新也不初始化）。
    """
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type 只能是 0 或 1，实为 {group_list_type}")
    if scale_alg not in (0, 1):
        raise ValueError(f"scale_alg 只能是 0(OCP) 或 1(cuBLAS)，实为 {scale_alg}")
    M, K, E, N = _check(x, weight, weight_scale, x_scale, group_list)
    NH = N // 2
    BN = _pick_block_n(NH, block_n, block_m)
    # BK 只能 64。交错 scale 布局 `[E,K/64,N,2]` 的索引在 BK=64 下是 N 方向
    # 1024 个相邻 pair = 2048 字节一个 burst；BK=128 起索引里的 `j//2` 开始
    # 跨 pair，退化成碎片 gather。实测（probe_tile.py，GLM-W8）：
    #   BK=64 → 181us，BK=128 → 1733us（9.6 倍），BK=256 → 编译失败
    # groupmatmul 在不带 epilogue 的同一条路上也是这个悬崖（BK=128 604us）。
    if block_k % 64:
        raise ValueError(f"block_k 必须是 64 的倍数，实为 {block_k}")
    if block_k != 64 and not allow_slow_bk:
        raise ValueError(
            f"block_k 只支持 64（BK={block_k} 实测慢一个数量级，"
            f"见 _pick_block_n 上方注释与 probe_tile.py）。"
            f"要复测这个悬崖请传 allow_slow_bk=True"
        )
    if K % block_k:
        raise ValueError(f"block_k={block_k} 不整除 K={K}")
    nz = _is_nz(weight) if weight_nz is None else weight_nz

    nkb = K // MX_BLOCK
    # A 和 A-scale 按 M+BLOCK_M 行补齐，这样内层 load 不用带 mask（实测
    # 在这条路径上 mask 值 37.5us）。补齐段的脏数据不影响结果：对应的
    # accumulator 行被 store 的 mask 丢掉。
    xp = torch.empty((M + block_m, K), device=x.device, dtype=x.dtype)
    xp[:M] = x
    # 补齐段用 empty 而不是 zeros：zeros 要额外起一个 ZerosLike kernel
    # （实测 1.1~1.4us 固定开销，Qwen 档占 4.7%），而补齐段的 scale 字节
    # 只会参与补齐行的 dot，那些 accumulator 行在 store 时被 mask 丢掉。
    # 同一个理由 xp 本来就是 empty。
    # （e8m0 的 0xFF 是 nan，但 nan 只污染它自己那一行：MX 的 scale 按
    # [行, 32列] 分块作用，不跨行。）
    sap = torch.empty((M + block_m, nkb), device=x.device, dtype=torch.uint8)
    sap[:M] = (
        x_scale.reshape(M, nkb).view(torch.uint8)
        if x_scale.dtype != torch.uint8
        else x_scale.reshape(M, nkb)
    )

    sb = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    nblk = NH // MX_BLOCK
    if split and epi == 2:
        epi = 3  # 拆开路径的 kernel 1
    y = torch.empty((M, NH), device=x.device, dtype=torch.float8_e4m3fn)
    ys = torch.empty((M, nblk), device=x.device, dtype=torch.uint8)
    if epi == 3:
        # kernel 1 存 bf16 的 SwiGLU 结果；量化交给 _mxquant_kernel。
        # 这块 bf16 中转是一次 [M,NH] 的 GM 往返（GLM 档 393KB，约 0.4us
        # 来回），换掉的是 ~30us 的 padding 浪费。
        mid = torch.empty((M, NH), device=x.device, dtype=torch.bfloat16)
    else:
        mid = y  # 不走拆开路径时不会被访问，占位即可

    if dump:
        # bf16 而不是 fp32：见 kernel 里 DUMP 分支的注释（fp32 store 会让编译器
        # abort）。诊断够用——要找的是结构性错位，不是 1 ULP。
        dg = torch.zeros((M, NH), device=x.device, dtype=torch.bfloat16)
        du = torch.zeros((M, NH), device=x.device, dtype=torch.bfloat16)
    else:
        dg = du = y  # 不走 DUMP 分支时不会被访问，占位即可

    cores = num_cores or _cube_core_num(x.device)
    _gmm_swiglu_mxquant_kernel[(cores,)](
        xp,
        sap,
        weight,
        sb,
        mid if epi == 3 else y,
        ys,
        group_list,
        GROUPS=E,
        NH=NH,
        K=K,
        N=N,
        NSP=nblk,
        GROUP_LIST_TYPE=group_list_type,
        BLOCK_M=block_m,
        BLOCK_N=BN,
        BLOCK_K=block_k,
        TILE_THRESHOLD=tile_threshold,
        MX=MX_BLOCK,
        WEIGHT_NZ=nz,
        CEIL=(scale_alg == 1),
        B_CACHE=b_cache,
        SB_CACHE=sb_cache,
        EPI=int(epi),
        DUMP=dump,
        DG=dg,
        DU=du,
    )
    if epi == 3:
        rows = quant_rows or _pick_quant_rows(M, cores, NH)
        progs = min(cores * 2, max(1, -(-M // rows)))
        _mxquant_kernel[(progs,)](
            mid, y, ys, M, NH=NH, NSP=nblk, ROWS=rows, NB=nblk, PROGS=progs
        )
    if dump:
        return y, ys.reshape(M, nblk // 2, 2), dg, du
    return y, ys.reshape(M, nblk // 2, 2)


def grouped_matmul_swiglu_quant_v2_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    scale_alg: int = 0,
):
    """纯 torch 参考，fp32 累加。精度比对的第二道判据（第一道是 vendor）。

    scale 的索引布局来自 groupmatmul/probe_scale_layout.py 的实测结论：
    x_scale 平凡展平；weight_scale 尾部 2 与 N 真交错。
    """
    M, K = x.shape
    E, _, N = weight.shape
    NH = N // 2
    nkb = K // MX_BLOCK

    sa = x_scale.reshape(M, nkb)
    sa = (sa.view(torch.uint8) if sa.dtype != torch.uint8 else sa).to(torch.int32)
    sa_f = torch.where(
        sa == 0,
        torch.zeros_like(sa, dtype=torch.float32),
        torch.pow(2.0, sa.float() - 127.0),
    )

    sb = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    sb = sb.reshape(E, nkb // 2, N, 2).to(torch.int32)
    sb = sb.permute(0, 1, 3, 2).reshape(E, nkb, N)  # (b//2, b%2) → b
    sb_f = torch.where(
        sb == 0,
        torch.zeros_like(sb, dtype=torch.float32),
        torch.pow(2.0, sb.float() - 127.0),
    )

    xf = x.to(torch.float32) * sa_f.repeat_interleave(MX_BLOCK, dim=1)
    wf = weight.to(torch.float32) * sb_f.repeat_interleave(MX_BLOCK, dim=1)

    gl = group_list.tolist()
    c = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        end = gl[e] if group_list_type == 0 else start + gl[e]
        if end > start:
            c[start:end] = xf[start:end] @ wf[e]
        start = end
    valid = start

    act, gate = c[:, :NH], c[:, NH:]
    s = torch.nn.functional.silu(act) * gate

    # MX 量化：先落 bf16，再按 32 分组
    sbf = s.to(torch.bfloat16).to(torch.float32)
    blk = sbf.reshape(M, NH // MX_BLOCK, MX_BLOCK)
    amax = blk.abs().amax(dim=-1)
    bits = amax.view(torch.int32)
    eb = ((bits >> 23) & 0xFF) - E4M3_EMAX
    if scale_alg == 1:
        eb = eb + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    ebs = torch.clamp(eb, min=0)
    ebs = torch.where(amax == 0, torch.zeros_like(ebs), ebs)
    rc = torch.pow(2.0, -(ebs.float() - 127.0)).unsqueeze(-1)
    q = torch.clamp(blk * rc, -448.0, 448.0)
    yq = q.reshape(M, NH).to(torch.float8_e4m3fn)
    return yq, ebs.to(torch.uint8), valid
