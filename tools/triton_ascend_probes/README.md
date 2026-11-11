# chunk_kda_fwd 算子在 Ascend950DT 上对编译器能力需求

## 背景介绍

FlagGems-vllm 仓库的 [chunk_kda](https://github.com/flagos-ai/FlagGems-vllm/tree/ascend_kda) 算子在 Ascend950DT 上进行了一轮优化，部分场景下性能超过 vllm-ascend 仓库调用 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)，但在长序列对齐分块场景下性能不足。

本轮重新核对了 AscendC 源码与 Triton 实现，把差距分成两类：一类是 Triton 侧的**算法/数据流选择**（Gram 用整块 64×64 vs AscendC 的因果前缀，已通过 R0 对齐）；另一类才是 **triton-ascend 相对 Ascend C 真正缺失的表达能力**，即下文的 R1–R4。

## 性能现状

AscendC 原生算子（`aclnnChunkKdaFwdV2`）之所以能把 Prepare 阶段压到 Triton 的水平附近，关键依赖三项技术：

1. **GM(ND)→L1(zN) 直通 + L1 常驻操作数**：源是 GM 上的 workspace payload，`Nd2NzParams` 直接搬进 L1，Cube 路径不经过 UB；
2. **16 行子块的因果前缀 MMAD**：每个 band 只和它的因果前缀做矩阵乘，不做整块 64×64；
3. **AIC / AIV 混核流水（CV Overlap）**：4 个 head 为一组，V1/V6 与 C2/C4 用跨核 flag 重叠。

Triton 版本已对齐三段式设计，并在本轮把 Gram 从“全矩阵 6 个 64×64 dot”改成“32 行 band + 因果前缀”的 6 个小 dot。典型配置（`T=32768, H=24, seqs=2`，总 token 数 32768）下分核耗时如下（单位：μs）：

| 算子阶段 / Kernel            | FlagGems (Triton)      | AscendC                        | 性能差异说明                 |
| ------------------------ | ----------------------:| ------------------------------:| ---------------------- |
| **Prepare**（prep + Gram） | 1353 + 1261 = **2614** | 1654 + 413 (l2norm) = **2067** | 约差 1.26×（本轮前为 1.9×） |
| **FwdH**                 | **1266**               | **840**                        | AIC MTE2 访存占比高        |
| **Finalize**             | **422**                | **718**                        | **Triton 已优于 AscendC** |
| **整算子**                 | **4299**               | **3636**                       | **0.85×**（本轮前 0.64×） |

四个基准 shape 相对 AscendC 的整算子加速比为 2.74× / 2.74× / 2.76× / 0.85×，几何平均由 1.56× 提升到 **2.05×**。

对新的 Gram 内核 Profiling 显示：`aiv_vec = 94%`，`aic_mac = 15%`。AIV 向量链仍是关键路径，AIC 大部分时间在等操作数。

> [!IMPORTANT]
> **评估结论（本轮更正）**：AscendC 的拷贝方向是 **GM(ND)→L1(zN)**，不是旧版本文档写的 UB→L1；Triton 缺的是 GM→L1 这条通路。但 `chunk_kda` 的 Gram 差距主要来自**算法与数据流**（整块 64×64 vs 因果前缀）以及**CV 重叠**，不是 L1 带宽。改用前缀分块后，Gram 由 2649 μs 降到 1261 μs，UB 峰值降到约 1.0 Mbit，**不再需要 R1 的显式 L1 常驻也能通过编译**。因此 R1 降级为 P2 通用能力需求，当前真正阻碍进一步对齐的是 **R2（向量链融合）与 R4（跨 head CV 流水）**。`causal_conv1d_fn` 侧仍保留阻断性需求 **R3**。

---

## 2. 核心需求（Gaps）—— AscendC 可做，Triton 尚不可用

### R0：因果前缀分块（已对齐，Gram 收益最大）

#### AscendC 的实现

`chunk_kda_fwd_prepare_cube.h:300-380` 不是“3 个秩项 × 2 个矩阵 = 6 个 64×64 dot”，而是：

- 16 行一个 band，共 4 个 band（`kSubChunkRows = 16`、`kSubChunkCount = 4`）；
- 每个 band 只和它的**因果前缀** `N ∈ {16, 32, 48, 64}` 做 MMAD（`kPrefixRows`）；
- 把 Q+ 与 K+ 在 M 维堆成 32 行，一次 `[32,128] @ [128,N]` 同时得到 Aqk / Akk 各 16 行；
- 4 个 band 合计约 **655k MAC/chunk-head**（Aqk + Akk 合计）。

旧 Triton 公式是 6 × 64 × 64 × 128 = **3.15M MAC**，约 4.8×。

#### Triton 的对齐实现

`src/flaggems_vllm/runtime/backend/_ascend/FLA/chunk_kda.py::_ascend_kda_fwd_gram_prefix_kernel` 保留 32 行 anchor，但按前缀拆分：

- band 0：`[32,128] @ [128,32]`（q/k 各一次）；
- band 1：`[32,128] @ [128,32]`（前缀 0）+ `[32,128] @ [128,32]`（前缀 1），q/k 各两次；
- 6 个 dot 合计 **786k MAC**（4.0× 降幅），面板降为 7 个 `[32,128]` bf16（56 KiB）；
- 默认 `NEUMANN_STAGES = 0` 时 `(I+L)^-1 = I - L` 精确成立，Akk 的四个象限可直接写出，不需要拼回 64×64；
- anchor 改为每个 band **有效部分**的中点（与 AscendC `gRef` 一致），同时修掉了旧实现在 33–47 行尾 chunk 上锚点落到 `T` 之外、`kM * exp2(a1-a0)` 溢出为 NaN 的潜在问题。

实测（`T=32768, H=24, seqs=2`）：

| 项目 | 改前 | 改后 |
| --- | ---: | ---: |
| Gram | 2649 μs | **1261 μs**（2.10×） |
| 整算子 | 5712 μs | **4299 μs** |
| 相对 AscendC | 0.64× | **0.85×** |
| 四 shape 几何平均 | 1.56× | **2.05×** |

正确性 4/4 PASS（`tools/test_kda_vs_vllm_ascend.py --ops chunk_kda --skip-perf`），BT=32/64 均验证。剩余差距是 `aiv_vec = 94%` / `aic_mac = 15%`，即 AIV 向量链与 CV 重叠，见 R2 与 R4。

---

### R1：GM(ND)→L1(zN) 直通（P2，已降级；chunk_kda 不再依赖）

#### 更正：AscendC 记录的是 GM→L1，不是 UB→L1

旧版本把 R1 描述为“UB→L1 的 ND→zN 转换”，与源码不符。`chunk_kda_fwd_prepare_cube.h:258-290` 的

```cpp
AscendC::Nd2NzParams qkCopy{};
...
AscendC::DataCopy(qPlusL1, payload[ScorePayload::kQPlus / sizeof(bfloat16_t)], qkCopy);
```

源 `payload` 是 V6 先写入 workspace 的 **GM** 地址，`DataCopy` 直接把 GM(ND) 搬成 L1(NZ)；Cube 路径自始至终不经过 UB。

Triton 的对应限制是：

- `al.copy` / `copy_from_ub_to_l1` 只接受 `bl.buffer`，且要求 src 在 UB、dst 在 L1/UB；
- `ascend_address_space` 只有 `L1 / UB / L0A / L0B / L0C`，没有 GM；
- `tl.load` 的结果总是落在 UB，因此构造 L1 上的 4D zN 只能先物化 UB，再 UB→L1。

#### 为什么 `g1` 的 `l1*` 会 UB 溢出，以及为什么它不再阻断

`gaps/g1_nd2nz_staging.py` 的 `l1*` 模式复现的是**旧的全矩阵 7 面板公式**：7 个 `[64,128]` bf16 面板（112 KiB）+ q/k/g 基础 tile，再加 ND→zN 转换的临时量，峰值 **2.50 Mbit**，超过 **2.03 Mbit** 上限。

这个溢出是公式选择导致的，不是“必须 L1 常驻”导致的。改成 R0 的前缀分块后，面板变成 7 个 `[32,128]`（56 KiB），基础 tile 约 72 KiB，峰值约 **1.0 Mbit**，无需任何显式 L1 常驻即可编译通过，且 Gram 快 2.10×。

因此 R1 从 P0 阻断性降为 **P2 通用能力需求**：GM→L1 直通仍是 Triton 相对 AscendC 缺失的能力（对别的多面板 kernel 仍有价值），但 **chunk_kda 的性能对齐不再依赖它**。`g1` 探针保留，用于继续追踪显式 L1 常驻的 UB 预算和 GM→L1 缺口。

---

### R2：长 `exp2/select` 链的向量融合（P1）

#### 需求描述

将 Gate 阶段的长依赖链（`sigmoid -> cumsum -> 2×exp2 -> 4×where -> 5×mul`）融合成少数几条向量微码指 令，避免每个 `tl` 基础操作都发生一次 UB 寄存器与内存的读写往返。

#### AscendC 证据（确认具备硬件能力）

源码参考：`chunk_kda_fwd_prepare_vec.h:1128, 1146, 1412`
通过 `AscendC::VF_CALL<Detail::StageV0Vf<...>>` 与 `StageV6Vf` 直接调用手写深度融合的向量计算核。

#### Triton 现状与实测分析（`gaps/g2_vec_chain_fusion.py`）

在全网格（12288 个 Program）上测试依赖链 `a = exp2(a * 1e-4) + where(a > 0.5, 1, 0)` 的扩展表现：

| 依赖链长度 N | 执行耗时 (μs)  | 边际耗时增量     | 编译与融合状态                  |
| -------:| ----------:| ----------:| ------------------------ |
| 0       | ~256 – 276 | —          | 基线                       |
| 4       | ~271       | ~0 μs/op   | **已有效融合**（几乎无额外开销）       |
| 8       | ~794       | ~65 μs/op  | **性能断崖**（融合被打破）          |
| 12      | ~1478      | ~101 μs/op | 线性累积（折合约 0.53 μs/op/AIV） |

- **编译选项表现**：开启 `vf_fusion_mode=all-op` 会直接触发编译崩溃；设置 `max-parallel` 或 `ub-aware-op` 时效果与默认一致。

#### 期望改进

支持超长链的分段自动融合（如每 4–8 个操作划为一个融合块），并修复 `vf_fusion_mode=all-op` 的编译崩溃问题。

本轮前缀分块落地后，Gram 的 `aiv_vec` 仍为 **94%**。即使把面板的 `exp2` 从 7 次降到 2 个 band + 1 个 cross 比值，向量链和每个 `tl` 操作在 UB 上的往返仍是关键路径；R2 的收益不再只是 `exp2` 本身。

### R4：跨 head 的 AIV/AIC 流水（P1，当前剩余主差距）

#### 需求描述

AscendC Prepare 以 **4 个 local head 为一组**：V1 生成 Q+/K+ 时，C2 在算上一组的 MMAD；V6 生成 RHS 时，C4/C5 已在读 L1 常驻数据。两侧用 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 和双缓冲槽位显式重叠，AIC 等操作数时 AIV 不会空闲。

#### Triton 现状（API 已有，最小 hand-off 已验证）

`al.scope("vector"|"cube")` + `al.sync_block_set/wait` 已经可以表达 CV 双缓冲，见 `supported/s6_scoped_cv_handoff.py`（PASS，maxerr 6.1e-5）。前提是编译选项 `enable_mixed_cv=True, disable_auto_inject_block_sync=True`；默认选项仍会崩溃（`defects/d1_scope_auto_sync.py`）。

但把这套用进 Gram 的 7 面板 / 2-chunk 双缓冲时，原型仍有两个问题：

- 面板写到 GM 后逐位正确（与 torch 参考 maxerr=0），但 cube scope 消费时得到错值（band 0 Aqk 误差约 0.026，band 1 约 17，Akk 出现 inf），说明 scope 内 `sync_block_wait` 之后的 GM load 排序/可见性还不完整；
- 即使不考虑正确性，T=4096 下原型 344 μs vs 拆分 169 μs，2× 更慢：每个 chunk-head 多出约 56 KiB 的 GM 写 + 56 KiB 读，而可被隐藏的 cube 时间只有约 190 μs（T=32768 时）。没有 GM→L1 直通（R1）时，这个往返成本吃掉了重叠收益。

新 Gram 的单 program 仍是 `AIV(面板) -> AIC(dot) -> AIV(后处理)`；`aiv_vec=94%`、`aic_mac=15%` 说明 AIC 大部分时间在等操作数。

#### AscendC 证据（确认具备硬件能力）

源码参考：`chunk_kda_fwd_prepare_cube.h:130-230`

- `ProcessChunkHeadRange` 每次处理 4 个 head，`kAivToAicPayloadReadyFlagId` / `kAicToAivSlotReusableFlagId` 构成 free/ready 双缓冲；
- `StageC2` 完成后立即 `CrossCoreSetFlag(scoreReady)`，V3 的 Aqk 后处理与 C4/C5 的下一 head 计算重叠。

#### 期望改进

- 修复 scope 内“wait 之后的 GM load 可见性/排序”，让 `s6` 的模式能安全扩展到多面板；
- 支持 GM→L1 直通（R1），让 cube 侧不再经过 UB，降低双缓冲的往返成本；
- 或由编译器在相邻 program/迭代之间自动做 CV 重叠。

验收口径：Gram 微基准的 `aic_mac_ratio` 提升到 50% 以上，或 Gram 端到端进入 600–800 μs 区间（当前 1261 μs，AscendC 413 μs）。

### R3：流水化 tile 的行偏移复用（P0，阻断性）

#### 需求描述

`causal_conv1d_fn` 这类 depthwise 卷积在每个输出 tile 上需要 `W` 个 tap（`x[t-W+1..t]`）。AscendC 只从 GM 读一份 tile，`W` 个 tap 是同一份 UB 缓冲的行偏移视图；Triton 里对应的写法是一次 `load` + `extract_slice` 取行偏移。但当前在 `num_stages=2` + auto multi-buffer 下，这类视图会读到旧版本缓冲（确定性错值，maxerr 约 16），换 `bind_buffer` 或先 `al.copy` 到独立 UB 缓冲也只是"有时正确"。结果是生产 kernel 只能按行偏移重读 `W` 次，MTE2 流量从 1.375x 变成 5x。

#### AscendC 证据（确认具备硬件能力）

源码参考：`causal_conv1d_tiling_planner.h` 的 `RunSeq` / `RunSeqFnRolling`
每行输入只从 GM 读一次，`W` 个 tap 在 UB 内按行偏移复用（滚动状态 `s_k` 只用于跨 chunk 的边界），没有重复 GM 访问。`flash-linear-attention-npu` 的因果卷积正是靠这一点领先。

#### Triton 现状与实测分析（`gaps/g3_pipelined_ub_view_shift.py`、`defects/d4_extract_slice_pipeline.py`）

生产配置（`C=9216, T=8192, BLOCK_N=2304, BLOCK_T=4, 32 split`）：

| 写法 | ns=1, mb=False | ns=2, mb=True | 耗时 (ns=2) |
| --- | --- | --- | ---: |
| `read4`（四次 GM 读，现网做法） | — | — | 245–273 us |
| 裸 `extract_slice(tile)` | PASS（0） | **FAIL**（maxerr≈16） | 186–202 us |
| `bind_buffer` + 视图 | PASS（0） | 不稳定（本轮 PASS / 另一轮 FAIL） | 202–238 us |
| `al.copy` 到独立 UB 缓冲 | PASS（0） | 不稳定（本轮 FAIL / 另一轮 PASS） | 201–274 us |

- 关掉流水/多缓冲后三种写法都逐位正确，但耗时 272–280 us，已慢于重读，"单读"的收益消失；
- 把能正确工作的写法搬进 production kernel（带 prologue、tail、多 range 网格）后，96 个回归用例中 20–32 个算错，因此尚未上线。

#### 期望改进

- auto multi-buffer / 软件流水对 `extract_slice`、`insert_slice` 的视图做别名与版本跟踪，保证 `num_stages=2` 下读到的是当前迭代版本；
- 或提供显式的"行偏移复用"原语（UB 缓冲上的静态行偏移视图，由编译器保证版本正确）。

验收口径：`gaps/g3_pipelined_ub_view_shift.py` 的两种移位写法在 `num_stages=2, multibuffer=True` 下都与 `read4` 逐位一致，且不慢于 `read4`。

#### 收益评估

`causal_conv1d_fn` 在 KDA prefill 形状上 MTE2 流量从 5x 降到 1.375x：token-major 目标约 111 us（现 155 us），channel-first 目标约 120 us（现 232 us）。后者是当前在诚实输入契约（channel-first 连续 layout）下唯一仍落后 AscendC 的路径（0.70x，几何平均 0.75x）；R3 落地后预计翻转为 1.3x 以上。

## 探针目录结构与验证指南

### 目录组织

```text
tools/triton_ascend_probes/
├── README.md                      # 本文档
├── run_all.sh                     # 一键运行全部探针脚本
├── supported/                     # 【已支持】验证可用能力，作为参考基线
│   ├── s1_slice_dot.py            # 行/列切片作为 dot 操作数（含 L1 zN）
│   ├── s2_l1_resident_dot.py      # 4D zN + al.dot(fractal) 显式 L1 驻留
│   ├── s3_l1_subview_assemble.py  # subview + copy 拼装 32 行子块
│   ├── s4_manual_cross_core_sync.py # 手工 sync_block_* 跨核同步
│   └── s6_scoped_cv_handoff.py    # al.scope + sync_block 的 CV 双缓冲 hand-off
├── gaps/                          # 【核心需求】Triton 缺失、AscendC 可做
│   ├── g1_nd2nz_staging.py        # R1：GM→L1 缺口 / 显式 L1 的 UB 预算验证
│   ├── g2_vec_chain_fusion.py     # R2：长向量链融合与性能断崖验证
│   └── g3_pipelined_ub_view_shift.py # R3：流水下单读 + tile 行偏移复用
└── defects/                       # 【缺陷】编译器 Bug，已有绕行方案
    ├── d1_scope_auto_sync.py      # D1：scope 崩溃与同步错误
    ├── d2_insert_slice_dot.py     # D2：insert_slice 作 dot 操作数崩溃
    ├── d3_l1_view_2d.py           # D3：2D ND 在 L1 上 view/to_tensor 失败
    └── d4_extract_slice_pipeline.py # D4：单读 + UB 切片移位读到旧版本缓冲
```

### 运行方式

- **一键全量测试**（在仓库根目录下执行）：

  ```shell
  bash tools/triton_ascend_probes/run_all.sh
  ```

- **单项探针测试**（示例）：

  ```shell
  PYTHONPATH=src:FlagGems/src python tools/triton_ascend_probes/gaps/g1_nd2nz_staging.py l1a
  ```

### 预期执行结果清单

| 分组            | 测试命令与参数                                       | 预期执行表现                                     |
| ------------- | --------------------------------------------- | ------------------------------------------ |
| **supported** | `s1_slice_dot.py col / row / l1_col / l1_row` | **PASS**（maxerr ≤ 7.7e-5）                  |
| **supported** | `s2_l1_resident_dot.py dot / copy`            | **PASS**（maxerr ≤ 6.1e-5）                  |
| **supported** | `s3_l1_subview_assemble.py`                   | **PASS**（maxerr ≤ 6.1e-5）                  |
| **supported** | `s4_manual_cross_core_sync.py`                | **PASS**（maxerr ≤ 1.5e-5）                  |
| **supported** | `s6_scoped_cv_handoff.py`                     | **PASS**（maxerr ≤ 6.1e-5；需 `enable_mixed_cv=True, disable_auto_inject_block_sync=True`） |
| **gaps**      | `g1_nd2nz_staging.py plain`                   | **PASS**（基线正常通过）                           |
| **gaps**      | `g1_nd2nz_staging.py l1a / l1b / l1`          | **FAIL**：UB 内存溢出（3.15 / 3.28 / 2.50 Mbit）  |
| **gaps**      | `g2_vec_chain_fusion.py`                      | N=4 已融合，N=8/12 出现显著耗时断崖                    |
| **gaps**      | `g3_pipelined_ub_view_shift.py`               | **GAP**：`read4` 245–273 us；单读+行偏移 1.3–1.4x，但两种写法只有时正确（判定为 0/2–2/2，随编译变化） |
| **defects**   | `d1_scope_auto_sync.py scope`                 | **FAIL**：SIGSEGV 崩溃                        |
| **defects**   | `d1_scope_auto_sync.py scope_noauto`          | 编译通过，但计算数值错误（maxerr ≈ 1e2）                 |
| **defects**   | `d2_insert_slice_dot.py insert / l1_insert`   | **FAIL**：`Unknown PIPE!` 触发 SIGABRT        |
| **defects**   | `d3_l1_view_2d.py nd / target`                | **FAIL**：报 `cbuf vs ub` 或 `unsupported op` |
| **defects**   | `d4_extract_slice_pipeline.py`                | **FAIL**：`ns=2` 且开多缓冲时裸视图读到旧版本（maxerr≈16，确定性）；bind/copy 写法时对时错 |

> [!NOTE]
> `g1` 的 `l1*` 是**旧全矩阵 7 面板公式**的 UB 预算探针，不是“L1 常驻必然溢出”的证据。生产 Gram 已改用 R0 的 32 行前缀分块（7×`[32,128]` 面板，峰值约 1.0 Mbit），无需 R1 的显式 L1 常驻即可通过编译。AscendC 的对应拷贝是 **GM→L1**；`g1` 里的 UB→L1 只是 Triton 当前唯一可写的路径。

生产验证（Ascend950DT，CANN 25.1.rc2 / triton-ascend 3.6.0）：

```shell
# 功能：4/4 PASS
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops chunk_kda --skip-perf
# 性能：T=32768 4299 us vs AscendC 3636 us（0.85x），四 shape 几何平均 2.05x
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops chunk_kda
# 分核：prep 1353 / Gram 1261 / FwdH 1266 / Finalize 422 us
BENCH_PERKERNEL=1 PYTHONPATH=src:FlagGems/src python tools/_kda_shape_bench.py chunk 3
```
