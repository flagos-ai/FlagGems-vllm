# chunk_kda_fwd 算子在 Ascend950DT 上对编译器能力需求

## 背景介绍

FlagGems-vllm 仓库的 [chunk_kda](https://github.com/flagos-ai/FlagGems-vllm/tree/ascend_kda) 算子在 Ascend950DT 上进行了一轮优化，部分场景下性能超过 vllm-ascend 仓库调用 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)，但在长序列对齐分块场景下性能不足。

目前定位到是 triton-ascend 相比 Ascend C 的表达能力限制，导致无法复刻 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) 中的计算逻辑。因此，对 FlagTree 编译器提出 特性需求。

## 性能现状

AscendC 原生算子（`aclnnChunkKdaFwdV2`）之所以能将 Prepare 阶段的耗时压减至 Triton 的一半左右，关键 依赖三项技术：

1. **L1 常驻操作数**：避免频繁进出 GM，大幅缓解 UB 读写压力；
2. **32 行子块门控**：细粒度子视图就地拼装；
3. **AIC / AIV 混核流水（CV Overlap）**：计算与搬运高密度掩盖。

Triton 版本当前已严格对齐三段式设计，并实现了源码级 AIV/AIC 重叠。在典型配置（`T=32768, H=24, seqs=2`）下，分核耗时对比如下（单位：μs）：

| 算子阶段 / Kernel            | FlagGems (Triton)      | AscendC                        | 性能差异说明                 |
| ------------------------ | ----------------------:| ------------------------------:| ---------------------- |
| **Prepare**（prep + Gram） | 1353 + 2649 = **4002** | 1654 + 413 (l2norm) = **2067** | **主要差距 所在（约差 1.9×）**   |
| **FwdH**                 | **1266**               | **840**                        | AIC MTE2 访存占比达 73%     |
| **Finalize**             | **421**                | **718**                        | **Triton 已优于 AscendC** |

对融合版 Prepare 进行 Profiling 剖析显示：`aiv_vec = 75.4%`，`aic_mac = 6.2%`。这表明 **AIV 向量指令依赖链是唯一的关键路径瓶颈**。

> [!IMPORTANT]
> **评估结论**：经过边界划分与技术解耦，真正阻碍性能对齐的**编译器核心需求仅剩 R1 与 R2**。其余能力 均已验证支持（第 1 节），或属于已有可用绕行方案的软件缺陷（第 3 节）。

---

## 2. 核心需求（Gaps）—— AscendC 可做，Triton 尚不可用

### R1：L1 操作数的 ND→zN 暂存直通（P0，阻断性）

#### 需求描述

需要支持将 2D ND 格式的 bf16 面板直接搬运为 L1 上的 4D zN 布局，供后续 `al.dot(..., format_*="fractal")` 多次复用，**避免在 UB 中额外物化并膨胀出第二份 4D Tile 显存**：

```python
# 期望能够直接将 UB/GM 的 2D ND buffer 复制为 L1 上的 4D zN 结构
l1 = bl.alloc(tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1)
al.copy(bl.to_buffer(a_nd_2d, al.ascend_address_space.UB), l1)  # ND(UB) -> zN(L1)
c = al.dot(bl.to_tensor(l1), tb, format_a="fractal", format_b="fractal")
```

#### AscendC 证据（确认具备硬件能力）

源码参考：`kda/chunk_kda_fwd_prepare/op_kernel/arch35/chunk_kda_fwd_prepare_cube.h`

- **L258–290**：通过 `AscendC::Nd2NzParams qkCopy` + `AscendC::DataCopy(qPlusL1, ...)`，将 Q+ / K+ / K- 按照 **GM(ND) → L1(NZ)** 一次性搬运入 L1（占用 72 KiB），源码注释明确写明：*“后面的四次 MMAD 不 再读 GM”*；
- **L423–452**：使用 `DataCopy` 生成标准 64×64 zN 布局，并按 32×32 象限将 B / X0 / negX1 直接写入 L1 的 `(M1=2, N1=0)` 槽位。

#### Triton 现状与实测分析（`gaps/g1_nd2nz_staging.py`）

针对真实 KDA Gram 阶段网格（12288 个 Program，包含 7 个 `[64, 128]` 面板、6 次 dot 计算）：

| 模式              | L1 暂存策略            | UB 内存需求                         | 实测结果                      |
| --------------- | ------------------ | -------------------------------:| ------------------------- |
| `plain`         | 无（常规重复访存）          | ~2.00 Mbit                      | **PASS**（同结构微基准约 1.84 ms） |
| `l1a`           | 暂存 4 个 A 面板（无转置）   | 3.15 Mbit                       | **FAIL：UB overflow 溢出**   |
| `l1b`           | 暂存 3 个共享 B 面板（含转置） | 3.28 Mbit                       | **FAIL：UB overflow 溢出**   |
| `l1`            | 暂存全部 7 个面板         | 2.50 Mbit                       | **FAIL：UB overflow 溢出**   |
| **2-chunk 完整版** | 14 个面板（真实双缓冲复现）    | 3.94 Mbit（L1 cbuf 另需 6.29/4.19） | **FAIL：UB 严重溢出**          |
| **硬件可用上限**      | —                  | **2.03 Mbit**                   | —                         |

- **问题根因**：当前 Triton 无法支持直接完成 ND 到 zN 的格式转换，只能在 UB 侧通过 `reshape` + `permute` + `trans` 手动构造，相当于在有限的 UB 空间里重复物化出整份 4D Tile。同时，若尝试将 2D ND 直接放入 L1，调用 `bl.to_tensor(l1)` 时又会报地址空间冲突（见缺陷 D3）。

#### 收益评估

缺少 R1，L1 常驻方案只能局限于单块 micro-kernel；KDA Gram 的 7 面板多操作数复用以及 2-chunk 双缓冲流水均无法落地。R1 解决后：

- Gram 阶段耗时预计由 **2649 μs 下降至 1900–2100 μs**；
- 算子整体耗时预计由 **5712 μs 压降至 5000–5200 μs**（提升至原先的 0.68–0.70x）。

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
│   └── s4_manual_cross_core_sync.py # 手工 sync_block_* 跨核同步
├── gaps/                          # 【核心需求】Triton 缺失、AscendC 可做
│   ├── g1_nd2nz_staging.py        # R1：ND→zN 暂存与 UB 内存预算验证
│   └── g2_vec_chain_fusion.py     # R2：长向量链融合与性能断崖验证
└── defects/                       # 【缺陷】编译器 Bug，已有绕行方案
    ├── d1_scope_auto_sync.py      # D1：scope 崩溃与同步错误
    ├── d2_insert_slice_dot.py     # D2：insert_slice 作 dot 操作数崩溃
    └── d3_l1_view_2d.py           # D3：2D ND 在 L1 上 view/to_tensor 失败
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
| **gaps**      | `g1_nd2nz_staging.py plain`                   | **PASS**（基线正常通过）                           |
| **gaps**      | `g1_nd2nz_staging.py l1a / l1b / l1`          | **FAIL**：UB 内存溢出（3.15 / 3.28 / 2.50 Mbit）  |
| **gaps**      | `g2_vec_chain_fusion.py`                      | N=4 已融合，N=8/12 出现显著耗时断崖                    |
| **defects**   | `d1_scope_auto_sync.py scope`                 | **FAIL**：SIGSEGV 崩溃                        |
| **defects**   | `d1_scope_auto_sync.py scope_noauto`          | 编译通过，但计算数值错误（maxerr ≈ 1e2）                 |
| **defects**   | `d2_insert_slice_dot.py insert / l1_insert`   | **FAIL**：`Unknown PIPE!` 触发 SIGABRT        |
| **defects**   | `d3_l1_view_2d.py nd / target`                | **FAIL**：报 `cbuf vs ub` 或 `unsupported op` |

