# chunk_kda_fwd 算子在 Ascend950DT 上对编译器能力需求

## 背景介绍

FlagGems-vllm 仓库的 [chunk_kda](https://github.com/flagos-ai/FlagGems-vllm/tree/ascend_kda) 算子在 Ascend950DT 上进行了一轮优化，部分场景下性能超过 vllm-ascend 仓库调用 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)，但在长序列对齐分块场景下性能不足。

目前定位到是 triton-ascend 相比 Ascend C 的表达能力限制，导致无法复刻 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) 中的计算逻辑。因此，对 FlagTree 编译器提出特性需求。

## 优化目标

KDA chunked prefill 在原生 AscendC（`aclnnChunkKdaFwdV2`）中分为 `Prepare → FwdH → Finalize` 三阶段 。其中 Prepare 阶段通过手写 AIC + 2×AIV 混核流水、4-head 乒乓调度以及 L1 驻留 Q+/K+/K- 面板实现了极 致的延迟隐藏。

在 FlagGems Triton 版本中，我们已对齐三阶段结构并完成了源码级 AIV/AIC 重叠。在典型评测形状（$T=32768, H=24, \text{seqs}=2$）下的分阶段耗时对比如下（单位：$\mu s$）：

| Kernel 阶段                           | FlagGems (Triton) | 原生 AscendC            | 差距分析与现状                |
|:----------------------------------- |:-----------------:|:---------------------:|:---------------------- |
| **Prepare**（Triton 拆分为 prep + Gram） | 1353 + 2649       | 1654 + 413 (含 l2norm) | **主要性能差距来源**（落后约 2 倍）  |
| **FwdH**                            | 1266              | 840                   | AIC MTE2 访存占比 73%      |
| **Finalize**                        | 421               | 718                   | **Triton 已优于 AscendC** |

性能剖析（Level 1，拆分前的融合版 Prepare）表明：`aiv_vec_ratio = 75.4%`，`aic_mac_ratio = 6.2%`。这说明 **AIV 向量指令计算链是当前的核心关键路径**，Cube 单元大部分时间处于空闲等待状态。

AscendC 主要依靠 **L1 常驻操作数**、**32 行子块门控** 以及 **AIC/AIV 混核跨 chunk 流水** 将 Prepare 阶段耗时压制到 Triton 的一半以内。下文详述阻碍 Triton 实现上述优化的 4 项核心编译器能力缺口，以及 1 项已经澄清的 API 用法问题（F1）。

## 需求与现状速览

### 待解决的编译器问题清单

| 编号     | 核心特性                      | 现状表现                                                         | 阻塞的算子优化                                    | 验证探针脚本                                                                                                                         |
|:------ |:------------------------- |:------------------------------------------------------------ |:------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------ |
| **F1** | 显式 L1 驻留的 dot 操作数         | **已澄清为 API 用法问题**：旧写法 `bl.to_tensor(l1, target_shape)+tl.dot` 仍失败；改用 4D zN 布局 + `al.dot(..., format_a/b="fractal")` 后可正常运行 | 已可实现 AscendC 式 L1 常驻面板，避免冗余搬运            | [`probe_01_l1_resident_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_01_l1_resident_dot.py)             |
| **F2** | `insert_slice` 拼装 dot 操作数 | **仍崩溃** (`LLVM ERROR: Unknown PIPE!` / SIGABRT)；给拼装结果显式 L1 地址空间也不能修复。可用 `bl.subview + al.copy + al.dot(fractal)` 规避 | 无法实施 32 行子块门控计算（导致向量工作量翻倍）                 | [`probe_03_insert_slice_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_03_insert_slice_dot.py)           |
| **F3** | `scope` 混核声明与自动跨核同步       | **默认配置下 SIGSEGV**；关闭自动同步后产生脏读导致精度错误                          | 无法通过声明式 `scope` 构建 AIC/AIV 混核流水            | [`probe_04_scope_sync.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_04_scope_sync.py)                       |
| **F4** | 跨 chunk 软件流水与容量预算         | **编译期空间溢出**（UB 溢出 5.57 Mbit，L1 溢出 6.29 Mbit）                 | 无法做跨 chunk 流水来掩盖 Cube 等待开销（影响约 0.5~1.0 ms） | [`probe_06_cross_chunk_l1_budget.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_06_cross_chunk_l1_budget.py) |
| **F5** | 长 `exp2/select` 链的向量融合    | **短链 (N≤4) 可融合，长链 (N≥8) 出现性能断崖**                             | 无法折叠复杂的门控因子计算链，UB 往返开销严重                   | [`probe_05_vec_chain_fusion.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_05_vec_chain_fusion.py)           |
| **F6** | L1 操作数的 ND→zN 暂存         | **只有 UB 侧 permute 一条路**：7 面板 kernel 需 2.37–3.28 Mbit UB（可用 2.03）；2D ND 的 L1 视图仍报 cbuf/ub | 使 F1 无法用于 KDA Gram，进而卡住 F4 流水 | [`probe_07_zn_staging_budget.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_07_zn_staging_budget.py) |

### 已确认可用能力（供编译器团队参考排除）

| 特性能力                                       | 验证现状                                     | 对应探针                                                                                                                               |
|:------------------------------------------ |:---------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------- |
| 列偏移 `extract_slice` 作为 dot 操作数             | **PASS**（精度误差 $\le 1.2 \times 10^{-5}$）  | [`probe_02_slice_as_dot_operand.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_02_slice_as_dot_operand.py) `col` |
| 行偏移 `extract_slice` 作为 dot 操作数             | **PASS**（精度误差 $\le 7.7 \times 10^{-5}$）  | [`probe_02_slice_as_dot_operand.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_02_slice_as_dot_operand.py) `row` |
| 手工插入 `al.sync_block_set/wait`（不使用 `scope`） | **PASS**（精度误差 $\le 1.5 \times 10^{-5}$ ） | [`probe_04_scope_sync.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_04_scope_sync.py) `noscope_sync`            |
| 显式 L1 驻留 dot（4D zN + `al.copy` + `al.dot(fractal)`） | **PASS**（精度误差 $\le 6.1 \times 10^{-5}$） | [`probe_01_l1_resident_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_01_l1_resident_dot.py) `dot` |
| deprecated `copy_from_ub_to_l1` + `al.dot(fractal)` | **PASS**（精度误差 $\le 4.6 \times 10^{-5}$，有 deprecation warning） | [`probe_01_l1_resident_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_01_l1_resident_dot.py) `copy` |
| 显式 L1 缓存 + 切片/子视图作为 dot 操作数 | **PASS**（`l1_col` $\le 7.6 \times 10^{-6}$，`l1_row` $\le 6.1 \times 10^{-5}$） | [`probe_02_slice_as_dot_operand.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_02_slice_as_dot_operand.py) `l1_col` / `l1_row` |
| `bl.subview` + `al.copy` 拼装 L1 dot 操作数 | **PASS**（精度误差 $\le 6.1 \times 10^{-5}$） | [`probe_03_insert_slice_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_03_insert_slice_dot.py) `l1_copy` |

## 详细需求与分析

### F1：显式 L1 驻留的 dot 操作数（已澄清为 API 用法问题）

#### 1. 业务场景与诉求

在 AscendC Prepare 实现中，Q+/K+/K- 操作数被保存在 L1 中，供后续 6 个点积操作多次复用，无需在每次计算时反复执行 UB $\rightarrow$ L1 staging 搬运。

当前 triton-ascend 已经提供 L1 驻留 dot 的完整路径（`al.dot` 的 `fractal` 布局参数），正确写法如下：

```python
# A [M, K] -> zN 布局 [K//16, M//16, 16, 16]（bf16；fp32 的 block 为 16x8）
a_nz = tl.permute(a.reshape(M, K // 16, 16), (1, 0, 2))
a_nz = a_nz.reshape(K // 16, M // 16, 16, 16)

l1 = bl.alloc(tl.bfloat16, [K // 16, M // 16, 16, 16], al.ascend_address_space.L1)
al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1)
at = bl.to_tensor(l1)  # 保持 4D zN，不要传 target_shape
c = al.dot(at, bt, format_a="fractal", format_b="fractal")
```

- A 操作数（`A [M,K]`）的 zN 布局是 `[K//16, M//16, 16, 16]`；
- B 操作数（`B [K,N]`）的 zN 布局是 `[N//16, K//16, 16, 16]`；
- 同一逻辑矩阵作为左/右操作数时物理布局不同，不能共用同一块 L1 面板；
- `al.dot` 的 `format_a` / `format_b` / `format_c` 取 `"fractal"` / `"nd"` / `""`，分别对应 zN 与 ND 布局。

#### 2. 旧写法的错误分析

| 调用方式                                                    | 错误表现                                                                                                                                                                                                          |
|:------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 使用 `bind_buffer=ub` 将 SSA Tensor 写入 UB，再搬运至 L1          | `'memref.alloc' op error: read before first write`。`bind_buffer` 只是别名，liveness 分析没有看到对 UB 的真实写入，与 `al.copy` 本身无关。 |
| `bl.to_tensor(l1, target_shape=[M, K])` + `tl.dot`      | `different memory spaces ... cbuf vs ub` 或 `LLVM ERROR: unsupported op to get out operands`。`target_shape` 路径会生成不支持的 `hivm.hir.convert_layout`。 |
| `tl.store` 写入 `bl.to_tensor(ub, writable=True)` 视图      | `Unsupported ptr type ... in tl.store`                                                                                                                                                                     |

> [!IMPORTANT]
> 旧探针的结论"L1 驻留 dot 不可用"是误判。真正的阻塞点是消费端：`bl.to_tensor(l1, target_shape=...)` 与 `tl.dot` 都不理解 L1 的 zN 布局。改用 `bl.to_tensor(l1)`（保留 4D）+ `al.dot(..., format_*="fractal")` 后，L1 操作数可以直接被 Cube 消费。

#### 3. 关于 `copy_from_ub_to_l1`

`al.copy_from_ub_to_l1` 只是 `al.copy` 的 deprecated 别名，语义完全相同；它本身并不能修复旧写法的 lowering 问题。把消费端改成 4D zN + `al.dot(fractal)` 后，`al.copy` 与 `al.copy_from_ub_to_l1` 都能正常工作（后者会打印 deprecation warning）。因此：

- **推荐**：`al.copy` + `bl.to_tensor(l1)` + `al.dot(fractal)`；
- **兼容**：`al.copy_from_ub_to_l1` 也能跑通，但不应在新代码中使用；
- **仍失败**：`bl.to_tensor(l1, target_shape=[M,K])` + `tl.dot`（见 `probe_01 ... old`）。

#### 4. 期望行为

- 保留并文档化 `al.dot` 的 `fractal` 路径，补充 L1 驻留 dot 的官方示例；
- 修复 `bl.to_tensor(l1, target_shape=...)` 在 L1 buffer 上的地址空间推导，或给出清晰的编译期诊断；
- `memref.alloc` 的活跃性分析应能识别 `al.copy` 与 `bind_buffer`，避免误报 read-before-write。

> [!NOTE]
> zN 布局的切片方向受物理布局限制：B 布局的 N 维位于第 0 维（最外层），按行切片是安全的；A 布局的 M 维位于 block 内层，直接对 A 面板做行切片虽然能编译运行，但会得到错误的物理视图。`probe_02 ... l1_row` 因此采用 "B 面板按行切片 + 每个行块单独分配 A 面板" 的方式；列方向（K 维）切片不受此限制。

### F2：`insert_slice` 拼装 dot 操作数

#### 1. 业务场景与诉求

在门控计算中，因子按 32 行子块（ScoreRefToken span）计算即可满足精度要求，能减少一半向量工作量。我们期望在 `[32, K]` 粒度计算向量操作，再使用 `al.insert_slice` 拼装为完整块送入 `tl.dot`：

```python
full = al.insert_slice(tl.zeros((64, K), tl.bfloat16), top, [0, 0], [32, K], [1, 1])
full = al.insert_slice(full, bot, [32, 0], [32, K], [1, 1])
c = tl.dot(full, tl.trans(full))
```

#### 2. 现状与错误分析

当 `insert_slice` 的拼接结果被用作 `tl.dot` 的输入时，BiShengIR 发生内部崩溃：

```text
[ConvertLinalgIRToBinary] encounters error:
LLVM ERROR: Unknown PIPE!
... died with <Signals.SIGABRT: 6>.
```

*(注：若 `insert_slice` 的结果仅用于 `tl.store`，则可正常执行。)*
若改用 `tl.join + tl.permute + tl.reshape` 规避，编译器会插入 layout 格式转换指令，实测耗时增加约 4 倍（Gram 核由 2649 $\mu s$ 恶化至 10000+ $\mu s$），不可接受。

**显式 L1 地址空间能否解决？** 不能。`probe_03 ... l1_insert` 把 `insert_slice`
写进显式 L1 的 4D zN tensor，再交给 `al.dot(..., format_a/b="fractal")`，仍然
报同一个 `LLVM ERROR: Unknown PIPE!`。因此这不是"结果落在哪个地址空间"的
问题，而是 `insert_slice` 生成的 SSA 拼接结果进入 Cube 管线时的 lowering 缺陷。

#### 3. 可用替代路径

`probe_03 ... l1_copy` 给出了一条已验证可用的路径：不拼装 SSA tensor，而是在
L1 上预先分配 A / B 两种 zN 布局的面板，用 `bl.subview` 定位上下子块，再用
`al.copy` 从 UB 写入：

```python
l1a = bl.alloc(tl.bfloat16, [K // 16, M // 16, 16, 16], al.ascend_address_space.L1)
sub_top = bl.subview(l1a, [0, 0, 0, 0],
                     [K // 16, (M // 2) // 16, 16, 16], [1, 1, 1, 1])
sub_bot = bl.subview(l1a, [0, (M // 2) // 16, 0, 0],
                     [K // 16, (M // 2) // 16, 16, 16], [1, 1, 1, 1])
al.copy(bl.to_buffer(top_nz, al.ascend_address_space.UB), sub_top)
al.copy(bl.to_buffer(bot_nz, al.ascend_address_space.UB), sub_bot)
c = al.dot(bl.to_tensor(l1a), bl.to_tensor(l1b),
           format_a="fractal", format_b="fractal")
```

该路径不需要 `insert_slice`，也不会触发 `join + permute + reshape` 的 layout
转换；实测精度误差 $\le 6.1 \times 10^{-5}$。

#### 4. 期望行为

- 支持 `insert_slice` 的拼接结果直接作为 `tl.dot` / `al.dot` 操作数，并在可能的情况下复用 layout；
- 若必须执行 layout 转换且不支持时，应在编译期抛出语义清晰的诊断错误，而非底层 SIGABRT 崩溃；
- 在修复前，文档应明确推荐 `bl.subview + al.copy + al.dot(fractal)` 的 L1 子块 staging 方案。

### F3：`scope` 混核声明与自动跨核同步

#### 1. 业务场景与诉求

通过 `al.scope("cube")` 与 `al.scope("vector")` 在单一算子内声明 Cube 与 Vector 的职责划分，由编译器自动分析依赖并插入跨核事件同步（对齐 AscendC 的 C2/C4/C5/C7 流水模式）：

```python
with al.scope("cube"):
    cb = tl.dot(a, tl.trans(b))
    tl.store(gm, cb)
with al.scope("vector"):
    tl.store(out, tl.load(gm) * 2.0)
```

#### 2. 现状与实测对比

| 执行模式           | 编译配置参数                                                      | 实测结果                                    |
|:-------------- |:----------------------------------------------------------- |:--------------------------------------- |
| `scope`        | 默认配置                                                        | **SIGSEGV (11)**（崩溃于 bishengir）         |
| `scope_noauto` | `enable_mixed_cv=True, disable_auto_inject_block_sync=True` | 编译通过，但**数值 错误**（`maxerr ≈ 1e2`，读入脏数据）   |
| `scope_sync`   | 同上 + 手工插入 `al.sync_block_set/wait`                          | **PASS**（精度误差 $\le 1.5 \times 10^{-5}$） |
| `noscope_sync` | 默认配置 + 仅手工插入 `al.sync_block_set/wait`（无 scope）              | **PASS**（精度误差 $\le 1.5 \times 10^{-5}$） |

> [!NOTE]
> 当前文档注释要求 `sync_block_*` 必须包含在对应的 `scope` 内；但实测在**完全不声明 `scope`** 的情况下直接使用手工同步标志反而能稳定运行（目前 GDN 算子即采用此临时规避方案）。

#### 3. 期望行为

- 修复默认配置下启用 `scope` 导致的编译器 SIGSEGV 问题。
- 在 `disable_auto_inject_block_sync=True` 时，若存在未显式同步的跨核数据流，提供编译期诊断警告。
- 明确并统一文档与实现中 `sync_block_*` 对 `scope` 上下文的依赖规则。

### F4：跨 chunk 软件流水与 L1/UB 容量预算

#### 1. 业务场景与诉求

在 chunk $c$ 的 6 个点积计算在 Cube 单元执行期间，由 Vector 单元提前生成 chunk $c+1$ 的门控因子与面 板数据，形成 AIC/AIV 间的乒乓软件流水，隐藏 Cube 的计算延迟。

#### 2. 现状与瓶颈

| 尝试方案                               | 实测结果与瓶颈分析                                                                       |
|:---------------------------------- |:------------------------------------------------------------------------------- |
| **动态 CV 流水** (`tl.range` + 动态流水属性) | 触发 UB 容量超限：`ub overflow, requires 5573120 bits while 1769472 bits available`    |
| **源码级 2-tile 展开流水**（同时保活两块的操作数与因子） | 触发 L1 容量超限：`cbuf overflow, requires 6291456 bits while 4194304 bits available!` |
| **顺序展开** (`CPT=2`，无并发保活)           | 编译通过但性能下降：Gram 由 2649 $\mu s$ 增加至 **3311 $\mu s$**（后处理操作仍然阻塞了后续 chunk 的计算）      |
| **当前单块实现**（无跨块流水）                  | Gram 耗时 2649 $\mu s$；剖析显示 AIV 利用率达 75.4%，AIC 利用率仅 6.2%，Cube 等待无法掩盖              |

#### 3. 期望行为

- 软件流水调度优化：当 UB 容量承载多缓冲受限时，支持将操作数平滑下溢（spill）至 L1（作为显式 staging buffer），而非直接在 UB 内多份复制导致溢出。
- 允许精细化指定多缓冲策略（目前 `multibuffer=False` 为全局开关，粒度过粗）。

### F5：长 `exp2/select` 链的向量融合

#### 1. 业务场景与诉求

门控因子计算包含连贯的向量运算链（$\text{sigmoid} \rightarrow \text{cumsum} \rightarrow 2\times\text{exp2} \rightarrow 4\times\text{where} \rightarrow 5\times\text{mul}$）。期望编译器将其自动融合成若 干连续计算的向量指令序列，消除逐指令读写 UB 的往返开销。

#### 2. 现状与性能断崖实测

使用探针 [`probe_05_vec_chain_fusion.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_05_vec_chain_fusion.py) 在全网格（12288 个 program，对齐典型形状）上测试计算链 `a = exp2(a * 1e-4) + where(a > 0.5, 1, 0)` 的执行表现：

| 连续操作数 (N) | 全网格耗时         | 边际耗时 (每新增 1 个 op) | 融合状态分析                          |
|:---------:|:-------------:|:-----------------:|:------------------------------- |
| **0**     | 255 $\mu s$   | —                 | 基准耗时                            |
| **4**     | 271 $\mu s$   | ~4 $\mu s$ / op   | **已成功融合**                       |
| **8**     | 794 $\mu s$   | ~67 $\mu s$ / op  | **出现性能断崖**（融合放弃，退化为逐 op 往返 UB）  |
| **12**    | ~1478 $\mu s$ | ~101 $\mu s$ / op | 稳定在每 op 每 AIV 约 0.53 $\mu s$ 开销 |

针对 `vf_fusion_mode` 编译开关的实测反馈：

- `all-op`: 直接引发编译失败；
- `max-parallel` 与 `ub-aware-op`: 与默认配置性能一致，未产生改善效果。

#### 3. 期望行为

- 改进向量融合策略：对于超长计算链，支持分段自适应融合（例如每 4~8 个 op 划分为一个融合块），避免超 过阈值后整体放弃优化。
- 修复 `vf_fusion_mode=all-op` 的编译异常。


### F6：L1 操作数的 ND→zN 暂存预算

#### 1. 业务场景与诉求

F1 已经确认 L1 驻留 dot 可用，但验证只在单块 `[64,128]` 上做过。KDA Gram 核每个
program 要建 **7 个** `[64,128]` bf16 面板并做 6 个 dot；要享受 L1 常驻（并为 F4 的
双 chunk 流水腾出 UB），必须把每个 ND 面板转成 zN 4D 再搬进 L1。

#### 2. 现状：**唯一构造 zN 的方式是 UB permute，代价放不下**

用 [`probe_07_zn_staging_budget.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_07_zn_staging_budget.py)
在真实 Gram 网格（12288 program）上实测：

| 模式 | 暂存到 L1 的面板 | UB 需求 | 结果 |
|:--|:--|--:|:--|
| `plain` | 无（6 个 `tl.dot`） | ~2.0 Mbit | **PASS**（同结构微基准约 1.84 ms / 12288 program） |
| `l1a` | 4 个 A 面板（无 trans） | 3.15 Mbit | ub overflow |
| `l1b` | 3 个共享 B 面板（含 trans） | 3.28 Mbit | ub overflow |
| `l1` | 全部 7 个面板 | 2.50 Mbit | ub overflow |
| 2-chunk 变体 | 14 个面板 | 3.94 Mbit | ub overflow |

zN 只能这样构造（没有别的前端入口）：

```python
a_tmp = tl.permute(a.reshape(R, K // 16, 16), (1, 0, 2))
a_nz = a_tmp.reshape(K // 16, R // 16, 16, 16)   # 第二份 4D tile 落在 UB
al.copy(bl.to_buffer(a_nz, al.ascend_address_space.UB), l1)
```

也就是说 F1 的“可用”只在单块/少面板场景成立；一旦按 Gram 的 7 面板规模使用，
UB 就超了。**2D ND 直通 L1 也不行**——`bl.to_tensor(l1)`（2D）仍然报
`different memory spaces ... cbuf vs ub`，这条路径没有便宜的替代。

#### 3. 期望行为

- 提供融合的 ND→zN 暂存：例如 `al.copy` 接受 2D ND 的 UB 源和 4D zN 的 L1 目标
  （等价 AscendC 的 `ND2NZ` DataCopy），不在 UB 里物化第二份 4D tile；
- 或 `al.dot` 接受 2D ND 操作数 + 内部 L1 暂存，复用编译器已有的（UB 友好的）
  ND→zN 转换；
- 修复 2D ND 的 `bl.to_tensor(l1)` 地址空间推导（这是最省事的旁路）。

#### 4. 影响

在此修复前，F1 的 L1 常驻无法用于 KDA Gram，F4 的跨 chunk 双缓冲也就没有
可行的落地形态（显式 L1 + 双 chunk 需要 3.94 Mbit UB）。这是当前从“API 可用”
到“算子可用”之间唯一的阻塞点。

## 测试与验证指南

### 快速一键验证

在仓库根目录下执行批处理脚本，可依次运行所有探针并打印关键日志：

```shell
bash tools/triton_ascend_probes/run_all.sh
```

### 单项探针运行与预期一览

```shell
# 环境准备
export PYTHONPATH="src:FlagGems/src"
cd tools/triton_ascend_probes
```

| 探针命令                                          | 预期运行结果（本机实测）                                           |
|:--------------------------------------------- |:------------------------------------------------------ |
| `python probe_01_l1_resident_dot.py dot`      | **PASS**: L1 zN + `al.dot(fractal)`，精度误差 $\le 6.1 \times 10^{-5}$ |
| `python probe_01_l1_resident_dot.py copy`     | **PASS**: deprecated `copy_from_ub_to_l1` + `al.dot(fractal)`，精度误差 $\le 4.6 \times 10^{-5}$（有 warning） |
| `python probe_01_l1_resident_dot.py old`      | **FAIL**（预期）: 旧 `to_tensor(target_shape)+tl.dot` 报 `unsupported op to get out operands` |
| `python probe_02_slice_as_dot_operand.py col` | **PASS**: UB 基线，精度误差 $\le 1.2 \times 10^{-5}$ |
| `python probe_02_slice_as_dot_operand.py row` | **PASS**: UB 基线，精度误差 $\le 7.7 \times 10^{-5}$ |
| `python probe_02_slice_as_dot_operand.py l1_col` | **PASS**: 显式 L1 + col 切片，精度误差 $\le 7.6 \times 10^{-6}$ |
| `python probe_02_slice_as_dot_operand.py l1_row` | **PASS**: 显式 L1 + 行切片，精度误差 $\le 6.1 \times 10^{-5}$ |
| `python probe_03_insert_slice_dot.py l1_copy` | **PASS**: L1 subview + `al.copy` + `al.dot(fractal)`，精度误差 $\le 6.1 \times 10^{-5}$ |
| `python probe_03_insert_slice_dot.py insert`  | **FAIL**: `insert_slice` + `tl.dot` 报 `LLVM ERROR: Unknown PIPE!` |
| `python probe_03_insert_slice_dot.py l1_insert` | **FAIL**: 显式 L1 地址空间仍报 `Unknown PIPE!` |
| `python probe_04_scope_sync.py noscope_sync`  | **PASS**: 手工标志同步，精度误差 $\le 1.5 \times 10^{-5}$         |
| `python probe_04_scope_sync.py scope`         | **FAIL**: 编译器段错误 SIGSEGV                               |
| `python probe_04_scope_sync.py scope_noauto`  | 编译通过但结果错误（`maxerr ≈ 1e2`）                              |
| `python probe_07_zn_staging_budget.py plain` | **PASS**：不暂存 L1 的 6-dot 基线 |
| `python probe_07_zn_staging_budget.py l1a` | **FAIL**: ub overflow 3.15 Mbit（可用 2.03） |
| `python probe_07_zn_staging_budget.py l1b` | **FAIL**: ub overflow 3.28 Mbit |
| `python probe_07_zn_staging_budget.py l1` | **FAIL**: ub overflow 2.50 Mbit |
| `python probe_04_scope_sync.py scope_sync`    | **PASS**: 带 scope 并显式手工同步                              |
| `python probe_05_vec_chain_fusion.py`         | 验证 N=4 融合与 N=8/12 的耗时突变                                |
| `python probe_06_cross_chunk_l1_budget.py`    | 简化模型 PASS；真实 kernel 触发 `cbuf overflow`                 |

## 优化优先级建议

按投入产出比与实施可行性排序：

1. **已关闭 - F1（显式 L1 操作数视图）**：不是编译器缺口。使用 4D zN 布局 + `al.copy` + `al.dot(format_*="fractal")` 已可正常工作；建议补充官方文档与示例，并修复 `bl.to_tensor(l1, target_shape=...)` 的地址空间推导。
2. **P0 - F6（L1 操作数的 ND→zN 暂存）**：这是把 F1 从“API 可用”变成“算子里可用”的唯一阻塞点。当前 7 面板 Gram 需要 2.37–3.28 Mbit UB（可用 2.03），显式 L1 + 双 chunk 需要 3.94 Mbit。修好后 F1 才能落地，F4 才有可行的双缓冲形态。
3. **P0 - F2（`insert_slice` 拼装 dot 操作数）**：修复底层 SIGABRT，支持常规 lowering 或给出友好报错，可直接减少门控计算中 50% 的向量负载。修复前可用 `bl.subview + al.copy + al.dot(fractal)` 规避。
4. **P1 - F3（`scope` 自动同步与稳定性）**：解决默认开启 `scope` 时的编译器崩溃，消除算子代码中硬编码手工事件同步的需求。
5. **P2 - F4（软件流水溢出调度与预算分配）**：使能跨 chunk 软件流水以掩盖 Cube 计算等待，提升整体硬件利用率。
6. **P3 - F5（长向量链分段融合）**：优化通用长计算链的生成质量，进一步缩减 AIV 执行耗时。
