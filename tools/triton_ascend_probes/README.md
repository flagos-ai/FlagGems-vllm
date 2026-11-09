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

AscendC 主要依靠 **L1 常驻操作数**、**32 行子块门控** 以及 **AIC/AIV 混核跨 chunk 流水** 将 Prepare 阶段耗时压制到 Triton 的一半以内。下文详述阻碍 Triton 实现上述优化的 5 项核心编译器能力缺口。

## 需求与现状速览

### 待解决的编译器问题清单

| 编号     | 核心特性                      | 现状表现                                                         | 阻塞的算子优化                                    | 验证探针脚本                                                                                                                         |
|:------ |:------------------------- |:------------------------------------------------------------ |:------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------ |
| **F1** | 显式 L1 驻留的 dot 操作数         | **前端 API 完备，但 lowering 阶段报错**（关键阻碍：`bl.to_tensor` 生成错误的地址空间） | 无法实现 AscendC 式 L1 常驻面板，产生大量冗余搬运            | [`probe_01_l1_resident_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_01_l1_resident_dot.py)             |
| **F2** | `insert_slice` 拼装 dot 操作数 | **BiShengIR 崩溃** (`LLVM ERROR: Unknown PIPE!` / SIGABRT)     | 无法实施 32 行子块门控计算（导致向量工作量翻倍）                 | [`probe_03_insert_slice_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_03_insert_slice_dot.py)           |
| **F3** | `scope` 混核声明与自动跨核同步       | **默认配置下 SIGSEGV**；关闭自动同步后产生脏读导致精度错误                          | 无法通过声明式 `scope` 构建 AIC/AIV 混核流水            | [`probe_04_scope_sync.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_04_scope_sync.py)                       |
| **F4** | 跨 chunk 软件流水与容量预算         | **编译期空间溢出**（UB 溢出 5.57 Mbit，L1 溢出 6.29 Mbit）                 | 无法做跨 chunk 流水来掩盖 Cube 等待开销（影响约 0.5~1.0 ms） | [`probe_06_cross_chunk_l1_budget.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_06_cross_chunk_l1_budget.py) |
| **F5** | 长 `exp2/select` 链的向量融合    | **短链 (N≤4) 可融合，长链 (N≥8) 出现性能断崖**                             | 无法折叠复杂的门控因子计算链，UB 往返开销严重                   | [`probe_05_vec_chain_fusion.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_05_vec_chain_fusion.py)           |

### 已确认可用能力（供编译器团队参考排除）

| 特性能力                                       | 验证现状                                     | 对应探针                                                                                                                               |
|:------------------------------------------ |:---------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------- |
| 列偏移 `extract_slice` 作为 dot 操作数             | **PASS**（精度误差 $\le 7.6 \times 10^{-6}$）  | [`probe_02_slice_as_dot_operand.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_02_slice_as_dot_operand.py) `col` |
| 行偏移 `extract_slice` 作为 dot 操作数             | **PASS**（精度误差 $\le 5.3 \times 10^{-5}$）  | [`probe_02_slice_as_dot_operand.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_02_slice_as_dot_operand.py) `row` |
| 手工插入 `al.sync_block_set/wait`（不使用 `scope`） | **PASS**（精度误差 $\le 1.5 \times 10^{-5}$ ） | [`probe_04_scope_sync.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_04_scope_sync.py) `noscope_sync`            |
| `al.copy(UB buffer → L1 buffer)` 前端类型检查    | **通过**（语法受支持，受阻于后端 lowering，见 F1）        | [`probe_01_l1_resident_dot.py`](file:///workspace/FlagGems/tools/triton_ascend_probes/probe_01_l1_resident_dot.py) `copy`          |

## 详细需求与分析

### F1：显式 L1 驻留的 dot 操作数

#### 1. 业务场景与诉求

在 AscendC Prepare 实现中，Q+/K+/K- 操作数被保存在 L1 中，供后续 6 个点积操作多次复用，无需在每次计 算时反复执行 UB $\rightarrow$ L1 staging 搬运。
在 Triton 中的理想书写模式如下：

```python
ub = bl.alloc(tl.bfloat16, [BT, K], al.ascend_address_space.UB)
l1 = bl.alloc(tl.bfloat16, [BT, K], al.ascend_address_space.L1)
al.copy(bl.to_buffer(a, bind_buffer=ub), ub)   # SSA Tensor -> UB Buffer
al.copy(ub, l1)                                 # UB -> L1
at = bl.to_tensor(l1)                           # 构造 L1 视图
c = tl.dot(at, tl.trans(at))                    # Cube 单元直接读取 L1 数据
```

#### 2. 现状与错误分析

前端语法虽然完整，但 lowering 阶段存在多处阻断：

| 调用方式                                                    | 错误表现                                                                                                                                                                                                          |
|:------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `al.copy(UB buffer → L1 buffer)`                        | 前端类型检查通过（`semantic.copy` 允许 src=UB, dst $\in$ {UB, L1}）                                                                                                                                                       |
| 使用 `bind_buffer=ub` 将 SSA Tensor 写入 UB，再搬运至 L1          | `'memref.alloc' op error: read before first write`                                                                                                                                                            |
| 使用位置参数 `bl.to_buffer(a, space=UB)` 填充 UB                | `'hivm.hir.pointer_cast' op addrs of PointerCastOp should not be empty!`                                                                                                                                      |
| 尝试通过 `tl.store` 写入 `bl.to_tensor(ub, writable=True)` 视图 | `Unsupported ptr type <['64','128'], fp32> in tl.store`                                                                                                                                                       |
| **`bl.to_tensor(l1)`（关键阻塞点）**                           | `different memory spaces specified for base memref type 'memref<64x128xbf16, #hivm.address_space<cbuf>>' and subview memref type 'memref<1x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>'` |

> [!IMPORTANT]
> `bl.to_tensor` 在 lowering 阶段为 L1 buffer 错误生成了带有 **UB** 地址空间的 `memref.subview`，与 基底类型的 `cbuf` 空间冲突。这是当前最关键的阻塞点。

#### 3. 期望行为

- `bl.to_tensor(l1_buffer)` 应正确输出带有 **cbuf 地址空间**的 Tensor，可直接作为 `tl.dot` 的输入操 作数；或提供显式接口（如 `al.dot(..., a_in_l1=...)`）。
- `memref.alloc` 的活跃性分析能够识别 `al.copy`，避免误报 read-before-write。

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

#### 3. 期望行为

- 支持 `insert_slice` 的拼接结果直接作为 `tl.dot` 操作数，并在可能的情况下复用 layout。
- 若必须执行 layout 转换且不支持时，应在编译期抛出语义清晰的诊断错误，而非底层 SIGABRT 崩溃。

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
| `python probe_01_l1_resident_dot.py copy`     | **FAIL**: 报 `memref.alloc ... read before first write` |
| `python probe_01_l1_resident_dot.py dot`      | **FAIL**: 报 `different memory spaces ... cbuf vs ub`   |
| `python probe_02_slice_as_dot_operand.py col` | **PASS**: 精度误差 $\le 7.6 \times 10^{-6}$                |
| `python probe_02_slice_as_dot_operand.py row` | **PASS**: 精度误差 $\le 5.3 \times 10^{-5}$                |
| `python probe_03_insert_slice_dot.py`         | **FAIL**: 报 `LLVM ERROR: Unknown PIPE!` 并触发 SIGABRT    |
| `python probe_04_scope_sync.py noscope_sync`  | **PASS**: 手工标志同步，精度误差 $\le 1.5 \times 10^{-5}$         |
| `python probe_04_scope_sync.py scope`         | **FAIL**: 编译器段错误 SIGSEGV                               |
| `python probe_04_scope_sync.py scope_noauto`  | 编译通过但结果错误（`maxerr ≈ 1e2`）                              |
| `python probe_04_scope_sync.py scope_sync`    | **PASS**: 带 scope 并显式手工同步                              |
| `python probe_05_vec_chain_fusion.py`         | 验证 N=4 融合与 N=8/12 的耗时突变                                |
| `python probe_06_cross_chunk_l1_budget.py`    | 简化模型 PASS；真实 kernel 触发 `cbuf overflow`                 |

## 优化优先级建议

按投入产出比与实施可行性排序：

1. **P0 - F1（显式 L1 操作数视图）**：仅需在 lowering 阶段修正 `bl.to_tensor` 在 L1 buffer 上的地址 空间推导逻辑。改动范围小，能直接解锁多点积共享 L1 面板的优化策略，性能收益最为显著。
2. **P1 - F2（`insert_slice` 拼装 dot 操作数）**：修复底层 SIGABRT，支持常规 lowering 或给出友好报错，可直接减少门控计算中 50% 的向量负载。
3. **P1 - F3（`scope` 自动同步与稳定性）**：解决默认开启 `scope` 时的编译器崩溃，消除算子代码中硬编 码手工事件同步的需求。
4. **P2 - F4（软件流水溢出调度与预算分配）**：使能跨 chunk 软件流水以掩盖 Cube 计算等待，提升整体硬 件利用率。
5. **P3 - F5（长向量链分段融合）**：优化通用长计算链的生成质量，进一步缩减 AIV 执行耗时。

