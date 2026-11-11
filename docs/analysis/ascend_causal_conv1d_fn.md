# Ascend 上 causal_conv1d_fn：fla_npu AscendC 方案、FlagGems-vllm 实现与性能差距

> 适用平台：Ascend950DT（arch35 / DAV_3510，64 个 AIV 核），CANN 25.1.rc2，
> PyTorch 2.10 + torch_npu，Triton 3.6.0（triton-ascend）。
> 数据来自 `tools/test_kda_vs_vllm_ascend.py --ops conv_prefill`
> （`torch_npu.profiler`，warmup=5 / active=5，取设备 kernel 时间）。

## 0. 摘要

| | fla_npu AscendC | FlagGems-vllm（重写后） |
|---|---|---|
| 语言/运行方式 | AscendC C++ kernel，`__aicore__`，arch35 走 RegBase(MicroAPI) | Triton-Ascend kernel |
| 并行划分 | host tiling：`baseDim × tokenBlockSize`，grid ≤ coreNum（一波） | host planner：token 段 × channel tile，grid ≈ coreNum |
| 每行 x 的读取次数 | 1（寄存器内移位） | W=4（同一 tile 按行偏移重读，靠 L2 吸收） |
| MTE2 流量（读+写） | 2 × 数据量 | 5 × 数据量（4 读 + 1 写） |
| 4 个 benchmark shape 的加速比 | 1.00x（基准） | 0.85x / 0.75x / 0.37x / 0.88x，几何平均 **0.68x** |

重写前 FlagGems 侧为 0.44x / 0.65x / 0.33x / 0.44x（几何平均 0.45x）。
**结论：AscendC 与 FlagGems 的差距基本等于「每行 x 读 1 次 vs 读 4 次」的
MTE2 流量差**；Triton 侧没有寄存器级移位 API，只能用重读换取正确性。

> 注：第 0–6 节是重写后（0.68x）的快照。第 7 节是后续优化：把通道 tile
> 做宽并整除 `dim` 后几何平均 **1.14–1.15x**（155.5 / 155.0 / 117.8 / 583.6 µs），
> 4 个 shape 中 3 个超过 AscendC。
> 第 7 节的数字基于「token-major 视图」输入；第 8 节改为真实 contract
> （channel-first 连续输入）、并把转置融进 kernel，融合后 0.72x，
> 而不融合是 0.48x。

## 1. 算子语义与调用路径

### 1.1 数学定义

逐通道（depthwise）因果一维卷积，`W = width`（Kimi K3 KDA 取 4）：

```
y[t, n] = bias[n] + Σ_{j=0..W-1} w[j, n] · x[t - (W-1) + j, n]
```

`x` 来自 varlen 连续批处理：多个请求的 token 首尾相接，`query_start_loc`
给每个请求的起点/终点。序列第一个 token 的窗口会越过序列起点，越过部分来自
`conv_state`（前缀缓存里的 W-1 个历史 token），没有历史时补 0。卷积结束后要
把序列末尾的 W-1 个 token 写回 `conv_state`，供后续 decode/前缀缓存使用。
激活为 SiLU（`x·sigmoid(x)`）。

### 1.2 两条调用路径

```
vLLM → vllm_ascend/ops/kimi_kda.py::_run_causal_conv1d
       → fla_npu.ops.ascendc.causal_conv1d_fn        （AscendC，本文基准）

flaggems_vllm.causal_conv1d_fn
       → src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py
                                                    （Triton-Ascend，本文重写对象）
```

### 1.3 布局契约

| | AscendC（fla_npu） | FlagGems（沿用上游 vLLM 契约） |
|---|---|---|
| x | `[T, dim]` token-major | `[dim, T]` channel-first（内部 `transpose(0,1).contiguous()` 成 `[T, dim]`） |
| weight | `[W, dim]` | `[dim, W]`（内部转置） |
| conv_state | `[pool, W-1, dim]`，dim stride = 1 | `[pool, dim, W-1]`，dim stride = 1（`transpose(1,2)` 视图即可互转，零拷贝） |
| 输出 | 同 x | 同 x（`empty_like`） |

FP32 累加、BF16/FP16 存取；`conv_states` 的 dtype 决定计算 dtype。

### 1.4 benchmark 形态

`head_dim=128`、`dim = 3·H·128`、`W=4`、bf16，TP=8 下 Kimi K3 的 prefill 扫点：

| # | T（总 token） | seqs | dim | 数据量（读+写） |
|---|---|---|---|---|
| 1 | 8192 | 2 | 9216 | 302 MB |
| 2 | 16384 | 4 | 4608 | 302 MB |
| 3 | 24576 | 8 | 2304 | 227 MB |
| 4 | 32768 | 2 | 9216 | 1208 MB |

---

## 2. fla_npu 的 AscendC 方案

代码：`flash-linear-attention-npu/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/`
（`op_host/causal_conv1d_tiling_planner.h`、`op_kernel/causal_conv1d.h`、
`op_kernel/causal_conv1d_fn_tasks.h`、`op_kernel/arch35/causal_conv1d_regbase.h`）。

整体是标准 AscendC 两段式：**host tiling 决定网格，device 每个核处理一个
`tokenBlockSize × baseDim` 的二维任务，内部用 ring buffer 流式跑完整段 token。**

### 2.1 主机侧 tiling：先 UB 预算，再核数

关键常量（`causal_conv1d_tiling_utils.h`、`causal_conv1d_common.h`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `MAX_WIDTH` | 4 | 支持的最大卷积宽度（W > 4 直接不处理） |
| `MAX_DIM_TILE_SIZE` | 4096 | 单 tile 通道上限 |
| `MAX_BLOCK_DIM` | 4096 | UB 中一行通道的上限 |
| `RING_SLOTS` | 5 | ring buffer 槽数 |
| `DIM_ALIGN_BYTES` | 32 | 通道分块对齐 |
| `FN_UB_RESERVED_BYTES` | 512 | UB 预留 |
| `RING_SLOT_CNT` / `FN_OUT_SLOT_CNT` / `FN_CALC_FP32_SLOT_CNT` | 5 / 2 / 8 | UB 每元素摊销 5×2 + 2×2 + 8×4 = **46 B/element** |

```cpp
// tiling_planner.h:92
inline int64_t ComputeFnUbLimitedBaseDim(uint64_t ubSize)
{
    const int64_t bytesPerElem = (RING_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_OUT_SLOT_CNT * BF16_FP16_ELEM_BYTES) +
                                 (FN_CALC_FP32_SLOT_CNT * sizeof(float));   // 46
    const int64_t budgetBytes = ubSize - FN_UB_RESERVED_BYTES;
    return std::min<int64_t>(MAX_DIM_TILE_SIZE,
                             AlignDownInt64(budgetBytes / bytesPerElem, DIM_ALIGN_ELEMS));
}
```

两条 case（`ChooseFnHostPlan`，tiling_planner.h:261）：

1. **token-first**（`dim ≤ 4096`）：`baseDim = dim`、`baseDimCnt = 1`，
   `tokenCoreBudget = coreNum`（64）。整行一次搬完。
2. **token-dim co-split**（`dim > 4096`）：`baseDim = ubLimitedBaseDim`（=4096），
   再把 `baseDimCnt` **微调成 coreNum 的因子**，并让 `baseDim` 不超 UB/上限：

```cpp
// tiling_planner.h:113
int64_t adjustedBaseDimCnt = result.baseDimCnt;
while (adjustedBaseDimCnt < coreNum && (coreNum % adjustedBaseDimCnt != 0)) ++adjustedBaseDimCnt;
const int64_t adjustedBaseDim = AlignUpInt64(CeilDivInt64(dim, adjustedBaseDimCnt), DIM_ALIGN_ELEMS);
```

token 轴按 **cuSeqlen（所有请求拼起来的总长）**分块，而不是按请求分：

```cpp
// tiling_planner.h:244
tokenBlockSize = CeilDivInt64(cuSeqlen, tokenCoreBudget);   // tokenCoreBudget = coreNum / baseDimCnt
tokenBlockCnt  = CeilDivInt64(cuSeqlen, tokenBlockSize);
gridSize       = tokenBlockCnt * baseDimCnt;                 // ≤ coreNum → 一波
```

因此一个 token tile 可能跨多个请求；host 用 `BuildFnTokenSeqRangePlan`
预先算出每个 tile 覆盖的 `[startSeq, endSeq)` 写进 tiling data
（`tokenTileStartSeq/tokenTileEndSeq`），device 只做一次查表，不再二分查找
（`ResolveExplicitTokenTileSeqRange`；缺失时回落到 `FindVarlenSeqByToken`）。

四个 benchmark shape 的 tiling 结果（`coreNum = 64`）：

| dim | T | baseDim | baseDimCnt | tokenBlockSize | grid |
|---|---|---|---|---|---|
| 9216 | 8192 | 2304 | 4 | 512 | 64 |
| 4608 | 16384 | 4096（第 2 块只有 512 通道） | 2 | 512 | 64 |
| 2304 | 24576 | 2304 | 1 | 384 | 64 |
| 9216 | 32768 | 2304 | 4 | 2048 | 64 |

即**无论 shape 如何，恒为 64 个 program = 一波**，每个核拿到一段连续 token
× 一段连续通道。

### 2.2 设备侧：任务、ring buffer、权重

```cpp
// causal_conv1d_fn_tasks.h:236  ProcessVarlenTokenTiled()
const auto blockTask = ResolveFnDirectBlockTask(blockIdx, tokenBlockCnt, tokenBlockSize, ...);
LoadWeightAndBias(blockTask.channelStart, blockTask.baseDimSize);
int32_t cursor = blockTask.tokenStart;
while (cursor < blockTask.tokenEnd && seq < seqUpperBound) {   // 跨请求
    ... ResolveSeqTaskWindow(seq, ...) 取该请求的 [seqStart, seqStart+curSeqLen)
    ProcessFnChunk(seq, cacheIdx, hasInit, seqStart, curSeqLen, cursor, tileLen, ...);
    cursor = tileEnd; ++seq;
}
```

每个核：

* UB 内 **5 槽 ring**（`RING_SLOTS`，每槽 `MAX_BLOCK_DIM` 元素）+ 输出 2 槽 +
  fp32 计算区；权重和 bias 一次性搬入并 **cast 成 fp32**，
  后续每个 tap 直接 `MulAddDst` 到 fp32 累加器。
* ring 的填充用 `DataCopyParams` **一次搬 2 个 token**（`kTokensPerCopy = 2`），
  减少 DMA 描述符数量；MTE2↔V 用 `SetFlag/WaitFlag` 事件对实现真正的
  "预取下一批、计算当前批" 流水。
* 段首历史（`InitRingSeqSplit`）：越过序列起点的 token 分别从
  **x**（同请求内前一段）/ **conv_state**（有初始状态）/ **0** 三者取。

### 2.3 滚动递推（rolling fast path）

无 bias、非投机解码时走 `RunSeqFnRolling`（`causal_conv1d.h:813`）。核心是把
W 个 tap 的"乘加历史行"改写成 **W-1 个寄存器常驻部分和** 的递推
（`w_row[i]` 是权重第 i 行，`h_i = x[t0-(W-1)+i]`）：

```
初始化（RestoreFnLocalPartials, :559）
    s_k = Σ_{i=0..W-2-k} w_row[i] · h_{i+k}          k = 0..W-2

每个 token（ComputeFnRollingOutput :630 / AdvanceFnLocalPartials :667）
    y(t)   = s_0 + w_row[W-1] · x[t]                 （再 Silu）
    s_k(t) = w_row[W-2-k] · x[t] + s_{k+1}(t-1)      k = 0..W-2, s_{W-1} ≡ 0
```

以 W=4 展开即 `AdvanceFnLocalPartials` 里的三条 `Mul` + 两条 `Add`：

```cpp
Mul(state0F, currF, weightF[(w0Idx + 2) * MAX_BLOCK_DIM], baseDim);  // x_t · w_row[2]
Add(state0F, state0F, state1F, baseDim);
Mul(state1F, currF, weightF[(w0Idx + 1) * MAX_BLOCK_DIM], baseDim);
Add(state1F, state1F, state2F, baseDim);
Mul(state2F, currF, weightF[ w0Idx      * MAX_BLOCK_DIM], baseDim);  // x_t · w_row[0]
```

要点：

* **每个 x 行只从 UB 取一次**（`ComputeFnRollingOutput` 读 `ring[slotCurr]`），
  而通用路径 `RunSeq`（`causal_conv1d.h:463`）每个输出 token 要把 W 个历史槽
  各取一次，等于每行在 UB 内被读 W 次。
* 向量指令数并不比通用路径少（约 9–10 条 baseDim 宽的操作/token），
  真正的收益是 **UB 读取次数**、状态常驻寄存器，以及便于 2-token 批 DMA 与
  输出双缓冲。
* 有 bias 时 `IsFnRollingFastPathEnabled()` 返回 false，退到 `RunSeq` 的
  通用逐 tap 实现（`W×(Cast+MulAddDst)`），因为 bias 需要每 token 重新初始化
  累加器。

### 2.4 conv_state 写回

* **序列最终状态**：由负责该序列末尾的 task 写（`WriteBackFnState`，
  `causal_conv1d.h:912`）。`seqLen ≥ W-1` 时一次 `DataCopyParams`
  搬最后 W-1 行；否则把 `conv_state` 里的旧历史与 x 拼接后写回。
* **前缀缓存（APC）块边界状态**：range k > 0 把自己块起点前的 W-1 行写到
  `cache_indices[seq, first + k - 1]`。
* **投机解码**：`WriteBackStateSpec` 按 `num_accepted_tokens` 滚动写。

### 2.5 arch35：RegBase（MicroAPI）

arch35 上 `RestoreFnLocalPartials` / `ComputeFnRollingOutput` /
`AdvanceFnLocalPartials` 走 `arch35/causal_conv1d_regbase.h`：直接操作
`__ubuf__` 地址与 `RegTensor`，`__VEC_SCOPE__` 内按
`REGBASE_VECTOR_LENGTH = VECTOR_REG_WIDTH/sizeof(float)` 分段
`LoadAlign → Cast → Mul/Add`。这是寄存器级编程，AscendC 可以做"一条指令处理
一个寄存器块"的紧循环；**Triton 没有等价 API**，只能退到 tile 级张量运算。

### 2.6 它为什么贴着带宽上限

AscendC 全程只有 1 次 x 读 + 1 次输出写（移位在寄存器/UB 内完成），
所以时间 ≈ `2 × 数据量 / 实测算力带宽`。实测该平台 2D tile copy
（同形态、8 行 × 1024 通道）能跑 ~2.7 TB/s，因此

```
T=8192, dim=9216: 302 MB / 2.7 TB/s ≈ 112 us   （实测 161 us，含 UB 内移位与 Silu）
```

## 3. FlagGems-vllm 的实现（重写后）

代码：`src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py`（819 行）。
对外契约、参数语义与重写前一致；实现按 AscendC 的 tiling 思路重排。

### 3.1 host planner（`_plan_ranges` / `_pick_tiles`，:548 / :593）

```python
budget = max(1, core_num // nct)                    # 一波的 token 段数
while sum(counts) < budget:                          # 贪心：每次给"剩余份额最大"的请求加一段
    j = max(cand, key=lambda i: (seqlens[i] / counts[i], -i)); counts[j] += 1
chunk = align_up(ceil(seq_len / counts[s]), BLOCK_T) # 段长对齐到 tile 高度
```

* **段绝不跨请求**，因此 kernel 不需要 AscendC 里那个跨序列 `while` 循环，
  也避免了"一个 program 循环多段"曾经触发的 triton-ascend 正确性问题。
* 贪心分配让 8 个请求 × 3072 token 也能摊成 32 段 × 2 个 channel tile = 64，
  而不是 21 段 × 3 = 63 里出现长短段。
* 每 call 只上传 **1 个** int32 张量（`[4, n]`：seq id / 段号 / 段首 token / 段长），
  缓冲区按 device 缓存（`_plan_workspace`），避免每次分配 4 个小张量。

实际 plan（与 AscendC 对照）：

| dim | T | BLOCK_N | nct | BLOCK_T | 段数 | grid |
|---|---|---|---|---|---|---|
| 9216 | 8192 | 1024 | 9 | 8 | 7（1024/1368/1360 token） | 63 |
| 4608 | 16384 | 1024 | 5 | 8 | 12 | 60 |
| 2304 | 24576 | 2048 | 2（2048 + 256） | 4 | 32（每段 768 token） | 64 |
| 9216 | 32768 | 1024 | 9 | 8 | 7 | 63 |

### 3.2 kernel 四段结构

```
_pid → (task, channel tile) → 读 (seq, 段首, 段长, 段号)
  ├─ A. 滚动 prologue（:279）     段首 W-1 个 token
  ├─ B. full-tile 主循环（:410）  BLOCK_T 行一块，affine 地址，无 mask 行判断
  ├─ C. peeled tail tile（:444）  不足一块的尾巴，唯一带行 clamp 的地方
  └─ D. conv_state 写回（:470）   序列末尾状态 + APC 块边界状态
```

*A. 滚动 prologue*：历史来自 `x` / `conv_state` / 0（`_history_row`，:83），
按 AscendC 的初始化公式算出 `s0..s_{W-2}`，再逐 token 递推并落盘。
bias 被折进所有部分和（`s_k += bias`），递推自然保持，所以主循环不需要
额外的 `+bias` 指令。

*B. 主循环*：每个 tap 用**同一个 tile 的行偏移**重新 load：

```python
for j in tl.static_range(KERNEL_WIDTH):
    xj = tl.load(x_ptr + (seq_start + rows - j)[:, None] * stride_x_token
                 + (idx_feats * stride_x_dim)[None, :], mask=m_feat[None, :], other=0.0)
    acc += wrows[KERNEL_WIDTH - 1 - j][None, :] * xj.to(tl.float32)
```

W=4 时等于每行 x 从 GM 读 4 次（相邻 tap 的 tile 高度重叠，后 3 次命中 L2）。
这是 AscendC "寄存器移位" 在 Triton 里的替代方案。

### 3.3 三个关键取舍（都有实测依据）

**(1) 为什么不用 `extract_slice` 做寄存器内移位？**
先读一次 `[BLOCK_T+W-1, BLOCK_N]` 再切 W 个 tap，理论 MTE2 流量只有 1.1×，
实测却更慢且不可靠（隔离 microbenchmark，T=8192 / dim=9216 / W=4）：

| 方案 | 时间 | 正确性 |
|---|---|---|
| 4 次按行偏移重读（`multibuffer=True`） | 162 µs | 正确（max err 1e-4） |
| 1 次读 + `extract_slice`，无 multibuffer | 260 µs（循环被串行化） | 正确 |
| 1 次读 + `extract_slice`，`multibuffer=True` | **156 µs** | **错误（max err 1.65，静默返回旧行）** |

triton-ascend 上 `multibuffer` 与 `extract_slice` 组合会读到被复用的 UB 缓冲区
旧内容。重读方案虽然多 3 倍读请求，但都命中 L2，是"能用且正确"的最快选择。

**(2) 为什么把 ragged 尾巴 peel 出去？**
行号 clamp（`min(max(row-j,0), seq_len-1)`）会让循环归纳变量上的地址变成
非 affine，triton-ascend 会退回逐行 DMA：

| 同一个 kernel（dim=9216, T=8192） | 时间 |
|---|---|
| 全 tile 都 clamp | 675 µs |
| 只对 peeled tail tile clamp | 185 µs |

所以主循环只跑整块（地址是 `t_first + it*BLOCK_T + arange` 的仿射式），
尾巴单独一块带 mask + clamp。**mask 本身免费，clamp 才贵**。

**(3) 为什么 tile 是 1024×8 / 2048×4？**
UB 里 W 个 tap 缓冲 ×`multibuffer`2 份 + fp32 累加器 ≈ `BLOCK_T·BLOCK_N·(4W+4)`
字节，254 KB 预算下 `BLOCK_T·BLOCK_N ≤ ~13000`：

| 行宽 | 2 的幂 tile | 可行 BLOCK_T | 纯 copy 上限（T=24576, dim=2304） |
|---|---|---|---|
| 1024 | 1024 | 8 | 158 µs（1.4 TB/s） |
| 2048 | 2048 | 4 | 107 µs（2.1 TB/s） |
| 4096 | 4096（覆盖整行 2304） | 2 | 101 µs（2.3 TB/s） |

宽度提升带宽，但 BLOCK_T 掉得更快，二者在 2048×4 与 1024×8 之间交叉 ——
实测 dim>2304 选 1024×8，dim≤2304 选 2048×4。

### 3.4 与 AscendC 的结构对照

| 维度 | fla_npu AscendC | FlagGems Triton |
|---|---|---|
| 网格 | host tiling，恒 64 program | host planner，60–64 program |
| token 划分 | 按 cuSeqlen 均分，可跨请求 | 按请求分段，不跨请求 |
| 段内处理 | ring + 逐 token（2-token 批 DMA） | BLOCK_T 行 tile |
| tap 实现 | 寄存器/UB 内移位，1 次读 | 同 tile 行偏移重读，W 次读 |
| 段首历史 | `InitRingSeqSplit` + 滚动初始化 | 滚动 prologue |
| 流水 | 手写事件双缓冲（5 槽 ring） | 编译器 `num_stages=2` + `multibuffer=True` |
| 状态写回 | `WriteBackFnState` / APC / spec | 同语义（已验证 APC 与旧实现逐位一致） |
| arch35 特化 | RegBase(MicroAPI) 寄存器级 | 无（triton-ascend 无此 API） |

### 3.5 正确性边界

* 支持 `width = 2..5`（AscendC 侧 `MAX_WIDTH = 4`）。
* `cache_indices=None` 时按"序列号即 cache 槽、原地更新"处理。
* `metadata` 参数保留但不再复用其预分配缓冲区（改为 per-device 计划缓冲）。
* APC（`block_idx_first/last_scheduled_token`、`initial_state_idx`、
  `num_computed_tokens`、`block_size_to_align`）语义与旧实现一致：段长强制等于
  缓存块大小。

---

## 4. 性能对比与差距归因

### 4.1 实测（设备 kernel 时间，µs）

| shape | 重写前 | 重写后 | AscendC | 前 → 后 |
|---|---|---|---|---|
| T=8192, seqs=2, dim=9216 | 377 | **190** | 161 | 0.44x → **0.85x** |
| T=16384, seqs=4, dim=4608 | 413 | **317** | 238 | 0.65x → **0.75x** |
| T=24576, seqs=8, dim=2304 | 357 | **310** | 114 | 0.33x → **0.37x** |
| T=32768, seqs=2, dim=9216 | 1449 | **716** | 633 | 0.44x → **0.88x** |
| **几何平均** | 0.45x | **0.68x** | 1.00x | |

（`speedup = AscendC 时间 / FlagGems 时间`；数字有 ±5% 抖动。）

### 4.2 带宽模型

设数据量 `B = T·dim·2 B`（读）＋同样一份写，共 `2B`。AscendC 搬 `2B`，
FlagGems 搬 `5B/2 × 2 = 5B`（4 读 + 1 写），即 **2.5 倍 MTE2 流量**。

| shape | 数据量 2B | AscendC 有效带宽 | FlagGems 有效带宽 | FlagGems MTE2 带宽 |
|---|---|---|---|---|
| dim=9216, T=8192 | 302 MB | 1.87 TB/s | 1.59 TB/s | **3.97 TB/s** |
| dim=9216, T=32768 | 1208 MB | 1.91 TB/s | 1.69 TB/s | 3.79 TB/s |
| dim=4608, T=16384 | 302 MB | 1.27 TB/s | 0.95 TB/s | 2.39 TB/s |
| dim=2304, T=24576 | 227 MB | 1.98 TB/s | 0.73 TB/s | **1.83 TB/s** |

对照纯 copy 上限（同形状、2D tile、`multibuffer`）：

| shape | copy 上限 | FlagGems MTE2 / copy 上限 |
|---|---|---|
| dim=9216, T=8192 | 111 µs（2.72 TB/s） | **1.46×**（L2 命中生效） |
| dim=9216, T=32768 | 437 µs（2.76 TB/s） | 1.37× |
| dim=4608, T=16384 | 181 µs（1.67 TB/s） | 1.43× |
| dim=2304, T=24576 | 107 µs（2.12 TB/s） | **0.86×**（几乎无 L2 收益） |

**读法**：FlagGems 的 MTE2 带宽能超过 copy 上限，说明 4 次 tap 读里的后 3 次
确实被 L2 服务了（dim≥4608 的三个 shape 都到 1.37–1.46 倍）。而 dim=2304 只到
0.86 倍，等价于"4 次读全部打到 HBM"，这正是它只有 0.37x 的直接原因
（具体为何该 shape 吃不到 L2，还没定位；它与 BLOCK_T=4、行宽 4 KB、以及
2048+256 的不均匀切分同时出现）。

### 4.3 逐 shape 归因

* **T=8192 / T=32768（dim=9216，0.85x / 0.88x）**：结构已经对齐，剩余差距
  就是 2.5 倍 MTE2 流量被 L2 部分吸收后的净开销（约 1.4× copy 时间）。
  这是"没有寄存器移位"的直接代价。
* **T=16384（dim=4608，0.75x）**：额外因素是 `nct=5` 的通道切分不均匀
  （1024×4 + 512），且该 shape 的 copy 上限本身只有 1.67 TB/s（行宽 2 KB、
  5 个 tile 分摊同一行）。tile 形状已扫过 2048×4、512×16 等，均更差。
* **T=24576（dim=2304，0.37x）**：要贴 AscendC 需要"整行一次搬"（2048/4096
  宽），但 UB 只允许 `BLOCK_T=2..4`，矮 tile 的 DMA 效率抵消了宽度收益；
  叠加 4 次重读没有 L2 收益。

### 4.4 若要继续追平，剩余可试方向（按风险排序）

1. **让 tap 重读真正命中 L2**：dim=2304 是唯一"重读全打 HBM"的 shape。
   可尝试把 4 个 tap 的 load 顺序/地址局部性显式化（例如同一 iteration 内
   先发 tap0，再 tap1..3），或缩小 `BLOCK_N`+ 提高并发 program 数。
   低风险，收益上限 ~2×（仅该 shape）。
2. **压缩 fp32 累加器以放宽 UB**：例如按 `BLOCK_T` 分块累加（每块只保留
   一行的 fp32 中间值），把 `BLOCK_T·BLOCK_N` 预算从 13000 提到 20000+，
   从而允许 2048×8 或 4096×4。中风险，可能同时改善 dim=2304/4608。
3. **寄存器级移位**：Triton 无对应 API；可行路径是 FlagTree 的 TLE 扩展、
   triton-ascend 的 `insert_slice` + 单缓冲手写流水（实测当前 229 µs，仍慢），
   或写自定义 AscendC/`register_custom_op` 与 Triton 混合。高成本，理论收益
   可到 2.5× 流量 → 接近 AscendC。
4. **直接复用 fla_npu 的 AscendC 算子**：性能最好，但违背 FlagGems
   "纯 Triton 实现" 的定位，且引入跨仓库依赖。

另外，AIC/Cube 在这里帮不上：这是**逐通道** depthwise 卷积，权重随 channel
变化，无法写成 `tl.dot`（矩阵乘法要求权重与 reduction 轴无关），所以
arch35 的 AIV/AIC 直通对本算子没有落点。

---

## 5. 验证与复现

### 5.1 正确性

* `tests/test_causal_conv1d_fn.py`：**48/48 通过**（batch 4/10、padding、
  dim 64/4096、seqlen 8/249/4096、bf16/fp16、随机 `has_initial_state`）。
* 与 fla_npu AscendC 对拍（`tools/test_kda_vs_vllm_ascend.py --ops conv_prefill`）：
  4/4 通过，输出 `max_rel ≤ 6.2e-4`，`conv_state` 逐位一致。
* 前缀缓存（APC）路径：与重写前实现输出和 `conv_state` **完全相同**（maxdiff = 0）。
* FP32 精度：对 torch 参考最大绝对误差 ~1e-6（width 2/3/4，含短序列与多段）。
* 重写过程中顺带修掉两个旧实现的隐性 bug：
  1. tile 路径漏加 bias（有 bias 时全错，benchmark 恰好是无 bias 才没暴露）；
  2. 短序列的 `conv_state` 历史行号应为 `rel + state_len` 而非位置下标 `pos`。

### 5.2 复现命令

```bash
# 功能 + 性能对比（4 个 prefill shape）
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops conv_prefill

# 只跑功能
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops conv_prefill --skip-perf

# 单测
PYTHONPATH=src python -m pytest tests/test_causal_conv1d_fn.py -q
```

### 5.3 本文数字的来源

* 上表 "重写后 / AscendC" 来自同一进程内同一轮的 profiler 采集，避免跨进程漂移；
  多次运行的偏差约 ±5%。
* copy 上限、per-token/tile 高度对比来自独立 microbenchmark（同一 kernel 形态，
  只保留 load/store），用于给出该 shape 的带宽上界，不是本算子的实测。

---

## 6. 文件索引

| 内容 | 路径 |
|---|---|
| AscendC tiling 规划 | `flash-linear-attention-npu/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/op_host/causal_conv1d_tiling_planner.h` |
| AscendC kernel 主体 / 滚动递推 | `.../op_kernel/causal_conv1d.h` |
| AscendC 任务划分（跨序列） | `.../op_kernel/causal_conv1d_fn_tasks.h` |
| arch35 RegBase 路径 | `.../op_kernel/arch35/causal_conv1d_regbase.h` |
| FlagGems 实现 | `src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py` |
| 对比脚本 | `tools/test_kda_vs_vllm_ascend.py` |
| 单测 | `tests/test_causal_conv1d_fn.py` |

---

## 7. 后续优化（2026-11-09）：宽通道 tile，追平并超过 AscendC

第 4 节把剩余差距归因于「每行读 4 次」的 MTE2 流量。按这个方向做的
「一次读 + UB 内 `extract_slice` 行移位」确实能到 142 / 142 / 109 / 534 µs
（比第 3 节的实现快约 9%），但在 `num_stages=2` 下**结果错误**：triton-ascend
的软件流水会把 `extract_slice` 的视图固定到流水某一级的缓冲区，读到上一轮
的旧行（第 3.3 节记录的同一个问题的更精确版本）。定位与取舍：

| 方案（W=4, T=8192, dim=9216） | 时间 | 正确性 |
|---|---|---|
| 1 次 halo 读 + `extract_slice`，`num_stages=2` | 142 µs | 错误（约 3–25% 元素是旧行） |
| 同上，`num_stages=1` | 208 µs | 正确 |
| 同上，`multibuffer=False` + 手动预取下一 tile | 210 µs | 正确 |
| `bl.alloc` 显式 UB buffer + `to_buffer` + `subview` | — | 编译失败：`memref.cast ... strided<[1], offset: N>` 无法转换（subview 结果参与算术时） |
| 同上，但拷进显式 buffer 后再 `extract_slice` | 142 µs 级 | 仍错误（流水把切片结果提前了一级） |

`num_stages=1` 证明数据本身没问题，是流水器的缓冲区版本化 bug；没有找到
既正确又保留流水的写法，于是保留「按行偏移重读、靠 L2 吸收」的形态，
转而把**通道 tile** 做成主优化点。

### 7.1 改动

1. **通道 tile 变宽且整除 `dim`**（`_pick_tiles`）：旧启发式在 dim=9216/4608/2304
   上分别落到 `1024+…`（9 块）、`1024×4+512`（5 块）、`2048+256`（2 块），
   窄行 + 不规则尾块是所有 shape 的共同瓶颈。新规则按 256 对齐、优先取
   `≤4608` 且整除 `dim` 的宽度，token tile 吃掉剩余 UB 预算：

   | dim | 旧 (BLOCK_N × BLOCK_T) | 新 |
   |---|---|---|
   | 9216 | 1024 × 8（9 块） | **4608 × 2**（2 块） |
   | 4608 | 1024 × 8（5 块，含 512 尾块） | **4608 × 2**（1 块） |
   | 2304 | 2048 × 4（2 块，含 256 尾块） | **2304 × 4**（1 块） |

   行传输从 2 KB 提到 9 KB，且不再有 masked 尾块——**几块通道 tile 与是否
   整除 `dim` 比 tile 形状本身更重要**（同为 9216 元素面积，1024×8 /
   1152×8 / 576×16 的实测差 ≤4%，而 2048+256 的尾块直接吃掉 2×）。
2. **UB 预算按 `W` 参数化**：每元素 `4W+4` 字节（W 个 tap buffer ×2 多缓冲 +
   fp32 累加器），`BLOCK_T·BLOCK_N ≈ 184 KB / (4W+4)`，W=4 时 9216 元素。
   这样 W=2..5 都不会因为 tile 过大而编译失败或溢出。
3. **去掉每次调用的 device 端 `diff` kernel**：`query_start_loc.cpu().diff()`
   用一次 D2H 拷贝代替「device diff + D2H」，每 call 少一个 kernel
   （profiler 里 kernels/iter 从 3 降到 2）。

### 7.2 结果（同一脚本、同一轮 profiler，设备 kernel 时间，µs）

| shape | 第 3 节实现 | **本次** | AscendC | 本次 speedup |
|---|---|---|---|---|
| T=8192, seqs=2, dim=9216 | 191 | **155.5** | 163.5 | **1.05x** |
| T=16384, seqs=4, dim=4608 | 318 | **155.0** | 238.1 | **1.54x** |
| T=24576, seqs=8, dim=2304 | 310 | **117.8** | 114.7 | **0.97x** |
| T=32768, seqs=2, dim=9216 | 717 | **583.6** | 635.7 | **1.09x** |
| **几何平均** | 0.68x | | | **1.14x**（多次运行 1.14–1.15x） |

* 4/4 功能对拍通过（`max_rel ≤ 5e-4`，`conv_state` 逐位一致）。
* APC 路径与改动前实现逐位一致（4 组 `block_idx_*` / `num_computed_tokens`
  场景，输出与 cache 状态 `torch.equal`）。
* `tests/test_causal_conv1d_fn.py` 48/48 通过（含 dim=4096、fp16、
  padding、seqlen 8/249/4096）。
* 剩余唯一没赢的 dim=2304（0.97x）已经是「4 读 + 1 写」在 MTE2 上的极限
  （283 MB / 117.8 µs = 2.4 TB/s，高于该 shape 的纯 copy 上限 2.12 TB/s），
  要再快只能减少读次数，而这条路径被上面的流水器 bug 挡住。

### 7.3 复现

```bash
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops conv_prefill
PYTHONPATH=src python -m pytest tests/test_causal_conv1d_fn.py -q
```

---

## 8. 对齐真实场景：benchmark 布局 + 把转置融进 kernel 加载（2026-11-09 续）

第 7 节的 1.14x 有一个前提：benchmark 把 `x` 以 **token-major 内存上的
channel-first 视图**（`x.transpose(0,1)`）交给 FlagGems，于是算子内部的
`x.transpose(0,1).contiguous()` 退化成 no-op，省掉了一次全量转置拷贝。
但算子对外的契约（`ops/causal_conv1d.py` 与上游 vLLM 一致）是
**x: (dim, total_tokens)**，真实调用方交来的是 channel-first 连续张量，
此时那一次拷贝实际会发生，而 fla_npu 的 AscendC 算子拿的是 token-major
原生布局——两边的工作量根本不同，benchmark 不公平。

### 8.1 benchmark 改为「同数据、各自原生布局」

`tools/test_kda_vs_vllm_ascend.py` 的 conv 用例现在构造一份数据：

```python
x = torch.randn(tokens, dim)          # AscendC 原生 [T, dim]
x_gems = x.transpose(0, 1).contiguous()  # FlagGems 原生 [dim, T]，同一批数值
```

两侧拿到数值完全相同、但各自 native layout 的连续张量（权重与 conv_state
仍是同一块内存的转置视图，零拷贝）。`benchmark/test_causal_conv1d_fn.py`
也改成直接构造 channel-first 连续 `x`。

### 8.2 kernel 融合转置：按 stride 选两种 tile 几何

算子现在按输入 stride 选择几何，**两种布局都不再拷贝**：

| `x.stride()` | 几何 | tile | 说明 |
|---|---|---|---|
| 通道轴 stride = 1（token-major 内存，vLLM 投影输出） | A | `[tokens, channels]` | 原路径，行宽 ~9KB，最快 |
| token 轴 stride = 1（`[dim, tokens]` 契约，连续） | **B（新）** | `[channels, tokens]` | tile load 本身就是转置，省掉整次拷贝 |
| 都不是 1 | — | — | 先 `contiguous()` 一次，走 B |

B 几何的第 4 个 tap 仍按 token 偏移重读（同一行内相邻 tap 只差 2B，几乎全部
命中 L2），但实现上踩到并解决了三个坑：

1. **tile 选择**（`_pick_tiles_channel_first`）：token 轴连续，一行只有
   `BLOCK_T*2` 字节，`BLOCK_N×BLOCK_T ≤ ~8192`；实测 32×256 最优（宽通道省
   program 数但缩短行，窄通道让每个 program 的 prologue 摊到太少通道）。
2. **通道 tile 不能太大**：`BLOCK_N ≥ 1024` 时，tile 外围的 per-channel 向量
   （权重、history、mask）按 vector fragment 形状补齐，单独就把 UB 撑爆
   （2048×2 的 tile 需要 462KB），因此 B 几何把 `BLOCK_N` 限制在 256。
3. **tail 的 clamp 是最大杀手**：行地址逐 lane clamp 会变成非 affine，triton-ascend
   退化成每元素一次 DMA。B 几何的 tail 改为**标量夹紧的短 tile 循环**：
   `start = min(tail_first + i*BLOCK_T_TAIL, tk_end - BLOCK_T_TAIL)`，行地址
   保持 affine、全部落在 `[0, tk_end)` 内、无需行 mask（末尾 tile 会重复计算
   几行，卷积是纯函数所以结果相同）；只有「整段短于一个 tail tile」的退化情形
   才回落到逐 lane clamp + mask。仅此一项就让 B 从 376 µs 降到 253 µs。

### 8.3 结果（同一脚本、同一轮 profiler，设备 kernel 时间，µs）

| shape | 不融合（转置拷贝 + A） | **融合 B** | AscendC | 融合后 speedup |
|---|---|---|---|---|
| T=8192, seqs=2, dim=9216 | 345.6 | **246.2** | 163.1 | 0.66x |
| T=16384, seqs=4, dim=4608 | 378.1 | **245.8** | 238.0 | 0.97x |
| T=24576, seqs=8, dim=2304 | 278.7 | **217.2** | 114.8 | 0.53x |
| T=32768, seqs=2, dim=9216 | 1478.2 | **820.4** | 636.2 | 0.78x |
| **几何平均** | 0.48x | **0.72x** | 1.00x | |

* 融合转置相对「拷贝 + A」快 **1.26–1.79x**（几何平均 1.45x）。
* 4/4 功能对拍通过；`tests/test_causal_conv1d_fn.py` 96/96（A/B 两种布局各
  48）；APC 路径与改动前实现输出和 cache 状态逐位一致。
* 若调用方能像 vLLM 投影层一样直接给出 token-major 张量（以 channel-first
  **视图**传入，零拷贝），算子仍走 A 几何，即第 7 节的 155/155/118/582 µs
  （对 AscendC 1.14x）。**这是推荐的接入方式**；交付 channel-first 连续张量
  时才走融合 B。

### 8.4 为什么 channel-first 布局追不上 token-major

B 几何每行只有 `BLOCK_T*2 = 512B` 连续内存，token-major 是 9KB。同样的
「4 读 + 1 写」下，A 的 MTE2 有效带宽约 4.9 TB/s（大 burst、L2 吸收重复读），
B 只有约 3.0 TB/s（512B burst 的 descriptor 开销先到顶）。B 的 246 µs 对应
302 MB 有效数据 = 1.2 TB/s，远低于纯 copy 上限 2.7 TB/s——瓶颈是 MTE2 的
**请求数**而不是 HBM 带宽。尝试过但无效/更差的方案：

* 末轴 `extract_slice`（单次读 `[BN, BT+3]` 再按 token 偏移切片）：
  1083 µs，2 字节偏移的切片被降级成逐元素移位，比 4 次重读慢 4x。
* 把 `BLOCK_N` 提到 512/1024 以减少 program 数：UB 直接溢出。
* `loop_unroll_factor`：A 从 155 µs 掉到 193 µs，其它 shape 编译失败。

要真正追平，只能减少读次数（第 7 节记录的 `num_stages=2` 流水器对切片视图的
版本化 bug 仍然存在），或者改变契约让调用方直接给 token-major 视图。

### 8.5 复现

```bash
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops conv_prefill
PYTHONPATH=src python -m pytest tests/test_causal_conv1d_fn.py -q
PYTHONPATH=src python -m pytest benchmark/test_causal_conv1d_fn.py --level core --iter 1 --warmup 1
```

---

## 9. 继续优化：UB 地址空间用法、流水器缺陷复核、tail 加高（2026-11-10）

### 9.1 先复核「是不是使用手法问题」

对照 `tools/triton_ascend_probes` 的结论重新试了「一次读 + UB 内移位」，发现之前
有一个**真实的使用错误**：`bl.alloc` 不传 `al.ascend_address_space.UB` 时，
`al.copy` / `bl.subview` 会报 `src's AddressSpace must be UB`，而去掉地址空间后
生成的是一块「通用空间」缓冲，流水器会把它当普通 SSA 值处理。补上
`al.ascend_address_space.UB` 之后：

* `al.copy(bl.subview(halo, [j,0], [BT,BN], [1,1]), tap)` + `bl.to_tensor(tap)`
  在小规模（BN=64, 4 次迭代）下**逐位正确**，ns=2 + multibuffer 也对；
* 但把同样的写法放进真实卷积（BN=2304、上万次迭代）后：`extract_slice` 视图
  版本会**确定性**地读到旧版本（maxerr≈16，两次运行结果相同；早期探针用 `T`
  当行 stride、只在 `T==C` 时自洽，所以曾误判成竞态），`bind_buffer` 与
  `al.copy` 两种写法「谁正确」随编译变化。

结论：**不是纯使用问题**。`halo` 缓冲在循环外分配、视图在循环内被当循环不变量
处理，加上 ns=2 的流水级重叠，导致读到的行可能被下一轮的 DMA/拷贝覆盖；
`num_stages=1`、`multibuffer=False`、bind 到显式缓冲、拷进稳定缓冲、
`debug_barrier`、动态 tap 循环都试过，没有一种能同时做到「正确 + 保持流水」。

### 9.2 试过并否决的方案（同一轮 profiler，T=8192/dim=9216）

| 方案 | 时间 | 结论 |
|---|---|---|
| 现状（4 次按行偏移重读） | 247 µs | 基准 |
| 动态 tap 循环（共享一份 tap buffer，BLOCK_T 可到 512/1024） | 295–330 µs | 内层变动态循环后失流水，反而更慢 |
| 动态 tap + 寄存器内 `tl.where` 选权重 | 295 µs | 同上 |
| `mb=False` + `num_stages=3` + 更高 tile | 268 µs | 更慢 |
| 满通道 tile 去掉 channel mask | 247 µs | 无变化（mask 免费） |
| `num_stages=3/4` | 251/250 µs | 无变化 |
| 单读 + `extract_slice` / `al.copy` 移位 | 186–202 µs（1.3–1.4x） | 视图读到旧版本（maxerr≈16）；bind/copy 谁正确随编译变化，搬进真实 kernel 后 96 例中 20–32 例算错 |
| prologue 用标量 `if rel >= 0` 分支掉无效的 x load | 未测完 | 单核编译时间从 ~2s 涨到 ~30s（96 例矩阵 10 分钟没编完），放弃 |
| conv_state 回写改成 2D tile（`[W-1, BLOCK_N]` + `tl.trans`） | 263 / 263 / 230 / 837 µs | 比现状（233 / 231 / 204 / 805）**慢 13%**，回退：该 tile 的每行只有 W-1 个 token（6–8 B），DMA 形态比逐行 1D 读更差，`trans` 与双 mask 又添了向量开销 |

### 9.3 落地的优化：CF 的 tail tile 加高

channel-first 路径的 peeled tail 原先固定 32 行（`BLOCK_T_TAIL=32`）。KDA 形状
上 `BLOCK_T=256`，一个 range 的尾巴约 253 行 → 8 个 64B 行的碎 tile，正好踩在
最差的 DMA 形态上。把 CF 的 tail 高度提到 128（仅当 `BLOCK_T ≥ 256`；小 tile
配置仍用 32，否则 UB 溢出）：

| shape | 改前 | 改后 | AscendC | speedup |
|---|---|---|---|---|
| T=8192, dim=9216 | 246.2 | **232.5** | 163.1 | 0.66 → **0.70x** |
| T=16384, dim=4608 | 245.8 | **231.4** | 237.2 | 0.97 → **1.03x** |
| T=24576, dim=2304 | 217.2 | **203.4** | 114.9 | 0.53 → **0.57x** |
| T=32768, dim=9216 | 820.4 | **807.9** | 635.3 | 0.78 → **0.79x** |
| **几何平均** | 0.72x | | | **0.75x** |

* `tests/test_causal_conv1d_fn.py` 96/96；APC 路径与改动前逐位一致。
* token-major 路径不受影响（155/155/118/582 µs，对 AscendC 1.05–1.54x）。
* 仍被否决：把 tail 直接设成 `BLOCK_T`（UB 溢出）、192 行（比 128 略慢）。

### 9.4 结论与下一步

要拿到 2.5× 的 MTE2 流量收益，必须先解决 triton-ascend「多缓冲流水 + 视图」
的版本跟踪问题：同一写法在不同编译下时对时错，出错时是确定性错值。该问题已
整理为编译器需求 **R3**（`tools/triton_ascend_probes/gaps/g3_pipelined_ub_view_shift.py`）
与缺陷复现 D4（`defects/d4_extract_slice_pipeline.py`），并写进该目录 README。在那之前，本算子的最优
表达仍是 4 次 L2 重读，剩余可挖的空间只有 tail/prologue 这类固定开销（合计
<10%）和集成侧的 token-major 视图传参。

---

## 10. 2026-11-11：压掉固定开销 + 分档 tile（0.75x → 0.80x）

第 9 节的结论是「剩余空间只有 tail/prologue 这类固定开销（合计 <10%）」。
这一轮先把 kernel 拆开量了固定开销，再分别处理，最后重新扫了 channel-first
的 tile。

### 10.1 逐段耗时（device kernel time，µs）

在 kernel 里加 `EXP` 位开关（跳过 full-tile 主循环 / tail / 前导 / state 回写），
用 `torch_npu.profiler` 只统计主 kernel：

| shape | kernel 总 | 4-tap 主循环 | 前导 W-1 输出 | tail | 其它 |
|---|---|---|---|---|---|
| T=8192, dim=9216 | 233.2 | 200.0 | 10.3 | 17.7 | 8.5 |
| T=24576, dim=2304 | 200.6 | 153.8 | 27.7 | 20.6 | 13.0 |
| T=32768, dim=9216 | 799.7 | 777.7 | 6.8 | 13.2 | 7.8 |

两处固定开销在短 shape 上合计 23–48 µs（10–14%）。旧的前导是 AscendC 那种
`s_k <- s_{k+1} + w_k x_t` 滚动递推，每个 program 串行走 3 个
load/compute/store 小步；在 576 个 program、9 波的情形下，这条依赖链本身
就要 10–28 µs。

### 10.2 改动

1. **前导输出改成并行组合。** 第 `i` 个前导输出
   `y_i = Σ_k w_k · x[tk_start + i - (W-1) + k]` 只依赖 `W-1` 个历史行和
   区间内前 `W-1` 行，输出之间没有递推关系，直接按 `i` 展开成 `W-1` 个固定
   组合即可。滚动状态 `s0..s3` 原本只服务于这段前导，主循环完全不使用，
   一并删除。
2. **tail 并入最后一个 full tile。** 主循环改成
   `n_full = ceil((tk_end - t_first) / BLOCK_T)`，最后一次迭代的起点
   标量回夹到 `tk_end - BLOCK_T`；重叠的几行重算同值（卷积是 `x` 的纯函数）。
   起点是标量，所有 tap 地址仍是 affine；只有「区间短于一个完整 tile」时
   才回落到原来的逐 lane mask 短 tile。这样 tail 从「2×128 行」变成
   「1×BLOCK_T 行」。
3. **channel-first tile 分档。** 重新用 profiler 扫（不是 wall-clock）后，
   长区间用 `16×512`（1 KB 行）、短区间用 `32×256`：

   | max_seqlen | 32×256 | 16×512 |
   |---|---|---|
   | 4096 | **112.0** | 116.6 |
   | 8192 | 221.4 | **199.0** |
   | 16384 | 424.1 | **376.1** |

   （dim=9216、单请求；dim=2304/4608 同趋势。）分界取 `max_seqlen >= 8192`。
   更长的行在长区间上把 4 次 tap 读的 DMA 描述符开销摊薄；短区间上
   channel tile 减半会让 program 数翻倍，前导/收尾摊不回来。
4. **顺带修掉 width=5。** 旧 `wrows` 只有 4 行，`wrows[KERNEL_WIDTH - 1]`
   在 W=5 时越界，W=5 实际不可用（单测只覆盖 W=4）。现在补上第 5 行权重，
   并让 W=5 固定用 16 宽通道 tile（32 宽 + 5 个 tap buffer 会 UB 溢出：
   32×128 需要 2.28 Mbit，超出 1.77 Mbit；16×256 可编译）。

### 10.3 结果（同一脚本、同一轮 profiler，µs）

| shape | 第 9 节 | **本次** | AscendC | speedup |
|---|---|---|---|---|
| T=8192, seqs=2, dim=9216 | 232.5 | **221.4** | 162.9 | 0.74x |
| T=16384, seqs=4, dim=4608 | 231.4 | **217.7** | 238.9 | 1.10x |
| T=24576, seqs=8, dim=2304 | 203.4 | **189.3** | 115.3 | 0.61x |
| T=32768, seqs=2, dim=9216 | 807.9 | **749.4** | 637.0 | 0.85x |
| **几何平均** | 0.75x | | | **0.80x** |

* `tests/test_causal_conv1d_fn.py` 96/96 通过。
* `tools/test_kda_vs_vllm_ascend.py --ops conv_prefill` 4/4 PASS，
  输出 `max_rel ≤ 6.2e-4`，`conv_state` 逐位一致。
* width 2/3/4/5 均与 torch 参考对齐（新增 W=5 覆盖）。

### 10.4 为什么还是 0.73x / 0.61x

固定开销已经压到 3–10%，剩余差距回到 4 次 tap 读。同一 tile 形态的
microbench（只保留 full-tile 循环）给出：

| shape | 1 次 tap + 写 | 4 次 tap + 写 | 增量 |
|---|---|---|---|
| T=8192, dim=9216 | 145.9 | 186.3 | +40.4 |
| T=24576, dim=2304 | 107.0 | 134.5 | +27.5 |

即后 3 次重读确实命中 L2（1→4 tap 只多 28%，不是 3×），但仍要付约
30–40 µs 的 L2 带宽。Silu 在 microbench 里完全免费（186.3 → 186.0）。
想再进一步只能把「每次 x 行读 4 次」变成 1 次：

* `al.extract_slice` 行移位：`num_stages=2` 下确定性读旧缓冲（D4）；
  `num_stages=1` 正确但串行，且 channel-first 末轴 2 B 偏移会退化成逐元素
  移位（第 8 节实测 1083 µs）。
* `tl.gather`：语义正确，但同 tile 实测 893–3648 µs，是 4 次重读的 4.5–5 倍。
* extension 的 depthwise `conv1d`（`triton.language.extra.cann.extension`）：
  非 depthwise 可编译；depthwise（groups=C）在 A5 regbase pipeline 下
  UB 溢出，`C×L≈4096` 以上就编不过，最大只能到 16×256 量级的 tile，
  对 KDA 的 dim 太小。

因此这一版仍是「4 次 L2 重读」的正确最优形态；真正的 2.5× 流量收益要等
R3（流水化 UB 行偏移视图）落地。

### 10.5 复现

```bash
PYTHONPATH=src:FlagGems/src python tools/test_kda_vs_vllm_ascend.py --ops conv_prefill
PYTHONPATH=src python -m pytest tests/test_causal_conv1d_fn.py -q
```
