# triton-ascend 编译器能力需求（KDA chunk 前向 / Ascend950DT arch35）

本文档逐个列出 FlagGems-vllm 的 KDA chunk 前向（`chunk_kda`）在 **stock
triton-ascend 3.6.0.post2（不使用 FlagTree/TLE）** 上落地 AscendC 级优化时，
遇到的编译器能力缺口。每个特性都附一个可直接运行的最小验证程序，报错文本
均为本机实测（Ascend950DT_9582，CANN 8.5，`torch_npu` 2.9）。

复现环境：

```shell
cd <flag-gems-vllm>
PYTHONPATH=src:FlagGems/src python tools/triton_ascend_probes/probe_XX_*.py ...
```

## 0. 背景：为什么需要这些能力

KDA chunked prefill 在 AscendC（`aclnnChunkKdaFwdV2`）里是 `Prepare → FwdH →
Finalize` 三段，Prepare 内部再用手写 AIC+2×AIV 流水、4 head/轮乒乓、L1 常驻
Q+/K+/K−。我们的 Triton 版本已经对齐了三段式并做了源级 AIV/AIC 重叠，当前
T=32768/H=24/seqs=2 的分核耗时（us）：

| kernel | FlagGems(Triton) | AscendC | 说明 |
|---|---:|---:|---|
| Prepare（我们拆成 prep + Gram） | 1353 + 2649 | 1654 + 413(l2norm) | 主要差距 |
| FwdH | 1266 | 840 | AIC MTE2 占 73% |
| Finalize | 421 | 718 | 已优于 AscendC |

性能剖析（Level1，拆分前的融合版 Prepare）显示 `aiv_vec_ratio=75.4%`、
`aic_mac_ratio=6.2%`：AIV 向量链是关键路径，AIC 大部分时间在等。AscendC 靠 L1 常驻操作数 + 32 行
子块因子 + 混核流水把 Prepare 压到我们的一半。下面 5 个特性就是这条路上
Triton 缺的东西。

## 1. 结论速览

| 编号 | 编译器特性 | 现状 | 被阻塞的优化 | 探针 |
|---|---|---|---|---|
| **F1** | 显式 L1 驻留的 dot 操作数 | **API 存在，链路中断（关键在 `bl.to_tensor`）** | AscendC 式 L1 常驻面板 | `probe_01_l1_resident_dot.py` |
| **F2** | `insert_slice` 拼装 dot 操作数 | **BiShengIR SIGABRT（Unknown PIPE!）** | 32 行子块门控（向量工作量减半） | `probe_03_insert_slice_dot.py` |
| **F3** | `scope` 混核 + 自动跨核同步 | **默认 SIGSEGV；关自动同步后结果错** | AscendC C2/C4/C5/C7 流水 | `probe_04_scope_sync.py` |
| **F4** | 跨 chunk 软件流水的 UB/L1 预算 | **UB 5.57 Mbit / L1 6.29 Mbit 溢出** | 隐藏 cube 等待（约 0.5–1 ms） | `probe_06_cross_chunk_l1_budget.py` + 真 kernel |
| **F5** | 长 `exp2/select` 链的向量融合 | **短链可融，N≥8 出现断崖** | gate cumsum/因子链 | `probe_05_vec_chain_fusion.py` |

已确认**可用**、无需改动的能力（供编译器团队排除）：

| 能力 | 现状 | 探针 |
|---|---|---|
| 列偏移 `extract_slice` 作 dot 操作数 | PASS（maxerr 7.6e-6） | `probe_02` col |
| 行偏移 `extract_slice` 作 dot 操作数 | PASS（maxerr 5.3e-5） | `probe_02` row |
| `al.sync_block_set/wait`（不加 scope） | PASS（maxerr 1.5e-5） | `probe_04` noscope_sync |
| `al.copy(UB buffer → L1 buffer)` 前端类型检查 | 通过（见 F1，后端仍断） | `probe_01` |

## 2. F1：显式 L1 驻留的 dot 操作数

### 需求

AscendC Prepare 把 Q+/K+/K− 放 L1，多个 dot 复用，不反复走 UB→L1 自动 staging。
我们想写：

```python
ub = bl.alloc(tl.bfloat16, [BT, K], al.ascend_address_space.UB)
l1 = bl.alloc(tl.bfloat16, [BT, K], al.ascend_address_space.L1)
al.copy(bl.to_buffer(a, bind_buffer=ub), ub)   # SSA -> UB
al.copy(ub, l1)                                 # UB -> L1
at = bl.to_tensor(l1)                           # L1 视图
c = tl.dot(at, tl.trans(at))                    # cube 直接读 L1
```

### 现状：**不支持**（API 齐全，链路多处断点）

| 步骤 | 结果 |
|---|---|
| `al.copy(UB buffer → L1 buffer)` 的前端类型检查 | 通过（`semantic.copy` 要求 src=UB、dst∈{UB,L1}） |
| 用 `bind_buffer=ub` 从 SSA tensor 填 UB，再 copy 到 L1 | `'memref.alloc' op error: read before first write` |
| 用位置参数 `bl.to_buffer(a, space=UB)` 填 UB | `'hivm.hir.pointer_cast' op addrs of PointerCastOp should not be empty!` |
| `tl.store` 写 `bl.to_tensor(ub, writable=True)` 的视图 | `Unsupported ptr type <['64','128'], fp32> in tl.store` |
| **`bl.to_tensor(l1)`**（关键最后一步） | `different memory spaces specified for base memref type 'memref<64x128xbf16, #hivm.address_space<cbuf>>' and subview memref type 'memref<1x128xbf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>'` |

最后一处说明：`bl.to_tensor` 在 lowering 时给 L1 buffer 生成了带 **UB**
地址空间的 `memref.subview`，与基址的 `cbuf` 不一致——这是整条链路上
**最致命**的一处（即使前面都修好，没有 L1 视图也无法把数据交给 dot）。
另外，用 `bind_buffer` 填 UB 的写法虽然能过前端，lowering 仍会报
read-before-write（见第一行），所以完整可用路径目前为 0。

### 最小验证程序

`probe_01_l1_resident_dot.py`：

```shell
python probe_01_l1_resident_dot.py copy   # 填 UB + UB->L1：read before first write
python probe_01_l1_resident_dot.py dot    # 加 to_tensor(L1)+dot：cbuf/ub 地址空间不一致
```

### 期望语义

- `bl.to_tensor(l1_buffer)` 产出 **cbuf 地址空间**的 tensor，可直接作
  `tl.dot` 的 A/B 操作数（无需 UB 中转）；
- 或提供 `al.dot(..., a_in_l1=...)` 之类的显式接口；
- `bl.to_tensor(ub, writable=True)` 应可作为 `tl.store`/`insert_slice` 的目标，
  且 `memref.alloc` 的 read-before-write 分析要把 `al.copy` 写入视为写。

### 影响

没有它就只能依赖编译器的自动 L1 staging（每个 dot 前重新搬一次），
这是 Gram 核 6 个 dot 每个都要一次 UB→L1 的直接原因，也是 F4 放不下的
原因之一。

## 3. F2：`insert_slice` 拼装 dot 操作数

### 需求

AscendC 的门控因子按 32 行子块（ScoreRefToken span）计算再拼成 `[64,K]`。
我们想在 `[32,K]` 上算 exp2/where（向量工作量减半），再用
`al.insert_slice` 拼成整块送进一个全宽 dot：

```python
full = al.insert_slice(tl.zeros((64, K), tl.bfloat16), top, [0, 0], [32, K], [1, 1])
full = al.insert_slice(full, bot, [32, 0], [32, K], [1, 1])
c = tl.dot(full, tl.trans(full))
```

### 现状：**BiShengIR 直接崩溃**

```text
[ConvertLinalgIRToBinary] encounters error:
LLVM ERROR: Unknown PIPE!
... died with <Signals.SIGABRT: 6>.
```

（`insert_slice` 只用于 store 时是正常的；只有把结果接进 dot 才崩。）

替代写法 `tl.join + tl.permute + tl.reshape` 能编译、数值正确，但会插入
layout 转换，实测比整块计算**慢 4 倍**（Gram 核 2649 → 10000+ us），
不可用。

### 最小验证程序

`probe_03_insert_slice_dot.py`（单块，直接 SIGABRT）。

### 期望语义

`insert_slice` 结果与源 tile 共享 layout（不产生 copy），且能作为 dot 操作数；
若必须转换，请在编译期报“unsupported layout”，而不是 abort。

## 4. F3：`scope` 混核 + 自动跨核同步

### 需求

用 `al.scope("cube")` / `al.scope("vector")` 表达同一个 kernel 里的 cube/vector
分工，由编译器插入跨核 flag（对应 AscendC Prepare 的 C2/C4/C5/C7）：

```python
with al.scope("cube"):
    cb = tl.dot(a, tl.trans(b)); tl.store(gm, cb)
with al.scope("vector"):
    tl.store(out, tl.load(gm) * 2.0)
```

### 现状：**默认不可用；关自动同步后结果错**

| 模式 | 编译选项 | 结果 |
|---|---|---|
| `scope` | 默认 | **SIGSEGV (11)** in bishengir |
| `scope_noauto` | `enable_mixed_cv=True, disable_auto_inject_block_sync=True` | 编译通过，`maxerr≈1e2`（参考值量级 ~2e1；每次运行数值不同），读到了未写入/陈旧数据 |
| `scope_sync` | 同上 + 手工 `sync_block_set/wait` | **correct，maxerr=1.5e-5** |
| `noscope_sync` | 默认 + 手工 `sync_block_set/wait`（无 scope） | **correct，maxerr=1.5e-5** |

注：`sync_block_set/wait` 的 docstring 要求“必须在匹配 sender 的 scope 内”，
但实测**不加 scope**反而是默认选项下唯一稳定正确的用法（GDN 算子
`chunk_gated_delta_rule_fwd.py` 就是这么写的）。

### 最小验证程序

`probe_04_scope_sync.py`，四种模式：

```shell
python probe_04_scope_sync.py noscope_sync   # PASS 1.5e-5
python probe_04_scope_sync.py scope          # SIGSEGV
python probe_04_scope_sync.py scope_noauto   # 编译过、结果错
python probe_04_scope_sync.py scope_sync     # PASS（手工 flag）
```

### 期望语义

- `scope` 在默认选项下不应崩溃；
- `disable_auto_inject_block_sync=True` 时，`scope` 边界的 GM 交换要么自动
  插 flag（推荐），要么在编译期报“需要显式 sync”；
- 明确 `sync_block_*` 是否必须配合 scope（文档与实现不一致）。

## 5. F4：跨 chunk 软件流水的 UB/L1 预算

### 需求

chunk c+1 的向量工作（门控因子、面板）在 chunk c 的 6 个 dot 在飞时执行，
即 AscendC 的 AIC/AIV ping-pong。

### 现状：**两条路都被容量卡死**

| 写法 | 结果 |
|---|---|
| `tl.range` + `tile_mix_vector_loop/tile_mix_cube_loop`（动态 CV 流水） | 触发 `--enable-mixed-cv=True ...`，但 `ub overflow, requires 5573120 bits while 1769472 bits available` |
| 源级 2 tile 交错（两块的 5+2 个 bf16 面板 + fp32 因子都保活） | `cbuf overflow, requires 6291456 bits while 4194304 bits available!` |
| 源级 `CPT=2` 顺序展开（一次只一块） | 能编译，但更慢：Gram 2649 → **3311 us** |
| 不做流水（当前提交） | Gram 2649 us；融合版 Prepare 的剖析是 `aiv_vec=75.4%`、`aic_mac=6.2%`，拆开后 AIV 仍是唯一关键路径（Gram 未单独剖析，但 6 个 dot 的 MAC 量级不变） |

### 最小验证程序

- `probe_06_cross_chunk_l1_budget.py`：两块、每块 7 个 bf16 面板 + 6 个 dot 的
  形状（单块能过；真 kernel 的 fp32 因子与 6 个累加器保活后才溢出）；
- 真 kernel 复现：`src/flaggems_vllm/runtime/backend/_ascend/FLA/chunk_kda.py`
  的 Gram kernel 改为每程序 2 chunk、dot 先行后处理（`4babebc` 之后的试验），
  即报 `cbuf overflow 6291456 / 4194304`。

### 期望语义

- 流水调度应把操作数**溢出到 L1**（显式 staging buffer）而不是在 UB 里
  双份复制；或在 UB 不足时给出“哪个 buffer 超了”的可操作报错；
- 多缓冲开关应按 buffer 生效（当前 `multibuffer=False` 是全局的，见 F6）。

## 6. F5：长 `exp2/select` 链的向量融合

### 需求

gate 链（sigmoid → cumsum → 2×exp2 → 4×where → 5×mul）希望融合成少量向量
指令，而不是每个 `tl` op 一次 UB 往返。

### 现状：短链能融，长链断崖

`probe_05_vec_chain_fusion.py` 在 12288 个 program（= T=32768/H=24 的 Gram 网格）
上测一条依赖链 `a = exp2(a*1e-4) + where(a>0.5, 1, 0)`：

| 链长 N | 耗时 | 边际 |
|---:|---:|---:|
| 0 | 255 us | — |
| 4 | 271 us | ~4 us/op（**已融合**） |
| 8 | 794 us | ~67 us/op |
| 12 | ~1478 us | ~101 us/op（≈0.53 us/op/AIV） |

即：≤4 个 op 的链能完全融合；到 8 个 op 出现断崖（UB 压力/融合放弃），
之后每加一个 `[64,128]` op 全网格多约 100 us——这正是 Gram 核 25+ 个
逐 op 链耗掉 2.6 ms 的量级来源。

### 最小验证程序

```shell
python probe_05_vec_chain_fusion.py
```

### 期望语义

- 长链应分块融合（例如每 8 个 op 一个 VF），或在超过融合上限时给出提示；
- 提供 `vf_fusion_mode=all-op` 的一个可编译实现（当前 `all-op` 直接编译失败，
  而 `max-parallel`/`ub-aware-op` 中性）。

## 7. 附：探针清单与预期

| 文件 | 预期（本机实测） |
|---|---|
| `probe_01_l1_resident_dot.py copy` | FAIL：`memref.alloc ... read before first write` |
| `probe_01_l1_resident_dot.py dot` | FAIL：`different memory spaces ... cbuf vs ub` |
| `probe_02_slice_as_dot_operand.py col` | PASS，maxerr 7.6e-6 |
| `probe_02_slice_as_dot_operand.py row` | PASS，maxerr 5.3e-5 |
| `probe_03_insert_slice_dot.py` | FAIL：`LLVM ERROR: Unknown PIPE!` + SIGABRT |
| `probe_04_scope_sync.py noscope_sync` | PASS，maxerr 1.5e-5 |
| `probe_04_scope_sync.py scope` | FAIL：SIGSEGV |
| `probe_04_scope_sync.py scope_noauto` | 编译过，maxerr ≈ 1e2（错） |
| `probe_04_scope_sync.py scope_sync` | PASS，maxerr 1.5e-5 |
| `probe_05_vec_chain_fusion.py` | N=4 已融合，N=8/12 断崖（见上表） |
| `probe_06_cross_chunk_l1_budget.py` | 单块形状 PASS；真 kernel 2-chunk 交错 cbuf 溢出 |

一键运行：

```shell
bash tools/triton_ascend_probes/run_all.sh
```

## 8. 优先级建议

1. **F1（L1 视图）**：一个 `bl.to_tensor` 的地址空间修复，直接解锁 AscendC
   的 L1 常驻策略，收益最大、改动面最小（只涉及 buffer→memref 的 lowering）。
2. **F2（insert_slice 进 dot）**：把 abort 改成合法 lower 或明确报错，
   解锁 32 行子块因子。
3. **F3（scope 自动同步）**：默认 SIGSEGV 是硬伤；修好后手工 flag 可以退场。
4. **F4（流水预算）**：需要流水 pass 支持 L1 溢出而不是 UB 双份，工作量最大。
5. **F5（长链融合）**：收益面广但依赖向量化 pass 的整体改造，可最后做。
