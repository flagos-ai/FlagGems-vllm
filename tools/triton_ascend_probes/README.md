# KDA chunk 前向在 Ascend950DT 上的 triton-ascend 能力清单

本文档把 KDA chunked prefill（`chunk_kda`）在 **stock triton-ascend
3.6.0.post2（不使用 FlagTree/TLE）** 上落地 AscendC 级优化时遇到的能力，按三类
分开：

- **已可用（supported/）**：已经能在 Triton 里做到，**不再作为需求**；
- **需求（gaps/）**：Triton 里做不到/不实用，且**已确认 AscendC 能实现**（附代码位置）；
- **缺陷（defects/）**：崩溃或错误，但有已验证的绕行方案，**不作为能力需求**，按 bug 处理。

环境：Ascend950DT_9582，CANN 8.5，`torch_npu` 2.9，triton-ascend 3.6.0.post2。

## 0. 背景与当前性能

AscendC（`aclnnChunkKdaFwdV2`）把 Prepare 压到 Triton 的一半，靠三件事：
L1 常驻操作数、32 行子块门控、AIC/AIV 混核流水。Triton 版本已经对齐三段式并做了
源级 AIV/AIC 重叠，T=32768/H=24/seqs=2 的分核耗时（us）：

| kernel | FlagGems(Triton) | AscendC | 说明 |
|---|---:|---:|---|
| Prepare（prep + Gram） | 1353 + 2649 | 1654 + 413(l2norm) | 主要差距 |
| FwdH | 1266 | 840 | AIC MTE2 占 73% |
| Finalize | 421 | 718 | 已优于 AscendC |

融合版 Prepare 的剖析为 `aiv_vec=75.4%`、`aic_mac=6.2%`：AIV 向量链是唯一关键路径。

**结论：两类分离后，真正剩余的编译器需求只有 R1、R2。** 其余能力已验证可用
（第 1 节），或属于有绕行的缺陷（第 3 节）。

## 1. 已可用 —— 不再作为需求

这些能力在 stock triton-ascend 上实测通过，算子代码可以直接用；探针在
`supported/`，只作为参考与回归基线。

| 能力 | 探针 / 模式 | 实测 |
|---|---|---|
| 列偏移 `extract_slice` 作 dot 操作数 | `s1_slice_dot.py col` | PASS，maxerr ≤1.2e-5 |
| 行偏移 `extract_slice` 作 dot 操作数 | `s1_slice_dot.py row` | PASS，maxerr ≤7.7e-5 |
| L1 zN 面板上再做列/行切片作 dot 操作数 | `s1_slice_dot.py l1_col / l1_row` | PASS，7.6e-6 / 6.1e-5 |
| **显式 L1 驻留 dot**：4D zN + `al.copy(UB→L1)` + `al.dot(format_*="fractal")` | `s2_l1_resident_dot.py dot` | PASS，maxerr ≤6.1e-5 |
| 同上，`copy_from_ub_to_l1`（deprecated 别名） | `s2_l1_resident_dot.py copy` | PASS，≤4.6e-5（有 warning） |
| **L1 子视图拼装**：`bl.subview` + `al.copy` 把 32 行子块写进 zN 槽位 | `s3_l1_subview_assemble.py` | PASS，≤6.1e-5 |
| **手工跨核同步**：`al.sync_block_set/wait`（不加 `scope`） | `s4_manual_cross_core_sync.py` | PASS，≤1.5e-5 |

说明：

- `s3` 是 `insert_slice`（缺陷 D2）的官方替代路径，也是 AscendC 自己的做法
  （`prepare_cube.h:432-452` 把 32×32 象限直接 DataCopy 进 L1 的 zN 槽位）；
- `s4` 是 `scope`（缺陷 D1）的替代路径，`chunk_gated_delta_rule_fwd.py` 已在用；
- **`s2` 的可扩展性受 R1 限制**：小 tile 可用，但 KDA Gram 的 7 面板规模会超 UB
  （见 R1 的数据）。

## 2. 需求 —— AscendC 可做，Triton 不可用

### R1：L1 操作数的 ND→zN 暂存直通（P0）

#### 需求

把 ND 的 bf16 面板直接搬成 L1 的 zN 布局，供 `al.dot(fractal)` 复用，不在 UB 里
物化第二份 4D tile：

```python
l1 = bl.alloc(tl.bfloat16, [K // 16, R // 16, 16, 16], al.ascend_address_space.L1)
al.copy(bl.to_buffer(a_nd_2d, al.ascend_address_space.UB), l1)  # ND(UB) -> zN(L1)
c = al.dot(bl.to_tensor(l1), tb, format_a="fractal", format_b="fractal")
```

#### AscendC 证据（确认可实现）

`kda/chunk_kda_fwd_prepare/op_kernel/arch35/chunk_kda_fwd_prepare_cube.h`：

- 258-290：`AscendC::Nd2NzParams qkCopy` + `AscendC::DataCopy(qPlusL1, ...)`，
  把 Q+/K+/K− 按 **GM(ND)→L1(NZ)** 一次搬入（72 KiB），注释明确
  “后面的四次 MMAD 不再读 GM”；
- 423-452：`DataCopy` 生成标准 64×64 zN，并按 32×32 象限把 B / X0 / negX1
  直接写入 L1 的 `(M1=2, N1=0)` 槽位。

#### Triton 现状（实测，`gaps/g1_nd2nz_staging.py`）

真实 Gram 网格（12288 program，7 个 `[64,128]` 面板、6 个 dot）：

| 模式 | 暂存到 L1 | UB 需求 | 结果 |
|:--|:--|--:|:--|
| `plain` | 无 | ~2.0 Mbit | PASS（同结构微基准约 1.84 ms） |
| `l1a` | 4 个 A 面板（无 trans） | 3.15 Mbit | **ub overflow** |
| `l1b` | 3 个共享 B 面板（含 trans） | 3.28 Mbit | **ub overflow** |
| `l1` | 全部 7 个面板 | 2.50 Mbit | **ub overflow** |
| 2-chunk 版（真 kernel 复现） | 14 个面板 | 3.94 Mbit（L1 6.29/4.19 的 cbuf 另计） | **溢出** |
| 可用预算 | — | 2.03 Mbit | — |

根因：zN 只能靠 UB 侧的 `reshape/permute/trans` 构造，等于再物化一份 4D tile。
2D ND 直通 L1 也不行（`bl.to_tensor(l1)` 报 cbuf/ub，见缺陷 D3）。

#### 影响

没有 R1，L1 常驻只能用于单块；KDA Gram 的 7 面板、以及 2-chunk 双缓冲都没有
可行形态。修好后 Gram 预期 2649 → 1.9–2.1 ms，整算子 5712 → 5.0–5.2 ms
（约 0.68–0.70x）。

### R2：长 `exp2/select` 链的向量融合（P1）

#### 需求

把 gate 链（sigmoid → cumsum → 2×exp2 → 4×where → 5×mul）融合成少量向量指令，
而不是每个 `tl` op 一次 UB 往返。

#### AscendC 证据（确认可实现）

`chunk_kda_fwd_prepare_vec.h:1128, 1146, 1412` 用
`AscendC::VF_CALL<Detail::StageV0Vf<...>>` / `StageV6Vf` 手写融合向量函数。

#### Triton 现状（实测，`gaps/g2_vec_chain_fusion.py`）

全网格（12288 program）上的依赖链 `a = exp2(a*1e-4) + where(a>0.5,1,0)`：

| 链长 N | 耗时 | 边际 |
|---:|---:|---:|
| 0 | ~256–276 us | — |
| 4 | ~271 us | ~0 us/op（**已融合**） |
| 8 | ~794 us | ~65 us/op（断崖） |
| 12 | ~1478 us | ~101 us/op（≈0.53 us/op/AIV） |

`vf_fusion_mode=all-op` 直接编译失败；`max-parallel`/`ub-aware-op` 与默认一致。

#### 期望

长链分段融合（例如每 4–8 op 一个融合块）；修复 `all-op` 编译失败。

## 3. 缺陷 —— 有绕行，不作为需求

| 编号 | 现象 | 复现 | 绕行（已可用） |
|---|---|---|---|
| **D1** | `al.scope` 默认 **SIGSEGV**；`enable_mixed_cv=True + disable_auto_inject_block_sync=True` 后编译通过但**结果错**（maxerr≈1e2） | `defects/d1_scope_auto_sync.py scope / scope_noauto` | `s4_manual_cross_core_sync.py`（手工 `sync_block_*`） |
| **D2** | `al.insert_slice` 结果作 dot 操作数 → `LLVM ERROR: Unknown PIPE!` + SIGABRT（放进显式 L1 也一样） | `defects/d2_insert_slice_dot.py insert / l1_insert` | `s3_l1_subview_assemble.py`（subview+copy） |
| **D3** | 2D ND 的 `bl.to_tensor(L1)` → `different memory spaces ... cbuf vs ub`；`to_tensor(L1, target_shape=...)` → `unsupported op to get out operands` | `defects/d3_l1_view_2d.py nd / target` | `s2_l1_resident_dot.py`（4D zN + `al.dot(fractal)`） |

另有一条**预算限制**（非缺陷、非需求）：编译器自动 CV 流水（`tl.range` +
`tile_mix_*`）在融合版 Prepare 上要 5.57 Mbit UB（可用 1.77）；这是自动调度的
选择问题——R1 修好后我们会手写 2-chunk 交错，不需要该 pass 自己展开。

## 4. 目录与复现

```text
tools/triton_ascend_probes/
  README.md                     # 本文档
  run_all.sh                    # 三组探针一次跑完
  supported/                    # 已可用，不再是需求
    s1_slice_dot.py             # 行/列切片作 dot 操作数（含 L1 zN）
    s2_l1_resident_dot.py       # 4D zN + al.dot(fractal) L1 驻留
    s3_l1_subview_assemble.py   # subview+copy 拼装 32 行子块
    s4_manual_cross_core_sync.py# 手工 sync_block_* 跨核同步
  gaps/                         # 需求（AscendC 可实现）
    g1_nd2nz_staging.py         # R1：ND→zN 暂存预算
    g2_vec_chain_fusion.py      # R2：长链融合
  defects/                      # 崩溃/错误，有绕行
    d1_scope_auto_sync.py       # D1
    d2_insert_slice_dot.py      # D2
    d3_l1_view_2d.py            # D3
```

一键运行（仓库根目录）：

```shell
bash tools/triton_ascend_probes/run_all.sh
```

单项运行：

```shell
PYTHONPATH=src:FlagGems/src python tools/triton_ascend_probes/gaps/g1_nd2nz_staging.py l1a
```

预期结果：

| 组 | 命令 | 预期 |
|---|---|---|
| supported | `s1 ... col/row/l1_col/l1_row` | PASS（≤7.7e-5） |
| supported | `s2 ... dot/copy` | PASS（≤6.1e-5） |
| supported | `s3` | PASS（≤6.1e-5） |
| supported | `s4` | PASS（≤1.5e-5） |
| gaps | `g1 plain` | PASS（基线） |
| gaps | `g1 l1a/l1b/l1` | **FAIL**：ub 3.15 / 3.28 / 2.50 Mbit |
| gaps | `g2` | N=4 已融合；N=8/12 断崖 |
| defects | `d1 scope` | **FAIL**：SIGSEGV |
| defects | `d1 scope_noauto` | 编译过，maxerr≈1e2 |
| defects | `d2 insert/l1_insert` | **FAIL**：`Unknown PIPE!` + SIGABRT |
| defects | `d3 nd/target` | **FAIL**：cbuf/ub / unsupported op |

## 5. 优先级

1. **R1（P0）**：一条 ND2NZ 暂存能力，直接决定 L1 常驻与 2-chunk 流水能否落地；
   预期整算子 0.68–0.70x。
2. **R2（P1）**：长链融合，影响所有向量密集 kernel 的 codegen 质量。
3. **D1/D2/D3（bug）**：都有绕行，不阻塞；但默认 `scope` SIGSEGV 和
   `insert_slice` SIGABRT 建议优先修，避免其他算子踩坑。
