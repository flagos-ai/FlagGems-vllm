# Ascend 950DT 上四个 KDA / Causal Conv1D 算子的优化总结

本文总结 `ascend_kda` 分支中四个算子的优化经验：KDA prefill 的 `chunk_kda`、KDA decode / speculative 的 `fused_recurrent_kda`，以及 Causal
Conv1D 的 prefill `causal_conv1d_fn` 和 decode `causal_conv1d_update`。

分析依据是分支提交说明、最终源码、[`ascend_causal_conv1d_fn.md`](ascend_causal_conv1d_fn.md) 和 `tools/test_kda_vs_vllm_ascend.py` 的对比口径。
本文未重新运行性能测试；数值均取自分支已有实测。除特别说明外，**加速比 = AscendC 设备 kernel 时间 / FlagGems 设备 kernel 时间**，大于 1 表示 FlagGems 更快。

测试平台为 Ascend 950DT（arch35 / DAV_3510，64 个 AIV 核），软件环境为 CANN 25.1.rc2、triton-ascend 3.6.0。不同轮次的 shape、head 数和计时范围并不完全一致，
因此只在相同测试条件下比较具体数值。

## 1. 核心结论

| 算子 | 场景 | 代表性结果 | 主要收益来源 |
|---|---|---|---|
| `chunk_kda` | prefill | 四个 shape 几何平均 2.05x；T=32768 为 0.85x | 拆分 AIV/AIC 依赖链、chunk-major 布局、gate 重基座、Gram 改因果前缀分块（MAC 降至 1/4） |
| `fused_recurrent_kda` | decode | bs=4/8/16：1.49x / 1.29x / 1.15x | 整 V tile、二维状态寻址、减少状态遍历 |
| `fused_recurrent_kda` | speculative | bs=16、k=7：0.70x | 将逐 token 窄传输合并成二维加载 |
| `causal_conv1d_fn` | prefill | 原生 channel-first 输入几何平均 0.75x；token-major 视图为 1.05–1.54x | 一波调度、宽通道 tile、按 stride 融合转置 |
| `causal_conv1d_update` | decode | bs=4/8/16：2.37x / 1.66x / 1.43x | 权重预取、批量处理序列、寄存器滑窗 |

950DT 的 AIV 处理标量和向量工作，AIC 处理 dot；单核 UB 约 248 KB，L1 约 512 KB。对这四个算子，**搬运请求的宽度和次数、UB 中同时存活的数据，以及 AIV/AIC 的依赖顺序**，往往比理论
FLOPs 或总搬运字节数更能解释性能。

### 共性方法

| 方法 | 判断依据 | 对应实例 |
|---|---|---|
| 按 UB 预算定 tile | 统计 tap buffer、多缓冲和 fp32 累加器的峰值占用 | Conv prefill 按 `4W+4` B/element 选 tile；KDA 用 BT/BV 控制 gate 与 state 占用 |
| 改善传输形态 | 比较连续行宽、DMA 请求数、尾块和跨步访问 | 宽通道 tile、chunk-major workspace、投机 token 二维 staging |
| 沿独立维度并行 | 确认 token recurrence 是否严格串行 | Conv prefill 的 token range；KDA 沿 chunk/head 或 batch/head 分配任务，state 尽量使用整 V tile |
| 缩短依赖链 | 查看 AIV vector、AIC MAC 和 MTE2 占比 | KDA Prepare 拆成 vector prep 与 Gram，FwdH 与 Finalize 分离 |
| 减少计算和遍历 | 判断 dot、状态遍历和预处理 kernel 是否必需 | Neumann 截断、decay 融合、gate/beta/SiLU 融合 |
| 控制低精度 dot 的动态范围 | 检查指数拆分后的中间操作数 | KDA 使用中点及 32 行子块 anchor，在 bf16 cast 前完成修正 |
| 保持地址仿射 | 检查非仿射索引、视图与流水组合 | Conv tail 使用标量 clamp；避开有版本化问题的 `multibuffer + extract_slice` |

以下按算子说明实现、收益和适用边界。

## 2. KDA prefill：`chunk_kda`

### 2.1 拆分串行状态更新和并行计算

最终实现分为四个 kernel：

```text
vector prep  L2 norm / safe gate / chunk 内 cumsum / ws packing
Gram         6 个 32×32 因果前缀 dot / Aqk、Akk / (I+L)^-1
FwdH         串行状态递推：w@h、(I+L)^-1@diff、kg^T@v_new
Finalize     按 chunk 并行输出：scale*qg@h + Aqk@v_new
```

原 Prepare 是一条 AIV → AIC → AIV 长链，Level1 中 `aiv_vec` 约 75%，`aic_mac` 仅约 6%。拆出不含 dot 的 vector prep 后，Gram 可以专注于 cube 计算；
prep 设置 `multibuffer=False` 控制 UB 占用，Gram 保留流水。Gram 内先发出 6 个 dot，再做 Aqk 后处理，使 AIV 后处理与 AIC 的 Akk dot 重叠。T=32768 时，这一调整使
intra kernel 从 4830 降至 4270 us。

本轮进一步把 Gram 从“3 个秩项 × 2 个矩阵 = 6 个 64×64 dot”改成 AscendC arch35 的因果前缀形式：两个 32 行 band，每个 band 只和它的因果前缀做
`[32,128] @ [128,32]`，6 个 dot 的 MAC 从 3.15M 降到 786k（4.0x）。面板从 7 个 `[64,128]`（112 KiB）降到 7 个 `[32,128]`（56 KiB），UB 峰值约 1.0 Mbit，
不再需要显式 L1 常驻；默认 `NEUMANN_STAGES=0` 时 `(I+L)^-1 = I-L`，Akk 的四个象限直接写出。T=32768 时 Gram 2649 → 1261 us，prep+Gram 4002 → 2614 us。

FwdH 只保留真正跨 chunk 依赖的 3 个 dot，把其余 2 个输出 dot 移到可按 chunk 并行的 Finalize。串行 scan 在 T=32768 时从 3143 降至 1266 us。

### 2.2 调整布局与并行切分

- 中间量 `ws/Aqk/Akk/v_new/h_buf` 改为 chunk-major 布局 `[NC, HV, BT, ...]`，使每个 chunk-head 成为 8–48 KiB 的连续对象。旧的 token-major
  布局让串行 scan 的 MTE2 和 scalar 占比分别达到 49% 和 36%。
- state kernel 使用 `BV=V`，使每个 `(seq, head)` 只执行一遍完整递推。T=24576、H=12 时，BV=32/64/128 分别耗时 5.66/4.38/3.90 ms。拆 V 并不能缩短 token
  方向的串行链。
- Prepare/Gram 沿 `(chunk, head)` 并行。线性 grid 上限为 32768；超过上限时通过 CPT 让一个 program 处理多个 chunk，控制 launch 规模。
- gate 前缀和改用 `tl.cumsum`，取代 `static_range` 的逐步读写链；T=32768 时节省约 440 us。

### 2.3 扩大 BT 前处理数值范围

调用方仍可传入 `chunk_size=16`，Ascend kernel 内部使用 BT=64，必要时重建 `chunk_indices`。BT 增大使串行 scan 长度减半，但 gate 因子不能直接拆成
`exp2(gc_i) * exp2(-gc_j)`：即使最终因果因子不超过 1，中间操作数也可能达到约 `2^62`，在 bf16 dot 前溢出或严重失真。

实现先以 chunk 中点为 anchor 计算指数差。BT=64 时再分成两个 32 行子块，各用自己的 anchor，并将跨子块修正 `exp2(anchor1 - anchor0)` **在 cast 和 dot 之前**折入 k
侧操作数。anchor 取每个 band 有效部分的中点（对齐 AscendC `gRef`），避免 33–47 行尾 chunk 把 anchor 落到 `T` 之外后 `kM * exp2(a1-a0)` 溢出为 NaN。dot 使用 bf16 操作数、fp32 累加；safe gate 的因果因子在 dot 后相乘，因为该因子位于 `[0, 1]`。修复前 BT=64 会产生 0.6M–17.8M 个非有限输出，修复后为 0；四个 prefill
shape 提升 1.18–1.20x。将修正放在 dot 之后逐元素相乘不满足原计算式，CPU probe 已验证这一点。

另一个直接减少计算的改动是截断 Neumann 展开。`L` 为严格下三角矩阵，级数会精确终止；在实测数据中 `||L||₂ ≈ 0.02–0.025`，stage-0 的截断误差约 `4–6e-4`，低于 bf16 的精度量级。默认不再做
3 个额外 dot，相对 AscendC 的加速比由约 1.37–1.41x 提升至 1.50–1.55x。该结论依赖所测输入分布，迁移到其他分布时需要重新验证精度。

### 2.4 结果与限制

| shape | FlagGems (us) | AscendC (us) | 加速比 |
|---|---:|---:|---:|
| T=8192，H=24，seqs=2 | 1120 | 3065 | 2.74x |
| T=16384，H=12，seqs=4 | 1121 | 3072 | 2.74x |
| T=24576，H=6，seqs=8 | 867 | 2394 | 2.76x |
| T=32768，H=24，seqs=2 | 4299 | 3636 | 0.85x |
| **几何平均** | | | **2.05x** |

T=32768 仍落后约 660 us：分核为 prep 1353 / Gram 1261 / FwdH 1266 / Finalize 422 us，AscendC 为 1654 (+413 l2norm) / 840 / 718 us。Gram 的 1261 vs 413 us 是主要剩余
差距，Profiling 为 `aiv_vec=94%`、`aic_mac=15%`，即 AIV 向量链与跨 head CV 重叠不足，而不是 L1 带宽或 ND2zN 拷贝。旧结论“需要 L1 常驻 7 个 panel”来自全矩阵公式的
UB 峰值（2.50–3.28 Mbit）；前缀分块后只需约 1.0 Mbit，不再依赖该能力。继续增大 BT 仍需要新的子块重基座方案。此外，T=32768、H=24 时 `h_buf` 约 402 MiB；
超过 `FLAGGEMS_ASCEND_KDA_H_BUDGET_MB` 后实现会回退到逐 chunk 的 portable Triton 路径，避免大 shape 耗尽内存。

## 3. KDA decode / speculative：`fused_recurrent_kda`

每个 token 做一次 rank-1 状态更新，token 方向严格串行，独立任务来自 `(batch, head)`。投机解码则在一个任务中连续处理 `width` 个候选 token。

### 3.1 保持状态 tile 完整

实现使用 `grid = min(total_tasks, AIV 核数)` 的 persistent grid，program 内部用 grid-stride 分配连续任务。dense 状态 `[V,K]` 通过二维地址
`rv[:, None] * SS[2] + rk[None, :]` 访问，避免先构造 `BV*BK` 个 flat index 再 reshape。旧形式在 BV=64/128 时分别需要 2.19/3.83 Mbit UB，超过约
1.77 Mbit 的可用空间；二维寻址允许 `BV=V`。bs=128 时，整 V tile 比 8-wide tile 快约 4.3x。

`A_log` 与 `dt_bias` 只依赖 `(head, key)`，因此移到 token 循环之外。每 token 的 decay 被折进两次 reduction：先计算 `kd = k * decay` 和
`delta = (v - sum(acc * kd)) * beta`，再更新 `acc = acc * decay + delta * k`；省去独立 `acc *= decay` 对 `[V,K]` 状态的第三次遍历，实测 121.7
→ 94.7 us。L2 norm、safe gate 和 beta sigmoid 也在 kernel 内完成。需要原地更新时直接写回 state pool；投机模式使用二维 slot table 和 accepted-token
列选择种子状态。

### 3.2 合并投机 token 的窄传输

当 `BTOK>1` 时，一次加载 `[BTOK,BK]` 的 q/k/g 和 `[BTOK,BV]` 的 v，循环内用 `tl.where(sel, ...)` 选择当前行。这样将 `4*BTOK` 次窄传输变为 4 次二维传输；
bs=16、k=7 时从 82.0 降至 75.6 us。`BTOK=1` 没有可摊销的加载开销，实测 staging 反而变慢，因此不启用。

### 3.3 结果与口径

| 场景 | 相对 AscendC | 测试轮次 |
|---|---:|---|
| decode bs=4/8/16 | 1.49x / 1.29x / 1.15x | 12 local heads，较新一轮 |
| decode bs=32/128 | 0.81x / 0.68x | 8 local heads，较早一轮 |
| speculative bs=8，k=3 | 0.68x | 二维状态寻址后 |
| speculative bs=16，k=7 | 0.70x | 二维 token staging 后 |

较早一轮还测得 decode bs=4 为 7.7 us、1.20x；bs=32/128 分别为 25.3/90.0 us。**两轮 head 数不同，不能把这些加速比视为同一条 batch 扩展曲线。**已有结果显示小 batch 领先，
而较早一轮的大 batch 落后；大 batch 仍需压缩串行链或改善 AIV 上的任务重叠。

## 4. Causal Conv1D prefill：`causal_conv1d_fn`

算子对 `W=2..5` 的 depthwise 因果卷积执行可选 SiLU，并更新 `conv_state`。计算主要由 MTE2 和 AIV 承担，AIC 不参与。

### 4.1 按 UB 和连续行宽选择 tile

每个元素的估算开销是 `W` 个 tap buffer × 2（多缓冲）× 2 B，加上 4 B 的 fp32 累加器，即约 `4W+4` B。host 以约 184 KB 的可用预算反推 `BLOCK_T * BLOCK_N`；W=4
时约可容纳 9216 个元素。通道 tile 对齐 256，优先整除 `dim`，避免 masked 尾块。dim=9216/4608/2304 分别选 `4608×2 / 4608×2 / 2304×4`，连续行宽达 9 KB，且没有尾块。
此前 `2048+256` 的通道尾块会使性能下降约 2x。

channel-first 路径中 token 轴连续，tile 方向相应调整，最终选择 `32×256`。tile 高度只取 2 的幂；非 2 的幂在所用 triton-ascend 版本上曾出现行计算错误。

### 4.2 一波调度与仿射地址

host 的 `_plan_ranges` 将每个 request 分为连续、互不跨 request 的 token range，长度按 `BLOCK_T` 对齐，并使 `range 数 × channel tile 数 ≈ AIV 核数`。
每个 program 流式处理整段，减少多波执行时重复支付的 prologue。卷积前 `W-1` 个 token 通过滚动递推 `s_k ← s_{k+1} + w_k x_t` 处理，主循环无需 masked history
correction。

主循环保持仿射地址 `t_first + it * BLOCK_T + arange`。tail 使用“标量 clamp tile 起点 + 短 tile 循环”，允许最后一块重复计算少量行，避免逐 lane clamp 退化成逐元素
DMA；这一改动在对应实验中使 376 降至 253 us。channel-first 且 `BLOCK_T≥256` 时，tail tile 高度从 32 提到 128，KDA shape 约再快 6%。full tile 使用
`num_stages=2` 和多缓冲来重叠加载与计算。

### 4.3 根据输入 stride 选择计算几何

| 输入布局 | 使用的 tile | 处理方式 |
|---|---|---|
| 通道轴 unit stride | `[tokens, channels]` | 沿通道宽行读取，速度最快 |
| token 轴 unit stride | `[channels, tokens]` | tile load 中完成转置 |
| 两轴均非 unit stride | 先 contiguous | 避免内核中的双向跨步读取 |

第二种路径相对“显式拷贝 + 第一种路径”快 1.26–1.79x。`query_start_loc.cpu().diff()` 将 device diff 与 D2H 合为一次 D2H，每次调用少一个 device kernel；
per-range plan 通过缓存 workspace 一次上传。

tap 保留“同一 tile 按行偏移重读”：相邻迭代的数据高度重叠，后续读取通常由 L2 服务。单次 halo 读加 UB 内移位虽然在 probe 中快 1.3–1.4x，但 `num_stages=2` 的实际 kernel
会读到旧行；`num_stages=1` 正确却慢约 35%，因此未用于最终路径。

### 4.4 结果与布局边界

| shape，原生 channel-first | FlagGems (us) | AscendC (us) | 加速比 |
|---|---:|---:|---:|
| T=8192，dim=9216 | 231.6 | 163.1 | 0.70x |
| T=16384，dim=4608 | 231.6 | 237.2 | 1.02x |
| T=24576，dim=2304 | 203.2 | 114.9 | 0.57x |
| T=32768，dim=9216 | 806.9 | 635.3 | 0.79x |
| **几何平均** | | | **0.75x** |

同一 kernel 在 token-major 视图下耗时分别为 155/155/118/582 us，相对 AscendC 为 1.05–1.54x。布局是集成时的关键条件：调用方若能交付通道轴 unit stride 的视图，
可使用更快的 `[tokens, channels]` 路径；真正的 channel-first 连续输入则走 `[channels, tokens]` 路径。

## 5. Causal Conv1D decode：`causal_conv1d_update`

每个 sequence 只有一个 token，投机解码时为 `k` 个 token；`dim` 较大，且 `conv_state` 需要原地 shift/append。

1. **预取权重并批量处理序列。** 每个 program 一次加载并 flatten `[W, BLOCK_N]` 权重，tap 循环用 `extract_slice(offset=j*BLOCK_N)` 从寄存器中读取。
   `B_TILE` 取 1/2/4/8，按 `batch*grid` 选择，目标约 80–160 个 program，摊薄预取与 prologue。
2. **保留寄存器滑窗。** `col0..col4` 持有前 `W-1` 个 token；每步只加载当前 token 并移动窗口，避免反复读取历史槽。
3. **分块更新 state。** 按 `T_CHUNK` 处理，而不分配完整的 `NP2_STATELEN × BLOCK_N` tile。`BLOCK_N=1024` 时 `T_CHUNK=1`，否则为 48。短序列执行“旧
   state 前移 + x 追加”，长序列直接用 x 尾部覆盖；mask 同时考虑 cache line 边界和 pad slot。
4. **保证原地语义。** 先读历史列，再写 state，允许 `out == x`，从而省去一次输出分配或拷贝。`HAS_BIAS`、`KERNEL_WIDTH`、`IS_VARLEN`、`IS_APC_ENABLED`、
   `IS_SPEC_DECODING` 均为 constexpr，编译期即可消除不适用的分支。
5. **使用 stock triton-ascend 切片接口。** 用 `cann.extension.extract_slice` 替代 `tle.dsa.extract_slice`，并将 width 1–6 的展开阶梯合并成
   `static_range(KERNEL_WIDTH)` tap 循环。

分支记录显示，conv decode 与 AscendC 逐位一致；bs=4/8/16 的加速比为 2.37x / 1.66x / 1.43x。这是四个算子中最明确的性能优势，但其 `B_TILE`、`T_CHUNK` 启发式仍需在其他
batch 和 `dim` 上重新评估。

## 6. 试过但未采用的方案

这些实验说明优化必须结合实际 DMA 形态、UB live range 和依赖链判断。

| 算子 | 方案 | 实测或否决原因 |
|---|---|---|
| `chunk_kda` | 为旧的全矩阵 Gram 方案在 L1 常驻 7 个 `[64,128]` panel | 需 2.50–3.28 Mbit UB，超过 2.03 Mbit；改成 32 行因果前缀后只需 7×`[32,128]`（约 1.0 Mbit），无需显式 L1 常驻 |
| `chunk_kda` | CPT≥2 合并 chunk | CPT=2 为 3.44 ms，CPT=1 为 3.11 ms；该场景不受 launch 数主导 |
| `chunk_kda` | 融合 prep + Gram，复用 q/k/g 寄存器以省掉二次加载和 norm/beta 重算 | 原型 4088 us vs 拆分 2644 us（1.55× 更慢）；`extract_slice` 引入的布局转换、UB 压力和重新拉长的 AIV→AIC→AIV 依赖链抵消了省下的搬运，拆分结构更优 |
| `chunk_kda` | `al.scope` + `sync_block_*` 做 Gram 的 7 面板 CV 双缓冲 | 最小 hand-off PASS（`supported/s6_scoped_cv_handoff.py`，需 `enable_mixed_cv=True, disable_auto_inject_block_sync=True`）；面板逐位正确且 cube 实际 load 到的数据与 GM 一致；但同一 6-dot body 不套 scope 时与 torch 一致（7e-6），套 `al.scope("cube")` 后出现 inf/NaN（最小复现 `defects/d5_scoped_cube_multidot.py`），是 scoped cube 多 dot codegen 问题，不是 sync/load 可见性。T=4096 仍为 344 us vs 169 us，GM 往返成本也尚未解决 |
| `chunk_kda` | Gram 用 CPT=2/4 合并 chunk | 1967/2127 us vs CPT=1 1290 us；每 program 的双份面板与 dot 让 UB/串行链变长 |
| `chunk_kda` | Gram 调 `vf_merge_level=1..4`、`multibuffer`、`num_stages=1..3` | 全部在 1289–1294 us，无差异；说明 Gram 已到当前代码形态的 AIV 向量下界，进一步收益需要跨 head CV 重叠 |
| `chunk_kda` | prep 保存 q/k rstd、beta sigmoid 供 Gram 复用 | 上限约 3%（Gram 1268→1228 us）；为 <1% 整算子收益增加 workspace/签名复杂度，暂不做 |
| `chunk_kda` | 只存 gate cumsum 最后一行 | 字节数降至 1/32，但窄 store 与标量偏移使 intra kernel 从 1395 升至 2948 us |
| `chunk_kda` | 把输出 rebase 到旧 state | 增加一次 `[V,K]` 遍历，只对最长 case 有益 |
| `fused_recurrent_kda` | 拆 V 增加 program；逐 token 分别加载 q/k/g/v | 前者重复执行串行链；后者在投机场景产生 `4*BTOK` 次窄传输 |
| `causal_conv1d_fn` | 单次 halo 读加 `extract_slice` | 多缓冲下静默读旧行（maxerr≈16）；关闭流水正确但慢约 35% |
| `causal_conv1d_fn` | channel-first 通道 tile 提到 512/1024；二维回写 `conv_state` | 前者 fragment 补齐后可达 462 KB，UB 溢出；后者每行仅 6–8 B，慢约 13% |
| `causal_conv1d_fn` | `loop_unroll_factor`；逐 lane clamp tail | 前者使一种布局 155 → 193 us，其他 shape 编译失败；后者触发逐元素 DMA |
| `chunk_kda` | 依靠 `num_stages` 流水化串行 state loop | 取 0/1/2/3 均约 3.14 ms，后端未流水化该 loop |

编译器相关的两点值得保留为实现约束：`bl.alloc` 要显式指定 `al.ascend_address_space.UB`；TLE 切片原语不可用时，`chunk_kda` 可使用 stock triton-ascend 的
`cann.extension.extract_slice` / `insert_slice`。对于 `multibuffer` 与视图组合，应先用最小 probe 验证正确性，再用于生产 kernel。

## 7. 复用时的检查顺序

1. 用 profiler 确认 AIV、AIC、MTE2 与 scalar 的时间占比，找出关键依赖链。
2. 列出每个 live tile 的大小、多缓冲倍数与 fp32 累加器，按 UB 预算定 tile。
3. 检查每次 DMA 的连续字节数、请求数、尾块与跨步；必要时调整 workspace 布局或将窄传输合并为二维 tile。
4. 只沿独立维度分配 program；对串行 recurrence 保留整 V tile，并控制 launch 的波次数。
5. 检查能否去掉 dot、状态遍历或独立预处理 kernel；再考虑 AIV/AIC 重叠。
6. 低精度 dot 前验证指数动态范围与累加误差，把重基座修正放在 cast 前。
7. 保持地址仿射，对视图、切片和流水组合做最小正确性 probe，并记录失败方案。

## 8. 代码与数据索引

| 内容 | 位置 |
|---|---|
| KDA prefill | `src/flaggems_vllm/runtime/backend/_ascend/FLA/chunk_kda.py` |
| KDA decode / speculative | `src/flaggems_vllm/runtime/backend/_ascend/FLA/fused_recurrent_kda.py` |
| Conv prefill | `src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py` |
| Conv decode | `src/flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_update.py` |
| Conv 详细分析 | `docs/analysis/ascend_causal_conv1d_fn.md` |
| 四算子对比脚本 | `tools/test_kda_vs_vllm_ascend.py` |
| 编译器 probe | `tools/triton_ascend_probes/` |
| KDA BT / Neumann 验证 | `tools/_kda_bt_accuracy.py`、`tools/_kda_neumann_order_check.py`、`tools/_kda_bt64_subblock_math_check.py` |

相关提交：

```text
e65d549  recurrent KDA decode + conv 去 TLE
4ae1c26  recurrent state 改 2D tile
de023a7  stock triton-ascend 上运行 KDA kernel + 重调 BV
904dd6a  chunk_kda BT=32/64、conv BLOCK_N 加宽
c52c5cb  BT=64 子块 gate rebasing
440d3c3  Neumann 截断到 0 个额外 dot
334b751  FwdH / Finalize 拆分 + chunk-major workspace
4babebc  Prepare 内 Aqk 后处理与 Akk dot 重叠
0725f74  Prepare 拆 vector prep + Gram
2ab1b70  conv prefill 按 AscendC tiling plan 重写
81629de  conv 融合 channel-first 转置
1727660  conv channel-first tail 提到 128 行
```
