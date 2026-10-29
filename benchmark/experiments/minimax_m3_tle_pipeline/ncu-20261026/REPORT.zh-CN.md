# MiniMax M3 attention：NCU 拆解与 #85 复用判断

日期：2026-10-26。结论：代表性大 q 上，新增 WS/深缓冲的退化已有硬件计数器支持；
对 #85 应采用按域评估，不能表述为框架不支持，也不能宣称它在所有输入上都慢。
[可直接使用的说明文字](REUSE_EXPLANATION.zh-CN.md)已单独保存。

## 范围与方法

- 单张 H100 80GB HBM3 / SM90 / 132 SM；Torch 2.13.0+cu129，FlagTree 0.7.0 /
  Triton 3.6.0，沿用已冻结的最小 #1415 load-mask 依赖修复工具链。
- q8192、seq16384、GQA8、Hkv1、D128、page128、TopK16。相同随机种子、页表、
  Q/KV 内容，分别生成 compact/recent、correlated8、independent scattered。
- 主域为 FP8-E4M3 KV 存储、BF16 QK/PV、FP32 累加。另设 BF16 存储控制组：将同一
  FP8 fixture 的值精确转换成 BF16，从而只改变存储及可用调度，不改变输入值。
- 对照精确 #85 head `027ebdc9eb23e37aa3238d4a8b09296bc64e0dd8`，原文件 SHA256
  `6c7bb5b5768b6c7c925da9457c20d812195a5b7f66c2fc9608d071d86ae64a60`。
  未修改其 kernel 或 dispatch；harness 仅提供 `flag_attn.utils.current_platform`
  的查询 stub，prefill 不使用其中的 PDL 查询。
- 基线 SHA256 `1ab42fc298abf8a950d02ac5204869ed0f2eeade5185b17b58416e1be845a789`，
  即此前冻结的 serving precursor，含后加 sparse-wide；它不是 #979 package wrapper。
- NCU 2025.2.1：14 份 detailed + SourceCounters 报告、6 份定向计数器报告，
  共 **20 份报告、45 次 kernel action**。已在实际 SM90 上枚举指标；不把缺失指标当零。
  所有报告通过 profiler Python API 解析，所有报告文件 SHA256 已复核。
- NCU 仅采集 30 次预热之后的一次完整 wrapper 调用；保留原有 clocks/driver 设置，
  `--cache-control none --clock-control none`。Profiler replay 的时间只用于诊断。
- 独立普通计时采用每 arm 30 warmup、3 rounds、每轮 100 samples、交替 A/B，分别
  测 eager 和 Graph。12 组通过门禁的对照，共 **14,400 次计时 wrapper 调用**。
  完整范围包含 flag、compact、wide、selected fallback，而非仅计算 kernel。

## 1. WS compact：同步与 local-memory 成本抵消了 TMA 收益

下表仅对比 compact 输入，主域均为 FP8 存储、BF16 QK/PV。
NCU 列对应主要 compact kernel，Graph 列对应完整调用链。

| 指标 | 冻结 compact-union | WS / consumer4 / ring2 |
|---|---:|---:|
| 完整 Graph median，µs | 318.72 | 645.26 |
| NCU Tensor Core 活跃率，elapsed | 24.22% | 11.51% |
| NCU achieved occupancy | 12.48% | 12.50% |
| 每 block shared memory，含 driver 字节 | 99,332 B | 132,332 B |
| barrier stall / active warp cycles | 20.06% | 46.30% |
| long-scoreboard stall / active warp cycles | 5.48% | 19.40% |
| local load sectors | 32,768 | 21,972,992 |
| local store sectors | 32,768 | 21,183,488 |
| local 请求流量，sector × 32 B | 2 MiB | 1.286 GiB |
| DRAM active cycles，elapsed | 3.48% | 3.31% |

WS 的 occupancy 没有提高，Tensor Core 活跃率反而下降，barrier 与 local 流量大幅
增加。这与此前 SASS 中的 stack/LDL/STL 证据一致。SourceCounters 的 WS 最大热点
是编译器映射到 kernel 入口的同步位置；不能把该入口行号误读成精确的某个 Python
barrier 语句。定向采集的 barrier 指标提供了独立量化。

这里的 local 流量是 L1TEX 请求 sector，不等于实际 DRAM 传输字节；stall 百分比是
活跃 warp-cycle 的比例，也不等于整个 kernel wall time 的占比。两者 DRAM 活跃率均
较低，本轮重复输入条件下没有证据支持“HBM 带宽打满导致退化”。

另一个反例：WS 把 shared-store bank-conflict 计数从 4,358,256 降到 557，但仍然
慢约一倍；局部访存指标改善不能覆盖新增同步和寄存器压力的代价。

## 2. 非 WS sparse-wide：第三槽触发 occupancy 降档和大量 local 流量

Independent scattered 输入，均为 BF16 QK/PV。NCU 对应真正处理数据的 wide kernel，
不是前面的 compact 早退 kernel。

| 指标 | 非 WS / ring2 | 非 WS / ring3 |
|---|---:|---:|
| 完整 Graph median，µs | 1,555.54 | 2,475.87 |
| 每 block shared memory，含 driver 字节 | 116,312 B | 149,100 B |
| shared memory 限制的 resident blocks / SM | 2 | 1 |
| achieved occupancy | 11.82% | 6.25% |
| Tensor Core 活跃率，elapsed | 20.54% | 13.04% |
| local load / store sectors | 0 / 0 | 26,325,312 / 13,400,224 |
| local 请求流量，sector × 32 B | 0 | 1.184 GiB |

增加一个槽没有改变注意力数学或增加有效工作，却越过 shared-memory 容量门槛，并伴随
local 流量出现。该变化与完整 Graph 耗时增长约 59% 一致，支持继续使用浅缓冲而非
盲目加深流水线。两槽版本相对冻结基线仍慢约 4%，此次普通计时复现了此前负结果。

## 3. #85 能接入，compact 大 q 不值得整体替换

为对齐计算精度，先比较同为 BF16 KV 存储、BF16 QK/PV 的精确原始 #85 和冻结基线。
在这个 q8192 形状，#85 的实际 dispatch 是其保留的
`_gqa_sparse_fwd_tle_upstream_kernel` / half-KV 路径；不能把耗时归因成它新增的
prefetch kernel。下表的时间是普通 Graph 微基准。

| 输入 | 冻结基线，µs | #85，µs | 基线耗时 / #85 耗时 |
|---|---:|---:|---:|
| compact/recent | 249.66 | 619.55 | 0.403x |
| correlated8 | 1,570.16 | 926.43 | 1.695x |
| independent scattered | 1,419.44 | 981.39 | 1.446x |

在 compact 域，#85 的耗时是 compact-union 的 **2.48 倍**。NCU 提供了更直接的原因：

| 同 BF16 compact 的主要 kernel 指标 | compact-union | #85 |
|---|---:|---:|
| CTA 数 | 512 | 8,192 |
| L2 TEX-source read sectors | 15,074,870 | 141,468,736 |
| Tensor Core 活跃率，elapsed | 30.44% | 11.25% |
| achieved occupancy | 12.48% | 36.37% |

#85 的 L2 read-sector 请求数约为 **9.38 倍**，尽管 occupancy 更高。逐 query 的
调度重复访问多个 query 共用的页，union 的跨 query 复用减少了这部分请求，并形成
更大的 GEMM tile。这是工作组织上的收益，不能仅靠更快的逐 query TMA 搬运补齐。
这个计数不是 HBM 流量，也不是对所有形状的理论放大倍数。

同一张表也反驳了“#85 总是更慢”：BF16 存储时，冻结基线不启用其 FP8 专用 wide arm，
而 #85 在两个散乱场景更快。该控制组不是对当前 FP8 sparse-wide 的同条件直接排名，
不能把它扩展成任一方的普遍优势。

BF16 KV + 散乱 TopK 适用于关闭 FP8 KV 的潜在部署，例如精度或平台能力受限的配置。
本轮使用合成输入，不代表当前部署正在命中该组合；当前主 attention KV 存储是 FP8。
Indexer 的另一份 BF16 KV 缓存不等于这里测量的主 attention KV 缓存。

## 4. 当前 FP8 KV 下，#85 prefill public entrypoint 改变了 QK 数值口径

当前 prefill 主路径是 FP8 KV 存储、BF16 QK/PV。#85 prefill 根据 FP8 KV dtype 选择 FP8 QK，
会额外量化 Q；它没有直接暴露保持该存储同时强制 BF16 QK 的同一 TLE 入口。
这种格式变化需要独立精度验收，不能作为数值不变的实现替换。

相同完整输入对独立 FP32 literal oracle 的结果如下，边界统一为
`abs(error) <= 0.004 + 0.04 * abs(reference)`，每格 8,388,608 个输出元素：

| FP8 存储输入 | 基线超界元素 | #85 超界元素 | #85 max abs error |
|---|---:|---:|---:|
| compact | 0 | 635 | 0.015481 |
| correlated8 | 0 | 632 | 0.012768 |
| independent scattered | 0 | 525 | 0.012350 |

同 BF16 compact 控制中，两者均通过 literal oracle，最大误差均约 0.000931。
FP8 结果仅说明 #85 不满足本轮选定的 BF16-QK 数值要求，不是宣称其违反上游对 FP8
算子的精度契约，也不据此推断模型质量。未通过门禁的三个 FP8 #85 分支没有计性能分，
也在进入 NCU 采集范围前退出；因此这些分支没有性能分或硬件计数器结果。

## 5. Decode 需要独立判断

上面的 BF16 散乱结果仍是 q8192 prefill，不是 decode。现有 decode 已按 batch 分流到
native 或 tuned split-K；本轮未与精确 #85 decode 做同条件对照，不能据此决定切换。

#85 general decode 会先把 FP8 KV 转为 Q 的 dtype 再做 QK，并没有沿用 prefill 的
Q-to-FP8 量化。它的 BF16 专用 fused decode 要求 parallelism>=512，且 GQA 正好
填满至少 16 的 head tile；当前 GQA8 不触发。后续若评估复用，需对齐 KV 存储、
QK 精度、活跃 batch/query length、split-K 和 Graph，并通过数值/replay 门禁。

## 决策和后续方向

保留当前部署主路径。NCU 支持的下一步是收缩 live ranges、减少 local 流量和同步，
避免第三槽的资源门槛；不是继续扩大 shared memory 或只提高 occupancy。
本次仅向 #992 补充研究记录；production、镜像、服务和 #979 均未修改。

#85 可作为特定 dtype/索引域的候选；若要在当前 FP8 KV + BF16 QK 域复用其实现，需要
显式适配计算精度，再做同输入数值、Graph 与性能门禁。不要把 BF16 存储控制组或两次
不同时间的公开 benchmark 当作当前部署的端到端收益。

本轮没有覆盖小 q、decode、全上下文矩阵、冷/轮换 KV、多层或模型端到端；也没有
compute-sanitizer 或 PM-sampling 因果时间线。因此尚不能量化各成本的独立可移除占比。

## 证据与复现

- [派生指标与普通计时汇总](data/metrics.json)
- [独立 oracle 结果](data/oracle-control.json)
- `data/ncu-summary.json.gz`：完整 NCU action 汇总。
- `data/ncu-exports/`：每份报告的完整 Python API 指标和源码采样文本，保留原始导出字节及 SHA256。
- `data/timings/`：原始普通计时及拒绝记录，gzip 解压后字节与采集结果相同。
- 原始 `.ncu-rep` 未纳入仓库；每份报告的 SHA256 已记录在 API 导出和汇总中。
- `measured_sources.tar.gz` 保留本轮 GPU 采集/计时所需的精确源码，包括原始 #85 文件。
  receipt 的 source inventory 还会列出当时存在、但不参与 GPU 执行的分析 helper。
- `MANIFEST.json` 记录发布文件、原始计时、API 导出和执行源码的哈希。

CPU-only 校验和源码准备命令见 [English report](README.md)。
本次发表整理未启动 GPU 任务，原采集任务已经停止并释放 GPU。
