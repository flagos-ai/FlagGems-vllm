---
title: cooperative_topk 昇腾与天数优化记录
weight: 30
---

# cooperative_topk 昇腾与天数优化记录

2026-11-13 整理昇腾与天数的实现、配置、导出与回归测试。PR 基于 `main` 的 `7120b84`，包含两个后端所需的共享 dispatch、Ascend 整数比较及 workspace 适配。两平台均达到六点加权目标。性能测量发生于 2026-11-11，本次保持算子实现和调优条目不变，不构成新的目标设备复测。

接口保持 `cooperative_topk(logits, lengths, output, workspace, k, max_seq_len)`：FP32 logits、INT32 lengths/output，外部 workspace 至少 16 KiB/row；支持三种 k=512/1024/2048、rows≤64。负长度按0，短行输出升序索引并补-1，长行索引无序且相同值的下标选择不固定。精确修正保留 NaN/Inf、次正规数和符号零语义；仅前向接口。

## 统计口径

- `加权耗时 = 原始耗时 × 系数`，`成绩 = H20 native 耗时 / 加权耗时`。表中原始值是三轮各自的中位数；“稳定通过”要求三轮成绩均不低于0.95，不用平均值掩盖失败点。
- 基线是给定的六点 H20 原生 vLLM 数据；系数取 [optest固定版本](https://github.com/flagos-ai/optest/blob/e160a8ae56dcc3594e8b34d6408f04647d5b7e16/data/op_vllm_perf_baseline.json)，仍以H800为系数参考。这是项目比较约定，不是重新标定的H20硬件资源归一化。
- 按同版collector规则对H20 NCU记录取 FMA、Tensor、DRAM 利用率的最大者：只有 `16×16385/k1024` 使用FP32 vector系数，其余五点使用带宽系数。此评分分类不表示真实瓶颈就是带宽饱和。
- 两平台正式计时为完整公共入口、无Graph、清理缓存在start event之前；昇腾验证2/4 GiB对照，天数验证256/512 MiB。两平台采样协议在下文单独说明。所有值均为µs，采用完整行长度、行stride向上对齐4元素的输入。
- [原始三轮数值、权重、来源hash和计算结果](measurements.json)可独立复算；未使用profiler kernel求和替代完整算子计时。

## 昇腾 910B4

额外工作：新增分段CANN sort/merge和FP32整数键修正；修复histogram mask兼容性、raw merge输出容量、独立中间scratch、INT64行地址运算。K2048小批量宽行复用既有精确sorted-topk，把布局复制、length mask和特殊值标记合并；最终正常归并与short/special修正互斥，避免UB生命周期交叠。新配置通过runtime/libtuner接入，固定单候选防止对原位stage重复调优。

正式计时：11月11日，2 GiB缓存清理，10次warmup/150次事件采样，三个独立seed；4 GiB交叉六点也达标，所有离群样本保留。系数0.389412/0.025008。**6/6 shape、18/18次测量通过**。

| rows × width / k | H20 | 原始三轮 | 加权中位数 | 最差成绩 | 稳定通过 |
| --- | ---: | --- | ---: | ---: | --- |
| 1 × 127 / 512 | 5.408 | 1.180 / 1.180 / 1.180 | 0.460 | 11.7692 | 是 |
| 4 × 4097 / 512 | 7.968 | 18.480 / 18.800 / 18.560 | 7.227 | 1.0884 | 是 |
| 16 × 16384 / 1024 | 12.736 | 31.600 / 31.890 / 31.640 | 12.321 | 1.0256 | 是 |
| 16 × 16385 / 1024 | 20.416 | 40.310 / 39.920 / 40.530 | 1.008 | 20.1426 | 是 |
| 4 × 65539 / 2048 | 16.960 | 45.350 / 45.220 / 45.390 | 17.660 | 0.9595 | 是 |
| 1 × 1048576 / 1024 | 47.168 | 120.800 / 119.930 / 119.220 | 46.702 | 1.0027 | 是 |

验证：冻结基础版254项并集为247 passed/7 skipped（超时后按nodeid续跑，0 missing）；最终入口另有40项相关/地址回归和36项API/导出/拒绝验证通过，不混记为同一源码的一次全量运行。最终profile为4个kernel。width>1048577显式NotImplementedError；sanitizer独立运行abort、无有效结论。验证依赖特定FlagTree/CANN构建与[一行运行时补丁](ascend-runtime.md)，不是无条件适用于所有Ascend版本。

## 天数 BI-V150

额外工作：新增TLE共享直方图、原子候选压缩和CAP64精确rank路径；覆盖到宽度131072、rows≤16（宽度≤8192仍遵循共享small调度）。宽行用histogram→threshold→compact→CAP256整数排序四阶段，候选溢出回到精确筛选；实测选取128分区、BLOCK2048等配置，并补齐vendor导出与tuning。

正式计时：11月11日，默认256 MiB缓存、warmup20ms/rep60ms，三个独立seed；512 MiB对照六点差异均小于0.6%。系数0.356378/0.230792。**6/6 shape、18/18次测量通过**。实际是32 GiB/16 SM的单逻辑GPU，物理BI-V150板有双GPU；optest记录63.5 GiB，本次未擅自减半系数。

| rows × width / k | H20 | 原始三轮 | 加权中位数 | 最差成绩 | 稳定通过 |
| --- | ---: | --- | ---: | ---: | --- |
| 1 × 127 / 512 | 5.408 | 3.308 / 3.408 / 3.327 | 1.186 | 4.4531 | 是 |
| 4 × 4097 / 512 | 7.968 | 12.665 / 12.665 / 12.000 | 4.514 | 1.7653 | 是 |
| 16 × 16384 / 1024 | 12.736 | 20.423 / 19.308 / 19.365 | 6.901 | 1.7499 | 是 |
| 16 × 16385 / 1024 | 20.416 | 22.050 / 21.019 / 20.981 | 4.851 | 4.0118 | 是 |
| 4 × 65539 / 2048 | 16.960 | 43.158 / 40.856 / 40.865 | 14.564 | 1.1027 | 是 |
| 1 × 1048576 / 1024 | 47.168 | 122.485 / 115.885 / 115.587 | 41.299 | 1.0806 | 是 |

验证：CPU 249项242 passed/7 skipped，再补4项集中分布；设备253项246 passed/7 skipped，另补跨设备1 passed；共享Ascend-only修正后补了长行回退/导出及六点复核。Core Torch对照9/9通过。六份Torch profiler trace有效，ixsys虽采到GPU指标，但kernel转换失败，未报告其计数器归因或sanitizer通过。

## 共用改动与复现边界

共用实现增加Ascend的masked histogram兼容处理：inactive元素进入独立dummy bin，再用整数shape操作取回原bins；其他后端保持原histogram路径。扩展262147宽行的特殊值、任意位模式、候选容量、混合长度与Graph replay，新增单段集中top-k、Ascend超宽拒绝和Ascend/Iluvatar注册验证。`conf/operators.yaml`同步说明vendor路径。

新增kernel通过runtime.get_tuned_config/libtuner接入。threshold等受workspace协议约束的固定归约明确免调优；Ascend raw sort/merge依赖192 KiB UB布局，使用固定配置，并复用既有helper的固定调度。生产Torch调用仅为内存分配、允许的contiguous复制和无复制view，未引入Torch compute fallback。

性能结论仅覆盖所列硬件、版本、输入分布与计时边界。聚集分布会进入更慢的精确路径；设备事件耗时不能代替端到端Python/vLLM墙钟性能。昇腾和天数达到所列六点加权目标。

2026-11-13 提取后的本地验证使用 RTX 5080、Torch 2.11.0+cu130、Triton 3.6.0，命令均在仓库根目录以 `PYTHONPATH=src .venv/bin/python -m pytest` 执行：

- `-q tests --collect-only`：15966项收集成功；目标测试单独收集254项，benchmark收集2项。
- `-q tests/test_cooperative_topk.py --quick --ref cpu`：**148 passed / 8 skipped**。
- `-q -s benchmark/test_cooperative_topk.py --level core --iter 1 --warmup 1`：**1 passed / 1 skipped**，Torch对照9个shape均执行成功；SM12x不支持的native-vLLM对照跳过。此项仅为本地benchmark smoke。
- `.venv/bin/pre-commit run --all-files`、diff检查及CI目标选择通过；选择器命中目标测试和benchmark。

两平台的36个原始计时值及评分计算已逐项复算；三个生产算子文件均与原提交逐字节一致；vendor 导出和调优条目合并到最新主线配置，测试适配主线已移除的注册配置接口。上述本地验证不替代目标平台真机结果。

功能复现入口为 `pytest -q tests/test_cooperative_topk.py --ref cpu`；仓库benchmark入口为 `benchmark/test_cooperative_topk.py`。Ascend默认do_bench_npu与完整多kernel事件计时不同，本表使用保留原始samples的独立正式runner。具体协议不可混算。

性能记录对应的源码与运行时hash保存在外部精简证据包。提交前仅更正了Ascend两处陈旧注释/docstring，去docstring后的Python AST完全一致；没有改写历史测量的source hash。测试、静态审计及性能数据是不同层级的证据。
