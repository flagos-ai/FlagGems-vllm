# 可直接使用的说明

FlagAttention #85 的算子在接口上可以接入当前框架，我们没有将它整体替换到 prefill 主路径，主要依据是工作划分和数值要求。它的主要 prefill 路径按单个 query 遍历 TopK 页，TLE/TMA 用来改善搬运和同步；我们的 compact-union / sparse-wide 则先在多个 query 间共享 KV 页，再进行计算。在 H100、q=8192、seq=16384、GQA=8 的 compact 输入上，同为 BF16 存储和计算的对照中，完整 Graph 调用耗时约为 250 µs 对 620 µs，NCU 测得 #85 的 L2 读 sector 请求数约为 compact-union 的 9.38 倍，说明跨 query 的页复用比继续优化逐 query 搬运更重要。另外，当前 prefill 部署使用 FP8 KV 存储、BF16 QK/PV，而 #85 的 FP8 prefill 入口会量化 Q 并启用 FP8 QK，不能直接视为保持现有数值要求的替换。我们也实际尝试了把 TLE/TMA 多缓冲叠加到 union/sparse-wide 上，但新增同步、local-memory 流量和 occupancy 损失抵消了收益。因此目前保留现有 prefill 主路径，并按具体输入域评估复用 #85；这不意味着它在所有形状上更慢，本轮 BF16 散乱索引对照中它仍有优势。

以上说明中的 250/620 µs 是同条件普通 Graph 微基准，9.38 倍来自 NCU。
BF16 控制组不是当前 FP8 KV 部署的端到端收益，也不是对 sparse-wide 的同存储对照。
详细数据和边界见 [NCU 报告](REPORT.zh-CN.md)。

上述结论不决定 decode 选型。#85 general decode 会先反量化 FP8 KV，再以 Q 的 dtype
做 QK；当前 GQA8 不触发其 BF16 专用 fused 路径。本轮未测同条件 decode A/B，
不能据此宣称替换现有 native/tuned split-K 会有额外收益。
