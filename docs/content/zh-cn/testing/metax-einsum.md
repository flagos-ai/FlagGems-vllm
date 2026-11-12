---
title: MetaX einsum 基线环境
weight: 40
---

# MetaX einsum 基线环境

MetaX INT8 einsum benchmark 使用 vLLM-MetaX 厂商提供的 BF16
`deep_gemm.einsum` 入口作为基线。FlagTree 开发镜像可能已有 PyTorch 和
Triton，但默认 Python 路径中没有厂商包。已验证的 Torch 2.8 / Triton 3.6
组合还需要补充 Inductor 使用的旧 `triton_key` 导入路径。

在已有 MetaX 环境中，通过 `tools/run_metax_einsum.sh` 运行测试。脚本为子进程
加载仓库、可选的厂商包路径和兼容层，检查并记录 BF16 基线来源。兼容别名仅针对
Torch 2.8 / Triton 3.6，已有 `triton_key` 保持不变。脚本不安装依赖，不修改
厂商实现。

## 环境要求

- 可运行的 MetaX PyTorch/FlagTree 环境及空闲 GPU。
- 来自兼容 vLLM-MetaX 环境的厂商 `deep_gemm` 包。
  `METAX_DEEP_GEMM_ROOT` 应指向包含 `deep_gemm/` 的父目录；如果包已经安装，
  可不设置此变量。
- 对应环境的 Python 解释器。默认 `python3` 不正确时，设置
  `METAX_EINSUM_PYTHON`。

厂商 BF16 入口本身使用 `torch.compile` 和 `torch.einsum`。复现时保持原实现及
编译开启。上游 NVIDIA DeepGEMM CUDA 内核属于不同后端。

## 运行方法

先检查设备占用，选择空闲的物理卡：

```bash
export METAX_EINSUM_PYTHON=/opt/conda/bin/python
export METAX_DEEP_GEMM_ROOT=/path/to/vendor-packages
export METAX_EINSUM_GPU=0
bash tools/run_metax_einsum.sh -m pytest -q tests/test_int8_einsum.py --quick
```

`METAX_EINSUM_GPU` 会同时设置 `CUDA_VISIBLE_DEVICES`、`MACA_VISIBLE_DEVICES`
和 `METAX_VISIBLE_DEVICES`。未设置时保留已有设备掩码。脚本在仓库根目录执行
传入的 Python 参数。

可通过 `METAX_EINSUM_EXPECTED_SHA256` 固定厂商 `deep_gemm/einsum.py` 的哈希。
不匹配时，脚本在运行用户命令前退出。每次启动均输出来源路径、哈希和实际加载的
Torch/Triton 版本。

```bash
bash tools/run_metax_einsum.sh -m pytest -q -s \
  'benchmark/test_int8_einsum.py::test_perf_int8_einsum[int8]' \
  --shape_file /path/to/accepted-shapes.yaml \
  --level comprehensive --mode kernel --parallel 0 --warmup 30 --iter 100
```

两侧使用相同 shape 集合、精度门槛和计时方法。加速比为 BF16 延迟除以 INT8
延迟，精度比较应包含量化误差。使用轨迹频次时，分别报告加权平均和算术平均。

## 常见问题

- `No module named deep_gemm`：选择正确的厂商环境，或通过
  `METAX_DEEP_GEMM_ROOT` 指定包的父目录。
- `cannot import name triton_key`：在受影响的 Torch 2.8 / Triton 3.6 组合中
  使用此入口，并检查解释器和实际加载的 Triton 路径。
- 基线哈希不匹配：先检查输出的基线来源，再决定是否更新预期哈希。

入口提供不依赖 GPU 的启动回归检查：

```bash
python3 tools/test_metax_einsum_env.py
```

这些检查使用隔离的包 fixture。发布性能结果前，仍需在 C550 上验证真实厂商算子
和数值精度。
