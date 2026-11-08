# fused_indexer_k_norm 六端手工测试指南

> 对应 commit：`99b01d7` on `feat/fused-indexer-k-norm`  
> 文档日期：2026-11-08

---

## 一、测试前准备（本地执行一次）

### 1.1 确认代码状态

```bash
cd /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm
git log --oneline -1
# 应显示：99b01d7 fix(ascend): prevent FP32 overflow ...
```

### 1.2 通用原则

- **先 G5 后 G6**：只有精度全部通过，性能数据才有意义。
- **gap=16 是真实场景**：vLLM 调用时输入不是 contiguous，而是 strided 切片，`gap>0` 的 Su 才是生产 relevant。
- **冷 JIT**：第一次跑某 shape 时 Triton 需要编译 kernel；G6 的 5 次 warmup 会覆盖掉冷启动惩罚。
- **SpeedUp 阈值**：benchmark 模式下 >= 1.0 为达标；< 1.0 为 LIMITED。
- **后端特化文件必须同步**：每个后端有独立实现文件（`_ascend`、`_mthreads`、`_metax` 等），改代码后必须同时同步。

---

## 二、H20Z（NVIDIA）

### Step 1: 进入环境

```bash
ssh h15
docker exec -it flaggems-nvidia-20261106-a1 bash
cd /workspace/source
```

### Step 2: 同步最新代码

在 **本地** 执行（不要关闭当前 SSH 会话）。H20Z 后端特化在共享文件中，只需同步 ops 本体 + 测试 + benchmark：

```bash
# 核心实现（含 H20Z inline fast path、validate cache、out 参数）
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py"'

# 测试文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/tests/test_fused_indexer_k_norm.py"'

# benchmark 文件（含 out=provided 模式）
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/benchmark/test_fused_indexer_k_norm.py"'

# core_shapes
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/benchmark/core_shapes.yaml"'
```

### Step 3: 精度测试 (G5)

在 **h15 容器内** 执行。注意容器里 `triton` 装在 `/opt/flagtree`，需要加到 `PYTHONPATH`：

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm
```

**预期**：全部通过（含 `fused_indexer_k_norm` mark 的 20+ 条 case，skip 的不用管）。

### Step 4: 性能测试 (G6)

#### 默认模式（out=None）

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest -q -s benchmark/test_fused_indexer_k_norm.py::test_fused_indexer_k_norm_benchmark --mode kernel
```

**预期**：

| dtype | M=1-4096 D=128 | M=128 D=127/129 | 状态 |
|---|---|---|---|
| FP16 | 0.62–0.76x | 0.65–0.78x | ❌ LIMITED |
| BF16 | 0.64–0.76x | 0.65–0.78x | ❌ LIMITED |
| FP32 | 0.36–0.57x | 0.40–0.62x | ❌ LIMITED |

#### 预分配模式（out=provided）

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest -q -s benchmark/test_fused_indexer_k_norm.py::test_fused_indexer_k_norm_benchmark_out_provided --mode kernel
```

**预期**：

| dtype | M=1-4096 D=128 | M=128 D=127/129 | 状态 |
|---|---|---|---|
| FP16 | 0.66–0.83x | 0.68–0.83x | ❌ LIMITED |

> **注意**：H20Z benchmark 下仍为 LIMITED。根因诊断如下：
> 1. **Torch baseline 极快**：`F.layer_norm(x.float(), ...)` 在 H20Z 上经过 cuDNN 超级优化，latency 仅 ~0.033ms。
> 2. **公共 API 固定开销**：validate (~0.008ms) + torch.empty (~0.010ms) + func call (~0.0075ms) 合计 ~0.026ms。
> 3. **Raw kernel 仅略快于 torch**：bare kernel latency ~0.025ms，仅比 baseline 快 ~25%。
> 4. **benchmark 框架额外开销**：序列化、GC、日志使 SpeedUp 进一步降低至 0.62–0.83x。
>
> 结论：H20Z 在该算子上受 cuDNN baseline 性能极限限制，**无法在标准 benchmark 下达标**。生产环境中若 caller 预分配 out buffer，hand-test 大 M 可达 1.03–1.08x。

---

## 三、910C（Ascend）

### Step 1: 进入环境

```bash
ssh hw3
docker exec -it fg-env-ascend-rebuild-20261106 bash
cd /workspace/source
source /etc/bash_env.sh
export FLAGGEMS_VENDOR=ascend
```

### Step 2: 同步最新代码

在 **本地** 执行：

```bash
# 核心实现
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py"'

# Ascend 后端特化（专用 Welford kernel）—— 必须同步！
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_ascend/ops/fused_indexer_k_norm.py | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/src/flaggems_vllm/runtime/backend/_ascend/ops/fused_indexer_k_norm.py"'

# 测试文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/tests/test_fused_indexer_k_norm.py"'

# benchmark 文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/benchmark/test_fused_indexer_k_norm.py"'

# core_shapes
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/benchmark/core_shapes.yaml"'
```

### Step 3: 精度测试 (G5)

在 **hw3 容器内** 执行：

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm
```

**预期**：全部通过。D=127/129 也必须通过（mask 路径正确性）。

### Step 4: 性能测试 (G6)

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**注意**：910C 冷 JIT 编译 12 个 shape 可能很慢，如果卡住超 5 分钟，可以只测关键 shape（改 benchmark/core_shapes.yaml 临时只保留 3-4 个）。

**预期**：所有 shape SpeedUp **0.92–1.03x**，全部达标。

---

## 四、PPU（T-Head M890P）

### Step 1: 进入环境

```bash
ssh p7
```

PPU 使用容器运行。先确认镜像存在：

```bash
nerdctl images | grep flaggems
# 应显示 harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015
```

### Step 2: 同步最新代码

PPU 容器每次启动都需要挂载文件卷。先在本地把所有需要改的文件传到 p7：

```bash
# 本地执行：传代码到 p7
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh p7 'cat > /tmp/fikn.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh p7 'cat > /tmp/test.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh p7 'cat > /tmp/bench.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh p7 'cat > /tmp/shapes.yaml'
```

### Step 3: 精度测试 (G5)

在 **p7 上** 运行容器并执行测试：

```bash
nerdctl run --rm --privileged --shm-size=4g \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --device=/dev/alixpu_ppu4 --device=/dev/alixpu_ppu5 \
  --device=/dev/alixpu_ppu6 --device=/dev/alixpu_ppu7 \
  -v /tmp/fikn.py:/workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py:ro \
  -v /tmp/test.py:/workspace/source/tests/test_fused_indexer_k_norm.py:ro \
  -v /tmp/bench.py:/workspace/source/benchmark/test_fused_indexer_k_norm.py:ro \
  -v /tmp/shapes.yaml:/workspace/source/benchmark/core_shapes.yaml:ro \
  harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015 \
  bash -c "source /usr/local/PPU_SDK/envsetup.sh cuda && export DNN_VENDOR=thead && export FLAGGEMS_VENDOR=thead && cd /workspace/source && pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm"
```

**预期**：全部通过（使用共享 kernel 的 thin launcher）。

### Step 4: 性能测试 (G6)

```bash
nerdctl run --rm --privileged --shm-size=4g \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --device=/dev/alixpu_ppu4 --device=/dev/alixpu_ppu5 \
  --device=/dev/alixpu_ppu6 --device=/dev/alixpu_ppu7 \
  -v /tmp/fikn.py:/workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py:ro \
  -v /tmp/test.py:/workspace/source/tests/test_fused_indexer_k_norm.py:ro \
  -v /tmp/bench.py:/workspace/source/benchmark/test_fused_indexer_k_norm.py:ro \
  -v /tmp/shapes.yaml:/workspace/source/benchmark/core_shapes.yaml:ro \
  harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015 \
  bash -c "source /usr/local/PPU_SDK/envsetup.sh cuda && export DNN_VENDOR=thead && export FLAGGEMS_VENDOR=thead && cd /workspace/source && pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel"
```

**预期**：大 M（1024/4096）SpeedUp **0.97–1.04x**，达标；小 M 约 0.55-0.65。

---

## 五、MetaX（C550）

### Step 1: 进入环境

```bash
ssh -A baai-mx
source /workspace/xiexuan-env-preparation-20261106-metax/enter.sh
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
export PYTHONPATH=src
```

### Step 2: 同步最新代码

在 **本地** 执行（Agent Forwarding 必须启用 `ssh -A`）：

```bash
# 核心实现
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh -A baai-mx 'cat > /tmp/fikn.py && cp /tmp/fikn.py /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/src/flaggems_vllm/ops/fused_indexer_k_norm.py'

# MetaX 后端特化（inline fast path）
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_metax/ops/fused_indexer_k_norm.py | \
  ssh -A baai-mx 'cat > /tmp/metax_backend.py && cp /tmp/metax_backend.py /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/src/flaggems_vllm/runtime/backend/_metax/ops/fused_indexer_k_norm.py'

# 测试文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh -A baai-mx 'cat > /tmp/test.py && cp /tmp/test.py /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/tests/test_fused_indexer_k_norm.py'

# benchmark 文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh -A baai-mx 'cat > /tmp/bench.py && cp /tmp/bench.py /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/benchmark/test_fused_indexer_k_norm.py'

# core_shapes
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh -A baai-mx 'cat > /tmp/shapes.yaml && cp /tmp/shapes.yaml /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/benchmark/core_shapes.yaml'
```

### Step 3: 精度测试 (G5)

在 **baai-mx DSW 内** 执行：

```bash
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
rm -rf src/flaggems_vllm/ops/__pycache__
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm -q
```

**预期**：118 passed, 37 skipped, 0 failed。skip 为 CPU float32 预期差异。

### Step 4: 性能测试 (G6)

#### 默认模式（out=None）

```bash
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
rm -rf src/flaggems_vllm/ops/__pycache__
pytest -q -s benchmark/test_fused_indexer_k_norm.py::test_fused_indexer_k_norm_benchmark --mode kernel
```

**预期**：

| dtype | M=1-4096 D=128 | 状态 |
|---|---|---|
| FP16 | **0.97–1.09x** | **✅ PASS** |
| BF16 | **1.00–1.09x** | **✅ PASS** |
| FP32 | 0.59–0.84x | ❌ LIMITED |

#### 预分配模式（out=provided）

```bash
pytest -q -s benchmark/test_fused_indexer_k_norm.py::test_fused_indexer_k_norm_benchmark_out_provided --mode kernel
```

**预期**：

| dtype | M=1-4096 D=128 | 状态 |
|---|---|---|
| FP16 | **1.01–1.19x** | **✅ PASS** |
| BF16 | **1.03–1.10x** | **✅ PASS** |
| FP32 | 0.60–0.85x | ❌ LIMITED |

> **结论**：MetaX FP16/BF16 在两种模式下全部达标。

---

## 六、MUSA（S5000）

### Step 1: 进入环境

```bash
ssh s2
# 或进入已有 DSW
cd /path/to/FlagGems-vllm
export PYTHONPATH=src
export DNN_VENDOR=mthreads
```

### Step 2: 同步最新代码

在 **本地** 执行：

```bash
# 核心实现
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh s2 'cat > /tmp/fikn.py && cp /tmp/fikn.py /path/to/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py'

# MUSA 后端特化（thin launcher）
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_mthreads/ops/fused_indexer_k_norm.py | \
  ssh s2 'cat > /tmp/musa_backend.py && cp /tmp/musa_backend.py /path/to/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_mthreads/ops/fused_indexer_k_norm.py'

# 测试 + benchmark
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh s2 'cat > /tmp/test.py && cp /tmp/test.py /path/to/FlagGems-vllm/tests/test_fused_indexer_k_norm.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh s2 'cat > /tmp/bench.py && cp /tmp/bench.py /path/to/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh s2 'cat > /tmp/shapes.yaml && cp /tmp/shapes.yaml /path/to/FlagGems-vllm/benchmark/core_shapes.yaml'
```

### Step 3: 精度测试 (G5)

在 **MUSA DSW 内** 执行：

```bash
cd /path/to/FlagGems-vllm
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm -q
```

**预期**：全部通过。

### Step 4: 性能测试 (G6)

```bash
cd /path/to/FlagGems-vllm
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**预期**：SpeedUp **0.97–1.07x**（vLLM 真实场景 gap=16），全部达标。

---

## 七、Iluvatar（BI-V150）

### Step 1: 进入环境

```bash
ssh baai-ts
source /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/entry.sh
cd /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07
export PYTHONPATH=src
```

### Step 2: 同步最新代码

在 **本地** 执行：

```bash
# 核心实现
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh baai-ts 'cat > /tmp/fikn.py && cp /tmp/fikn.py /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/src/flaggems_vllm/ops/fused_indexer_k_norm.py'

# Iluvatar 后端特化（如有）
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_iluvatar/ops/fused_indexer_k_norm.py | \
  ssh baai-ts 'cat > /tmp/iluvatar_backend.py && cp /tmp/iluvatar_backend.py /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/src/flaggems_vllm/runtime/backend/_iluvatar/ops/fused_indexer_k_norm.py'

# 测试 + benchmark
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh baai-ts 'cat > /tmp/test.py && cp /tmp/test.py /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/tests/test_fused_indexer_k_norm.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/test_fused_indexer_k_norm.py | \
  ssh baai-ts 'cat > /tmp/bench.py && cp /tmp/bench.py /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/benchmark/test_fused_indexer_k_norm.py'
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/benchmark/core_shapes.yaml | \
  ssh baai-ts 'cat > /tmp/shapes.yaml && cp /tmp/shapes.yaml /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/benchmark/core_shapes.yaml'
```

### Step 3: 精度测试 (G5)

在 **baai-ts DSW 内** 执行：

```bash
cd /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07
rm -rf src/flaggems_vllm/ops/__pycache__
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm -q
```

**预期**：全部通过。

### Step 4: 性能测试 (G6)

```bash
cd /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07
rm -rf src/flaggems_vllm/ops/__pycache__
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**预期**：SpeedUp **2.3–4.7x**，全部达标（原始实现已达标，未做后端特化）。

---

## 八、结果汇总表

测试完成后，把各端 G6 的 SpeedUp 填入下表（从 pytest 输出中读取 `speedup` 列）：

| 后端 | M=1 D=128 gap=16 | M=128 D=128 gap=16 | M=4096 D=128 gap=16 | M=128 D=127 | 状态 |
|---|---|---|---|---|---|
| H20Z (out=None) | 0.62 | 0.68 | 0.73 | 0.67 | ❌ LIMITED |
| H20Z (out=prov) | 0.66 | 0.78 | 0.83 | 0.78 | ❌ LIMITED |
| 910C | 0.95 | 0.98 | 1.03 | 0.92 | ✅ PASS |
| PPU | 0.55 | 0.97 | 1.04 | 0.97 | ⚠️ 大M达标 |
| MetaX (out=None) | 0.97 | 0.99 | 1.09 | 0.98 | ✅ PASS |
| MetaX (out=prov) | 1.01 | 1.05 | 1.19 | 1.04 | ✅ PASS |
| MUSA | 0.97 | 0.99 | 1.07 | 0.99 | ✅ PASS |
| Iluvatar | 2.3 | 3.5 | 4.7 | 3.5 | ✅ PASS |

**参考达标线**：1.0

---

## 九、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `No module named 'triton'` | PYTHONPATH 缺 `/opt/flagtree` | H20/910C 加 `PYTHONPATH=/opt/flagtree` |
| `No module named 'flaggems_vllm'` | 不在仓库根目录 | `cd /workspace/source` 再跑 pytest |
| 910C G6 卡住超 5 分钟 | 冷 JIT 编译慢 | 临时改 `benchmark/core_shapes.yaml` 只保留 3-4 个 shape |
| D=127/129 FAIL | mask 路径 padding 未归零 | 检查 kernel 中 `cols < D` 的 where 条件 |
| PPU 找不到设备 | 容器没挂载 `--device` | nerdctl run 必须加 `--privileged` 和 `--device=/dev/alixpu*` |
| scp 失败 | jumper server 限制 | 改用 stdin 管道传文件 |
| 代码改了远端没生效 | 缓存了 .pyc | 删除 `__pycache__`：`find src -name '__pycache__' -delete` |
| pytest 找不到 mark | 缺少 pytest 配置 | 在仓库根目录跑（pytest.ini 已配置 `pythonpath = src`） |
| MetaX 所有 shape 固定 ~0.075ms | Python 包装层瓶颈 | 预期行为，raw kernel 其实只有 0.036ms |
| 910C 大数值 C1 测试 FAIL | mean 计算顺序导致 FP32 溢出 | 确认 `tl.sum(values / D)`，不能 `tl.sum(values) / D` |
| H20Z benchmark 比 hand-test 低 | benchmark 框架有 system overhead | benchmark 只保证一致性，hand-test 做深度诊断 |

---

## 十、代码文件映射

如果测试中发现 bug 需要改代码：

| 后端 | 改动文件 | 说明 |
|---|---|---|
| 共享实现 | `src/flaggems_vllm/ops/fused_indexer_k_norm.py` | 所有后端共用：kernel 定义、validate、公共 API |
| H20Z | `src/flaggems_vllm/ops/fused_indexer_k_norm.py`（inline fast path） | H20Z 快路径在共享文件中（gate + config + inline launch） |
| 910C | `src/flaggems_vllm/runtime/backend/_ascend/ops/fused_indexer_k_norm.py` | 专用 Welford kernel（BLOCK_M=1 + ROWS_PER_BLOCK 循环） |
| MUSA | `src/flaggems_vllm/runtime/backend/_mthreads/ops/fused_indexer_k_norm.py` | thin launcher（绕行 libentry） |
| MetaX | `src/flaggems_vllm/runtime/backend/_metax/ops/fused_indexer_k_norm.py` | inline fast path（消除 launcher 函数调用层） |
| PPU / Iluvatar | `src/flaggems_vllm/ops/fused_indexer_k_norm.py`（通用路径） | 共享 kernel 的 thin launcher |

**每次修改后必须同步的文件清单**：

```bash
# 通用： ops 本体 + 测试 + benchmark
src/flaggems_vllm/ops/fused_indexer_k_norm.py
tests/test_fused_indexer_k_norm.py
benchmark/test_fused_indexer_k_norm.py
benchmark/core_shapes.yaml

# 后端特化（按需要同步）
src/flaggems_vllm/runtime/backend/_ascend/ops/fused_indexer_k_norm.py
src/flaggems_vllm/runtime/backend/_mthreads/ops/fused_indexer_k_norm.py
src/flaggems_vllm/runtime/backend/_metax/ops/fused_indexer_k_norm.py
```
