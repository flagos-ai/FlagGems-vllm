# fused_indexer_k_norm 六端手工测试指南

> 对应 commit：`725fbd8` on `feat/fused-indexer-k-norm`  
> 文档日期：2026-11-07

---

## 一、测试前准备（本地执行一次）

### 1.1 确认代码状态

```bash
cd /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm
git log --oneline -1
# 应显示：725fbd8 refactor: move all backend-specific launchers ...
```

### 1.2 通用原则

- **先 G5 后 G6**：只有精度全部通过，性能数据才有意义。
- **gap=16 是真实场景**：vLLM 调用时输入不是 contiguous，而是 strided 切片，`gap>0` 的 Su 才是生产 relevant。
- **冷 JIT**：第一次跑某 shape 时 Triton 需要编译 kernel；G6 的 5 次 warmup 会覆盖掉冷启动惩罚。
- **SpeedUp 阈值**：>= 0.9 为达标；< 0.9 为 LIMITED。

---

## 二、H20Z（NVIDIA）

### Step 1: 进入环境

```bash
ssh h15
docker exec -it flaggems-nvidia-20261106-a1 bash
cd /workspace/source
```

### Step 2: 同步最新代码

在 **本地** 执行（不要关闭当前 SSH 会话）：

```bash
# 源码
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py"'

# 测试文件
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/tests/test_fused_indexer_k_norm.py | \
  ssh h15 'docker exec -i flaggems-nvidia-20261106-a1 bash -c "cat > /workspace/source/tests/test_fused_indexer_k_norm.py"'

# benchmark 文件
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

```bash
cd /workspace/source
export PYTHONPATH=/opt/flagtree
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**预期**：

| dtype | M=1-4096 D=128 | M=128 D=127/129 | 状态 |
|---|---|---|---|
| FP16 | 0.55–0.65x | 0.58–0.60x | LIMITED |
| BF16 | 0.56–0.60x | 0.58–0.60x | LIMITED |
| FP32 | 0.32–0.47x | 0.35–0.47x | LIMITED |

> **注意**：H20Z 实际测试的 SpeedUp **远低于** review 中声称的 2.2–3.4x。根因诊断如下：
> 1. **Torch baseline 极快**：`F.layer_norm(x.float(), ...)` 在 H20Z 上高度优化，4096×128 仅 ~0.034ms。
> 2. **Python 包装层开销大**：`_validate_inputs` (~0.019ms) + `torch.empty` (~0.010ms) 合计 ~0.029ms，已超过 torch baseline 的 80%。
> 3. **Raw kernel 仅略快于 torch**：即使绕过全部 Python 层直接测 raw kernel，latency 也只从 ~0.034ms 降到 ~0.028ms（Su ≈ 1.2x）。
> 4. **review 中的 2.2–3.4x 数据无法复现**：candidate-06 的正式 benchmark JSON 同样显示 0.5–0.7x，与本次手工测试一致。历史 probe 数据可能来自错误 baseline 或测量方法。
>
> 结论：在当前 benchmark 协议（`F.layer_norm` FP32 baseline）下，H20Z 的 public API SpeedUp **无法达到 0.9x**。

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
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh hw3 'docker exec -i fg-env-ascend-rebuild-20261106 bash -c "cat > /workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py"'
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

PPU 容器每次启动都需要挂载源码目录。先传代码到 p7 本地：

```bash
# 本地执行：传代码到 p7
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh p7 'cat > /tmp/fikn.py'
```

### Step 3: 精度测试 (G5)

在 **p7 上** 运行容器并执行测试：

```bash
cat > /tmp/g5.sh << 'EOF'
cd /workspace/source
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm
EOF

nerdctl run --rm --privileged --shm-size=4g \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --device=/dev/alixpu_ppu4 --device=/dev/alixpu_ppu5 \
  --device=/dev/alixpu_ppu6 --device=/dev/alixpu_ppu7 \
  -v /tmp/fikn.py:/workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py:ro \
  -v /tmp/g5.sh:/tmp/g5.sh:ro \
  harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015 \
  bash -c "source /usr/local/PPU_SDK/envsetup.sh cuda && export DNN_VENDOR=thead && export FLAGGEMS_VENDOR=thead && bash /tmp/g5.sh"
```

**预期**：全部通过。

### Step 4: 性能测试 (G6)

```bash
cat > /tmp/g6.sh << 'EOF'
cd /workspace/source
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
EOF

nerdctl run --rm --privileged --shm-size=4g \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu0 --device=/dev/alixpu_ppu1 \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --device=/dev/alixpu_ppu4 --device=/dev/alixpu_ppu5 \
  --device=/dev/alixpu_ppu6 --device=/dev/alixpu_ppu7 \
  -v /tmp/fikn.py:/workspace/source/src/flaggems_vllm/ops/fused_indexer_k_norm.py:ro \
  -v /tmp/g6.sh:/tmp/g6.sh:ro \
  harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015 \
  bash -c "source /usr/local/PPU_SDK/envsetup.sh cuda && export DNN_VENDOR=thead && export FLAGGEMS_VENDOR=thead && bash /tmp/g6.sh"
```

**预期**：大 M（1024/4096）SpeedUp **0.97–1.04x**；小 M 约 0.55-0.65。

---

## 五、MetaX（C550）

### Step 1: 进入环境

```bash
ssh baai-mx
source /workspace/xiexuan-env-preparation-20261106-metax/enter.sh
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
export PYTHONPATH=src
```

### Step 2: 同步最新代码

在 **本地** 执行：

```bash
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh baai-mx 'cat > /tmp/fikn.py && cp /tmp/fikn.py /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07/src/flaggems_vllm/ops/fused_indexer_k_norm.py'
```

### Step 3: 精度测试 (G5)

在 **baai-mx DSW 内** 执行：

```bash
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm
```

**预期**：全部通过。

### Step 4: 性能测试 (G6)

```bash
cd /workspace/xiexuan-env-preparation-20261106-metax/fused-indexer-k-norm-20261106/candidate-07
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**预期**：SpeedUp **0.75–0.85x**，LIMITED。Python 包装层开销是刚性瓶颈（raw kernel 仅 0.036ms，公共 API 固定 ~0.075ms）。

---

## 六、Iluvatar（BI-V150）

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
cat /Users/xiexuan/Projects/BAAI-FlagGems/FlagGems-vllm/src/flaggems_vllm/ops/fused_indexer_k_norm.py | \
  ssh baai-ts 'cat > /tmp/fikn.py && cp /tmp/fikn.py /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07/src/flaggems_vllm/ops/fused_indexer_k_norm.py'
```

### Step 3: 精度测试 (G5)

在 **baai-ts DSW 内** 执行：

```bash
cd /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07
pytest tests/test_fused_indexer_k_norm.py -m fused_indexer_k_norm
```

**预期**：全部通过。

### Step 4: 性能测试 (G6)

```bash
cd /workspace/xiexuan/flaggems-vllm-env-iluvatar-20261106/candidate-07
pytest -q -s benchmark/test_fused_indexer_k_norm.py --mode kernel
```

**预期**：SpeedUp **2.3–4.7x**，全部达标（原始实现已达标，未做后端特化）。

---

## 七、结果汇总表

测试完成后，把各端 G6 的 SpeedUp 填入下表（从 pytest 输出中读取 `speedup` 列）：

| 后端 | M=1 D=128 gap=16 | M=128 D=128 gap=16 | M=4096 D=128 gap=16 | M=128 D=127 | 状态 |
|---|---|---|---|---|---|
| H20Z | | | | | |
| 910C | | | | | |
| PPU | | | | | |
| MetaX | | | | | |
| Iluvatar | | | | | |

**参考达标线**：0.9

---

## 八、常见问题

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

---

## 九、代码文件映射

如果测试中发现 bug 需要改代码：

| 后端 | 改动文件 |
|---|---|
| H20 | `src/flaggems_vllm/ops/fused_indexer_k_norm.py`（`_launch_fused_indexer_k_norm` 中 H20 分支） |
| 910C | `src/flaggems_vllm/runtime/backend/_ascend/ops/fused_indexer_k_norm.py` |
| MUSA | `src/flaggems_vllm/runtime/backend/_mthreads/ops/fused_indexer_k_norm.py` |
| MetaX | `src/flaggems_vllm/runtime/backend/_metax/ops/fused_indexer_k_norm.py` |
| PPU / Iluvatar | `src/flaggems_vllm/ops/fused_indexer_k_norm.py`（共享 kernel） |
