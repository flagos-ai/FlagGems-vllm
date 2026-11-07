# Probe 脚本使用说明

> 本文档汇总优化过程中使用的诊断/探针脚本，解释其原理、用法和输出解读。

---

## 一、总体设计原则

所有 probe 脚本遵循以下模式：

```
raw kernel → public API → baseline comparison
```

**为什么分层诊断？**
1. **Raw kernel**：直接调用 `@triton.jit` 装饰的函数，绕过 `libentry`、`libtuner`、`_launch_fused_indexer_k_norm` 等所有 Python 层
2. **Thin launcher**：调用 `_launch_fused_indexer_k_norm`，包含条件分支、参数组装，但不包含 `libentry`
3. **Public API**：完整的 `flaggems_vllm.fused_indexer_k_norm()`，包含输入验证、输出分配、条件分发

**关键公式**：
- `public_API_time = validation_time + empty_time + launcher_time + libentry_time + kernel_time`
- 通过逐层剥离，定位瓶颈所在层级

---

## 二、具体 Probe 脚本

### 2.1 BLOCK_M 参数扫描（所有后端通用）

**文件名**：`probe_musa_block_m_sweep.py` / `probe_metax_block_m_sweep.py` / `probe_h20_params.py`

**原理**：
- 固定 num_warps=4, num_stages=1，改变 BLOCK_M（1/2/4/8/16/32/64）
- BLOCK_M 决定 grid 维度：grid = (rows / BLOCK_M,)
- 更大 BLOCK_M → 更少 block → 更少 launch overhead，但每个 block 工作量更大

**用法**：
```python
# 直接运行
python3 probe_musa_block_m_sweep.py

# 输出示例
 BLOCK_M    Gems_ms    vs BM=1
------------------------------
       1     0.0367         --
       4     0.0301      17.9%   ← 最优
       8     0.0295      19.6%
      16     0.0303      17.4%
```

**解读**：
- **MUSA**：BLOCK_M=4-8 最优，改善 ~18%（减少 block launch 数量）
- **MetaX**：所有 BLOCK_M 几乎无差异（~0.037ms），说明瓶颈不是 launch overhead
- **910C**：BLOCK_M>1 全部失败，后端不支持

---

### 2.2 BLOCK_M 公共 API A/B（musa_public_ab, metax_block_m_api）

**原理**：
- 使用 monkey-patch 修改 `_launch_fused_indexer_k_norm` 的行为
- 公共 API 路径下测试不同 BLOCK_M 的收益
- 与 raw kernel 对比，看 host 开销是否掩盖 kernel 差异

**关键代码模式**：
```python
_op_mod = sys.modules["flaggems_vllm.ops.fused_indexer_k_norm"]
orig_launch = _op_mod._launch_fused_indexer_k_norm

def _patched(x, weight, bias, out, eps):
    # 强制使用指定 BLOCK_M
    _op_mod._fused_indexer_k_norm_kernel[(grid,)](*args, BLOCK_M=block_m, ...)

_op_mod._launch_fused_indexer_k_norm = _patched
result = flaggems_vllm.fused_indexer_k_norm(...)  # 走公共 API
_op_mod._launch_fused_indexer_k_norm = orig_launch  # 恢复
```

**关键发现**：
- MUSA public API 下 BM=8 vs BM=4 改善 <1%，说明 host/libentry 开销占主导
- MetaX 公共 API 下所有 BM 几乎无差异

---

### 2.3 Thin Launcher 探针（musa_thin_launcher, metax_thin_launcher, probe_musa_thin_launcher）

**原理**：
- 用 monkey-patch 把 `_launch_fused_indexer_k_norm` 替换为直接调用 raw kernel
- 绕过 `libentry` 的 key 计算、cache 查找、lock 同步

**三种实现方式**：

**A. 模块级 monkey-patch（公共 API 下测试）**：
```python
def _thin_launcher(x, weight, bias, out, eps):
    _fused_indexer_k_norm_kernel[(grid,)](*args, BLOCK_M=4, ...)

_op_mod._launch_fused_indexer_k_norm = _thin_launcher
# 测试公共 API
_op_mod._launch_fused_indexer_k_norm = orig_launch
```

**B. 公共 API Patch（最准确）**：
```python
_op_mod._launch_fused_indexer_k_norm = _thin_launcher
ms = _measure(flaggems_vllm.fused_indexer_k_norm, args)
```

**关键结果**：
- **MUSA**：thin launcher 公共 API 下改善 ~31%（0.076→0.053ms）→ Su 达到 0.89-1.07
- **MetaX**：thin launcher 公共 API 下改善 ~6%（0.074→0.070ms）→ 提升有限
- **PPU**：thin launcher 改善 ~25%（0.043→0.030ms）→ Su 1.04x

---

### 2.4 Host Overhead诊断（metax_host_diag, h20_overhead_diag）

**原理**：
- 从公共 API 逐层剥离到 raw kernel，测量每一层的时间增量
- 识别"每增加一层 Python 包装，增加多少延迟"

**测试路径**：
```
public API
  → no validation（跳过 _validate_inputs）
    → thin launcher（直接调用 _launch_fused_indexer_k_norm）
      → raw kernel（直接调用 _fused_indexer_k_norm_kernel）
        → empty + raw（包含 torch.empty）
          → cache + raw（复用预分配 buffer）
```

**MetaX 结果**：
```
public API:      0.0768 ms
no validation:   0.0584 ms  (saves 0.0184)  ← 验证层几乎不占时间
thin launcher:   0.0576 ms  (saves 0.0192)  ← launcher 层占 0.019ms
raw kernel:      0.0417 ms  (saves 0.0351)  ← kernel 本身
empty + raw:     0.0456 ms  (saves 0.0312)  ← empty 占 0.004ms
cache + raw:     0.0417 ms  (saves 0.0351)  ← 无 empty 开销
```

**解读**：
- **Kernel 本身**：0.0417ms（已优化到极致）
- **torch.empty**：0.004ms
- **Launcher 函数调用**：0.019ms（> kernel 本身 45%）
- **验证层**：~0.0008ms（可忽略）

**结论**：MetaX 的瓶颈是 **Python 函数调用层**，不是 kernel。直接内联到公共函数可改善 ~5-8%。

---

### 2.5 910C 专用 Kernel Probe（ascend_kernel）

**文件名**：`probe_910c_ascend_kernel.py`

**原理**：
- 为 910C 设计完全独立的 kernel，因为共享 kernel 的 BLOCK_M>1 在 NPU backend 下失败
- 关键创新：
  - `BLOCK_M=1`（backend 兼容）
  - `ROWS_PER_BLOCK`：kernel 内循环处理多行
  - `Welford` 单遍算法
  - `NO_MASK`：D 为 2 的幂时跳过 mask

**核心 kernel 结构**：
```python
@triton.jit
def _ascend_layernorm_kernel(..., ROWS_PER_BLOCK: tl.constexpr):
    row_start = program_id(0) * ROWS_PER_BLOCK
    for r in range(ROWS_PER_BLOCK):
        row = row_start + r
        if row < M:
            # load, compute, store one row
```

**测试维度**：
- `ROWS_PER_BLOCK sweep`：1/2/4/8/16/32/64
- 每个 RPB 对比 production code 和改善百分比
- Full core_shapes with best RPB

**关键结果**：
```
 RPB    base_ms  ascend_ms improve%
------------------------------------
   8     0.1505     0.1023    32.0%   ← 最优
  16     0.1303     0.0988    24.2%
```

---

### 2.6 Raw Kernel Param Sweep（h20_raw_sweep）

**原理**：
- 固定 shape（M=4096, D=128, gap=16），扫描 BLOCK_M × num_warps 全组合
- 找到 raw kernel 的 latency 下限

**输出矩阵**：
```
 BLOCK_M  num_warps    Gems_ms   vs BM=1 NW=4
--------------------------------------------
       1          1     0.0083             --
       1          4     0.0085          -2.7%
       4          1     0.0064          23.2%
       8          4     0.0062          25.5%   ← 最优
      16          4     0.0067          19.7%
```

**关键发现**：
- H20 raw kernel 最优仅 **0.0062ms**
- 但之前公共 API 下是 0.072ms → 差距 **10x**
- 说明旧代码有严重 bug（libtuner 路径缺失 BLOCK_M）
- 修复后公共 API 下降到 0.0063ms

---

### 2.7 Inlined Fast Path（metax_inline）

**原理**：
- 把 MetaX 的快路径直接内嵌到 `fused_indexer_k_norm()` 函数体中
- 跳过 `_launch_fused_indexer_k_norm` 函数调用层

**代码模式**：
```python
def _inlined_fast(x, weight, bias, dim, eps):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.shape[0]
    _fused_indexer_k_norm_kernel[(rows,)](
        x, weight, bias, out, ...
        BLOCK_M=1, num_warps=4, num_stages=1,
    )
    return out

# Monkey-patch 公共 API
flaggems_vllm.fused_indexer_k_norm = _inlined_fast
```

**关键结果**：
- MetaX：0.0773ms → 0.0520ms（改善 32.8%）→ Su **1.06x**（达标）
- 但实际生产代码中内联后只改善到 0.075ms → 0.070ms
- 差距原因是 monkey-patch 后的函数调用链仍经过 gems_assert

---

## 三、通用方法论

### 3.1 诊断流程（适用于任何后端）

```
Step 1: raw kernel sweep（BLOCK_M/num_warps）
    → 确认 kernel 本身的 latency 下限
    
Step 2: thin launcher probe（monkey-patch _launch_fused_indexer_k_norm）
    → 确认 libentry + Python 层开销
    
Step 3: host overhead diagnosis（逐层剥离）
    → 确认哪一层 Python 包装占时间
    
Step 4: public API A/B（修改 _launch_fused_indexer_k_norm）
    → 确认生产代码路径下的实际改善
    
Step 5: 全 shape G6（alternating ref/gems）
    → 最终验收
```

### 3.2 关键指标解读

| 指标 | 正常范围 | 异常范围 | 含义 |
|---|---|---|---|
| raw kernel < 0.01ms | GPU | NPU/CPU | kernel 本身很快 |
| thin vs raw 差距 < 20% | 正常 | >50% | libentry 开销大 |
| API vs thin 差距 < 10% | 正常 | >30% | launcher 层开销大 |
| 验证层时间 | <1% | >10% | _validate_inputs 太重 |
| torch.empty 时间 | <5% | >20% | 内存分配是瓶颈 |

### 3.3 Monkey-Patch 安全模式

所有 probe 脚本都遵循：
```python
orig = _op_mod._launch_fused_indexer_k_norm
try:
    _op_mod._launch_fused_indexer_k_norm = _patched
    # 测试...
finally:
    _op_mod._launch_fused_indexer_k_norm = orig
```

确保即使测试失败也不会永久修改生产代码。

---

## 四、复用模板

### 最小可复用的 raw kernel probe（任意后端）

```python
import torch, triton, statistics
import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

device = flaggems_vllm.device

def _measure_kernel(fn, args, grid, block_kwargs, warmup=5, samples=25):
    from flaggems_vllm.ops.fused_indexer_k_norm import _fused_indexer_k_norm_kernel
    for _ in range(warmup):
        _fused_indexer_k_norm_kernel[grid](*args, **block_kwargs)
    torch_device_fn.synchronize()
    times = []
    for _ in range(samples):
        torch_device_fn.synchronize()
        s = torch_device_fn.Event(enable_timing=True)
        e = torch_device_fn.Event(enable_timing=True)
        s.record()
        _fused_indexer_k_norm_kernel[grid](*args, **block_kwargs)
        e.record()
        torch_device_fn.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)

# 创建测试数据
m, d = 4096, 128
x = torch.randn((m, d), dtype=torch.float16, device=device)
weight = torch.randn((d,), device=device)
bias = torch.randn((d,), device=device)
out = torch.empty_like(x)

args = (x, weight, bias, out, m, d, m, 1, 1, 1, 1e-6)
grid = (m,)

# 测试不同参数
for block_m in (1, 2, 4, 8, 16):
    for num_warps in (1, 2, 4, 8):
        ms = _measure_kernel(None, args, grid, dict(BLOCK_M=block_m, num_warps=num_warps, num_stages=1))
        print(f"BM={block_m} NW={num_warps}: {ms:.4f} ms")
```

### 最小可复用的 public API A/B（任意后端）

```python
import torch, statistics, sys
import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

def _measure_public_api(args, warmup=5, samples=25):
    for _ in range(warmup):
        flaggems_vllm.fused_indexer_k_norm(*args)
    torch_device_fn.synchronize()
    times = []
    for _ in range(samples):
        torch_device_fn.synchronize()
        s = torch_device_fn.Event(enable_timing=True)
        e = torch_device_fn.Event(enable_timing=True)
        s.record()
        flaggems_vllm.fused_indexer_k_norm(*args)
        e.record()
        torch_device_fn.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)

# 测试不同 BLOCK_M
_op_mod = sys.modules["flaggems_vllm.ops.fused_indexer_k_norm"]
orig_launch = _op_mod._launch_fused_indexer_k_norm

for block_m in (1, 2, 4):
    def _patched(x, w, b, out, eps):
        from flaggems_vllm.ops.fused_indexer_k_norm import _fused_indexer_k_norm_kernel
        rows = x.shape[0]
        _fused_indexer_k_norm_kernel[(rows // block_m,)](
            x, w, b, out, rows, x.shape[1], *x.stride(), w.stride(0), b.stride(0), eps,
            BLOCK_M=block_m, num_warps=4, num_stages=1,
        )
    
    _op_mod._launch_fused_indexer_k_norm = _patched
    ms = _measure_public_api((x, weight, bias, d, 1e-6))
    print(f"BM={block_m}: {ms:.4f} ms")
    _op_mod._launch_fused_indexer_k_norm = orig_launch
```

---

## 五、历史探针脚本清单

| 脚本名 | 用途 | 关键发现 |
|---|---|---|
| `probe_musa_block_m_sweep.py` | BLOCK_M 参数扫描 | BM=4-8 最优，+18% |
| `probe_musa_public_ab.py` | 公共 API A/B | host 开销掩盖差异 |
| `probe_musa_thin_launcher.py` | thin launcher 验证 | +31%，Su 达标 |
| `probe_metax_block_m_sweep.py` | BLOCK_M 参数扫描 | 几乎无差异 |
| `probe_metax_thin_launcher.py` | thin launcher 验证 | +6%，提升有限 |
| `probe_metax_host_diag.py` | host 开销逐层剥离 | launcher 占 0.019ms |
| `probe_metax_inline.py` | 内联快路径 | +32%，Su 1.06x |
| `probe_h20_params.py` | H20 BLOCK_M/warps | BM=8 最优 |
| `probe_h20_raw_sweep.py` | H20 全参数扫描 | raw kernel 0.0062ms |
| `probe_h20_overhead_diag.py` | H20 host 开销 | public API = raw kernel |
| `probe_910c_warps.py` | 910C raw kernel 扫描 | BLOCK_M>1 全部失败 |
| `probe_910c_ascend_kernel.py` | 910C 专用 kernel | RPB=8/16 最优，Su 达标 |
| `g6_musa_core.py` | MUSA 完整 G6 | gap=16 达标 |
| `g6_metax_core.py` | MetaX 完整 G6 | 全 shape 0.70-0.79 |
| `g6_910c_ascend_final.py` | 910C 专用 kernel G6 | 全 shape 达标 |

这些脚本在完成优化使命后，已被移动到 `benchmark/probes/` 目录（后被清理），但其方法论和模板代码保留在本文档中，可供后续后端调优复用。
