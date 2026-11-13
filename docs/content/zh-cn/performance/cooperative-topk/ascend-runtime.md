<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# cooperative_topk 的 Ascend 运行时复现

2026-11-11 的 Ascend910B4-1 验证使用下列版本及一个 custom-op 补丁。
本页只重建 `custom_ops.bc`，不代替 Torch、CANN 或 FlagTree 的安装流程。

| 组件 | 实测版本 |
| --- | --- |
| 硬件 | Ascend910B4-1，20 Cube / 40 Vector cores |
| Driver / CANN | 25.2.0 / 9.1.0 |
| Torch / Torch-NPU | 2.12.0+cpu / 2.12.0 |
| FlagTree / Triton | 0.7.0+ascend.gitca04d7d6 / 3.5.1 |
| FlagTree 源码 | `ca04d7d6000994816ab1bdc8b447a1656e16a50b` |
| Template 源码 | `a205c9574907907d608da6029403415ca2f98d3c` |

FlagTree 的输入目录为
`python/triton/experimental/tle/language/dsa/ascend/custom_ops`。
已逐文件核验：实际 B 构建使用的 16 个 C++ 源文件及头文件与上述
[FlagTree commit](https://github.com/flagos-ai/FlagTree/commit/ca04d7d6000994816ab1bdc8b447a1656e16a50b)
一致，唯一源码改动是随本页提供的补丁。
Template 取自
[FlagTree-AscendNPU-IR commit](https://github.com/flagos-ai/FlagTree-AscendNPU-IR/commit/a205c9574907907d608da6029403415ca2f98d3c)
的 `bishengir/lib/Template`；实测副本的 138 个文件与该干净 checkout 一致。

## 补丁和验证边界

[ascend-merge-pointers.patch](ascend-merge-pointers.patch) 只修改
`sort_ops/merge_pack_sort.cpp`：先将 `xn[4]` 的所有元素初始化为合法 UB 地址，
再填入 active lanes 的实际地址。它不改变排序算法或精度，也不包含后来弃用的
空 padding 同步优化。

原 bundled bitcode、未修改源码的重建 A、打补丁的 B，在最小 2/3/4-way
合并测试中均各通过 30 轮。因此，这个补丁是防御性修补，不能将之前所有 VEC
异常归因于它，或将算子性能收益归因于它。
正式性能和最终相关功能验证使用 B；未将其他运行时版本自动视作同等验证通过。

## 在独立构建目录中重建 B

在已加载 **CANN 9.1.0 开发环境**的 aarch64 开发容器中，从本仓库根目录执行。
需要 Git、Python 3、`patch`、CANN 的 `ccec` 及同目录的 `llvm-link`。
下面的步骤只修改新建的工作目录，不覆盖已安装或共享的运行时。

```bash
set -euo pipefail
coop_repo_root="$(pwd -P)"
coop_work="$(mktemp -d "${TMPDIR:-/tmp}/cooperative-topk-ascend.XXXXXX")"
coop_flagtree_rev=ca04d7d6000994816ab1bdc8b447a1656e16a50b
coop_template_rev=a205c9574907907d608da6029403415ca2f98d3c

# sparse checkout 只取重建 custom_ops 所需目录。
git clone --filter=blob:none --no-checkout \
  https://github.com/flagos-ai/FlagTree.git "$coop_work/FlagTree"
git -C "$coop_work/FlagTree" sparse-checkout set \
  python/triton/experimental/tle/language/dsa/ascend/custom_ops
git -C "$coop_work/FlagTree" checkout --detach "$coop_flagtree_rev"

git clone --filter=blob:none --no-checkout \
  https://github.com/flagos-ai/FlagTree-AscendNPU-IR.git "$coop_work/AscendNPU-IR"
git -C "$coop_work/AscendNPU-IR" sparse-checkout set bishengir/lib/Template
git -C "$coop_work/AscendNPU-IR" checkout --detach "$coop_template_rev"

coop_source="$coop_work/FlagTree/python/triton/experimental/tle/language/dsa/ascend/custom_ops"
coop_template="$coop_work/AscendNPU-IR/bishengir/lib/Template"
coop_build="$coop_work/custom_ops"
coop_patch="$coop_repo_root/docs/content/zh-cn/performance/cooperative-topk/ascend-merge-pointers.patch"
mkdir "$coop_build"
cp -a "$coop_source/." "$coop_build/"

# 拒绝在未知源文件上直接套用补丁。
python3 - "$coop_build/sort_ops/merge_pack_sort.cpp" <<'PY'
import hashlib
import sys
from pathlib import Path

actual = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
expected = "330e288cc0c43360447522dd4f5f67e903382afcdd6050ac4ac67d94a4a0fa84"
if actual != expected:
    raise SystemExit(f"Unexpected merge source SHA256: {actual}")
PY
patch --dry-run -p1 -d "$coop_build" < "$coop_patch"
patch -p1 -d "$coop_build" < "$coop_patch"

# 上游脚本会自行赋值 TEMPLATE_ROOT，仅在外部设置同名变量不会生效。
# 保留上游所有编译/链接参数，只将这个路径改为显式环境输入。
sed 's|^TEMPLATE_ROOT=.*$|TEMPLATE_ROOT="${COOP_TEMPLATE_ROOT:?Set COOP_TEMPLATE_ROOT}"|' \
  "$coop_source/build_custom_ops.sh" > "$coop_build/build_custom_ops.sh"

command -v ccec
coop_ccec="$(readlink -f "$(command -v ccec)")"
test -x "$(dirname "$coop_ccec")/llvm-link"
COOP_TEMPLATE_ROOT="$coop_template" bash "$coop_build/build_custom_ops.sh"
sha256sum "$coop_build/custom_ops.bc"
```

脚本会编译各 custom-op 的 Vector/Cube 目标和 mix 变体，并链接 Template 的
Arange、Elementwise、Sort 与 UB Copy 实现。必须保留完整输入目录；仅复制
`merge_pack_sort.cpp` 和构建脚本不足以重建。

输出位于 `$coop_build/custom_ops.bc`。部署到自己的开发环境时，应先备份该环境
原有的 bitcode，再放入
`triton/experimental/tle/language/dsa/ascend/custom_ops/custom_ops.bc`；
该目录可通过 `triton.experimental.tle.language.dsa.ascend.custom_ops.__file__`
定位。安装后使用新的 `TRITON_CACHE_DIR`，避免把旧编译结果混入验证。
在此次运行时中，独立测试脚本在首次 JIT 前显式执行 `import triton.knobs`。

## 校验值与重新验证

| 实测产物 | SHA256 |
| --- | --- |
| B 的 `merge_pack_sort.cpp` | `fca7d26e2f90391417a994139f7fde45aeb7de69c5faf56f376cdb86b1f4247f` |
| B 的 `custom_ops.bc` | `69ef9df02127671d3ee29b9045d16c9e3f18772ecb82d58aec5a756482174e44` |
| 实测 `libtriton.so` | `90aa426e038560622cc63f26ee73e70b9f3b4ddba957d0a7fc28cc82cdaf5382` |
| 实测 Ascend `cooperative_topk.py` | `c761c66536acf37b03a32db76ce31da8d9f229e42f380207c143fe3789945e1c` |
| 复用的 Ascend `topk.py` | `7fc56629255969d07e6df1a3ed729a56b0242595546b8cfe611d3d462bc8e5f5` |

上述 bitcode hash 标识**实际验收过的二进制**。构建路径、编译器构建版本和
bitcode 元数据可能改变重建文件的字节表示，不能承诺任意机器上逐字节相同。
应记录新构建的 hash，并重新运行相关正确性、graph replay 和性能验证；只看到
“Triton 3.5.1”或“CANN 9.1”版本字符串不足以沿用历史性能结论。

本次提交对算子源码另有注释/docstring 修正，去掉 docstring 后的 Python AST
保持一致；历史性能文件保留上述原实测 hash，不将其重标为整理后的源码 hash。

在具备 GPU 环境时，从 FlagGems-vllm 根目录运行完整契约测试：

```bash
DNN_VENDOR=ascend python -c \
  'import triton.knobs, pytest; raise SystemExit(pytest.main(["-q", "tests/test_cooperative_topk.py", "--ref", "cpu"]))'
```

特别覆盖混合及负长度、NaN/Inf/subnormal、任意 FP32 bit patterns、集中 top-k、
output/workspace 保护和 graph replay。此次 Ascend 算子对非空输入限制
`width <= 1048577`，更宽输入明确报错。
性能复验须沿用完整公共入口的非 graph 设备事件协议、2 GiB 计时前缓存清理、
三次独立进程 seed42/43/44，并保留较大缓存清理量的交叉测量；不要用各 kernel
平均时间或 profiler 求和替代整条算子耗时。

启用 `TRITON_ENABLE_SANITIZER=1` 的历史尝试在 driver launch 处 abort，
没有有效 sanitizer 报告，也没有最小 fill 对照；因此没有 sanitizer 通过结论，
亦未确定 abort 根因。
