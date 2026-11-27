# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""iHC (independent Hyper-Connections) operators for HY V4.

Unlike mHC, iHC has no Sinkhorn and no cross-stream mixing matrix: each
residual channel has a single scalar sigmoid gate. See ihc_pre / ihc_post.
"""

from flaggems_vllm.runtime.backend._enflame.ops.ihc.ihc_post import (
    ihc_post,
    ihc_post_ref,
)
from flaggems_vllm.runtime.backend._enflame.ops.ihc.ihc_pre import ihc_pre, ihc_pre_ref


def _install():
    """Wire the GCU implementation into the direct-import entrypoint.

    ihc is called via direct module import
    (``from flaggems_vllm.ops.ihc import ihc_pre, ihc_post``) in
    tests/test_ihc.py and benchmark/test_ihc_perf.py, so the
    SpecOpRegistrar namespace swap cannot reach it. Replace the attributes
    on the already-imported package (loaded during ``import flaggems_vllm``),
    mirroring the qwen4/ple_fusion wiring.
    """
    import sys

    mod = sys.modules.get("flaggems_vllm.ops.ihc")
    if mod is None:
        return
    replacements = {
        "ihc_pre": ihc_pre,
        "ihc_post": ihc_post,
    }
    for name, gcu_fn in replacements.items():
        current = getattr(mod, name, None)
        # Only patch the untouched generic binding (the generic submodule's
        # own function); skip if already patched or customized.
        generic_mod = sys.modules.get(f"flaggems_vllm.ops.ihc.{name}")
        if generic_mod is not None and current is getattr(generic_mod, name, None):
            setattr(mod, name, gcu_fn)


try:
    _install()
except Exception:
    # Under pytest assertion-rewrite, locating @triton.jit sources may fail
    # (triggered by dynamically touching generic modules during collection); skipping wiring does not affect op routing
    # (guaranteed by SpecOpRegistrar); direct-import wiring is covered by the runtime retry fallback.
    pass
__all__ = [
    "ihc_post",
    "ihc_post_ref",
    "ihc_pre",
    "ihc_pre_ref",
]
