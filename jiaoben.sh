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

cd /share/project/tle/pr/flash-linear-attention

SRC=perf/gdn2-fused-infer
BASE=$(git merge-base origin/main "$SRC")

GEMS=/share/project/tle/pr/FlagGems-vllm
DST=$GEMS/src/flaggems_vllm/ops/FLA/gdn2_native

mkdir -p "$DST"
touch "$DST/__init__.py"

for f in chunk_fwd.py chunk_intra.py chunk_intra_token_parallel.py wy_fast.py; do
    git show "$BASE:fla/ops/gdn2/$f" > "$DST/$f"
done
