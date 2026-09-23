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


def bmm_heur_divisible_m(args):
    """Check if M dimension is divisible by TILE_M"""
    return args["M"] % args["TILE_M"] == 0


def bmm_heur_divisible_n(args):
    """Check if N dimension is divisible by TILE_N"""
    return args["N"] % args["TILE_N"] == 0


def bmm_heur_divisible_k(args):
    """Check if K dimension is divisible by TILE_K"""
    return args["K"] % args["TILE_K"] == 0


HEURISTICS_CONFIGS = {
    "bmm": {
        "DIVISIBLE_M": bmm_heur_divisible_m,
        "DIVISIBLE_N": bmm_heur_divisible_n,
        "DIVISIBLE_K": bmm_heur_divisible_k,
    }
}
