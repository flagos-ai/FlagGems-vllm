# Revalidation evidence, 2026-11-27

These results use current-main `12210ab628e0b8f5beed55791958e3bd57accb2b` with the listed PR changes. They are local controlled revalidation, not a claim that each PR's final GitHub required CI passed or that reviewer approval exists.

| Scope | Source and result |
| --- | --- |
| #1349 logprob | PR head7bbc34f + main, treec9bd8301: 47/47 functional cases, 22/22 native-reference performance cases >=0.9, minimum0.981303435. Original25ms warmup/100ms repetition, fresh cache, one real run. The source audit records only CRLF/LF conversion in the staged archive; no other source difference. |
| #1351 MegaMoE | PR head867ad20 + main, tree63d1b62b: full24-case module,18 passed/6 existing Hygon-only skips; actual NVIDIA two-rank FP8 accuracy passed. Canonical Linux archive matches all1809 raw Git members. Test/toolchain change, no new kernel performance claim. |
| #1356 + #1361 | Current-main candidate1a613878/treea0aefee9 integrates selector coverage and strict async-failure reporting with main's new teardown wrapper. Linux CPU97 helpers/10 policy tests, zero failures/skips; validators and formatting passed. A real pytest subprocess verifies failure+later skip+JSON failed; unchanged main's negative control exposes false green. Full candidate diff included. |

NVIDIA target: H20-3e, Torch2.13.0+cu130, vLLM0.32.0, FlagTree0.7.0/Triton3.6.0, FlagGems5.4.0, FlagCX0.14.1+cuda13.3; MegaMoE uses real Clang22.1.8 and two-way P2P. Source/stack paths, original outputs, timings and cleanup records are retained. Separate original setup failures remain in the local task archive and are not counted as successful numerical runs. Every qualified GPU/container exited naturally and selected devices were released.

Source PRs: https://github.com/flagos-ai/FlagGems-vllm/pull/1349, https://github.com/flagos-ai/FlagGems-vllm/pull/1351, https://github.com/flagos-ai/FlagGems-vllm/pull/1356, https://github.com/flagos-ai/FlagGems-vllm/pull/1361.

Independent final-source #1356 and #1361 Linux CPU validation is in `results/ci-independent-1356-1361`: respectively 100 and104 positive tests, all validators/style checks and unchanged-main negative controls passed; original PR branches were synchronized by normal push. GitHub final-head required CI remains a separate condition.
