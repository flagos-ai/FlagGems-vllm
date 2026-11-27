# #1356 shared PR range completion

Final local candidate `a497988d866ee8527d434a315c2cf1bbb18654a3`, tree `d25451e07ec424fb6c8a0d7c9934e29a53d3a397`, directly extends published `541a13c849a5ff90fe2345a61089b9b807769508` and permits a normal fast-forward push. It is not yet pushed by this worker. The incremental change is four files,127 additions/31 deletions: both existing workflows, one 20-line Bash helper and expanded real-Git regressions.

The same reviewed merge identity/parent validation now lives in tools/resolve-ci-pr-range.sh. Both rule-check and unit-test preprocess invoke it after their actual checkout. It requires HEAD==GITHUB_SHA, exactly two parents and second-parent==event PR head, then emits the immutable first-parent target plus head and merge-base. No moving-main fetch or stale event-base fallback is used. Existing rule checks use these same outputs; unit preprocessing now writes target...head filenames to its original NUL-delimited changed_files.bin. Push/first-push behavior, downstream selectors, prohibition rules, job counts, optional observation/gate requirements and all test skip conditions are unchanged.

## Complete Linux CPU result

One fresh formal run used immutable cached c7b68ce18668b8f796b112a9632f32898f4aed802f078288077bbfe32309fd0e with Git 2.43.0 and Python 3.12.3. runc, no GPU/device requests, network none, read-only root/source and capdropALL isolation remained unchanged.

- **98/98 helper tests and 10/10 workflow-policy tests passed, zero skips.** All 8 real-Git/workflow-shell regression methods actually executed: the original 5 rule-check regressions and 3 new unit-preprocess regressions.
- **All 14 phases passed**, including the shared helper's bash -n, policy/export validators, Black 24.8/isort 5.12/flake8 7.1 and three fixed negative controls.
- Unit tests execute the actual workflow collect step with a real Git graph containing stale event base, common main updates and later generic/vendor-only main changes. Only the PR path reaches the NUL list, including a filename with spaces and a literal newline. All 4 wrong merge identity/parent structures fail closed through the actual unit range step; all 3 normal/zero/empty-before push variants retain their prior path results.
- Both workflows execute the exact production Bash helper copied into the fixture checkout. Git is not stubbed. The original production torch/derive checks still reject PR violations and violations in current merged contents; the formatter shim only captures its actual CLI arguments.
- Fixed negative controls retain causal evidence: unchanged main 12210 selector misses two tuning target sets; old 317282 rule workflow wrongly checks common legacy and merge-only files; old 541a unit workflow wrongly includes common, main-only and MetaX vendor paths. All expected failures occurred with no skips/errors.
- All 1810 raw Git blobs matched canonical LF. Source hashes before/after passed. Final own CID f8bd4495224115fd9a42cb07b0e01d6dd565be6180ef7fc6de098233f79c2968 naturally exited0, was removed and exact CID absence was positively verified. No shared host environment or GPU state changed.

Evidence: summary.json, final.diff and raw.tar.gz. Raw SHA256 `2fae84b1bf4fa838d94a8619c0287bb8234b3e501768c60e6e7714d889ed543c`. The previous rule-only run and its missing-Git first attempt remain preserved separately in ../range-fix. This increment was tested once on the already suitable Git-enabled CPU image; no source/test/rule changes were needed after the reviewed diff.

## CI status boundaries

The vendor-container-tests job is an optional observation lane. Its Volcano queue failure alone does not block unittest-required. Required H20 and rule-check results must be reported separately. This patch changes no infrastructure, observation-job requirements or skip/allow policy. The independent main-tree A5 export-order validator fix is tracked by https://github.com/flagos-ai/FlagGems-vllm/pull/1504 and is not a backend runtime dependency. Parent owns publication and exact-head live CI status.
