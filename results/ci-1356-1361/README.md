# CI-only current-main revalidation

Local candidate **1a6138780c359e2ded226e352fe489c35c8975e8**, tree **a0aefee9de4c767461b54f0e24fedb84d7fd9eb0**. Base is main12210ab6, with #1356 head7a733d1c and #1361 head6f9436c6 merged locally. No push or PR merge was performed.

The integration preserves main#1500's fixture-first teardown hook and fallback device backend, then adds #1361's first asynchronous teardown failure and failed accuracy JSON. The old direct-call CPU tests now advance the hook generator. The main helper's poisoned-context test expects the new first-failure contract. The original healthy real-pytest runtime-monkeypatch/fixture-restoration test remains intact.

A new real-pytest subprocess regression verifies that an initially successful call whose stub backend faults at teardown causes exit1, marks its accuracy JSON failed, skips the next case before its body executes, and retains the expected failure reason. This is explicitly a CPU test of real pytest hooks with a stub device backend, not GPU numerical evidence. The unchanged main12210 conftest fails that regression as expected: it exits0 while skipping the later case. Its original failure traceback is retained as a negative control.

## Linux CPU result

On h20_65, cached image bd8a5d7a… under an isolated own namespace and Docker runc:

- **97/97 CI helper tests passed, zero skips**: absolute-import3, backend-policy27, device-context6, general helpers61.
- **10/10 workflow-policy tests passed, zero skips**; repository workflow-policy validator passed.
- No-torch-fallback, init-export and export-registry validators passed.
- Black24.8.0, isort5.12.0 and flake8 7.1.0 checks passed for all four changed Python files.
- The current-main negative fault regression failed exactly as expected; the negative-control wrapper reports this successful detection separately from the107 positive tests.
- Container naturally exited0, then was removed; exact CID absence was positively checked. No GPU devices/device requests, privileged mode or added capabilities; network none, read-only root/source, CPU2/memory4GiB. Only own results and tmpfs were writable.

Every exported Git blob in the canonical LF source archive was compared against its candidate tree; the exact counts and three safe source symlinks are in canonical-source-proof.json. The Windows checkout uses core.autocrlf=true, so archive generation explicitly disables it for Linux. Source identities and individual phase commands/log hashes are recorded in summary.json and the raw archive.

## Preserved initial failures

The initial local artifact used Windows-converted archive line endings and a noexec tmpfs. It produced seven Bash CRLF failures and prevented the Black native extension from loading; the new fault contract already passed. Flake8 also found four pre-existing f-string banner spacing violations in conftest.py, corrected in the final formatting-only commit. All first-run logs remain in ci-cpu-43456792-v1-raw.tar.gz. The final rerun used canonical Git LF bytes and executable CPU-only tmpfs without changing any test, assertion, timeout or skip condition.

## Evidence and limits

- summary.json: final candidate/source identity, phase exits, counts, environment and cleanup.
- ci-cpu-final-v2-raw.tar.gz: complete final phase logs, original negative-control trace, admission and natural cleanup.
- Raw SHA256: **be7951b63d38aeeddec82b2ec8cd1448718ef44121a1eeb1d6bdeb2e6c0a5825**.
- Worktree: D:/Codex/workspaces/FlagGems-vllm-revalidation-ci-20261127, clean after the local commits.

This combined local validation does not replace either PR's required GitHub CI and makes no GPU accuracy, native-reference performance, platform Gate or merge-readiness claim. The image's GPU capability was not used.
