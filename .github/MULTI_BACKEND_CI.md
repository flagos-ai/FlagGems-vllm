# Multi-backend CI operations

The multi-backend workflow intentionally separates environment validation from
operator capability testing. A backend is not assumed to support every test or
benchmark merely because FlagGems can create its vendor environment.

## Scheduling and trust boundary

- Pull requests select a non-NVIDIA backend either through the exact `label`
  values in FlagGems' pinned `.github/backends.json` or by changing a path
  under `src/flaggems_vllm/runtime/backend/_<vendor>/`. Path inference lets a
  fork test the backend implementation it changes without waiting for a
  maintainer to add a routing label.
- `ci/all-vendors` selects every enabled non-NVIDIA backend. Use it only for a
  deliberate maintainer-approved validation run.
- `ci/benchmark` enables the selected core benchmarks. Benchmarks also run on
  `main` pushes or when `run_benchmarks` is selected in `workflow_dispatch`.
- `workflow_dispatch` can select all non-NVIDIA backends with
  `run_non_nvidia`; it also runs the H20 preflight/baseline so the result has
  a known reference lane.
- During backend bring-up, a `main` push does not automatically select every
  non-NVIDIA backend. Enable that fan-out only after every production runner
  has passed its individual validation.
- Fork pull requests may enter selected non-NVIDIA self-hosted jobs without a
  per-run maintainer gate. H20 remains restricted to same-repository pull
  requests, and Dependabot pull requests remain excluded.
- Treat every file in a fork checkout as arbitrary code, including workflow
  files, local actions, package build hooks, setup scripts, and tests. Every
  non-NVIDIA runner exposed to this public repository must therefore be a
  one-job disposable environment with no secrets, persistent workspace,
  internal-network access, cloud metadata, host Docker socket, or production
  credentials. Deregistering an ephemeral runner is not enough: destroy or
  reimage the worker after the job.
- The currently pinned FlagGems reusable workflow accepts its runner label
  from the caller. Runner-group workflow restrictions and caller-side guards
  do not make a persistent machine safe from a malicious fork. Do not merge
  this mode while any matching vendor runner retains sensitive state.
- Do not use `pull_request_target` to check out and execute fork code.

The repository must provide these labels (spelling and case are significant):

```text
vendor/Ascend
vendor/Enflame
vendor/Hygon
vendor/Iluvatar
vendor/Kunlunxin
vendor/MetaX
vendor/MooreThreads
vendor/SpaceMit
vendor/Sunrise
vendor/Thead
vendor/TsingMicro
ci/all-vendors
ci/benchmark
```

`vendor/Thead` follows the backend registry exactly; do not use
`vendor/THead`.

## Capability rollout

`.github/backend-capabilities.json` is fail closed. Unknown backends receive an
empty operator allowlist and cannot run benchmarks. The setup action still runs
`tools/check_backend_env.py`, which verifies imports, the configured vendor,
device discovery, and a small float32 allocation/addition.

After a backend passes the preflight on a trusted same-repository branch, add
only tests confirmed on that hardware to its `tests_allow` list. Enable and
allowlist benchmarks separately after correctness is stable. The NVIDIA H20
profile is the only initial `allow_all_tests` profile.

`iluvatar` currently allows only `tests/test_mul.py` so PR #50 can provide its
first hardware validation; a failure must be fixed or the candidate removed,
not hidden by widening or bypassing the policy.

The generic preflight is not a substitute for a vendor health query. In
particular, the current SpaceMit descriptor exposes a CPU-compatible device
name. Enflame, SpaceMit, and Sunrise should gain dedicated `gpu_check` scripts
in FlagGems before their hardware availability is considered fully verified.

## Reuse contract

FlagGems remains the source of truth for backend profiles and runner routing:

1. `.github/backends.json` supplies `backend`, `runner_label`, `label`,
   `gpu_check`, and `enabled`.
2. `.github/workflows/backend-test.yaml` owns the shared self-hosted backend
   job and bootstraps the caller repository.
3. FlagGems-vllm's `.github/actions/setup-flaggems/action.yml` asks the pinned
   FlagGems checkout to create the vendor environment, then installs only this
   repository into that environment. The adapter keeps the caller checkout at
   `$GITHUB_WORKSPACE`, the FlagGems checkout at
   `$GITHUB_WORKSPACE/.ci/flaggems`, and the physical vendor environment at
   `$GITHUB_WORKSPACE/.ci/flaggems/.venv`; a root `.venv` symlink exists only
   for compatibility with the reusable workflow.
4. `.github/backend-capabilities.json` records only the FlagGems-vllm tests
   proven on each backend. It is not a second backend registry.

Do not add a hand-maintained vendor matrix or copy vendor setup scripts into
FlagGems-vllm. Add or change a backend in FlagGems first, then advance all
three pinned FlagGems references together.

The current pin includes FlagGems' split of optional native extensions from
the default pure-Python package and its compatible setuptools-scm constraint.
Do not roll the pin back before those changes: even with C++ extensions
disabled, the older scikit-build package path still configured a C++ compiler
on every vendor host.

GitHub documents that a called workflow can use only self-hosted runners made
available in the caller repository's context. Sharing an organization alone
does not grant runner access:
https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations

## Runner installation and access

### 1. Create organization runner groups

As a `flagos-ai` organization owner, open `Settings -> Actions -> Runner
groups -> New runner group`. Prefer one group per vendor or security domain,
for example `flaggems-ascend`, `flaggems-kunlunxin`, and `flaggems-metax`.

For each non-NVIDIA group:

1. Set repository access to `Selected repositories`.
2. Allow `flagos-ai/FlagGems-vllm`; runner eligibility is evaluated in the
   caller repository's context. Also allow `flagos-ai/FlagGems` only if its
   own CI must use the same runner group.
3. Explicitly allow public repositories; GitHub disables public-repository
   access to runner groups by default.
4. Where workflow restrictions are available, choose `Selected workflows`
   and allow the fixed revision:

   ```text
   flagos-ai/FlagGems/.github/workflows/backend-test.yaml@<FLAGGEMS_CI_SHA>
   ```

Only jobs directly defined in that selected reusable workflow can then enter
the group. The H20 job is currently defined directly in `basic-ci.yml`, so its
group cannot use this FlagGems-only restriction until the H20 job is migrated
to the pinned reusable workflow. Prefer an ephemeral H20 runner in the
meantime.

For non-NVIDIA runners, `Selected workflows` reduces accidental access but is
not the fork security boundary in the currently pinned design: a fork can edit
the caller workflow and invoke the allowed reusable workflow with a vendor
runner label. Consequently, every runner matching `ascend`, `iluvatar`, or any
other enabled non-NVIDIA label and visible to FlagGems-vllm must be disposable
and isolated. Do not leave a persistent trusted runner with the same accessible
label.

### 2. Prepare each accelerator host

Use a dedicated non-root runner account. Install the vendor driver, runtime,
SDK, and device query utility, plus `bash`, `git`, `curl`, and `tar`. Ensure:

- Actions Runner is v2.327.1 or newer for the pinned Node 24 actions;
- outbound HTTPS/DNS can reach GitHub, FlagOS resources, vendor package
  indexes, Astral/uv, and the configured Python mirror;
- the runner work directory, home directory, and uv cache are writable, and
  any location holding uv-managed Python is executable;
- no long-lived SSH key, cloud credential, or production secret remains on
  the machine; and
- the runner is isolated from unrelated production networks and state.

Prefer an ephemeral/JIT VM, container, or host image. If the runner is
persistent, retain the workflow cleanup and verify the machine after cancelled
jobs.

To match FlagGems' own CI, the caller adapter preserves a writable inherited
`HOME` and its existing uv cache and managed Python. If `HOME` is unusable
(for example, an unprivileged runner inherited `HOME=/root`), it tries the
account database home and only then uses a job-local fallback under
`$RUNNER_TEMP`. The setup log prints the selected home and effective uv paths
for diagnosis. Do not configure a root-owned home or uv cache for an
unprivileged runner account.

Because fork code can write to this reused home and cache, treat all cached
content as untrusted. Public fork jobs require ephemeral runners or a reliable
machine reset between jobs; changing `HOME` alone is not a security boundary.

### 3. Register the organization runner

Open `flagos-ai -> Settings -> Actions -> Runners -> New runner -> New
self-hosted runner`, select the operating system and architecture, and execute
the exact download and registration commands generated by GitHub. The
registration token is short lived; never copy an example token into a script
or repository.

The final registration command should have this shape:

```shell
./config.sh \
  --url https://github.com/flagos-ai \
  --token '<REGISTRATION_TOKEN_FROM_GITHUB>' \
  --name ascend-runner-01 \
  --runnergroup flaggems-ascend \
  --labels ascend \
  --work _work \
  --unattended
```

Use the exact custom label from the pinned registry:

```text
h20
ascend
enflame
hygon
iluvatar
kunlunxin
metax
mthreads
spacemit
sunrise
thead
tsingmicro
```

The runner label is not the PR label. For example, `mthreads` is selected by
`vendor/MooreThreads`, and `thead` is selected by `vendor/Thead`.

For a persistent Linux runner with systemd, install and verify the service
after registration:

```shell
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

Confirm that GitHub shows the runner as `Online` and `Idle`.

For an ephemeral/JIT deployment, use an external launcher to create a fresh
execution environment and forward runner logs before teardown. Automatic
runner deregistration after one job does not clean a reused physical host, so
the launcher must destroy or reset that environment explicitly; do not install
that one-job runner as the persistent service above.

### 4. Validate the host before CI rollout

Run the vendor device query, then verify the vendor PyTorch and Triton imports.
The trusted CI run must subsequently prove this complete path:

```text
FlagGems-vllm checkout
  -> pinned FlagGems setup
  -> vendor torch/triton/flag_gems imports
  -> vendor and device-count checks
  -> small float32 allocation/add
  -> allowlisted FlagGems-vllm operator tests
```

Enflame, SpaceMit, and Sunrise currently have no dedicated `gpu_check` entry.
Add those checks in FlagGems before treating their generic tensor preflight as
a complete hardware health check.

## FlagGems-vllm repository settings

### Actions policy

Open `Settings -> Actions -> General` and configure:

1. Enable GitHub Actions. Organization or enterprise policy may impose a more
   restrictive setting than the repository page.
2. Require actions to use a full-length commit SHA.
3. Keep the default `GITHUB_TOKEN` read only.
4. Under `Approval for running fork pull request workflows from contributors`,
   choose the least restrictive policy permitted by the organization if fork
   CI should start without a per-run maintainer action.
5. Review changes to workflows, composite actions, checkout refs, `runs-on`,
   setup scripts, and test runners as security-sensitive code before merge.

For least privilege, use the specified-action patterns rather than enabling
all actions created by GitHub, and allow the exact revisions used by the
caller and called workflows, currently including:

```text
actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1
flagos-ai/FlagGems/.github/workflows/backend-test.yaml@<FLAGGEMS_CI_SHA>
```

If administrators instead enable all GitHub-authored actions for convenience,
the repository files must still pin their action SHAs. GitHub's "require full
SHA" setting applies to actions, not reusable workflows; the FlagGems workflow
is protected separately by `uses: ...@<full-sha>`, the pin check, and the
runner-group selected-workflow policy.

For public repositories GitHub does not expose a completely disabled fork
approval policy: even its least restrictive option can still require approval
for a first-time contributor using a new GitHub account. YAML cannot bypass
that platform gate. Established contributors such as the author of PR #50 can
run automatically once the repository/organization policy permits it.

Approval is a review control, not a runtime isolation boundary. Automatic fork
CI makes disposable accelerator workers mandatory.

### Labels and branch rules

Create the labels listed in the scheduling section under `Issues -> Labels`.
Their spelling and case must match the pinned registry exactly.

Under `Settings -> Rules -> Rulesets`, protect `main` with:

- pull requests and approvals required;
- code-owner review required;
- the stable `multi-backend summary` status check required;
- branches required to be up to date if that matches the repository's merge
  policy; and
- bypass permissions limited to the smallest maintainer set.

Confirm that every account named in `.github/CODEOWNERS` has repository write
access; otherwise GitHub cannot request an effective code-owner review.
CODEOWNERS is read from the pull request's base branch, so the new CODEOWNERS
file in the initial CI pull request cannot protect that same pull request.

GitHub only offers a status check for selection as required after it has
completed successfully in the repository recently. Run `multi-backend
summary` successfully first, then add it to the ruleset. A successful summary
means expected accelerator jobs succeeded or were deliberately skipped by the
selection policy. For a non-Dependabot fork that selects a non-NVIDIA backend,
the summary now requires that backend job to succeed; H20 is still expected to
skip.

If the repository enables merge queue, add and validate a `merge_group`
trigger before requiring this check for the queue. The current workflow does
not claim merge-queue support.

## Adding a local action

The existing `.github/actions/setup-flaggems/action.yml` already owns shared
cleanup, FlagGems setup, FlagGems-vllm installation, GPU checks, and the
portable preflight. Extend that action for behavior shared by all backends;
do not create one setup action per vendor.

`tools/prepare-flaggems-ci-env.sh` owns the job-local HOME/uv paths used by the
action. Keep the FlagGems directory, physical venv, caller installation target,
and GPU-check path explicit; a shell step must not depend on the working
directory left behind by a previous composite step.

If another repeated caller-side step needs a composite action, place
`action.yml` under `.github/actions/<name>/`, pass untrusted values through
`env`, and quote them in the shell:

```yaml
name: backend helper
description: Run a backend-neutral helper

inputs:
  backend:
    required: true
    description: FlagGems backend profile

runs:
  using: composite
  steps:
    - shell: bash
      env:
        BACKEND: ${{ inputs.backend }}
      run: |
        set -euo pipefail
        python tools/check_backend_env.py \
          --expected-vendor "${BACKEND%%-*}" \
          --require-flaggems
```

The caller must successfully check out its repository before using a local
action. A local action therefore cannot recover the first failed checkout.
Add every new CI action or helper to `.github/CODEOWNERS`.

## Adding a backend

If the backend already exists in the pinned FlagGems registry:

1. Register a runner with the registry's exact `runner_label`.
2. Grant its runner group to the caller repository `FlagGems-vllm`, and select
   the pinned `FlagGems/backend-test.yaml@SHA` under workflow access. Grant
   repository access to `FlagGems` too only if its own CI uses that group.
3. Create the registry's exact `vendor/*` label in FlagGems-vllm.
4. Run that label from a trusted branch in `flagos-ai/FlagGems-vllm`.
5. After preflight succeeds, add only hardware-proven tests to that backend's
   `tests_allow`; enable benchmarks separately after correctness is stable.

If the backend is not in FlagGems, first submit a FlagGems change that adds its
`backend`, `runner_label`, `label`, `gpu_check`, and `enabled` fields, makes
`./setup.sh <backend>` work, and provides a device health script. After it
lands, advance the registry checkout, reusable workflow, and setup checkout to
the same FlagGems commit and run `python tools/check_ci_pins.py`.

## FlagGems reusable-workflow dependency

The FlagGems revision is intentionally repeated in three syntax locations that
cannot use an expression: the registry checkout, reusable workflow call, and
setup checkout. `python tools/check_ci_pins.py` prevents those pins from
drifting.

The currently pinned FlagGems workflow still declares the unused
`RUNNER_SSH_KEY` input and calls a caller-local checkout retry action after its
bootstrap checkout. A failed bootstrap cannot load that local action. Until a
FlagGems companion change is merged, the caller passes only its short-lived,
read-only `GITHUB_TOKEN`; do not create a long-lived SSH private-key secret.

The FlagGems companion change must:

1. validate canonical backend/runner pairs inside the pinned called workflow,
   detect fork pull requests from the caller context, keep NVIDIA blocked, and
   force non-NVIDIA forks onto an immutable disposable-runner route;
2. inline the checkout attempts in `backend-test.yaml` and leave `ref`
   unspecified (or fix it to `github.sha`);
3. add a 60-minute timeout to the called backend job;
4. make `RUNNER_SSH_KEY` optional, then remove the unused interface after all
   callers stop passing it; and
5. pass `test_script`, `backend`, and `pr_id` through quoted environment
   variables rather than interpolating workflow inputs into shell source.

After that change lands, update all three FlagGems SHAs together, remove
`.github/actions/checkout-retry`, stop passing `RUNNER_SSH_KEY`, and rerun the
pin check.

That companion hardening is not part of this FlagGems-vllm-only change. Until
it is available and pinned, isolation of every matching vendor runner is the
only boundary protecting infrastructure from fork-controlled code.

Before changing the repository files, add the new FlagGems SHA alongside the
old SHA in both the Actions allowlist and every non-NVIDIA runner group's
`Selected workflows` policy. Then update the three code pins and validate a trusted
backend run. Remove the old management-plane allowlist entries only after the
new run succeeds; changing the code pin first can make the workflow ineligible
for its runner group.

## Bring-up sequence

1. Before merging automatic fork CI, remove secrets and sensitive state from
   every non-NVIDIA worker visible to this public repository and make each job
   run in a fresh worker that is destroyed or reimaged afterwards.
2. Confirm the runner groups allow `flagos-ai/FlagGems-vllm`, public-repository
   jobs, and the pinned FlagGems reusable workflow.
3. Configure the least restrictive fork-workflow approval policy permitted by
   the organization. This removes per-run approval for established external
   contributors, subject to GitHub's new-account exception described above.
4. Merge this change into `main`, then synchronize PR #50. Its changed
   `_iluvatar` path automatically selects the `iluvatar` backend; the existing
   `vendor/Iluvatar` label is also a valid route but is no longer required.
5. Confirm that H20 remains skipped while `iluvatar tests and benchmarks`
   starts on the runner carrying the `iluvatar` label.
6. Confirm the preflight succeeds and the target log runs
   `tests/test_mul.py`. The Iluvatar policy deliberately drops all other tests
   and keeps benchmarks disabled.
7. Repeat with one backend at a time. Add tests to each backend allowlist only
   after they pass on that hardware.
8. Use `ci/all-vendors` or `workflow_dispatch(run_non_nvidia=true)` only for an
   intentional full-matrix run, and enable benchmarks only after correctness
   is stable.
9. After FlagGems gains immutable fork routing, advance all three pins and
   tighten runner-group policy before considering any persistent trusted
   vendor pool.

A `main` push intentionally runs the normal NVIDIA selection but does not
reserve every non-NVIDIA runner.

## GitHub references

- Adding organization runners:
  https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners
- Runner group repository and workflow access:
  https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access
- Installing the Linux runner service:
  https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application
- Repository Actions policy and fork approvals:
  https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository
