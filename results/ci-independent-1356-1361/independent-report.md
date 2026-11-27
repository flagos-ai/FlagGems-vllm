# Independent current-main CI PR revalidation

Both candidates preserve the respective public PR head as first parent, can be pushed normally, and remain unpushed by this worker. The current-main base is `12210ab628e0b8f5beed55791958e3bd57accb2b`.

| PR | Candidate | Tree | Complete helper tests | Workflow-policy tests | Result |
|---|---|---|---:|---:|---|
| #1356 | `317282b1b2db9647fc4f4267b0ea8288de1d5984` | `33f390daa4d5634979b9c3b2545b5f97dcae776d` | 90 passed, zero skips | 10 passed, zero skips | All 11 phases passed |
| #1361 | `f9268b77c8e1c3ccfdf86a080f3e656a38083a99` | `170ccb3d94d7b40fc39e327f77bf5657491e7f16` | 94 passed, zero skips | 10 passed, zero skips | All 11 phases passed |

Each was independently archived with canonical LF and every raw Git blob checked (1808 and1809 blobs respectively). The extracted source was hashed before and after execution. Each own namespace used a separate runc container, no GPU/device requests, no privileges/capabilities, network none and read-only root/source. Only own results and a tmpfs were writable. Both naturally exited0, were removed and had exact CID absence positively verified.

The 11 phases were offline lint dependency installation, all CI helpers, workflow-policy validator, policy unit tests, no-torch-fallback validator, init-export validator, export-registration validator, Black24.8, isort5.12, flake8 7.1, and an appropriate fixed negative control. Style checks cover each candidate's actual changed files.

#1356's unchanged-main selector negative control runs the new original tuning-selection regression against the actual main selector and detects both `_nvidia` and nested `_nvidia/hopper` omitted functional-target sets. Its expected two subtest failures are preserved. This is evidence that the patch fixes target selection, not evidence that its selected smoke profile executes the full hardware suite.

#1361's unchanged-main conftest negative control uses a real pytest subprocess with the original asynchronous-fault stub backend: main returns0 after a successful call and skips the following test, so the regression fails as expected. The independent candidate passes that same real subprocess regression, including the first teardown failure, accuracy JSON failed record and skipping the later body. It also preserves the original healthy fixture-restoration regression and current-main#1500 fixture-first teardown ordering. These are CPU hook/recording semantics, not GPU numerical evidence.

Full evidence is in `independent-summary.json` and `independent-raw.tar.gz`. Raw SHA256: `b08d46afbbc0a8e3b971d7f43da8e662a3af15a530238f91dd15fc57d0474d19`. These local results do not replace either PR's exact pushed-head GitHub required CI.
