# Final shared PR-range validation for #1356

Published head a497988d/tree d25451e0 uses one checked-merge range resolver in the existing rule-check and unit-preprocess jobs. The actual target is the checked-out merge's first parent, not a stale PR event base or a newly fetched moving main. Rule checks retain current merged file contents and original compute prohibitions; unit selection retains NUL paths. Existing main-push behavior, standard CI --quick execution and optional-vendor gate policy are unchanged.

NVIDIA tuning-table edits select the complete functional file set; benchmark edits select matching functional tests, including Marlin aliases. This is not merely an environment-smoke-only selection.

Linux CPU:98 helper tests plus10 workflow-policy tests passed, zero skips/failures. All8 real-Git/workflow-shell methods ran;14 phases including Bash syntax, validators, style and three unchanged-code negative controls passed. Actual PR and merge-content compute violations remain rejected. The new source1810 raw blobs matched before/after, no GPU/network/privilege was used and the container exited naturally and was removed. Prior missing-Git setup evidence is retained separately and not counted as an executed test pass. GitHub final-head required CI/review remain separate, including independent main A5 sort repair #1504.
