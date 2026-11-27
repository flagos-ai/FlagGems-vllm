# Formal CI failures are tracked separately from local results

Representative original logs show two source-side issues: stale event-base selection included inherited main files in PR checks, and new main A5 exports were out of order. They are repaired independently by #1356 and #1504; original job URLs and exact first causes are indexed here.

Affected NVIDIA/MetaX/MThreads/Hygon container-smoke runners are rejected before checkout by Volcano because their configured queue objects are missing. This is runner/cluster infrastructure, not an operator execution result. The repository config treats these vendor-container jobs as optional observations, so their failure alone does not fail unittest-required. Required H20/rule checks and human approval must still be assessed on each final head. No queue, gate requirement or operator allowlist was modified.
