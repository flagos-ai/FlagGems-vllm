# Marlin #1347 corrected full functional validation

Published head c237ed46/tree4be9e45c corrects only the pruning test's production-table assumption: explicit stage1/2/3 fixtures preserve all original pruning assertions, and a separate strict test checks the real Hopper single-stage table. No production kernel, tuning table, benchmark input, numerical tolerance or skip changed.

The complete659-case module ran once in a new container/cache:483 passed/176 original THead-only skips/zero failures,1973.23s, actual conftest context not poisoned. Source1808 raw members, natural exit, exact-container absence and GPU6 idle were verified. Original1F and exactmain unit control remain in the diagnostics folder.

All production and benchmark blobs are identical to the completed14-case G6 diagnostic. Its14 numerical passes/154 matching input identities and **9/14 performance passes** therefore remain applicable; five cases still fail>=0.9, minimum0.328542. This PR is **not accepted for merge by the complete performance Gate**. No additional performance rerun was used after this test-only correction.
