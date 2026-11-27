# Revalidation snapshot, 2026-11-27

Hardware candidates use frozen main12210ab6 plus the listed PR changes. Subsequent main a7809702 CI regressions are tracked separately. These results do not claim final GitHub required-CI success or human approval.

| PR | Verified local result / boundary |
| --- | --- |
| #1234 Ascend | Device admission/access blocked; zero original UTs ran. |
| #1248 Kunlunxin |11 original provider tests passed; all3 pure-native modes failed compiler preflight; no accepted G6 timing. |
| #1343 bad-words |6 accuracy tests and45 native0.32 benchmark workloads passed;45/45 ratios>=0.9,min0.966507. Declared legacy0.20/CUDA128 tuple not freshly qualified. |
| #1345 KDA | Original bounded40/40 and new9 resolver regressions passed; recurrent186 passed/24 existing native-ABI skips. Head1dac2904. |
| #1346 test contracts | Explicit #1345 integration:237 unique original cases+9 inherited resolver regressions passed; complete70-case dual concat filled12 earlier resource skips. Independent pushed headbb3551f4 still needs parent merge/sync/final-head CI. |
| #1347 Marlin | Full corrected659 cases:483 passed/176 original THead skips; native numerical14/14 but performance9/14,min0.328542. Headc237ed46; NOT complete-Gate accepted. |
| #1349 logprob |47 functional and22 native performance cases passed,min0.981303. |
| #1350 MHC | Cold/warm26+26,nearby182/33skip,540 numeric checks passed; native12/12>=0.9,min0.945676. Headcd1e5146. |
| #1351 MegaMoE |18 passed/6 existing Hygon skips; actual two-rank FP8 accuracy passed. |
| #1356 CI | Independent98 helpers+10 policy tests, zero skips,14 phases/3 negative controls passed. Heada497988d. |
| #1361 CI | Independent94 helpers+10 policy tests, zero skips; actual pytest first-fault/failed-JSON control passed. Headf9268b77. |
| #1504 new metadata repair | One A5 export-list ordering correction;13 validators/style pass, formal rule-check green; required H20/review separate. |

Every completed local GPU test container exited naturally, exact-owned containers were removed and selected devices were released. No operator preference/allowlist, changed numerical tolerance or native-reference substitution was used. Detailed raw timing, source/stack hashes, skipped-case reasons, failed attempts and scope limits are in the named results folders. Vendor container observation failures from missing Volcano queues remain an infrastructure task; they are not alone a failure of unittest-required.
