# MHC #1350 current-main revalidation

Frozen main12210ab6 + original head aa3e24a, resolved tree615c4205, now published as normal synchronized source head cd1e5146. Main's two AABS guards are preserved; the actual tested bytes did not change on push.

G5: cold26/26 and warm26/26 norm tests; all215 nearby tests collected,182 passed/33 existing skips; all15 reader configurations x12 original shapes x3 repeats =540 numerical checks passed at the original tolerances. Configuration forcing was confined to that correctness process. Three actual conftest observers recorded context not poisoned.

G6: new process/container/cache, actual automatic source tuning, all12 original shapes passed numerical checks and native-reference SpeedUp>=0.9, minimum0.9456759503314002. Original30ms warmup/80ms repetition/3 native-candidate pairs; ratio protocol median(native3)/median(candidate3). Native callable was the installed vLLM0.32 tilelang.mhc_pre_tilelang with recorded source SHA, not a Torch replacement. Source1808 raw Git members matched before/after; both containers exited naturally, were removed and GPU5 was released. New final-head required CI and reviewer approval remain separate.
