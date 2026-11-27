# P800 #1248 native-reference preflight: blocked

The image-baked provider and current-main candidate independently passed the11 original operator tests; this native-only preflight is separate. Runtime: vLLM0.32.0+flagos, Torch2.9+cu129, FlagTree0.7.0+xpu3.6, FlagCX distribution0.14.1+xre5.37.1. Each mode used a fresh independent container/cache and actual unpatched installed vLLM callable. No FlagGems-vllm/plugin import, altered fixture, Torch replacement or timing.

- Grammar and penalties: original block8192 failed ConvertTritonToTritonXPU/GlobalEncodingAttr compilation.
- Bincount: the first original fixture(167,185) failed arith.divsi type legalization; the remaining three fixtures, including2048, were not executed in this run.

All three native processes exited1 naturally, exact owned containers were removed and selected P800 physical4 was released. Controller exit0 only confirms evidence collection. Native G6 remains blocked; no performance pass is claimed. Existing upstream requirements: FlagTree issues1700/1701; production compiler/image pins unchanged. Single-device operator results do not requalify rank1 NoC,35B or two-host FULL.
