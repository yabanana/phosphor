# bench/f9_spike — F9 ray-tracing spikes S0–S5, measurement tool, not engine code

Results live in `bench/results/f9-spike/`; decisions in `docs/opt-log.md`,
"F9 — Spike".  Spikes only inform the choices the F9 plan leaves open
(`docs/plans/F9.md`); a spike number never replaces the measurement on the
engine's own workloads.  Every number in the opt-log was measured, none is
assumed.

Runner: the bench/soc harness (`--list`, `--only F9-S1`, `--runs N`,
`--quick`, `--validate`, `--force-family apple9`, `--out FILE`).  Shared
helpers: `f9_common.{h,cpp}` (exact double-precision watertight CPU
reference with a BVH, Sponza through the engine's GltfLoader, BLAS per mesh
reading the engine's `GPUVertex`/index layout in place, TLAS of indirect
instance descriptors, `traceNearest`).  MSL: `shaders/` (`f9_rt.h` = ray/hit
layout).

GPU safety as in F5/F6: bounded loops, no inter-threadgroup waits, command
buffers far below 1 s, one GPU process at a time.

## S0 — smoke (`F9-S0`, `s0_smoke.cpp`)

Procedural corpus (5 meshes) and Sponza (103 meshes, 262,267 triangles):
BLAS per mesh, TLAS, camera rays traced with the intersector and compared
with the exact CPU BVH (t within 2e-4 relative, the named triangle must
contain the hit up to 1e-5 barycentric).  Negative control: the procedural
rays checked against a soup whose first instance moved by 0.15 must fail.

```sh
./build/release/f9_spike --only F9-S0 --runs 1
```
