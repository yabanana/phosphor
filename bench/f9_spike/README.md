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

Full measurement (quiet machine, as recorded in `docs/opt-log.md`):

```sh
caffeinate -d ./build/release/f9_spike --only F9-S0,F9-S1,F9-S1b,F9-S1c,F9-S1d,F9-S2,F9-S3,F9-S4,F9-S5 \
    --runs 3 --out bench/results/f9-spike/m5max-macos27.2.json
./build/release/f9_spike --only F9-S5 --force-family apple9 --runs 3 \
    --out bench/results/f9-spike/m5max-macos27.2-s5-apple9.json
./build/release/f9_spike --only F9-S0,F9-S1,F9-S1b,F9-S1c,F9-S1d,F9-S1e,F9-S2,F9-S3,F9-S4,F9-S5 --validate
```

## S1 — BLAS lifecycle (`F9-S1`, `F9-S1b`, `F9-S1c`, `F9-S1d`, `s1_blas.cpp`)

Corpus: procedural (5 meshes) and Sponza (103 meshes); refit on a sphere, a
64x64 plane and the largest Sponza mesh; ordering on a ~1M-triangle plane
(~250k with --quick).  Every GPU result is checked against the exact CPU BVH.
- F9-S1: BLAS in device allocations, one placement heap each and several
  packed in one heap (sizes, align, allocation delta, correctness); the 103
  Sponza BLAS built with one shared scratch + AS->AS barriers vs disjoint
  scratch ranges and no barriers; build/refit scratch sizes and AS bytes per
  usage flag (None, Refit, PreferFastBuild, MinimizeMemory,
  PreferFastIntersection).
- F9-S1b: asynchronous compaction (size query in one command buffer, later
  copyAndCompact into a device allocation and into a packed heap): ratios,
  bitwise identical hits.
- F9-S1c: compute deformation of the shared GPUVertex buffer, barrier
  Dispatch->AS, refit in place and out of place (source untouched), refit vs
  rebuild time, trace time after a large deformation (refit vs rebuild).
- F9-S1d: ordering with a slow producer (two states, y=0 / y=1): build or
  refit -> trace (same-encoder AS->Dispatch barrier, none, Dispatch->Dispatch,
  two encoders with a queue barrier, two encoders without) and vertex write ->
  build (barrier / none, also with a spinning producer kernel).  20
  repetitions (6 with --quick).
Controls: shifted soup (S1, S1b), refit checked against the undeformed mesh
(S1c), the operation skipped + at least one unordered variant failing (S1d).
Env knobs: F9_S1_SCRATCH_ALIGN, F9_S1_SCRATCH_SKEW (scratch offsets).
Placement heaps are kept alive on purpose (see S1e).

## S1e — heap release under shader validation (`F9-S1e`, `s1e_heap_release.cpp`)

Reduction of a crash in MetalTools (`HeapUsageTable::processHeapEntry` at
`preCommit`): a placement heap holding a buffer or an acceleration
structure is released and the SAME command buffer object is reused.
`F9_S1E_CONTENT=buffer|as|as_built`, `F9_S1E_RELEASE=together|split|heap_first`,
`F9_S1E_CMD=fresh|reused` (default fresh).  Under
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1` every reused configuration with a
used or resident heap aborts the process; every fresh one passes.  The engine
works around it in `MetalContext::refreshCommandBuffer`.

The report distinguishes completed AS builds, verified AS traces and completed
buffer fills. `F9_S1E_CMD=fresh` with `as_built` builds the AS and waits for GPU
completion, but does **not** trace it: the existing trace helper reuses the
harness command buffer. Only `reused` verifies the one-ray hit before release.
The fresh result establishes survival across build/release/subsequent commits,
not traversal correctness. Earlier S1e reports that called fresh builds
"built and traced" overstated that scope; they are retained as historical data.

```sh
F9_S1E_CMD=reused MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
    ./build/release/f9_spike --only F9-S1e,F9-S2 --quick   # aborts: the repro
```

## S2 — TLAS written by compute from the GPU scene (`F9-S2`, `s2_tlas.cpp`)

Can the engine's GPUInstance buffer (capacity >= live count, invalid slots,
recycled generations, mirrored transforms) feed a per-frame TLAS built or
refitted from descriptors written by a compute kernel, and how fast (target:
100K dynamic instances <= 0.5 ms on T2).  Corpus: 4 BLAS (cube, sphere,
torus, icosphere), N = 1K/10K/100K live instances (--quick: 1K/10K), random
rotation, scale 0.5..1.5, ~10% mirrored, 5% invalid slots, animated by a
kernel each frame.  Strategies (`tlas.<size>.<A|B|Bs>.<metric>`): A =
descriptor per slot (invalid: mask 0); B = atomic compaction; Bs = stable
compaction; B/Bs use IndirectInstanceAccelerationStructureDescriptor with a
GPU-written count.  Per strategy: descriptors.ms, build_none/fast/refit.ms,
refit.ms, update_*.ms (descriptors + barrier + build/refit), bytes.  userID:
`.user_instance_id` of intersector<triangle_data, instancing>; instance_id =
descriptor index.  Checks (0 wrong): an exact CPU two-level reference after
build, 1 and 5 refit frames, rebuild, delete/reuse of recycled slots (new
mesh + generation read on the GPU through userID); front face for every
instance option.  Negative controls: transposed 4x3, deleted slots with mask
0xFF, mirrored front face read without the CCW option.

## S3 — traversal (`F9-S3`, `s3_traversal.cpp`)

Cost per ray of the Metal RT traversal APIs on Sponza and robustness of
secondary-ray origins; 1920x1080 (quick 960x540), two cameras (`.camB`), a
directional sun.  Ray types: primary, shadow, AO (cosine hemisphere, tmax
0.5), diffuse (cosine hemisphere).  A primary-hit kernel stores point,
normal, ids and Waechter-Binder offset points; the secondary kernels read
them.  Variants: intersector closest (with/without assume_geometry_type),
intersector any-hit, intersection_query closest/any, non-opaque instances
with/without force_opacity.  Self-intersection strategies: none, tmin 1e-4 /
1e-3, W&B offset in object space, W&B offset in world space; metrics
`selfhit.<s>.*` and `shadow.<s>.agree_pct/leaks/acne` against an exact CPU
shadow ray.  Control: strategy none must self-hit.

## S4 — RT proxy geometry (`F9-S4`, `s4_proxy.cpp`)

Index-only proxies (meshoptimizer) over the original GPUVertex buffer.
Levels per mesh: full (identity control), r50/r25/r10 (ratio, border free),
r50b/r25b/r10b (border locked), e3/e2 (error driven), s10/s01 (sloppy).  A
reference TLAS (full BLAS) and a proxy TLAS with identical instances are
traced: primary hit/miss and |dt| (3 cameras at 960x540), sun shadows from
the full-geometry hit points against all instances and against the
receiver's own instance (acne), error by distance and blame per mesh.  Three
policies start coarse and upgrade the blamed meshes until shadow <= 0.5%,
primary bad <= 0.2%, dt95 <= 1 cm, acne <= 0.5%.  Controls: full-vs-full is
exactly 0, s01 exceeds the thresholds, a sabotaged mesh is detected.

## S5 — alpha test in RT (`F9-S5`, `s5_alpha.cpp`)

Alpha test in intersection functions with the raster rule (alpha =
baseColor.a * texture.a, accept when >= cutoff).  Corpus: Sponza (3 masked
materials) in 4 views + shadow rays, and a synthetic scene with known holes.
Variants: A (one generic function at offset 0, masked geometry non-opaque),
A_pd (UVs from primitive data), A_all (every geometry non-opaque), B
(per-material table slots, Apple10 only), B_all, C (all opaque, negative
control).  Checks: CPU reference with the same rule and identical CPU-built
mips (texels within 2/255 of the cutoff reported as ambiguous), analytic
holes, explicit LOD 2, function-invocation counts (0 on opaque geometry),
shadow leaks through opaque occluders.  LOD study (CPU): raster-like
derivative LOD vs LOD 0 vs ray-cone LOD.
