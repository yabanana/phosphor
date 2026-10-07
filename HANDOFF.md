# F10 writer handoff — NON VERIFIED

Writer worktree: `/Users/danielsan/.codex/worktrees/f10-shadow-contracts/phosphor`.
Branch: `codex/f10-shadow-contracts`.
Base: `7997f12713f07895bdf27d7ab27f7947f3da7bd7`.

No configure/build, shader compilation, tests, renderer execution, benchmark,
GPU probe, profiler or large download was executed. Only source reading,
source writing, commits and `git diff --check` were performed. No ROADMAP
status was changed. This is a writer package; host integration and all
verification belong to the parent/tester.

## Commit boundaries

- `091e844` — F10.1/F10.2 shared scalar ABI and pre-resolve guide contract.
- `d07c1bc` — F10.1/F10.5 portable cascades, history/caster/bias predicates,
  bounded cache policy and authored CPU tests.
- `62f2f03` — F10.1/F10.2/F10.4 full shader paths, bindings and contract.
- The handoff follow-up commit includes variance-aware spatial filter weights
  and this document; no empirical choice is represented as an adopted preset.

## Delivered files

- `src/renderer/gpu_types.h`: GPUShadowCascade 112 B, GPUShadowParams 864 B,
  GPUShadowSurface 48 B, GPUShadowHistory 64 B, GPUShadowCounters 32 B.
- `src/renderer/shadow_layout.h`: common stable argument table slots.
- `src/renderer/shadow_math.h`: shared CPU/MSL conservative sphere selection,
  reverse-Z visibility, orthographic solar penumbra and history identity checks.
- `src/renderer/shadow_settings.{h,cpp}`: opt-in ShadowTechnique/settings,
  split generation, stabilized overlapping orthographic cascades, uniform
  solid-angle solar sample oracle, changed-caster dirty tile mask and bounded
  revision/timeline-safe static-cache admission policy.
- `shaders/shadows.metal`: clear counters, surface guides, full-scene caster
  flags, indexed/mesh alpha depth raster, four-map PCSS, one solar RT ray,
  temporal moments, bounded spatial filter and contact composition.
- `tests/test_shadows.cpp`: tests WRITTEN ONLY, including invalid settings,
  geometric coverage/snapping/off-camera caster checks, independent PCSS
  units, solar distribution, history isolation and cache budget/lifetime.
- `docs/implementation/F10-CONTRACT.md`: complete kernel bindings, APIs,
  graph/ownership requirements, mathematical estimator and pending checks.

## Integration contracts and dependencies

Guide extraction reads F7 meshlet candidate A/B lists and V-buffer IDs, with
phase B based at candidateCapacity. It reconstructs world position from depth
using current jittered inverse VP, and geometric world normal from the actual
raster triangle. Outputs are both the surface buffer and world-position
RGBA32Float/geometric-normal RGBA16Float textures, with validity in alpha.
The camera-visible geometric normal is reoriented toward each outgoing RT ray
before W&B. Never use a normal map or proxy triangle for this offset.

CSM uses four independent Depth32Float graph refs and four raster passes,
clear zero/Greater. PCSS textures are 2/7/8/9. Raster input is the full scene
slot set and each mesh's full geometry. The host now reconstructs all motion/hierarchy world matrices with the exact uploaded renderer sin/cos table and passes full-scene caster bounds to makeShadowCascades. The finite casterReach preset no longer defines the maximum scene reach. CSM normal/depth bias values and
solar radius are tuneable experimental presets, with no quality measurements.

The RT consumer must create its IFT from its OWN resolved PSO, linked to the
single rt_alpha_generic TU, and recreate it after pipeline hot reload. Use
F9 readonly typed slot TLAS ref plus BLAS dependencies at Dispatch. Mask is
RT_MASK_SHADOW, traversal any-hit, alpha LOD0, nominal sample count ONE per
valid selected-light pixel. No diagnostic F9 IFT reuse is permitted.

Each history is per-view and per-light/signal, not per buffer-ring slot.
Advance HistoryRegistry after actual submission, use its previous jittered
VP, and enforce its requiredCompletion at graph ownership boundaries. Moved
receiver world points reject temporal reuse when prior object poses are
unavailable. Caster/light/alpha-material revisions clear stale visibility.
Use RGBA16Float for temporal/filter masks if consuming variance/sample count;
R16Float retains only visibility. Store separate stable graph refs for raw,
temporal, filter and contact outputs in encoding lambdas.

Only the selected directional direct-light contribution is multiplied by the
visibility signal. HDR total, ambient, emissive and all other direct-light
terms are preserved. Contact composes by min. F10.3 local-light visibility
joins F11.1; complete lighting-denoised acceptance depends on F13.

F10.5 now includes actual 8x8 tile GPU updates per view/cascade. Missing/stale static tiles are rasterized CURRENT every frame; at most cacheUpdateBudget are copied into persistent R32Float cache. Dynamic casters always use a separate current depth texture. The fullscreen composite emits Depth32Float MAX(static cached-or-current depth, dynamic depth) before PCSS. Exact tile revisions are GPU-validated and persistent resources are graph imported. This implementation is WRITTEN, with all runtime verification pending.

## Commands for the tester — NOT EXECUTED

After CMake and host integration in the parent worktree:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

Compile all MSL and app on macOS; run the Linux CI metal_syntax_check in its
supported configuration. Then preserve exact manifests and image references:

```sh
./build/phosphor --render-path visibility --shadows csm --shadow-map-size 1024 --frames 240 --warmup 30 --debug-lighting 1
./build/phosphor --render-path visibility --rt on --shadows rt --frames 240 --warmup 30 --debug-lighting 1
./build/phosphor --render-path visibility --shadows csm --contact-shadows on --shadow-cache on --frames 240 --warmup 30 --debug-lighting 1
./build/phosphor --render-path visibility --rt on --shadows rt --force-family apple9 --frames 240 --warmup 30 --debug-lighting 1
```

These are the parent's authored CLI spellings, not confirmed runnable results.
Repeat relevant scenes under API/shader validation. Exercise camera cuts,
slow translation/rotation, cascade boundaries, thin and alpha geometry,
upstream off-camera casters, moving casters and alpha-material changes,
resize, view switching and scene switching. Preserve unshadowed A/B/A
baseline and direct-light-only composition golden/readback. Record GPU memory
allocation count, per-pass work/cost and leaks in the tester's own lane.

The following controls MUST fail with exit status nonzero after independent
readback comparison; merely accepting the flags does not validate them:

```sh
./build/phosphor --render-path visibility --rt on --shadows rt --frames 120 --debug-lighting 1 --debug-lighting-corrupt bias
./build/phosphor --render-path visibility --shadows csm --frames 120 --debug-lighting 1 --debug-lighting-corrupt caster
./build/phosphor --render-path visibility --rt on --shadows rt --frames 120 --debug-lighting 1 --debug-lighting-corrupt history
./build/phosphor --render-path visibility --shadows csm --shadow-cache on --frames 120 --debug-lighting 1 --debug-lighting-corrupt cache
```

Shader bias flips the W&B hemisphere, caster flips the first cascade flag,
history mutates previous generation. Cache corruption forces tile zero into the ready mask, suppresses its update, and alters the expected projection revision. shadow_cache_validate must increment errors for the stale/uninitialized exact tuple. The tester must
ensure the chosen scene actually contains the relevant caster/history/cache
case and that each failure occurs for the intended invariant.

## Outstanding checks and real limitations

All CPU/GPU/runtime checks above are NOT EXECUTED. MSL and host ABI compilation
are unverified. Parent source review has identified the stable-mask-ref,
history publication, temporal texture-channel and full caster-depth-bound
requirements above. There is no measurement or phase acceptance in this
handoff, and no physical M3 certification. Only M5 Max 128GB is available;
an effective Apple9 run on that machine exercises the software fallback.

The direct all-pairs mesh grid is deliberately a correctness baseline and
may require the indexed fallback or later compaction at scale. Cache update
rendering is parent-owned. Deformed receivers have safe history rejection
rather than per-instance previous-pose motion reuse. Generic alpha must be
defined only once in the aggregator's extracted rt_intersections.metal TU;
the original F9 base has an unguarded rt_common body. No local duplicate-TU
compilation was attempted.

## F10.5 regional GPU follow-up — NON VERIFIED

Parent host baseline snapshot is `72ec3fb`, identical to the parent's
`a8c6a14` shadow files. DO NOT cherry-pick the snapshot commit; cherry-pick
only the following regional implementation delta onto the common parent
host. It owns shadow_passes.cpp/h, adds shadow_cache.metal and adds only
GPUShadowCacheParams/GPUShadowCacheTile to the shared GPU header. The parent
F9 PipelineCache leak fix `e72adfc` is an integration dependency; this writer
did not modify PipelineCache.

New host hooks:

- Copy the exact `motionSinCos_` values already uploaded to SceneRenderer
  into `ShadowPasses::Frame::motionSinCos` and set `motionSinCosValid=true`.
  GPU-motion scenes without this input fail explicitly. Parent-before-child
  mat4Mul and motionWorld reuse F5's shared arithmetic, with preallocated CPU
  scratch. Full-scene bounds include every valid casting slot independently
  of camera visibility.
- Read `shadow.depth()` for the LAST receiver-prepass depth version in DI
  material prepass and visibility resolve.
- Add `shaders/shadow_cache.metal` to the parent's metallib source list and
  harvest the cache initialize/publish/validate/indexed/mesh/composite PSOs.
- `cacheState(view)` exposes CPU admission counts (ready/current/update tiles,
  bounded update budget, static/dynamic classifier counts). These are not
  measured GPU results or completion proof.
- `--shadow-cache on` explicitly requires `--shadows csm`. Constructor also
  rejects other modes; RT solar history remains its independent signal.

Each of four views owns four persistent R32Float depth maps and four 64-entry
GPU revision buffers (each metadata record is 48 bytes). Per-frame expected
revision records and static classifications use frameUploads. Resource
creation/release goes through GpuMemory. Tables are allocated at startup/scene
load per slot/cascade/class; indexed draws have a distinct table per mesh.
Cache initialization, publication and validation each have distinct tables.
No table is mutated for another encoded draw.

Static classification requires VALID+castsShadows+static and excludes every
GPU motion root and hierarchy child. Changed static slots invalidate the union
of OLD and NEW conservative bounds, expanded for the maximum PCSS footprint,
for ALL previously active views. Alpha-material/global light changes invalidate
all affected cascade tiles conservatively. Projections compare exact snapped
cascade data; pipeline generation changes invalidate alpha materials as well.
There are no age-only hits or hashed equality substitutes.

The steady graph always records bounded work: current stale static region
raster (tile discard mask), current dynamic raster, admitted-tile copy, exact
GPU validation and depth composition. The update quota is shared across all
256 active-view tiles and rotates admission. Stale tiles not admitted still
render CURRENT static fallback. Ready tile depths are not rerasterized or
rewritten; dynamic texels are never stored in static cache. A temporal/spatial/
contact RGBA16Float mask preserves moments/sample count.

Additional tester requirements: freeze static camera/light, fill the cache
across frames and verify ready tiles rise to 256; never exceed update budget;
move/remove a static caster and change alpha material, then verify old/new
regions fall back to current depth on the SAME frame; animate roots/children
misflagged static and verify they remain dynamic; move the light/camera and
verify exact projection invalidation; switch views while a caster changes and
verify the dormant view's cache was invalidated. Compare composed DEPTH with
uncached full-scene CSM exactly before PCSS, including overlapping static and
dynamic occluders (MAX reverse-depth). Negative cache control must fail on
actual GPU metadata validation, not merely on accepted CLI flags.

Source-only DirectLightingPasses review additionally found texture slot12
bound while lighting::table default maxTextureBindCount was12 (valid indices
0..11). Parent owns the correction to at least13/16. No DirectLighting source
was edited by this writer.
