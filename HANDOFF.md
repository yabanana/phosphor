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
slot set and each mesh's full geometry. Full-scene caster bounds must be
passed to makeShadowCascades; casterReach=500 is an experimental bounded
preset, not a proof for arbitrary scenes. CSM normal/depth bias values and
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

Cache API is complete portable admission/lifetime logic. The parent host must
connect it to region rendering/composition. Key includes view/light/cascade/
tile and revision includes light/caster/alpha-material/projection. Union old
and new caster region masks. Dynamic/stale/budget-starved regions use CURRENT
dynamic CSM. Never reuse stale static depth. The parent currently starts with
coarse per-cascade regions and full-current dynamic fallback; fine 8x8 tile
region rendering is explicitly still an integration expansion, not verified
F10.5 completion.

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
history mutates previous generation; parent cache corruption must inject a
stale revision/publication into its independent checker. The tester must
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
