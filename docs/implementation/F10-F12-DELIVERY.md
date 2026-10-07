# F10-F12 writer delivery — NON VERIFIED

Owner-authorized source writing only. Base: `7997f12713f07895bdf27d7ab27f7947f3da7bd7`.
Managed worktree: `/Users/danielsan/.codex/worktrees/f10-f12-development/phosphor`.
Delivery ref: `codex/f10-f12-delivery` (frozen after this document). Development
continues separately for F13-F14, then STOPS. No F15/OPT activation.

NO configure/build, C++/MSL compile, CPU/GPU test, renderer, benchmark, profiler,
simulator/container or reference generation was executed. The only executed
check is Git diff formatting (`git diff --check`). None of these phases is
accepted or ticked; `docs/ROADMAP.md` is untouched.

## Integration prerequisite

The source baseline predates the aggregator's F9 `e72adfc` linked-compute archive
leak correction and `drainFrameReadbacks` change. Keep that F9 PipelineCache
implementation; this writer did NOT change PipelineCache. Reconcile Engine
teardown so new snapshot/capture/check readbacks are consumed after completion
BEFORE old scene resources or textures are reloaded/released. F9 execution,
compilation, CI and merge remain aggregator-owned. Do not cherry-pick the
subagent baseline snapshot `72ec3fb`.

The delivery contains small commits per task; F10.3 joins the F11 direct-light
package. Shared integration fixes are intentionally shared commits. Use the
complete frozen range for the first source review/build; standalone subagent
ranges do not constitute renderer integration.

## F10 writing

- ABI and portable cascadings/oracles: `e064b69`, `b387c00`.
- CSM/RT/contact kernels: `69bcd73`, `84d57ef`.
- PSO-relative IFT/AS API and renderer integration: `21eaf73`, `a8c6a14`, `16bb31e`.
- Regional static cache and full-scene world bounds: `e5b0762`, `01ef014`.
- Four independent Depth32 maps; independent caster flags over ALL scene slots.
  Stabilized unjittered splits, orthographic PCSS, reverse-Z MAX composition
  of cached/current-static and dynamic depth before filtering. Eight cache
  tile admissions per frame; stale tiles ALWAYS use current static raster.
- Point/geometric normal guide -> indexed overflow receiver prepass -> pack ->
  shadow visibility -> resolve. Only selected sun DIRECT contribution is masked.
  RT sun samples the physical disk once; any-hit distance is never a blocker
  distance. Alpha LOD0 and world W&B hemisphere offset.
- Solar histories per view/signal, exact CPU content epochs, moments, bilateral
  filtering and optional contact; caster/material/light changes reject history.
  Conservative global rejection on moving casters is a quality/cost experiment.
- Cache `on` requires CSM; RT cache flags are rejected. Forced Apple9 exercises
  indexed casters on the M5, without any physical M3 claim.

## F11 writing

Portable sampling/reservoir/STBN: `687e815`, `068cf0f`, `67772ca`.
Shaders/reference tests: `c74b00e`, `83d6420`, `3608e02`.
Host integration/actual emissive vertices/checks: `9f2a3b3`, `290dbd9`.

Candidate -> temporal -> spatial -> selected-sample RT shading are separate
passes/buffers. Local radiance REPLACES only the non-directional direct-light
loop, preserving sun/ambient/emissive. Area rectangle/disk/tube and mesh emissive
sampling use area PDFs/Jacobians. Alias proposal has full support; final target
uses actual material/texture data. Three-dimensional clusters have bounded lists
and FULL-light fallback on overflow. Brute and reduced presets stay selectable.

`GPUEmissiveSurface` is now **96 B**: local geometry/UV/material/slot plus three
live raster vertex indices. `light_emissive_update` additionally binds current
vertices at buffer6, after graph geometry producers. Rigid emitter/light motion
can reuse DI; identity/domain revision is separate from full radiance revision
used by GI. STBN is a defined original generator at loading time, with no copied
asset or measured spectrum/quality claim. All target/PDF/M/normalization and
negative controls are documented in `docs/implementation/F11-CONTRACT.md`.

GPU checker reads actual reservoir error flags, normalization, identities and
output. PDF/light negative controls mutate actual state. Full lighting denoise
remains F13 work; no F11 denoised-quality exit claim is made here.

## F12 writing

Portable/GPU packages: `536f00f`, `fdd5dc2`, `b5438e8`, `8de120c`, `888b6fb`.
Tests/material/solar refinements: `28129b6`, `d2e0997`, `40df0a2`.
Capture/graph: `5b5ec31`, `4811885`; independent checks: `68896cc` and following
source-review fix commits. See `docs/implementation/F12-CONTRACT.md`.

DDGI traces, classification/relocation and irradiance/distance-moment atlases are
connected to the renderer. Default volume comes from evaluated world poses and
conservative motion envelopes, with explicit grid/spacing/embedded-probe overrides.
Directional radiance hash cache is bounded, compares full keys, has generations,
eviction and serial bounded updates. It stores outgoing RADIANCE, not irradiance.
Cache mode is fresh cosine-path RIS; ReSTIR GI uses separate temporal/spatial
reconnection, with documented area estimator/shift and explicit BASIC-REUSE BIAS.
DDGI remains selectable. GI replaces only diffuse ambient; no direct/emissive
energy is counted twice. Complete denoise/rough reflection consumers are F13.

`--export-reference` copies exact same-frame GPU world poses, materials, CURRENT
full raster vertices, world sampled lights, metadata and retained LINEAR decoded
texture texels. Camera jitter/infinite-far and solar radius are exported. The
reference runner uses an already installed independent Mitsuba CPU renderer;
it never installs/downloads dependencies. No reference image was generated.

Linear PFM captures are before exposure/upscale/tonemap/UI. Select `hdr`, `direct`
or `indirect-diffuse`. The latter is E*texturedAlbedo*(1-metallic)/pi BEFORE the
engine's artistic AO/specular hemisphere and matches the stated diffuse reference
signal. It does not pretend to validate the full GGX model. Model differences
are explicitly gated by the offline tool; freeze ROI/tolerances before rendering.

## Files/commands for the tester — ALL NOT EXECUTED

Core: new shadow settings/math, light sampling/reservoir/STBN/local-light scene,
probe grid/cache/GI reservoir/offline exporter and doctest files.
Metal: `rt_consumer`, `shadow_passes`, `direct_lighting_passes`, `gi_passes`,
`reference_snapshot`, `linear_capture`; Engine/VisibilityRenderer composition.
MSL: shadows/cache, DI/cluster/light visibility, DDGI/cache/GI, geometric/material
receiver guides, independent checks, capture and the single `rt_intersections`
export. F9-K1 includes that single intersection source explicitly.

Prepare a tester-only build (never the active writer's/main checkout):

```sh
cmake -S . -B build/lighting -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/lighting
ctest --test-dir build/lighting --output-on-failure
python3 tools/f10_f12_check.py --app build/lighting/phosphor --output build/f10-f12-results
python3 tools/f10_f12_check.py --app build/lighting/phosphor --output build/f10-f12-results --run
bash tools/harvest_lighting_pipelines.sh build/lighting
```

The first Python invocation emits a NOT_EXECUTED manifest. `--run` is explicitly
TESTER work and serializes jobs. Its functional results are NOT phase acceptance.
Use API/shader validation, native/reduced Apple9 paths, 1-4 views, resize/DRS,
camera cuts, moving lights/emissive/casters, switch/hot reload, archive fallback,
leaks, zero steady-state GPU allocations, negative controls and F9/F7/F8 regressions.
Per-phase raw files/commands are in the writer handoffs and runner manifest.

Cornell/thin-wall/offscreen/bias/disocclusion/cache-stress fixtures are selected
with `--bench 6 --lighting-scene NAME`. To actually test an embedded DDGI probe,
use `--gi-probe-anchor 0.9,1,-0.8 --gi-spacing 0.5`. A default auto grid does NOT
prove a probe was embedded. Freeze snapshot/candidate at matching frame numbers.
Then execute independent convergence/reference and temporal ROI comparisons.

## Pending source and evidence review

- Every new C++/MSL file remains UNCOMPILED. Static source review caught and fixed
  mutable mask output refs, history publication/all-view reallocation invalidation,
  moment-channel loss, placeholder world bounds, XOR content epochs, physical sun
  PDF, actual vertex snapshot dependencies and incompatible comparison signals.
  These source fixes remain runtime UNVERIFIED.
- Committed pipeline archive is **NOT re-harvested**. The new harvest runner is
  written only; do not claim new pipeline archive coverage before tester harvest.
- Probe/cache/GI presets, tolerances, spectral STBN quality, cache/ray budgets,
  footprint and convergence are experimental; no fabricated measurement.
- F13 denoise and reflection/AO composition remain the next writer packages.
- Physical M3/T0 certification remains EXTERNAL_VALIDATION_PENDING. No OS work,
  push, PR, merge, roadmap tick or performance-log number was performed here.
