# F13 host handoff — SOURCE ONLY / NON VERIFIED

Base `524bf1f187b13f1cd69d053805d53b9b3bfd1f24`, branch
`codex/f13-host-signals`, managed worktree
`/Users/danielsan/.codex/worktrees/f13-host-signals/phosphor`.
No configure/build/C++/MSL compilation, test, renderer, benchmark, profiler or
simulator ran. Only source/SDK reads and `git diff --check` were performed.
No phase checkbox/acceptance, push, PR or merge is delivered. Stop at F13 host.

## Connected implementation and commits

- `34c8e99`: new DenoisePasses, composite/reduction/probe check kernels and
  shared compose/reduction layouts.
- `3f5c831`: new ReflectionPasses, actual scene/static capture, every face/mip,
  cooked PFM/analytic source and F13 graph/bind/lifetime/composition wiring.
- `9180ed3`: minimal existing F13 shader fixes preserve numerical faults before
  black sanitization (SPECULAR_SAMPLE_ERROR and capture alpha=-1). No F11 edit.
- `e0e30bb`: cached graphs receive live exact DI/GI/SPEC/AO revisions each frame;
  per-view/signal tuple normalization and monotonic structural version.
- subsequent host-handoff commit: bounded static OBB capture-position heuristic
  and this source handoff. All are NON VERIFIED.

The root owns Engine, Visibility, MaterialShading, CMake, CLI, Post, MetalFX
adapter and atmosphere. New files do not modify them. The root's later
material-occlusion guide write, old-specular-omit flag8, physical-GI flag16 and
RESOLVE_EXTERNAL_DIFFUSE flag32 are prerequisites for exact composition; material occlusion is read directly
from GPUDISurface.pad[0], including a legitimate zero value.

## Public API and caller order

ReflectionPasses constructor takes Context,PipelineCache,SceneRenderer,
DirectLightingPasses&,AccelerationStructures*,GiPasses*,LaunchOptions.
API: loadScene(GpuScene,SceneStore), ready(),
prepareFrame(SceneStore,ShadowPasses::Frame,useCustom,externalContentEpoch),
addToGraph(graph,baseHDR,depth)->newHDR, bindFrame(executor), version(),
check(completedSlot), rawSpecular/rawAO/filteredSpecular/filteredAO/hitDistance,
metadataRef and probeSource.

Wait/gate initial async pipeline readiness using ready() BEFORE the first
prepareFrame (consumer IFT preparation requires a real resolved PSO). The root
calls loadScene after scene geometry/store upload; prepares F13 after Direct
and GI preparation; declares F13 after Direct/GI producers and the base HDR
resolve; passes the returned HDR to Post/SDK packing; and includes version()
in the cached graph key. bindFrame follows graph compilation and binds all
imports for this frame slot/view. check(slot) is read only after that slot's
GPU completion, before preparing it again. Numerical error words are kept
even when debugLighting is off; the root must read check for those failures.

The primary residual HDR excludes external DI/GI and hemisphere diffuse under
flag32, retaining sun/emission/legacy direct and old hemisphere SPECULAR only
when F13 reflections are off. Residual storage may initially remain RGBA16;
the host adds no quantized-signal subtraction. A separate RAW positive assembly
pass adds raw DI and raw GI E (or ambient without AO) into RGBA32 before SSR.
SSR reads that complete pre-reflection Lo, never a stripped residual or final
reflection image. Final output and pre-reflection source are always RGBA32 with
the residual backing extent; the root selects physical Float32 Post storage
so this sum is not immediately copied back to half before exposure. Padding
is copied from the defined residual. The half-range predicate remains conditional
on actual half output and is inactive for these Float32 outputs.

Positive final composition is:
`residual + DI_selected + GI_selectedE*albedo*(1-metal)/pi + SPEC_selected`.
Internal custom GI denoises IRRADIANCE E, not albedo-baked Lo; the material is
applied once at the current receiver. Raw/MetalFX mode selects raw DI/E/specular;
custom mode selects filtered DI/E/specular. When GI is OFF, add guide ambient
diffuse times AO (or1 when AO is off), using the current mapped normal/albedo/
metallic/material occlusion. DI, GI,
specular, emission and HDR total are not AO multiplied.

## Cached graph data and history identity

DenoisePasses is reusable for4 signals. prepareFrame receives the exact live
array `[DI radianceRevision, GI cacheGeneration, SPEC externalRadiometricEpoch,
AO geometryEpoch]` before every frame, regardless of graph recompilation.
addSignal carries only resource references; it no longer caches a revision.
Each view/signal compares exact `(scene,externalEpoch,revision)` and advances
a monotonic content epoch on change. Registry.begin and GPU signalRevision
use that normalized epoch, with no XOR identity collision. The root's external
content epoch must include full analytic AND sampled light radiance, geometry,
materials, RT geometry and pipeline programs, including sunlight with GI OFF.
The host derives AO geometry epoch from exact structure/scene/RT geometry and
CPU instance/node/material/motion update counters plus pipeline generation.

History pairs are allocated lazily per active view/signal. Resize reallocating
one pair invalidates that view/signal; F13 signal pool growth invalidates ALL
view/signal registries. Cut/reset/DRS/scene/domain changes are explicit. Both
history imports remain persistent; temporal reads one side and writes the
physically different side at Dispatch, so the graph's same-queue cross-frame
barrier orders both. Raw signals/moments remain separate from atrous feedback.
All frame upload addresses, sides and params refresh in prepareFrame, including
the cached graph. Custom/raw path toggles invalidate histories and change the
structural key. Reflection.version compares the exact structural-component
pair and emits a monotonic published version, not an XOR of component versions.

## Reflections, AO, static captures and imports

Reflection samples1..8 each own an argument table and output record slice.
Record-buffer versions retain every earlier slice, so graph culling cannot
drop prior proposals. Reduction averages raw BRDF-weighted Lo before denoise;
mixed secondary identities/paths carry SPECULAR_SAMPLE_MIXED and hitDistance0
rather than a fabricated geometry hit. RT/SSR/probe-only and RTAO/GTAO are
actual separate PSOs; NoRT paths bind no acceleration structure. Every RT
PSO owns its own RtConsumer/IFT, and traversal passes use rt.declareTraceReads
for BLAS, typed TLAS and AS→Dispatch ordering. GI uses the read-only accessor
resources and safe initialized zero/dummy bindings when off.

One64² cube-array probe has separate raw storage and a seven-mip destination.
All14 views (six face views, raw array view, seven destination mip views) are
created/labelled/resident/retired through GpuMemory.newTextureView; parent
storage remains alive through view retirement. Filter reads raw mip0 and
writes a DIFFERENT destination; each mip has its own table and write version.
The whole parent and individual views have explicit dependencies, and consumers
wait for every producer. A GPU publication pass samples completed filtered
storage and writes persistent metadata before enabling it. Probe validation
failure is sticky per source generation across per-slot counter clears, so an
invalid capture cannot become accepted merely on the next frame.

`--reflection-capture-probe` selects actual scene RT capture when AS is active,
otherwise actual indexed full STATIC scene capture with an explicitly unshadowed
local/direct baseline. The indexed list excludes GPU-motion slots and hierarchy
children; original mesh index/vertex ranges and unique per-face/per-draw tables
are used, not the main camera's culled ICB. Structure changes rebuild that list
at a GPU-idle resource event. Capture positioning starts from scene bounds and
uses a bounded conservative static OBB escape heuristic for the trivial
centre-inside-cube/thin-wall case; it is not a general point-in-mesh certificate.
Reference quality still requires inspecting the chosen probe volume/position.

`--reflection-probe-path DIR` expects six64×64 LINEAR RGB Float32 PFM files:
px.pfm,nx.pfm,py.pfm,ny.pfm,pz.pfm,nz.pfm. Reader validates size/header/endian/
scale/finite nonnegative radiance and flips bottom-to-top PFM rows. Loading
uploads use stagingAllocate/enqueueUpload/flushUploads. No path/capture selects
an explicitly labelled analytic environment; it is not actual scene capture.
probeSource reports that distinction. Physical32-bit float filtering capability
selects RGBA32 probes only when not forced/reduced; otherwise RGBA16 is the
compatible floor and unrepresentable cooked HDR is rejected instead of clamped.
No physical M3 or image-quality claim follows from that selection.

## Checks, remaining tester work and stop

F13 shared error words:0 checked pixels,1 raw specular record errors,
2 raw probe faults,3 filtered/sticky probe faults,4 composite faults,
6 invalid raw/output values. Per-signal denoise checks also cover history
domain/length/moments and output. Capture faults are stored as alpha=-1 BEFORE
sanitizing RGB, then validated over EVERY face pixel; reflection faults survive
via the metadata high-bit BEFORE zero. Raw moment/check controls remain
independent of plausible black final output. A safe negative history kernel
skips reset/uninitialized history and corrupts only a previous valid view ID.

Root registers new.cpp/.metal dependencies and harvests the real descriptors;
TESTER performs C++/MSL/Metal syntax, pipeline/startup/archive/hot reload,
raw/SSR/RT/probe-only and GTAO/RTAO, actual/cooked/analytic sources, every
face/mip/parallax direction, custom/raw/SDK fallback, four signals/two views,
cut/resize/DRS/history corruption, finite overflow/capture fault sticky recovery,
1 versus8 reflection samples, missing RT/GI, material occlusion includingzero,
no AO double-count and physically authored reference comparison. Frame-slot
checks need real GPU completion. Memory/residency/lifetime/zero steady-state
GPU allocations and timings remain entirely unmeasured.

## Incremental authored negative hooks and always-on state checks

These hooks depend on the root CLI/report commit4e9773f, which adds
LaunchOptions.debugReflectionCorrupt. Codes1..3 mean history,motion,normal.
Root requires debugLighting for authored controls and custom denoise for
history. They are distinct from the older F8 guide/motion/history controls.

Code1 writes a FOREIGN VIEW into an actually produced valid NEXT history
record after temporal/atrous and before the independent checker. It does not
set an artificial error flag or counter: denoise_check must observe the real
domain mismatch. The old debugHistoryCorrupt still poisons PREVIOUS history
to test rejection; it is skipped on reset/uninitialized history. Both passes
own different argument tables even if requested together.

Code2 copies pre-resolve motion into an F13-owned RG32 texture and injects
quiet NaN at valid receiver pixels. Code3 copies DISurfaces into an F13-owned
per-slot buffer and injects quiet NaN in geometric AND shaded normals, leaving
background and original Direct/F8 resources unchanged. Reflections, AO,
custom denoise and composite consume those actual owned negative inputs.
reflection_input_check inspects the consumed values before any sanitization
and counts actual malformed valid receivers in F13 error word5; it does not
use the requested control code as its failure condition.

Denoise numeric buffers, clear and independent checker now run ALWAYS for
each defined signal, not only under debugLighting. F13 check must be consumed
on completed submitted F13 slots regardless of future pipeline readiness or
debug cadence; the root records per-slot F13 submission and reads failures
outside its cadence-only block. Numerical faults in moments/history flags can
therefore not be hidden by finite final RGB. Authored portable fixtures in
test_surface_signal_controls.cpp compare actual bad-domain/receiver counts,
nonfinite motion rejection, normal transport/history faults and finite-HDR
moment overflow; they do not search strings or merely assert a CLI code.
All controls/tests remain NON EXECUTED and require the tester's actual images,
readbacks, rejection recovery and zero-baseline-fault runs.

No shader/host compile or test result is asserted. This handoff is source
delivery only; F13 acceptance and F14 integration are the root's separate work.
