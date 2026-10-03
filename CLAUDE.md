# Phosphor — notes for AI coding sessions

Native **Metal 4** renderer for Apple silicon (C++20 + metal-cpp, MSL shaders).
Design rationale: `reports/Engine AAA nativo per Apple Silicon.md`.
Implementation plan (phases F0–F41, tasks, exit criteria, optimisation rules
O1–O12): `docs/ROADMAP.md` — cite task IDs (e.g. `F7.3`) in commits, tick a
task only once it is verified for its declared scope on the available device
(per `docs/plans/HARDWARE_VALIDATION.md`), and log measurements in
`docs/perf-log.md`. The report's older "F1–F5" numbering is superseded by it.
Planning horizons: read `docs/plans/SEQUENCING.md` and `docs/plans/README.md`.
All 58 F/OPT plans now contain advance implementation and verification detail.
F5–F8 implementation is integrated on main (F6 merge e600887; F7/F8 PR #14,
merge 91b51c2). F7 baseline has M5 development acceptance; F8.4 retains an
open MetalFX SDK lifetime gate (see `docs/F7_F8_HANDOFF.md`). Integration
does not close that failed check; native remains the default. The SDK/API
comparison and rejected denoised experiment are recorded in
`docs/research/2026-10-03-metalfx-lifetime.md`. The owner requested a stop before OPT.
F7.4/F7.5 were implemented as opt-in experiments and not adopted in the
measured preset. Material binning is available; generic resolve is the measured
default. `docs/RENDERING_F7_F8.md` records contracts and validation commands. Later
plans remain revisable specifications, and open OPT/EDGE work is a candidate
catalog: planning detail does not activate implementation. Before adding optimization
infrastructure, demonstrate a bottleneck on real engine workloads, compare a
simple baseline, timebox one experiment and adopt only a measured benefit.
Read the matching phase plan and `docs/plans/METHOD.md` when promoting it.
The roadmap owns task status; unselected candidates stay unchecked and do not
block the next functional phase. Report delivered scope and outstanding
candidates explicitly; keep correctness and validation of selected work mandatory.
Required product capabilities: editor, ECS/runtime, community content, 2D,
game UI and ecosystem reuse (`docs/plans/PRODUCT_PLATFORM.md`, F39–F41).
Prefer evaluating actual Bevy crates/App in a Rust host with a measured batched
bridge to the C++ Metal renderer before writing equivalents. A Bevy-inspired
C++ ECS does not make Rust/wgpu plugins compatible; verify versions, adapters
and behavior. These functional requirements do not need an FPS improvement
to justify implementation. Preserve the integrated F5 contracts; Vulkan is a distant
unscheduled hypothesis only, with no legacy revival or speculative RHI work.
SoC research
and experimental hypotheses: `docs/research/2026-10-01-apple-soc.md`.
After each era, select relevant OPT tasks (OPT-0…OPT-15) from measured needs, driven by
research (`docs/RESEARCH_REFERENCES.md`, keys `[Rn]`) and by the SoC playbook
(`docs/APPLE_SOC_PLAYBOOK.md`, item IDs like `S-TBDR-3`, benchmarks `B-xx`).
Record every OPT spike, successful or not, in `docs/opt-log.md`. Never state
an undocumented hardware number as fact: measure it with `bench/`.
The Vulkan renderer in `legacy/vulkan/` is reference only — never build or
extend it; port algorithms from it.

## Available hardware and acceptance (owner decision, 2026-10-02)

Only M5 Max 128 GB is available. `DEVELOPMENT_ACCEPTED` on this device permits
integration and the next phase; physical T0/other-device certification is
`EXTERNAL_VALIDATION_PENDING` and does not block development. Preserve the
Apple9 feature floor and test relevant fallbacks, but never present a forced
Apple9 path or a 16 GB application budget on M5 as M3/Base measurements.
Read `docs/plans/HARDWARE_VALIDATION.md` for the ledger and reporting rules.
Explicit multi-device measurement tasks (F0.8, OPT-0.2, F28.7, F37.2) remain
partial/unticked until their actual measurements exist. Bugs or failed checks
on M5 are not hardware exemptions. Since F6 the app has `--force-family apple9`
(effective capabilities only) and reports physical and effective families
separately (report schema 7 `hardware`, plus `rendering`).
Spike results choose variants within the authorized phase; do not invent
measurements or introduce another routine owner-approval gate before coding.

## Non-negotiable working rules (set by the project owner)

1. **Severe residue audit before declaring a phase done.** Before saying an
   `Fxx` or `OPT-xx` phase is finished, re-read every task line of that phase
   in `docs/ROADMAP.md` literally, the approved plan, the exit criteria and
   every "later"/"arrives with Fx" note or TODO left in code and docs, and
   check each against evidence (tests, validation runs, measurements).
   Anything partial is reported as partial and either completed or left
   explicitly unticked. Intermittent failures are never "passes": reproduce
   and explain them first.
2. **Autonomy: do not hand actions to the owner.** Do the work yourself,
   including verification. When a step seems to need the owner's
   interaction or `sudo`, find another non-privileged method (e.g. sign dev
   builds with `get-task-allow` so profiling tools attach without root, use
   in-process measurements, simulate events). Never try to bypass the
   harness permission system; if no alternative exists, say so explicitly.
3. **Chat in Italian.** Code, comments and commit messages stay in English.
4. **Detailed handoff at the end of every phase**: what was done (per task
   ID), how it was verified (commands and numbers), bugs found, deviations
   from the plan, what is still open and why, risks, and the exact next
   steps. Put it in the PR description and in the final chat message.

## Build / verify

- macOS (real target): `cmake -S . -B build -G Ninja && cmake --build build && ./build/phosphor`
- Linux / cloud sessions (no Mac, no GPU):
  ```bash
  cmake -S . -B build/linux -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  cmake --build build/linux && ctest --test-dir build/linux
  cmake --build build/linux --target metal_syntax_check
  ```
  `metal_syntax_check` type-checks every file in `src/platform/metal`, `src/app`
  and `src/imgui` against the real metal-cpp headers with the stubs in
  `tools/apple-sdk-stubs`. Run it after touching Metal host code. It cannot
  catch runtime errors, and **MSL shaders are only compiled on macOS** (CI job
  `app-macos`), so be extra careful with `.metal` edits and say so when they are
  unverified.
- Visual/validation check on macOS: `tools/visual_check.sh build build/reference`
  (all benches under API + shader validation, pixel diff against references
  taken with `--update` before a change). Bench switching:
  `./build/phosphor --frames 300 --warmup 0 --switch-every 20` with the same
  environment variables.
- F3 pipeline checks (macOS): `tools/archive_check.sh build/release <ref>`
  (archive miss scenarios, API validation only), `tools/hot_reload_check.sh`
  (Debug), `tools/hitch_check.sh build/release` (bench-switch hitches),
  `tools/variant_check.sh build/release` (every forward variant against the
  generic pipeline, pixels), `tools/harvest_pipelines.sh` (regenerate
  `shaders/pipelines.mtl4-json` when pipeline descriptors change; the
  `pipelines script` unit test fails when it is stale).
- F4 observability (macOS): the report JSON (`--report`, schema v3; v2 +
  per-pass `work` since OPT-0.4) has GPU
  time per timed unit (`passes`) with `--frames`; `tools/bench_all.sh
  [--stats] [--passes] [--vsync]`, `tools/perf_record.sh` (appends to
  `docs/perf-history*.csv`), `tools/perf_table.py latest|compare|passes`,
  `tools/tracy_check.sh build/tracy` (build with `-DPHOSPHOR_TRACY=ON`;
  Tracy runs on demand: events only while a profiler is connected),
  `tools/gpu_trace.sh` (headless Metal System Trace per pass; no hardware
  counters exist headless), `--gpu-capture*` (.gputrace).  Compare per-pass
  times at saturated clocks (`--no-vsync --gpu-timing-serial`): with vsync
  the GPU lowers its clocks (DVFS) and times do not scale with work.
- OPT-0 SoC suite (macOS, a measurement tool, not engine code):
  `soc_bench` (`bench/soc`, README there): `--list`, `--only B-08`,
  `--quick`, `--runs N`, `--validate`, `--force-family apple9`, `--window`
  (B-26), `--soak MIN` (B-27); exit battery `tools/soc_bench_all.sh` (x3,
  validation, Apple9 paths, leaks; keeps the display on with caffeinate).
  Results `bench/results/<chip>-<os>.json`; cost model
  `src/diagnostics/soc_model.h` + CLI `soc_model`, `docs/soc-model.md`
  (`tools/soc_model.py`), roofline `tools/soc_roofline.sh`
  (`tools/air_ops.py --forward-variant`, `tools/roofline.py`).  Measured
  traps: uniform or identical ALU chains are computed once per SIMD-group
  or merged (seed per thread and chain), use random input data (zeros read
  25% faster in the spike, 1.00x in B-08 at 16 MiB), a small render pass between compute encoders costs either
  ~6-15 or ~60 us (compare variants on per-round minima), single-thread
  chases run with the GPU fabric (IOReport AFR) at a low state, the display
  must stay on.  Every benchmark proves its negative control can fail.
- OPT-1 graph scenarios and plans (macOS): `--graph-scenario N` (0 deferred,
  1 forward-plus, 2 post-chain, 3 async-compute; `--graph-scenario-views N`,
  `-size`, `-work`, `-wide`, `--graph-remat`, `--graph-order`) replaces the
  scene with a portable graph of synthetic passes (`rendergraph/scenario.h`,
  `shaders/scenario.metal`) whose image is identical for any correct
  schedule: compare modes with `image_diff` (0 pixels).  `--graph-opt
  off|greedy|plan` (default off = end-of-F4 compiler) and `--graph-plan`;
  report schema 4 has a `graph` object.  Plans: `build/release/graph_opt
  --top K --top-dir D` (candidates; also `--gamma/--beta` for memory/bytes),
  `tools/graph_select.py --candidates D` (adopts by measurement into
  `shaders/graph-plans.json`), `tools/graph_scenarios.sh` (off/greedy/plan
  table, pixels, validation).  The cost model ranks but under-predicts
  latency-bound passes: never adopt a plan without measuring it.  Measure
  only on a quiet machine (no agent, solver or other benchmark running:
  frame times become bimodal).
- F5 GPU scene (macOS): `--gpu-driven off|on` (default on; off = one CPU draw
  per bucket, same image), bench 8 `--instances N --scene-meshes K
  --dynamic-cpu PCT --churn N`, `--cull-distance D --cull-min-pixels P`,
  `--debug-gpu-scene N` (exact readback self-check every N frames, exit 1 on
  FAIL) with `--debug-gpu-scene-corrupt delta|plane|command|touch` (each must
  FAIL); report schema 5 (`scene`, `cpu_phases`) and the `SCENE` line.
  Kernel checks: `f5_spike --only F5-K2,F5-K3` (`bench/f5_spike`).
  `visual_check` covers bench 8 (`BENCH8_ARGS`); captures are only
  comparable in the same mode: Debug vs Release metallibs differ by 1 pixel
  (bench 1) and running with vs without the validation layers by 9 pixels
  (bench 6) -- `archive_check` needs Release references
  (`tools/visual_check.sh build/release <dir> --update`).  Comparing GPU
  times of builds with different CPU cost per frame needs equal CPU time
  (`--gpu-timing-serial` is DVFS-sensitive: measured 2.8 vs 1.8 ms for the
  same forward pass).
- F6 mesh path (macOS): `--geometry-path indexed|mesh` (default indexed, the
  F5 reference; mesh needs `--gpu-driven on`), `--meshlet-cull
  off|frustum|two-phase` (default two-phase), `--hiz-path auto|compute|sampler`
  (auto = compute SIMD-group; sampler needs effective Apple10),
  `--force-family apple9` (EFFECTIVE capabilities only: physical device,
  memory and budget unchanged; report schema 6 `hardware`), `--debug-meshlets
  N` (CPU-reference check of candidates, every decision, B list, pyramids and
  lost surfaces) with `--debug-meshlets-corrupt id|depth|count` (each must
  FAIL), `--debug-view meshlets|cull|hiz`, `--culling-script` (bench 7 F6
  scenario = the gate preset), `--resolution WxH`, cook options
  `--meshlet-builder`/`--meshlet-max-*`, options `--meshlet-min-pixels`
  (approximate) and `--meshlet-triangle-cull on`.  Battery:
  `tools/f6_check.sh build build/release [--quick] [--perf]` (references
  `build/reference-f6base` = indexed, `build/reference-f6mesh` = mesh off for
  bench 8); spikes `f6_spike --only F6-S3|F6-S4` (`bench/f6_spike`), cook
  `meshlet_cook`.  The app prints `EXIT <code>` last: a shell status that
  differs means a signal.
- Before calling a Metal API, check its exact signature in the fetched
  metal-cpp headers (`build/linux/_deps/metal_cpp-src/Metal/MTL4*.hpp`); Metal 4
  names differ from Metal 3 (e.g. no `setVertexBytes`, draws take GPU addresses,
  queue labels are set through the descriptor).

## Architecture rules

- `phosphor_core` (src/core, scene, renderer, testbench, diagnostics) must stay
  free of Metal/Apple headers so it builds and is unit-tested on Linux.
- GPU struct layouts live once in `src/renderer/gpu_types.h`, shared with MSL:
  scalars only (MSL `float3` is 16-byte aligned), keep the `static_assert`s.
- Metal 4 does not retain or make resources resident: create every GPU buffer
  or texture through `context.memory()` (`GpuMemory`: labels, residency,
  per-category accounting, deferred release), write per-frame data into
  `context.frameUploads()` and loading-time data through `stagingAllocate` /
  `enqueueUpload` / `flushUploads`. Never call `device->newBuffer/newTexture`
  elsewhere. Frame-lifetime intermediates go in `TransientHeap` at offsets
  chosen by the render graph (aliasing: first use after another resource
  needs a barrier with `VisibilityOptionResourceAlias`). Insert explicit
  stage-to-stage barriers (no hazard tracking).
- Benchmark mode (`--frames`) ignores keyboard/mouse input: the window takes
  focus at launch; `tools/visual_check.sh` injects input to prove it.
- Benchmarks report `GPU allocations` during the measured frames: it must be
  0 (rule O7).
- Builds are signed with `get-task-allow` (`PHOSPHOR_DEBUGGABLE`), so the
  agent can profile without root: `leaks --atExit -- ./build/release/phosphor
  ...`, `heap <pid>`, `MallocStackLogging=lite` + `malloc_history <pid>
  -allByCount` (diff two snapshots per stack), `xcrun xctrace record
  --template Allocations --launch -- ...`.
- The frame is a render graph (`src/rendergraph/`, portable; executed by
  `MetalGraphExecutor`): add passes in `Engine::buildFrameGraph` declaring
  every access with its exact stages, never encode passes or barriers by
  hand. The graph is compiled only when its key changes (drawable size, UI,
  capture, debug flags). Barrier rules come from the F2.3 spike
  (`barrier_plan.h`): Fragment is fine on the consumer side of a queue
  barrier, Tile synchronises nothing, no fragment/tile producer inside a
  render encoder. Import resources that change every frame (drawable,
  per-slot buffers) with `ImportPerFrame`; any other import written by the
  graph is treated as persistent and its first access waits for the previous
  frame. Passes fused in one render encoder share its state: leave
  Metal's defaults (cull none, clockwise winding) when you change them, the
  validation layer rejects redundant state.
- Graph optimiser (OPT-1, `src/rendergraph/optimizer/`, portable): a plan is
  keyed by family + structural `graphKey` and applied only when the built
  graph matches (else logged fallback); plans name passes, so renaming a
  pass or changing its accesses invalidates them (regenerate with
  `graph_opt` + `graph_select.py`).  `AliasPolicy`/`BarrierPolicy`/
  `LintMode` in `CompileOptions`; `BarrierPolicy::Minimal` is proven by
  argument and tests, not by a GPU race (OPT-1.4), so the engine default
  stays Conservative.
- Pipelines (F3): create every render/compute pipeline through
  `PipelineCache::request(pipe::PipelineDesc)` and resolve the handle while
  encoding (`render(h)` / `compute(h)`); never call an `MTL4::Compiler`
  elsewhere.  Requests compile on utility-QoS threads (archive lookup →
  flexible fallback → full variant) and become visible only at
  `PipelineCache::beginFrame()`, so the cached graph never recompiles for
  them.  Forward variants are declared in `shaders/variants.def` (generated
  C++/MSL tables); a variant must only remove branches that are dead for the
  scenes that select it (images stay identical).  The archive comes from the
  committed `shaders/pipelines.mtl4-json` (`tools/harvest_pipelines.sh`
  regenerates it when pipeline names/constants/state change) through
  `metal-tt` at build time; Metal rejects archive lookups under
  `MTL_SHADER_VALIDATION`, so archive checks (`tools/archive_check.sh`) run
  with API validation only.  Self-checks: `--pipeline-sync`,
  `--debug-pipeline-fallback`, `--debug-hot-reload`, `--no-pipeline-archive`.
- Frame pacing: normally one MTL4 command buffer per frame (scene + ImGui
  overlay fused in one render pass) and one `MTLSharedEvent`; value `n+1` =
  frame n done. With `--debug-split-encoding` a render pass is suspended/
  resumed across several command buffers (one commit); with
  `--debug-async-compute` the frame is several submissions on two queues
  synchronised by per-queue timeline events. `makeResident` allocations are
  committed right before each commit, so they can be used by the frame
  being recorded.
- Debug self-checks (all must stay green in `tools/visual_check.sh` with
  `EXTRA_ARGS=...`): `--debug-graph-transients`, `--debug-split-encoding`,
  `--debug-async-compute`; `--resize-every N` exercises recompilation;
  `--dump-graph FILE` writes the Graphviz dump with DRAM estimates;
  `--debug-gpu-cost N` (known-cost pass, negative control of the pass
  timings), `--gpu-timing-unfused`, `--gpu-timing-serial`, `--no-gpu-timing`.
- GPU timing (F4.1, `GpuTimestamps` via `MetalGraphExecutor`): only END
  timestamps (encoder-start ones are written late) plus a commit-start
  timestamp in an "anchor" compute encoder (1-thread dispatch + producer
  barrier; never `writeTimestampIntoHeap`, whose driver bookkeeping grows
  forever on reused command buffers, nor an encoder without a dispatch,
  which is dropped); a fused render group is one timed unit
  (the GPU runs its passes per tile together; at most 4 timestamps are
  written per render encoder); a unit's time is its exclusive contribution
  to the queue timeline, so passes that overlap on the GPU share it.  Never
  call `invalidateCounterRange` on a range while frames are in flight (it
  wipes later writes).  GPU captures need the capture layer before
  `SDL_Init`, the graphics MTL4 queue as capture object and no
  `MTL4Archive` (all measured).
- Engine conventions follow glTF: counter-clockwise front faces (set
  explicitly: Metal defaults to clockwise), UV origin top-left, bitangent =
  `cross(N, T) * w` towards decreasing V. Mirrored instances carry
  `INSTANCE_FLAG_MIRRORED` and are drawn in their own batches with front-face
  culling; back faces are culled (glTF `doubleSided` materials are not).
  `tests/test_procedural.cpp` enforces the meshes.
- GPU scene (F5): instance, material, node and motion data reach the GPU
  only through `SceneStore` deltas (`renderer/scene_store.h`); benches change
  components through mutable ECS access (`getComponent`/`modify` mark them
  changed, `const` access never does) and the engine calls `ECS::endFrame()`
  after the sync; EntityIDs are recycled.  The vertex shaders read
  `instances[visible[instance_id]]`.  Scene passes (`Scene update`, `Scene
  transforms`, `Instance cull`, `Draw build`) chain their own dispatches with
  Dispatch->Dispatch encoder barriers; a draw that consumes indirect
  arguments or an ICB written by compute waits at the **Vertex** stage
  (Fragment/Object/Mesh do not synchronise, spike S5).  ICBs: one command per
  bucket, three fixed ranges per cull class, the state set by the CPU; never
  `executeCommandsInBuffer` inside a render encoder resumed in another
  command buffer (GPU fault and recovery every frame) and never an indirect
  execution range (shader validation aborts).  A split render pass (F2.5) is
  a separate commit behind a fence: barriers do not order resumed pieces
  after earlier work.  Shared C++/MSL math that must match the CPU bit for
  bit uses `fp contract(off)` (`cull_math.h`, `transform_math.h`).  Pipelines
  that draw the scene set `PipelineDesc::indirectCommandBuffers`.
- GPU safety: every kernel loop has a hard bound, no kernel waits on another
  threadgroup, command buffers stay far below 1 s, and one GPU test process
  at a time (a 60 s job made the WindowServer watchdog kill the compositor).
- Mesh path (F6, `platform/metal/mesh_renderer`, `hiz_builder`, contract
  `renderer/meshlet_layout.h`, maths `renderer/meshlet_cull_math.h` shared
  with the CPU reference): passes Meshlet candidates (between Instance cull
  and Draw build) → Forward (phase A) → Hi-Z A → Meshlet B → Forward B → Hi-Z
  final; three indirect mesh draws per phase (one per cull class, F5 cull
  states).  Mesh indirect arguments and object-shader reads of compute data
  order at **Object|Mesh**, never Vertex (spike S4, the opposite of the F5
  ICB rule); object writes → compute: after Object|Mesh.  The F5 ICB stays
  in phase A as the overflow fallback (Draw build gated by the meshlet
  overflow word): never truncate a list.  The Hi-Z history is only a hint
  (every history rejection is retested against the current depth), one
  persistent pyramid read by phase A and rewritten by Hi-Z final.  Culling
  tests must stay conservative: sphere radius from the Gershgorin
  spectral-norm bound (`cullScaleBound`, also F5), cone test in mesh space
  (camera through M^-1), footprint/near-plane rules in the header; a change
  needs the property tests and `--debug-meshlets` green.  Passes that may
  record nothing in normal frames must not be in the graph (Metal drops an
  empty encoder and the next timed unit loses its timestamp).  Mesh grids
  without an object stage need 2D grids (a 1D grid of ~700K threadgroups
  drew silently part of the scene).
- Reverse-Z infinite projection (clear depth 0, compare Greater), NDC y up.
- Hardware floor is Apple9 (M3); anything needing Apple10 (M5) must have a
  fallback or be an explicitly higher tier.

## Conventions

- Match the surrounding style: 4-space indent, `camelCase_` members, `LOG_*`
  macros, `u32`-style aliases from `core/types.h`.
- New portable logic gets doctest coverage in `tests/`.
- User-facing docs and research are partly in Italian; code and comments in English.
