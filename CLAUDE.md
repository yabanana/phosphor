# Phosphor — notes for AI coding sessions

Native **Metal 4** renderer for Apple silicon (C++20 + metal-cpp, MSL shaders).
Design rationale: `reports/Engine AAA nativo per Apple Silicon.md`.
Implementation plan (phases F0–F38, tasks, exit criteria, optimisation rules
O1–O12): `docs/ROADMAP.md` — cite task IDs (e.g. `F7.3`) in commits, tick a
task only once it is verified on the device, and log measurements in
`docs/perf-log.md`. The report's older "F1–F5" numbering is superseded by it.
After each era come OPT phases (OPT-0…OPT-15): optimisation only, driven by
research (`docs/RESEARCH_REFERENCES.md`, keys `[Rn]`) and by the SoC playbook
(`docs/APPLE_SOC_PLAYBOOK.md`, item IDs like `S-TBDR-3`, benchmarks `B-xx`).
Record every OPT spike, successful or not, in `docs/opt-log.md`. Never state
an undocumented hardware number as fact: measure it with `bench/`.
The Vulkan renderer in `legacy/vulkan/` is reference only — never build or
extend it; port algorithms from it.

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
- F4 observability (macOS): the report JSON (`--report`, schema v2) has GPU
  time per timed unit (`passes`) with `--frames`; `tools/bench_all.sh
  [--stats] [--passes] [--vsync]`, `tools/perf_record.sh` (appends to
  `docs/perf-history*.csv`), `tools/perf_table.py latest|compare|passes`,
  `tools/tracy_check.sh build/tracy` (build with `-DPHOSPHOR_TRACY=ON`;
  Tracy runs on demand: events only while a profiler is connected),
  `tools/gpu_trace.sh` (headless Metal System Trace per pass; no hardware
  counters exist headless), `--gpu-capture*` (.gputrace).  Compare per-pass
  times at saturated clocks (`--no-vsync --gpu-timing-serial`): with vsync
  the GPU lowers its clocks (DVFS) and times do not scale with work.
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
- Reverse-Z infinite projection (clear depth 0, compare Greater), NDC y up.
- Hardware floor is Apple9 (M3); anything needing Apple10 (M5) must have a
  fallback or be an explicitly higher tier.

## Conventions

- Match the surrounding style: 4-space indent, `camelCase_` members, `LOG_*`
  macros, `u32`-style aliases from `core/types.h`.
- New portable logic gets doctest coverage in `tests/`.
- User-facing docs and research are partly in Italian; code and comments in English.
