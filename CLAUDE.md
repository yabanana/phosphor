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
  elsewhere. Insert explicit stage-to-stage barriers (no hazard tracking).
- Benchmarks report `GPU allocations` during the measured frames: it must be
  0 (rule O7).
- Builds are signed with `get-task-allow` (`PHOSPHOR_DEBUGGABLE`), so the
  agent can profile without root: `leaks --atExit -- ./build/release/phosphor
  ...`, `heap <pid>`, `MallocStackLogging=lite` + `malloc_history <pid>
  -allByCount` (diff two snapshots per stack), `xcrun xctrace record
  --template Allocations --launch -- ...`.
- Frame pacing: one MTL4 command buffer per frame (scene + ImGui overlay in
  the same render pass) and one `MTLSharedEvent`; value `n+1` = frame n done.
  `makeResident` allocations are committed right before each commit, so they
  can be used by the frame being recorded.
- Engine conventions follow glTF: counter-clockwise front faces (set
  explicitly: Metal defaults to clockwise), UV origin top-left, bitangent =
  `cross(N, T) * w` towards decreasing V. Mirrored instances carry
  `INSTANCE_FLAG_MIRRORED`. `tests/test_procedural.cpp` enforces the meshes.
- Reverse-Z infinite projection (clear depth 0, compare Greater), NDC y up.
- Hardware floor is Apple9 (M3); anything needing Apple10 (M5) must have a
  fallback or be an explicitly higher tier.

## Conventions

- Match the surrounding style: 4-space indent, `camelCase_` members, `LOG_*`
  macros, `u32`-style aliases from `core/types.h`.
- New portable logic gets doctest coverage in `tests/`.
- User-facing docs and research are partly in Italian; code and comments in English.
