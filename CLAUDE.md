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
- Before calling a Metal API, check its exact signature in the fetched
  metal-cpp headers (`build/linux/_deps/metal_cpp-src/Metal/MTL4*.hpp`); Metal 4
  names differ from Metal 3 (e.g. no `setVertexBytes`, draws take GPU addresses,
  queue labels are set through the descriptor).

## Architecture rules

- `phosphor_core` (src/core, scene, renderer, testbench, diagnostics) must stay
  free of Metal/Apple headers so it builds and is unit-tested on Linux.
- GPU struct layouts live once in `src/renderer/gpu_types.h`, shared with MSL:
  scalars only (MSL `float3` is 16-byte aligned), keep the `static_assert`s.
- Metal 4 does not retain or make resources resident: add long-lived allocations
  with `MetalContext::makeResident`, release in-flight objects with
  `deferRelease`, and insert explicit stage-to-stage barriers (no automatic
  hazard tracking).
- Frame pacing: one `MTLSharedEvent`; values `2n+1` scene done, `2n+2` frame done.
  ImGui still renders through a Metal 3 queue ordered after the scene.
- Reverse-Z infinite projection (clear depth 0, compare Greater), NDC y up.
- Hardware floor is Apple9 (M3); anything needing Apple10 (M5) must have a
  fallback or be an explicitly higher tier.

## Conventions

- Match the surrounding style: 4-space indent, `camelCase_` members, `LOG_*`
  macros, `u32`-style aliases from `core/types.h`.
- New portable logic gets doctest coverage in `tests/`.
- User-facing docs and research are partly in Italian; code and comments in English.
