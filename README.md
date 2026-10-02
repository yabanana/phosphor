# Phosphor

**Phosphor** is a real-time renderer being rebuilt as a **native Metal 4 engine
for Apple silicon**, with the goal of AAA-grade rendering on Mac (and later
iPad).  It is written in C++20 on top of [metal-cpp](https://github.com/apple/metal-cpp).

The direction, and the research behind it, is in
[`reports/Engine AAA nativo per Apple Silicon.md`](reports/Engine%20AAA%20nativo%20per%20Apple%20Silicon.md)
(Italian).  In short:

- **Metal 4 programming model** end to end: `MTL4CommandQueue`, per-frame
  command allocators, `MTLSharedEvent` frame pacing, residency sets, argument
  tables, `MTL4Compiler`.
- **Designed for TBDR** and unified memory: memoryless attachments, no depth
  pre-pass, a thin visibility buffer (next phase), bandwidth measured in bytes
  per pixel.
- **Hardware floor: Apple9 (M3/M4)** so hardware ray tracing, mesh-shader ICBs,
  64-bit atomics and the MetalFX denoiser are always available.  The app still
  starts on M1/M2 (Apple7+) with a warning.
- **Ray tracing in compute** (Metal forbids RT inside mesh-shader pipelines),
  radiance caching for GI, MetalFX for reconstruction.

The original Vulkan renderer is archived in [`legacy/vulkan/`](legacy/vulkan/README.md).

---

## Status and roadmap

The implementation roadmap — F0–F41 and OPT-0…OPT-15, organized by dependencies
in seven eras — is in [`docs/ROADMAP.md`](docs/ROADMAP.md) (Italian).
The product scope includes an editor, ECS/runtime, community content, complete
2D and game UI, and verified Bevy ecosystem reuse. These are planned capabilities;
see [`docs/plans/PRODUCT_PLATFORM.md`](docs/plans/PRODUCT_PLATFORM.md).
Metal is the active renderer; a future Vulkan backend remains unscheduled.
After each era, select **OPT tasks** justified by real engine measurements, using research
([`docs/RESEARCH_REFERENCES.md`](docs/RESEARCH_REFERENCES.md)) and a
component-by-component guide to squeezing Apple silicon
([`docs/APPLE_SOC_PLAYBOOK.md`](docs/APPLE_SOC_PLAYBOOK.md)).

| Era | Phases | Content |
|---|---|---|
| I · Foundations | F0 ✅, F1–F4 | Metal 4 context, memory/heaps, render graph with automatic barriers, async/AOT pipelines, profiling |
| II · GPU-driven geometry | F5 ✅, F6 ✅, F7–F8 | Persistent GPU scene, GPU-built ICBs, mesh shaders + two-phase culling, visibility buffer, HDR/EDR, MetalFX |
| III · Light | F9–F14 | Ray tracing infrastructure, hybrid shadows, ReSTIR DI, radiance-cache GI, reflections, atmosphere and clouds |
| IV · World | F15–F22 | Advanced materials, on-tile OIT, particles, post, virtualised geometry, terrain, water, cooker, streaming |
| V · Simulation (parallel) | F23–F27 | Job system and CPU optimisation, physics, animation, ray-traced acoustics, gameplay runtime |
| VI · Frontier | F28–F33 | Scalability, per-device autotuning, neural rendering (incl. in-house MLX-trained networks), path tracing, Gaussian splatting |
| VII · Product | F34–F41 | Editor, automated QA, Apple platforms, vertical slice, distribution, complete 2D/game UI and ecosystem reuse |


---

## Requirements

- Apple silicon Mac, **macOS 26** or later (Metal 4). M3 or later recommended.
- Xcode 26 with the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain` if `xcrun metal` is missing).
- CMake 3.25+ and Ninja (`brew install cmake ninja`).

All other dependencies (SDL3, glm, meshoptimizer, tinygltf, MikkTSpace, Dear ImGui,
metal-cpp, doctest) are fetched and pinned by CMake.

## Build and run

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build
./build/phosphor            # or: ./build/phosphor --bench 4
ctest --test-dir build      # unit tests for the portable core
```

For GPU captures and shader debugging, generate an Xcode project instead:
`cmake -S . -B build-xcode -G Xcode`.  Debug builds enable the Metal API
validation layer automatically; set `MTL_SHADER_VALIDATION=1` for shader
validation.

### Command line and benchmarks

| Option | Effect |
|---|---|
| `--bench N` | Start on test bench N (1–8) |
| `--frames N` | Benchmark mode: measure N frames, print a `BENCH` summary line, exit |
| `--warmup N` | Frames skipped before measuring (default 120) |
| `--no-vsync` | Uncapped frame rate |
| `--no-ui` | Hide the ImGui overlay |
| `--capture FILE` | Write a PNG of the last measured frame (first frame when interactive) |
| `--report FILE` | Write the benchmark summary as JSON |
| `--fixed-timestep` | Simulate 1/60 s per frame (deterministic captures) |
| `--switch-every N` | Cycle to the next bench every N frames (same path as the 1–8 keys) |
| `--gpu-driven off\|on` | Scene submission: GPU culling + GPU-built indirect command buffers (on, default) or one CPU draw per bucket (off); same image |
| `--instances N`, `--scene-meshes K`, `--dynamic-cpu PCT`, `--churn N` | Bench 8 ("1M Instances"): size, meshes, CPU-updated share, spawn/despawn per frame |
| `--cull-distance D`, `--cull-min-pixels P` | Extra GPU instance culling (off by default) |
| `--debug-gpu-scene N` | Read the GPU scene back every N frames and compare it exactly with the CPU mirror and references (exit 1 on failure) |
| `--memory-stress N` | Create/destroy N GPU resources and check memory returns to baseline (exit 1 on failure) |
| `--simulate-pressure` | Inject memory-pressure warning/critical events |
| `--transient-test` | Aliasing self-test of the transient heap (exit 1 on failure) |
| `--inject-input` | Push synthetic key/mouse events every frame (benchmarks must ignore them) |

Baseline numbers live in [`docs/perf-log.md`](docs/perf-log.md); measure them
on a Release build without validation:

```bash
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release
tools/bench_all.sh build/release   # all benches, Markdown rows for the perf log
```

Visual regression and validation check of every bench (capture references
with `--update` before a change):

```bash
tools/visual_check.sh build build/reference [--update]
```

### Building on Linux (no GPU)

The portable core, its tests and a **syntax check of all Metal host code**
(against the real metal-cpp headers plus minimal SDK stubs in
`tools/apple-sdk-stubs`) run on Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
cmake --build build && ctest --test-dir build
cmake --build build --target metal_syntax_check
```

MSL shaders can only be compiled on macOS; CI does that on a hosted Mac.

---

## Test benches

Press **1–7** or use the ImGui combo box:

| # | Bench | Purpose |
|---|---|---|
| 1 | Torus Demo | Gold torus on a plane; end-to-end sanity check |
| 2 | PBR Material Grid | Metallic × roughness sweep |
| 3 | Stress Test | 100,000 instanced spheres |
| 4 | Scene Viewer | glTF/GLB from `assets/` (see `assets/README.md`) |
| 5 | Many Lights | 1,024 point lights (ReSTIR target) |
| 6 | Cornell Box | GI reference scene |
| 7 | Culling Viz | 10,000 buildings for occlusion culling |

## Controls

| Input | Action |
|---|---|
| W A S D, Q / E | Move, down / up (FPS benches) |
| Right mouse drag | Look (FPS) |
| Left mouse drag, wheel | Orbit, zoom (orbit benches) |
| Shift | Sprint |
| 1 – 8 | Switch bench |
| F1 / F2 / F3 | Lit / normals / base color |
| Esc | Quit |

---

## Layout

```
src/
  core/        types, logging, input, timer
  scene/       ECS, camera, components, procedural meshes, glTF loader,
               TextureManager (API-agnostic front end)
  renderer/    GpuScene (CPU-side scene geometry), meshlet builder,
               SceneStore (CPU mirror of the persistent GPU scene, deltas),
               cull/transform math and references, gpu_types.h and
               gpu_scene_layout.h (shared with MSL)
  testbench/   the eight benches
  platform/metal/  Metal 4 backend: context/frame loop, textures, forward pass
  app/         Engine (SDL3 window + Metal layer, main loop, benchmark mode)
  imgui/       debug panels and the Metal 4 ImGui renderer
shaders/       MSL (compiled to build/shaders/phosphor.metallib)
tests/         doctest unit tests for the portable core
tools/         apple-sdk-stubs for Linux syntax checks, bench_all.sh
legacy/vulkan/ archived Vulkan renderer (reference only)
reports/, research_notes/   design research (Italian / English)
```

## Planning and validation

All 58 F/OPT phase plans now include implementation packages and task-level
acceptance checks: [plan index](docs/plans/README.md). F5 is integrated;
F6–F8 remain the immediate renderer priority. Advance detail does not activate
the optional research catalog.

Development acceptance uses the available M5 Max 128 GB. Apple9 fallback
checks on that device do not certify M3/T0 hardware or performance; missing
physical-device validation remains explicit and does not block development.
See the [hardware policy](docs/plans/HARDWARE_VALIDATION.md).

## License

No license has been chosen yet; all rights reserved by the author.
