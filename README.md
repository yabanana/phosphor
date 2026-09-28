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

The full implementation plan — 38 phases in seven eras, each with tasks, exit
criteria and a cross-cutting ultra-optimisation track — is in
[`docs/ROADMAP.md`](docs/ROADMAP.md) (Italian). After every era, dedicated
**OPT phases** (OPT-0…OPT-15) do nothing but optimise, using research
([`docs/RESEARCH_REFERENCES.md`](docs/RESEARCH_REFERENCES.md)) and a
component-by-component guide to squeezing Apple silicon
([`docs/APPLE_SOC_PLAYBOOK.md`](docs/APPLE_SOC_PLAYBOOK.md)).

| Era | Phases | Content |
|---|---|---|
| I · Foundations | F0 ✅, F1–F4 | Metal 4 context, memory/heaps, render graph with automatic barriers, async/AOT pipelines, profiling |
| II · GPU-driven geometry | F5–F8 | Persistent GPU scene, GPU-built ICBs, mesh shaders + two-phase culling, visibility buffer, HDR/EDR, MetalFX |
| III · Light | F9–F14 | Ray tracing infrastructure, hybrid shadows, ReSTIR DI, radiance-cache GI, reflections, atmosphere and clouds |
| IV · World | F15–F22 | Advanced materials, on-tile OIT, particles, post, virtualised geometry, terrain, water, cooker, streaming |
| V · Simulation (parallel) | F23–F27 | Job system and CPU optimisation, physics, animation, ray-traced acoustics, gameplay runtime |
| VI · Frontier | F28–F33 | Scalability, per-device autotuning, neural rendering (incl. in-house MLX-trained networks), path tracing, Gaussian splatting |
| VII · Product | F34–F38 | Editor, automated QA, iPad/iPhone/visionOS, vertical slice, distribution |

---|---|---|
| **F0 — Foundations** | Portable core, Metal 4 context and frame loop, bindless textures, forward PBR pass, ImGui, CI | **in progress** |
| F1 — RHI and render graph | Heaps, transient aliasing, stage-to-stage barriers, async pipeline compilation | planned |
| F2 — GPU-driven geometry | Object/mesh shaders, two-phase Hi-Z culling, visibility buffer, MetalFX temporal | planned |
| F3 — Ray tracing | BLAS/TLAS, RT shadows, ReSTIR DI, DDGI in compute, MetalFX denoiser | planned |
| F4 — Content scale | Cluster LOD (meshoptimizer `clusterlod`), MTLIO streaming, pipeline archives | planned |
| F5 — High tiers | Frame interpolation, radiance cache + ReSTIR GI, neural features on M5 | planned |

---

## Requirements

- Apple silicon Mac, **macOS 26** or later (Metal 4). M3 or later recommended.
- Xcode 26 with the Metal toolchain
  (`xcodebuild -downloadComponent MetalToolchain` if `xcrun metal` is missing).
- CMake 3.25+ and Ninja (`brew install cmake ninja`).

All other dependencies (SDL3, glm, meshoptimizer, tinygltf, Dear ImGui,
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
| 1 – 7 | Switch bench |
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
               scene extraction, gpu_types.h (shared with MSL)
  testbench/   the seven benches
  platform/metal/  Metal 4 backend: context/frame loop, textures, forward pass
  app/         Engine (SDL3 window + Metal layer, main loop, UI)
  imgui/       debug panels
shaders/       MSL (compiled to build/shaders/phosphor.metallib)
tests/         doctest unit tests for the portable core
tools/         apple-sdk-stubs for Linux syntax checks
legacy/vulkan/ archived Vulkan renderer (reference only)
reports/, research_notes/   design research (Italian / English)
```

## License

No license has been chosen yet; all rights reserved by the author.
