# Phosphor -- Vulkan Mesh-Shader Renderer

**Phosphor** is a real-time Vulkan 1.3 renderer built from scratch in C++20.  It
showcases modern GPU-driven rendering techniques: task/mesh shaders, a visibility
buffer, bindless descriptors, hierarchical-Z occlusion culling, ReSTIR direct
illumination, DDGI global illumination, and more.

The engine is structured as a set of self-contained render passes wired together
by a dependency-tracking render graph.  Seven built-in test benches let you
explore different workloads without writing any code.

---

**Phosphor** e' un renderer real-time Vulkan 1.3 scritto da zero in C++20.
Utilizza tecniche di rendering GPU-driven di ultima generazione: task/mesh
shader, visibility buffer, descriptor bindless, Hi-Z occlusion culling, ReSTIR
per l'illuminazione diretta, DDGI per l'illuminazione globale e molto altro.

Il motore e' organizzato come un insieme di pass di rendering indipendenti
collegati da un render graph con tracking automatico delle dipendenze.  Sette
scene di test integrate permettono di esplorare diversi carichi di lavoro senza
scrivere codice.

---

## What You Can Do / Cosa Puoi Fare

Press **1-7** or use the ImGui dropdown to switch between test benches:

| # | Test Bench | Description |
|---|-----------|-------------|
| 1 | **Torus Demo** | A rotating PBR gold torus on a ground plane.  The default scene -- verifies the full pipeline end-to-end. |
| 2 | **PBR Material Grid** | 10x10 spheres with metallic (rows) and roughness (columns) varying from 0 to 1.  Four point lights for even illumination. |
| 3 | **Stress Test** | 100,000 randomly placed low-poly spheres.  Tests GPU culling throughput, mesh shader dispatch, and memory bandwidth. |
| 4 | **Scene Viewer** | Loads a glTF/GLB file from `assets/` (Sponza, DamagedHelmet, etc.).  Falls back to procedural geometry if no assets are present. |
| 5 | **Many Lights** | 1,024 coloured point lights orbiting inside a large room with pillars.  Designed to stress the ReSTIR direct illumination pipeline. |
| 6 | **Cornell Box** | Classic Cornell box (red/green walls, two cubes, ceiling light).  Ideal for testing DDGI global illumination. |
| 7 | **Culling Viz** | 10,000 buildings in a city-grid layout.  Use the Meshlets/Overdraw debug overlay to visualise Hi-Z occlusion culling. |

---

## How It Works / Come Funziona

Each frame follows this pipeline:

```
Acquire Swapchain Image
        |
        v
  Upload Scene Data            ECS -> GPU buffers (instances, materials, lights)
        |
        v
  +--- Mesh Pass ---+          Task + Mesh shaders write a visibility buffer
  |  (Vis Buffer)   |          Each pixel stores instance ID + triangle ID
  +-----------------+
        |
        v
  +-- Material -----+          Compute shader reads vis buffer, fetches material
  |   Resolve       |          data via bindless, evaluates PBR + lighting
  +----- (HDR) -----+          Output: RGBA16F HDR render target
        |
        v
  +--- Tonemap -----+          ACES filmic tone mapping (HDR -> LDR)
  +----- (LDR) -----+
        |
        v
  +-- Composite ----+          Blit LDR result onto swapchain image
  +-(Swapchain)-----+
        |
        v
  +---- ImGui ------+          Debug UI overlays (dynamic rendering)
  +-----------------+
        |
        v
     Present
```

### Extended Pipeline (when enabled)

```
  Scene Upload
       |
       v
  Hi-Z Pyramid Build  <-+     Build mip chain from previous frame's depth
       |                 |
  Phase 1 Cull           |     Cull instances against LAST frame's Hi-Z
       |                 |
  Mesh Pass (visible)    |     Render surviving instances
       |                 |
  Phase 2 Cull           |     Cull rejects against THIS frame's Hi-Z
       |                 |
  Mesh Pass (late) ------+     Render newly-visible instances
       |
  Shadow Cascades              4 cascaded shadow maps (2048x2048 each)
       |
  Material Resolve             PBR shading with CSM + bindless textures
       |
  ReSTIR DI                    Spatiotemporal resampled direct lighting
       |   +---DDGI---+
       +-->| Probe RT  |      Ray-trace probe rays
           | Irradiance|      Update irradiance atlas
           | Visibility|      Update visibility atlas
           +-----------+
       |
  AA Pass (FXAA / TAA)        Anti-aliasing
       |
  Tonemap -> Composite -> ImGui -> Present
```

---

## Requirements / Requisiti

### GPU
- **Vulkan 1.3** with the following extensions:
  - `VK_EXT_mesh_shader` (task + mesh shaders)
  - `VK_KHR_dynamic_rendering`
  - `VK_KHR_synchronization2`
  - `VK_KHR_buffer_device_address`
  - `VK_EXT_descriptor_indexing` (bindless)
  - `VK_KHR_ray_tracing_pipeline` (optional, for DDGI)
  - `VK_KHR_acceleration_structure` (optional, for DDGI)
- **Minimum**: NVIDIA RTX 2060 / AMD RX 6600 or equivalent
- **Recommended**: NVIDIA RTX 3070+ / AMD RX 6800+ for 1024-light ReSTIR

### OS
- Linux (primary target, tested on Arch Linux)
- Windows 10/11 (should work, minor path adjustments may be needed)

### Build Dependencies
- C++20 compiler (GCC 13+, Clang 17+, MSVC 19.36+)
- CMake 3.24+
- Vulkan SDK 1.3.261+ (with `glslangValidator` or `glslc` for shader compilation)
- SDL3
- VulkanMemoryAllocator (VMA)
- GLM
- Dear ImGui (with SDL3 + Vulkan backends)
- tinygltf (for glTF loading)
- stb_image (for texture loading, usually bundled with tinygltf)

On Arch Linux:
```bash
sudo pacman -S vulkan-devel sdl3 glm cmake ninja
# VMA, Dear ImGui, tinygltf are typically pulled via CMake FetchContent
```

---

## Build & Run / Compilazione e Avvio

```bash
# Clone
git clone https://github.com/user/phosphor.git
cd phosphor

# Configure
cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPHOSPHOR_ENABLE_VALIDATION=ON

# Build
cmake --build build -j$(nproc)

# Run
./build/phosphor
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `PHOSPHOR_ENABLE_VALIDATION` | `ON` (Debug), `OFF` (Release) | Enable Vulkan validation layers |
| `CMAKE_BUILD_TYPE` | `Debug` | `Debug`, `Release`, `RelWithDebInfo` |

### Shader Compilation

Shaders are compiled to SPIR-V at build time using `glslangValidator`.
Pre-compiled `.spv` files are placed in `build/shaders/`.

---

## Controls / Controlli

### Camera
| Key | Action |
|-----|--------|
| **W / A / S / D** | Move forward / left / backward / right |
| **Space / Ctrl** | Move up / down |
| **Shift** (hold) | Sprint (3x speed) |
| **Right Mouse** (hold) | Look around |
| **Scroll Wheel** | Zoom (orbit mode) |

### Test Benches
| Key | Action |
|-----|--------|
| **1 - 7** | Switch test bench directly |

### Debug
| Key | Action |
|-----|--------|
| **F1** | Overlay: None |
| **F2** | Overlay: Meshlet colours |
| **F3** | Overlay: World normals |
| **F4** | Overlay: Depth buffer |
| **F5** | Overlay: Overdraw heatmap |
| **F12** | Trigger RenderDoc capture |
| **Escape** | Quit |

### ImGui Panels

All rendering parameters are adjustable through the ImGui panels:

- **Test Bench** -- combo box to switch scenes
- **Performance** -- FPS, CPU/GPU ms graphs, per-pass GPU timings
- **GPU Resources** -- live allocation counts and memory usage
- **Debug & Rendering** -- overlay mode, AA mode, exposure, ReSTIR/DDGI toggles
- **RenderDoc** -- single-frame capture button (only visible when launched from RenderDoc)

---

## Diagnostics / Diagnostica

### Debug Overlays

Switch overlays with F1-F5 or the ImGui panel.  Overlays replace the lit
output with a false-colour visualisation:

- **Meshlets** -- each meshlet gets a unique colour, showing how geometry is
  partitioned for the mesh shader.
- **Normals** -- world-space normals mapped to RGB.
- **Depth** -- linearised depth, white = near, black = far.
- **Overdraw** -- heatmap showing how many fragments are written per pixel.
  Blue = 1 (ideal), red = 4+ (wasteful).
- **Light Heatmap** -- number of lights contributing to each pixel.

### RenderDoc Integration

Launch Phosphor from within **RenderDoc** and the engine will automatically
detect the API.  Press **F12** to capture a single frame.  The capture
includes all Vulkan calls with debug labels (one per render pass) and named
objects for easier debugging.

### GPU Profiler

The built-in GPU profiler uses Vulkan timestamp queries to measure each pass.
Results are displayed in the Performance panel.  The profiler smooths values
over 60 frames to reduce jitter.

---

## Architecture / Architettura

```
phosphor/
  src/
    core/           Window, Input, Timer, Engine (composition root), types
    rhi/            VulkanDevice, GpuAllocator, Swapchain, CommandManager,
                    SyncManager, BindlessDescriptorManager, PipelineManager,
                    ShaderModule, AccelerationStructureManager
    render_graph/   RenderGraph, ResourceRegistry, BarrierBuilder, RenderPass
    renderer/       FrameContext, GpuScene, VisibilityBuffer, MeshPass,
                    MaterialResolve, TonemapPass, CompositePass, HiZPass,
                    ShadowPass, ReSTIRPass, AAPass, DDGIPass, MeshletBuilder
    scene/          ECS, Camera, Components, ProceduralMeshes,
                    TextureManager, GltfLoader
    diagnostics/    GpuProfiler, DebugUtils, FrameStats, RenderDocCapture,
                    ResourceTracker, DebugOverlay, Validation
    imgui/          ImGuiLayer, UIPanels
    testbench/      TestBench base + 7 concrete benches
  shaders/
    common/         Shared GLSL includes (types.glsl, etc.)
    mesh/           Task + mesh shaders
    visibility/     Visibility buffer write
    lighting/       Material resolve, ReSTIR stages
    post/           Tonemap, FXAA, TAA
    culling/        Hi-Z build, phase 1/2 culling
    gi/             DDGI probe tracing, atlas update
    debug/          Debug overlay shaders
```

### Key Design Decisions

1. **Visibility Buffer** -- instead of a traditional G-buffer, the mesh pass
   writes only an instance ID and triangle ID per pixel (32 bits).  The
   material resolve pass reconstructs all attributes on the fly using buffer
   device addresses.  This saves bandwidth and scales to millions of triangles.

2. **Bindless Descriptors** -- a single descriptor set with 16K texture slots
   and 4K buffer slots.  Materials reference textures by index, avoiding
   per-draw descriptor updates.

3. **Mesh Shaders** -- geometry is pre-processed into meshlets (64 vertices,
   124 triangles).  Task shaders perform per-meshlet frustum + cone culling,
   mesh shaders output surviving triangles.

4. **Two-Phase Occlusion Culling** -- Phase 1 culls against last frame's Hi-Z
   pyramid; Phase 2 culls phase-1 rejects against this frame's depth.  This
   eliminates temporal lag for fast-moving cameras.

5. **ReSTIR DI** -- spatiotemporal reservoir-based resampling for direct
   lighting.  Handles 1000+ lights without per-pixel loops.

6. **DDGI** -- Dynamic Diffuse Global Illumination via probe ray tracing with
   irradiance and visibility atlases.

---

## Technologies / Tecnologie

| Feature | Technique | Reference |
|---------|-----------|-----------|
| Geometry pipeline | Task/Mesh shaders + meshlets | [Mesh Shading -- NVIDIA, 2018](https://developer.nvidia.com/blog/introduction-turing-mesh-shaders/) |
| Visibility buffer | Instance+triangle ID per pixel | [Wihlidal, "Optimizing the Graphics Pipeline with Compute", GDC 2016](https://www.gdcvault.com/play/1023109/) |
| Bindless textures | `VK_EXT_descriptor_indexing` | [Vulkan Guide -- Descriptor Indexing](https://docs.vulkan.org/guide/latest/extensions/VK_EXT_descriptor_indexing.html) |
| Occlusion culling | Hi-Z pyramid + two-phase cull | [Wihlidal, "Optimizing the Graphics Pipeline with Compute", GDC 2016](https://www.gdcvault.com/play/1023109/) |
| Shadows | Cascaded Shadow Maps (4 cascades) | [Dimitrov, "Cascaded Shadow Maps", GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-10-parallel-split-shadow-maps-programmable-gpus) |
| Direct lighting | ReSTIR DI (WRS + temporal/spatial) | [Bitterli et al., "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting", SIGGRAPH 2020](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct) |
| Global illumination | DDGI (probe irradiance + visibility) | [Majercik et al., "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields", JCGT 2019](https://jcgt.org/published/0008/02/01/) |
| Tone mapping | ACES filmic curve | [Narkowicz, "ACES Filmic Tone Mapping Curve", 2015](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/) |
| Anti-aliasing | FXAA 3.11 / TAA | [Lottes, "FXAA", NVIDIA 2011](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf) |
| Memory allocation | VMA (Vulkan Memory Allocator) | [GPUOpen -- VMA](https://gpuopen.com/vulkan-memory-allocator/) |

---

## License / Licenza

See `LICENSE` file for details.
