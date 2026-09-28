# Legacy Vulkan implementation (archived)

This directory holds Phosphor's original Vulkan 1.3 renderer, kept as a
reference while the engine moves to a native Metal 4 backend for Apple
silicon.  It is **not built** by the top-level CMake project.

What is here and why it is kept:

| Path | Use during the port |
|---|---|
| `src/rhi/` | Vulkan RHI; replaced by `src/platform/metal/`. |
| `src/render_graph/` | Resource/barrier logic to be rewritten around Metal 4 stage-to-stage barriers (phase F1). |
| `src/renderer/*_pass.*` | Pass structure for Hi-Z, shadows, ReSTIR DI, DDGI, TAA/FXAA. |
| `shaders/` | GLSL reference for the algorithms being ported to MSL. DDGI used a Vulkan RT pipeline (rgen/rchit/rmiss) that Metal lacks; it becomes compute + `intersector`. |
| `src/diagnostics/` | Aftermath/RenderDoc/Vulkan validation tooling. Metal uses Xcode's GPU tools instead (RenderDoc does not support Metal). |

To build it you need the Vulkan SDK on Linux/Windows; `git checkout` the
last pre-port commit on `main` rather than trying to build from here.
