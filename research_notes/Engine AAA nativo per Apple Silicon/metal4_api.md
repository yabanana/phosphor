# Metal 4 API (and still-relevant Metal 3.x features) for a AAA native engine on Apple Silicon — state as of Sept 2026

Scope note: primary sources are Apple's WWDC25 sessions (205 "Discover Metal 4", 254 "Explore Metal 4 games", 211 "Go further with Metal 4 games", 262 "Combine Metal 4 machine learning and graphics"), Apple Tech Talks on M3 (111375) and M5/A19 (111431, 111432), the WWDC26 Metal guide and session 330, Apple docs (.md symbol pages), the **Metal Feature Set Tables PDF dated May 21, 2026** (text extracted directly from the PDF), and Xcode 27 release notes. Session content was read through transcript summaries, so exact method spellings from sessions should be cross-checked against SDK headers (metal-cpp / Metal.framework) before use.

---

## 1. Metal 4's new programming model and how it differs from Metal 3

### Takeaway
Metal 4 (shipped with the "26" OSes in fall 2025) is a second, explicit, D3D12/Vulkan-like API surface (the `MTL4*` types) that sits alongside the old Metal 3 API rather than replacing it: app-owned command memory (`MTL4CommandAllocator`), command buffers decoupled from queues, argument tables instead of per-encoder `set*` binding, residency sets instead of implicit residency and `useResource`, no automatic hazard tracking (stage-to-stage barriers), a unified compute encoder, and an explicit compiler object (`MTL4Compiler`) with flexible/unspecialized pipelines and an ahead-of-time archive flow. It runs on Apple7+ GPUs (A14 / M1 and later).

### Cited Findings
**Availability / coexistence**
- Metal 4 supports "M1 and later" and "A14 Bionic and later". — [Discover Metal 4 (WWDC25-205)](https://developer.apple.com/videos/play/wwdc2025/205/)
- Feature tables (May 21, 2026): "The Metal 3 programming model is the combined Metal 1, 2, and 3 API surface and is available on all Apple GPU families; the Metal 4 programming model is available as of Apple7." Device mapping: A14=Apple7, A15/A16=Apple8, A17 Pro/A18=Apple9, A19=Apple10, M1=Apple7, M2=Apple8, M3/M4=Apple9, M5=Apple10. Every row in the table from A14/M1 onward is labeled "Metal 3 & 4". — [Metal Feature Set Tables PDF (May 2026)](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- The table lists these as **Metal 4-only** (Apple7+ unless noted): Argument tables; Command allocators; Decoupled command queues and command buffers; Command barriers; Dedicated compilation contexts; Pipeline dataset serialization; Flexible render pipeline state; Machine learning encoding; Tensors; Performance counter heaps; Placement sparse buffers/textures (Apple8, some Apple7 incl. all Apple7 Macs — check `MTLDevice.supportsPlacementSparse`); Address-driven acceleration structure builds (Apple9). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Features usable from both models ("Metal 3 & 4") include: Texture view pools (Apple7), Color attachment mapping (Apple7), Residency sets (Apple6), Intersection function buffers (Apple9), Acceleration structure build options (Apple9). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Tessellation is listed as "Metal 3" only (not Metal 4) — Tessellation (Apple3), Indirect tessellation arguments (Apple5), Tessellation in ICBs (Apple5). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Build requirements at launch: Xcode 26 and a device on OS version 26. — [Metal by Example: Getting Started with Metal 4](https://metalbyexample.com/metal-4/)
- Xcode 26 includes a Metal 4 game project template. — [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)

**Command model**
- `MTL4CommandQueue`, `MTL4CommandBuffer` (created from the device, independent of any queue), `MTL4CommandAllocator` ("In Metal 4, this memory is managed by a MTL4CommandAllocator... take direct control of your app's command buffer memory use"). — [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- Command buffers are "long-lived objects" that "no longer manage their own memory"; encoding starts with `beginCommandBuffer(allocator:)`; allocators "cannot be reused while the commands they encoded are in-flight". Submission: `waitForDrawable(_:)`, `commit(_:)`, `signalDrawable(_:)` on the queue, then `drawable.present()`. — [Metal by Example](https://metalbyexample.com/metal-4/); [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- Critical behavioral change: "command buffers do not implicitly retain their resources nor make them resident" (app owns lifetime). — [Metal by Example](https://metalbyexample.com/metal-4/)
- Allocators: `reset` only after GPU completion; not thread-safe (one per encoding thread); memory grows until reset. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- Typical frame pacing: one allocator per in-flight frame plus `MTLSharedEvent` (`signalEvent(_:value:)` on the queue, CPU `wait(untilSignaledValue:timeoutMS:)`). — [Metal by Example](https://metalbyexample.com/metal-4/)
- **Unified compute encoder** `MTL4ComputeCommandEncoder` encodes blits, dispatches and acceleration-structure builds; commands without dependencies run concurrently with no implicit sync. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/); [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- `MTL4RenderCommandEncoder` supports **color attachment mapping** (`MTLLogicalToPhysicalColorAttachmentMap`, `setPhysicalIndex:forLogicalIndex:`, descriptor `supportColorAttachmentMapping = YES`, `setColorAttachmentMap:`) to cut the number of render encoders. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- **Render pass suspend/resume across command buffers**: `MTL4RenderEncoderOptionSuspending` / `MTL4RenderEncoderOptionResuming`, requires batched commit (`[commandQueue commit:cmdbufs count:N]`), merging parallel-encoded parts into one GPU render pass (avoids store/load). — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)

**Binding: argument tables**
- `MTL4ArgumentTable`, created from `MTL4ArgumentTableDescriptor` (`maxBufferBindCount`, `maxTextureBindCount`); buffers bound by GPU address (`setAddress(gpuAddress:index:)`, offsets via address arithmetic), textures by `gpuResourceID` (`setTexture(resourceID:index:)`); set with `setArgumentTable(_:stages:)`; "Argument table state is effectively copied for each command". For bindless, a table typically needs a single buffer binding (to a top-level argument buffer). — [Metal by Example](https://metalbyexample.com/metal-4/); [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- Per-function binding limits (Apple7–Apple10): 31 buffers, 128 textures, 16 samplers; footnote: "These values are identical to the maximum number of bindings in an MTL4ArgumentTable of the same type." Through argument buffers (tier 2): unlimited buffers, 1M textures per stage, samplers 996 (Apple7/8) and 500K (Apple9/10). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- **Texture view pools**: `MTLTextureViewPool` from `MTLResourceViewPoolDescriptor` (`resourceCount`), `newTextureViewPoolWithDescriptor:error:`, `setTextureView:descriptor:atIndex:` returns an `MTLResourceID` — no allocations while encoding. Max entries: 128M (Apple7/8), 256M (Apple9/10). — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/); [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

**Residency sets**
- `MTLResidencySet` is available since macOS 15 / iOS 18 (introduced in Metal 3.2, mandatory mental model in Metal 4). API: `addAllocation(s)`, `removeAllocation(s)`, `removeAllAllocations`, `commit()`, `requestResidency()`, `endResidency()`, `allocatedSize`. Residency sets "don't track hazards"; "don't support sparse heaps or sparse textures, and their methods aren't thread-safe"; adding a heap-allocated resource makes the whole heap resident; Metal makes the union of all sets resident. — [MTLResidencySet docs](https://developer.apple.com/documentation/metal/mtlresidencyset)
- Attach to queue (`addResidencySet`) or command buffer (`useResidencySet(s)`); max 32 residency sets per queue and 32 per command buffer (Apple6+). — [Metal by Example](https://metalbyexample.com/metal-4/); [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Guidance: fewer sets with many resources; populate at startup; `CAMetalLayer` exposes its own residency set to add once to the queue. Control Ultimate Edition saw "significant reductions in the overheads of managing residency and lower memory usage when ray-tracing is disabled." — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/); [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)

**Synchronization**
- Barriers work "stage to stage" (e.g., dispatch→fragment). Intra-encoder "pass barriers": `barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:`; cross-encoder "queue barriers": `barrierAfterQueueStages:beforeStages:visibilityOptions:`. Stages: `MTLStageVertex`, `MTLStageFragment`, `MTLStageDispatch`, `MTLStageBlit`, `MTLStageAccelerationStructure`, `MTLStageMachineLearning`; visibility e.g. `MTL4VisibilityOptionDevice`. Fences and `MTLEvent`/`MTLSharedEvent` remain for cross-queue/CPU sync. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/); [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/); [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- Apple sample/doc titles cited in session: "Synchronizing passes with producer/consumer/fence barriers", "Understanding the Metal 4 core API", "Using the Metal 4 compilation API", "Drawing a triangle with Metal 4". — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)

**Pipeline compilation**
- `MTL4Compiler` (from `newCompilerWithDescriptor:`) — compilation moved off the device; inherits the calling thread's QoS; sync and async variants. Function descriptors are mandatory (`MTL4LibraryFunctionDescriptor`, `MTL4SpecializedFunctionDescriptor` + `MTLFunctionConstantValues`); `MTL4RenderPipelineDescriptor` takes function descriptors, `MTL4RenderPipelineColorAttachmentDescriptor` uses a `blendingState` enum; no depth attachment descriptors in the pipeline. — [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/); [Metal by Example](https://metalbyexample.com/metal-4/); [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- **Flexible render pipeline states**: build unspecialized pipeline once using `MTLPixelFormatUnspecialized`, `MTLColorWriteMaskUnspecialized`, `MTL4BlendStateUnspecialized`, then `newRenderPipelineStateBySpecializationWithDescriptor:pipeline:error:` — only the fragment-output part is regenerated (Metal IR reused). Small GPU-side overhead (jump to output part), so profile hot shaders. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- **AOT flow**: `MTL4PipelineDataSetSerializer` (config `...CaptureDescriptors`) → `serializeAsPipelinesScriptWithError:` (JSON, `.mtl4-json`) → offline `metal-tt` builds a Metal archive → runtime `MTL4Archive` (`newArchiveWithURL:`), lookup miss must fall back to `MTL4Compiler`. Apple claims runtime pipeline load time "near zero". — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)

### Inferences
- Metal 4 is a clear match for an engine RHI modeled on D3D12/Vulkan (explicit allocators, bindless via one argument-table slot pointing at a descriptor-heap-like argument buffer, explicit barriers). Engines must now implement their own hazard tracking and resource lifetime/deferred-deletion, which Metal 3 did implicitly.
- Because tessellation is Metal-3-only in the table, a Metal 4 renderer should plan on mesh shaders (or compute-generated geometry) instead of fixed-function tessellation.
- Metal 3 and Metal 4 objects coexist in one app (e.g., placement sparse mappings synchronized with an existing `MTLCommandQueue` via events per WWDC25-205), allowing an incremental port.

### Gaps
- Exact current spelling of some selectors (e.g., whether queue barrier selector is `barrierAfterQueueStages:beforeStages:` vs `...beforeQueueStages:` — sessions show both forms) should be verified in SDK headers; ML session shows `barrierAfterStages:beforeQueueStages:visibilityOptions:`.
- No Apple-published CPU-overhead benchmark (Metal 3 vs Metal 4) found. A GitHub PR "B12 — a real Metal 4 backend, measured against the MTL3 path" exists (github.com/noah-qin/Corta/pull/72) but was not read/verified.
- Intel/AMD Macs: the May 2026 tables list only Apple families; Metal 4 is stated for M1+/A14+, so Metal 4 appears Apple-silicon-only (inference from absence; no explicit Apple statement fetched).

---

## 2. Machine learning in Metal 4 (tensors, Shader ML, ML encoder) and neural rendering uses

### Takeaway
Metal 4 makes tensors first-class (`MTLTensor` in API and MSL), offers two execution paths — (a) whole networks on the GPU timeline via `MTL4MachineLearningCommandEncoder` from Core ML models converted to `.mtlpackage`, and (b) small networks inlined into any shader via Metal Performance Primitives "TensorOps" (`matmul2d`, `convolution2d`) — and on M5/A19 these hit per-core Neural Accelerators. Apple's demo neural material compression hit ~50% of BC memory/disk footprint with no perceived quality loss. 2026 (OS 26.x/27) added bf16, int8/int4, then fp8/fp4/int2 and MX block-scaled formats.

### Cited Findings
- `MTLTensor`: rank, extents, dataType, usage (`MTLTensorUsageMachineLearning | Compute | Render`); created from device (`newTensorWithDescriptor:offset:error:`, opaque optimized layout = best performance) or from an `MTLBuffer` with explicit strides (innermost stride 1). — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- MSL: `#include <metal_tensor>`; device tensors e.g. `tensor<device half, dextents<int,2>>` bound via buffer slots/argument buffers; inline tensors created in-shader: `tensor(inputs, extents<int, N, 1>())`. — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- ML encoder flow: PyTorch → Core ML ML Program (`coremltools`, `convert_to='mlprogram'`) → `metal-package-builder model.mlpackage` → load `.mtlpackage` as `MTLLibrary` → `MTL4MachineLearningPipelineDescriptor` (+`setInputDimensions:atBufferIndex:`) → `[compiler newMachineLearningPipelineStateWithDescriptor:]` → `machineLearningCommandEncoder`, `setPipelineState`, `setArgumentTable`, `dispatchNetworkWithIntermediatesHeap:`; intermediates heap is `MTLHeapTypePlacement` sized by `pipeline.intermediatesHeapSize`; synchronization via barriers with `MTLStageMachineLearning`. — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- Shader ML / Metal Performance Primitives: `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, `mpp::tensor_ops::matmul2d_descriptor(M,N,K, transposeL, transposeR, reducedPrecision)`, `matmul2d<desc, execution_thread>` (single thread, e.g., per-fragment) or `execution_simdgroups<N>` (requires uniform control flow; no divergence at call site). "Compiler inlines optimized code directly into shaders." — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/); [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- Neural material demo: 4 latent textures + UV → 2 fully-connected layers with ReLU → base color (3ch) + tangent-space normal (3ch), evaluated per fragment; "50% of block-compressed format footprint", 50% disk, "no perceived quality loss". Rationale for Shader ML: avoids device-memory round-trip between sampling, inference and shading. — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- Tooling: Metal debugger Dependency Viewer, ML Network Debugger (graph, per-op intermediate tensors), MTLTensor viewer. — [WWDC25-262](https://developer.apple.com/videos/play/wwdc2025/262/)
- Feature tables: "Machine learning encoding" and "Tensors" are Metal 4, Apple7+ (i.e., M1+). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- M5/A19 **Neural Accelerators**: dedicated matmul hardware in each GPU shader core; scales with core count; GEMM 4–8x faster depending on precision; TensorOps code is portable M1–M5 (falls back to optimized shaders without accelerators). Data-type timeline: bf16 in OS 26.1; cooperative tensors as matmul inputs in 26.3; INT8/INT4 in 26.4. Demo 4Kx4K matmul: SIMD-group-matrix ~2 s → TensorOps ~0.5 s → +Morton order ~0.33 s. Frameworks using accelerators automatically: MPS, MPSGraph, Core ML, MetalFX, MLX, llama.cpp, PyTorch. — [Tech Talk 111432 "Accelerate your ML workloads with the M5 and A19 GPUs"](https://developer.apple.com/videos/play/tech-talks/111432)
- WWDC26 (session 330): quantized formats — macOS/iOS 26: 4- and 8-bit ints; macOS/iOS 27: 4- and 8-bit floats, 2-bit ints, FP8 E8M0 block-wise scale factor format; multi-plane tensors (data plane + scales plane, e.g., `blockFactors={32,1}`); names include `MTLTensorDataTypeMetalFloat8E4M3`, `MTLTensorDataTypeMetalFloat8UE8M0`; MSL `tensor_blockwise<tensor_plane_scales, ...>`; TensorOps additions `reduce_rows`, `map_iterator`, `get_left_input_cooperative_tensor`, `is_compatible_as_left_input`; dequantization handled by `matmul2d`; FlashAttention example. Session was ML/compute-focused, not rendering. — [WWDC26-330 "Optimize custom ML operations with Metal tensors"](https://developer.apple.com/videos/play/wwdc2026/330)
- Apple's WWDC26 Metal page: "new quantized tensor formats... native support in Metal Performance Primitives... scale factors... weight compression", targeting Neural Accelerators on M5 Pro/M5 Max. — [Metal What's New](https://developer.apple.com/metal/whats-new/); [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)
- Xcode 27 notes mention "CoreAI models" (a Metal API Validation fix), and session 330 mentions "Core AI" conversion of PyTorch models with custom Metal kernels (`TorchMetalKernel`). — [Xcode 27 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes); [WWDC26-330](https://developer.apple.com/videos/play/wwdc2026/330)

### Inferences
- Neural texture/material compression is the most engine-ready use: per-fragment MLP via `execution_thread` works on M1+ but will be far faster on M5-class Neural Accelerators; budget a fallback (BC/ASTC) path for Apple7–9.
- Neural denoising: either use MetalFX denoised upscaler (Apple9+) or a custom network via the ML encoder on the GPU timeline, barriered against the RT pass with `MTLStageMachineLearning`.
- Quantized weights (int4/fp8/MX) in OS 26.4/27 make larger per-pixel networks viable for neural materials at lower bandwidth — worth tracking for M5+.
- "Core AI" appears to be a 2026 successor/companion to Core ML for model conversion; its relationship to `.mtlpackage` is unclear.

### Gaps
- WWDC25-262 summary said `coremltools` `minimum_deployment_target=ct.target.macOS16`; this likely reflects the pre-rename beta target naming (macOS 26) — unverified.
- No Apple-published ms-cost numbers for neural materials per resolution/GPU; no official neural texture compression SDK found (only the demo).
- Details of Core AI framework not researched.

---

## 3. Mesh shaders (object/mesh stages)

### Takeaway
Mesh shading (object + mesh stages, `metal::mesh`) is available on Apple7+ (M1/A14) since Metal 3 and remains in Metal 4; Apple9 (M3/A17 Pro) added hardware-accelerated mesh scheduling that keeps meshlet data on-chip, indirect mesh draws and mesh draws in ICBs; Apple10 (M5) raises grid limits and roughly doubles geometry throughput vs M4.

### Cited Findings
- Mesh shading: Apple7+; Indirect mesh draw arguments: Apple9+; Indirect command buffers containing mesh draws: Apple9+. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Limits: max threadgroups per object grid "No limit" (Apple7+); max threadgroups per mesh grid 1024 (Apple7/8), 1,048,575 (Apple9), 4,194,303 (Apple10); max payload 16,384 B (all); payload reduced by 16 B if using `[[threadgroups_per_grid]]`/`[[threads_per_grid]]`, and another 16 B when viewing geometry in the Metal debugger; Apple7/8 can use up to 4 GB of payload + mesh geometry per draw. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Constraint: "Support for function pointers and ray tracing in render pipelines isn't compatible with mesh shading. You can only use Metal IR linking through MTLLinkedFunctions.privateFunctions in render pipelines using mesh shading." — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- M3/A17 Pro: "much more efficiently schedule object and mesh threadgroups to keep intermediate meshlet data on chip"; mesh grid limit raised from 1,024 to >1M; mesh draws in ICBs. Best practices: minimize `metal::mesh` template sizes (drop unused attributes), set max vertices/primitives only as large as needed, omit vertex positions for culled primitives to let hardware cull. No numeric speedup published. — [Tech Talk 111375 (M3/A17 Pro)](https://developer.apple.com/videos/play/tech-talks/111375/)
- M5: geometry throughput 2x vs M4. — [Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431)
- Developer reports of poor mesh-shader performance on earlier hardware exist (Apple forums thread "Bad mesh shader performance"). — [Apple Developer Forums 722047](https://developer.apple.com/forums/thread/722047) (not read in full)

### Inferences
- For a Nanite-style virtualized geometry pipeline, target Apple9+ as the "fast path" (on-chip meshlet scheduling + indirect/ICB mesh draws) and consider compute-rasterization or vertex-pulling fallback for Apple7/8, where mesh grids are capped at 1024 threadgroups per draw and no indirect mesh draws exist.
- Because ray queries in render pipelines are incompatible with mesh shading pipelines, inline RT in a mesh-shaded G-buffer pass is not possible; do RT in compute/fragment passes of non-mesh pipelines.

### Gaps
- No published Apple mesh-shader throughput numbers (tris/clock) or max vertices/primitives per meshlet in the tables fetched (MSL spec would have per-mesh limits).

---

## 4. Ray tracing

### Takeaway
Metal RT (acceleration structures, intersector/intersection-query APIs, intersection & visible function tables) runs on Apple6+ in compute and render pipelines, with hardware RT from Apple9 (M3/A17 Pro) and third-generation RT on M5. Metal 4 adds intersection function buffers (DXR-SBT-like), per-build AS options, and address-driven AS builds (Apple9); M5 hardware-accelerates IFB indexing and instance transforms and shrinks AS alignment from 16 KB to 1 KB.

### Cited Findings
- Ray tracing in compute pipelines: Apple6+; in render pipelines: Apple6+ (incompatible with mesh shading); row-major-matrix AS, per-component motion interpolation, direct access to on-chip ray-intersection result storage: Apple9+; Intersection function buffers: "Metal 3 & 4", Apple9; Acceleration structure build options: Apple9; Address-driven AS builds: Metal 4, Apple9. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Limits: intersector can traverse 32 AS levels; intersection query 16 levels (includes one primitive level); IFB min alignment 64 B, stride alignment 8 B, max stride 4096 B (Apple7+). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Intersection function buffers: per-geometry/instance `intersectionFunctionTableOffset`; MSL `intersector<intersection_function_buffer, instancing, triangle>`, `set_geometry_multiplier(N)` (ray types), `set_base_id(k)`, `intersection_function_buffer_arguments` (buffer, size, stride); "Shader Binding Tables map directly to Metal intersection function buffers". — [WWDC25-211](https://developer.apple.com/videos/play/wwdc2025/211/)
- New AS build flags in Metal 4 (per build): refit-optimized, larger scenes, prefer fast intersection, minimize memory. — [WWDC25-211](https://developer.apple.com/videos/play/wwdc2025/211/)
- AS builds now encoded in `MTL4ComputeCommandEncoder` with `MTLStageAccelerationStructure` barriers. — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- M3 hardware RT: fixed-function traversal; a reorder stage groups intersection-function calls across SIMD-groups to reduce divergence. Best practices: prefer the intersector object API over intersection query (query increases scratch traffic and disables reordering); separate intersection functions per logical routine (not an uber-function); minimize payload. Blender Cycles converges "significantly faster" on M3 vs M2 (no %). — [Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)
- M5: third-generation hardware RT; faster hardware instance transforms; fully hardware-accelerated IFB indexing (~70% GPU time reduction vs emulation); AS memory alignment 16 KB → 1 KB ("eliminates hundreds of MB padding"). — [Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431)

### Inferences
- For a hybrid renderer on M3+, use intersector (not `intersection_query`) in compute for reflections/GI/shadows to benefit from the reorder stage; IFBs make a DXR-style material/hit-group model portable.
- Apple6–8 have RT API support but no RT hardware (hardware RT starts Apple9 per Apple's M3 talk); RT effects should be scalable/optional there.

### Gaps
- No Apple-published absolute RT throughput numbers (rays/s) or M3→M4→M5 percentage speedups were found.
- Whether "ray queries in any shader stage" includes mesh/object stages: tables say RT in render pipelines is incompatible with mesh shading.
- Indirect AS build details (address-driven builds) API names not extracted.

---

## 5. MetalFX (spatial, temporal, denoised upscaling, frame interpolation)

### Takeaway
MetalFX provides spatial (Apple3+) and temporal (Apple7+) upscaling, frame interpolation (listed Apple5+) and a joint denoiser+upscaler for ray/path tracing (Apple9+, i.e., M3/A17 Pro+), with Metal 4 variants (`MTL4FX*`) that encode into `MTL4CommandBuffer` (OS 26+). WWDC26 announced a redesigned temporal upscaler using the Neural Engine/Neural Accelerators on M5 Pro/Max, plus sub-rectangle processing, motion-vector passthrough, and distortion fields.

### Cited Findings
- Availability by family: MetalFX spatial upscaling Apple3; temporal upscaling Apple7; frame interpolation Apple5; denoised upscaling Apple9. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- `MTLFXFrameInterpolator` (inherits `MTLFXFrameInterpolatorBase`, `encode(commandBuffer:)`) and `MTL4FXTemporalScaler` (`encode(commandBuffer: any MTL4CommandBuffer)`, inherits `MTLFXTemporalScalerBase`, `MTLFXFrameInterpolatableScaler`) — both iOS/iPadOS/macOS/tvOS/Mac Catalyst 26.0+. — [MTLFXFrameInterpolator docs](https://developer.apple.com/documentation/metalfx/mtlfxframeinterpolator); [MTL4FXTemporalScaler docs](https://developer.apple.com/documentation/metalfx/mtl4fxtemporalscaler)
- Temporal upscaler: inputs color (linear), motion vectors, depth, exposure; now supports dynamically sized inputs (dynamic resolution) with max recommended 2x scale; optional reactive mask (`temporalUpscaler.reactiveMask`); placement after jittered rendering, before post/tonemap/UI; exposure debugger `MTLFX_EXPOSURE_TOOL_ENABLED`. — [WWDC25-211](https://developer.apple.com/videos/play/wwdc2025/211/)
- Frame interpolator: inputs color, previous color, motion vectors, depth, output; `MTLFXFrameInterpolatorDescriptor` can link a scaler (`desc.scaler`); `motionVectorScaleX/Y`, `depthReversed`; three UI modes (composited, offscreen, every-frame UI); min ~30 FPS input; pacing verified with Metal HUD (≤2 histogram buckets). — [WWDC25-211](https://developer.apple.com/videos/play/wwdc2025/211/)
- Denoised upscaler: `newTemporalDenoisedScalerWithDevice:` from `MTLFXTemporalScalerDescriptor`; required noise-free aux inputs: world-space normals (signed format), diffuse albedo, roughness (linear), specular albedo (incl. Fresnel), depth, motion; optional specular hit distance, denoise-strength mask, transparency overlay. Pitfalls: correlated RNG, metallic albedo split. Recommended pipeline: RT low-res → denoised upscaler → exposure/tonemap → frame interpolation → UI → paced present. — [WWDC25-211](https://developer.apple.com/videos/play/wwdc2025/211/)
- WWDC26: "Temporal Upscaler (Redesigned)" leveraging Neural Engine and Neural Accelerators on M5 Pro and M5 Max; new capabilities: process subrectangles for dynamic resolution, pass motion vectors to reduce preprocessing time, handle post-process effects with distortion fields. — [Metal What's New](https://developer.apple.com/metal/whats-new/); [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)
- Xcode 27: Metal Performance HUD shows more MetalFX metrics (jitter sequence length, motion vector scale) and an "Overrides" panel (jitter multiplier, MV scale, exposure visualization); GPU capture option "Include MetalFX temporal scaler history". — [Xcode 27 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes)
- M5 talk: MetalFX benefits from Neural Accelerators automatically. — [Tech Talk 111432](https://developer.apple.com/videos/play/tech-talks/111432)

### Inferences
- A AAA engine can treat MetalFX as the default TAAU/denoiser/frame-gen stack, analogous to DLSS SR+RR+FG, with the denoiser tier limited to M3+/A17 Pro+.
- WWDC26 API names for sub-rect/distortion-field features were not captured; expect them on `MTLFXTemporalScalerBase`/descriptor in the 27 SDK.

### Gaps
- Exact API names and OS version for the 2026 MetalFX additions (sub-rects, distortion fields, motion-vector passthrough) and whether the redesigned upscaler is M5 Pro/Max-only or has fallbacks — not verified.
- Frame interpolation "Apple5" table entry seems surprisingly low; reported as-is.
- No published MetalFX cost (ms) per resolution.

---

## 6. Fast resource loading (MTLIO), sparse textures, placement heaps, streaming

### Takeaway
Streaming building blocks are: MTLIO (`MTLIOCommandQueue`, Metal 3, Apple2+) for direct file→buffer/texture loads with built-in decompression; classic sparse textures (Apple6+, automatic heap backing); and Metal 4 **placement sparse** buffers/textures (Apple8+, most Apple7) mapped onto `MTLHeapTypePlacement` heaps with 16/64 KB pages via queue-level mapping operations.

### Cited Findings
- Fast resource loading: Apple2+, "Metal 3 & 4"; max 8192 I/O commands per buffer. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- `MTLIOCommandQueue`/`MTLIOCommandBuffer` load file data directly into Metal buffers/textures; inline decompression; built-in codecs incl. zlib, LZBITMAP, LZFSE, LZ4, LZMA; custom codecs supported; LZ4 when decompression speed is critical. — [WWDC22 "Load resources faster with Metal 3"](https://developer.apple.com/videos/play/wwdc2022/10104/); [MTLIOCommandQueue docs](https://developer.apple.com/documentation/metal/mtliocommandqueue); [MTLIOCompressionMethod.lzma](https://developer.apple.com/documentation/metal/mtliocompressionmethod/lzma?language=_3)
- Sparse textures (automatic heap backing): Apple6+, no sparse buffers or 1D/1DArray/TextureBuffer; sparse depth/stencil Apple8 (some Apple7 incl. all Apple7 Macs via `supportsPlacementSparse`). Placement sparse buffers/textures: Metal 4, Apple8 (and some Apple7 incl. all Apple7 Macs). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Placement sparse API: heap `MTLHeapTypePlacement` with `maxCompatiblePlacementSparsePageSize` (e.g., `MTLSparsePageSize64`); page sizes 16 KB and 64 KB (larger pages = faster mapping, more padding); resources set `placementSparsePageSize`; mapping ops `MTL4UpdateSparseBufferMappingOperation`, `MTL4UpdateSparseTextureMappingOperation` via `[cmdQueue updateBufferMappings:heap:operations:count:]` (and texture equivalent). — [WWDC25-254](https://developer.apple.com/videos/play/wwdc2025/254/)
- Placement sparse resources are allocated without pages; pages come from a placement heap; can sync with existing `MTLCommandQueue` via events. — [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- Residency sets don't support sparse heaps or sparse textures. — [MTLResidencySet docs](https://developer.apple.com/documentation/metal/mtlresidencyset)
- M5 raises max 1D/2D/cube texture size to 32,768 px (Apple10) vs 16,384 (Apple3–9). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431)

### Inferences
- Virtual texturing: placement sparse textures + MTLIO page loads into heap-backed staging, then mapping ops on the queue; virtualized geometry: placement sparse buffers (page pool) + MTLIO for cluster pages. MTLIO has no GPU decompression path documented comparable to DirectStorage GDeflate — decompression is inline in the IO pipeline (CPU/HW codecs; not verified).

### Gaps
- Whether MTLIO can target `MTL4` queues/placement sparse resources directly and its sync model with MTL4 events was not verified.
- Whether decompression is CPU or dedicated hardware — not verified.

---

## 7. Other relevant features (GPU-driven rendering, bindless, function pointers, dynamic libraries, validation, M5 features, Game Mode, function stitching)

### Takeaway
GPU-driven rendering relies on ICBs (render/compute, Apple3+; mesh draws in ICBs Apple9+; raster/depth-stencil state in ICBs Apple10), argument buffers tier 2 (Apple6+, 1M textures/stage) plus Metal 4 argument tables, function pointers/visible function tables and dynamic libraries (Apple6+). M5 (Apple10) adds depth-bounds test, sampler min/max reduction, LOD bias, pre-raster per-vertex values (visibility-buffer friendly), universal texture compression incl. shader-write textures, 8x MSAA, 32K textures.

### Cited Findings
- ICBs (rendering and compute): Apple3+; ICB memory barriers (rendering) Apple9; argument buffers tier 2 Apple6; function pointers in compute/render pipelines Apple6 (render not with mesh shading); render/compute dynamic libraries Apple6; binary archives Apple3; per-pipeline shader validation and shader logging Apple6; residency sets Apple6; 64-bit atomics full set Apple9 (Apple8 min/max ulong on macOS only); floating-point atomics Apple7; SIMD-scoped matrix multiply Apple7; lossy texture compression Apple8; VRR and vertex amplification Apple6; BC formats all Apple9 (Apple7/8 Macs all support). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Apple10-only (M5/A19): sampler min/max reduction, sampler LOD bias, access to pre-raster per-vertex values, depth bounds testing, ICB support for raster and depth/stencil state, atomics on cube/cube-array textures, universal texture compression; max MSAA 8x (16x16 tile); implicit imageblock 256 KB (Apple9/10). — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- M5 talk: GPU-encoded ICB commands can set device state (e.g., culling mode, depth bias) in-shader; visibility-buffer rendering encouraged using non-interpolated vertex values; universal compression automatically applies to `MTLTextureUsageShaderWrite` textures; FP16 and complex ALU 2x vs M4; geometry 2x; memory bandwidth up to +30%; second-gen Dynamic Caching with redesigned Occupancy Management Unit; Xcode 26.4 occupancy counters (`occupancy_target`, `..._register_pressure`, `..._l1_cache_pressure`, `..._memory_stalls`, `..._texture_decompression_stalls`, `texture_memory_red_compression_ratio`). — [Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431)
- M3 (Apple9) Dynamic Caching: registers dynamically allocated, register file acts as a cache; flexible on-chip memory for threadgroup/tile/stack/buffer; more on-chip stack helps function pointers/visible function tables/dynamic libraries; up to 2x ALU via parallel FP16/FP32/int issue. — [Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)
- Metal 4 tooling: API and shader validation, Metal debugger, Metal Performance HUD, Metal System Trace all support Metal 4. — [WWDC25-205](https://developer.apple.com/videos/play/wwdc2025/205/)
- Performance counter heaps: Metal 4, Apple7+, 32 per process. — [Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Xcode 27: more Metal validation options in the scheme Diagnostics panel (log non-fatal actions, validate load/store actions, allocation stack traces, GPU stack-overflow detection); "Optimize shared memory capture" option. — [Xcode 27 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes)
- WWDC26: Game Porting Toolkit 4 with a GitHub companion repo of AI-agent skills/sample code; session "Speedrun your game port with agentic coding" (WWDC26-357). — [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)

### Inferences
- A AAA engine can implement a visibility-buffer + GPU-driven pipeline on M1+ (primitive ID and barycentrics are Apple7+), with M5 adding hardware helpers (pre-raster per-vertex values, depth bounds, ICB state).
- Given universal compression for shader-write textures on M5, compute-written render targets no longer lose bandwidth compression there.

### Gaps
- Game Mode (macOS 14+ CPU/GPU prioritization) and `MTLFunctionStitchingGraph` were not researched in this pass — no sources collected; background knowledge only (function stitching since macOS 12/iOS 15 for compute "stitched" visible functions; Game Mode auto-activates for fullscreen games) — must be verified.
- Metal Shading Language version number for the 26/27 SDKs (e.g., MSL 4.0 / 4.1) not verified.

---

## 8. GPU family / OS support matrix and WWDC26 (June 2026) updates

### Takeaway
Metal 4 = Apple7+ (M1/A14 and newer) on OS 26+; everything important for AAA (placement sparse Apple8, hardware RT/mesh efficiency/IFB/denoiser Apple9, Neural Accelerators and new raster features Apple10) tiers above that. WWDC26 (June 8–12, 2026; OS 27 = macOS 27 "Golden Gate", iOS 27) was an incremental year for Metal: quantized tensors, redesigned neural MetalFX upscaler, MetalFX sub-rects/distortion fields, tooling, GPTK 4 — no new core Metal 4 API model.

### Cited Findings
- Family map and Metal 4 floor Apple7 — see §1. — [Feature Set Tables (May 21, 2026)](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Metal 4 OS baseline: OS 26 (Xcode 26); MetalFX Metal 4 variants 26.0+. — [Metal by Example](https://metalbyexample.com/metal-4/); [MTL4FXTemporalScaler docs](https://developer.apple.com/documentation/metalfx/mtl4fxtemporalscaler)
- Point releases: bf16 tensors 26.1, cooperative tensors as matmul inputs 26.3, int8/int4 26.4; Xcode 26.4 occupancy counters. — [Tech Talk 111432](https://developer.apple.com/videos/play/tech-talks/111432); [Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431)
- Xcode 27 ships SDKs for iOS/iPadOS/tvOS/watchOS/macOS/visionOS 27, requires macOS Tahoe 26.6+. — [Xcode 27 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-27-release-notes)
- WWDC26 held June 8–12, 2026; macOS 27 named "Golden Gate". — [MacRumors (Jun 10, 2026)](https://www.macrumors.com/2026/06/10/apple-lists-250-changes-ios-27-and-more/) (secondary source)
- WWDC26 Metal items: quantized tensor formats + scale factors in MPP; redesigned MetalFX temporal upscaler (Neural Engine + Neural Accelerators on M5 Pro/Max); sub-rectangle processing, motion vector passthrough, distortion fields; GPTK 4. Sessions: WWDC26-330 (tensors), WWDC26-357 (agentic porting), plus tech talks 111431/111432 (M5/A19). — [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/); [Metal What's New](https://developer.apple.com/metal/whats-new/)
- OS 27 tensor types: fp4/fp8, int2, FP8 E8M0 scales. — [WWDC26-330](https://developer.apple.com/videos/play/wwdc2026/330)

Summary matrix (from the May 2026 Feature Set Tables unless noted):
| Feature | Min family | Example chips |
|---|---|---|
| Metal 4 core (allocators, argument tables, barriers, MTL4Compiler, flexible PSO, tensors, ML encoder) | Apple7 | M1, A14 |
| Mesh shading | Apple7 | M1 |
| MetalFX temporal | Apple7 | M1 |
| Placement sparse buffers/textures | Apple8 (some Apple7 incl. all M1 Macs) | M2 / M1 Macs |
| Indirect mesh draws, mesh in ICBs, IFBs, AS build options, address-driven AS builds, 64-bit atomics | Apple9 | M3, M4, A17 Pro, A18 |
| MetalFX denoised upscaling | Apple9 | M3+ |
| Hardware RT | Apple9 (per M3 talk) | M3+ |
| Neural Accelerators, depth bounds, sampler min/max & LOD bias, ICB raster state, universal compression, 8x MSAA, 32K textures | Apple10 | M5, A19 |

### Inferences
- Practical AAA baseline: Metal 4 on Apple9+ (M3/M4) for full-featured path; M1/M2 as a reduced tier (no hardware RT, no denoiser, capped mesh grids); M5 as the "neural" tier.
- The feature tables no longer list Mac1/Mac2 (Intel/AMD) families, consistent with Metal 4 being Apple-silicon only.

### Gaps
- No evidence found of a "Metal 5" or of new core `MTL4` object types at WWDC26; searches returned no Metal ray tracing announcements for 2026. Could not access full WWDC26 session list beyond the Metal guide, so smaller additions (e.g., new MSL features in the 27 SDK) may be missed.
- Official feature table rows specific to OS 27 additions (e.g., new MetalFX features family requirements) were not present in the May 2026 PDF (pre-WWDC).
