# Languages, Shader Toolchains and Developer Tooling for a Native Metal 4 Engine on macOS (as of Sept 2026)

Research date: 2026-09-28. Context: macOS 26 ("Tahoe") is shipping; macOS 27 / Xcode 27 SDK artifacts already appear (metal-cpp changelog, GitHub `xcode-27` runner label in public preview). MSL spec on developer.apple.com is now version 4.1.

## Host language options (metal-cpp, Objective-C++, Swift, Rust) and what shipping engines use

### Takeaway
C++ with metal-cpp is now a first-class, Apple-maintained option that fully covers Metal 4 (added with the macOS 26 SDK, updated for 26.4 and 27). Objective-C++ (.mm) remains what major open engines actually use for their Metal backends. Swift/C++ interop is officially promoted by Apple for games and is improving quickly (Swift 6.2 safe interop, Swift 6.4 std::span bridging), but it is aimed at mixing, not at writing the hot core. In Rust, metal-rs is deprecated in favour of objc2-metal, which does include MTL4 types.

### Cited Findings
- metal-cpp is "a low overhead and header only C++ interface" mapping 1:1 to the Metal Objective-C interfaces, and it requires C++17. — [apple/metal-cpp GitHub](https://github.com/apple/metal-cpp); [Apple metal-cpp page](https://developer.apple.com/metal/cpp/)
- metal-cpp changelog (verbatim entries): "macOS 27, iOS 27 — Add all the new Metal APIs in macOS 27 and iOS 27"; "macOS 26.4, iOS 26.4 — Add all the Metal APIs in macOS 26.4, iOS 26.4. Improvements and fixes."; "macOS 26, iOS 26 — Add all the Metal APIs in macOS 26, iOS 26, including support for the Apple10 GPU family. Add support for Metal 4 and new denoiser and temporal scalers in MetalFX." Earlier: MetalFX added with macOS 14; `NS::SharedPtr<T>` added with macOS 13. — [metal-cpp README](https://raw.githubusercontent.com/apple/metal-cpp/main/README.md)
- metal-cpp can be combined into one header with `./SingleHeader/MakeSingleHeader.py`. — [metal-cpp README](https://raw.githubusercontent.com/apple/metal-cpp/main/README.md)
- Maintenance signal: the GitHub repo is small (about 120 stars, 16 forks, about 10 commits on main). It is versioned per SDK and not per semver release. — [apple/metal-cpp GitHub](https://github.com/apple/metal-cpp)
- Apple's games "Get Started" page on C++: "Use this popular programming language when you need fine-grained control in performance-critical code. Note that it works only with CoreFoundation and other C-based frameworks…"; on interop: "With Swift-C++ interoperability, you can start using Swift and accessing all Apple frameworks, then move to C++ performance-critical and cross-platform portions of your game. Or start with C++ and integrate Swift as you adopt platform frameworks." It also says "Metal-cpp helps you add Metal functionality to graphics apps, games, and game engines written in C++." — [Apple Developer: Games Get Started](https://developer.apple.com/games/get-started/)
- Swift 6.2 introduced a "safe interoperability mode" that lets Swift use C/C++ pointers and view types such as std::span safely through header annotations. — [Swift 6.2 Released (swift.org)](https://www.swift.org/blog/swift-6.2-released/) (summarised via search)
- Swift 6.4, released on 2026-09-27, "transparently bridges C++20's std::span with Swift's Span". It also adds `UniqueBox` (a non-reference-counted smart pointer) and `UniqueArray` (no copy-on-write), plus borrowing accessors for Span and InlineArray. — [InfoQ, Swift 6.4](https://www.infoq.com/news/2026/09/swift-6-4-released/)
- The official Swift C++ interop documentation and its "Supported Features and Constraints" status page are maintained at swift.org. — [Mixing Swift and C++](https://www.swift.org/documentation/cxx-interop/); [Status](https://www.swift.org/documentation/cxx-interop/status/)
- Rust: gfx-rs/metal-rs is marked "Deprecated Rust bindings for Metal". Its README says to use objc2 and objc2-metal for new development, and says it will only receive basic maintenance while wgpu migrates to objc2. — [gfx-rs/metal-rs](https://github.com/gfx-rs/metal-rs)
- objc2-metal 0.3.2 (published 2026-08-04 according to docs.rs) includes MTL4 types such as `MTL4CommandQueue`, `MTL4Compiler`, `MTL4ArgumentTable`, `MTL4CommandBuffer` and `MTL4RenderCommandEncoder`. — [docs.rs objc2-metal](https://docs.rs/objc2-metal/latest/objc2_metal/)
- Godot's Metal rendering driver is implemented in Objective-C++ (`drivers/metal/rendering_device_driver_metal.mm`). Its README says portions were derived from MoltenVK and lists placement heaps, hazard tracking and MetalFX as future work. — [Godot drivers/metal README](https://github.com/godotengine/godot/blob/master/drivers/metal/README.md); [rendering_device_driver_metal.mm](https://cocalc.com/github/godotengine/godot/blob/master/drivers/metal/rendering_device_driver_metal.mm); [Godot PR #88199](https://github.com/godotengine/godot/pull/88199)
- There is an open Godot proposals discussion on adopting Metal 4 features. — [godot-proposals #13449](https://github.com/godotengine/godot-proposals/discussions/13449)

### Inferences
- For a new C++ engine, metal-cpp now covers the Metal 4 API surface (MTL4 command queues, allocators, argument tables, compiler, archives) from the macOS 26 SDK onward. Because it tracks every SDK release (26, 26.4, 27), it is effectively maintained in lockstep with the OS. Its low star and commit count reflects Apple publishing it as periodic drops, not neglect.
- Objective-C++ gives the most direct access to new APIs (no wait for a metal-cpp drop) and to AppKit, GameController and CAMetalLayer. Many engines use a thin .mm platform layer plus C++ for the renderer core. metal-cpp removes most of the need for .mm files in the renderer.
- Swift is viable for app shell, UI and tooling layers. For a performance-critical renderer core, C++ or Objective-C++ remains the lower-risk choice, given ARC/retain-release overhead and interop constraints (inference; I found no benchmarks).
- Rust via objc2-metal is feasible (MTL4 present), but the ecosystem is thinner for AAA use.

### Gaps
- I could not verify from primary sources which language Unreal Engine 5's or Unity's Metal RHI uses in 2026. It is commonly understood to be Objective-C++ for Unreal, but this is not verified here.
- I found no published benchmarks comparing Swift and C++ for engine hot paths on Metal.
- The exact metal-cpp release dates per SDK drop are not in the README.

## Shader language: MSL 4.x, Slang, HLSL via Metal Shader Converter, GLSL/SPIR-V via SPIRV-Cross

### Takeaway
MSL is the only path that gets every Metal 4 feature on day one: tensors and cooperative tensors (MSL 4.0), with the spec now at 4.1. Metal Shader Converter 4.0 (beta) handles HLSL/DXIL up to SM 6.6, including ray tracing and mesh shaders, but it requires Argument Buffers Tier 2 and ships only for macOS and Windows. Slang's Metal backend is active and improving (mesh shaders, RayQuery, printf, ParameterBlock mapped to argument buffers), but full Metal ray-tracing pipelines were still being implemented in July to Sept 2026. I found no evidence that Slang supports Metal 4 tensors.

### Cited Findings
**MSL**
- Apple hosts the "Metal Shading Language Specification Version 4.1". — [MSL spec PDF](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (title seen in search; the PDF was too large to fetch in full)
- Metal 4 adds tensor types and tensor operators to MSL (convolution, matmul, reduction) and cooperative tensors, "which your app's shader code can use when working with tensors and their data in parallel during any GPU stage". Cooperative tensors distribute data across threads in thread-private or threadgroup memory to reduce bandwidth. — [Apple docs: Machine-learning passes](https://developer.apple.com/tutorials/data/documentation/metal/machine-learning-passes.md) (via search snippet); [WWDC25 "Combine Metal 4 machine learning and graphics"](https://developer.apple.com/videos/play/wwdc2025/262/)
- Third-party analysis describes a `tensor_inline` (a non-owning shader-side view over a buffer), a host-bound `tensor_handle` with strides, and a `cooperative_tensor` whose layout the spec "deliberately leaves opaque". A forum thread discusses FP8/FP4 tensors in Metal 4.1. — [Rigel arXiv 2606.12765](https://arxiv.org/html/2606.12765v1); [TechBoards: FP8/FP4 tensors in Metal 4.1](https://techboards.net/threads/fp8-fp4-tensors-in-metal-4-1.5676/) (secondary sources; FP8/FP4 not verified against the spec)
- On paravirtual GPUs, cooperative-tensor shaders fail to compile ("error: unsupported deferred-static-alloca-size function body in metal::cooperative_tensor"), but they compile on physical M1 hardware. — [kornia #4207](https://github.com/kornia/kornia/issues/4207)

**Metal Shader Converter (HLSL → DXIL → metallib)**
- Current version: 4.0 (beta), which enables 2D compute derivatives and a NaN/Inf optimisation by default. Host requirements: macOS 13 with Xcode 15 or later, or Windows 10 with Visual Studio 2019 or later. Linux is not listed. Output requires devices with Argument Buffers Tier 2, and full features need macOS 14 / iOS 17 or later. It ships both a CLI and a C API library. — [Apple Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)
- Supported shader models: SM6.0 (wave intrinsics, 64-bit ints), 6.1 (barycentrics), 6.2 (16-bit), 6.3 (ray tracing), 6.4 (packed dot), 6.5 (mesh and amplification), 6.6 (dynamic resources, compute derivatives, IsHelperLane). It also supports function constants, framebuffer fetch and shader debug info. — [Apple Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)
- Apple says converted shaders are compatible with all Metal debugging and profiling tools, including debugging converted shaders. — [Apple Metal developer tools](https://developer.apple.com/metal/tools/)
- DXC issue proposing HLSL→MSL on macOS through DXC plus Metal Shader Converter. — [DXC #6057](https://github.com/microsoft/DirectXShaderCompiler/issues/6057)
- Critique (2023, now dated): Raph Levien's note on Metal Shader Converter discusses its closed-source, binary-only nature and its implications. — [Raph Levien blog](https://raphlinus.github.io/gpu/2023/06/12/shader-converter.html)

**Slang**
- Slang Metal target docs say `ParameterBlock` maps to Metal argument buffers ("potentially containing nested resources"). Mesh shaders are supported. The page lists `RaytracingAccelerationStructure` as unsupported, conservative rasterization as unsupported, and SubpassInput as fragment-only. It does not mention Metal 4 or tensors. — [Slang Metal-specific docs](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-02-metal-target-specific.html)
- Release notes contradict the "no ray tracing" line for inline ray queries. v2026.14.1 (2026) includes "[Metal] Fix RayQuery TriangleFrontFace emission" and "Add Metal support for printf". v2026.17.1 fixes Metal DispatchMesh and mesh index emission. v2026.18.3 (2026-09-25) "Emit noinline on Metal functions". — [Slang releases](https://github.com/shader-slang/slang/releases)
- Full Metal ray-tracing pipeline (intersection, closest-hit and miss stages) is in progress. The design issue #4576 was opened in 2024. "[Metal RayTracing]: Start the implementation – part 1" (#12241) was opened 2026-07-27. Part 2 (#13067) plans to port samples (Falcor2). Crash bugs in August 2026 involve the structural RT API. — [#4576](https://github.com/shader-slang/slang/issues/4576); [#12241](https://github.com/shader-slang/slang/issues/12241); [#13067](https://github.com/shader-slang/slang/issues/13067); [#12740](https://github.com/shader-slang/slang/issues/12740)
- Open Metal-backend correctness bugs as of 2026:
  - matrix multiply order emitted wrongly on Metal (#13141);
  - entry-point `uniform` pointer parameter dropped on struct-returning vertex shaders "on macOS with Apple Silicon and Metal 4 backend" (#11606);
  - mesh/fragment `[[user(...)]]` semantic name mismatch (#12997);
  - dual-source blend ignored on Metal (#8003).

  — [#13141](https://github.com/shader-slang/slang/issues/13141); [#11606](https://github.com/shader-slang/slang/issues/11606); [#12997](https://github.com/shader-slang/slang/issues/12997); [#8003](https://github.com/shader-slang/slang/issues/8003)
- A 2026 project (LichtFeld-Studio) building a Metal 4 tensor backend chose to port its Vulkan Slang shaders to hand-written MSL compiled at runtime, not to use Slang's Metal output. — [LichtFeld-Studio PR #2436](https://github.com/MrNeRF/LichtFeld-Studio/pull/2436)

**SPIRV-Cross / MoltenVK lineage**
- Godot's native Metal driver derives portions from MoltenVK, which is itself built on SPIRV-Cross for SPIR-V→MSL. — [Godot drivers/metal README](https://github.com/godotengine/godot/blob/master/drivers/metal/README.md)

### Inferences
- MSL-first (possibly with a thin macro or codegen layer) is the safest choice for an Apple-exclusive AAA engine. It gets tensors and cooperative tensors, Metal 4 argument tables and residency semantics, intersection function tables and new features in each OS release without waiting on a translator.
- Slang is attractive for multi-platform plus differentiable or neural shading. On Metal in 2026 it is still catching up: rasterization, compute, mesh and ray query work, but the RT pipeline is still under construction and there are open miscompilation bugs. Plan for per-feature validation and fallbacks to MSL.
- Metal Shader Converter suits HLSL-centric teams porting from D3D12. It emulates a D3D-style root signature and descriptor-heap model on top of Tier-2 argument buffers. Expect some overhead from that indirection and loss of direct access to Metal-only features (tensors, MSL-specific intrinsics). Performance impact is not quantified by Apple on the page fetched.
- SPIRV-Cross is mature for Vulkan-style GLSL/HLSL→MSL, but it translates to source MSL. It will not expose Metal 4-only constructs.

### Gaps
- I could not read the MSL 4.1 spec's revision-history table directly (the PDF exceeds the fetch limit). I did not verify an exact list of 4.0 vs 4.1 changes (for example, FP8/FP4 tensor element types) against the spec.
- I found no Apple-published performance comparison between MSC-converted and hand-written MSL.
- I found no evidence either way on whether Metal Shader Converter emits code targeting Metal 4 argument tables (MTL4ArgumentTable) or tensors.
- I did not fetch SPIRV-Cross's 2026 MSL-version support status.

## Offline shader compilation and pipeline caching in Metal 4

### Takeaway
Metal 4 moves compilation to an explicit `MTL4Compiler` object and adds a full ahead-of-time (AOT) pipeline workflow. You harvest pipeline descriptors at runtime into a `.mtl4-json` script, compile them offline with `metal-tt` plus Metal IR libraries into an archive, and load them via `MTL4Archive`, falling back to on-device compilation on a miss. Flexible (unspecialized) render pipelines and multithreaded, QoS-tuned compilation address stutter.

### Cited Findings
- "In Metal 4, shader compilation is performed with a compiler object rather than via the device interface". MTL4Compiler supports pipeline creation and serialization to reduce startup time. — [Metal by Example: Getting Started with Metal 4](https://metalbyexample.com/metal-4/); [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)
- Harvesting: create an `MTL4PipelineDataSetSerializer` with configuration `CaptureDescriptors`, attach it via `MTL4CompilerDescriptor.pipelineDataSetSerializer`, then call `serializeAsPipelinesScriptWithError`. The file suffix must be `mtl4-json`: "This is the suffix expected by the GPU toolchain." "A pipeline script is just a JSON formatted file." — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- "Feed your pipeline configuration script and Metal IR libraries to metal-tt. it will output the GPU binaries packed in a Metal archive." You must edit the Metal IR library paths in the script to match the build machine. — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- At runtime: `[device newArchiveWithURL:]` → `MTL4Archive`, then `newRenderPipelineStateWithDescriptor:`. "The lookup into the archive can miss for multiple reasons, such as no matching pipeline, incompatible OS, or incompatible GPU architecture. You need to handle such misses yourself in Metal 4." — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- Flexible render pipelines: set `MTLPixelFormatUnspecialized`, `MTLColorWriteMaskUnspecialized` and `MTL4BlendStateUnspecialized`, then call `newRenderPipelineStateBySpecializationWithDescriptor:pipeline:`. There is "a small GPU performance overhead… usually small, but can be large in some fragment shaders". Apple recommends compiling everything unspecialized first, then compiling important pipelines with full state in the background, identified via Metal System Trace. — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- Use `device.maximumConcurrentCompilationTaskCount` compile threads. Set the compile threads' QoS below the render thread's ("the hitch is gone"); Apple recommends `QOS_CLASS_DEFAULT` for prewarming. — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- Residency sets: "Prefer having fewer residency sets with more resources each". "A single argument table can be set on many encoder stages." — [WWDC25 Explore Metal 4 games](https://developer.apple.com/videos/play/wwdc2025/254/)
- Pre-Metal-4 background: Metal 3 introduced offline GPU binary generation from pipeline scripts and binary archives. — [WWDC22 Target and optimize GPU binaries with Metal 3](https://developer.apple.com/videos/play/wwdc2022/10102/)
- Toolchain packaging: since Xcode 26 the Metal toolchain (metal, metallib, air-lld) is unbundled from Xcode and is an optional download of about 700 MB (`xcodebuild -downloadComponent MetalToolchain`, or Settings › Components). There are known issues with visibility across users and on CI. — [Pol Piella: Metal Toolchain on CI/CD](https://www.polpiella.dev/metal-toolchain-ci-cd); [actions/runner-images #13080](https://github.com/actions/runner-images/issues/13080); [Apple forums 803174](https://developer.apple.com/forums/thread/803174); [OpenRadar FB20389216](https://openradar.appspot.com/FB20389216)

### Inferences
- Recommended pipeline: compile .metal sources to Metal IR (.metallib) at build time. Run a harvesting build or QA pass that emits `.mtl4-json`. Run `metal-tt` in CI per target GPU family and OS to produce archives. At runtime, load the archive first, fall back to unspecialized plus background full compile, and log misses to feed back into the harvest.
- Because archive misses can come from OS or GPU-architecture mismatch, shipping titles will likely need archives regenerated per major macOS release and per GPU family (Apple7 through Apple10). Treat on-device fallback as mandatory, not optional.
- CI must explicitly install the Metal toolchain component on Xcode 26+ images.

### Gaps
- I did not verify the exact `metal-tt` CLI flags or the list of target-architecture options.
- I did not confirm whether harvested archives can be built on Windows (with Metal Developer Tools for Windows) or only on macOS.

## Build systems, code signing and distribution (Mac App Store vs Steam)

### Takeaway
There is no Metal-specific build-system requirement. CMake (with the Xcode or Ninja generator) and native Xcode projects are both used. Signing requires hardened runtime plus notarization for Steam and direct distribution, and App Store signing for the Mac App Store. Sourcing on this topic is thinner and mostly secondary.

### Cited Findings
- Notarization for games: a missing hardened runtime is a common failure. Games often need explicit entitlements (JIT, unsigned executable memory, debugging-adjacent features), and mismatched entitlements are "a classic reason notarization fails". `notarytool` ships with modern Xcode. — [GamineAI: macOS notarization for Steam builds 2026](https://www.gamineai.com/blog/macos-notarization-stapling-ninety-minute-pass-unity-godot-steam-builds-2026) (secondary blog); [GameMaker: Notarizing your apps](https://gamemaker.io/en/help/articles/macos-notarizing-your-apps)
- Example of CMake-driven codesign and notarization of a macOS app. — [tony-go/codesign-macos](https://github.com/tony-go/codesign-macos); [giada issue #478 (hardened runtime in CMake)](https://github.com/monocasual/giada/issues/478)
- Apple forum threads on Xcode notarization with hardened runtime. — [Apple forums 129544](https://developer.apple.com/forums/thread/129544)

### Inferences
- Practical setup: a CMake source of truth that generates an Xcode project (`-G Xcode`) for debugging and GPU capture ergonomics, and Ninja for fast CI builds. Compile .metal files with custom commands (`xcrun metal` / `metallib`), or rely on the Xcode generator's native .metal handling.
- For Steam: sign with Developer ID plus hardened runtime, notarize, and staple. For the Mac App Store: App Sandbox is required, and file-system and JIT entitlements are constrained.

### Gaps
- I did not fetch CMake documentation on the Xcode generator's handling of `.metal` sources or `XCODE_ATTRIBUTE_*` signing properties.
- I did not verify Steamworks' official macOS notarization requirements or Mac App Store sandbox constraints for games from primary sources.

## Debugging and profiling

### Takeaway
Apple's first-party stack is comprehensive and Metal-4-aware: the Xcode Metal debugger (including tensor inspection), shader debugger, performance timeline with counters, heat maps and cost graphs, Metal System Trace and the Game Performance template in Instruments, Metal Performance HUD, and API and shader validation. New command-line tools (`gpucapture`, `gpudebug`, `metalperftrace`) help automation. RenderDoc does not support Metal. Tracy supports Metal GPU zones since v0.12.0. Superluminal does not support macOS.

### Cited Findings
- The Metal debugger inspects "rendering, compute, and machine learning pipelines", including buffers, textures, tensors and ray-tracing acceleration structures. It offers shader debugging with variable inspection, a dependencies viewer and memory reports. There is a performance timeline with hardware counters, shader execution heat maps and a shader cost graph. Runtime API and shader validation (for example, bounds checking) is included. — [Apple Metal developer tools](https://developer.apple.com/metal/tools/)
- New command-line tools: `gpucapture` (attach to a running process and capture), `gpudebug` (debug and profile from the CLI), and `metalperftrace` (performance traces with human-readable summaries). — [Apple Metal developer tools](https://developer.apple.com/metal/tools/)
- "Analyze slowness in your game's frame rate with the Game Performance template in Instruments, which combines threading and system call information with the Metal System Trace instrument." "Overlay the Metal Performance HUD on your game to view CPU and GPU metrics." "Run your game in Xcode to validate your Metal code and catch shader execution errors." — [Apple Games Get Started](https://developer.apple.com/games/get-started/)
- RenderDoc FAQ: supports "Vulkan 1.4, D3D11 (up to D3D11.4), D3D12, OpenGL 3.2+, and OpenGL ES 2.0 – 3.2". It says "Future API support is at this point not clear; Metal, WebGL, and perhaps D3D9/D3D10 all being possible." The macOS tracking issue says Metal/iOS support "is not planned at the moment". — [RenderDoc FAQ](https://renderdoc.org/docs/getting_started/faq.html); [RenderDoc #1272](https://github.com/baldurk/renderdoc/issues/1272)
- A "RenderDoc Meta Fork for Mac" exists from Meta, oriented to Quest/Android device tracing, not Metal. — [Meta for Developers](https://developers.meta.com/horizon/downloads/package/renderdoc-meta-fork-for-mac-installer/)
- Tracy NEWS: v0.12.0 (2025-05-30): "GPU profiling is now available with Metal and CUDA." v0.13.0 (2025-11-11): "Prototype implementation of system tracing on Apple devices." v0.14.0 (2026-08-09): "Tracing on Arm macOS will now have more precise timer readings." — [Tracy NEWS](https://raw.githubusercontent.com/wolfpld/tracy/master/NEWS)
- The Tracy Metal implementation has constraints: timestamp query buffers are limited to 4096 queries, with restrictions on resetting. — [Tracy PR #793](https://github.com/wolfpld/tracy/pull/793) (via search snippet)
- Superluminal supports Windows, Xbox, PlayStation and (per its site) Linux targets. macOS is not listed. — [Superluminal supported platforms](https://superluminal.eu/applications/); [Superluminal Linux](https://superluminal.eu/applications/linux/)

### Inferences
- Workflow: Tracy for continuous CPU plus GPU-zone instrumentation across platforms. Xcode GPU capture and the shader profiler for per-frame deep dives. Instruments (Metal System Trace / Game Performance) for stutter and CPU–GPU overlap. The Performance HUD in playtests. Validation layers always on in debug CI. `gpucapture` and `metalperftrace` enable scripted captures in automated perf tests on physical Macs.
- There is no RenderDoc/PIX equivalent outside Apple's tools for Metal, so the engine should be designed to be capture-friendly (debug labels, `MTLCaptureManager` programmatic captures).

### Gaps
- I did not verify which macOS/Xcode version introduced `gpucapture`, `gpudebug` and `metalperftrace` (likely Xcode 26 or 27).
- I did not confirm whether Tracy's Metal back-end already supports MTL4 command buffers and counter heaps, or only the legacy MTLCommandBuffer path.

## Testing without a GPU: Linux/Windows compilation and macOS CI

### Takeaway
You can compile shaders off-Mac only partially. The Apple Metal compiler and metallib, and Metal Shader Converter, run on Windows as well as macOS, but not officially on Linux. Slang can emit MSL source anywhere. GitHub-hosted Apple Silicon runners expose only a paravirtualized "Apple M1 (Virtual)" GPU, which has feature gaps (it cannot compile Metal 4 cooperative-tensor shaders, and `newArgumentEncoderWithLayout:` is missing) and non-representative timing. Real Metal 4 testing needs self-hosted or physical Macs.

### Cited Findings
- Metal Developer Tools for Windows include the Metal compiler, metallib, and headers and libraries for building Metal shader programs. Linux is not mentioned. — [Apple Metal developer tools](https://developer.apple.com/metal/tools/)
- Metal Shader Converter runs on macOS 13+ and Windows 10+. Linux is not listed. — [Apple Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)
- GitHub-hosted macOS runners: arm64 labels `macos-latest`, `macos-14`, `macos-15`, `macos-26` and `xcode-27` (public preview), with 3 CPUs (M1) and 7 GB RAM. Intel labels are `macos-15-intel` and `macos-26-intel`. "Nested-virtualization is not supported due to the limitation of Apple's Virtualization Framework." GPU and Metal are not documented. — [GitHub Docs: GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- On `macos-26-arm64`, the paravirtual GPU ("Apple M1 (Virtual)") failed to compile Metal 4 cooperative-tensor shaders, and 664 of 670 tests failed. On a physical M1 with macOS 26.5 the same shaders compiled. The project pinned GPU tests to a `macos-15` (Metal 3) job. The issue was opened 2026-09-03. — [kornia #4207](https://github.com/kornia/kornia/issues/4207)
- Godot's Metal driver crashes on the paravirtual device: "-[AppleParavirtDevice newArgumentEncoderWithLayout:]: unrecognized selector". The paravirtual GPU reports as Apple5 family and is "too limited" for Forward+. — [Godot #101773](https://github.com/godotengine/godot/issues/101773)
- A long-standing request for GPU passthrough (Metal) on GitHub-hosted macOS runners remains open. — [actions/runner-images #7085](https://github.com/actions/runner-images/issues/7085)
- Commentary that hosted runners are usable for Metal correctness tests but not for performance benchmarks. — [VeloxQuant-MLX #395](https://github.com/rajveer43/VeloxQuant-MLX/issues/395) (low-authority source)
- The Xcode 26 Metal toolchain was missing on runner images (a component download is needed). — [actions/runner-images #13080](https://github.com/actions/runner-images/issues/13080)

### Inferences
- Linux CI can run Slang→MSL source generation, SPIRV-Cross, and static checks. Producing `.metallib` or `.air` requires macOS or Windows (Apple's tools). Linux is possible only through unofficial means (not researched).
- GitHub-hosted macOS runners are fine for building, signing, notarization and offline metallib compilation (after installing the Metal toolchain). They are unreliable for Metal 4 runtime tests, since the paravirtual Apple5-class GPU lacks many Apple7+ / Metal 4 features. For GPU tests and perf regression, use self-hosted Apple Silicon Macs (M-series) or Mac cloud providers with bare-metal hosts.

### Gaps
- I did not verify whether `metal-tt` (archive generation) or `gpucapture` run on Windows or in CI VMs.
- I did not research third-party macOS CI providers (bare-metal Mac hosting) in terms of GPU exposure and pricing.
- I did not verify whether any Apple tool runs on Linux (for example, under Wine or via Docker) as an official offering. None was found.
