# Open-source / free middleware for an Apple-only AAA engine (macOS on Apple Silicon, iPadOS/iOS) — state as of Sept 2026

Methodology note: GitHub's REST API was blocked in this session, so versions come from `git ls-remote --tags` against each public repo (run 2026-09-28), and licenses/platform claims come from shallow clones of each repo (LICENSE files, README, release notes). Where a fact comes from background knowledge and was not re-checked this session, it is marked **[not re-verified]**. The dates are commit dates of the tagged commit or HEAD.

## Physics: Jolt, PhysX 5, Box2D v3, Bullet

### Takeaway
Jolt Physics is the clear first choice. It is MIT-licensed, uses NEON on ARM64, officially supports macOS and iOS on ARM64, and the simulation is always deterministic in the development branch. v5.6.0 (July 2026) also added a GPU compute layer with a Metal backend. PhysX 5 is open source (BSD-3), but its current release supports only Linux and Windows at runtime, so it is effectively unavailable on Apple platforms. Box2D v3 (MIT, NEON SIMD) is the pick for 2D.

### Cited Findings
- **Jolt: license and version.** MIT license, "Copyright 2021 Jorrit Rouwe". Latest tags: v5.4.0, v5.5.0, v5.6.0. The v5.6.0 tagged commit is dated 2026-07-11, and master was active on 2026-09-28. — [JoltPhysics repo](https://github.com/jrouwe/JoltPhysics)
- **Jolt: platforms.** The README lists "macOS x64/ARM64" and "iOS x64/ARM64", and says "On ARM64 (AArch64) the library uses NEON and is compatible with Armv8-A." Samples, TestFramework and the viewer run on Windows, macOS and Linux. — [Jolt README](https://github.com/jrouwe/JoltPhysics/blob/master/README.md)
- **Jolt: production users.** The README says: "Used by Horizon Forbidden West and Death Stranding 2: On the Beach". ProjectsUsingJolt.md also lists Gaijin's open-source Dagor Engine (War Thunder). There is a GDC 2022 talk, "Architecting Jolt Physics for 'Horizon Forbidden West'". — [Jolt README](https://github.com/jrouwe/JoltPhysics/blob/master/README.md), [ProjectsUsingJolt.md](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/ProjectsUsingJolt.md), [GDC Vault](https://gdcvault.com/play/1027560/Architecting-Jolt-Physics-for-Horizon)
- **Jolt: determinism.** The README says: "The simulation runs deterministically. You can replicate a simulation to a remote client by merely replicating the inputs." The unreleased notes (after v5.6.0) say: "Removed `PhysicsSettings::mDeterministicSimulation`… The simulation is now always deterministic." v5.6.0 also "Fixed an issue that broke cross platform determinism between ARM64 and x64 builds when compiling with double precision." — [Jolt ReleaseNotes.md](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/ReleaseNotes.md)
- **Jolt v5.6.0 new features:**
  - An "interface to run compute shaders on the GPU with implementations for DX12, Vulkan and Metal" (flags `JPH_USE_MTL` and others). Building on macOS requires dxc and spirv-cross.
  - GPU strand-based hair simulation (Cosserat rods; still a work in progress).
  - A new, cheaper friction model.
  - "Up to 40% performance improvement … up to 70% memory reduction" (scene dependent).
  - Support for glTF `KHR_physics_rigid_bodies` constraint motors.
  - Source: [Jolt ReleaseNotes.md](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/ReleaseNotes.md)
- **PhysX: version and platforms.** The repo's latest SDK is `physx/version.txt` = 5.11.0. The CHANGELOG for v5.11.0 lists these supported runtime platforms:
  - "Linux (tested on Ubuntu LTS 22.04, 24.04)"
  - "Microsoft Windows 10 or later (64 bit)"
  - GPU acceleration through CUDA 12.8 on Volta or newer GPUs

  The public build presets are only linux, linux-aarch64 and vc16/vc17win64. There is no macOS or iOS preset. Other tags in the repo are `ovphysx-0.6.3` (Omniverse) and `release/104.0`. The README also says GPU binaries are downloaded on demand from CloudFront via packman. — [NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX)
- **PhysX license.** BSD-3-Clause **[not re-verified]**: LICENSE.md is present, but its text was not inspected this session. — [PhysX repo](https://github.com/NVIDIA-Omniverse/PhysX)
- **Box2D.** MIT (Copyright 2022 Erin Catto). Latest tag v3.1.1, whose commit is dated 2025-06-03; master was active on 2026-09-24. The README says: "Box2D uses SSE2 and Neon (AArch64) SIMD math … can be disabled by defining `BOX2D_DISABLE_SIMD`". Other listed features are "Extensive multithreading and SIMD" and continuous collision. Build presets cover Xcode. — [box2d repo](https://github.com/erincatto/box2d)
- **Bullet.** Latest tag is 3.25 (tags 3.23, 3.24, 3.25). — [bullet3](https://github.com/bulletphysics/bullet3)

### Inferences
- The PhysX 5 source may be portable in principle (it historically had a macOS CPU build in PhysX 4). However, NVIDIA ships no macOS or iOS preset in 5.x, and GPU simulation depends on CUDA, which is unavailable on Apple Silicon. A port would mean maintaining your own fork. Recommendation: Jolt for 3D, Box2D v3 for 2D.
- Jolt's new Metal compute interface (v5.6.0) suggests that future GPU features (hair, possibly soft bodies or cloth) will run on Apple GPUs without a porting layer. It needs dxc and spirv-cross at build time, so the shaders are cross-compiled from HLSL.
- Bullet appears to be in maintenance mode. The last tag is 3.25; I could not confirm the date of any recent activity. It is not recommended for a new AAA engine.

### Gaps
- Could not confirm Box2D v3's cross-platform determinism guarantees from the README grep. Its docs say it has them, but I did not verify this here.
- Could not check the date of the last Bullet commit.

## Geometry processing: meshoptimizer (meshlets / cluster LOD), MikkTSpace, xatlas

### Takeaway
meshoptimizer is MIT-licensed and very active: v1.3 is dated 2026-09-25, and v1.0 shipped in Dec 2025. It is the de-facto standard for this work and now covers a complete "Nanite-style" pipeline:
- meshlet building, including spatial/SAH clusters for ray tracing
- cluster partitioning
- `clusterlod.h`, a single-header cluster-LOD hierarchy builder
- meshlet compression (the `meshopt_encodeMeshlet` codec)
- an experimental voxel remesher
- opacity maps

For tangent space, MikkTSpace remains the standard. meshoptimizer's src now also contains a `tangentspace.cpp`.

### Cited Findings
- **meshoptimizer: license and versions.** MIT ("Copyright (c) 2016-2026 Arseny Kapoulkine"). Recent tags are v1.1.1, v1.2 and v1.3. The v1.3 commit is dated 2026-09-25 and v1.0 is dated 2025-12-06. — [meshoptimizer repo](https://github.com/zeux/meshoptimizer)
- **clusterlod.h.** The README says: "clusterlod.h, a single-header C/C++ library for continuous level of detail using clustered simplification". The header describes itself as "a small 'library'/example built on top of meshoptimizer to generate cluster LOD hierarchies… intended to either be used as is, or as a reference". Its config options map to `meshopt_buildMeshlets*`, `meshopt_partitionClusters` (with a spatial partition option), and `meshopt_buildMeshletsSpatial` / `meshopt_buildMeshletsFlex`. The demo folder contains `nanite.cpp` and `meshletdec.slang`. — [meshoptimizer README](https://github.com/zeux/meshoptimizer/blob/master/README.md), [demo/clusterlod.h](https://github.com/zeux/meshoptimizer/blob/master/demo/clusterlod.h)
- **Ray-tracing clusters.** `meshopt_buildMeshletsSpatial` "builds clusters using surface area heuristic (SAH) to produce raytracing-friendly cluster distributions". The README discusses cluster acceleration structures (NV). — [README](https://github.com/zeux/meshoptimizer/blob/master/README.md)
- **Meshlet compression.** The codec is `meshopt_encodeMeshlet`. The README says: "Meshlet data uses a frameless format without an embedded version header, so it's compatible across all library versions (starting with meshoptimizer v1.1)." — [README](https://github.com/zeux/meshoptimizer/blob/master/README.md)
- **Voxel remeshing.** The `meshopt_remesh` API (options `meshopt_RemeshShell` and `meshopt_RemeshSolve`) is marked: "This feature is still experimental and is subject to change". — [README](https://github.com/zeux/meshoptimizer/blob/master/README.md)
- **Other source files.** src/ also includes `opacitymap.cpp`, `partition.cpp`, `spatialorder.cpp`, `rasterizer.cpp`, `meshletcodec.cpp` and `tangentspace.cpp`. — [meshoptimizer/src](https://github.com/zeux/meshoptimizer/tree/master/src)
- **Proprietary-platform index limits.** The README says: "Some proprietary platforms have additional restrictions on the index range that can be referenced by a meshlet. Building the library with `MESHOPTIMIZER_CLUSTERIZER_INDEXLIMIT` defined will ensure…" — [README](https://github.com/zeux/meshoptimizer/blob/master/README.md)
- **xatlas.** The repo has no git tags, which means no formal releases. — [jpcy/xatlas](https://github.com/jpcy/xatlas)
- **MikkTSpace.** The repo is at [mmikk/MikkTSpace](https://github.com/mmikk/MikkTSpace). Its zlib-style license and dormant status are **[not re-verified]**. It is still the tangent basis used by Blender, Unreal and the glTF spec.

### Inferences
- On Apple GPUs that support Metal mesh shaders (Apple7+/M1 and later), meshoptimizer meshlets plus clusterlod.h are the fastest route to a virtualized-geometry pipeline. The comment "NVidia 64/126 recommended" does not apply to Apple GPUs, so tune `max_vertices` and `max_triangles` empirically.
- The fact that `nanite.cpp` and the spatial partitioning live in the core repo suggests that cluster LOD is now a first-class focus of the project.

### Gaps
- Could not confirm a detailed changelog for v1.2 and v1.3. The repo has no CHANGELOG file; release notes are on GitHub Releases, which could not be reached through the API.
- Did not check whether xatlas has had any commits in 2025–2026.

## Asset import and formats: glTF, OpenUSD, texture compression

### Takeaway
For glTF:
- **fastgltf** (MIT, C++20, simdjson, ARM CI) is the fastest modern parser.
- **cgltf** (single-header C) is the simplest.
- **tinygltf** v3 is also available.

OpenUSD is very active (v26.08) and matters on Apple platforms because RealityKit, Reality Composer Pro and Quick Look are built around USD.

For textures:
- ARM's **astcenc** 5.7.0 (Apache 2.0) ships a macOS universal binary with a NEON build.
- **Basis Universal** v2.50 (Apache 2.0) now adds XUASTC, a supercompressed ASTC format with all 14 block sizes, plus ASTC HDR 6x6. This fits Apple GPUs, which support ASTC natively.
- **KTX-Software** 5.0 is at release-candidate stage (rc2).

### Cited Findings
- **fastgltf.**
  - License and dependencies: MIT (2022–2026 Sean Apeler); it depends on simdjson (Apache 2.0).
  - Language: "written in modern C++20"; also available as a C++20 named module. "For C++17 compatibility, please use v0.9.x. Later versions require C++20."
  - CI and activity: it has a "CI ARM" workflow. Latest tag is v0.9.1, and master was active on 2026-09-28.
  - Source: [fastgltf](https://github.com/spnda/fastgltf)
- **cgltf and tinygltf versions.** cgltf's latest tag is v1.15. tinygltf's latest tags are v3.0.0 and v3.0.1. Their MIT licenses are **[not re-verified]**. — [cgltf](https://github.com/jkuhlmann/cgltf), [tinygltf](https://github.com/syoyo/tinygltf)
- **OpenUSD.** Latest tags are v26.03, v26.05 and v26.08, a roughly bimonthly cadence. The license is the modified Apache 2.0 "Tomorrow Open Source Technology License" **[not re-verified]**. — [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD)
- **astcenc.**
  - Apache 2.0; latest tag 5.7.0, with HEAD dated 2026-08-29.
  - The README lists builds for `astcenc-sve_256`, `astcenc-sve_128` and `astcenc-neon`.
  - The README says: "For macOS, we provide a single universal binary `astcenc`", where the arm64 slice "uses the `astcenc-neon` build".
  - Source: [astc-encoder](https://github.com/ARM-software/astc-encoder)
- **Basis Universal.**
  - The README header reads "basis_universal v2.5"; the latest tag is v2_50. The encoder library is Apache 2.0, and some third-party parts carry their own licenses (see the DEP5 file).
  - Formats: ETC1S, UASTC LDR 4x4, UASTC HDR 4x4, ASTC HDR 6x6, UASTC HDR 6x6 Intermediate, ASTC LDR 4x4–12x12, **XUASTC LDR 4x4–12x12** ("A latent-space GPU texture codec supporting ASTC LDR supercompression") and **XUBC7**.
  - Transcoding targets include ASTC LDR 4x4–12x12.
  - Source: [basis_universal](https://github.com/BinomialLLC/basis_universal)
- **KTX-Software.** Latest tags are v4.4.2, v5.0.0-rc1 and v5.0.0-rc2. — [KTX-Software](https://github.com/KhronosGroup/KTX-Software)

### Inferences
- Every Apple GPU from the A8 onward supports ASTC LDR, and M-series and A13+ GPUs support ASTC HDR. That makes a pure-ASTC pipeline viable on Apple: astcenc offline, or Basis XUASTC for supercompressed distribution. BCn (BC1–7) is also supported on Apple Silicon Macs, so desktop assets could use BC7 through XUBC7.
- Apple's own texture tools (the `texturetool` CLI, Metal texture converters, and MTLIO / Metal fast resource loading) exist, but none of them replaces astcenc or Basis for quality control. **[not re-verified this session]**
- A reasonable asset policy: glTF (fastgltf) as the engine's interchange format and USD as an optional bridge to Apple's tools (Reality Composer Pro) and DCC pipelines. OpenUSD is heavy (Boost and TBB historically), so keep it in tools only.

### Gaps
- Apple's texture-tool documentation and Metal 4's texture compression support were not fetched.
- License texts for cgltf, tinygltf and KTX were not inspected.

## Animation: ozz-animation, ACL, motion matching

### Takeaway
- **ozz-animation** (MIT, 0.17.0) provides a runtime and offline toolchain (glTF/FBX import) tested on macOS and ARM.
- **ACL** (MIT, v2.1.0) is the reference for animation compression. It is shipped inside Unreal Engine, and its CI tests macOS on ARM64.
- There is no mature, production-grade open-source motion matching library in C++. The best reference is Daniel Holden's (orangeduck) implementation, and the feature would realistically be built in-house.

### Cited Findings
- **ozz-animation: license, version and platforms.** MIT; latest tags 0.15.0, 0.16.0 and 0.17.0, with HEAD dated 2026-08-01. The README says it "is tested on WebAssembly, Linux, macOS and Windows, for x86, x86-64 and ARM architectures." The runtime (ozz_base, ozz_animation, ozz_geometry) depends only on C++17. — [ozz-animation](https://github.com/guillaumeblanc/ozz-animation)
- **ozz-animation: toolchain.** It "comes with the toolchain to convert from major Digital Content Creation formats (gltf, Fbx, Collada, Obj, 3ds, dxf)". — [ozz-animation](https://github.com/guillaumeblanc/ozz-animation)
- **ACL: license, version and platforms.**
  - MIT (2017 Nicholas Frechette & contributors); latest tag v2.1.0. HEAD is dated 2025-09-13, a slower cadence than the others.
  - It is 100% header-only.
  - CI covers "OS X XCode 15+: ARM64" and Windows ARM64. "Each release is also manually tested on iOS and Android."
  - Source: [ACL](https://github.com/nfrechette/acl)
- **ACL in Unreal Engine.** The README says Unreal support "is now distributed with a more recent version as part of each engine release since UE 5.13". "5.13" is most likely a typo in the README for UE 5.3. — [ACL README](https://github.com/nfrechette/acl)
- **Motion matching references:**
  - orangeduck/Motion-Matching: "Learned Motion Matching example implementation and source code for the article 'Code vs Data Driven Displacement'" (raylib demo).
  - Other hobby projects: dreaw131313/Open-Source-Motion-Matching-System, nashnie/MotionMatching, and an Unreal parkour sample rewrite.
  - Source: [orangeduck/Motion-Matching](https://github.com/orangeduck/Motion-Matching), [GitHub topic motion-matching](https://github.com/topics/motion-matching)

### Inferences
- Stack: ozz for the runtime (sampling, blending, IK) and ACL for compression. A custom motion-matching layer (feature database plus KD-tree or brute-force search, which can use NEON or Accelerate/BNNS) would follow Holden's reference code.

### Gaps
- orangeduck's license was not verified (a permissive license is believed but not confirmed).
- I found no GDC talk that confirms production use of ozz in a shipped AAA title.

## Audio: Apple frameworks vs FMOD/Wwise vs miniaudio vs Steam Audio

### Takeaway
- **Apple native:** AVAudioEngine for mixing plus PHASE for geometry-aware spatial audio. Both are free and zero-dependency, and PHASE was built by Apple for games.
- **FMOD:** free Indie license below $200K yearly revenue and a $500K budget.
- **Wwise:** free tiers are limited (a 200-asset trial for non-commercial projects; an indie model tied to budget); commercial tiers are priced by budget.
- **miniaudio:** a single-file low-level engine.
- **Steam Audio:** Apache 2.0 and open source. It supports macOS and iOS and ships FMOD and Wwise plugins.

### Cited Findings
- **PHASE.** Apple introduced PHASE at WWDC21 for "geometry-aware audio… complex, interactive, and immersive audio scenes for apps and games". Its concepts are Sources, Listeners, Acoustic Geometry and Materials. — [WWDC21 session 10079](https://developer.apple.com/videos/play/wwdc2021/10079/)
- **PHASE in Godot.** A search-engine summary claimed Godot gained a PHASE audio plugin around WWDC 2026. I could not verify this against a primary source, so treat it as unconfirmed. — [Apple Games videos](https://developer.apple.com/videos/graphics-and-games/games)
- **FMOD.** The Indie license is free for developers "whose yearly revenue is less than $200,000" and who have "less than $500K USD in funding/budget". It includes all features except email support; gambling and simulation projects are excluded. — [Game Developer](https://www.gamedeveloper.com/audio/small-developers-and-creators-can-now-use-fmod-studio-for-free), [GameFromScratch](https://gamefromscratch.com/fmod-studio-now-free-for-indie-game-developers/). The primary fmod.com/licensing page did not render in this session.
- **Wwise: free trial and indie model.**
  - The free trial or non-commercial license is "limited to 200 media assets in the Wwise SoundBanks".
  - Audiokinetic introduced "a free licensing model… for smaller studios and individual developers, with no asset limit".
  - Source: [Audiokinetic pricing for games](https://www.audiokinetic.com/pricing/for-games/), [Audiokinetic blog: Free Wwise for indie developers](https://blog.audiokinetic.com/en/free-wwise-for-indie-developers/)
- **Wwise: commercial tiers.**
  - Indie covers budgets up to $250K and Pro up to $2M; Premium and Platinum apply above $2M.
  - There is a 1% post-launch royalty option and a Games-as-a-Service option.
  - The pricing page returned 403 to direct fetch, so these figures come from the search snippet.
  - Source: [Audiokinetic pricing](https://www.audiokinetic.com/en/wwise/pricing/)
- **Steam Audio.** Apache 2.0 (LICENSE.md). Latest tag is v4.8.1, with HEAD dated 2026-03-25. The README says it "supports Windows…, Linux…, macOS, Android…, and iOS platforms". The repo contains `fmod`, `wwise`, `unity` and `unreal` integration folders. — [steam-audio](https://github.com/ValveSoftware/steam-audio)
- **miniaudio.** Latest tag is 0.11.25. Its dual public-domain / MIT-0 license and CoreAudio backend are **[not re-verified]**. — [miniaudio](https://github.com/mackron/miniaudio)
- **Apple audio-porting guidance.** Apple publishes "Porting your audio code to Apple silicon", linked from its P/E-core guidance. — [Apple developer news](https://developer.apple.com/news/?id=vk3m204o)

### Inferences
- For an Apple-only engine, AVAudioEngine + PHASE (native spatial audio, personalized HRTF and AirPods head tracking) + Steam Audio (for geometric occlusion/reverb, if PHASE's acoustics prove insufficient) avoids any commercial audio license.
- FMOD or Wwise mainly add sound-designer authoring tools. That matters for AAA, but it costs money above the indie thresholds.
- miniaudio is a good portable low-level fallback. On an Apple-only engine, AVAudioEngine or AudioUnit covers the same ground.

### Gaps
- The Steam Audio README says "macOS" without naming arm64 or universal binaries. I did not confirm native Apple Silicon binaries, though it is very likely.
- I did not find PHASE's WWDC25/26 updates or evidence of PHASE use in a shipped AAA game.
- Exact 2026 Wwise prices were not fetched.

## Job systems / task schedulers, and P/E-core scheduling

### Takeaway
All three main candidates are active and permissively licensed:
- **enkiTS** (zlib, v1.12): has task priorities and pinned tasks.
- **Taskflow** (MIT, v4.1.0): C++20, graph-based.
- **GCD** (`dispatch_apply` / `concurrentPerform`): Apple's native option.

marl has no tags. Apple's guidance is to use QoS classes, split work into many pieces with work stealing, and use at least 3x the core count in iterations. Background-QoS threads are confined to E-cores.

### Cited Findings
- **enkiTS.**
  - License: zlib (License.txt: "Copyright (c) 2013 Doug Binks… provided 'as-is'").
  - Version: latest tag v1.12, with HEAD dated 2026-09-02.
  - Platforms: "Windows, Linux, Mac OS, Android (should work on iOS)" on x64/x86/ARM, though it is "somewhat less frequently tested… on Mac OS".
  - Features: up to 5 task priorities, pinned tasks, and waiting on pinned tasks (for IO threads).
  - Users: Avoyd.
  - Source: [enkiTS](https://github.com/dougbinks/enkiTS)
- **Taskflow.** MIT ("Copyright (c) 2018-2026 Dr. Tsung-Wei Huang"). Tags are v3.11.0, v4.0.0 and v4.1.0. The examples compile with `-std=c++20`. It has a macOS CI workflow, and its GPU tasking is CUDA-only (CUDA Graph), which is unusable on Apple. — [Taskflow](https://github.com/taskflow/taskflow)
- **marl.** `git ls-remote --tags` returned no tags. — [google/marl](https://github.com/google/marl)
- **Apple on QoS.** Apple says: "QoS classes are the primary way for you to categorize work… and provide the OS with semantic information." Parallel work should "subdivide parallel problems into a large number of pieces and use a work-stealing algorithm", and should "use the `concurrentPerform` / `dispatch_apply` API… Set the number of iterations to at least three times the total number of cores". — [Apple: Optimize for Apple Silicon with performance and efficiency cores](https://developer.apple.com/news/?id=vk3m204o)
- **QoS and core placement.** Background QoS (9) threads "are run on E cores, and can't be promoted to run on P cores". Threads at QoS 17 or higher prefer P-cores but spill over to E-cores when the P-cores are busy. — [Eclectic Light: What is QoS](https://eclecticlight.co/2025/05/09/what-is-quality-of-service-and-how-does-it-matter/), [Eclectic Light: Can you game core allocation](https://eclecticlight.co/2022/11/28/can-you-game-core-allocation-on-apple-silicon/)
- **Game Mode.** Apps that declare a game category in Info.plist get Game Mode, which gives "highest priority access" to CPU cores and reduces background use of E-cores. This is a secondary source summarizing Apple's behavior. — [Eclectic Light](https://eclecticlight.co/2022/11/28/can-you-game-core-allocation-on-apple-silicon/)
- **Thread affinity.** Apple's developer forums confirm that you cannot bind threads directly to P- or E-cores; QoS is the lever. — [Apple Developer Forums thread 674456](https://developer.apple.com/forums/thread/674456)

### Inferences
- For a custom job system on Apple, create worker threads with `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` for frame-critical workers and `UTILITY` or `BACKGROUND` for streaming and compilation.
- Size the worker pool to the P-core count (read from the `hw.perflevel0.physicalcpu` sysctl) or to all cores, and rely on work stealing. The asymmetric core speeds make static partitioning bad.
- Avoid long spin-waits: they waste the energy budget and can hurt P-core boost on iPad and iPhone.
- enkiTS's priorities map naturally onto QoS tiers.

### Gaps
- The Apple documents "Tuning your code's performance for Apple silicon" and the Game Mode docs were not fetched directly.
- marl's archival or maintenance status is unconfirmed.

## ECS: EnTT vs flecs

### Takeaway
Both are MIT-licensed and active.
- **EnTT** (header-only C++) just tagged **v4.0.0**. It is used in Minecraft (Bedrock), Minecraft Legends and Minecraft Earth.
- **flecs** (C99 core plus a C++17 API, v4.1.6) adds queries, relationships, a scripting module and the web-based Flecs Explorer. It is used in Tempest Rising.

### Cited Findings
- **EnTT.** MIT ("Copyright (c) 2017-2026 Michele Caini"). Latest tags are v3.15.0, v3.16.0 and v4.0.0, with HEAD dated 2026-09-17. It requires CMake 3.28 or later. The README says it is "used in **Minecraft** by Mojang"; the links page adds Minecraft Legends and Minecraft Earth. — [EnTT](https://github.com/skypjack/entt)
- **flecs.** MIT; latest tag v4.1.6, with HEAD dated 2026-09-12. It offers a "zero dependency C99 API" and a "Modern type-safe C++17 API that doesn't use STL containers". The Flecs Explorer is a web tool. Tempest Rising is listed as a project using it. — [flecs](https://github.com/SanderMertens/flecs)

### Inferences
- EnTT 4.0 is a major version, so expect API breaks relative to 3.x tutorials.
- flecs's built-in explorer, reflection and serialization reduce editor-tooling work.

### Gaps
- EnTT 4.0's changes and minimum C++ standard were not examined.

## Networking: GameNetworkingSockets, Apple Network framework, GameKit

### Takeaway
**GameNetworkingSockets** (BSD-3, v1.6.0) provides reliable and unreliable messages over UDP, encryption, and P2P with ICE/WebRTC NAT traversal. It has macOS and iOS CMake presets and CI. On macOS you must supply your own OpenSSL: don't use the system LibreSSL.

### Cited Findings
- **GNS: license, version and platforms.**
  - License: BSD-style ("Copyright (c) 2018, Valve Corporation. All rights reserved.").
  - Version: tags v1.5.0, v1.5.1 and v1.6.0, with HEAD dated 2026-08-26.
  - Platforms: CI badges cover macOS, iOS, Ubuntu and Windows. BUILDING.md mentions "CMake 3.21 or later to use the macOS/iOS presets".
  - Source: [GameNetworkingSockets](https://github.com/ValveSoftware/GameNetworkingSockets)
- **GNS: features.** "NAT traversal through google WebRTC's ICE implementation" with pluggable signaling. It "has shipped on consoles, mobile platforms, and non-Steam stores". Steam-only services (SDR relay, authentication) are not included. — [GNS README](https://github.com/ValveSoftware/GameNetworkingSockets/blob/master/README.md)
- **GNS: OpenSSL on macOS.** BUILDING.md says: "Do **not** try to use the OpenSSL that ships with macOS. `/usr/lib/libcrypto.dylib` is LibreSSL, Apple ships no headers for it, and it is not a drop-in substitute." — [GNS BUILDING.md](https://github.com/ValveSoftware/GameNetworkingSockets/blob/master/BUILDING.md)

### Inferences
- The Network framework (`nw_connection`, QUIC support) and GameKit (Game Center matchmaking, real-time `GKMatch`, achievements, leaderboards) are Apple-native alternatives. GameKit gives matchmaking and friends for free on Apple, which GNS lacks outside Steam. A likely split: GNS or Network.framework for transport, GameKit for identity and matchmaking. **[Apple docs not fetched this session]**

### Gaps
- Network framework and GameKit documentation were not fetched.
- GNS's crypto options (OpenSSL vs libsodium) were not examined in detail.

## Input and windowing: GameController, SDL3, native AppKit/MTKView/CAMetalLayer

### Takeaway
SDL3 is current (3.4.16) and handles windowing and input across Apple platforms. For an Apple-only engine, the native route is simpler to optimize and exposes Apple-only features:
- AppKit/UIKit with `CAMetalLayer` or `MTKView`
- the GameController framework, which covers controllers, keyboard, mouse and haptics
- Game Mode and MetalFX

### Cited Findings
- **SDL3.** Latest tags are release-3.4.12, 3.4.14 and 3.4.16. — [SDL](https://github.com/libsdl-org/SDL). The license is zlib **[not re-verified]**.
- **Apple game-development resources.**
  - WWDC25 "Level up your games" covers Metal 4 and Game Porting Toolkit 3.
  - metal-cpp "allows you to integrate Metal into your existing C++ code base, and it comes with full Metal 4 support".
  - Tooling: Metal Performance HUD, the Xcode Metal debugger, and Instruments Metal System Trace.
  - Source: [WWDC25 session 209](https://developer.apple.com/videos/play/wwdc2025/209/), [AppleInsider](https://appleinsider.com/articles/25/06/09/metal-4-game-porting-toolkit-3-boost-frame-rate-ray-tracing-performance), [apple/game-porting-toolkit](https://github.com/apple/game-porting-toolkit/tree/main)

### Inferences
- In a C++ engine, metal-cpp plus a thin Objective-C++ layer (NSWindow/UIWindow, CAMetalLayer, GCController) avoids SDL's abstraction cost and gives direct access to `CAMetalDisplayLink`, EDR/HDR and Game Mode.
- SDL3 is still useful for rapid prototyping, and its SDL_GPU backend is also targeted by RmlUi (see below).

### Gaps
- Pages for the GameController framework (GCKeyboard, GCMouse, the virtual controller on iOS) and for MTKView vs CAMetalLayer were not fetched this session.

## UI and debug tools: Dear ImGui, RmlUi, Tracy

### Takeaway
- **Dear ImGui** (v1.92.9) ships an official Metal renderer backend and an OSX platform backend.
- **RmlUi** (MIT, HTML/CSS-like game UI) has no native Metal renderer, but it does have an SDL_GPU renderer that would run on Metal.
- **Tracy** (BSD-3, v0.14.1) supports macOS, including a `TracyMetal.hmm` GPU-zone header, Apple Mach timing, and binary macOS releases.

### Cited Findings
- **Dear ImGui.** Latest tags are v1.92.9 and v1.92.9-docking. `backends/imgui_impl_metal.mm` is the "Renderer Backend for Metal… used along with a Platform Backend (e.g. OSX)". It supports `MTLTexture` as the texture identifier and large meshes. — [imgui_impl_metal.mm](https://github.com/ocornut/imgui/blob/master/backends/imgui_impl_metal.mm)
- **RmlUi.** MIT; latest tag release-1.3.0.0, with HEAD dated 2026-09-13. Its backends are GL2/GL3, VK, DX11/12 and `SDL_GPU` (`RmlUi_Renderer_SDL_GPU.cpp`); there is no dedicated Metal renderer. — [RmlUi](https://github.com/mikke89/RmlUi)
- **Tracy: license and Apple support.**
  - Licensed "under the 3-clause BSD license"; latest tag v0.14.1, with HEAD dated 2026-09-27.
  - Apple-specific code: `public/tracy/TracyMetal.hmm` and `public/client/apple/TracyMach.cpp`, plus a `profiler/macos` folder.
  - The NEWS file notes "Binary releases are now also provided for macOS", "Tracing on Arm macOS will now have more precise timer readings", and "Prototype implementation of system tracing on Apple devices".
  - Source: [Tracy](https://github.com/wolfpld/tracy)

### Inferences
- Stack: Dear ImGui (Metal) for in-engine debug UI, Tracy for CPU and Metal-GPU profiling, and Xcode Instruments / the Metal debugger for deeper GPU analysis.
- For in-game UI, either write a small Metal renderer for RmlUi (its render interface is small) or use SDL_GPU.

### Gaps
- The completeness of TracyMetal (timestamp sampling on Apple GPUs) was not inspected.

## Scripting: Lua/Luau, Wren, C#/.NET, others

### Takeaway
- **Luau** (MIT, Roblox, very active: tag 0.740) is the strongest embeddable choice. It is sandboxed, gradually typed and fast.
- **Wren** (MIT) appears stagnant: its latest tag is 0.4.0.
- **C#:** .NET NativeAOT is supported on iOS and Mac Catalyst since .NET 9. Engines that use it must root reflection-reached code against trimming. Hosting CoreCLR JIT is possible on macOS but not on iOS.

### Cited Findings
- **Luau.** License: MIT ("Copyright (c) 2019-2025 Roblox Corporation; Copyright (c) 1994–2019 Lua.org, PUC-Rio"). The latest numeric tags are 0.739 and 0.740; a stray tag named "696" is a sort artifact. — [luau LICENSE](https://github.com/luau-lang/luau/blob/master/LICENSE.txt), [luau](https://github.com/luau-lang/luau)
- **Wren.** MIT ("Copyright (c) 2013-2021 Robert Nystrom and Wren Contributors"); latest tags 0.3.0 and 0.4.0. — [wren LICENSE](https://github.com/wren-lang/wren/blob/main/LICENSE)
- **.NET NativeAOT.** "iOS/Mac Catalyst now supports NativeAOT since .NET 9, whereas experimental support was added in .NET 8." The publish command is `dotnet publish -f net10.0-ios -r ios-arm64 …`. — [Microsoft Learn: Native AOT deployment on iOS and Mac Catalyst](https://learn.microsoft.com/en-us/dotnet/maui/deployment/nativeaot?view=net-maui-9.0)
- **Godot C# on Apple.** Godot notes that "NativeAOT uses trimming and game engines may use reflection… bindings and game project assemblies need to be rooted", and that some reflection may break at runtime. — [Godot: platform state in C#](https://godotengine.org/article/platform-state-in-csharp-for-godot-4-2/)

### Inferences
- iOS forbids JIT (except for debuggers), so iOS can run only interpreters (Lua, Luau, Wren) or AOT-compiled C#. Luau's interpreter is fast without JIT; its x64/arm64 native codegen would be usable on macOS only.
- An Apple-only engine could also use **Swift** as the gameplay language via C++ interop. This is not researched here.

### Gaps
- The latest PUC Lua release (5.4.x vs 5.5) was not checked.
- Did not verify whether Luau's native codegen supports arm64 macOS in production.
- The .NET 10 runtime-hosting API (hostfxr) on macOS ARM64 was not fetched.

## Editor tooling: SwiftUI vs Dear ImGui

### Takeaway
I found no direct, citable comparative source in this session. The analysis below is inference.

### Cited Findings
- Dear ImGui has official Metal and OSX backends, so it can host an editor inside the engine's Metal view. — [imgui_impl_metal.mm](https://github.com/ocornut/imgui/blob/master/backends/imgui_impl_metal.mm)
- flecs ships a web-based entity explorer that can serve as an editor inspector. — [flecs](https://github.com/SanderMertens/flecs)

### Inferences
- **Dear ImGui (docking branch):**
  - Fastest to iterate on, and the same code path serves the in-game debug UI.
  - Needs custom work for native menus, accessibility, drag-and-drop from Finder, and text input with IME.
- **SwiftUI/AppKit shell hosting the engine's CAMetalLayer view:**
  - A native Mac look and good document or inspector patterns.
  - Needs a C++/Swift interop boundary (Swift 5.9+ C++ interop).
  - Weaker for dense, high-frequency tool widgets such as curve editors and node graphs.
- **Hybrid (common in practice):** an AppKit or SwiftUI shell with menus, file browser and inspectors, with the viewport and specialist tools drawn in ImGui inside the Metal view.

### Gaps
- No verified production example of a SwiftUI-based game editor was found.
