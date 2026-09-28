# State-of-the-art real-time rendering techniques (2024-2026) and how they map onto Apple Silicon (Metal 4, M3/M4/M5)

Research date: 2026-09-28. Primary Apple reference used throughout: Apple "Metal Feature Set Tables" PDF dated May 21, 2026 (https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), text extracted and grepped locally. Family mapping from that PDF: M1 = Apple7, M2 = Apple8, M3 and M4 = Apple9, M5 and A19 = Apple10; A17 Pro and A18 = Apple9.

Legend used in Inferences: "Feasible" = can be built today with public Metal API at reasonable cost; "Feasible with work" = needs non-trivial re-engineering/emulation; "Blocked/weak" = missing API/HW feature or large perf gap.

---

## 1. Virtualized geometry (Nanite cluster-LOD DAG, micro-triangle software raster, clusterlod, RTX Mega Geometry / cluster AS, open-source implementations)

### Takeaway
Nanite-style virtualized geometry is feasible on Apple Silicon from M2 onward. 64-bit ulong atomic min/max on buffers AND textures (the exact primitive a visbuffer software rasterizer needs) exists on M2 (macOS only), with the full 64-bit atomic set on M3/M4/M5. M1 lacks it, so it needs a 32-bit workaround. Epic shipped Nanite on Mac as beta for M2+ via an SM6 Metal path starting in UE 5.3. The offline side (cluster DAG building) is now commodity through meshoptimizer's clusterlod.h. The ray-tracing side of cluster geometry is where Apple has no answer: no cluster acceleration structures or partitioned TLAS. NVIDIA/Vulkan have them, and DXR 2.0 standardized them at GDC 2026.

### Cited Findings
**Apple hardware and API support for 64-bit atomics**
- Apple's feature table lists "64-bit atomics" starting at Apple9. Footnote 7: "GPU devices in the Apple8 family support 64-bit atomic minimum and maximum using ulong, on both buffers and textures, only on macOS. The full set of 64-bit atomic operations is supported on all platforms starting with Apple9." In practice that means M2 has only min/max on macOS, while M3/M4/M5 and A17 Pro+ have the full set. — [Apple Metal Feature Set Tables (May 2026)](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- The same tables list "Texture atomics" from Apple6, "Floating-point atomics" from Apple7, and "Atomics on cube map and cube map array textures" from Apple10 (M5). — [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

**How Nanite and similar software rasterizers use 64-bit atomics**
- UE5's software rasterizer does 64-bit atomic compare/max, with depth in the upper 32 bits and payload in the lower 32. Philip Turner (independent researcher) suggests Apple may have added UInt64 min/max "precisely to get Nanite running on M2". He also showed that Nanite can run "entirely through 32-bit atomics, without creating data races", at an estimated 2.5x (bandwidth) to 5x (latency) cost. His repo was archived on Aug 16, 2024. — [philipturner/ue5-nanite-macos](https://github.com/philipturner/ue5-nanite-macos), [AtomicsWorkaround README](https://github.com/philipturner/ue5-nanite-macos/blob/main/AtomicsWorkaround/README.md)
- The same repo points to a Nanite persistent-thread culling requirement: the assertion "GRHIPersistentThreadGroupCount must be configured correctly in the RHI". This relates to forward-progress guarantees. — [philipturner/ue5-nanite-macos](https://github.com/philipturner/ue5-nanite-macos)

**Nanite on Mac in Unreal Engine**
- UE 5.2 blog, as seen in the search snippet: "Nanite relies on image atomics and forward-progress guarantees that Apple M1 devices may not support". Experimental support for M2 Macs existed but was disabled by default, and hardware ray tracing was "not currently supported on macOS". — [Epic: UE 5.2 native Apple Silicon support](https://www.unrealengine.com/en-US/tech-blog/unreal-engine-5-2-brings-native-support-for-apple-silicon-and-other-developments-for-macos)
- UE 5.3 release notes: "Unreal Engine 5.3 brings beta support for Nanite rendering technology on Apple Silicon M2 devices running macOS", enabled by default with SM6 in the launcher binaries. — [UE 5.3 Release Notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.3-release-notes?application_version=5.3)
- Epic published a Feb 2025 progress report, "Bringing Unreal Engine on macOS up to feature parity with Windows". Only the search snippet was readable (the page returned 403 to the fetcher). The snippet says the refined Metal RHI supports SM6, bringing Nanite to M2 and later without recompiling the editor, and that SM6 requires macOS 15+. — [Epic progress report](https://www.unrealengine.com/tech-blog/bringing-unreal-engine-on-macos-up-to-feature-parity-with-windowsprogress-report)
- Conflicting signal: Epic's current (5.8) Nanite page says only "DirectX 12 with Shader Model 6 (SM6)" and does not mention Mac. The "Supported features by rendering path" page shows macOS/Metal SM5 with Nanite "N", Virtual Shadow Maps "N", Lumen HWRT "N" and Path Tracer "N". That page appears to describe only the SM5 Mac path. — [Nanite docs 5.8](https://dev.epicgames.com/documentation/unreal-engine/nanite-virtualized-geometry-in-unreal-engine); [Supported features by rendering path](https://dev.epicgames.com/documentation/en-us/unreal-engine/supported-features-by-rendering-path-for-desktop-with-unreal-engine)
- A secondary source (StraySpark blog, Apr 2026, not Epic) claims "Nanite on Metal works across modern M-series GPUs and is in the same performance neighborhood as D3D12 on comparable workloads, though gaps remain on heavy geometry". Treat this as unverified. — [StraySpark: Apple Silicon M5 for UE development 2026](https://www.strayspark.studio/blog/apple-silicon-m5-unreal-engine-development-2026)

**Nanite features in UE 5.x (context)**
- Epic's 5.8 docs describe: hierarchical cluster decomposition at import; static displacement via offline adaptive tessellation; dynamic (material-driven) tessellation; and "Nanite Foliage" (instancing, skinned meshes, voxelization). — [Nanite docs 5.8](https://dev.epicgames.com/documentation/unreal-engine/nanite-virtualized-geometry-in-unreal-engine)

**Open-source cluster-LOD tooling**
- meshoptimizer v1.0 ships `clusterlod.h`, a single-header continuous-LOD builder (cluster, group, simplify with locked boundaries, recurse; "similarly to Nanite"). — [meshoptimizer clusterlod.h](https://github.com/zeux/meshoptimizer/blob/master/demo/clusterlod.h); [meshoptimizer releases](https://github.com/zeux/meshoptimizer/releases)
- zeux (Sep 30, 2025) processed NVIDIA's Zorah scene (1.64B triangles) into a cluster DAG, going from about 7-9 min down to about 2m35s on 16 threads, with 54-60 GB RAM. clusterlod includes a "raytracing-optimized clusterizer" and is used together with NVIDIA's `vk_lod_clusters`. — [zeux.io: Billions of triangles in minutes](https://zeux.io/2025/09/30/billions-of-triangles-in-minutes/)
- NVIDIA's `vk_lod_clusters` is a sample for "cluster-based continuous level of detail rasterization or ray tracing". — [nvpro-samples/vk_lod_clusters](https://github.com/nvpro-samples/vk_lod_clusters)

**Bevy's virtual geometry ("meshlets")**
- Bevy (Rust/wgpu) implemented Nanite-like virtual geometry starting in 0.14, with improvements through 0.15-0.17. In 0.16 it rasterizes into an R64Uint/R32Uint storage texture holding "packed depth + cluster ID + triangle ID", does two-pass occlusion culling, and builds the DAG with METIS. — [JMS55: Virtual Geometry in Bevy 0.16](https://jms55.github.io/posts/2025-03-27-virtual-geometry-bevy-0-16/); [Bevy 0.14 post](https://jms55.github.io/posts/2024-06-09-virtual-geometry-bevy-0-14/)
- Bevy PR #17765 moved to texture atomics, roughly 40% faster on an RTX 4070/Vulkan. On Metal it requires an M2 Mac or newer, or A17+ on mobile. M1 is unsupported because neither `TEXTURE_INT64_ATOMIC` nor `SHADER_INT64_ATOMIC_MIN_MAX` is available there. A reviewer confirmed testing on M2 macOS. — [bevy PR #17765](https://github.com/bevyengine/bevy/pull/17765)

**RTX Mega Geometry and cluster acceleration structures**
- RTX Mega Geometry adds Cluster Acceleration Structures (CLAS, batches of up to 256 triangles, GPU-driven) plus partitioned TLAS. Alan Wake 2 (title update 1.2.8, early 2025) applied it to existing assets for about 5-20% FPS and about 300 MB less VRAM. — [Tom's Hardware test](https://www.tomshardware.com/pc-components/gpus/testing-nvidias-rtx-mega-geometry-tech-vram-reducing-tech-a-leap-forward-for-path-traced-rendering); [NVIDIA-RTX/RTXMG](https://github.com/NVIDIA-RTX/RTXMG); [NVIDIA blog: Mega Geometry Vulkan samples](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples)
- RTX Mega Geometry 2.0 (RTX Kit 2026.3) adds Cluster LOD. It streams pre-baked clusters from a continuous-LOD hierarchy, shares/caches/merges BLAS, and uses `VK_NV_cluster_acceleration_structure` and `VK_NV_partitioned_acceleration_structure`. — [Hardware Busters](https://hwbusters.com/news/rtx-mega-geometry-2-0-streams-ray-traced-geometry-into-vram-and-sheds-detail-before-it-runs-out-of-memory/); [WindowsForum summary](https://windowsforum.com/news/nvidia-rtx-mega-geometry-2-0-adds-cluster-lod-for-ray-tracing.445726/)
- GDC 2026: Microsoft announced "DXR 2.0", which includes CLAS (up to 256 vertices/triangles per cluster), cluster templates for animated objects, compressed position encoding, partitioned TLAS, and indirect AS builds. — [asawicki.info: DirectX 12 news from GDC 2026](https://asawicki.info/news_1801_directx_12_news_from_gdc_2026_-_my_comments)
- Apple's closest features: Metal 4 "Address-driven acceleration structure builds" (Apple9+), "Acceleration structure build options" (Apple9+), plus the build flags `preferFastIntersection` and `minimizeMemoryUsage`. The May 2026 feature table has no cluster-AS, partitioned-TLAS or opacity-micromap entry. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [WWDC25 "Go further with Metal 4 games"](https://developer.apple.com/videos/play/wwdc2025/211/)
- On M5, acceleration-structure memory alignment dropped from 16 KB to 1 KB, which "eliminates hundreds of megabytes of padding overhead in scenes with many small objects". This helps when there are many small per-cluster BLASes. — [Apple Tech Talk: Boost graphics performance with M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111431/)

### Inferences
- **Rasterized virtual geometry: Feasible on M2+, best on M3+.** A native engine should require Apple8/macOS as a minimum for the visbuffer 64-bit `atomic_max` path. For M1 (and older iOS), use the 32-bit split-atomic workaround, or hardware-raster everything through mesh shaders. Mesh shaders are available from Apple7, with hardware acceleration and ICB mesh draws from Apple9.
- **Streaming benefits from unified memory.** There is no PCIe upload, so page-in can be a CPU memcpy or DirectStorage-like `MTLIOCommandQueue` load straight into GPU-visible memory. Large unified pools also relax the resident-set budget. This is inferred from the architecture, not from a benchmark.
- **Ray tracing against Nanite-density geometry is the real blocker.** Without CLAS/PTLAS, an Apple engine must ray trace a simplified proxy/fallback mesh (UE's approach on non-Mega-Geometry hardware), or build many small BLASes per cluster group. M5's 1 KB alignment and faster instance transforms make the latter less painful. RT against full-detail cluster LOD, as in Mega Geometry 2.0, has no Metal equivalent as of the May 2026 tables.
- **Persistent-thread culling and forward progress.** Nanite's persistent-threads hierarchy traversal assumes a forward-progress guarantee that Metal does not document. A native design should prefer multi-pass, level-by-level BVH/DAG traversal with indirect dispatches, which is what Bevy and many hobby engines do.

### Gaps
- No primary Epic statement found for Nanite/VSM/Lumen-HWRT status on Mac in UE 5.6-5.8 (the Epic progress report was 403 to the fetcher). No first-party performance numbers for Nanite on M3/M4/M5.
- No Apple statement on forward-progress guarantees between threadgroups.
- Whether Apple plans cluster acceleration structures or opacity micromaps is unknown; nothing appeared at WWDC25/WWDC26 per the pages reviewed.

---

## 2. GPU-driven rendering, two-phase occlusion culling, visibility buffers, and work graphs (and Metal's equivalent)

### Takeaway
Classic GPU-driven rendering maps well onto Metal: indirect command buffers encoded on the GPU, mesh shaders, argument buffers and tables, and residency sets. The visibility-buffer approach is compatible with TBDR, but a compute-based software raster bypasses the tile hardware. Metal has no equivalent of D3D12 Work Graphs. That matters little in practice so far, because Work Graphs had no shipping games as of early 2026.

### Cited Findings
**Metal's GPU-driven toolkit**
- Metal's Indirect Command Buffers (ICBs) can be encoded on the CPU or GPU. Tellusim notes Metal ICBs are similar in expressivity to `VK_NV_device_generated_commands`, supporting full pipeline changes and the equivalent of binding descriptor sets. — [Apple: Encoding ICBs on the GPU](https://developer.apple.com/documentation/Metal/encoding-indirect-command-buffers-on-the-gpu); [Tellusim: MultiDrawIndirect and Metal](https://tellusim.com/metal-mdi/)
- Feature-table entries:
  - Mesh shading: Apple7+.
  - ICBs containing mesh draws: Apple9+.
  - ICB support for raster and depth/stencil state: Apple10 (M5) only.
  - Sampler min/max reduction (useful for single-tap Hi-Z pyramid builds): Apple10 only.
  - Depth bounds testing: Apple10 only.
  - Max threadgroups per mesh grid: 1024 (Apple7/8), 1,048,575 (Apple9), 4,194,303 (Apple10).
  - Footnote 6: "Support for function pointers and ray tracing in render pipelines isn't compatible with mesh shading."

  — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- M3 tech talk: hardware-accelerated mesh shading keeps intermediate meshlet data on-chip, mesh draws can go into ICBs, and max mesh-grid threadgroups rose from 1,024 to over 1 million. — [Apple Tech Talk: Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)
- M5 tech talk: geometry throughput doubled, FP16 and complex ALU rate doubled, up to 30% more memory bandwidth, and second-generation Dynamic Caching. Universal texture compression now works on shader-write textures. — [Apple Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431/)
- Metal 4 core-API features (Apple7+): argument tables, command allocators, decoupled command queues/buffers, command barriers, placement sparse buffers/textures (Apple8+), flexible render pipeline state, pipeline dataset serialization. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)

**TBDR constraints**
- Apple GPUs are TBDR. Tile size is 32x32 without MSAA, and on-chip tile memory reduces main-memory traffic. — [WWDC20 Harness Apple GPUs with Metal](https://developer.apple.com/videos/play/wwdc2020/10602/); [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Footnote 3: Apple3 through Apple10 "don't support memory barriers that include the MTLRenderStages.fragment or .tile stages in the after argument". — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

**Work Graphs**
- D3D12 Work Graphs went 1.0 in March 2024 (compute-only). Mesh nodes (draws inside a graph) went to public preview, and AMD's `VK_AMDX_shader_enqueue` added mesh nodes on Vulkan. — [D3D12 Work Graphs blog](https://devblogs.microsoft.com/directx/d3d12-work-graphs/); [D3D12 Mesh Nodes preview](https://devblogs.microsoft.com/directx/d3d12-mesh-nodes-in-work-graphs/); [GPUOpen: Work Graphs mesh nodes in Vulkan](https://gpuopen.com/learn/gpu-workgraphs-mesh-nodes-vulkan/)
- A search-result summary says no games shipped with Work Graphs as of early 2026. A rumour report says next-gen consoles won't use them in their first years. VKD3D-Proton's emulation effort "disappoints". — [GamingBolt (rumour)](https://gamingbolt.com/ps6-next-xbox-wont-use-d3d12-work-graphs-feature-in-the-first-few-years-rumour); [Phoronix: VKD3D-Proton work graphs](https://www.phoronix.com/news/VKD3D-Proton-Work-Graphs)
- The Apple Metal feature table (May 2026) and the WWDC26 Metal guide contain no work-graph or GPU-enqueue feature. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)

### Inferences
- **Two-phase occlusion culling: Feasible.** The pattern is: previous-frame HZB, cull, draw, rebuild HZB, re-test the rejects. It is implementable with compute plus ICBs or mesh/object shaders on all M-series. M5 adds sampler min/max reduction and depth-bounds testing, which cheapen Hi-Z generation and tests.
- **Visibility buffer with compute-based shading: Feasible.**
  - Compute-shaded material passes work normally.
  - Tile-memory advantages (programmable blending, imageblocks, memoryless attachments) favour hybrid designs: hardware-raster primary visibility into tile memory, then resolve.
  - A pure compute software rasterizer writing to a 64-bit storage texture bypasses TBDR hidden-surface removal and tile memory entirely.
  - Expect micro-triangle software raster to be ALU/atomic bound, similar to desktop GPUs. M5's doubled geometry throughput and full 64-bit atomics on Apple9+ help.
- **Work graphs: no Metal equivalent.** Emulate with multi-pass "append buffer + indirect dispatch" chains, or persistent-thread queues (with the forward-progress caveat). Since no shipping titles depend on work graphs, this is a low-impact gap for a 2026-2028 engine.

### Gaps
- No published benchmark comparing visibility-buffer vs deferred G-buffer on Apple TBDR at AAA scale.
- No Apple statement on plans for GPU work graphs.

---

## 3. Real-time path tracing, many lights and radiance caching (ReSTIR DI/GI/PT, SHaRC, NRC, Lumen, GI-1.0/Brixelizer, MegaLights, surfel GI)

### Takeaway
The algorithms are API-agnostic compute plus ray queries, so all of them can be ported to Metal. The limit is ray-tracing throughput and the lack of NVIDIA-specific accelerators (SER, OMM, cluster AS). Hardware RT arrived with M3 and improved with M5's third-generation engine; M5 Pro/Max report up to 30-35% RT uplift over the previous generation. Shipping evidence (Cyberpunk 2077 on Mac) shows RT Medium at 1080p-class resolutions and 30-60 fps on M3 Pro/Max, with path tracing possible but not yet a comfortable default. Radiance-cache-heavy approaches are the practical sweet spot on Apple GPUs today: Lumen-style caches, SHaRC-style hash caches, and GI-1.0/Brixelizer.

### Cited Findings
**Apple ray-tracing hardware**
- M3 introduced hardware ray tracing, mesh shading and Dynamic Caching on Mac. — [Apple Newsroom, M3 (Oct 2023)](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/)
- M3 tech talk details:
  - Fixed-function traversal plus box/triangle intersection.
  - Custom intersection functions go through a "reorder stage" that groups calls from different SIMD-groups. This is a limited analogue of SER.
  - Apple advises the `intersector` object API over `intersection_query`, because the query API "increases scratch memory I/O and disables the reorder stage".

  — [Apple Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)
- M5 third-generation RT: hardware-accelerated instance transforms, fully hardware intersection-function-buffer indexing ("GPU time spent in IFB indexing dropped up to 70% compared to emulation"), and AS alignment reduced from 16 KB to 1 KB. — [Apple Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431/)
- Apple claims M5 gives "up to 45 percent graphics uplift in apps using ray tracing" (Oct 2025). For M5 Pro/Max (Mar 2026) the claim is up to 35% generational RT uplift, about 30% for M5 Max vs M4 Max (per search summary). — [Apple Newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/); [Apple Newsroom M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- Metal 4 intersection function buffers (Apple9+) map directly from DXR shader binding tables (same ray-type index / geometry multiplier concept). — [WWDC25 session 211](https://developer.apple.com/videos/play/wwdc2025/211/); [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Offline reference point: M4 Max averaged 5,205 on Blender Open Data, just below the RTX 4080 Laptop at 5,326. — [TechSpot](https://www.techspot.com/news/105649-apple-m4-max-beats-rtx-4070-blender-but.html)

**Cyberpunk 2077 on Mac (shipping reference)**
- Released July 17, 2025, requiring M1+ with 16 GB. Ray tracing needs M3+ and 16 GB. RT Medium runs at about 30 fps on M3 Pro and about 60 fps on M3 Max, at 1800x1125 or 1080p with MetalFX DRS. RT is off by default. — [CDPR support: Ray tracing on Mac](https://support.cdprojektred.com/en/cyberpunk/mac/sp-technical/issue/2897/ray-tracing-on-mac); [AppleInsider](https://appleinsider.com/articles/25/07/15/cyberpunk-2077-ultimate-edition-coming-to-apple-silicon-macs-on-july-17)
- The launch coverage headline mentions path tracing and FSR frame generation on macOS. MetalFX Denoising (for real-time path tracing) and MetalFX Frame Interpolation were promised as a post-launch update. — [Notebookcheck](https://www.notebookcheck.net/Cyberpunk-2077-to-be-released-on-Thursday-for-macOS-with-FSR-frame-generation-and-path-tracing.1058826.0.html); [AppleInsider](https://appleinsider.com/articles/25/07/15/cyberpunk-2077-ultimate-edition-coming-to-apple-silicon-macs-on-july-17)
- M4 Max reportedly reaches about 61 fps at 1600p with RT and MetalFX upscaling. — [TweakTown](https://www.tweaktown.com/news/106804/cyberpunk-2077-runs-at-1600p-61fps-on-macbook-pro-16-with-apple-m4-max-chip-and-ray-tracing/index.html)
- WWDC25 demoed real-time path tracing with the MetalFX denoised upscaler, and WWDC26 coverage says Apple showed Cyberpunk 2077 path tracing. — [WWDC25 session 211](https://developer.apple.com/videos/play/wwdc2025/211/); [search summary of WWDC26 coverage](https://developer.apple.com/wwdc26/guides/games/)

**ReSTIR family (research and production)**
- ReSTIR DI (Bitterli et al. 2020), ReSTIR GI (Ouyang et al. 2021), GRIS/ReSTIR PT (Lin et al. 2022). Cyberpunk 2077 used ReSTIR DI for many-light night scenes plus ReSTIR GI, and NVIDIA's UE branch uses ReSTIR GI and PT. — [Wikipedia: Spatiotemporal reservoir resampling](https://en.wikipedia.org/wiki/Spatiotemporal_reservoir_resampling)
- 2025-2026 papers:
  - "ReSTIR PT Enhanced" (Lin, Kettunen, Wyman, 2026).
  - ReSTIR PG (path guiding, SIGGRAPH Asia 2025).
  - ReSTIR BDPT with caustics (2025).
  - ReSTIR SSS (2024).
  - SIGGRAPH 2025 Advances talk "Real-Time Subsurface Scattering via Hybrid ReSTIR-Path-Tracing and Diffusion" (NVIDIA).

  — [NVIDIA: ReSTIR PT Enhanced](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/); [ReSTIR PG paper](https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/zeng2025restirpg_paper.pdf); [ReSTIR BDPT](https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir/hedstrom2025restir.pdf); [ReSTIR SSS](https://dl.acm.org/doi/10.1145/3675372); [Advances 2025](https://advances.realtimerendering.com/s2025/index.html)
- RTX Kit put ReSTIR PT into the NvRTX 5.6 UE branch. — [NVIDIA blog (GDC 2025 RTX Kit updates)](https://developer.nvidia.com/blog/announcing-the-latest-nvidia-gaming-ai-and-neural-rendering-technologies/)
- Open-source reference, Bevy "Solari" (0.17, Sep 2025):
  - ReSTIR DI (2 rays/px) plus ReSTIR GI (2 rays/px), a spatially hashed "world cache", and DLSS-RR.
  - Total 8.2-14.6 ms on an RTX 3080 at 1600x900, upscaled to 3200x1800.
  - Currently NVIDIA-only; no Metal mention.

  — [JMS55: Realtime Raytracing in Bevy 0.17 (Solari)](https://jms55.github.io/posts/2025-09-20-solari-bevy-0-17/)

**Radiance caches**
- SHaRC (Spatially Hashed Radiance Cache) and NRC (Neural Radiance Cache) are both in the RTXGI v2 SDK for D3D12/Vulkan; NRC is "currently experimental". Doom: The Dark Ages' path-tracing update uses SHaRC plus SER. — [NVIDIA-RTX/RTXGI](https://github.com/NVIDIA-RTX/RTXGI); [NVIDIA-RTX/SHARC](https://github.com/NVIDIA-RTX/SHARC); [NVIDIA GeForce news: Doom TDA PT update](https://www.nvidia.com/en-us/geforce/news/doom-the-dark-ages-path-tracing-dlss-ray-reconstruction-update/)
- AMD FSR "Redstone" includes Neural Radiance Caching, ML Ray Regeneration and ML Frame Generation. The ML parts require RDNA4, and Radiance Caching was due in games in 2026. — [GPUOpen: FSR Redstone for developers](https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/); [GPUOpen: FSR Radiance Caching](https://gpuopen.com/amd-fsr-radiancecaching/)
- SIGGRAPH 2026 Advances talk: "Speeding up Path Tracing via ORCA" (EA SEED), a "custom radiance cache designed specifically for real-time rendering" without temporal dependencies. — [Advances 2026](https://advances.realtimerendering.com/s2026/index.html)

**Non-hardware-RT GI and production GI talks**
- AMD Brixelizer GI is a simplified GI-1.0: screen probes plus a world-space radiance/irradiance cache over sparse distance-field "bricks" (64^3 voxel cascades). It is compute-only and "does not require hardware-accelerated ray-tracing". — [GPUOpen Brixelizer GI manual](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/brixelizer-gi/); [GPUOpen Brixelizer](https://gpuopen.com/fidelityfx-brixelizer/)
- Lumen:
  - Surface cache built from mesh "cards" (default 12 per mesh).
  - Screen tracing first, then software (distance field) or hardware RT.
  - Targets 60 fps on consoles, about 8 ms at 1080p for Epic scalability.
  - Epic's Lumen doc lists HWRT platforms as Windows DX12, PS5, Xbox Series and Switch 2, and does not mention Mac.

  — [Lumen Technical Details](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine)
- The secondary StraySpark source claims "Lumen on Metal supports both software and hardware ray tracing" in UE 5.7-era builds. This contradicts Epic's docs; treat it as unverified. — [StraySpark](https://www.strayspark.studio/blog/apple-silicon-m5-unreal-engine-development-2026)
- MegaLights (UE 5.5+, SIGGRAPH 2025 Advances) is stochastic direct lighting that traces a fixed number of rays per pixel toward importance-sampled lights, to support many shadowed area lights on current consoles. — [MegaLights slides](https://advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf); [UE docs MegaLights](https://dev.epicgames.com/documentation/unreal-engine/megalights-in-unreal-engine?lang=en-US)
- Arm published a blog on bringing MegaLights to mobile. — [Arm community blog](https://developer.arm.com/community/arm-community-blogs/b/mobile-graphics-and-gaming-blog/posts/lighting-at-scale-bringing-hundreds-of-dynamic-lights-to-mobile-with-unreal-megalights)
- Other 2025 production GI talks: "Ray Tracing the World of Assassin's Creed Shadows" (Ubisoft), "Fast as Hell: idTech8 Global Illumination" (id), "Stochastic Tile-Based Lighting in HypeHype" (mobile-to-PC). — [Advances 2025](https://advances.realtimerendering.com/s2025/index.html)
- Assassin's Creed Shadows ships natively on macOS. On PC it has hardware RTGI plus a proprietary software-RT fallback for non-RT GPUs; consoles use baked GI in 60 fps modes. — [Ubisoft AC Shadows Tech Q&A](https://www.ubisoft.com/en-us/game/assassins-creed/news/4XbPPtFyQEtIMWrA9xVDmZ/assassins-creed-shadows-tech-qa)

**Features NVIDIA/DXR have that Metal lacks**
- DXR 1.2 (GDC 2025) added Opacity Micromaps (up to 2.3x in path-traced games) and Shader Execution Reordering (up to 2x). — [DirectX blog: DXR 1.2](https://devblogs.microsoft.com/directx/announcing-directx-raytracing-1-2-pix-neural-rendering-and-more-at-gdc-2025/)
- Metal's feature table has no OMM or general SER; its only reordering is the intersection-function reorder stage. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [Apple Tech Talk 111375](https://developer.apple.com/videos/play/tech-talks/111375/)

### Inferences
- **ReSTIR DI (many lights) and MegaLights-style stochastic direct lighting: Feasible on M3+.** The cost is 1-2 shadow rays per pixel plus reservoir bandwidth. Unified memory bandwidth (M-Max class) is adequate. This is likely the highest-value RT feature per millisecond on Apple.
- **ReSTIR GI plus a hash-grid world cache (SHaRC/Solari-style): Feasible with work on M3 Max, M4 Max and M5 Pro/Max** at reduced internal resolution with MetalFX denoised upscaling. On base M3/M4/M5 it is marginal. Alpha-tested foliage will hurt: there are no OMMs, so any-hit via intersection functions is needed. Use the `intersector` API to keep the reorder stage active.
- **Full ReSTIR PT / Cyberpunk-Overdrive-class path tracing: Weak today.** It becomes demo/"ultra" material on M5 Max-class hardware only with the MetalFX denoised upscaler plus frame interpolation. The missing SER/OMM and lower raw RT throughput than desktop RTX are the blockers. Apple's own messaging frames PT as "fewer rays + denoise".
- **Lumen-like hybrid (SDF/surface-cache software tracing plus optional HWRT) and GI-1.0/Brixelizer: Feasible on M1+.** These are compute-only and scale across the whole Apple line. They are the safest baseline GI for a native Apple engine, with HWRT as an M3+ quality tier.
- **NRC (online-trained MLP): Feasible in principle on M5.** WWDC26 shows in-shader inference and backprop with TensorOps cooperative tensors (see section 6). No Metal NRC implementation was found.

### Gaps
- No published fps for Cyberpunk path tracing on M3/M4/M5, and no confirmation that the MetalFX-denoiser update shipped.
- No independent RT micro-benchmarks (rays/s) for M3/M4/M5 against RTX/RDNA.
- No primary Epic confirmation of Lumen HWRT or MegaLights on Mac.
- EA SEED GIBS/surfel GI: not re-researched here (it is pre-2024 foundational work). SEED's newer ORCA (2026) has few details beyond the talk abstract.

---

## 4. Denoising (NRD ReBLUR/ReLAX, DLSS Ray Reconstruction, MetalFX denoised upscaler, ML denoisers)

### Takeaway
The state of the art has moved from hand-tuned spatiotemporal denoisers (NRD) to joint ML denoise+upscale: DLSS RR 4/4.5, FSR Ray Regeneration, and MetalFX's denoised upscaler. Apple provides a first-party joint denoiser/upscaler on M3+ (Apple9+) that needs no per-scene tuning. NRD has no Metal backend but is portable HLSL/compute.

### Cited Findings
**NRD**
- NRD is an "API agnostic" spatio-temporal denoising library: ReBLUR for low spp, ReLAX for preserving lighting detail. It integrates into D3D12, Vulkan or D3D11 engines; no Metal backend is documented. — [NVIDIA-RTX/NRD](https://github.com/NVIDIA-RTX/NRD)

**DLSS Ray Reconstruction**
- DLSS 4 (Jan 2025) moved Ray Reconstruction, Super Resolution and DLAA to a vision-transformer model. DLSS 4.5 Ray Reconstruction (second-gen transformer, August 2026) has "35% more compute capability" and 20% more parameters at similar cost. — [NVIDIA DLSS 4](https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation-ai-innovations/); [NVIDIA DLSS 4.5 RR](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-ray-reconstruction-1000-rtx-games-apps-out-now/); [TechPowerUp review](https://www.techpowerup.com/review/nvidia-dlss-4-5-ray-reconstruction/)

**AMD FSR Ray Regeneration**
- ML denoising for RT, requiring RDNA4. — [GPUOpen FSR Redstone](https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/)

**MetalFX denoised upscaler**
- Requires Apple9+ (M3/M4/M5, A17 Pro+). — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- Inputs:
  - The standard upscaler inputs: color, motion, depth.
  - World-space normals (with sign bit), diffuse albedo, linear roughness, and specular albedo (noise-free, including Fresnel).
  - Optional: specular hit distance, denoiser-strength mask, transparency overlay.
- Apple claims "no per-scene parameter tuning". Pitfalls it flags: correlated random numbers, and missing NEE/importance sampling.

  — [WWDC25 "Go further with Metal 4 games"](https://developer.apple.com/videos/play/wwdc2025/211/)
- WWDC26 "Build real-time neural rendering pipelines with Metal":
  - The joint denoiser handles 1 spp path tracing.
  - Best practices: clean aux inputs ("diffuse albedo is the strongest denoising signal"), Fresnel-blended primary-surface replacement for mirrors/glass, dejittered motion vectors.
  - Maxon Redshift Live renders path-traced viewports at 1 spp plus MetalFX denoising.

  — [WWDC26 session 359](https://developer.apple.com/videos/play/wwdc2026/359/)
- Other studios' ML denoising talk: Sony's "Upgrading PSSR on PlayStation 5 Pro" (Advances 2026). — [Advances 2026](https://advances.realtimerendering.com/s2026/index.html)

### Inferences
- **Recommended path on M3+: MetalFX denoised upscaler.** Design the G-buffer to produce Apple's required aux channels (specular albedo including Fresnel, roughness, sign-carrying normals) from day one.
- **Keep an NRD-like or custom compute denoiser (e.g. ReLAX port) for:**
  - M1/M2, since the MetalFX denoiser needs Apple9+;
  - effects needing per-signal control, such as separate shadow and AO denoising;
  - cases where MetalFX quality is insufficient.

  Porting NRD's HLSL via Metal Shader Converter or a hand port is feasible, but no existing Metal port was found.
- Quality versus DLSS RR 4.5 is unknown (see Gaps). DLSS RR is transformer-based and heavily trained on game content, so parity should not be assumed.

### Gaps
- No independent image-quality comparison of the MetalFX denoised upscaler against DLSS RR or NRD was found.
- No published cost (ms) of the MetalFX denoiser at given resolutions and chips.

---

## 5. Shadows (virtual shadow maps, ray-traced shadows, cost)

### Takeaway
Virtual shadow maps (16k x 16k virtual, 128x128-texel pages, cached) pair naturally with a virtualized-geometry rasterizer. They need the same GPU-driven machinery (page tables, atomics, indirect draws) and run on Apple via UE's SM6 Metal path, per secondary sources. Ray-traced shadows are cheap per ray relative to GI and are the natural HWRT entry point on M3+. MegaLights-style stochastic shadows unify many-light shadowing.

### Cited Findings
**Virtual shadow maps**
- VSM uses a 16k x 16k virtual resolution split into 128x128 pages, allocated on demand and cached across frames unless invalidated. — [Epic: VSM in Fortnite Chapter 4](https://www.unrealengine.com/en-US/tech-blog/virtual-shadow-maps-in-fortnite-battle-royale-chapter-4); [UE 5.8 VSM docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)
- Fortnite's moving sun and deforming trees undercut caching, unlike the earlier "Lumen in the Land of Nanite" and "The Matrix Awakens" demos. — [Epic: VSM in Fortnite Chapter 4](https://www.unrealengine.com/en-US/tech-blog/virtual-shadow-maps-in-fortnite-battle-royale-chapter-4)
- Secondary sources report console shadow passes of around 9 ms in some projects, and one-pass projection light loops at 1.08 ms vs 1.56 ms. These are non-primary optimization blogs; treat the numbers as indicative only. — [StraySpark VSM 5.7](https://www.strayspark.studio/blog/virtual-shadow-map-optimization-open-worlds-ue5-7); [PerfGuard VSM tuning](https://getperfguard.com/tutorials/virtual-shadow-maps)
- Epic's 5.8 support table shows VSM "N" on the macOS SM5 path. StraySpark claims VSM is available on Mac (SM6). — [Supported features by rendering path](https://dev.epicgames.com/documentation/en-us/unreal-engine/supported-features-by-rendering-path-for-desktop-with-unreal-engine); [StraySpark](https://www.strayspark.studio/blog/apple-silicon-m5-unreal-engine-development-2026)

**Stochastic many-light shadows**
- MegaLights traces a fixed number of shadow rays per pixel toward stochastically selected lights, enabling "orders of magnitude more dynamic and shadowed area lights". — [MegaLights SIGGRAPH 2025](https://advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf)

**Relevant Metal features**
- Sparse textures are Apple6+, and Metal 4 placement sparse buffers/textures are Apple8+. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

### Inferences
- **VSM: Feasible on M2+** using the same 64-bit atomic/visbuffer-style page rasterization as Nanite, or hardware raster into sparse depth textures. Placement sparse (Apple8+) and ICBs support page-granular rendering.
- On TBDR, rendering many small shadow pages with hardware raster costs a render-pass/tile setup per page. Batching pages into atlas passes, or compute software raster for small clusters, is likely preferable. This is an inference; there is no Apple-specific VSM data.
- **RT shadows: Feasible on M3+** for sun, local and area lights. They are among the cheapest RT effects, and denoising can use the MetalFX denoiser or a dedicated shadow denoiser.

### Gaps
- No Apple-GPU VSM cost data from Epic.
- No primary measurements of RT-shadow cost on M3/M4/M5.

---

## 6. Neural rendering (neural materials, NTC, RTX Neural Shaders, cooperative vectors; Metal 4 tensors; Gaussian splatting)

### Takeaway
The cross-vendor primitive for neural shading is "matrix ops callable from any shader stage". On D3D12 this is cooperative vectors (SM 6.9), evolving into the LinAlg Matrix type (SM 6.10, GDC 2026). On Vulkan it is `VK_NV_cooperative_vector`. Apple's equivalent is Metal 4 tensors plus Metal Performance Primitives TensorOps with cooperative tensors, callable from any stage. It runs on Apple7+ and is hardware-accelerated by M5's per-core Neural Accelerators. Apple's public neural-rendering guidance (WWDC26) covers denoise/upscale, neural tone mapping and tiny online-trained MLPs. It does not yet cover neural materials or neural texture compression, so those would be engine-owned ports.

### Cited Findings
**NVIDIA and Microsoft**
- Real-Time Neural Appearance Models (Zeltner et al., ACM TOG / SIGGRAPH 2024): learned hierarchical latent textures decoded by small MLPs, with importance sampling and LOD. Deep layered material graphs are baked into a compact neural representation. — [arXiv 2305.02678](https://arxiv.org/abs/2305.02678); [NVIDIA Research](https://research.nvidia.com/publication/2023-05_real-time-neural-appearance-models)
- RTX Neural Texture Compression (NTC SDK v0.10):
  - Three modes: inference on load (transcode to BCn), inference on sample, inference on feedback.
  - About 5 bits/texel for a 9-10-channel PBR bundle vs 24 bits/texel for BCn, at "PSNR of 40 to 50 dB".
  - APIs: D3D12 (LinAlg, SM 6.10 preview) and Vulkan 1.3 (`VK_NV_cooperative_vector`), with "2-4x improvement in inference throughput" from cooperative vectors.
  - DP4a fallback on any SM6 GPU; validated on GTX 1000, RX 6000 and Arc A.
  - OS support: Windows and Linux only; no Metal/macOS.

  — [NVIDIA-RTX/RTXNTC README](https://github.com/NVIDIA-RTX/RTXNTC/blob/main/README.md)
- Microsoft added Cooperative Vector to D3D12/HLSL (SM 6.9, Agility SDK preview April 2025). — [D3D12 Cooperative Vector blog](https://devblogs.microsoft.com/directx/cooperative-vector/); [DXR 1.2 / neural rendering GDC 2025](https://devblogs.microsoft.com/directx/announcing-directx-raytracing-1-2-pix-neural-rendering-and-more-at-gdc-2025/)
- GDC 2026 D3D12 additions: long vectors, a wave-scope `Matrix<...>` linear-algebra type, and a "DirectX Compute Graph Compiler" for ML models. — [asawicki.info GDC 2026](https://asawicki.info/news_1801_directx_12_news_from_gdc_2026_-_my_comments)
- NVIDIA RTX Kit bundles Neural Shaders, NTC, NRC, Mega Geometry and RTX Character Rendering. — [NVIDIA RTX Kit](https://developer.nvidia.com/rtx-kit)

**Apple Metal 4 and M5**
- Metal 4 adds `MTLTensor` as a first-class resource, an ML command encoder (runs whole networks on the GPU timeline), and "Shader ML" (small networks inside fragment/vertex/compute shaders). — [WWDC25 "Combine Metal 4 machine learning and graphics"](https://developer.apple.com/videos/play/wwdc2025/262/); [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)
- The feature table lists "Machine learning encoding" and "Tensors" as Metal 4, Apple7+ (M1 and later). — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- M5 puts a Neural Accelerator (matrix-multiply unit) in each GPU core, with ">4x the peak GPU compute performance [for AI] compared to M4". The access path is Metal Performance Primitives `mpp::tensor_ops` (e.g. `matmul2d`) at simdgroup or threadgroup scope. — [Apple Newsroom M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/); [Apple ML Research: MLX and M5 Neural Accelerators](https://machinelearning.apple.com/research/exploring-llms-mlx-m5); [BaseRT arXiv 2607.19438](https://arxiv.org/html/2607.19438v1)
- WWDC26 Metal additions: quantized tensor formats with scale factors, native quantized-weight support in MPP, Neural Accelerator support on M5 Pro/Max, and a redesigned MetalFX temporal upscaler using the Neural Engine and Neural Accelerators. — [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)
- WWDC26 session 359 describes three integration levels:
  - MetalFX.
  - The ML command encoder with `MTLPackage` models.
  - TensorOps inline in shaders, with "cooperative tensors", supporting inference and backpropagation.

  Its examples are neural tone mapping (HDRNet-style) and a per-frame online-trained 3-4-4-3 MLP sky-visibility probe. Neural materials, NTC and NRC are "not covered". — [WWDC26 session 359](https://developer.apple.com/videos/play/wwdc2026/359/)

**3D Gaussian splatting**
- In games, 3DGS is still plugin-based: no first-party UE 5.7 module; Luma and Polycam integrations; Aras P's UnityGaussianSplatting works on Mac; Unigine 2.20 experimental. These are mostly virtual-production and capture uses. — [StraySpark: GS in UE5 2026](https://www.strayspark.studio/blog/gaussian-splatting-unreal-engine-5-capture-to-game-pipeline); [CG Channel: Unigine 2.20](https://www.cgchannel.com/2025/07/unigine-2-20-adds-support-for-3d-gaussian-splatting/); [Volinga 2025/2026 trends](https://web.volinga.ai/2025-turning-point-and-2026-trends-blog/)
- Research on cluster-LOD for splats: "Virtualized 3D Gaussians" (2025). — [arXiv 2505.06523](https://arxiv.org/pdf/2505.06523)

### Inferences
- **Mapping:** DX cooperative vectors / LinAlg correspond to Metal 4 TensorOps with cooperative tensors at simdgroup scope. Neural materials (Zeltner-style) and NTC inference-on-sample are therefore technically portable to Metal. They are only performance-viable at scale on M5-class GPUs with Neural Accelerators. On M1-M4 the same code runs on the regular ALUs (SIMD-group matrix ops exist from Apple7), akin to NTC's DP4a fallback.
- **NTC inference-on-load** (decode to ASTC/BC at load) is feasible on all Apple Silicon, since BC formats are universal on Apple9+ Macs. It is the low-risk first step: disk and download savings without per-pixel inference cost.
- **NRC / online-trained caches:** WWDC26 demonstrates in-shader backprop, so a Metal NRC is plausible on M5. No public implementation exists.
- **3DGS:** a niche for AAA gameplay rendering in 2026. Useful for captured backdrops. Sorting and blending map well to compute plus TBDR blending on Apple.

### Gaps
- No published Metal implementation or benchmark of neural materials or NTC.
- No Apple numbers for TensorOps throughput inside fragment shaders on M5 vs M4.
- No shipping game on any platform confirmed to use neural materials in production as of Sep 2026. None was found in this search; this was not exhaustively verified.

---

## 7. Upscaling and frame generation (DLSS 4/4.5, FSR 4/Redstone, XeSS, MetalFX) and MetalFX quality

### Takeaway
PC vendors are now on transformer-class (DLSS 4/4.5) or CNN-class (FSR 4) ML upscalers, with multi-frame generation up to 6x (DLSS 4.5). MetalFX is a mature temporal upscaler (Apple7+), plus frame interpolation (1 interpolated frame per 2 rendered; Apple5+) and the denoised upscaler (Apple9+). WWDC26 redesigned the temporal upscaler around the Neural Engine and Neural Accelerators on M5 Pro/Max. Independent quality comparisons of the 2025-2026 MetalFX against DLSS 4.x or FSR 4 are missing.

### Cited Findings
**PC vendors**
- DLSS 4: transformer model for SR/RR/DLAA and Multi Frame Generation. DLSS 4.5: second-gen transformer SR, Dynamic MFG, and 6x MFG (up to 5 generated frames per rendered frame). — [NVIDIA DLSS 4](https://www.nvidia.com/en-us/geforce/news/dlss4-multi-frame-generation-ai-innovations/); [NVIDIA DLSS 4.5](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/)
- FSR 4 is reported to approach recent DLSS quality. FSR Redstone adds ML frame generation, Ray Regeneration and Neural Radiance Caching, all RDNA4-only. — [Notebookcheck FSR4 vs DLSS](https://www.notebookcheck.net/FSR-4-vs-FSR-3-vs-DLSS-FSR-4-shows-remarkable-improvement-over-FSR-3-and-approaches-latest-DLSS-version-in-quality.973137.0.html); [GPUOpen FSR Redstone](https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/)
- Sony's SIGGRAPH 2026 talk "Upgrading PSSR on PlayStation 5 Pro" "walk[s] much of it back" to focus the network on pattern recognition. — [Advances 2026](https://advances.realtimerendering.com/s2026/index.html)

**MetalFX feature availability**
- Spatial upscaling: Apple3+. Temporal upscaling: Apple7+. Frame interpolation: Apple5+. Denoised upscaling: Apple9+. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

**MetalFX in Metal 4 (WWDC25)**
- Dynamic-resolution input, reactive mask, and exposure debugger. Apple recommends at most 2x scaling unless higher ratios are needed.
- Frame interpolation needs at least 30 fps base and generates 1 frame between N-1 and N. It offers three UI-compositing strategies and guidance on pacing via the Metal HUD.

  — [WWDC25 session 211](https://developer.apple.com/videos/play/wwdc2025/211/)

**MetalFX in WWDC26**
- Redesigned temporal upscaler using the Neural Engine plus Neural Accelerators on M5 Pro/Max, "reconstruct fine detail from significantly lower render resolutions". Also subrectangle processing, motion-vector support and distortion-field handling. — [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/)

**Historic and anecdotal quality evidence**
- Digital Foundry (Resident Evil Village, 2022-era MetalFX) found good detail restoration and disocclusion handling, but weaknesses on transparencies and hair. Tom's Hardware reported MetalFX's legal notices reference AMD FSR. — [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/amd-fsr-is-the-building-block-for-apples-metalfx-upscaling-tech-the-apps-legal-info-references-the-usage-of-amd-fsr); [MacRumors forum discussion](https://forums.macrumors.com/threads/observations-discussion-on-apple-silicon-graphics-performance-with-metalfx-upscaling.2368474/page-5)
- Cyberpunk on base M4 with FSR 3.1 frame generation averaged about 40 fps but "feels jittery and off-timed" (per search summary). — [Beebom](https://beebom.com/i-tested-cyberpunk-2077-on-m4-mac-and-its-really-good/)

### Inferences
- A native Apple engine should target MetalFX temporal (or the denoised upscaler on M3+) as the primary AA/upscaler, plus MetalFX Frame Interpolation. Keep an engine TSR (FSR 3.1-class, open source) as a fallback and reference.
- Expect the M5 Pro/Max neural MetalFX to narrow the gap to DLSS 4 / FSR 4. This is unverified: no independent image-quality study was found.
- Multi-frame generation beyond 2x (DLSS 4.5 style) has no Apple equivalent. MetalFX interpolation is 1 generated frame per rendered frame.

### Gaps
- No Digital Foundry-class comparison of Metal 4 / WWDC26 MetalFX against DLSS 4.5 or FSR 4.
- No XeSS 2/3 specifics were researched, as they are out of scope for Apple.
- MetalFX cost in ms per resolution and chip is unpublished.

---

## 8. SIGGRAPH/HPG/GDC 2025-2026 developments that change the picture

### Takeaway
Three shifts matter for an Apple-native AAA engine:
1. Cluster geometry is becoming a cross-vendor ray-tracing primitive (DXR 2.0 CLAS/PTLAS at GDC 2026, RTX Mega Geometry 2.0). Apple has no equivalent, which widens the RT gap for Nanite-density content.
2. Neural shading is standardizing (SM 6.9 cooperative vectors, then SM 6.10 LinAlg), and Apple has a comparable primitive (Metal 4 TensorOps plus M5 Neural Accelerators).
3. Production rendering is converging on stochastic lighting, radiance caches and ML denoise/upscale, all of which Apple now officially supports at the MetalFX level.

### Cited Findings
**GDC 2025 and GDC 2026 (Microsoft DirectX)**
- GDC 2025: DXR 1.2 with OMM (up to 2.3x) and SER (up to 2x); cooperative vectors announced for DirectX. — [DirectX blog GDC 2025](https://devblogs.microsoft.com/directx/announcing-directx-raytracing-1-2-pix-neural-rendering-and-more-at-gdc-2025/)
- GDC 2026: DXR 2.0 with CLAS, cluster templates, compressed positions, PTLAS and indirect AS builds; LinAlg Matrix type; DirectX Compute Graph Compiler; DirectStorage 1.4 (Zstd). — [asawicki.info](https://asawicki.info/news_1801_directx_12_news_from_gdc_2026_-_my_comments)

**NVIDIA**
- RTX Mega Geometry 2.0 (cluster LOD streaming for RT) in RTX Kit 2026.3. — [Hardware Busters](https://hwbusters.com/news/rtx-mega-geometry-2-0-streams-ray-traced-geometry-into-vram-and-sheds-detail-before-it-runs-out-of-memory/)

**SIGGRAPH Advances courses**
- 2025: MegaLights (Epic); AC Shadows RT (Ubisoft); idTech8 GI (id); Indiana Jones strand hair (MachineGames); HypeHype stochastic tile lighting; adaptive voxel OIT (Activision); ReSTIR+diffusion SSS (NVIDIA). — [Advances 2025](https://advances.realtimerendering.com/s2025/index.html)
- 2026: ORCA radiance cache for PT (EA SEED); PSSR upgrade (Sony); Variable Rate Ray Tracing in CoD MW4 (Infinity Ward); Smolder volumetrics (IOI); Roblox SLIM; adaptive tessellation/subdivision in compute (Meta). — [Advances 2026](https://advances.realtimerendering.com/s2026/index.html)

**NVIDIA research**
- ReSTIR PT Enhanced (2026), ReSTIR PG (SIGGRAPH Asia 2025), ReSTIR BDPT (2025). — [NVIDIA research](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/)

**Apple**
- WWDC25: Metal 4 with tensors, MetalFX frame interpolation, the denoised upscaler and intersection function buffers. — [WWDC25 session 211](https://developer.apple.com/videos/play/wwdc2025/211/)
- M5 / A19 (Apple10): third-gen RT, Neural Accelerators, sampler min/max reduction, depth bounds, ICB raster state, universal texture compression including shader-write textures. — [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf); [Apple Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431/)
- WWDC26: neural MetalFX upscaler on M5 Pro/Max, quantized tensors, neural rendering pipelines session, and Game Porting Toolkit 4. — [WWDC26 Metal guide](https://developer.apple.com/wwdc26/guides/metal/); [WWDC26 session 359](https://developer.apple.com/videos/play/wwdc2026/359/)

### Inferences
- The RT feature gap between Metal and DXR is widening at the top: SER, OMM, CLAS and PTLAS are all absent in Metal. The ML/neural gap is narrowing, with Metal TensorOps comparable to cooperative vectors/LinAlg.
- Apple is betting on ML reconstruction (the MetalFX denoiser plus neural upscaler) to compensate for fewer rays. An engine designed around a low ray budget plus radiance caching plus MetalFX aligns with Apple's hardware direction.

### Gaps
- HPG 2025/2026 and EGSR 2025/2026 papers were not surveyed individually.
- GDC 2026 Apple sessions, if any, were not found.

---

## 9. Summary feasibility matrix for Apple Silicon (synthesis of sections 1-8)

### Takeaway
For a native Apple AAA engine in 2026-2028, a pragmatic stack is:
- **Geometry:** virtualized cluster-LOD rasterization on M2+ (M1 via a 32-bit fallback or hardware mesh-shader path).
- **Shadows:** VSM-style or RT shadows.
- **Lighting:** stochastic many-light direct lighting.
- **GI:** Lumen/GI-1.0-style radiance-cache GI with HWRT as an M3+ tier.
- **Reconstruction:** MetalFX (denoised) upscaling plus frame interpolation.
- **Neural:** optional features on M5 via Metal 4 TensorOps.

Full path tracing and ray tracing against full-detail cluster geometry remain the weakest fits.

### Cited Findings
- Family/feature gates: [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).
- Shipping RT data point: [CDPR Mac RT requirements](https://support.cdprojektred.com/en/cyberpunk/mac/sp-technical/issue/2897/ray-tracing-on-mac).
- Nanite on M2+: [UE 5.3 release notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.3-release-notes?application_version=5.3).
- M5 RT and ML changes: [Apple Tech Talk 111431](https://developer.apple.com/videos/play/tech-talks/111431/); [WWDC26 session 359](https://developer.apple.com/videos/play/wwdc2026/359/).

### Inferences
Each row is an inference built on the findings above.

| Technique | Min Apple HW | Feasibility | Key blocker / note |
|---|---|---|---|
| Cluster-LOD DAG (offline build) | any | Feasible | meshoptimizer clusterlod.h; CPU/memory heavy offline |
| Visbuffer SW raster (64-bit atomic max) | M2 (macOS) / M3+ full | Feasible | M1 needs 32-bit workaround (est. 2.5-5x atomic cost) |
| Mesh-shader HW raster of clusters | M1 (HW-accel M3+) | Feasible | RT and function pointers incompatible with mesh pipelines |
| Two-phase HZB occlusion culling | M1 | Feasible | M5 adds sampler min/max reduction and depth bounds |
| Work graphs | n/a | Not available | Emulate with indirect dispatch chains; low impact |
| Virtual shadow maps | M2+ | Feasible with work | Many small pages vs TBDR pass overhead |
| RT shadows / MegaLights-style | M3+ | Feasible | Alpha-test cost (no OMM) |
| ReSTIR DI | M3+ | Feasible | Use `intersector` API to keep reorder stage |
| ReSTIR GI + hash radiance cache | M3 Max / M4 Max / M5 Pro+ | Feasible with work | Ray throughput; no SER |
| ReSTIR PT / full PT | M5 Max (demo/ultra tier) | Weak | Ray throughput, no SER/OMM; rely on MetalFX denoiser + FI |
| Lumen-like SDF/surface cache GI, GI-1.0/Brixelizer | M1+ | Feasible | Compute-only; safest baseline |
| RT vs full-detail cluster geometry (Mega Geometry) | n/a | Blocked | No CLAS/PTLAS in Metal |
| NRD-class denoisers | M1+ | Feasible with work | No Metal port; HLSL port needed |
| MetalFX denoised upscaler | M3+ | Feasible | Requires specific aux G-buffer channels |
| Neural materials / NTC-on-sample | M5 (HW), M1-M4 slow | Feasible with work | No Apple sample; TensorOps port |
| NTC-on-load | M1+ | Feasible | Transcode to BC/ASTC at load |
| NRC (online-trained) | M5 | Feasible with work | In-shader backprop shown at WWDC26; no implementation |
| MetalFX upscaling + frame interpolation | M1+ (FI Apple5+) | Feasible | Only 1 interpolated frame; quality vs DLSS 4.5 unverified |

### Gaps
- Quantitative performance comparisons (ms per technique on M3/M4/M5 vs RTX 40/50 or consoles) are largely unpublished. The matrix is qualitative.
