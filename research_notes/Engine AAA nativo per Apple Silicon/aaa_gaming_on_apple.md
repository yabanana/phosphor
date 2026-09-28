# AAA Gaming on Apple Platforms (2026) and Existing Engines' Metal Backends

Research date: 2026-09-28. Scope: native AAA on Apple Silicon (macOS, plus iPadOS/iOS), porting pipelines and lessons, Apple's tools (GPTK, Metal Shader Converter, D3DMetal, Metal 4 / MetalFX), third-party engine Metal backends, market data, cross-device convergence.
Source-quality note: many performance figures come from secondary tech press (Notebookcheck, Wccftech, TweakTown, Tom's Guide). Apple developer session pages (developer.apple.com/videos) are primary sources. Some search-engine summaries were cross-checked by fetching the page; items marked "(snippet only)" were seen only in search-result summaries and were not verified on the page.

---

## 1. Recent native AAA releases/ports on Mac: engines, Metal features, performance

### Takeaway
By late 2025/2026 the native Mac AAA catalog includes Cyberpunk 2077 Ultimate Edition (July 2025, with ray/path tracing and MetalFX), Assassin's Creed Shadows/Mirage (Anvil), the Capcom RE Engine titles, Death Stranding DC (Decima), and Crimson Desert (Pearl Abyss BlackSpace Engine, a day-one Mac launch in March 2026 with MetalFX upscaling, frame generation and denoiser plus hardware RT on M3/M4). An M4 Max performs roughly like a 100 W RTX 5060 laptop GPU. Heavy RT is usable only with upscaling, and several ports drop back to software RT or lower presets.

### Cited Findings
**Cyberpunk 2077 (CD Projekt Red, REDengine)**
- Cyberpunk 2077: Ultimate Edition (base game + Phantom Liberty + Patch 2.3) became available on July 17, 2025 on the Mac App Store, Steam, GOG and Epic — [Inven Global](https://www.invenglobal.com/articles/22679/how-cyberpunk-2077-succeeded-in-running-at-60-fps-on-mac) (snippet only); see also [TechPowerUp](https://www.techpowerup.com/338984/cyberpunk-2077-lands-on-apple-silicon-with-full-m-series-support).
- WWDC26 session "Bringing Cyberpunk 2077 to Mac": the pipeline used native Apple Silicon builds, Metal Shader Converter (automated DXIL→Metal in the build), a native Metal rendering foundation, ray tracing and path tracing "with performance optimization while maintaining visual parity", and MetalFX Upscaling with Dynamic Resolution Scaling. A "For This Mac" preset system sets a 30 or 60 FPS target per Mac model. Example given: M5 Max MacBook Pro, Ultra preset, 60 FPS target, MetalFX DRS at 50–80% internal resolution, output 2336×1460, HDR auto-on, VSync 60 — [Apple Developer WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- The same session reports stable 60 FPS in Dogtown (the heaviest area), Mac Game of the Year in the Apple 2025 App Store Awards, and 35M copies sold on all platforms plus 10M Phantom Liberty — [Apple Developer WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- Notebookcheck (Nov 20, 2025), M4 Max 40-core GPU MacBook Pro 16, native macOS: 1080p Low 136.1 / Medium 109 / High 95.5 / Ultra 84.3 fps; 1440p Ultra 53.8 fps; 4K Ultra 23.6 fps (34.1 fps with MetalFX Quality); RT Ultra at QHD 37 fps. Power draw about 119 W at 1080p Ultra. The native version was 46% more efficient than running under CrossOver. Overall the M4 Max is "roughly on the level of laptops with fast versions of the mobile GeForce RTX 5060 with a TGP of 100 Watts" — [Notebookcheck](https://www.notebookcheck.net/Cyberpunk-AC-Shadows-on-Apple-s-M4-Max-Gaming-performance-comparable-to-the-RTX-5060-Laptop.1166765.0.html).
- Reported 1600p, 61 fps average on M4 Max with RT and MetalFX — [TweakTown](https://www.tweaktown.com/news/106804/cyberpunk-2077-runs-at-1600p-61fps-on-macbook-pro-16-with-apple-m4-max-chip-and-ray-tracing/index.html) (snippet only). Apple's WWDC25 demo of 120 fps on M4 Max most likely ran without path tracing and relied on MetalFX upscaling plus frame interpolation — [Wccftech](https://wccftech.com/m4-max-running-cyberpunk-2077-ultimate-edition-at-120fps-deeper-dive/) (snippet only; this is analysis, not confirmation).
- Cross-generation M1–M4 benchmarks exist — [Tom's Guide](https://www.tomsguide.com/gaming/we-benchmarked-cyberpunk-2077-on-mac-heres-how-well-it-runs-on-m1-m4-macs-vs-windows); [Notebookcheck M1 Air→M4](https://www.notebookcheck.net/Cyberpunk-2077-tested-on-M1-Air-M1-Max-M3-Max-and-M4-MacBook-Pro-Here-s-how-well-it-runs-on-each.1061925.0.html) (not fetched).

**Assassin's Creed Shadows (Ubisoft, Anvil)**
- On M4 Max: 1080p Ultra High 33 fps, 1440p Ultra High 29 fps, 1080p High about 50 fps — [Notebookcheck](https://www.notebookcheck.net/Cyberpunk-AC-Shadows-on-Apple-s-M4-Max-Gaming-performance-comparable-to-the-RTX-5060-Laptop.1166765.0.html).
- On macOS, AC Shadows uses software selective ray tracing, not hardware RT, even though the M4 Max supports hardware RT and mesh shading — [Notebookcheck](https://www.notebookcheck.net/Cyberpunk-AC-Shadows-on-Apple-s-M4-Max-Gaming-performance-comparable-to-the-RTX-5060-Laptop.1166765.0.html) (via search summary).
- Sold on the Mac App Store — [App Store listing](https://apps.apple.com/kh/app/assassins-creed-shadows/id6497794841?mt=12).

**Crimson Desert (Pearl Abyss, BlackSpace Engine)**
- Launched on macOS March 19, 2026, the same day as all other platforms. The Mac version uses MetalFX Upscaling, Frame Generation and Denoiser, with hardware RT and mesh shading on M3/M4 — [Crimson Desert Wiki](https://thegameswiki.com/crimson-desert/wiki/macos-version) (fan wiki, secondary; snippet only). An M4 Max test at 1440p/4K with MetalFX FG is on [YouTube](https://www.youtube.com/watch?v=16_EV9TnPek) (not viewed).
- It appeared among Mac titles teased at WWDC25 — [AppleInsider](https://appleinsider.com/articles/25/06/09/all-the-mac-games-apple-teased-at-wwdc-25).

**Capcom RE Engine, Kojima Productions (Decima), others**
- Capcom runs a dedicated "Resident Evil Series Releases for Apple Devices" page — [Capcom](https://www.capcom-games.com/apple/en-us/) (HTTP 403, content not verified). RE Village launched on Mac in 2022 (older) — [TechRadar](https://www.techradar.com/news/apple-macbook-owners-can-finally-join-in-the-gaming-fun-with-resident-evil-village).
- Resident Evil Requiem (Feb 27, 2026) has no native macOS port. It runs through CrossOver 26 at about 70 fps on an M4 Max after settings tweaks, and crashes on M1/M2 — [Notebookcheck](https://www.notebookcheck.net/Resident-Evil-Requiem-runs-at-70-FPS-on-Apple-silicon-despite-no-macOS-port.1241116.0.html); [Wccftech](https://wccftech.com/m4-max-decent-performance-running-resident-evil-requiem/) (snippets only).
- Death Stranding Director's Cut was announced at WWDC23 by Kojima ("blown away by the power of Apple silicon and Metal 3... MetalFX upscaling") and released on iPhone, iPad and Mac on Jan 30, 2024. Kojima Productions pledged that future titles will come to Apple platforms — [PC Gamer](https://www.pcgamer.com/kojima-makes-surprise-appearance-on-apple-stream-to-praise-the-macs-rendering-pipeline-and-announce-death-stranding-for-mac/); [MacRumors](https://www.macrumors.com/2024/01/30/death-stranding-directors-cut-iphone-mac/); [Kojima Productions](https://www.kojimaproductions.jp/en/deathstranding_dc_mac).
- Aggregator lists of native Apple Silicon AAA: RE4/Village/RE2/RE3, Death Stranding DC, AC Shadows/Mirage, Baldur's Gate 3, Control, Stray, No Man's Sky, WoW, FFXIV, Diablo IV, Lies of P (UE4), GTA Vice City DE, and for 2026 Cronos: The New Dawn, HITMAN World of Assassination, Crimson Desert, inZOI. One aggregator counts about 25 major AAA titles with official Apple Silicon builds and claims "RE4 hitting 90 fps at 1440p High on M5 Pro" — [tekknological](https://tekknological.com/2026/02/20/native-mac-games-2025/); [Macworld best Mac games](https://www.macworld.com/article/670824/best-mac-games.html) (low-quality aggregators, snippet only; treat as unverified).
- Sniper Elite 5 was slated for Mac, iPhone and iPad in Q1 2026 — [PCQuest](https://www.pcquest.com/news/mac-gaming-2026-apple-gaming-metal-4-game-porting-toolkit-12004351) (snippet only).
- In GPTK 4 testing (a translation layer, not native ports), 007 First Light, Battlefield 6, Subnautica 2 and Red Dead Redemption 2 were run. So these titles are still unported and are being run through translation — [AppleInsider](https://appleinsider.com/articles/26/06/17/apples-game-porting-toolkit-4-is-a-big-improvement-for-modern-game-coders).

### Inferences
- The in-house engines behind the flagship ports (REDengine, Anvil, RE Engine, Decima, BlackSpace) all reached Metal through DXIL/HLSL→Metal via Metal Shader Converter or their own cross-compilers. The native engine port is not the barrier; per-title business cases are.
- M4 Max ≈ RTX 5060 laptop (100 W) is a useful planning anchor. Path tracing on Apple Silicon is practical only with aggressive upscaling (50–80% DRS) plus denoising and frame interpolation.
- Studios sometimes pick software RT on Mac (AC Shadows), which suggests Metal HW-RT integration or its performance is not yet a drop-in win for every engine.

### Gaps
- No Digital Foundry deep-dive on a 2025–2026 Mac port was found in this pass.
- No verified M5 Max per-title fps numbers beyond CDPR's preset example.
- The Capcom Apple page was blocked (403). The exact list of RE titles and their Mac features (MetalFX, RT) is unverified here.
- Baldur's Gate 3, No Man's Sky, Control, Stray, Hitman, Lies of P and Mirage Mac technical details (feature sets, fps) were not researched individually.

---

## 2. Developer postmortems and talks on porting to Metal

### Takeaway
Apple's canonical porting flow (WWDC23 onward) is: evaluate the unmodified Windows build in GPTK with the Metal HUD, convert shaders with Metal Shader Converter, then build a native Metal renderer with MetalFX, EDR/HDR, the Game Controller framework and CAMetalDisplayLink. CDPR's WWDC26 postmortem follows this path and stresses CPU-side bottlenecks, per-device presets and platform-native polish. Detailed TBDR or memory lessons are thin in public talks.

### Cited Findings
- WWDC23 "Bring your game to Mac, Part 1": run the unmodified Windows build in GPTK and use the Metal Performance HUD (which shows translation-specific data such as D3D version, encoder counts, geometry shader and tessellation use) and Instruments Metal System Trace. Apple stresses that translated performance includes overhead and does not predict native performance. The native path adds MetalFX, fast resource loading, offline shader compilation, mesh shaders, RT, EDR via CAMetalLayer, CAMetalDisplayLink for pacing, and the Game Controller framework. Apple suggests keeping existing audio middleware (Wwise/FMOD) for audio — [Apple Developer WWDC23 #10123](https://developer.apple.com/videos/play/wwdc2023/10123/) (older, 2023).
- CDPR (WWDC26) ran the Windows build in GPTK first and gathered three data streams: in-engine frame-time stats, the Metal HUD, and per-thread engine profiling. They replayed repeatable "hotspot sequences" for these runs.
  - In GPTK, GPU time was "healthy" on high-spec hardware. Dense city driving with traffic and crowds produced CPU spikes.
  - Live shader-translation oscillation and audio-middleware overhead turned out to be translation artifacts that disappeared once native.
  - The native port proceeded in this order: native builds → data pipeline with macOS as a parallel platform → "architecture bridge" unit tests for x86→ARM64 assumptions → automated Metal Shader Converter in the build → validation from stationary to dynamic scenes → RT/PT → MetalFX + DRS → per-device presets.
  - Platform integration covered EDR auto-calibration, occlusion-state pause, display-change handling, cursor handling, trackpad support, head-tracked spatial audio (AVAudioEngine), Game Mode, and iCloud saves.
  - Source: [Apple Developer WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- One report says CDPR engineers spent about 18 months rebuilding the REDengine pipeline for Metal — [TechPowerUp / Inven Global via search](https://www.invenglobal.com/articles/22679/how-cyberpunk-2077-succeeded-in-running-at-60-fps-on-mac) (snippet only; not confirmed in the Apple session transcript).
- Interview in which CDPR argued that Cyberpunk shows Macs are good for gaming — [The Escapist](https://www.escapistmagazine.com/news-apple-cyberpunk-2077-mac-interview/) (not fetched).
- WWDC26 "Speedrun your game port with agentic coding" (session 357) ported Microsoft's D3D12 MiniEngine to native Metal and added Metal 4 to Godot's Metal 3 backend "in a few days". Lessons from the session:
  - Register resources in residency sets before GPU access.
  - Use Metal Shader Converter reflection (IRShaderReflection) for root-signature and argument-buffer offsets rather than hardcoded layouts.
  - Map D3D12 barriers precisely to Metal 4 stages instead of using blanket barriers.
  - MetalFX jitter must be in pixel space, and motion vectors need correct scale and conventions.
  - Frame interpolation needs a dedicated present thread with precise timing.
  - Source: [Apple Developer WWDC26 #357](https://developer.apple.com/videos/play/wwdc2026/357/).
- WWDC25 "Go further with Metal 4 games" gave this guidance:
  - Frame interpolation needs at least 30 FPS before interpolation. UI can be composited, rendered offscreen, or drawn every frame.
  - Check pacing with the Metal HUD frame-interval graph.
  - For the denoised upscaler, noisy or correlated random numbers hurt quality, normals need a signed format, and metallic materials need darker diffuse albedo.
  - Source: [Apple Developer WWDC25 #211](https://developer.apple.com/videos/play/wwdc2025/211/).

### Inferences
- Frame-pacing and presentation infrastructure should be designed early in a native engine: a dedicated present thread, CAMetalDisplayLink, and interpolation-aware UI layers. So should explicit residency management.
- CPU-side systems (crowds, traffic, streaming) were the main risk in Cyberpunk, not the GPU. A native engine should budget for Apple's P/E-core topology and job systems that scale across cores.
- Per-device auto-presets tied to a 30/60 target with DRS are Apple's recommended UX pattern.

### Gaps
- No public talks with specific TBDR lessons were found for Capcom RE Engine, Ubisoft Anvil, Hello Games (No Man's Sky) or Larian (BG3) in this pass: tile memory, memoryless render targets, load/store actions, programmable blending.
- No specific unified-memory or streaming numbers (memory budgets, PSO cache sizes, shader compile times) appear in CDPR's session.

---

## 3. Game Porting Toolkit (v1–v4), Metal Shader Converter, D3DMetal

### Takeaway
GPTK has grown each WWDC from an evaluation environment into a full porting suite:
- GPTK 1 (2023): Wine-based evaluation plus D3DMetal plus Metal Shader Converter.
- GPTK 3 (2025): Metal 4 integration and MetalFX interpolation/denoiser guidance.
- GPTK 4 (June 2026): DX12→Metal 4 translation (DX11 falls back to Metal 3), open-source agent skills, metal-cpp, command-line `gpucapture`/`gpudebug` for macOS 27, and an explicitly agentic porting workflow.

### Cited Findings
- GPTK 1 (WWDC23) translates x86 instructions and Windows APIs (DX12, input, audio, networking, file system) to evaluate unmodified Windows builds. Metal Shader Converter converts HLSL (DXIL) to Metal for all stages, including geometry, tessellation, mesh and RT — [Apple Developer WWDC23 #10123](https://developer.apple.com/videos/play/wwdc2023/10123/) (older).
- GPTK 3 (WWDC25) integrates with Metal 4 and adds metrics for developers. Games can be translated to use MetalFX Frame Interpolation and Denoising — [AppleInsider](https://appleinsider.com/articles/25/06/09/metal-4-game-porting-toolkit-3-boost-frame-rate-ray-tracing-performance); [Wccftech](https://wccftech.com/apple-metal-4-api-adds-interpolation-to-boost-gaming-performance/).
- GPTK 4 (WWDC26, June 2026):
  - Metal 4 is Apple Silicon-only. DX12→Metal 4 translation improved, and DirectX 11 games fall back to Metal 3.
  - Cyberpunk 2077 on an M3 Max MacBook Pro ran 10% faster under DX12→Metal 4 translation than under Metal 3.
  - RDR2 on "MacBook Neo" improved 25% overall (+7 fps average).
  - 007 First Light reached 60–70 fps at 1080p Medium (it crashed on GPTK 3). Subnautica 2 on a Mac mini M4 improved 6%, and Battlefield 6 memory usage stabilized.
  - Source: [AppleInsider](https://appleinsider.com/articles/26/06/17/apples-game-porting-toolkit-4-is-a-big-improvement-for-modern-game-coders); also [AppleInsider on agentic coding](https://appleinsider.com/articles/26/06/08/game-porting-toolkit-4-ushers-in-support-for-agentic-coding).
- The Apple GitHub repo `apple/game-porting-toolkit` (Apache 2.0, feedback only, no PRs) contains:
  - an agent-skills collection (for example `using-metalfx-frame-interpolation`)
  - metal-cpp
  - end-to-end porting samples
  - references to GPTK 4, macOS 27 and Xcode 27, `gpucapture`/`gpudebug`, and MCP tools for Xcode/LLDB
  - a four-phase-per-milestone workflow (prepare, execute, validate with GPU capture and leak checks, hand off) whose state persists in `.porting/`
  - Source: [GitHub apple/game-porting-toolkit](https://github.com/apple/game-porting-toolkit); [SKILL.md example](https://github.com/apple/game-porting-toolkit/blob/main/game-porting-skills/skills/using-metalfx-frame-interpolation/SKILL.md).
- Apple says the WWDC26 agentic workflow has three stages: Discover, Plan (milestones mapped to expert skills), and Execute & Validate (app launch, Metal API validation, screen-capture comparison against ground truth from the evaluation environment, anti-pattern and memory checks) — [Apple Developer WWDC26 #357](https://developer.apple.com/videos/play/wwdc2026/357/).
- CDPR used GPTK only for evaluation and Metal Shader Converter in production — [Apple Developer WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- Third-party wrappers (CrossOver 26, which bundles D3DMetal) are how unported titles such as RE Requiem run on Mac — [Notebookcheck](https://www.notebookcheck.net/Resident-Evil-Requiem-runs-at-70-FPS-on-Apple-silicon-despite-no-macOS-port.1241116.0.html) (snippet only).

### Inferences
- A new native engine can use Metal Shader Converter to keep an HLSL authoring path while targeting Metal natively, or it can author MSL directly. The converter's IR runtime model (argument buffers emulating root signatures) carries some overhead and constraints compared with hand-designed Metal 4 argument tables.
- Translation of D3D12 to Metal 4 is now good enough that end users run unported AAA games. This lowers the relative value of "native-only" claims unless native brings clear wins (efficiency: native Cyberpunk was 46% more efficient than CrossOver per Notebookcheck).

### Gaps
- Exact GPTK version dates and the 2.x changelog (2024: AVX2 support etc.) were not verified in this pass.
- No public figures on D3DMetal translation overhead beyond the per-title deltas above.

---

## 4. Engine Metal backends: Unreal Engine 5, Unity 6.x, Godot 4.x, others

### Takeaway
UE5 on Mac has moved from a native Apple Silicon editor (5.2) to Nanite beta on M2+ (5.3), a simpler Nanite setup (5.5) and bindless Metal. Mac shader compilation was still beta as of 5.5, and the status of Lumen hardware RT on Metal is poorly documented officially. Unity 6 uses Metal by default but has no Metal ray tracing, only an "eventual" intent as of Dec 2025. Godot 4.4 shipped a native Metal driver (Apple Silicon only) with MetalFX, and Apple itself demonstrated adding Metal 4 to it. The Forge (Confetti) and proprietary engines (REDengine, Anvil, RE Engine, Decima, BlackSpace) are the proven Metal-native AAA paths.

### Cited Findings
**Unreal Engine 5**
- UE 5.2 brought native Apple Silicon support for the editor and other macOS improvements — [Epic tech blog](https://www.unrealengine.com/en-US/tech-blog/unreal-engine-5-2-brings-native-support-for-apple-silicon-and-other-developments-for-macos) (older, 2023; not fetched). Epic also published a "feature parity with Windows" progress report — [Epic tech blog](https://www.unrealengine.com/tech-blog/bringing-unreal-engine-on-macos-up-to-feature-parity-with-windowsprogress-report) (HTTP 403; content not verified).
- Nanite relies on image atomics and forward-progress guarantees that M1 may lack. Experimental support on M2 was off by default, and UE 5.3 brought beta Nanite on Apple Silicon M2 with SM6 by default. In earlier versions Lumen used only the software ray tracer on Apple Silicon — [UE 5.3 release notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5.3-release-notes?application_version=5.3) (search-snippet summary; not fetched).
- UE 5.5 release notes contain these Apple items:
  - "Experimental support for visionOS 2.0 and Mixed Immersive Mode which enables passthrough support with Metal".
  - Metal is listed with DX12 and Vulkan as receiving bindless resource support.
  - "for macOS and Linux host build machines, C++ compilation is stable... however shader compilation on those platforms remains in beta".
  - Source: [UE 5.5 release notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-5-release-notes).
- The official UE 5.8 "Hardware Ray Tracing" doc page does not mention Mac or Metal at all. It names DX12 and Vulkan (Windows/Linux) only — [Epic docs](https://dev.epicgames.com/documentation/unreal-engine/hardware-ray-tracing-in-unreal-engine). A third-party blog claims that "UE5.7's Metal path is the most capable Metal RHI in any shipping UE release", that "Lumen on Metal supports both software and hardware ray tracing" and that Nanite on Metal is "in the same performance neighborhood as D3D12" — [StraySpark blog](https://www.strayspark.studio/blog/apple-silicon-m5-unreal-engine-development-2026) (unverified secondary; this conflicts with Epic's HWRT doc, which is silent on Metal).
- Epic's "Supported Features by Rendering Path for Desktop" page (UE 5.8) exists and probably holds the authoritative matrix — [Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/supported-features-by-rendering-path-for-desktop-with-unreal-engine) (not fetched).
- Lies of P, a native Mac title, is built on UE4 — [tekknological](https://tekknological.com/2026/02/20/native-mac-games-2025/) (aggregator).

**Unity 6.x**
- Unity uses Metal by default on iOS, tvOS and macOS players — [Unity Manual 6000.3 Metal](https://docs.unity3d.com/6000.3/Documentation/Manual/Metal.html); [Metal requirements](https://docs.unity3d.com/6000.2/Documentation/Manual/metal-requirements-and-compatibility.html) (snippets).
- Dec 2025, Unity staff (rasmusn): "It is our current intention to eventually support Metal ray tracing, but it will take time". No timeline was given — [Unity Discussions](https://discussions.unity.com/t/path-tracing-for-mac-hardware-that-supports-it/1699747).
- MetalFX support in Unity has been a long-standing forum request. No confirmation of built-in MetalFX in Unity 6.3 was found — [Unity forum](https://forum.unity.com/threads/will-metalfx-upscaling-be-supported.1306038/).

**Godot 4.x**
- Godot 4.4 added a direct Metal rendering driver instead of MoltenVK. It is "at least as fast as Vulkan and in many cases much faster" on Apple hardware, supports Apple Silicon only (no x86_64), and offers MetalFX upscaling as an option — [Godot 4.4 release](https://godotengine.org/releases/4.4/); [Godot Metal driver README](https://github.com/godotengine/godot/blob/master/drivers/metal/README.md).
- Community proposal to adopt Metal 4 features — [godot-proposals #13449](https://github.com/godotengine/godot-proposals/discussions/13449). At WWDC26 Apple showed Godot's Metal 3 backend updated to Metal 4 "in a few days" with agentic tooling — [Apple WWDC26 #357](https://developer.apple.com/videos/play/wwdc2026/357/).

**Other / proprietary**
- The Forge (ConfettiFX) is a cross-platform rendering framework with Metal, DX12 and Vulkan backends, including ray tracing on macOS/iOS. It powers Supergiant's engine for Hades (Windows, macOS, Switch, 2020) and Bethesda's rendering layer for Starfield — [The Forge GitHub](https://github.com/ConfettiFX/The-Forge); [README](https://github.com/ConfettiFX/The-Forge/blob/master/README.md) (snippets).
- Proprietary engines with shipped Metal ports: REDengine (Cyberpunk; [WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/)), BlackSpace (Crimson Desert; [wiki](https://thegameswiki.com/crimson-desert/wiki/macos-version)), Anvil (AC Shadows; [Notebookcheck](https://www.notebookcheck.net/Cyberpunk-AC-Shadows-on-Apple-s-M4-Max-Gaming-performance-comparable-to-the-RTX-5060-Laptop.1166765.0.html)), and Death Stranding DC, which uses Metal 3 and MetalFX ([PC Gamer](https://www.pcgamer.com/kojima-makes-surprise-appearance-on-apple-stream-to-praise-the-macs-rendering-pipeline-and-announce-death-stranding-for-mac/)). Sources do not name Decima explicitly in the Mac announcement.

**Metal 4 feature set (context for engine design)**
- WWDC25 "Go further with Metal 4 games" covered:
  - MetalFX Upscaling now accepts dynamic-resolution input and a reactive mask.
  - New MetalFX Frame Interpolation.
  - New MetalFX Denoised Upscaler, which takes normals, diffuse and specular albedo, and roughness.
  - Ray tracing gains Intersection Function Buffers (for porting DXR shader-table-style hit groups) and per-acceleration-structure flags `preferFastIntersection` / `minimizeMemoryUsage`.
  - Source: [Apple WWDC25 #211](https://developer.apple.com/videos/play/wwdc2025/211/).
- Metal 4 is Apple Silicon-only — [AppleInsider](https://appleinsider.com/articles/26/06/17/apples-game-porting-toolkit-4-is-a-big-improvement-for-modern-game-coders).

### Inferences
- The biggest opening for a new native Apple engine is that no major third-party engine treats Metal 4 as first-class for high-end features. Unity lacks Metal RT. UE's Mac HWRT and Lumen status is under-documented, and its Mac shader tooling was still beta as of 5.5. Godot is only now adopting Metal 4.
- Proven AAA-on-Metal architectures come from engines designed around explicit modern APIs (D3D12-style), then ported. None found was designed Metal-first for TBDR.

### Gaps
- No authoritative Epic statement was found (5.6–5.8) on Metal HWRT, Lumen HWRT, Virtual Shadow Maps or MegaLights on Mac. The feature-parity blog and the rendering-path matrix were not accessible or not fetched.
- Unity 6.x MetalFX, mesh shader or Metal 4 adoption status is unconfirmed.
- No postmortem on Decima's Metal backend architecture was found.

---

## 5. Market data and Apple's gaming initiatives

### Takeaway
macOS is about 2–2.4% of Steam users in 2026, now below Linux. Mac Steam users are mostly on base or low-end chips (M4 18%, M1 15%, M2 11%, M5 10%; 16 GB RAM 43%, 8 GB 29%), and high-end Max chips are a small minority. Apple's push includes the Games app, Game Overlay, Game Mode, Low Power Mode for games, and Metal 4. Premium AAA ports on iPhone sold very poorly in 2024.

### Cited Findings
- macOS share on the Steam survey: 2.35% in March 2026 (Linux passed macOS), 2.01% in April 2026, and 2.32% in July 2026 (Windows 93.67%, Linux 4.01%) — [ResetEra (March)](https://www.resetera.com/threads/steam-hardware-survey-for-march-2026-has-linux-at-5-33-another-new-record.1480495/); [GamingOnLinux (May)](https://www.gamingonlinux.com/2026/06/steam-survey-for-may-2026-is-out-linux-down-at-3-99-percent-but-still-above-macos/); [Steam survey](https://store.steampowered.com/hwsurvey/Steam-Hardware-Software-Survey-Welcome-to-Steam) (monthly figures from search summaries).
- Steam Mac-only survey, August 2026. The largest macOS versions are 26.x (26.5.2 at 30.6%); macOS 27.0.0 is at 3.86%.
  - Chips: M4 18.24%, M1 14.99%, M2 10.72%, M5 9.85%, M3 5.94%, M1 Pro 5.71%, M4 Pro 4.93%, A18 Pro 4.03%, Intel 11.08%.
  - RAM: 16 GB 43.41%, 8 GB 29.44%, 24 GB 10.74%.
  - Source: [Steam HW Survey (Mac)](https://store.steampowered.com/hwsurvey/?platform=mac).
- December 2025: M4 was 19.6% of Steam Macs, M1 18.06% and M5 about 1% — [iDropNews](https://www.idropnews.com/news/m4-most-popular-mac-gaming-chip-steam-2026/258210/) (snippet only).
- Apple Games app (announced June 2025) ships with iOS 26, iPadOS 26 and macOS Tahoe 26 (fall 2025). It includes a Library, Play Together/Challenges, Apple Arcade, and a Game Overlay on Mac and iPad. Low Power Mode for gaming was added — [Apple Newsroom](https://www.apple.com/newsroom/2025/06/introducing-the-apple-games-app-a-personalized-home-for-games/); [MacRumors](https://www.macrumors.com/guide/ios-26-games-app/).
- Game Mode raises CPU/GPU priority, reduces the impact of background tasks, and doubles the Bluetooth sampling rate for controllers and AirPods — [Apple WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- 2024 iPhone AAA port sales per mobilegamer.biz estimates, as reported by Game Rant and Kotaku: AC Mirage about 3,000 buyers; RE4 Remake about 7,000 paid (357k downloads, $208k revenue at $29.99); RE Village about 5,750; Death Stranding about 10,000 downloads. RE7 was also called a flop. The reasons given were pricing against $5–10 and F2P norms, performance, and touch controls — [Game Rant](https://gamerant.com/aaa-games-not-selling-on-iphone/); [Kotaku](https://kotaku.com/re7-aaa-ios-iphone-village-remake-flop-low-sales-why-1851595596) (older, 2024; third-party estimates).
- Over 1,700 games run on Apple Silicon (as of 2025), more than 340 of them native — [Intego](https://www.intego.com/mac-security-blog/is-2025-the-year-of-mac-gaming-top-5-reasons-to-be-a-mac-gamer/) (snippet; unclear methodology).
- Cyberpunk 2077 won Mac Game of the Year in the Apple 2025 App Store Awards — [Apple WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).

### Inferences
- Steam's 2% share across its large monthly user base is still millions of Mac users. However, most of them have base-tier chips with 8–16 GB RAM, so the engine's scalability floor (M1/M2/A18 Pro, 8 GB) matters more than the M-Max ceiling.
- The A18 Pro showing up on the Steam Mac survey (4%) suggests a low-cost, iPhone-chip Mac (the "MacBook Neo" referenced by AppleInsider) is now part of the Mac gaming base. That further lowers the hardware floor.
- The commercial case for Apple-exclusive AAA is weak on the evidence: poor iPhone AAA sales and a small Steam share. Apple-exclusive high-end titles appear to be Apple-funded or strategic (Arcade, launch showcases) rather than standalone commercial successes.

### Gaps
- No verified total install base of Apple Silicon Macs or official Apple "Mac gamers" figure was found in this pass.
- No verified Mac-specific sales figures for Cyberpunk, AC Shadows or Crimson Desert on Mac.
- No true Apple-exclusive high-end AAA title was identified. Apple Arcade premium or exclusive high-end titles were not researched.

---

## 6. Apple platform convergence (Mac, iPad, iPhone Pro, Vision Pro)

### Takeaway
"One binary, many devices" is real for Apple Silicon ports: Death Stranding DC, RE Village/RE4/RE7 and AC Mirage shipped on iPhone 15 Pro-class, M-series iPad and Mac, and Sniper Elite 5 was slated for all three. The Games app and iCloud saves provide cross-progression. Vision Pro support is mostly via engines (UE 5.5 experimental visionOS). Commercial results on iPhone have been poor.

### Cited Findings
- Death Stranding Director's Cut launched on iPhone, iPad and Mac together (Jan 30, 2024) — [MacRumors](https://www.macrumors.com/2024/01/30/death-stranding-directors-cut-iphone-mac/).
- AC Mirage, RE4 Remake, RE Village, RE7 and Death Stranding shipped as iPhone ports in 2024 — [Game Rant](https://gamerant.com/aaa-games-not-selling-on-iphone/); [Kotaku](https://kotaku.com/re7-aaa-ios-iphone-village-remake-flop-low-sales-why-1851595596).
- Sniper Elite 5 was planned for Mac, iPhone and iPad in Q1 2026 — [PCQuest](https://www.pcquest.com/news/mac-gaming-2026-apple-gaming-metal-4-game-porting-toolkit-12004351) (snippet only).
- Metal 4 and GPTK 3 target Mac, iPad and iPhone — [macitynet](https://www.macitynet.it/con-metal-4-e-game-porting-toolkit-3-apple-migliora-frame-rate-e-ray-tracing-per-giochi-su-mac-ipad-e-iphone/).
- Apple's Games app is shared across iOS, iPadOS and macOS, with a shared library and progression — [Apple Newsroom](https://www.apple.com/newsroom/2025/06/introducing-the-apple-games-app-a-personalized-home-for-games/).
- CDPR used iCloud Drive saves plus an in-house solution for cross-progression ("continue on any device") — [Apple WWDC26 #356](https://developer.apple.com/videos/play/wwdc2026/356/).
- UE 5.5 added experimental visionOS 2.0 support, including Mixed Immersive Mode passthrough with Metal — [UE 5.5 release notes](https://dev.epicgames.com/documentation/unreal-engine/unreal-engine-5-5-release-notes).
- Testing of AAA games on iPhone 17 Pro Max exists as YouTube and blog content — [Appleosophy](https://appleosophy.com/2025/12/10/aaa-gaming-on-iphone-17-pro-transforms-mobile-gaming/) (not fetched).

### Inferences
- A native Apple engine should treat scalability as a first-class axis, from A18 Pro/A19 Pro phones and 8 GB Macs up to M5 Max. It should also share one Metal 4 renderer with per-device presets, following CDPR's model.
- Touch controls and pricing, not rendering, limited iPhone AAA uptake. Controller-first design plus cross-buy or universal purchase is a likely prerequisite.

### Gaps
- No shipped AAA title running on Vision Pro as a native immersive experience was found.
- Cyberpunk 2077 availability on iPad or iPhone was not confirmed.
- M-series iPad Pro performance data for AAA ports was not found.
