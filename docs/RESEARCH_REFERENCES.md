# Bibliografia per le fasi OPT

La ricerca del 2026-10-01 aggiunge [R87–R111](#ricerca-soc-e-piattaforma--2026-10-01)
con fonti primarie consultate e limiti espliciti. R1–R86 sono il catalogo
preesistente: verificarne testo, implementazione e disponibilità prima di
adottare una tecnica; questa integrazione non certifica nuovamente ogni voce.

Riferimenti citati nelle fasi OPT di [`ROADMAP.md`](ROADMAP.md) con la chiave
`[Rn]`. Sono raggruppati per fase OPT. I lavori del 2024–2026 sono stati
verificati in rete a settembre 2026; per i classici vale la citazione
standard (autori, titolo, sede, anno).

Come usarla: all'inizio di ogni fase OPT, leggere (o farsi riassumere da
Claude) i riferimenti della fase, poi scegliere le direzioni da provare come spike.

---

## OPT-1 · Memoria, grafo e banda

- **[R1]** Y. O'Donnell, *FrameGraph: Extensible Rendering Architecture in Frostbite*, GDC 2017.
- **[R2]** H.-K. Arntzen, *Render graphs and Vulkan — a deep dive*, blog "Maister's Graphics Adventures", 2017.
- **[R3]** P. Jain et al., *Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization*, MLSys 2020. <https://arxiv.org/abs/1910.02653>
- **[R4]** S. Williams, A. Waterman, D. Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM, 2009.

## OPT-2 · Shader, pipeline e occupancy

- **[R5]** P. Sitthi-amorn, N. Modly, W. Weimer, J. Lawrence, *Genetic Programming for Shader Simplification*, ACM TOG (SIGGRAPH Asia) 2011.
- **[R6]** R. Wang, X. Yang, Y. Yuan, W. Chen, K. Bala, H. Bao, *Automatic Shader Simplification using Surface Signal Approximation*, ACM TOG (SIGGRAPH Asia) 2014.
- **[R7]** P. Turner, *metal-benchmarks* — microarchitettura e throughput delle GPU Apple. <https://github.com/philipturner/metal-benchmarks>

## OPT-3 · Geometria, culling, dati di vertice

- **[R9]** U. Haar, S. Aaltonen, *GPU-Driven Rendering Pipelines*, SIGGRAPH 2015 Advances in Real-Time Rendering.
- **[R10]** J. Hasselgren, M. Andersson, T. Akenine-Möller, *Masked Software Occlusion Culling*, HPG 2016.
- **[R11]** M. B. Jensen, J. R. Frisvad, J. A. Bærentzen, *Performance Comparison of Meshlet Generation Strategies*, JCGT 2023.
- **[R12]** B. Kuth et al., *Towards Practical Meshlet Compression*, VMV 2024 (best paper). <https://arxiv.org/abs/2404.06359>
- **[R13]** J. Barczak, C. Benthin, D. McAllister, *DGF: A Dense, Hardware-Friendly Geometry Format for Lossily Compressing Meshlets with Arbitrary Topologies*, HPG 2024. <https://gpuopen.com/dgf/>
- **[R14]** Z. Cigolle et al., *A Survey of Efficient Representations for Independent Unit Vectors*, JCGT 2014.
- **[R22]** J. Hable, *Adaptive Tessellation and Subdivision*, SIGGRAPH 2026 Advances in Real-Time Rendering. <https://advances.realtimerendering.com/s2026/index.html>

## OPT-4 · Shading, banda, ricostruzione

- **[R15]** C. Burns, W. Hunt, *The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading*, JCGT 2013.
- **[R16]** G. Liktor, C. Dachsbacher, *Decoupled Deferred Shading for Hardware Rasterization*, I3D 2012.
- **[R17]** K. Hillesland, J. Yang, *Texel Shading*, Eurographics 2016 (short papers).
- **[R18]** M. Drobot, *Software-based Variable Rate Shading in Call of Duty: Modern Warfare*, SIGGRAPH 2020 Advances. <https://research.activision.com/publications/2020-09/software-based-variable-rate-shading-in-call-of-duty--modern-war>
- **[R19]** B. Karis, *High Quality Temporal Supersampling*, SIGGRAPH 2014 Advances.
- **[R20]** AMD, *FidelityFX Single Pass Downsampler (SPD)*, GPUOpen.
- **[R21]** *Training and Predicting Visual Error for Real-Time Applications*, arXiv 2310.09125. <https://arxiv.org/abs/2310.09125>

## OPT-5 · Budget di raggi e campionamento

- **[R23]** C. Wyman, A. Panteleev, *Rearchitecting Spatiotemporal Resampling for Production*, HPG 2021.
- **[R24]** D. Lin, M. Kettunen, B. Bitterli, J. Pantaleoni, C. Yuksel, C. Wyman, *Generalized Resampled Importance Sampling: Foundations of ReSTIR*, SIGGRAPH 2022.
- **[R25]** M. Kettunen et al., *Conditional Resampled Importance Sampling and ReSTIR*, SIGGRAPH Asia 2023.
- **[R26]** S. Zhang, D. Lin, M. Kettunen, C. Yuksel, C. Wyman, *Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing*, SIGGRAPH 2024.
- **[R27]** J. Liu, D. Lin, M. Kettunen, C. Wyman, R. Ramamoorthi, *Reservoir Splatting for Temporal Path Resampling and Motion Blur*, SIGGRAPH 2025. <https://github.com/Jebbly/Reservoir-Splatting>
- **[R28]** O. Junkins, M. Kettunen, D. Lin, R. Ramamoorthi, C. Wyman, *Compatibility-Guided Neighbor Selection for ReSTIR*, HPG 2026 (best paper).
- **[R29]** Y. Tokuyoshi, S. Ikeda, P. Kulkarni, T. Harada, *Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting*, SIGGRAPH Asia 2024.
- **[R30]** A. C. Estevez, C. Kulla, *Importance Sampling of Many Lights with Adaptive Tree Splitting*, HPG 2018.
- **[R31]** C. Yuksel, *Stochastic Lightcuts*, HPG 2019.
- **[R32]** M. Olejnik, *Variable Rate Ray Tracing in Call of Duty: Modern Warfare 4*, SIGGRAPH 2026 Advances.
- **[R33]** K. Garanzha, C. Loop, *Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing*, Eurographics 2010.
- **[R34]** C. Benthin, S. Woop, I. Wald, A. Áfra, *Improved Two-Level BVHs using Partial Re-Braiding*, HPG 2017.
- **[R35]** J. Haydel, A. Kensler, C. Yuksel, E. Brunvand, *Memory-Efficient Bounding Volume Hierarchies with Merged Nodes*, HPG 2026 (best paper).
- **[R36]** H. Gruen, C. Benthin, M. Kern, D. McAllister, *Ray Tracing Massive Amounts of Animated Geometry*, HPG 2026 (best paper, ex aequo).
- **[R42]** A. Wolfe, N. Morrical, T. Akenine-Möller, R. Ramamoorthi, *Spatiotemporal Blue Noise Masks*, EGSR 2022.
- **[R43]** W. Donnelly, A. Wolfe, J. Bütepage, J. Valdés, *FAST: Filter-Adapted Spatio-Temporal Sampling for Real-Time Rendering*, I3D 2024. <https://github.com/electronicarts/fastnoise>

## OPT-6 · Illuminazione globale, cache, ammortamento

- **[R37]** G. Boissé et al., *GI-1.0: A Fast and Scalable Two-level Radiance Caching Scheme for Real-time Global Illumination*, AMD, arXiv 2310.19855 (2023). <https://arxiv.org/abs/2310.19855>
- **[R38]** N. Binder, S. Fricke, A. Keller, *Fast Path Space Filtering by Jittered Spatial Hashing*, SIGGRAPH 2018 Talks.
- **[R39]** A. Sannikov, *Radiance Cascades: A Novel Approach to Calculating Global Illumination* (2023); R. Freeman, A. Sannikov, *Split Radiance Cascades: Real-Time Global Illumination via Sparse Radiance Probes*, arXiv 2607.20384 (2026). <https://arxiv.org/abs/2607.20384>
- **[R40]** J. Greenberg, *Speeding up Path Tracing via ORCA (Online Radiance Cache Acceleration)*, SIGGRAPH 2026 Advances (EA SEED).
- **[R41]** Z. Majercik, A. Marrs, J. Spjut, M. McGuire, *Scaling Probe-Based Real-Time Dynamic Global Illumination for Production*, JCGT 2021.
- **[R44]** C. Schied et al., *Spatiotemporal Variance-Guided Filtering*, HPG 2017.
- **[R45]** S. Hillaire, *A Scalable and Production Ready Sky and Atmosphere Rendering Technique*, EGSR 2020.
- **[R46]** A. Schneider, *Nubis Cubed: Methods (and madness) to model and render immersive real-time voxel-based clouds*, SIGGRAPH 2023 Advances.
- **[R47]** A. Mueller, *Smolder – Real-Time Volumetric Effect Rendering in Glacier and 007 First Light*, SIGGRAPH 2026 Advances.
- **[R48]** *Radiance Caching with On-Surface Caches for Real-Time Global Illumination*, HPG 2024 (PACMCGIT).

## OPT-7 · Geometria virtualizzata e mondo

- **[R49]** B. Karis, R. Stubbe, G. Wihlidal, *Nanite: A Deep Dive*, SIGGRAPH 2021 Advances.
- **[R50]** A. Kapoulkine, *Billions of triangles in minutes*, zeux.io, 2025. <https://zeux.io/2025/09/30/billions-of-triangles-in-minutes/>
- **[R51]** S. Laine, T. Karras, *High-Performance Software Rasterization on GPUs*, HPG 2011.
- **[R52]** M. Kenzel, B. Kerbl, D. Schmalstieg, M. Steinberger, *A High-Performance Software Graphics Pipeline Architecture for the GPU*, SIGGRAPH 2018.
- **[R53]** L. Lipp, A. Jarabo, M. Wimmer, L. Bode, *High-Performance Real-Time Implicit Strand-Based Hair Rendering via Software Rasterization*, PACMCGIT 9(4), 2026. <https://arxiv.org/abs/2607.04230>
- **[R54]** A. Benyoub, J. Dupuy, *Concurrent Binary Trees for Large-Scale Game Components*, HPG 2024. <https://arxiv.org/abs/2407.02215>
- **[R55]** D. van Antwerpen et al., *Real-time Path Tracing of Massive Dynamic Foliage*, HPG 2026 (best paper, ex aequo).
- **[R56]** S. Jeschke et al., *Water Surface Wavelets*, SIGGRAPH 2018.
- **[R57]** M. Salvi, K. Vaidyanathan, *Multi-Layer Alpha Blending*, I3D 2014.
- **[R58]** C. Münstermann et al., *Moment-Based Order-Independent Transparency*, I3D 2018.

## OPT-8 · Streaming, texture, materiali

- **[R59]** P. Krajcevski, S. Pratapa, D. Manocha, *GST: GPU-decodable Supercompressed Textures*, SIGGRAPH Asia 2016.
- **[R60]** S. Fujieda, T. Harada, *Neural Texture Block Compression*, EGSR 2024. <https://arxiv.org/abs/2407.09543>
- **[R61]** K. Chen, *Adaptive Virtual Texture Rendering in Far Cry 4*, GDC 2015.
- **[R62]** A. Kaplanyan, S. Hill, A. Patney, A. Lefohn, *Filtering Distributions of Normals for Shading Antialiasing*, HPG 2016.
- **[R63]** S. Makeev, *SLIM: Scaling User-Generated 3D Worlds on Roblox*, SIGGRAPH 2026 Advances.

## OPT-9 · CPU, thread, memoria unificata

- **[R64]** C. Gyrling, *Parallelizing the Naughty Dog Engine Using Fibers*, GDC 2015.
- **[R74]** J. Guo et al., *ExtraNet: Real-time Extrapolated Rendering for Low-latency Temporal Supersampling*, SIGGRAPH Asia 2021.

## OPT-10 · Simulazione

- **[R65]** A. H. Chen, Z. Liu, Y. Yang, C. Yuksel, *Vertex Block Descent*, SIGGRAPH 2024; e *Augmented Vertex Block Descent*, SIGGRAPH 2025.
- **[R66]** M. Macklin et al., *Small Steps in Physics Simulation*, SCA 2019.
- **[R67]** D. Holden, O. Kanoun, M. Perepichka, T. Popa, *Learned Motion Matching*, SIGGRAPH 2020.
- **[R68]** N. Raghuvanshi, J. Snyder, *Parametric Directional Coding for Precomputed Sound Propagation*, SIGGRAPH 2018.

## OPT-11 · Rendering neurale

- **[R69]** T. Müller, F. Rousselle, J. Novák, A. Keller, *Real-time Neural Radiance Caching for Path Tracing*, SIGGRAPH 2021.
- **[R70]** T. Müller, A. Evans, C. Schied, A. Keller, *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*, SIGGRAPH 2022.
- **[R71]** K. Vaidyanathan et al., *Random-Access Neural Compression of Material Textures*, SIGGRAPH 2023.
- **[R72]** T. Zeltner et al., *Real-Time Neural Appearance Models*, ACM TOG 2024.
- **[R73]** L. Xiao et al., *Neural Supersampling for Real-time Rendering*, SIGGRAPH 2020.
- **[R75]** D. Craig, *Upgrading PSSR on PlayStation 5 Pro*, SIGGRAPH 2026 Advances.
- **[R76]** T. Zakharko, *Apple Neural Accelerators benchmark*. <https://tzakharko.github.io/apple-neural-accelerators-benchmark/>

## OPT-12 · Autotuning, path tracing, splatting

- **[R77]** J. Ragan-Kelley et al., *Halide*, PLDI 2013; A. Adams et al., *Learning to Optimize Halide with Tree Search and Random Programs*, SIGGRAPH 2019.
- **[R78]** L. Zheng et al., *Ansor: Generating High-Performance Tensor Programs for Deep Learning*, OSDI 2020.
- **[R79]** E. O. Hellsten et al., *BaCO: A Fast and Portable Bayesian Compiler Optimization Framework*, ASPLOS 2023.
- **[R80]** H. Lu, W. Chang, T. Hedstrom, T.-M. Li, *Real-Time Path Guiding Using Bounding Voxel Sampling*, SIGGRAPH 2024. <https://github.com/SuikaSibyl/vxpg>
- **[R81]** Q. Hou et al., *Sort-free Gaussian Splatting via Weighted Sum Rendering*, ICLR 2025. <https://arxiv.org/abs/2410.18931>
- **[R82]** L. Radl et al., *StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering*, SIGGRAPH 2024.
- **[R83]** N. Moenne-Loccoz et al., *3D Gaussian Ray Tracing*, SIGGRAPH Asia 2024.

## OPT-13 · QA e prestazioni

- **[R84]** P. Andersson et al., *FLIP: A Difference Evaluator for Alternating Images*, HPG 2020.
- **[R85]** D. Daly et al., *The Use of Change Point Detection to Identify Software Performance Regressions in a Continuous Integration System*, ICPE 2020.

## OPT-14 · Avvio, dimensioni, energia

- **[R86]** W. Xia et al., *FastCDC: a Fast and Efficient Content-Defined Chunking Approach for Data Deduplication*, USENIX ATC 2016.

---

### Fonti generali da seguire ogni anno

- Corso *Advances in Real-Time Rendering in Games* (SIGGRAPH): <https://advances.realtimerendering.com/>
- High-Performance Graphics (HPG), EGSR, I3D: atti e premi
- Sessioni Metal della WWDC e tech talk Apple sulle GPU
- Elenco dei paper di Ke-Sen Huang: <https://www.realtimerendering.com/kesen/>

## Ricerca SoC e piattaforma — 2026-10-01

Fonti primarie consultate il 2026-10-01. «Documentazione» descrive un contratto
pubblico; «sorgente» descrive quella revisione, non il binario installato;
«paper» e «preprint» riportano risultati degli autori, da riprodurre sul nostro
carico. I link sono fonti, non dipendenze da installare automaticamente.

- **[R87]** Apple, [Tune CPU job scheduling for Apple silicon games](https://developer.apple.com/videos/play/tech-talks/110147/). **Documentazione/talk**: granularità, pool e QoS; riferimento per F23 e OPT-9. Non promette pinning o deadline hard real-time.
- **[R88]** Apple OSS, [Clutch/Edge scheduler](https://github.com/apple-oss-distributions/xnu/blob/main/doc/scheduler/sched_clutch_edge.md). **Sorgente/documento di progetto**: gruppi, raccomandazioni e migrazione fra cluster; base del laboratorio F23.9, da fissare a un commit prima dell'esperimento.
- **[R89]** Apple, [Meet Audio Workgroups](https://developer.apple.com/videos/play/wwdc2020/10224/), e [implementazione libdispatch](https://github.com/apple-oss-distributions/libdispatch/blob/main/src/workgroup.c). **Documentazione/sorgente**: coordinamento dei thread audio; verificare il contratto di ogni tipo di workgroup prima di estenderne l'uso.
- **[R90]** Apple, [Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/). **Documentazione/talk**: command allocator, residency, sparse placement e sincronizzazione esplicita. Consultare anche le firme metal-cpp della build.
- **[R91]** Apple, [Boost your graphics performance with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111431/). **Documentazione/talk**: dynamic caching, cause di limitazione dell'occupancy e compressione; i rapporti pubblicizzati non sono guadagni del renderer Phosphor.
- **[R92]** Apple, [Accelerate your machine learning workloads with the M5 and A19 GPUs](https://developer.apple.com/videos/play/tech-talks/111432/), [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf). **Documentazione**: percorso GPU tensoriale; distinto dal Neural Engine.
- **[R93]** Apple, [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw?language=objc) e [cpuAndNeuralEngine](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine). **API pubbliche**: piano/costi previsti e selezione delle unità ammesse; non equivalgono a un trace dell'esecuzione.
- **[R94]** Apple ML Research, [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers), 2022. **Ricerca degli autori**: layout e riduzione di copie/intermedi; non trasferire limiti o performance del modello ai chip nuovi senza misura.
- **[R95]** Apple OSS, [ARM Scalable Matrix Extension in XNU](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md). **Sorgente/documentazione OS**: stato SME e supporto del sistema; verificare ABI e feature effettive prima dei kernel custom.
- **[R96]** [Hello SME! Generating Fast Matrix Multiplication Kernels Using the Scalable Matrix Extension](https://arxiv.org/abs/2409.18779), SC Workshops 2024. **Paper**: caratterizzazione e generazione di kernel; riferimento OPT-9.12, con packing e shape reali inclusi.
- **[R97]** Apple, [Installing a custom kernel extension](https://developer.apple.com/documentation/apple-silicon/installing-a-custom-kernel-extension). **Documentazione**: vincoli di installazione su Apple Silicon; non documenta una via per rimpiazzare lo scheduler o i driver GPU di macOS.
- **[R98]** C. Augonnet et al., [StarPU: A Unified Platform for Task Scheduling on Heterogeneous Multicore Architectures](https://starpu.gitlabpages.inria.fr/publications.html), e [codelet/features](https://starpu.gitlabpages.inria.fr/features.html). **Paper/progetto degli autori**: ispirazione per costi e implementazioni alternative del job; non si assume un backend Metal/ANE pronto per Phosphor.
- **[R99]** R. D. Blumofe, C. E. Leiserson, [Scheduling Multithreaded Computations by Work Stealing](https://www.cs.cornell.edu/courses/cs612/2006sp/papers/blumofe94.pdf). **Paper**: proprietà per computazioni fully strict; le garanzie richiedono ipotesi che il DAG GPU/CPU può non soddisfare.
- **[R100]** M. Willsey et al., [egg: Fast and Extensible Equality Saturation](https://arxiv.org/abs/2004.03082), POPL 2021. **Paper**: e-graph e riscritture; uso proposto offline con semantica e precondizioni esplicite.
- **[R101]** P. Jain et al., [Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization](https://proceedings.mlsys.org/paper_files/paper/2020/hash/0b816ae8f06f8dd3543dc3d9ef196cab-Abstract.html), MLSys 2020. **Paper**, stesso lavoro di R3 riverificato: ispirazione per il tradeoff memoria/ricalcolo, non un compilatore Metal già disponibile.
- **[R102]** B. Bitterli et al., [Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct lighting](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct), 2020. **Paper**: fondamento ReSTIR, da mantenere corretto nella gestione di pesi e riuso.
- **[R103]** K. Vaidyanathan et al., [Random-Access Neural Compression of Material Textures](https://research.nvidia.com/publication/2023-08_random-access-neural-compression-material-textures), SIGGRAPH 2023. **Paper**, R71 riverificato: confronto proposto con formati classici su GPU Apple.
- **[R104]** E. Hellsten et al., [BaCO: A Fast and Portable Bayesian Compiler Optimization Framework](https://arxiv.org/abs/2212.11142), ASPLOS 2023. **Paper**, R79 riverificato: ricerca su parametri misti e vincoli; validare su scene holdout.
- **[R105]** R. Kumaresan, [Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference](https://arxiv.org/abs/2603.06728), 2026. **Preprint**: usa API ANE private secondo gli autori; candidato per laboratorio F30.7, risultati non riprodotti qui.
- **[R106]** [Apple Neural Engine: Architecture, Programming, and Performance](https://arxiv.org/abs/2606.22283), 2026. **Preprint** di caratterizzazione: separare osservazioni sperimentali, API private e percorso Core ML supportato; risultati non riprodotti qui.
- **[R107]** Jolt Physics, [release notes ufficiali](https://github.com/jrouwe/JoltPhysics/releases). **Upstream**: interfaccia compute con Metal e simulazione hair; fissare versione e verificare quali solver siano effettivamente GPU prima di pianificare il port.
- **[R108]** Apple, [Metal Feature Set Tables](https://developer.apple.com/metal/capabilities/). **Specifiche**: matrice di disponibilità e limiti; verificare la revisione insieme all'SDK, interrogare le capacità a runtime.
- **[R109]** [Demystifying ARM SME to Optimize General Matrix Multiplications](https://arxiv.org/abs/2512.21473), 2025. **Preprint**: candidato di ricerca per tiling e riuso SME; benchmark degli autori non esteso automaticamente ai piccoli batch del motore.
- **[R110]** A. H. Chen et al., [Vertex Block Descent](https://arxiv.org/abs/2403.06321), SIGGRAPH 2024, e [testo degli autori](https://graphics.cs.utah.edu/research/projects/vbd/vbd-siggraph2024.pdf). **Paper**, parte di R65 riverificata: candidato per simulazione, con validazione di stabilità e collisioni separata dal solo throughput.
- **[R111]** Apple, [What’s new in Metal](https://developer.apple.com/metal/whats-new/), consultato 2026-10-01. **Documentazione corrente**: formati tensoriali quantizzati con scale factor, MetalFX neurale su ANE e GPU Neural Accelerators, ingressi per DRS/motion/distortion; disponibilità da verificare nell'SDK e sul device.

## Piattaforma Bevy e riuso — 2026-10-01

Fonti primarie consultate per la progettazione; docs.rs riportava Bevy 0.19.1.
Fissare versione e feature al kickoff, e verificare la compatibilità dei plugin.
Queste fonti non attestano alcuna integrazione già funzionante in Phosphor.

- **[R112]** Bevy, [README bevy_ecs](https://github.com/bevyengine/bevy/blob/main/crates/bevy_ecs/README.md). Uso standalone e semantica ECS; non implica compatibilità di tutti i plugin del motore.
- **[R113]** Bevy, [Plugin](https://docs.rs/bevy/latest/bevy/app/trait.Plugin.html) e [DefaultPlugins](https://docs.rs/bevy/latest/bevy/struct.DefaultPlugins.html). Contratto App, lifecycle e composizione per feature; base dell'host da verificare.
- **[R114]** Bevy, [bevy::ui](https://docs.rs/bevy/latest/bevy/ui/index.html). Componenti/layout della UI di riferimento; la copertura finale richiede prove di interazione e rendering.
- **[R115]** Bevy, [bevy::sprite](https://docs.rs/bevy/latest/bevy/sprite/index.html). Dati/primitivi 2D da censire per la matrice di parità.
- **[R116]** Bevy, [AssetLoader](https://docs.rs/bevy/latest/bevy/asset/trait.AssetLoader.html). Interfaccia di caricamento con tipi e dipendenze da preservare nell'adapter.
- **[R117]** Bevy, [UiRenderPlugin](https://docs.rs/bevy/latest/bevy/ui_render/struct.UiRenderPlugin.html). Integrazione del rendering UI distinta dal layout; non riusabile automaticamente sul backend Metal Phosphor.
- **[R118]** Bevy, [release 0.19](https://bevy.org/news/bevy-0-19/) e [catalogo Bevy Assets](https://bevy.org/assets/). Inventario di funzionalità ed estensioni, non garanzia di qualità o compatibilità universale.
- **[R119]** [bevy-inspector-egui upstream](https://github.com/jakobhellermann/bevy-inspector-egui). Esempio di riuso della reflection per inspector; dipendenze egui/render da verificare.
- **[R120]** [bevy_egui upstream](https://github.com/vladbat00/bevy_egui) e [sorgente dell'integrazione](https://github.com/vladbat00/bevy_egui/blob/main/src/lib.rs). Esempio di plugin con collegamento al renderer Bevy, candidato a integrazione mirata.
- **[R121]** [bevy_ecs_ldtk upstream](https://github.com/Trouv/bevy_ecs_ldtk) e [API LDtk](https://ldtk.io/api/). Loader e percorso tilemap da verificare separatamente.
