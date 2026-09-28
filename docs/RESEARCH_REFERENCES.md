# Bibliografia per le fasi OPT

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
