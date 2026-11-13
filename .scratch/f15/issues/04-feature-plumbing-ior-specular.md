# 04: Feature plumbing proven by ior and specular (resolve and forward)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_ior` and `KHR_materials_specular` render for analytic lights, the sun disk and the environment term in the resolve and forward paths, through the new chain: extension records via SceneStore deltas, feature bits and scene feature-union specialisation of the forward and resolve pipelines, the shared BSDF module and the support report flipping to implemented for those paths. Runtime parameter edits reach the next frame.

**Blocked by:** 01, 02, 03

**Status:** ready-for-agent

- [ ] Extension records reach the GPU only through SceneStore deltas; editing specularFactor/ior on an entity changes the next frame (fixture test)
- [ ] Scenes with an empty feature union keep identical pipelines and images (visual_check 0 px); Sponza D2 GPU p95 within noise (A/B/A)
- [ ] ior and specular grids pass decision-11 direct-light thresholds; ior = 0 special mode and f90 = specular weight covered; furnace bound holds
- [ ] Negative control (specular extension ignored) fails the grid bound
- [ ] Forward and visibility resolve give the same specular-grid image within the existing lit-mode envelope
- [ ] 0 GPU allocations in measured frames; IORTestGrid and SpecularTest captured under the declared Khronos-like rig
- [ ] Extension-record deltas count as material deltas for the GI radiance state and every history keyed on material revision (today `gi_passes.cpp:91` bumps `materialRevision` on base material deltas only); after an edit, indirect/denoised signals recover within the F13.6 recovery criterion
