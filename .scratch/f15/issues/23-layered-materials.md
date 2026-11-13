# 23: Layered materials (layer stack) on a terrain fixture

Roadmap: F15.4

**What to build:** Up to four layers per pixel from an RGBA weight mask with optional endpoint-exact height contrast (metres), parameter-space blending with one BSDF evaluation, surface-gradient normals, weights normalised to sum 1, layer presence in the feature class; a procedural terrain patch with four pinned CC0 materials and a cavity coat finish; final binning decision on chess + terrain.

**Blocked by:** 21

**Status:** ready-for-agent

- [ ] Doctest seam 3: weight normalisation, endpoint exactness, monotonic transitions, no NaN with all-zero weights
- [ ] Each layer at weight 1 renders exactly like the single material (0 pixels, visibility path); weight 0 leaves the others unchanged
- [ ] Coverage and feature layers distinct: a coat finish changes coat weight only, never the base F0 below
- [ ] Transition ramp monotonic; furnace ≤ 1 + 0.5 % for blended BSDFs; additive-blend negative control exceeds it; deviation from a BSDF-space mix recorded
- [ ] Resolve GPU time at 1080p per active-layer count (1–4) vs single material, A/B/A, 3 replicas, quiet machine; final binning decision (≥ 5 % rule on chess + terrain) in opt-log
- [ ] CC0 fetch pinned with hashes; sparse top-K masks recorded as an option only
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
