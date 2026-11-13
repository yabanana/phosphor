# 16: Layered materials (layer stack) on a terrain fixture

Roadmap: F15.4

**What to build:** Up to four material layers per pixel from an RGBA weight mask with optional endpoint-exact height contrast, parameter-space blending with one BSDF evaluation, surface-gradient normal blending, weights normalised to sum 1, layer presence in the feature class; a procedural terrain patch with four pinned CC0 materials and a cavity coat finish; cost per active-layer count measured.

**Blocked by:** 14

**Status:** ready-for-agent

- [ ] Doctest seam 3: weight normalisation, endpoint exactness, monotonic transitions, no NaN with all-zero weights
- [ ] Each layer at weight 1 renders exactly like the single material (0 pixels, visibility path); weight 0 leaves the others unchanged
- [ ] Transition ramp monotonic; white furnace ≤ 1 + 0.5 % for blended BSDFs; an additive-blend negative control exceeds it; deviation from a BSDF-space mix recorded
- [ ] Resolve GPU time at 1080p per active-layer count (1–4) vs single material, A/B/A, 3 replicas, quiet machine, in `docs/perf-log.md`; binning rule applied to the terrain view
- [ ] CC0 fetch pinned with hashes; Track Studio audit lessons reviewed and recorded
- [ ] Coverage and feature layers stay distinct: a coat finish layer changes coat weight only, never the base F0 of the layer below (Track Studio audit: a chrome finish took the soil colour as F0); height contrast expressed in metres
- [ ] Recorded option, not built unless measured need: sparse top-K (layer id, weight) masks for palettes larger than four (Track Studio found ~95 % zeros in dense weights)
