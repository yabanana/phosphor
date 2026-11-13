# 12: Subsurface feature, volumetric mode (screen-space diffusion)

Roadmap: F15.1, F15.3

**What to build:** Volumetric subsurface renders per ADR-0005: the resolve and DI write a diffuse-irradiance target for subsurface pixels (DI diffuse split only in extended scenes), a Burley normalised-diffusion pass in screen space (21/55 samples, metres→pixels radius, sub-pixel skip, same-material and depth test) runs after diffuse denoising, and the result is recombined with the specular terms. The draft `KHR_materials_scatter` volumetric mode maps through its Kulla–Conty albedo inversion.

**Blocked by:** 11

**Status:** ready-for-agent

- [ ] Side-lit and back-lit volumetric fixtures compared with a Mitsuba 3.9.1 random-walk ground truth (homogeneous medium inside an ior 1.5 boundary, `volpath`) on the piece mask: mean ≤ 10 %, p95 ≤ 25 %
- [ ] Decision rule applied: if the bound is missed, the hybrid RT single-scattering variant is implemented within this ticket (timeboxed ≤ 2 weeks) and re-measured; otherwise it is recorded as a candidate
- [ ] Radius sweep monotonic; sub-pixel radius leaves the image equal to the subsurface-off image within FP noise
- [ ] Negative control (diffusion off) fails the bound
- [ ] No change to scenes without subsurface (visual_check 0 px); cost of the diffusion pass measured
