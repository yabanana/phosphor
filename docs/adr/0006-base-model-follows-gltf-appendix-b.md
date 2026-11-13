# The base material model follows glTF 2.0 Appendix B

Owner decision, 2026-11-13. The F15 review found that Phosphor's base model
weighted the direct diffuse by the Fresnel of the metallic-blended F0 (about
half the Appendix B value at metallic 0.5) and left the environment/GI diffuse
without dielectric Fresnel weighting (a white dielectric reflected more than it
received). Every KHR material extension is defined relative to Appendix B, so
F15 corrects the base model in the shared BSDF module (dielectric
`mix(diffuse, specular, F(0.04))`, metal `specular · F(baseColor)`,
`mix(dielectric, metal, metallic)`; environment/GI diffuse weighted by the
dielectric directional Fresnel) before any lobe lands.

## Consequences

- visual_check references, the declared model of the independent references
  that mirror the engine BSDF and the affected F13/F14 reference clips are
  re-taken deliberately, with before/after evidence; the F12 diagnostic signal
  `gi_reference_diffuse` stays pure Lambert so the Mitsuba Cornell oracle stays
  valid. Sponza D2 is re-measured.
