# 24: F15 phase closure: residue audit, regressions, measurements and handoff

Roadmap: F15.1–F15.4

**What to build:** Severe residue audit (roadmap lines, plan, exit criteria, TODOs), full regression battery, Sponza budget re-measured, Apple9 forced path, documentation and per-ID handoff; roadmap boxes ticked only for verified scope.

**Blocked by:** 15, 18 (if triggered), 20, 21, 22, 23

**Status:** ready-for-agent

- [ ] `ctest`, `metal_syntax_check`, `tools/visual_check.sh` (0 px vs the ticket-03 baseline), F13/F14 corpus, `variant_check`, `archive_check`, `hitch_check` green; API validation clean (shader validation where the F14.4 defect does not apply)
- [ ] Sponza D2 GPU p95 A/B/A against the ticket-03 baseline within the decision-11 band; 0 GPU allocations
- [ ] `--force-family apple9` run of the material bench with features
- [ ] ROADMAP F15.1–F15.4 ticked only where verified; approximations and deferred items listed (caustics, dispersion, multiple-scattering compensation, back-lit thick subsurface if the hybrid is not triggered or not met, non-uniform texture transforms, coated emission of sampled emitters, GI/probe rays hitting glass as opaque)
- [ ] perf-log, opt-log, RENDERER.md and per-ID handoff written; no push or PR without the owner

## Comments

2026-11-14 — orchestrator: consolidated carry list. Keep it current until closure. Each item names its source ticket and record.

Partial criteria (stay unticked):
- 13: the temporal clip with transmission fails F13.6 on the gated smooth glass (p95 0.14-0.17). The cause is refracted-image
  reprojection under the orbit. A per-branch history was tried as a spike and rejected. The refract-axis control is not
  demonstrable, and the rough ROIs are not judgeable (control cost).
- 17 and 18: volumetric subsurface misses the side-lit bound on valid OpenPBR ground truth (gated mean 0.25-0.36) because of
  confinement in near-lossless media; ticket 18 ended in its fallback.
  - The tabulated profile (0.14-0.27) and the V1 dmfp mapping are measured candidates, not adopted.
  - The back-lit view is near black.
  - Later EDGE candidate: an RT volumetric random walk.
- 20: the ticket-17 bound on the chess pieces cannot be built (the reference is texture-free and homogeneous). The scattering
  weight map is not in the pinned subset (weight 1).
- 21: class pipelines are not bit-identical to the all-lobes pipeline (up to 37 ULP, compiler scheduling), so binning stays
  opt-in; the A/B/A binning-vs-union measurement on the chess fixture still needs a quiet machine.
- 22: Khronos side by side is qualitative by design (no IBL, different tonemap). No numeric texture fixtures for coat normal,
  anisotropy direction, iridescence thickness, or subsurface weight/colour textures (the reference refuses textures).

Declared approximations and limits (support table entries):
- 13: one Fresnel-chosen sample per pixel (ADR-0004); at most 4 crossings / 8 events.
- 14: GI shadow rays and reflection secondary lighting through glass have no numeric fixture.
- 15: single-layer map; glass is opaque with `--shadow-cache on`; the froxel fog ignores glass; the receiver bias applies only
  while the map is active; CSM at 2048^2 cannot resolve 2 cm casters (the chess shadow rig uses 8192).
- 19: GI at -N is a DDGI probe lookup only (cache, ReSTIR GI and the legacy resolve path keep E(N)). The iridescence diffuse
  weight is evaluated at N.V (ADR-0006 energy form).
- 22:
  - D4 fix: refracted first opaque hits get the environment term, but F13 reflection estimators do not (asymmetry);
  - with `--gi ddgi` the shadowed wall behind glass stays black (latent, not analysed);
  - transport rays hitting the back of a single-sided opaque plane return black (latent);
  - D5 is the single-bounce reflection of neighbours; D7 is unbiased 1-spp noise;
  - refusals: KHR_materials_unlit and KHR_lights_punctual are unsupported by design; non-uniform KHR_texture_transform is
    deferred.

Found in F15, outside F15 code:
- C22-D3, a pre-existing F13 defect: on silhouette pixels the specular is exactly 0 while the composite has already removed the
  diffuse weight 1 - E(N.V -> 0), so those pixels are dark with RT or SSR reflections.
- GPU safety: large emissive meshes (chess kings/queens, ~75k triangles) hung frame 0 with the clustered DI path. An agent is
  investigating it.
- O7 caveat: a non-archived RT configuration with a cold compiler cache allocates one IFT per frame slot when its final variant
  arrives (ticket 13 record).

Quiet-machine measurements for this ticket:
- Sponza D2 A/B/A band;
- cost of transmission RT, transmissive shadows (RT and the CSM map), the subsurface diffusion pass, the iridescence per light
  (ticket 12) and the GI-back lookup;
- binning vs union on chess.
