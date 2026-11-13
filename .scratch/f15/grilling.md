# F15 grilling record (self-answered)

The owner asked for the grilling to be answered from code, documents and
research, with only owner-level choices put to the owner in one question.
Each entry: question, recommended answer, evidence. "Owner" marks the
entries that went to the owner.

## Round 1 — facts from the code (2026-11-13, base `e62bb9d`)

**Q1. Where is the material model evaluated today?**
➡️ In four independent copies of the same GGX + Lambert model:
`shadeSurface`/`evaluateLight` (`shaders/material_shading.h`, forward and
visibility resolve, analytic lights + ambient), `diBRDFWeighted`
(`shaders/restir_common.h`, ReSTIR/clustered DI on `GPUDISurface` guides),
`reflectionSpecularBRDF`/`reflectionDirection` (`shaders/reflection_common.h`,
RT/SSR/probe reflections), `giMaterial` (`shaders/gi_common.h`, secondary
hits, diffuse reduction) plus the CPU references (`src/renderer/rt_reference.cpp`,
`tools/f13_reference.py`, independent by design). The composition in the
visibility path is residual + DI + GI·albedo(1−metallic)/π + specular
(`reflection_composite`).
Decision: a prefactor ticket consolidates the three GPU copies of the primary
BSDF into one MSL module before any lobe lands; the Python reference stays an
independent implementation written from the specification.

**Q2. Can extension parameters go into `GPUMaterial`?**
➡️ No. `GPUMaterial` is 80 bytes and equals the 80-byte payload of
`GPUDeltaRecord` (`gpu_types.h:204`), the only path materials take to the GPU
(`SceneStore` deltas, F5 rule). Growing it grows every material/instance delta
and every RT/GI/cull read, including Sponza. Decision: the base record stays
frozen; extension parameters live in a separate buffer of 80-byte records
(reusing `GPUDeltaRecord`) referenced from the base record's padding, with
feature bits in `flags`. ADR candidate.

**Q3. Which render path is the product path for the new lobes?**
➡️ Visibility buffer + F13 lighting (`--render-path visibility`); forward is
still the default path of the legacy benches (`launch_options.h:416`). Both
call `shadeSurface`, so the analytic-light and environment terms reach both;
DI, GI and reflections exist only in the visibility path. The post-F8 review
states that the product HDR path (visibility + compute resolve) uses analytic
derivatives and is not affected by the forward helper-lane divergence.

**Q4. Does F15.2 need OPT-2.0 (analytic derivatives in forward) first?**
➡️ No. Generated material permutations specialise the visibility resolve,
whose derivatives are analytic, so generic-vs-permutation equivalence can be
checked exactly there. Forward keeps the existing axes plus a scene-level
feature mask; its checks use the lit-mode contract of `variant_check.sh`
(≤1 LSB, ≤100 px) or untextured fixtures. OPT-2.0 stays an unselected
catalog item. Evidence: `docs/research/2026-10-04-post-f8-review.md` §1.

**Q5. How do variants work today, and what must the generator add?**
➡️ `shaders/variants.def` declares axes; `tools/variant_gen` emits the full
mixed-radix product (42 forward variants); the resolve has five classes
(`FC_RESOLVE_CLASS`: normal-map × emissive bits, plus the generic reference
4) chosen per 8×8 tile by `visibility_classify`; the archive holds 296 compute,
170 render, 142 mesh and 3 tile pipeline descriptors. A full product over six
lobe bits would multiply the forward table by 64 (2688). The generator must
therefore enumerate only the feature combinations that material descriptions
actually contain, not the product ("Non esplorare prodotto cartesiano di
varianti senza consumer", plan).

**Q6. How do material edits reach the renderer (Track Studio edits parameters)?**
➡️ A `MaterialComponent` change marks the entity; `SceneStore::sync` re-sends
the persistent per-entity material as a delta (`scene_store.h:42`). Extension
records must follow the same path so a parameter edit changes the next frame
without a reload; a fixture test edits a parameter and checks the image.

**Q7. What does the loader do with extensions today?**
➡️ Nothing: tinygltf parses them into `material.extensions` but
`GltfLoader::processMaterial` reads only core PBR fields and silently drops
the rest (only `alphaMode == BLEND` logs a warning). The F15.1 acceptance
"feature non implementata segnalata dal loader" needs a structured support
report (per material: feature, status implemented/approximated/unsupported)
in the log and in the benchmark report JSON.

**Q8. How are secondary hits shaded, and how does that constrain transmission?**
➡️ `reflection_rt` samples a GGX direction from the guide, traces it, shades
the hit with direct lights, sampled emitters and GI irradiance
(`reflectionSecondary`), weights by BRDF·cos/pdf and falls back to probes or
the environment on a miss. A refracted direction is the same transport with a
different sampling routine; a closed volume needs a bounded number of
interface crossings and Beer–Lambert attenuation along the inside segment.

**Q9. Is there an independent reference toolchain to reuse for F15.3?**
➡️ Yes: `exportOfflineReference` snapshots, linear captures
(`--capture-linear`), and Python references that use Mitsuba 3.9.1 for ray
casting with their own BSDF (`tools/f13_reference.py`); Mitsuba 3.9.1 is in
the local `uv` cache. Mitsuba's own rough dielectric and homogeneous media
inside a dielectric boundary give independent ground truth for transmission,
volume attenuation and volumetric subsurface scattering.

**Q10. What fixes the regression budget?**
➡️ Sponza contains no extension materials, so with specialisation its
pipelines and images must stay identical: `tools/visual_check.sh` 0 pixels and
the D2 gate (Sponza native 1080p, every feature, GPU p95 ≤ 16.67 ms; last
measured 15.37–15.48 ms after `4505ee7`) within noise, A/B/A on a quiet machine.

## Round 2 — owner-level choices and review pass (2026-11-13)

- Owner (AskUserQuestion): transmission = RT in F15, raster in F16
  (ADR-0004); subsurface = OpenPBR semantics + draft KHR_materials_scatter
  (ADR-0005).
- Owner direction: Phosphor is a separate product; F15 considers only Phosphor;
  Track Studio is paused (ADR-0001 addendum).
- Review pass: the F13 denoiser does not demodulate by albedo (roughness
  edge-stopping on BRDF-weighted radiance) → the lobe guide supplies the
  effective transport roughness; ticket 04 split (plumbing / lobe guide) to fit
  one context each; Mitsuba 3.9.1 verified offline; glTF §3.2 rule verified;
  `GPUMaterial` padding unused; kernel buffer slots available (max used 19).

## Round 3 — independent review and owner decisions (2026-11-13)

- Independent adversarial review: `.scratch/f15/review-2026-11-13.md`
  (2 blockers, 9 majors, 8 minors), each verified in code before acting.
- Owner (AskUserQuestion): base model corrected to glTF Appendix B in F15
  (ADR-0006); transmissive shadows attenuated and coloured in F15 (ADR-0004
  amendment, supersedes "opaque").
- Tickets restructured to 24: test-first Appendix B harness before the base
  fix, lobe guide split into producers/DI/transport+denoiser with all layouts
  frozen early, RT and CSM transmissive-shadow tickets, a conditional hybrid
  subsurface ticket with its own bound and fallback, binning decision
  provisional on chess and final with the terrain.
