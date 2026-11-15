# 22: Khronos corpus run and defect report

Roadmap: F15.3

**What to build:** A pinned fetch script brings Tier A/B models (glTF-Sample-Assets 8f49239, hashes and licence records, never committed); the bench renders each under the declared Phosphor rig; a side-by-side page against the Khronos screenshots/goldens documents per-feature defects and the explained Sample Renderer deviations.

**Blocked by:** 19

**Status:** resolved with open items (2026-11-14; 21 of 24 models render, 3 refused with stated reasons; C22-D3 FIXED in e9569ac (section 'F13 residues fixed inside F15'), D4 fixed, D5/D7/A1 named approximations, R2 non-uniform texture transform deferred; numeric texture fixtures of tickets 09/11/12/16 NOT built)

- [x] Fetch script reproducible, verifies hashes, stores licence/attribution, skips Draco/BasisU/WebP variants (`--verify`, offline, unit-tested; the committed manifest records no skipped file: the default variant of all 24 models is plain .gltf/.bin/.png/.jpg)
- [x] Every Tier A model loads or is refused for a stated reason (e.g. DiffuseTransmissionTest requires unlit/lights_punctual) (Tier A: 17 of 19 render, DiffuseTransmissionTest and TextureTransformMultiTest refused; Tier B: 3 render, GlamVelvetSofa refused; ABeautifulGame renders)
- [x] Side-by-side page with per-feature defects; no global parity claim; Sample Renderer deviations cited (page generated under build/, never committed; generator `tools/f15_corpus_page.py`; report `docs/research/2026-11-14-f15-khronos-corpus-report.md`)
- [x] Captures run serially with the runner's GPU guard (lock + process check before every capture, waiting for other cooperating jobs, never killing)

## Verification record (2026-11-14, agent for ticket 22)

Commits (all `F15.3 (ticket 22)`): `5efabce` fetch `--verify` + unit test; `6781925` `--corpus-rig` (rig + bounds framing, doctests); `89870f6` runner and
page generator, exact corner framing; `4d6ecec` targeted archive update; `ea2a7d8` log classification, notes, tests, ctest entries; `a05bb3d` defect report.

Declared rig (`src/testbench/corpus_rig.h`, `--corpus-rig YAW,PITCH` on `--bench 4 --scene FILE`): one white sun of 3 W/m2 from the upper left behind the camera,
constant environment radiance 0.5, exposure 1.0, ACES tonemap, 60 degree vertical field of view framed over the eight world-bounds corners (+12 %), 1024x768
(1600x640 for IridescenceSuzanne), `--rt on --shadows rt --lighting restir --reflections rt --lighting-denoise custom --post --upscaler native --pipeline-sync`,
48 warm-up + 48 measured frames, one settled frame captured (no accumulation). No image-based lighting, so no parity is claimed anywhere.

Commands (corpus at the orchestrator's `build/khronos-corpus`, read-only):
`python3 -I tools/f15_corpus.py --static|--run|--validate|--harvest|--archive-update|--reclassify --corpus DIR --out build/f15-corpus [--wait-busy-minutes N]`;
`python3 -I tools/f15_corpus_page.py --run build/f15-corpus --corpus DIR --page build/f15-corpus/page --report-md docs/research/2026-11-14-f15-khronos-corpus-report.md`;
`python3 -I tools/fetch_khronos_corpus.py --verify DIR` (24 models match the manifest, exit 0).

Results (final run, `build/f15-corpus/summary.json`): 21 rendered, 3 refused, 0 unexpected failures. Refusals: DiffuseTransmissionTest (requires KHR_materials_unlit and
KHR_lights_punctual: by design, decision 02), TextureTransformMultiTest and GlamVelvetSofa (a non-uniform KHR_texture_transform: the loader bakes only a uniform one
into UV0; open, C22-R2/R2b). Archive: 110 misses over 10 models before the targeted update (`--harvest-pipelines` per model, `filter_pipeline_harvest.py --constant 22 --constant 23
--skip-present`, `merge_pipeline_harvests.py`: +82 compute, +13 render, +4 tile descriptors, purely additive (`git diff --minimal`: 3194 insertions, 0 deletions), constant 23
added nothing); after it 0 misses and 0 fallback draws in all 24 captures, `pipelines script` doctests pass. O7: 0 GPU allocations in the measured frames of all 21 rendered models.
Validation (API + shader, archive off): 21 of 21 rendered captures exit 0 with no validation failure; the only messages are the known baseline notices (96 redundant
setDepthStencilState + 96 setViewport calls per capture, identical to the ticket 15 record) and loader warnings (BLEND rendered opaque, volume without transmission).
Unit tests: doctest 889+ (corpus rig 4 cases, `--corpus-rig` parsing), python `fetch_khronos_corpus_python` (9), `f15_corpus_python` (13), `khronos_corpus_verify`
(skipped with a message when no fetched corpus exists); ctest 24 / 24 passed (with `-DKHRONOS_CORPUS_DIR` pointing at the fetched corpus; 23 / 23 plus the skip message without it).

Defects (report has the per-model root-cause hypotheses): C22-D3 dark pixels on the outermost silhouette row of smooth spheres (present with reflections rt and ssr, absent
with reflections off: diagnostic runs `build/f15-corpus-diag`; open, reflection composite/denoiser at geometry edges, F13); C22-D4 dark grain where glass overlaps single-sided label
planes in TransmissionRoughnessTest (NOT missing GI: unchanged with `--gi ddgi`; hypothesis: transmitted ray hits the back of a culled single-sided plane; open);
C22-D5 speckle on near-mirror strongly anisotropic spheres in one settled frame (open); C22-D7 grain at high IOR in one settled frame (open); C22-A1 single-bounce reflections
make mirror-in-mirror dark (named approximation); C22-R1/R2/R2b refusals. Nothing was fixed in the engine: the candidates are shader/denoiser changes of F13/F9 that this
ticket cannot verify against an independent reference.

Not done: numeric texture fixtures (coat normal, anisotropy direction, iridescence thickness/factor, subsurface weight/colour): the reference refuses texture words and the
corpus exercises those textures only qualitatively; no per-slot texture transform; no Render Fidelity goldens (not approved). Deviation from the brief: the corpus framing uses
a fixed front or three-quarter view per model (`MODELS` in the runner), not the Khronos cameras. Process note: one probe run with the first framing build started while another
agent's phosphor was running (before the runner's guard existed); it lasted a few seconds, finished with exit 0 and is not used as evidence.

## Defect triage follow-up (2026-11-14, agent for ticket 22 defects)

- **C22-D4 fixed** (`reflections.metal`: the refract walk lights its first opaque hit with the declared environment diffuse term when transported GI is off).
  The back-face hypothesis was disproved (double-sided labels: 2 of 786432 pixels; a debug counter found no back-face hit); the dark squares were the wall
  behind the glass in the sun shadow of the label panel, lit by the sun only. Check: `tools/f15_shadowed_hit_check.py` (ratio through glass / direct 1.001 shadowed,
  1.001 lit; control binary without the fix 0.000 / 0.926, fails as required), ctest `f15_shadowed_hit_check_python`. 17 transmission f15_check keys green
  (roughness-ior numbers identical to the pre-fix run: worst mean 0.0497, p95 0.1349), `visual_check` 0 px on 8 benches, shader + API validation clean on 6
  transmissive corpus models, O7 0 allocations, corpus rerun: ClearCoatTest and IORTestGrid 0 px against the pre-fix binary, transmissive models changed as intended.
  Latent, reported: reflection estimators (F13) keep the sun-only lighting of a shadowed hit with GI off; a transport ray hitting the back of a single-sided opaque
  plane returns black (not triggered by the corpus, no fixture).
- **C22-D3 open (F13)**: pre-existing code path (reproduced on extension-free spheres); raw and filtered specular are exactly 0 at the outermost silhouette pixel,
  the composite already removed the diffuse share: not fixed (F13 composite/kernels, F13 exact captures not available), details in the report.
- **C22-D5 / C22-D7 named approximations**: D7 is unbiased noise converging as 1/sqrt(N) (ratio 7.9 for 64 frames); D5 is converged deterministic structure
  (reflections of the neighbours), not noise. Numbers in the report.

## F13 residues fixed inside F15 (2026-11-14, owner decision; integrated at 6b6bf04)

- **C22-D3 fixed** (e9569ac): shading N.V <= 0 on the silhouette ring has no specular estimate; the composite now keeps the full diffuse
  weight there. Fixture `tools/f15_residue_check.py --case grazing` (control 11/11 ring pixels below 70 %, fixed 0/0). Corpus: SpecularTest
  ring count 1165 -> 937 (includes label/background edges); IORTestGrid stays at 67 (transmissive sphere edges, N.V > 0, open: ticket 24).
- **Latent cases fixed**: reflection estimators and probe capture get the environment term (bf0799f); single-sided back faces no longer end
  transport rays (26fe229); DDGI behind glass (4422a13). Corpus rerun on ten models: all render, 0 archive misses, validation clean; the
  changed-pixel counts per model are in docs/implementation/F13-F14-CLOSURE-2026-11-12.md.

2026-11-15 — closure audit (ticket 24):

- 1 fetch script: MET (--verify 24 models, 9 unit tests). 2 every Tier A model loads or is refused: MET (17/19 render; DiffuseTransmissionTest, TextureTransformMultiTest, GlamVelvetSofa refused with reasons). Ticked.
- 3 side-by-side page: MET qualitative (page under build/, report docs/research/2026-11-14-f15-khronos-corpus-report.md; own cameras, no IBL, no parity claim). 4 serial captures with GPU guard: MET. Ticked.
Open: numeric texture fixtures (coat normal, anisotropy direction, iridescence thickness/factor, subsurface weight/colour; sheen textures) because the reference refuses textures; per-slot texture transform; Render Fidelity goldens not approved; A1 (single-bounce mirror-in-mirror dark) not in 24.

2026-11-15 — per-slot KHR_texture_transform (closure agent texxform; owner decision: implement the non-uniform transform now; details in ticket 04, same date):

- **C22-R2 (TextureTransformMultiTest) and C22-R2b (GlamVelvetSofa) fixed: both now load and render** (`build/f15-corpus`, serial, GPU lock;
  `tools/f15_corpus.py --run --only TextureTransformMultiTest --only GlamVelvetSofa --only TextureTransformTest`): status rendered, 0 fallback draws, 0 failures,
  O7 0 GPU allocations in the measured frames, TextureTransformTest (the uniform baked path) still renders.  API + shader validation (`--validate`, archive off): 3 of 3
  exit 0, no validation failure, only the known baseline notices (96 redundant setDepthStencilState + 96 setViewport per capture).
- What renders: **TextureTransformMultiTest** - all nine rows (BaseColor, Emission, Normal, Metal/Rough, Occlusion, Unlit, Clearcoat factor / roughness / normal) draw the
  same check-mark pattern in the UV 0, UV 1 and Sample columns (the README's criterion).  Its 28 textured materials have exactly ONE textured slot each (a UNIFORM transform,
  baked as before); what was blocking it was texCoord 1 (the UV 1 column): the loader now copies TEXCOORD_1 into UV0 for a material whose textures sample set 1.  The model therefore
  does not exercise the per-slot GPU path.  The "Unlit" row still renders lit: KHR_materials_unlit is only USED here (3 materials), reported unsupported and ignored (agent loader-misc owns
  it; nothing else blocks this model).  **GlamVelvetSofa** - renders (blue velvet, default variant): the fabric normal texture carries scale 5 while the base colour does not (PerSlot status,
  the per-slot GPU path with a rotated tangent frame), specular + sheen on the fabric.  KHR_materials_variants and KHR_lights_punctual are used, not required: the default
  material renders, the asset's lights are not used (the rig lights it); that does not block the render (notes C22-R4, named approximation).
- Numeric fixture (the corpus is qualitative): `tools/f15_texture_transform_check.py --run` (GPU lock; ctest `f15_texture_transform_python` runs its CPU self-test): a synthetic glTF with five
  quads, every material carries four DIFFERENT transforms on four slots (baseColor, emissive, occlusion, normal; thresholds fixed in the docstring before any comparison) and each quad
  measures one slot (baseColor / emissive: 3-bit colour code per cell; occlusion: luminance bit; normal: bright / dark class of N'.L under the rig sun, which includes the rotation of the tangent
  frame; alpha: a MASK material's cut-out through the base colour transform) against a CPU texel lookup with the README formula.  Result: agreement 1.0 on every slot (valid pixels
  3265 / 3192 / 3254 / 2916 / 3278; threshold 0.995); negative controls (the same pixels against ANOTHER slot's transform = the uniform-bake result, against the identity, normal: frame not rotated,
  frame rotated the wrong way) 0.09 - 0.58 (must stay below 0.90, otherwise exit 3): normal frame-not-rotated 0.478, wrong way 0.0.  The same run under API + shader validation (archive off): 1.0, clean.
- Featureless scenes unchanged: `f15_exact.sh` 5 cases (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky) PFM byte-identical (cmp) to build/litefix/exact-new;
  `visual_check` 8 benches 0 px against build/reference-f15-appB.  One finding on the way: the first version (restructured shadeSurface expressions) changed benches 1 / 2 / 6 by 9 / 1 / 9 px of delta 1
  with the constant compiled out (a 3ef8f41 binary: 0 px); the fix keeps every plain statement verbatim and adds the transform as a guarded alternative (commit fc94500).  Rule for later edits of shadeSurface:
  never restructure an existing expression, add a guarded branch.
- Sites NOT covered (named approximation `texture_transform_secondary_sites`): a slot with a per-slot transform samples UV0 untransformed at reflection / GI / probe-capture hits
  (reflections.metal, gi_common.h, reflection_probes.metal, restir_common.h emitter UVs), at the RT any-hit alpha test and transmission tint (rt_intersections.metal rtAlphaMaskedTest and the tint at
  shadows.metal), and in the CSM / static-cache shadow-map alpha tests (shadow_depth_fragment, shadow_cache_depth_fragment).  Uniform transforms are baked into the vertex UVs, so none of those sites is
  wrong for them.  Covered: forward, meshlet forward, visibility resolve (generic, adaptive, class bins), indexed overflow fallback, DI guide kernel and receivers, the volumetric-subsurface resolve, the visibility
  alpha raster (visibility_alpha(_lit)_tt_fs) and the overflow receiver prepass (forward_shadow_receiver_tt_fs).  What covering the rest takes: the extension buffer bound at each of those kernels
  (rt_alpha_generic already receives it at buffer 7), the `On` helpers of shaders/texture_transform.h and one separate entry point per alpha function (metal-tt cannot lower an unspecialised descriptor
  whose function references a function constant: found when the archive step failed with "cannot lower module with unresolved function constants").
- Layer stacks (F15.4): a layer sub-material's own transforms and the mask / height UV are not applied (a transform or UV set on a layer texture is Unsupported); the layer tiling `uvScale` is the only per-layer UV control.
- Archive: targeted update (`.scratch/f15/harvest/texxform-1.json` KEPT, constant 26 plus the three `*_tt_fs` functions, `--skip-present`, from the fixture and GlamVelvetSofa harvests; TextureTransformMultiTest adds nothing,
  its transforms are baked): +12 compute, +1 mesh, +7 render descriptors, purely additive (`diff --minimal`: 0 deleted lines, 813 added).
- Still open for the orchestrator: regenerate the page and `docs/research/2026-11-14-f15-khronos-corpus-report.md` from a FULL corpus run (the notes json carries the new status of C22-R2 / R2b, new C22-R3 unlit, C22-R4
  variants / lights; this agent ran only three models and did not edit the generated report), and the timings below.
- Timing for a quiet machine (not measured here: other agents were running): A/B/A of the textured fixture with `--no-gpu-timing` off and `--gpu-timing-serial`: `tools/f15_texture_transform_check.py` scene with the constant
  (default) against the same scene with every transform removed (unused: remove the `KHR_texture_transform` blocks from the generated glTF: a copy of build_gltf without them) - the per-pixel cost of one extra 2x3 matrix per slot
  (about 40 flops per slot sampled); Sponza D2 `--frames` A/B against 3ef8f41 (bit-identical output; the shader code of the generic and class pipelines differs only by dead branches, expected equal within noise); GlamVelvetSofa
  at 1080p with and without the constant (same scene, `withTextureTransform` skipped) for the real cost of the transform on a textured model.

## F15 closure, loader-misc (2026-11-15): DiffuseTransmissionTest renders; the redundant-call notices are gone

- **DiffuseTransmissionTest** (`C22-R1`) was refused because it REQUIRES KHR_materials_unlit and KHR_lights_punctual.  Both are rendered now (ticket 04 section, commit e8b6399): 22 of 24 models render, 2 are refused
  (TextureTransformMultiTest and GlamVelvetSofa: non-uniform texture transform, owner "texxform" agent).  Capture `build/f15-corpus-dt/run` (declared rig, 1024x768): exit 0, `status rendered`, the loader prints the known
  `BLEND rendered opaque` warning only, 0 archive misses after the targeted update (19 misses of 15 pipelines before: the unlit + thin-subsurface union had never been harvested; `.scratch/f15/harvest/loader-misc-2.json`:
  +17 compute, +3 render, +1 tile descriptors, constant 22), 0 fallback draws, 0 GPU allocations; under API + shader validation (`validate`, archive off): 0 validation messages and 0 notices.  Side by side: the factor row
  goes from the opaque teal at factor 0 to the transmitted white at factor 1, the colour factor row to red, the stripe and logo textures modulate the transmitted lobe (qualitative, own camera, declared rig sun: the file's
  directional light is not imported under the rig).  Notes, page text and the research report are updated (tools/f15_corpus_notes.json, tools/f15_corpus_page.py, docs/research/2026-11-14-f15-khronos-corpus-report.md).
- **KHR_materials_unlit numerically** (tools/f15_unlit_lights_check.py, `build/f15-unlit-lights2`): a flat and a 55-degree tilted unlit panel show their base colour with error 0.00000 (limit 0.002, half-float ulps), the lit
  panels of the same colours differ by 0.13 / 0.20; the lit twin of the whole file fails the unlit predicate (error 0.17).  GI (tools/f15_unlit_gi_check.py, DDGI, `build/f15-unlit-gi`): an unlit panel lights nothing (wall patch delta
  +0.00000, noise floor 0.00000) while its emissive twin lights the wall by +0.327; shader validation of the `gi_common.h` edit: 0 messages.  KHR_lights_punctual numerically: directional 0.743 vs 0.764 (Lambert), node turned 180 degrees
  0.000, grazing 0.003 (disk bound 0.0036), point 0.956 / 0.581 vs 0.978 / 0.600, spot on axis 1.100 vs 1.132, outside the cone 0.000 (all within the declared 6 %; the spot between the cones follows the engine's smoothstep,
  angular factor 0.60 measured vs 0.616 smoothstep and 0.578 linear, the informative formula of the specification would give 0.334: declared in ticket 04).
- **The `Redundant call to setDepthStencilState / setViewport` baseline of this ticket (96 + 96 per capture, "known notices") is gone** (commit 65f2f10): the cause was the fused "Visibility opaque B" + "Visibility alpha B"
  encoder repeating the same two calls; the corpus runner's KNOWN_NOTICE family stays for older binaries.  DiffuseTransmissionTest is the first corpus capture with 0 notices; the other models need the next full corpus run
  (not repeated here: the orchestrator owns the final corpus run).
