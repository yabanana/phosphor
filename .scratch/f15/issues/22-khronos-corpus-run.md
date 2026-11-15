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

## C22-A1 multi-bounce mirror reflections (2026-11-15, agent reflbounce, branch f15c-reflbounce-b)

- **Change** (`shaders/reflections.metal`, `--reflection-bounces 0..3`): the RT reflection kernel follows up to 3 extra ideal-mirror bounces while the secondary
  hit is a metal-like mirror (roughness <= 0.1, max F0 >= 0.5, Fresnel >= 0.1): throughput *= Schlick F, shading of every hit through the unchanged
  `reflectionSecondary`, miss -> probes/environment. Counters: `--debug-lighting` prints extra traces / chains ended by the bound. Payload stays `RtPayloadLite`.
  Shared files touched (additive): `src/renderer/gpu_types.h` (`REFLECTION_BOUNCE_SHIFT/MASK`), `src/core/launch_options.{h,cpp}`, `tests/CMakeLists.txt`
  (python test), `tests/test_launch_options.cpp`. No pipeline descriptor changed (0 archive misses): no archive update, no KEPT file.
- **Fixtures** (`tools/f15_reflection_bounce_model.py`, `tools/f15_reflection_bounce_check.py`, `tools/test_f15_reflection_bounce_check.py`, ctest
  `f15_reflection_bounce_check_python`): corridor (two facing mirrors) and ball + mirror wall turned 65 degrees, 640x360, classes = number of mirror events
  from an independent analytic model (Schlick F0 0.8, environment 0.5). Stable pixels per class (7x7 window class-constant, sun glint masked): corridor
  29084 / 4788 / 2164 / 1848 (classes 1-4), ball 54562 / 12845 / 2219 (classes 1-3); gate >= 400. Gates fixed before comparing: mean relative error per class
  <= 5 % (floor 5 % of the class mean), background within 5 % of 0.5. With `--reflection-bounces 3`: corridor 0.20 / 0.32 / 0.39 / 0.70 %, ball 0.63 / 2.43 /
  4.7 % (class 3 is close to the gate: sphere tessellation, 96x48 smooth normals vs the analytic sphere), background 0.5000. Negative control
  `--reflection-bounces 0`: classes >= 2 error 98-100 % (dark), class 1 unchanged (0.21 % / 0.63 %). Counters: corridor 208037 extra traces (2665 chains ended by the
  bound), ball 227633 (3202). Re-run reproduces the numbers exactly.
- **Featureless scenes**: the first criterion (no F0 condition) made Sponza D2 trigger ~70 extra traces and change 828 of 921600 linear pixels (max 0.0031, median
  6e-9; PNG 0 px). With max F0 >= 0.5 (metal-like mirrors only; smooth dielectrics keep the single bounce) zero extra traces: the 5 `f15_exact.sh` cases are
  byte-identical (cmp) to `build/litefix/exact-new`, visual_check 8/8 0 px (BENCH8_ARGS=--instances 100000), `--reflection-bounces 0` byte-identical as well.
- **F13/F14 reflection subset** (20 cases, binary with bounces 3, `--references` re-bound to the subset manifest hash): control 11/11, functional 3/3,
  independent-clip 4/6. Two independent clips flip against the t03 verdicts: `roughness-custom` and `metalfx-fallback` (spatial_p95 false, frame-0 linear RMSE
  0.104 vs 0.071); with `--reflection-bounces 0` the same case passes again (0.0712 = t03). Cause: the roughness scene has five gold spheres at roughness 0.04-0.9;
  the two smoothest now show the neighbour spheres through the ideal-mirror chain (431 extra traces, 15 bound-ended per frame), while the declared F13
  reference (`tools/f13_reference.py`, `secondary_specular`) models no specular continuation (a secondary vertex only light-samples its specular lobe). The engine is
  closer to the physics, the clip is stale. Not fixed here (re-rendering the roughness / metalfx-fallback clips with a continuation model is a reference change
  with battery pacing; thresholds untouched).
- **Default**: `--reflection-bounces` defaults to 0 (byte-identical to the F15 tip, all F13/F14 verdicts kept) until the reference models the continuation; ticking
  A1 as resolved needs `--reflection-bounces 3` as the product default plus that clip update (owner decision). Quiet D2 A/B (bounces 3 vs 0): see the
  agent report; Sponza D2 itself triggers no extra trace, so the expected cost is the unused-branch cost of the kernel only.

### C22-A1 follow-up: reference updated, default 3 (2026-11-15, orchestrator decision)

- **Reference** (`tools/f13_reference.py`, additive: `Context(mirror_bounces=0)`, `mirror_hit`, `mirror_chain`, `tail(skip_specular, mirror_continuation)`, CLI `--mirror-bounces N`,
  also in `tools/f13_reference_clip.py render`): from the physics of a smooth surface, not from the engine kernel. The first hit of the camera's specular lobe continues along
  its ideal mirror direction when its roughness <= 0.1 (any f0, dielectric or metal: the engine's F0 restriction is measured against it), weighted by its Schlick Fresnel, up to
  N further hits, each lit and continued with its diffuse lobe like the first hit. Light accounting (D5): the environment reaches the camera through the continuation, emitters
  (sun disk, panels) through the mirror hit's regularised light connection; the first version counted the sun through both (seed NRMSE 0.51 at 256 spp vs 0.094, fireflies up to 880): fixed.
  Unit test `tools/test_f13_reference_mirror.py` (ctest `f13_reference_mirror_python`, needs Mitsuba through uv, skipped otherwise): two facing mirrors against the analytic
  model of tools/f15_reflection_bounce_model.py, 160x90, 8 spp: bounces 3 classes 1-3 error 0.24 / 0.16 / 0.22 %, the 1.0x8.0 engine corridor classes 1-4 within 0.3 % (ref
  0.4175/0.3679/0.2856/0.2205 vs model 0.4178/0.3678/0.2859/0.2205); negative control bounces 0: classes 2 and 3 error > 0.5 (fails as required); bounces 1 reaches class 2 only.
  `f13_reference.py validate` 27/27 unchanged.
- **Clip**: `roughness` re-rendered with `--mirror-bounces 3` (build/refclips/roughness-m3 in the agent worktree, 4096 spp, seed NRMSE 0.0131 < 0.02, 38 min with 4 threads).
  Same-seed 64 spp, bounces 3 vs 0, relative rmse of the frame: roughness 4.3 %, mirror 0.09 %, moving-light 0.09 %, disocclusion 0.05 %, rough-emitter 0 (0 pixels differ). The
  mirror / moving-light / disocclusion clips are therefore NOT re-rendered here (the rough floor lobe hits the mirror plane in ~0.7 % of pixels; moving-light needs ~5 h at 12 threads):
  their verdicts are insensitive at 0.1 %, the orchestrator's final re-render after the merge covers them.
- **F13/F14 reflection subset** (20 cases, default-3 binary, API validation, the runner's own gates): control 11/11, functional 3/3, independent-clip **6/6**. Independent-clip
  linear RMSE p95, tip (t03) -> bounces 3 with the old clip -> bounces 3 with the re-rendered clip: roughness-custom 0.0303 -> 0.1023 (fail) -> 0.0394; metalfx-fallback 0.0303 -> 0.1023 -> 0.0393;
  roughness-raw 0.1762 / 0.2513 / 0.2262 and roughness-rt8-control 0.1048 / 0.1898 / 0.1423 (non-gating controls, fail by design at every stage); mirror-custom 0.0184, mirror-raw 0.1068,
  mirror-rt8 0.0398, moving-light 0.0515 / 0.1060 / 0.0421, disocclusion 0.0568 / 0.1103 / 0.0431, rough-emitter 0.0098 / 0.1768 / 0.1768 unchanged to 4 decimals in all three columns.
  The gating roughness number rose 0.0303 -> 0.0394 against the continuation reference: the engine's chain still differs from the reference in the glossy gold sphere pair
  (regularised sun glint at the continuing hit, probe environment vs exact sky); within the existing threshold, not loosened.
- **Default 3**: `launch_options.h` and `tests/test_launch_options.cpp`; `f15_exact.sh` 5 cases cmp-identical to build/litefix/exact-new, visual_check all benches 0 px, fixtures unchanged
  (corridor 0.20 / 0.32 / 0.39 / 0.70 %, ball 0.63 / 2.43 / 4.7 %, control 98-100 %).
- **Scope F0 >= 0.5 (exclusion b of the plan)**: dielectric and low-F0 secondary hits keep the single bounce. Dielectric numbers from the reference (`mirror_hit` models every smooth
  surface): a corridor of two facing smooth dielectrics (F0 0.04, roughness 0.04, sky 0.5) gains 1.95 % mean radiance over the covered pixels from the continuation, up to 27 % at the
  most grazing pixels (470 of 1884 covered pixels gain > 5 %). The subset scenes contain no smooth dielectric secondary hit, so the clips are not affected. Sponza D2: with the criterion
  without the F0 floor 828 of 921600 linear pixels changed (max 0.0031) through its smooth dielectric texels; with F0 >= 0.5 zero extra traces.
- Quiet D2 A/B (bounces 3 vs 0): Sponza D2 triggers no extra trace, so expect the unused-branch cost only; use the commands in the agent report.

2026-11-15 (second round) — per-slot KHR_texture_transform at the secondary sites (orchestrator review: the declared gaps are closed):

- Sites now applying a slot's transform (commit message of this round lists the files): RT any-hit alpha test and the transmittance tint (`rt_intersections.metal`; the lite
  entry gained buffer 7, which every IFT already bound), CSM alpha (`shadow_depth_fragment`) and static-cache alpha (`shadow_cache_depth_fragment`) plus the transmittance tint
  (`shadows.metal`; the extension buffer is bound at SB_EXTENSIONS 22 for every raster draw), RT reflection hit surfaces and glass hits (`reflections.metal`, incl. the normal slot's
  rotated tangent frame), GI hit surfaces (`gi_common.h giMaterial`), probe capture surfaces (`reflection_probes.metal`), emitter textures of DI / reflections / GI / probes
  (`restir_common.h diSampleTexturedLight`: emissive and base-colour-alpha slots).  Mechanism: `shaders/texture_transform_core.h` (constant-free helpers, `texXfSlotUv`), the
  extension-buffer address as a new last field `materialExtensions` of GPUDIParams (112 -> 120 B), GPUProbeGridParams (176 -> 184), GPUReflectionParams (304 -> 312),
  GPUProbeCaptureParams (112 -> 120), filled where each params struct is built.  No function constant (the plain entries stay constant-free for metal-tt); every transform is a guarded
  alternative: a material without MATERIAL_FLAG_TEXTURE_TRANSFORM hands the sampler the very same uv.
- Fixtures (`tools/f15_texture_transform_secondary.py`, CPU self-test = ctest `f15_texture_transform_secondary_python`; thresholds fixed in its docstring before any comparison; a build without the
  site change = `build/pre-sec`, the binary of 096675f):
    rt-shadow / csm-shadow / cache-shadow: an alpha-masked quad (per-slot transform on the alpha channel of the base colour texture, DIFFERENT transforms on two decoy slots) shadows a floor
    behind it from the sun (`--shadows rt` / `csm` / `csm --shadow-cache on`); the reference is the planar sun ray and the README formula; only floor pixels whose sun ray crosses the caster count.
      before: agreement of the reference 0.592 (rt) / 0.586 (csm, cache) against thresholds 0.99 / 0.97 / 0.97 (FAIL; the identity control scores 0.84 / 0.82: the shadow ignored the transform);
      after: 1.0 on 2219 valid pixels in all three; negative controls (the transform of another slot = the uniform-bake result, identity) 0.41 - 0.59 (< 0.90).
    reflection: a palette quad (transformed base colour slot, decoys on two others) seen in a mirror floor (`--reflections rt`, pitch 30): before 0.113 (threshold 0.95, FAIL), after 1.0 on 1972
      valid pixels; controls 0.03 - 0.09.
  The same four cases under API + shader validation (archive off): 1.0, clean (only the known baseline notices; the csm / cache captures add redundant setDepthStencilState / setRenderPipelineState
  notices of the shadow passes, not new failures).  O7: 0 GPU allocations in all four.
- Featureless scenes unchanged (this round's final build): `f15_exact.sh` 5 cases PFM byte-identical (cmp) to build/litefix/exact-new (sponza-d2 included: its RT alpha path runs the changed
  `rtAlphaMaskedTest`), `visual_check` 0 px on 8 benches, the five-slot primary fixture still 1.0 on every slot.
- NOT covered by a numeric fixture (implemented through the same helper, covered by review and the shaders compiling into the archive build): GI hit surfaces (`giMaterial`: indirect colour bleeding
  is smooth, no per-pixel reference exists), the static probe capture (the cube map holds a parallax-corrected, filtered environment: a mirror of it shows a blur, not the cells: tried, rejected as a
  fixture), emitter textures (the texel of a randomly sampled area-light point is not observable per pixel), the transmittance tint of the RT / CSM glass passes (existing transmission fixtures
  do not use transformed slots).  `fog.metal` samples textured emitters without the extension buffer (its params struct has none): a textured emitter with a per-slot transform is lit untransformed
  by the fog in-scatter only.
- Still unsupported, by decision: a second UV set per material (GPUVertex 48 -> 56 B) is Out of Scope in `.scratch/f15/spec.md` (the Out of Scope list, lines 217-219: "...per-texture (non-uniform)
  texture transforms, a second UV set"; the owner override covered the non-uniform transform only): the Unsupported reason stays named ("the textures of this material use different UV sets (the
  engine has one UV stream per mesh)").  Layer sub-material transforms and the layer mask / height UV wait for the layers agent's integrated code (orchestrator follow-up).
- Archive: texxform-2.json is EMPTY by measurement: the four fixture scenes (RT, CSM, cache, reflection) request only pipelines the archive already holds (`PIPELINES ... misses 0` in all four, and
  `filter_pipeline_harvest.py --constant 26 --function ... --skip-present` keeps 0 descriptors from their harvests): the secondary-site changes edit shader bodies, not descriptors.
- For the orchestrator's quiet-machine timing: the guards add one flag test (`m.flags & TEXTURE_TRANSFORM`) at each of the sites above and a pointer field to four params structs.  Sponza D2 (RT alpha
  tests, RT shadows) A/B against 096675f is the number to take; output is bit-identical.
