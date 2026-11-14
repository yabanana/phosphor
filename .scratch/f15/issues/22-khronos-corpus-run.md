# 22: Khronos corpus run and defect report

Roadmap: F15.3

**What to build:** A pinned fetch script brings Tier A/B models (glTF-Sample-Assets 8f49239, hashes and licence records, never committed); the bench renders each under the declared Phosphor rig; a side-by-side page against the Khronos screenshots/goldens documents per-feature defects and the explained Sample Renderer deviations.

**Blocked by:** 19

**Status:** resolved with open defects (2026-11-14; 21 of 24 models render, 3 refused with stated reasons; defects C22-D3/D4/D5/D7/R2 open; numeric texture fixtures of tickets 09/11/12/16 NOT built)

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
