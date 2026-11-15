# F15: advanced materials and shader variants — handoff

Branch `cl/f15-plan-c38b51`, base 75924b2 (plan approved by the owner on 2026-11-13), M5 Max 128 GB, macOS 27.3.
Plan: [plans/F15.md](plans/F15.md); tickets `.scratch/f15/issues/01..24`; decisions `.scratch/f15/decisions/`; ADR-0003 (permutations by
function constants), ADR-0004 (RT transmission, one Fresnel-chosen sample per pixel), ADR-0005 (OpenPBR subsurface + draft KHR), ADR-0006
(base model = glTF Appendix B). The ticket 24 record (`.scratch/f15/issues/24-phase-closure.md`) is the detailed ledger behind this page.

**Status.** Development work is integrated and verified on the M5 for the scope stated per ID below. **No roadmap box F15.1–F15.4 is ticked**:
each ID keeps partial items that are listed under "Open and partial". Physical Apple9/T0 certification stays `EXTERNAL_VALIDATION_PENDING`.
The Sponza D2 band is met (owner option A, see "Measured cost"). Nothing is pushed; no PR is opened without the owner.

## Delivered scope per roadmap ID

### F15.1 — extended material model (tickets 01, 03–20)

- One shared BSDF module for resolve, DI and reflections (01, bac820d); base model conforming to glTF Appendix B with DFG environment terms
  (03, aa58b66), every model-dependent baseline retaken with before/after evidence (`build/reference-f15-appB`).
- Versioned material description with feature blocks, glTF `KHR_materials_*` parsing (clearcoat, sheen, anisotropy, transmission, volume,
  ior, specular, iridescence, emissive_strength, uniform texture_transform bake, diffuse_transmission RC, draft scatter), refusal of
  unsupported required extensions, URI containment, `MATERIALS` support report (report schema 12) (04).
- GPU extension records through SceneStore deltas, reserved function constant 22, per-scene feature-union specialisation, runtime edits
  without GPU allocations (05); lobe guide (06) read by DI (07) and by the specular transport, composite and denoiser contract (08).
- Lobes: clearcoat (09), sheen (10), anisotropy (11), iridescence (12), transmission + volume with ray-traced refraction transport and a
  declared `--rt off` model (13), attenuated coloured shadows through transmissive casters in RT (sun, ReSTIR/clustered DI) (14) and a
  single-layer CSM transmittance map (15), thin subsurface / diffuse transmission (16), volumetric subsurface as screen-space diffusion
  (17, partial; 18 closed by its fallback), lobe combinations and furnace (19), the Track Studio chess fixture (20, partial).

### F15.2 — permutation generator (ticket 21)

Feature-class manifests exported from the fixtures (81 scenes), the three chess rigs, 21 Khronos corpus models and the terrain; `variant_gen
classes` writes the class table, MSL constants and pipeline descriptors; targeted archive updates (descriptors now compute 916, mesh 143,
render 288, tile 35 against 296 / 142 / 170 / 3 at the base); 41 permutations (<= 64), <= 16 classes per scene, archive build +9.4 to
+11.7 %. Material binning per class works and stays opt-in: it never lowers the resolve by the >= 5 % of decision 05 (chess rigs -1.8 /
+6.7 / +9.8 %, terrain CC0 full +3.5 %).

### F15.3 — material testbench against the Khronos models (tickets 02, 03, 22 and every lobe fixture)

Validation bench with named scenarios and frozen rigs; independent Python reference written from the specifications (Appendix B,
declared environment, every lobe), Mitsuba 3.9.1 ground truth for transmission and subsurface; `tools/f15_check.py` (141 keys, decision-11
thresholds, a negative control per gated fixture). Khronos glTF-Sample-Assets at 8f49239 (default variants, hashes verified, never
committed): 21 of 24 models render, 3 are refused with named reasons; the side-by-side page is qualitative by design (no IBL, different
tonemap).

### F15.4 — layered materials (ticket 23)

Up to four layers from an RGBA mask with optional endpoint-exact height contrast, parameter-space blending with one BSDF evaluation,
surface-gradient normals, a coat finish layer; terrain fixture (procedural heightfield + the four owner-approved CC0 ambientCG sets).
Weight 1 = the single material at 0 pixels (procedural and CC0), weight 0 has no effect, monotone ramp, furnace with a failing additive
control. Resolve cost at 1080p (CC0): single material 0.087 ms, 1 / 2 / 3 / 4 active layers 0.136 / 0.159 / 0.183 / 0.217 ms, the full
generated mix 0.201 ms.

## Defects found during F15 outside F15 code (owner: fixed inside F15)

Recorded with fixtures (failing before, passing after) in [the F13/F14 closure document](implementation/F13-F14-CLOSURE-2026-11-12.md),
section "Defects found during F15": clustered-DI GPU hang with emissive meshes (1bf0e19, 128 lights per pixel, stratified); the C22-D3
silhouette ring (e9569ac); shadowed secondary hits black with GI off (bf0799f); the DDGI wall behind glass (4422a13); transport rays ending
black on single-sided back faces (26fe229); the declared environment missing from primary surfaces in every GI mode (4fc1c8a). Proven not
to be defects: the IORTestGrid ring (white-furnace proof, eac5233) and the DI-cap 0.973 ratio (the a-trous luminance weight, 7e3f730; the
owner's timeboxed energy-preserving a-trous experiment removed the bias but cost +13.3 ms on Sponza D2, so the residue is declared with its
bound, owner option B; GI temporal 0.926 isolated as history lag, 0.998 at convergence).

## Measured cost

Quiet-machine numbers are in [perf-log](perf-log.md) (entries of 2026-11-15) and [opt-log](opt-log.md).

- **Sponza D2 (decision-11 band: median of three A/B/A GPU p95 deltas <= 0.10 ms; absolute <= 16.67 ms): MET.** The plan's cost baseline is the
  ticket-03 state 2e1057a. A 124-byte RT payload added by ticket 14 slowed every RT traversal (RTAO +0.46 ms); 8fbda9c gave RTAO, fog and the sun
  visibility the 16-byte payload (images bit-identical). The remaining p95 excess sat in `F13 reflection sample 0`, where the 26fe229 back-face
  continuation re-traces up to five times (~200 rays per frame exhaust the bound). Owner decision, option A: the continuation's cost is the F13
  fix's, and the F15 band (still 0.10 ms) is measured against the ticket-03 baseline plus 26fe229 (a never-merged backport,
  `cl/f15-d2-t03-plus-26fe229`). Measured that way the closure tree still missed it (+0.198 ms, rotation +0.104): the reflection pass carried the
  124-byte payload, and the continuation multiplied its traversals. Fixed in F15 (720ee63): the base `reflection_rt` traces with the 16-byte payload
  (F13 exactness 0 px, PFM bit-identical). Final numbers: F15 band -0.038 ms (pairs -0.04 / +0.04 / -0.10); against the plain ticket-03
  baseline +0.020 ms; F13 continuation cost on the shipped code, same executable with and without it, +0.065 ms (+0.009 at the baseline); GPU p95
  15.33-15.43 ms, 0 GPU allocations. The hardware back-face culling alternative was discarded (it changes the first surface of 24-35 rays per frame
  and recovered 0.07 ms). Records: perf-log "F15 ticket 24: Sponza D2 band under the owner's option A", opt-log "follow-up 3".
- Feature costs (A/B/A x 3, recorded, not gated): transmission on the chess back-light rig +10.67 ms frame p50 (mostly `Local clustered RT
  shading` +9.18: an optimisation candidate), RT sun shadows through glass +0.15 ms, the CSM transmittance map at 8192 +1.03 ms and +1028 MiB,
  at 2048 about 4 x 0.021 ms, subsurface diffusion 0.43 ms at 1080p, the thin GI-back lookup 0.175 ms. The iridescence per light has no clean
  A/B in the current options (static estimate 187 flops per `bsdfThinFilm`).
- Startup: Sponza D2 pipelines ready in 14–20 ms with the archive; 12 compiler calls for the linked RT compute pipelines.

## Open and partial (the reason no box is ticked)

- F15.1: volumetric subsurface misses the side-lit bound on OpenPBR ground truth (gated mean 0.25–0.36; ticket 18 fallback); the chess
  subsurface bound cannot be built (texture-free reference); the transmission temporal clip fails F13.6 on smooth glass under orbit;
  the refract-axis denoiser control is not demonstrable; numeric texture fixtures missing for coat normal, anisotropy direction, iridescence
  thickness, sheen and subsurface weight/colour; image equality of the glTF encodings vs the description file not built.
- F15.2: class pipelines are not bit-identical to the all-lobes pipeline (77 of 116 bit-identical, the rest within 37 ULP on fixtures and up
  to 1.0e-2 relative on textured real assets, 8-bit within 1 LSB): the "exactly 0 pixels" criterion is met only in 8-bit.
- F15.3: two thresholds were fixed after the first measurement (21: 1e-5 envelope; 23: 5e-4 coat-zero gate) and are reported as such.
- F15.4: layers blend base parameters only (feature blocks, emissive and alpha are not blended).
- The AGX compiler crash seen once in the corpus (`negative-f14-lut`, SIGSEGV inside the driver's linked-intersection compile) is
  unexplained: about 200 attempts in seven configurations did not reproduce it; `bench/agx_link_stress` stays for a rerun.
- The complete list, with source tickets, is the carry list of ticket 24.

Declared approximations and deferrals (support table and plan "Rischi"): caustics, dispersion, multiple-scattering compensation,
non-uniform texture transforms, back-lit thick subsurface (hybrid not met), coated emission of sampled emitters, GI/probe/AO/fog rays that
see glass as opaque, one Fresnel-chosen transmission sample per pixel with at most 4 crossings / 8 events, single-bounce reflections
(mirror-in-mirror dark), secondary RT hits shaded with the base reduction (coat, sheen, anisotropy, iridescence ignored there).

## Closure verification

The battery ran on 29814ac (Release, `PHOSPHOR_DEBUGGABLE`, executable sha256 `8c0c002b…`, metallib `3bab6e23…`; records in
`build/closure-battery/` and `build/f15-corpus-closure/`).  The reflection lite-payload fix that followed the D2 measurement was re-verified on
the final tree (records in `build/litefix/`, rows marked "final" or "fix").

| Check | Command | Result |
|---|---|---|
| Unit and script tests | `ctest --test-dir build/release` | 33 / 33 |
| `metal_syntax_check` | the CMake target exists only on non-Apple hosts; emulated on macOS with the same flags and `tools/apple-sdk-stubs` | 48 / 49 files pass; `metalfx_lifetime.cpp` fails only because its `#ifdef __APPLE__` block (Objective-C runtime calls) is compiled under the emulation; the file is unchanged by F15 and identical to `origin/main`, whose Linux job is green. The one warning F15 introduced (misleading indentation, 26fe229) is fixed (29814ac); the other five warnings predate F15. The real target runs in CI on the owner's push |
| Visual regression | `BENCH8_ARGS="--instances 100000" tools/visual_check.sh build/release build/reference-f15-appB` (references of ticket 03) | 0 px on 8 benches, API + shader validation |
| Forward variants | `tools/variant_check.sh build/release`, again with `--geometry-path mesh --meshlet-cull off` | pass, pass |
| Pipeline archive | `tools/archive_check.sh build/release build/reference-f15-appB` | every scenario passes |
| Bench-switch hitches | `tools/hitch_check.sh build/release` | 0 hitches, 21 / 21 archive hits, 0 render-thread compiles; the same on the final tree 720ee63 (`build/litefix/hitch.log`) |
| Subsurface forced overflow | `tools/f15_subsurface_overflow_check.sh build/release/phosphor` | PASS (worst 1.6e-5 against 2e-3) |
| Apple9 forced path | `--force-family apple9` on combo-all-lobes-sun, subsurface-volumetric-sun, layers-patch, transmission-glass-behind-glass, transmissive-shadow-pane-sun@rt | all exit 0, 0 validation lines (API + shader validation where F14.4 allows), report `effective_capabilities: apple9` on physical apple10; transmission and transmissive shadows bit-identical to native; the denoised scenes differ by the Apple9 preset (fewer iterations, shorter history), as declared |
| Material testbench | `tools/f15_check.py --run --expect green` (Mitsuba 3.9.1 under uv) | GREEN: 124 gated scenarios, 122 pass, 2 expected fails (volumetric subsurface side-lit bound, ticket 18 fallback), every negative control fails; identical outcome on the final tree 720ee63 (`build/litefix/f15-check.log`) |
| F13/F14 corpus | `tools/f13_f14_check.py --run` with the F13/F14 closure's frozen cases, independent reference clips (`build/f13-reference-clips-20261110`) and ROI configuration, API + shader validation | exit 0: 89 / 89 cases executed, functional and gating pass (control 30/30, cross-check 2/2, full-rate-pair 17/17, functional 13/13, independent-clip 12/12, numerical 14/14, repeatability-pair 1/1), case by case identical in state to the F13/F14 final4 run and to the ticket-03 baseline run; the AGX compiler crash did not recur; phase acceptance is not claimed by the tool (its F14 hard stop), as in final4; rerun on the final tree 720ee63 (`build/f15-corpus-final`): exit 0, 89 / 89, every case identical in state to the 29814ac run |
| Sponza D2 band (final tree) | `f15_d2_aba.sh` A/B/A x 3, 6-replica rotation, owner option A | F15 band -0.038 ms against ticket 03 + 26fe229, +0.020 ms against ticket 03; GPU p95 15.33-15.43 ms; 0 GPU allocations |
| Reflection lite-payload fix (720ee63) | ctest; F13 exactness 5 cases against 29814ac; `visual_check`; `archive_check` | 33 / 33; 0 px and PFM bit-identical; 0 px on 8 benches; every scenario passes |

## Next steps

1. Owner review of this branch; with the owner's go-ahead, push `cl/f15-plan-c38b51` and open the F15 PR (nothing is pushed). CI then runs the
   real `metal_syntax_check` (Linux) and the `app-macos` job.
2. After the merge, F16 (on-tile OIT, particles, VFX) per the owner direction of 2026-11-10; F16.2 (glass and liquids) should reuse the F15
   transmission contract (ADR-0004) rather than add a second glass model.
3. To tick F15.1–F15.4, close the partial items above, in this order of value: numeric texture fixtures (coat normal, anisotropy direction,
   iridescence thickness, sheen, subsurface weight/colour) and the glTF-encoding equality images (cheap, test-only); the transmission temporal clip
   on smooth glass (F13.6); bit-identical class pipelines (F15.2) or a declared 8-bit criterion decided by the owner; layer blending of feature
   blocks (F15.4); the volumetric subsurface bound (an RT random walk is the recorded EDGE candidate).
4. Optimisation candidates with measured evidence, not activated: `Local clustered RT shading` with transmission (+9.18 ms on the chess back-light
   rig); the lobe kernels of feature scenes without glass still trace with the 124-byte payload.
5. If the AGX linked-intersection compiler crash recurs, rerun `bench/agx_link_stress` with the failing configuration and file a Feedback report.
6. The local branch `cl/f15-d2-t03-plus-26fe229` (worktree `f15-t03-baseline`, whose build now holds the backport binary; the pristine 2e1057a
   binaries are in `build/d2-skips/t03`) is a measurement baseline only; delete it after the owner accepts the D2 record.
