# 16: Subsurface feature, thin mode (diffuse transmission)

Roadmap: F15.1, F15.3

**What to build:** The subsurface feature (OpenPBR parameters, ADR-0005) in its thin mode: a diffuse-transmission lobe for analytic lights on the back hemisphere, back-side environment and GI irradiance at −N; fed by `KHR_materials_diffuse_transmission` (RC), the draft `KHR_materials_scatter` thin mode and the material-description file.

**Blocked by:** 07, 08

**Status:** resolved with carried items (2026-11-16, round 2: image equality of the glTF encodings (diffuse_transmission, draft scatter) vs the description file is built and met (tools/f15_encoding_check.py, 0 pixels differ, repeat capture 0, controls differ); weight / colour textures `texture-thin-sun` / `-env-resolve` GREEN; thin under sheen / anisotropy / iridescence / ior is composed in the engine and gated (`combo-thin-lobes-sun`, `-env@resolve`, exact-environment `combo-thin-lobes-env` worst sphere mean 1.95 % after the declared transmitted term was fitted per ior / film, controls fail); GI at -N is a probe-grid lookup in every GI mode and the legacy resolve (quad errors 6.7 / 6.2 / 4.7 % in all six mode x consumer combinations vs the 25 % indicative gate, `--thin-gi-back off` fails; DDGI back lookup +0.175 ms at 720p recorded in ticket 24); the 8 background compiler calls are explained (linked RT compute pipelines routed through the compiler by policy); the RT / DI visibility of a back-lit thin receiver is verified (`thin-backlit`, 3.83x unlit, occluder control 0.000), while CSM self-shadowing of a back-lit thin receiver is NOT addressed by any record in this ticket and stays open)

- [x] Thin fixtures (backlit plane, factor and colour sweeps) pass decision-11 thresholds; furnace bound holds (energy moved, not added)
- [ ] Both glTF encodings and the description file produce the same description (doctest) and image — description doctests exist (ticket 04); the IMAGE comparison needs a glTF test asset → ticket 20 (the chess fixture uses both a glTF and a description file)
- [x] Negative control (lobe on the front hemisphere) fails; support report labels RC/draft inputs
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — test-first harness (independent reference, fixtures, runner keys, targeted archive update); no engine or shader
change. Commits: reference `df5195f`, fixtures and keys `d878e09`, archive `482e643`, furnace control list (next commit), and the
commit with this note.

Reference (`tools/f15_reference.py`, section "Subsurface thin mode / diffuse transmission", from the RC text in
research/khr-materials.md section 10.1; never from the MSL). Interpretations, each one open for confirmation:
- Hemispheres relative to the interpolated SHADING normal; front = the camera side (N.V > 0), back = the -N side. Thin-walled:
  same point, no refraction, no offset, no thickness. Lights with N.L < 0 light the visible face through the diffuse BTDF.
- Formula (the RC's normative composition under the Appendix B `fresnel_mix`): reflected `(1 - w)(1 - metallic)(1 - F(V.H))
  baseColor / pi` for N.L > 0 (specular lobe untouched: "the specular reflection BSDF and Fresnel weighting are not affected");
  transmitted `w (1 - metallic)(1 - F(V.H)) color / pi |N.L|` for N.L < 0; F = Schlick(0.04, |V.H|), H = normalize(V + L) of the
  literal text, ALSO for the transmitted lobe (the glTF Sample Renderer does the same). For a back light V.H = sqrt((1 + V.L)/2)
  <= 0.71, so a light straight behind the camera ray (V = -L) transmits nothing and the exact albedo is T(1) = 0.8075,
  T(0) = 0.933. ORCHESTRATOR: confirm the literal V.H reading (alternative: 1 - F(N.V), which the text does not say). Degenerate
  H (|V+L|^2 <= 1e-20) gives 0, the engine's rule. Metals carry no transmission. The transmission colour is not multiplied by
  baseColor; the RC README's text/formula inconsistency (colour texture times the factor) is resolved as in research 10.1
  (formula wins, the mix carries w once).
- Sun: reflected and transmitted terms point-evaluated at the disk centre (the engine's diffuse rule).
- Exact environment (constant L): `(1 - w) c Cd(N.V) + w color T(N.V) + F0 A + B`, T the back-hemisphere quadrature of the same
  integrand (`thin_transmission_table`, 192x192 Gauss-Legendre, agrees with 256x256 to 1e-6 and with an independent
  cosine-weighted Monte Carlo to 4e-3). T <= Cd at every N.V, so reflected + transmitted never exceeds the unextended model.
- Declared environment (the engine's, to implement): the declared front diffuse `c (1 - E_d)` is scaled by `(1 - w)` and the
  transmitted lobe adds `w color (1 - E_d(mu)) (1 - 0.16 mu)`, E_d = clamp(0.04 A + B, 0, 1) from the DFG table at the material
  roughness. The ratio r(mu) = T / Cd is 1 - 0.16 mu within 1.4e-3 (tested), so the transmitted weight never exceeds the diffuse
  weight it replaces (the furnace bound holds by construction; measured declared-vs-exact sphere-weighted mean error 2.9 %).
  Rejected: a plain `(1 - E_d)` transmitted weight (16 % over the exact T at N.V = 1).
- GI at -N (decision 06) and the secondary-hit base reduction are engine-only: no GI, no RT hits in the fixtures.
- Shadow rays of the DI keys start on the side of the light (offset `1e-4 * sign(N.L) * N`): a thin carrier does not occlude its
  own transmitted light. The engine's DI must do the same for a back light (ORCHESTRATOR: engine requirement).
- Refused (ValueError, runner exit 2): volumetric mode (ticket 17), subsurface textures (words 10-12; ticket 22), non-zero
  anisotropy (the draft's forward/backward scatter split is not modelled: with `KHR_materials_scatter` thin the loader maps
  weight = transmission * scatterStrength and keeps g), any other feature bit alongside (combinations not modelled; the RC's
  "transmission overrides diffuse transmission" is a ticket 13/19 matter). Radius/radiusScale are volumetric-only and ignored.
- Controls: `transmission-ignored` (w = 0, the engine today), `-untinted` (colour white), `-front-only` (the BTDF on the FRONT
  hemisphere: back lights ignored, a front light gets an extra w color/pi term: covers both "back lights ignored" and "lobe on
  the front hemisphere" of the ticket), `-double-counted` (diffuse not reduced by w: energy), `-fresnel-ignored` (extra, no 1 - F
  on the transmitted lobe). Declared twins: `declared-transmission-ignored / -untinted / -front-only / -double-counted`.
- `tools/test_f15_reference.py`: 134 tests OK (16 new: hand value of the literal fresnel_mix, hemisphere and metal rules, reflected
  reduction by exactly w * (1 - F) baseColor / pi, w = 0 equals the base model bit for bit in every term, direct sun/point sums,
  control directions, T against a Monte Carlo and T <= Cd, environment albedo identities and monotonicity in w, declared hand
  value and furnace bound by construction, record 2 decoding and every refusal, Mitsuba snapshot tests including the shadow-ray
  side).

Fixtures (`src/testbench/material_validation.*`, `tests/test_material_validation.cpp`; 40 scenarios, doctest OK): rows weight
{0, 0.25, 0.5, 1} (weight 0 carries no extension, the neutral row), columns transmission colour {white, warm (1, .55, .25), teal
(.1, .7, .6)}, blue-grey base (.3, .5, .8), roughness 0.5, metallic 0.
- `subsurface-thin-sun`: spheres, sun 3 W/m2 BEHIND the grid (toward (-.55, .4, -.73), V.L -0.73 at the centre): the visible face is
  lit through the transmitted lobe, a crescent from the front through the reduced reflected lobe.
- `subsurface-thin-env-resolve` (declared gate) and `subsurface-thin-env` (exact quadrature, transport): environment 0.5.
- `subsurface-thin-point`: single-sided quads facing the camera (generatePlane rotated +90 degrees about X; no self-occlusion of the
  DI shadow rays) with one 250 W/sr point light behind (9 m, same direction), rows {0.25, 0.5, 1} (a weight-0 row would have no lit
  pixel and the direct gate fails an empty group), keys legacy / `@restir` / `@clustered`.
- `furnace-thin-subsurface` (+ `@resolve`): white base, white transmission colour, environment 1, columns vary the base roughness
  {0.2, 0.5, 0.9}.
What the DI receivers do today: `diBRDFWeighted` returns 0 for a light with N.L <= 0 (restir_common.h:143), the resolve's lights
saturate N.L (material_shading.h), so a light behind the surface contributes nothing anywhere: the engine image of the back-lit
fixtures is black today.

Runner (`tools/f15_check.py`, `tools/test_f15_check.py`): keys `subsurface-thin-sun`, `-env-resolve` (declared), `-env`,
`-point`, `-point@restir`, `-point@clustered`, `furnace-thin-subsurface`, `furnace-thin-subsurface@resolve`; all
`"pending": "ticket 16"` with `pending_matches` = `transmission-ignored` / `declared-transmission-ignored`, except the furnace
transport bound (asserted green: the blind engine satisfies an upper bound; the double-counted control fails it) and the literal
gate of `@resolve` (`pending_gate: declared`). Two runner changes: `relative_error` no longer divides 0 by 0 when the reference is
black over the whole ROI (the transmission-ignored image of the back-lit quads is exactly black: NaN made the pending twin fail);
`F15_REFERENCE_THREADS` caps the Mitsuba threads (default unchanged). 7 new runner tests + the zero-reference test.

Measured red state (M5 Max, `build/f15-check/t16-red`, 1280x720, `--run --expect green`: GREEN, exit 0; every capture 0 archive
misses and 0 render-thread compiles after the archive update; the first run failed with 11 archive misses on the unmasked union 128):
- subsurface-thin-sun: FAIL, p99 per weight group 0.0003 (w 0, neutral, passes) / 1.0 / 1.0 / 3.88; passes against
  transmission-ignored.
- subsurface-thin-env-resolve: FAIL, worst sphere mean 2.13 (limit 0.005); twin (declared-transmission-ignored) worst mean 3.9e-5.
- subsurface-thin-env: FAIL, worst sphere mean 2.12 (limit 0.05); twin worst mean 0.0099 (so the declared environment of the base
  matches the exact quadrature at 1 % on this fixture).
- subsurface-thin-point, @restir, @clustered: FAIL, p99 1.0 in every group (black against lit); twin passes in all three.
- furnace-thin-subsurface: gate PASS (sphere mean max 1.00001, pixel max 1.0098), asserted green; the double-counted control
  fails (2.22).
- furnace-thin-subsurface@resolve: engine-pixel gate PASS (1.0000001), declared gate FAIL (worst mean 0.122); twin passes; controls
  fail (double-counted 2.22 literal, 1.10 declared).
Every control fails its gate on every key it is listed for (env-resolve declared: ignored 2.13, untinted 5.79, front-only 0.119,
double-counted 2.94; env exact: ignored 2.13, untinted 5.81, front-only 0.118, double-counted 2.95, fresnel-ignored 0.166).
Dropped / not registered where they cannot differ: `transmission-double-counted` on the point keys (the single back light leaves no
front-lit pixel; the sun and environment keys own it), `declared-transmission-untinted` and `-front-only` on the furnace (white
colour; front-only is the ignored image in a uniform environment, measured identical), `fresnel-ignored` has no declared twin (the
declared model has no V.H).

Archive: targeted update, +17 descriptors with MATERIAL_FEATURES 128 (compute 417 -> 429, render 194 -> 198, tile 9 -> 10, mesh 142
unchanged), harvested from the eight captures; a scene combining subsurface with another feature needs its own union.

Open / not done:
- Not run: `tools/archive_check.sh`, `tools/hitch_check.sh`, O7 allocation counts of measured frames (the BENCH line counts the
  whole run), `--debug-lobe-guide` (the guide has no subsurface check yet), shader validation.
- The restir and clustered captures show 8 background compiler calls (0 misses, 0 render-thread compiles, 0 fallbacks): not
  investigated here.
- Textured weight/colour, volumetric mode, subsurface + transmission/clearcoat combinations, metals with weight (the metallic rule
  is unit-tested only), the draft's anisotropy split, the Track Studio piece mask and the Mitsuba random walk (tickets 17/18/22).
- Acceptance items 2 and 3 (both encodings produce the same description and image, support report labels) are loader-side and
  were already covered by ticket 04's doctests; the image equality needs the engine.

2026-11-13, orchestrator decision on the harness question:
- The transmitted lobe is weighted by 1 - F(V.H) with H = normalize(V + L).
  This is the literal composition of the RC text (fresnel_mix around
  mix(diffuse_brdf, diffuse_btdf, w)), and the Sample Renderer does the
  same. Confirmed.
- The declared environment weight w * color * (1 - E_d) * (1 - 0.16 N.V) is
  what the engine implements.
- DI receivers must evaluate lights on the back hemisphere. Their shadow
  rays leave from the light's side.

2026-11-13 — engine side (F15.1), commits `99e0d3d` (BSDF, resolve, DI, composite, guide, support table, tests), `fa3c7e1` (targeted
archive update), `558f62f` (runner keys no longer pending), `3ad271e` (comment only).
What was built:
- `bsdf.h`: `BsdfThin {weight, colour}`; reflected diffuse x (1 - w), transmitted `w (1 - metallic)(1 - F(V.H)) colour / pi |N.L|`
  for N.L < 0 with H = normalize(V + L) (degenerate H gives 0), declared environment weight `w colour (1 - metallic)(1 - E_d)(1 - 0.16 N.V)`.
- Resolve (`shadeSurface`): record 2 read for mode 0 only (volumetric: base model, no subsurface bit published; ticket 17); analytic
  lights and the sun disk centre keep back-hemisphere lights (no N.L clamp to the front), specular lobe unchanged; environment: front
  diffuse x (1 - w), transmitted term on the radiance at -N (declared environment; legacy hemisphere at -N; with GI, see below).
  The term travels with the diffuse ambient (the composite adds it when the resolve strips it).
- DI `_lobe` kernels (ReSTIR candidates/temporal/spatial/shade and clustered share `diBRDFLobe`/`diTarget`): `DILobe` carries the thin
  parameters; target and shading evaluate back-hemisphere lights. The visibility ray needed NO change: `diEndpointVisible` already flips
  the geometric normal towards the light before `rtOffsetRay`, so a back light's ray leaves from the light's side.
- Composite `_lobe`: front diffuse x (1 - w) plus the transmitted environment/GI term. GI at -N is NOT a probe lookup: the pixel's own
  GI irradiance (the denoised irradiance at N) stands for E(-N) (declared approximation; no fixture has GI; support table id
  `gi_irradiance_at_n_stands_for_minus_n`). A genuine -N lookup needs a second irradiance signal and is carried.
- `LOBE_GUIDE_RESOLVED_FEATURES` gains SUBSURFACE (the volumetric flag is untouched); guide words carry weight and colour; CPU
  expectation reads record 2 (mode 0, textures unverified, volumetric clears the bit); the corrupt control shifts the weight (+0.25).
- Support table: subsurface = approximated (`thin_mode_only;gi_irradiance_at_n_stands_for_minus_n;secondary_rt_hits_ignore_transmission`),
  the volumetric mode has its own unsupported entry (`MaterialSupportTable::subsurfaceVolumetric`); tests that used the subsurface
  feature as unsupported were updated. No harness or reference change.
Verification (M5 Max, release build, `-j 6`): ctest phosphor_tests 840 cases OK; `tools/test_f15_check.py` 70 tests OK.
Runner `--run --expect green` over ALL keys (59 scenarios, `build/f15-check/t16-all`): GREEN exit 0, every gate pass, every expectation met,
every negative control fails its gate, every capture 0 archive misses and 0 render-thread compiles (the restir/clustered captures keep
the 8 background compiler calls noted above). Thin keys (`t16-b`): sun p99 <= 0.0003 per weight group; env-resolve worst sphere mean
0.00032 (limit 0.005); env worst sphere mean 0.0098 (limit 0.05); point legacy p99 0.0000, @restir / @clustered p99 0.0004 (same as the
base DI images); furnace transport max sphere mean 0.99999 (limit 1.005); furnace@resolve pixel max 1.00000 and declared gate worst
mean 0.00000 (limit 0.005). Archive: +7 compute descriptors with MATERIAL_FEATURES 128 (429 -> 436; render 198, tile 10, mesh 142).
Guide: `--debug-lobe-guide 8` PASS on subsurface-thin-env (14952 extended pixels), corrupt control FAIL (14952 mismatches, exit 1);
`tools/f15_lobe_guide_overflow_check.sh` on subsurface-thin-sun PASS (dumps identical, 18432000 bytes). Validation
(`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`): 1500 frames of subsurface-thin-point with ReSTIR + RT reflections + custom denoiser exit 0,
all LIGHTING and LOBE-GUIDE checks PASS; 200 frames of subsurface-thin-env with DDGI (the GI composite path) exit 0, guide PASS (no
reference image for GI).
Carried / not done: `tools/archive_check.sh` and `tools/hitch_check.sh` (not run, orchestrator); O7 per-frame allocation counts (the BENCH
line counts the whole run); image equality of the glTF encodings against the description file (loader doctests exist, no engine image
comparison); GI at -N as a real second lookup; CSM/RT sun shadows of a back-lit thin receiver use the standard sun visibility (the carrier can
shadow its own transmitted sun: transmissive shadows are tickets 14/15; the fixture has no shadows); weight/colour textures (guide expectation
unverified, no reference: ticket 22); combinations with other lobes (not composed in the thin branch: a thin material in the resolve skips the
coat/sheen/aniso light branches); metals with weight (rule unit-tested only); volumetric mode (ticket 17).

2026-11-14 — orchestrator verification on M5 after integration (engine
99e0d3d-equivalents). Batch log: `build/f15-exact/battery16.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t16`): GREEN on
  all 59 keys, every control failing.
  - `subsurface-thin-sun`: worst p99 0.03 %.
  - `-point`: 0; `@restir` and `@clustered`: 0.04 %.
  - `-env-resolve`: worst mean 3.2e-4.
  - `-env` (transport): worst sphere mean 0.98 %.
  - `furnace-thin-subsurface`: max sphere mean 0.99999 (gated); max pixel 1.0079 on the stochastic transport (reported, not gated, decision-11 furnace amendment); `@resolve`: max pixel 1.0.
- `--debug-lobe-guide 16` on subsurface-thin-env: `pixels 20052 extended 14952
  | PASS`. The corrupt control shifts the weight.
- Shader validation: subsurface-thin-point with ReSTIR and RT reflections
  exits clean.
- O7 (warmup 64, 64 measured frames): subsurface-thin-env in both lighting
  modes gives 0 GPU allocations and 0 render-thread compiles.
- Scenes without features: F13 exact captures and visual_check 0 px.
  archive_check: every local scenario PASS. hitch_check OK. The clip runner
  and the coat-emitter check meet every expectation.
- Decision-06 deviation, declared (`gi_irradiance_at_n_stands_for_minus_n`):
  the transmitted part uses the pixel's own GI irradiance at N instead of
  E(-N). A real second lookup needs a second irradiance signal → ticket 19,
  and the residue audit of ticket 24 lists it.
- Declared:
  - the sun's shadow of a back-lit thin receiver is the standard sun
    visibility (tickets 14/15);
  - a thin material in the resolve does not compose coat, sheen or
    anisotropy (ticket 19);
  - weight and colour textures have no reference (ticket 22).

2026-11-14 - ticket 19 closed the carried item "GI at -N second lookup" for DDGI (commit c0de265: second probe-grid lookup for thin pixels, --thin-gi-back, declared check tools/f15_thin_gi_back_check.py) and composed the thin lobe with a clearcoat on the sun and delta-light paths of the resolve (commit ab42590); details in the last comment of ticket 19. Cache/ReSTIR GI and the legacy resolve GI path keep E(N) for E(-N).

### Ticket 22 closure (2026-11-14)

- DiffuseTransmissionTest is REFUSED (requires KHR_materials_unlit and KHR_lights_punctual, decision 02), so the subsurface weight/colour TEXTURES still have no reference and no corpus coverage: open. The glTF + description IMAGE comparison stays with ticket 20.

2026-11-15 — closure audit (ticket 24):

- 1 thin fixtures + furnace: MET (t16 59 keys GREEN; furnace transport 0.99999). Ticked.
- 2 both glTF encodings and description file same description and image: PARTIAL / NOT MET (description doctests exist in ticket 04 and 2dd468d; IMAGE equality never built; ticket 20 has no such criterion, DiffuseTransmissionTest refused in 22; ticket 24 did not list it). Open, orphaned.
- 3 negative control + support report labels: MET (transmission-front-only control fails; 4727b0f). Ticked.
- 4 archive/hitch/O7: MET (+17 then +7 descriptors). Ticked.
Open: GI at -N is DDGI only (c0de265, ticket 19), its cost unmeasured; thin + coat composed on sun and delta lights only (ab42590), thin + sheen/aniso/iridescence/ior refused by the reference; back-lit thin receiver uses standard sun visibility; 8 background compiler calls in restir/clustered captures never explained.

## F15 closure, loader-misc (2026-11-15): the 8 background compiler calls explained

Open item: "8 background compiler calls in restir/clustered captures never explained".  DONE: they are exactly the linked RT compute pipelines, routed through the compiler by policy; there is no archive gap to close.

- `PipelineCache::resolve` (src/platform/metal/pipeline_cache.cpp) sends every COMPUTE pipeline with `linkedFunctions` (the RT kernels linking `rt_alpha_generic` / `rt_alpha_lite`) through the Compiler, never through
  the archive: "the direct Archive static-link path retained a linked function in the F9 lifetime checks" (comment at the routing line).  They are counted as compiler calls by design, with `archive` NotTried (so they are
  not archive misses either).  The log now names them (`Pipeline linked compute (compiler by policy, never archived): <label>`).
- Evidence (`build/f15-irid-smoke/iridescence-grid-point-clustered.log`, 960x540, `--lighting clustered --rt on`): `compiler calls 8 (335.0 ms, max 73.3 ms)`, archive hits 155, misses 0, render-thread compiles 0,
  and exactly eight such lines: rt_trace_rays, restir_di_shade_rt, light_cluster_shade_rt, reflection_rt, ao_rtao, reflection_capture_rt and light_cluster_shade_rt_lobe at the two feature masks (1 and 511); the ReSTIR
  runs add restir_di_shade_rt_lobe and reflection_rt_lobe (the Khronos corpus rig, `--lighting restir`, `--reflections rt`: 9 lines, the rest of its 28 calls are archive misses of an unharvested model).  Same count as
  ticket 07's "6 base RT plus 2 lobe RT on workers".  The compile runs on the utility-QoS workers (335 ms of summed CPU at startup, 0 on the render thread), so it never reaches a frame.
- Not changed: archiving them again would reintroduce the lifetime retention of ticket F9 (the policy is older than F15 and measured); the cost is startup CPU on workers only.

2026-11-15 — F15 closure, agent "thinwall": the carried "GI at -N" item is closed for every GI mode and for the legacy resolve (details and numbers in ticket 19, same date): E(-N) is the
probe-grid lookup of the back normal for DDGI, the radiance cache and ReSTIR GI, read by the reflection composite and by the legacy visibility/forward resolve; `--thin-gi-back off` (the old E(N)
stand-in) fails the declared check in all six mode x consumer combinations, `on` is within 4.7-6.7 % of the independent expectation (gate 25 %, indicative).

### 2026-11-15 — weight / colour textures and encoding equality (agent `tex`, branch f15c-tex-2)

- Thin weight (A) and colour (sRGB RGB) textures: `texture-thin-sun` and `texture-thin-env-resolve` GREEN (`build/f15-check/tex4`): sun worst p99 0.18 %, max outside mask
  2.5 % (one dielectric sphere, below the 3 % bound), env-resolve worst mean 0.053 %, p99 0.12 %. Controls texture-v-flipped, thin-weight-ignored, thin-color-ignored,
  thin-color-srgb-encoded FAIL (sun and declared twins). Closes the "textures have no reference" carry.
- Encoding equality (box 2, was orphaned): `tools/f15_encoding_check.py` renders one generated scene (two spheres with `KHR_materials_diffuse_transmission`, one with the draft
  `KHR_materials_scatter` + `KHR_materials_transmission`) and the same glTF with plain materials plus `--material-overrides` (the description file, schema
  phosphor-material-overrides) holding the blocks the loader maps the encodings to. Result (`build/f15-encoding`, corpus rig, 1280 x 720 PNG): **0 pixels differ** between
  the glTF encodings and the description file; the repeat capture of the glTF scene differs by 0 pixels (the capture is deterministic, so the equality is meaningful);
  negative controls differ as they must: weight +0.02 on one sphere 21 523 pixels (max delta 3), blue channel +0.02 on one sphere 14 479 pixels (max delta 2).
  Unit tests: `tools/test_f15_encoding_check.py` (scene, override mapping, pixel counting). Run: `F15_GPU_LOCK_HELD=1 python3 gpulock.py uv run ... tools/f15_encoding_check.py
  --binary build/release/phosphor` (or alone: the capture takes the lock itself).

### 2026-11-15 (2) — thin under sheen / anisotropy / iridescence / ior (agent `tex`, round 2)

Semantics, pinned to the plan's primary source (glTF-Sample-Renderer cc3d5ff; the source tree is not in this checkout, so the citations are the research note that
quotes it, `.scratch/f15/research/khr-materials.md` section 12, lines 453 and 460, and the layering order on line 440):
- Dielectric base: `f_dielectric_brdf = mix(f_diffuse, f_specular_dielectric, F)` with `f_diffuse = mix(diffuse_brdf, diffuse_btdf, w)`: the diffuse transmission sits INSIDE the
  base of the same Fresnel mix as the diffuse reflection (line 453). Consequence implemented: the transmitted lobe takes the diffuse weight of the reflected lobe,
  `(1 - metallic) (1 - max3 F)` with F the extended dielectric Fresnel (ior/specular) or the thin-film Fresnel (iridescence) at V.H of the real half vector
  (`bsdfThinTransmittedSpec`, `tools/f15_reference.py thin_transmitted`), instead of the default 0.04 Fresnel.
- Back lights: the Sample Renderer evaluates the BTDF with the lambertian term of `-n` (line 460, "lightIntensity * clampedDot(-n, l) / pi * color"); the sheen scaling
  `albedoSheenScaling = min(1 - m E(N.V), 1 - m E(N.L))` uses `N.L = clampedDot(n, l) = 0` behind the surface, i.e. E at the smallest table N.L (clamped to 1), and the sheen
  lobe is added for `N.L > 0` only: the whole base, transmitted lobe included, is scaled (`thinSheenBackScale`, reference `point_radiance` / `sun_radiance`).
- Anisotropy changes D V of the specular lobes only (line 435: "anisotropy changes D/Vis of both specular lobes"): the transmitted lobe is a diffuse lobe and is unaffected.
- Coat above sheen above the base (line 440), unchanged from ticket 19.
Engine: `shadeSurface` thin branch keeps its statements for a thin-only material and takes new functions `thinLayerLight`, `sunThinLayer`, `thinSheenBackScale` (material_shading.h)
and `bsdfThinTransmittedSpec` (bsdf.h) when the material also carries ior/specular, sheen, anisotropy or iridescence (the diffuse base scale 1 - w is the existing `diffuseScale`
argument of `evaluateLightSpec` / `evaluateLightAniso`). Reference: refusal lifted (`check_combination`), `declared_extension_diffuse_weight` feeds the declared thin environment.
Archive/class table: new fixture scene `combo-thin-lobes` (classes 129, 132, 136, 144 and the union), manifest re-exported, archive updated with `tools/material_class_archive.py`.
Result (round 2, `build/f15-check/tex11`, `tex12`, `tex17`; the thin + coat, thin-only and sheen keys stay GREEN with the same binary):
- `combo-thin-lobes-sun` (2 x 4 grid: thin weight 1 white / 0.6 warm under sheen 0.4, anisotropy 1 at pi/4, a film 1.3 / 400 nm, ior 1.8 / specular 0.7; sun from the side so
  the left half of every sphere is front-lit and the right half back-lit): GREEN, p99 <= 0.39 % (sheen column), <= 0.02 % elsewhere, max outside mask <= 0.44 %. Seven controls
  FAIL: transmission-ignored, sheen-ignored, aniso-ignored, iridescence-ignored, specular-ignored, transmission-default-fresnel (the engine before this change for
  ior / film), transmission-sheen-unscaled (the engine before this change for sheen).
- `combo-thin-lobes-env@resolve` (declared model): GREEN, worst sphere mean 0.022 %, p99 0.034 %; six declared controls FAIL. The exact-environment twin does NOT exist:
  the per-ior transmitted-albedo tables of the exact environment are not built, so the key is gated by the declared model only (`--no-exact-environment`, like iridescence +
  anisotropy was). OPEN: exact-environment tables per ior / film.
- First run (`tex10`/`tex11`): sheen column 28 % off. Cause: N.L = 0 behind the surface is not resolved by the engine's sheen table (flat below N.L 1e-3 at E 0.7695) while the exact
  albedo falls to 0.678 at N.L 1e-4; the back-light albedo is now read at N.L = 1e-3 in both (`THIN_BACK_NDOTL`, `thinSheenBackScale`).
- Archive / class table: +26 descriptors in all (22 compute, 3 render, 1 tile): 8 class resolve pipelines (classes 129, 132, 136, 144) through `tools/material_class_archive.py` (harvest `tools/f15_permutations.py harvest`) and 18 constant-22
  descriptors (`.scratch/f15/harvest/tex-3.json`); `shaders/material_classes/material.json` re-exported; budget 46 / 64 archived permutations.
- Touched engine functions (shaders/material_shading.h: new `thinSheenBackScale`, `thinLayerLight`, `sunThinLayer`, and in `shadeSurface` ONLY the thin branch `if (thinResolved) {...}`;
  shaders/materials/bsdf.h: new `bsdfThinTransmittedSpec`). `bsdfThinTransmitted` and every other statement are untouched; a thin-only material still takes the old statements.
  Featureless: the five F13 exactness PFMs are byte-identical (`build/tex-exact4`).

### 2026-11-15 (3) — exact environment per ior / per film (agent `tex`, round 3)

- `thin_transmission_table_for(f0, weight, film)` (tools/f15_reference.py): T(mu) = integral over the back hemisphere of W(V.H) |N.L| / pi with W the diffuse weight of the reflected lobe
  (extension Fresnel, or (1 - s) extension + s film Fresnel at V.H); `environment_radiance` adds (1 - m) (w color T - (1 - reflect scale) base Dint) to the thin-free radiance (Dint = the material's
  diffuse integral). The neutral dielectric still returns the original table bit for bit. Unit test: the generic quadrature reproduces the default table to 3e-4, ior 2.4 lowers T by > 1e-3,
  a film changes it by > 5e-3. New key `combo-thin-lobes-env` (exact environment, controls transmission-ignored, sheen-ignored, aniso-ignored, iridescence-ignored, specular-ignored,
  transmission-default-fresnel): the capture and gate were NOT run (battery hold): UNVERIFIED on the GPU.
Result (round 3, `build/f15-check/tex22`): `combo-thin-lobes-env` (EXACT environment with the per-ior / per-film tables) GREEN: worst sphere mean 4.94 % (ior 1.8 / specular 0.7 cell) of the 5 %
bound, the other spheres 1.1 - 4.1 %; controls transmission-ignored, sheen-ignored, aniso-ignored, transmission-default-fresnel FAIL. Redesigns (thresholds untouched): the film column had
declared-vs-exact 5.3 % at index 1.3 / 400 nm (measured 5.04 % on the engine, reference prediction 5.3 %), so the film is index 1.5 / 400 nm (2.8 %); `specular-ignored` and `iridescence-ignored` move the exact
environment sphere mean by less than 5 % and stay on the sun and declared keys. The sun and declared keys stay GREEN with the index-1.5 film (`combo-thin-lobes-sun`, `-env@resolve`, all
controls fail). The environment margin of the ior cell (4.94 %) is thin: the declared transmitted ratio fit `1 - 0.16 N.V` was derived for the default dielectric.

### 2026-11-15 (4) — the declared transmitted term fitted per ior / per film (agent `tex`, round 4; replaces the film-index redesign of round 3)

- The film 1.3 / 400 nm cell is back (next to 1.5 / 400 nm and ior 1.8 / 0.7): it measured 5.04 % (exact environment) and was a real engine miss, not a fixture defect. Cause measured with the reference:
  replacing ONLY the declared transmitted term by the exact table T gives 1.1 % (film 1.3), 0.9 % (film 1.5) and 0.0 % (ior 1.8), against 5.3 / 2.8 / 5.2 % with the dielectric-only ratio `1 - 0.16 N.V`: the whole
  gap is the transmitted ratio (T / (1 - max F at N.V) is 0.82 .. 2.3 for the film, not 0.86 .. 0.98).
- Engine fix (`bsdfThinEnvAlbedo`, `bsdfThinEnvWeightFit`, `bsdfThinEnvIsDefault` in shaders/materials/bsdf.h; used in the thin environment of `shadeSurface` and in `reflection_composite_lobe` only when the
  material is NOT the default dielectric: `useSpec && !bsdfThinEnvIsDefault`): T(N.V) = sum over 4 V.H nodes {0.15, 0.4, 0.65, 0.9} of a_k(N.V) W(vh_k), W = the diffuse weight `1 - max3 F` of
  `bsdfSpecDiffuseIrid` (extension Fresnel or film), a_k cubic in N.V fitted offline on 96 films and 24 dielectrics (`THIN_ENV_NODES`, `THIN_ENV_POLY` in tools/f15_reference.py, the same literals in the shader).
  Held-out error of T (film 1.3 / 400, 1.5 / 400, 1.4 / 600, 1.7 / 250, ior 1.8, ior 2.4, default): <= 0.13 % sphere-weighted. The coat / sheen / transmission scalings carry over as `envDiffuse / envDiffuseBase`
  (a new local, the diffuse weight before those scalings). Default-dielectric thin materials keep the old statements, the featureless path is untouched.
- Result (`build/f15-check/tex23`): `combo-thin-lobes-env` (exact environment, unchanged 5 % bound) GREEN with worst sphere mean 1.95 % (was 5.04 % / 4.94 %); controls transmission-ignored, sheen-ignored,
  aniso-ignored, iridescence-ignored (now discriminating again with the 1.3 film), transmission-default-fresnel FAIL; `combo-thin-lobes-sun` and `-env@resolve` (worst mean 0.026 %) GREEN, all controls fail;
  `subsurface-thin-env(-resolve)`, `combo-thin-coat-env(@resolve)`, `texture-thin-env-resolve`, `furnace-thin-subsurface(@resolve)`, `combo-irid-aniso-env@resolve`, `iridescence-grid-env-resolve`,
  `specular-grid-env-resolve` stay GREEN. The reference mirrors the fit per pixel (`thin_env_albedo`, `thin_env_is_default`): a first version decided "default" over the whole image and broke the sheen / aniso
  columns by 1.7 % (found by the gate, fixed). Unit test `test_declared_thin_albedo_fit_tracks_the_exact_table_and_keeps_the_radiance_within_the_bound` (fit < 2 %, the old fit > 4 %).
- Featureless: the five F13 exactness PFMs are byte-identical; visual_check with `EXTRA_ARGS=--pipeline-sync` on this tree: all 8 benches 0 px (`build/visual3.log`).
2026-11-15 — back-lit thin receiver and the RT sun visibility (agent "transport"; ticket 16 residue "CSM/RT sun shadows of a back-lit thin receiver use the standard sun visibility").

Analysis (code): the RT sun kernel (`shadowSunRt`, shaders/shadows.metal) flips the surface normal to the light side before `rtOffsetRay`, so the shadow ray of a single-sided thin
receiver starts on the SUN side and cannot hit its own carrier; the DI visibility (`diEndpointTransmittance`) does the same for local lights (and the existing t16 keys
`subsurface-thin-point@restir|@clustered` already gate it at 0.04 % on quad carriers with RT visibility rays).  The CSM path (depth bias along the camera-facing normal, "shadow depth stays
two-sided") is the one that can self-shadow a back-lit thin receiver: it belongs to the "thinwall" agent's CSM work, not to this item.
The corpus rig puts the sun on the camera side, so a sun-lit back-lit receiver cannot be built from a generated glTF; `subsurface-thin-sun` uses SPHERES (their geometric terminator is a real
shadow: RT and CSM both darken the camera side to 0.17 / 0.14 of the unshadowed image, which proves nothing about a thin quad).  Fixture written instead (`tools/f15_transport_check.py thin-backlit`):
a thin diffuse-transmission quad lit only by a sampled emitter BEHIND it (ReSTIR DI): sum(D, `--rt on`) / sum(D, `--rt off`) in [0.95, 1.05] (D = capture - capture with the emitter off), control = an opaque
occluder between emitter and receiver must take the ratio below 0.45.  NOT RUN: the GPU batch was stopped by the owner's battery rule (27 % below 40 %); the fixture and its pure-Python
geometry tests are in the tree.  Status: PARTIAL - no code change is justified without a failing fixture; the code reading says RT is already correct.  Command for the orchestrator:
`python3 -I tools/f15_transport_check.py thin-backlit --binary <phosphor> --out DIR` (takes the GPU lock per capture; 4 captures).

2026-11-16 — GPU evidence on the integrated tip (branch f15c-transport-3 = 06dd3a7 + 3 commits; binary linked by hand because the pipeline archive step fails in this build tree, see below).

- Glass weights (ticket 13), `tools/f15_transport_check.py glass-weights`: rough ior-2 glass sphere (roughness 0.2, alpha 0.04), `--capture-linear-signal specular`, 1 sample per pixel, denoiser off, 4 seeds,
  1024x768, environment 0.5; counted pixels whose raw radiance exceeds 0.5 x 1.2.  Fix: 0 / 0 / 0 / 0 (peak 0.5002 = the environment, 180 729 lit pixels per frame).  Tip binary (control): 1302 / 1275 / 1289 / 1303
  pixels, peak 7.82 (a weight of 15.6).  Negative control, same counter at half the environment: 180 620 pixels on the fixed build (the statistic sees the sphere).
- Ticket 16, `thin-backlit`: the thin diffuse-transmission quad lit only by an emitter behind it (ReSTIR DI + RT visibility) is lit 3.83x its unlit value; with an opaque occluder between emitter and quad the
  light drops to exactly 0.000 of it: the RT visibility ray does not self-occlude the back-lit thin receiver (DI path; the sun cannot back-light a camera-visible receiver in the corpus rig, RT sun origin is on the light
  side by code).  There is no visibility-off arm (`--rt off` is refused with restir DI): the occluder is the control.  CSM self-shadowing stays with the thinwall work.
- Exactness: the 5 F13 captures (sponza-d2, cornell-restir, cornell-clustered-ssr, manylights-restir, grid-sky) with the final binary are `cmp`-identical to the same 5 captured with the orchestrator's tip binary.
  `visual_check` (EXTRA_ARGS=--pipeline-sync, BENCH8_ARGS="--instances 100000", references build/reference-f15-appB): all 8 benches 0 pixels, max delta 0.
- Archive: `metal-tt` ends with "applegpu-nt: Internal compiler error ... plugin wrapper-nt.dylib cannot emit pipeline" (no descriptor named) on every `phosphor-archive.metallib` step in this worktree: with my changes (3 runs),
  with all my shader changes reverted to the tip (2 runs, at fa6786d and at 06dd3a7), identical pipelines.mtl4-json to the orchestrator's (cmp).  I linked `phosphor` from `ninja -t commands` without the archive and ran with
  "archive hits 0, unavailable 136" (every pipeline compiled at start, `--pipeline-sync`): fallbacks 0.  The archive ICE is therefore environmental to this tree, not caused by this work; the orchestrator's archive build of the same content succeeds.
