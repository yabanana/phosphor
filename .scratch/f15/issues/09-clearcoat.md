# 09: Clearcoat lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_clearcoat` (factor, roughness, coat normal) per decision 06: coat GGX with F0 0.04 over the base and emission (sun-disk quadrature covers the coat lobe), coat term in the environment, coat-or-base lobe choice in the transport, coat lobe id/roughness to the denoiser, emitter ownership by coat roughness.

**Blocked by:** 07, 08

**Status:** resolved with carried items (2026-11-13; Khronos side-by-side delivered by ticket 22; coat-normal-texture numeric fixture still open: box 1 PARTIAL; coat-only union (mask 2) not archived)

- [ ] Clearcoat grids (factor × roughness, coat normal) pass decision-11 thresholds against the independent reference; furnace bound holds — coat normal = the shading normal on the grids; coat-normal TEXTURE fixtures → ticket 22 — PARTIAL (audit 2026-11-15): grids with coat normal = N met; the coat-normal TEXTURE numeric fixture is not built (ticket 22: qualitative only)
- [x] Rough base + smooth coat: the coat reflection survives the denoiser and matches the reference within the transport bound
- [x] Emitter ownership: an emissive-panel fixture shows no double count between DI and the coat lobe's transport
- [x] Negative control (coat ignored, or coat F0 from the material ior as in the Sample Renderer) fails the bound
- [x] ClearCoatTest and ClearCoatCarPaint side by side under the declared rig (ticket 22, 2026-11-14, `docs/research/2026-11-14-f15-khronos-corpus-report.md`, qualitative, no parity claim); Sample Renderer deviations listed; temporal clip within F13.6 — temporal clip done (coat cases of tools/f15_clip_check.py); the Khronos side-by-side → ticket 22
- [x] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

## Comments

2026-11-13 — test-first harness (independent reference, fixtures, runner keys); no engine change.
Commits `6487a0b` (reference), `4feed22` (fixtures, keys, tests), `d15d58d` (archive) and the commit with this note.

Reference (`tools/f15_reference.py`, written from the ratified text, research/khr-materials.md section 2):
- `F_c = 0.04 + 0.96 (1 - |V.Nc|)^5`, `f = (1 - c F_c) f_material + c F_c D Vis(a_c)`; the coat term has its own
  cosine max(Nc.L, 0), the base its own; Nc = the shading normal (fixtures have no coat normal texture; a coat
  texture in the snapshot makes the reference refuse, like the ior/specular texture rule); the material under the
  coat is the full ior/specular dielectric or metal. Clamps as the base: roughness in [0.04, 1], `a_c = max(r_c^2, 0.002)`.
- Sun disk: the coat lobe uses the declared disk rule with its own alpha, reflection axis and sample count; the
  `narrow` mask is the union over lobes; the material terms carry the pixel-wide `1 - c F_c`.
- Exact environment: `(1 - c F_c(mu)) L_material + c F_c(mu) E_coat(mu, a_c) L` (E_coat = A + B of the Fresnel-free
  D Vis quadrature). Declared environment (split sum, DFG table): `(1 - c F_c(mu)) declared_material + c F_c(mu) clamp(A_c + B_c, 0, 1)`.
- Controls: `coat-ignored` (c = 0), `coat-f0-from-ior` (Sample Renderer: coat F0 from the material ior),
  `coat-fresnel-vh` (F_c at V.H per light sample; in the exact environment only the coat term integrates the V.H
  Schlick weight), `coat-base-unattenuated`; declared twins `declared-coat-ignored`, `-f0-from-ior`,
  `-base-unattenuated` (no V.H in a split-sum model, so no declared fresnel-vh).
- `tools/test_f15_reference.py`: 56 tests OK (16 new: F_c hand values, layering formula, |V.Nc| independence of the
  light, c = 0 exact reduction in every term, own cosine, E_coat <= 1 at every mu/roughness and vs wi-space Monte
  Carlo, white metal under the coat <= 1, control directions, union mask and 20000-sample sun convergence, record
  words 7/8/9 and the texture words 10/11/12 refusal).

Fixtures (`src/testbench/material_validation.*`, `tests/test_material_validation.cpp`): six rows x five columns,
base roughness 0.6 (rough base under a smooth coat); rows = coat factor {0, 0.25, 0.5, 1}, coat 1 over a base of
ior 2.4 (added: with the ratified F0 = 0.04 the `coat-f0-from-ior` control is invisible at ior 1.5, so one row
needs another ior), metal under a full coat; columns = coat roughness {0, 0.05, 0.15, 0.4, 1}. Coat factor 0 with
no ior block carries no extension (neutral material). Scenarios: `clearcoat-grid-sun`, `-env-resolve`, `-env`,
`-point`, `furnace-clearcoat` (17 scenarios in all; ctest 16/16).

Runner keys (`tools/f15_check.py`, `tools/test_f15_check.py`: 41 tests OK):
- `clearcoat-grid-sun`, `clearcoat-grid-point` (legacy), `@restir`, `@clustered`: direct gate per (metallic, ior,
  coat) row; controls coat-ignored, coat-f0-from-ior, coat-fresnel-vh, coat-base-unattenuated.
- `clearcoat-grid-env-resolve`: declared gate (mean 0.5 %, p99 1 %); controls declared-coat-ignored, -f0-from-ior,
  -base-unattenuated.
- `clearcoat-grid-env` (SSR transport): environment gate (mean <= 5 % per sphere); controls coat-ignored,
  coat-f0-from-ior, coat-base-unattenuated.
- `furnace-clearcoat` (transport mean <= 1.005) and `furnace-clearcoat@resolve` (per-pixel <= 1.005 plus the
  declared gate); control coat-base-unattenuated (+ declared-coat-ignored, declared-coat-base-unattenuated).
- All gates the current engine can fail carry `pending: "ticket 09"` with `pending_matches` = coat-ignored (or its
  declared twin). New runner rule: `--expect green` also requires `pending_matches_met` (the engine image passes the
  gate against that control), and `pending_gate: "declared"` limits the pending state to the declared gate.

Measured red state (M5 Max, `build/f15-check/t09-red`, 1280x720, `--run --expect green`: GREEN, exit 0, 0 archive
misses in every capture). Gate (pending, must FAIL) and the same gate against the coat-ignored control (must PASS):
- clearcoat-grid-sun: FAIL, rows with a coat p99 0.26 / 0.44 / 0.63 / 0.56 (ior 2.4) / 0.75 (metal), coat-0 row p99 0.0007; vs coat-ignored: PASS (0 failing rows).
- clearcoat-grid-env-resolve: FAIL, worst sphere mean 0.284 (25 of 30 spheres fail); vs declared-coat-ignored: PASS (worst mean 0.0000, p99 0.0001).
- clearcoat-grid-env: FAIL, worst sphere mean 0.284 (22 spheres); vs coat-ignored: PASS (worst mean 0.028, bound 0.05).
- clearcoat-grid-point (legacy), @restir, @clustered: FAIL (same five coat rows; p99 0.19-0.83); vs coat-ignored: PASS in all three.
- furnace-clearcoat: gate PASS (sphere mean max 1.000, pixel max 1.0035), asserted green; furnace-clearcoat@resolve: engine-pixel gate PASS (1.0), declared gate FAIL (worst mean 0.039) and PASS vs declared-coat-ignored.
- Controls (all fail their gate, every key): sun/point rows failing: coat-ignored and the other three on every coat row,
  coat-f0-from-ior only on the ior 2.4 row; environment keys: declared-coat-ignored mean 0.284, f0-from-ior 0.292,
  base-unattenuated 0.072; furnace: coat-base-unattenuated pixel max 1.438, sphere mean 1.067.

Dropped or limited controls and why:
- `coat-fresnel-vh` is not on the environment/furnace keys: the exact quadrature would need the V.H Fresnel inside
  the base integral, and the split-sum declared model has no V.H; it guards the direct keys (sun, point).
- `coat-ignored` is not a furnace control: the furnace is an upper bound, which the coat-ignored image satisfies
  (a coat redistributes energy in a white furnace). The furnace controls are therefore the unattenuated base, which
  gains energy, and (resolve) the declared twins.
- `furnace-clearcoat` (transport) has no red state: the engine-pixel gates stay asserted green and protect the
  implementation against energy gain; the pending red state of the furnace lives in the declared gate of `@resolve`.

Interpretations of the specification: coat IOR fixed at 1.5 (F0 0.04); `|V.Nc|` (not V.H) for both the coat
reflectance and the base attenuation; the coat cosine is Nc.L, equal to N.L here; the base specular lobes of the
sun disk keep their own sample counts and are scaled by the pixel-wide attenuation; the diffuse of the sun is
evaluated at the disk centre and attenuated at the centre's `F_c`.

Still open for ticket 09 (not built by this harness): coat normal texture / `clearcoatNormalScale` fixtures (the
reference refuses coat textures; words 9 and 12 are decoded only), the emissive-panel fixture of the emitter
ownership criterion, the denoiser/lobe-id criterion (the sun and env keys cover the coat reflection over a rough
base only through the image gate), ClearCoatTest/ClearCoatCarPaint side by side, and the temporal clip (F13.6).
Archive: +19 descriptors with MATERIAL_FEATURES 3 (compute 330 -> 344, render 178 -> 182, tile 5 -> 6); a
specialisation by this ticket for other masks (e.g. coat-only scenes) needs its own targeted update.

2026-11-13 — engine side verified on M5. Code 251e231 (Sonnet agent,
reviewed), runner 027edbd, coat-emitter fixture cfaa7a3, clip cases 575d079.
Batch logs: `build/f15-exact/battery09.log`, `battery09b.log`,
`t09/clip2.log`.
- `tools/f15_check.py --run --expect green` (`build/f15-check/t09`): GREEN on
  all 25 keys. Every control fails: coat-ignored, coat-f0-from-ior,
  coat-fresnel-vh, coat-base-unattenuated and the declared ones.
  - `clearcoat-grid-sun`: worst row p99 0.071 %.
  - `-point`: 0.081 %; `@restir` and `@clustered`: 0.138 %.
  - `-env-resolve` (declared): worst mean 2.6e-5.
  - `-env` (transport): worst sphere mean 2.9 %, against 5 %.
  - `furnace-clearcoat`: 1.00001; `@resolve`: max pixel 1.0.
- Rough base under a smooth coat (`tools/f15_clip_check.py` coat cases,
  `build/f15-clip-t09b`): on the dielectric smooth-coat spheres, the custom
  denoiser against the 8 spp control gives p95 0.0095 / 0.0090 and flicker
  0.0091 / 0.0088. The coat reflection survives the denoiser.
  - Metal rows under the coat (p95 0.08, flicker 0.077) and the whole-grid
    flicker (0.0216) are diagnostic only: they are the 1 spp
    strong-specular noise level that ticket 08 also measured on base
    metals.
  - The per-frame lobe-key control does NOT discriminate on these spheres:
    its error is lower than the stable key's. Reported, not gated.
- Emitter ownership (`tools/f15_coat_emitter_check.py`, coat roughness 0.3
  and 0.5): MET. The ROI mean ratios against legacy-ssr are brute-ssr
  1.12 / 1.14, legacy-rt 1.03 / 1.00 and brute-rt 1.15 / 1.14. The
  double-count control (2.12) fails.
  - The tolerance is 0.20 because the uncoated `rough-emitter` twin already
    shows the same +8..16 % DI/SSR difference (F13 behaviour).
- `--debug-lobe-guide 16` on clearcoat-grid-env: `pixels 52356 extended 43228
  base 9128 unverified 0 | PASS`. Coat factor and roughness are compared bit
  for bit; the coat normal by angle.
- Shader validation: clearcoat-grid-env with RT reflections and ReSTIR exits
  clean, and so does the F13 mirror DI fixture.
- Scenes without features: F13 exact captures and visual_check 0 px.
  archive_check: every local scenario PASS. hitch_check OK. The transport
  and lobe-guide overflow checks PASS.
- Transport fixes made during the ticket:
  - emitter and RT ownership are drawn per lobe in proportion to the coat's
    share of the BSDF value in the sampled direction;
  - a sample in the extreme lobe tail (mixture pdf 0) contributes zero
    instead of an error.
- Lobe key: the denoiser key is the coat when the coat is significant (at
  least 3 % of the albedo weight), with edge-stopping roughness
  min(base, coat). Unit-tested on the CPU mirror.
- Support report: clearcoat approximated,
  `secondary_rt_hits_ignore_coat;di_emitters_uncoated` (decision 06).
- Carried:
  - ClearCoatTest / ClearCoatCarPaint side by side → ticket 22;
  - coat-normal-texture fixtures need textures in the reference → ticket 22;
  - the coat-only union (mask 2) is not archived because no fixture uses it.
- Process note: the first clip run failed `repeat_identical`. The
  orchestrator had rebuilt `build/release` while the runner was running,
  so the two repeats used different binaries. The rerun at a fixed binary
  passes. Rule: never rebuild a binary under a running verification.

### Ticket 22 closure (2026-11-14)

- Side by side done (`docs/research/2026-11-14-f15-khronos-corpus-report.md`). ClearCoatTest also exercises the coat factor texture, the coat roughness texture and the coat normal texture visually (qualitative); the NUMERIC coat-normal-texture fixture was not built (the reference refuses texture words): still open.
- Open defect C22-D3: isolated dark pixels on the silhouette of smooth spheres (ClearCoatCarPaint), present with the RT and SSR reflection paths, absent with reflections off.

2026-11-15 — closure audit (ticket 24):

- 1 grids + furnace: PARTIAL (t09 GREEN 25 keys, furnace 1.00001; coat normal = N only; texture fixture not built). Unticked.
- 2 coat survives denoiser: MET (f15_clip_check coat cases, 575d079; metal rows diagnostic only). Ticked.
- 3 emitter ownership: MET (cfaa7a3, ratios 1.00-1.15 vs tolerance 0.20, control 2.12 fails). Ticked.
- 4 negative controls: MET. Ticked.
- 5 side by side: MET qualitative (ticket 22 report; C22-D3 fixed in e9569ac). Ticked.
- 6 archive/hitch/O7: MET (+19 descriptors). Ticked.
Open: coat-normal-texture numeric fixture (ticket 22 closure); coat-only union (mask 2) not archived (no fixture uses it).

## F15 closure, loader-misc (2026-11-15): coat-only union (mask 2) archived and fixture

Open item: "coat-only union (mask 2) not archived (no fixture uses it)".  DONE.

- Fixture `clearcoat-grid-coat-only-sun` (bench scenario, no reference, no gate): `clearcoat-grid-sun` without the ior block of row 4, so every sphere carries the clearcoat block only and the scene's feature union is 2
  (clearcoat-grid-sun is 3).  tests/test_material_validation.cpp pins the union of both (negative control: the reference is 3) and the manifest `shaders/material_classes/material.json` exports the scene (union 2).
- Before the update the fixture capture reported 2 archive misses (tools/f15_coat_only_check.py, 1280x720, legacy + SSR); targeted update (rules-file recipe, commit 271d751, KEPT file
  `.scratch/f15/harvest/loader-misc-1.json`, +3 compute descriptors specialised with MATERIAL_FEATURES = 2; the coplanar fixture of ticket 06 shares the mask).  After the rebuild: 0 archive misses, 0 render-thread compiles
  (both the coat-only capture and its clearcoat-grid-sun reference).
- Image criterion (fixed from float rounding, see the docstring: the first version asked for bit equality and failed with a largest difference of 1.8e-7, the two specialisations are different compiled programs):
  every pixel outside row 4's band equals the reference pixel to within 1e-6 (0 pixels above, largest 1.79e-7) and inside the band 9 610 pixels differ by more than 1e-3 (largest 0.384: row 4 lost its ior 2.4 block,
  the negative control).  The coat model itself is unchanged: the scene is the sun gate's grid, only the specialisation differs.

### 2026-11-15 — texture closure (agent `tex`, branch f15c-tex-2)

- The coat-normal / factor / roughness texture fixture is built: `texture-coat-sun` (direct) and `texture-coat-env-resolve` (declared), 2 x 3
  spheres (dielectric and copper rows) binding {normal}, {factor, roughness} and all three slots. The reference reads the EXPORTED texels
  (level-0 bilinear, repeat, binary16, ROI without ambiguous-footprint pixels; `tools/f15_reference.py` "KHR texture fixtures") and
  applies the coat normal per pixel (T x + B y + N z with the engine frame). Thresholds are the existing ones (sun p99 <= 1 %, max <= 3 %;
  declared sphere mean <= 0.5 %, p99 <= 1 %), every sphere its own row.
- Result (`build/f15-check/tex2`, `tex3`): GREEN. Sun: worst p99 0.34 % (sphere 0), max outside mask 1.47 %; env-resolve: worst mean 0.048 %,
  p99 0.22 %. Controls, all FAIL as they must: texture-v-flipped, coat-normal-ignored, coat-normal-y-flipped, coat-factor-ignored,
  coat-roughness-ignored (sun and declared twins).
- Coat-only union (mask 2) archived: the first capture logged 2 archive misses (`light_cluster_shade_lobe`, `reflection_ssr_lobe`, constant 22 = 2).
  Targeted update, +2 compute descriptors, `.scratch/f15/harvest/tex-1.json` (harvested from the 10 texture scenarios, `--constant 22
  --skip-present`); the class manifest `shaders/material_classes/material.json` was re-exported for the 10 new scenes.
- Open: nothing in this ticket's list.
