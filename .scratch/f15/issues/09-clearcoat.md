# 09: Clearcoat lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_clearcoat` (factor, roughness, coat normal) per decision 06: coat GGX with F0 0.04 over the base and emission (sun-disk quadrature covers the coat lobe), coat term in the environment, coat-or-base lobe choice in the transport, coat lobe id/roughness to the denoiser, emitter ownership by coat roughness.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Clearcoat grids (factor × roughness, coat normal) pass decision-11 thresholds against the independent reference; furnace bound holds
- [ ] Rough base + smooth coat: the coat reflection survives the denoiser and matches the reference within the transport bound
- [ ] Emitter ownership: an emissive-panel fixture shows no double count between DI and the coat lobe's transport
- [ ] Negative control (coat ignored, or coat F0 from the material ior as in the Sample Renderer) fails the bound
- [ ] ClearCoatTest and ClearCoatCarPaint side by side under the declared rig; Sample Renderer deviations listed; temporal clip within F13.6
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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
