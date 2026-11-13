# 16: Subsurface feature, thin mode (diffuse transmission)

Roadmap: F15.1, F15.3

**What to build:** The subsurface feature (OpenPBR parameters, ADR-0005) in its thin mode: a diffuse-transmission lobe for analytic lights on the back hemisphere, back-side environment and GI irradiance at −N; fed by `KHR_materials_diffuse_transmission` (RC), the draft `KHR_materials_scatter` thin mode and the material-description file.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Thin fixtures (backlit plane, factor and colour sweeps) pass decision-11 thresholds; furnace bound holds (energy moved, not added)
- [ ] Both glTF encodings and the description file produce the same description (doctest) and image
- [ ] Negative control (lobe on the front hemisphere) fails; support report labels RC/draft inputs
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames

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
