# 11: F15.3 reference method, fixtures and threshold policy

Type: grilling
Status: resolved
Blocked by: 01, 02, 06

## Question

Which fixtures (per lobe, critical combinations, furnace, extreme roughness,
Track Studio case, Khronos corpus), which frozen light rig/camera/exposure, which
independent reference per lobe, which metrics and how thresholds are fixed
before the comparison (METHOD), and which negative controls must fail?

## Answer

Fixtures (a new material-validation bench with named scenarios, like the
lighting/reflection scenarios of bench 6):
1. Per-lobe parameter grids on texture-free spheres (and a plane for
   anisotropy), each lobe alone, then the critical combinations: clearcoat +
   sheen (no Khronos sample has it), iridescence on metal and dielectric,
   anisotropy rotation, transmission × roughness × ior, thin vs volume,
   subsurface radius sweep.
2. White furnace: constant environment of radiance 1, no sun, albedo 1, every
   lobe and combination.
3. Extreme roughness 0 and 1 (Phosphor clamps roughness to 0.04: kept and
   declared as an approximation of roughness 0).
4. Track Studio case: ABeautifulGame (+ subsurface override file) under a
   back-light rig (pawns) and a side-light rig (kings/queens), plus parameter
   sweeps (transmission 0→1, tint, subsurface weight 0→1, scale ×0.5/×2).
5. Khronos Tier A/B corpus under a declared Phosphor light rig (two
   directional lights + constant environment), qualitative only.

Light rig, camera, exposure and resolution frozen per scenario; linear (PFM)
captures; snapshots exported with `exportOfflineReference`.

References: per-lobe Python BSDF written from the specification by the verifier
(direct light + exact environment quadrature, analytic sphere intersection;
Mitsuba 3.9.1 only for ray casting on meshes), Mitsuba's rough dielectric and
volumetric path tracing for transmission and subsurface ground truth.

Thresholds (fixed now, before any comparison):
- Direct-light-only grids: per-pixel relative error p99 ≤ 1 %, max ≤ 3 %
  (half-precision textures and fp32), ROI = lit sphere pixels.
- Environment grids: mean relative error ≤ 5 % per sphere (split-sum fit
  declared); the per-sphere error table is published.
- Furnace: no pixel above 1.0 + 0.5 % for any lobe or combination; energy loss
  per roughness recorded, not gated.
- Transmission vs Mitsuba ground truth on the glass mask (static accumulation):
  mean relative luminance error ≤ 5 %, p95 ≤ 15 %.
- Subsurface vs random-walk ground truth on the piece mask: mean relative
  luminance error ≤ 10 %, p95 ≤ 25 %.
- Parameter sweeps: monotonic response of the declared mask statistic at every
  step (no reversal beyond 1 % noise).
- Discrimination rule: for every gated fixture the negative control (lobe
  disabled, or wrong parameter) must exceed its bound; a fixture whose control
  passes is redesigned, never the threshold.
- Khronos corpus: no numeric gate; side-by-side page with per-feature defects
  and the explained Sample Renderer deviations; parity is never ticked globally.

## Review amendments (2026-11-13, fixed before any comparison)

- Relative errors use an absolute floor: `|a − b| / max(b, 0.05 · mean_ROI(b))`,
  reported as mean and p95 over the ROI (aligned with F13's D1 absolute-floor
  radiometry), for every gated fixture.
- Sponza D2 band: the median of three A/B/A GPU p95 deltas ≤ 0.10 ms, and the
  absolute GPU p95 ≤ 16.67 ms.
- Gating policy for transmission: gated fixtures have nothing seen through glass
  lit through curved glass; planar interfaces (pane, flat shallow water, sun
  within 30° of the normal) are gated against Mitsuba now that shadows are
  attenuated; curved-glass shadows are qualitative (caustics excluded).
- Subsurface: screen-space diffusion is gated on side-lit fixtures only; the
  back-lit thick case is measured and reported; Mitsuba's boundary uses the
  engine's material roughness.
- The reference models the declared roughness clamp (0.04) and the engine's
  sun-disk quadrature; the direct-light "max ≤ 3 %" bound applies outside
  pixels where the quadrature limits narrow highlights (mask declared with the
  fixture).
- Before ticket 03, the base fixtures at intermediate metallic and the furnace
  are expected to fail (test-first red state).

- ROI amendment from the first GPU run (2026-11-13, before any gate was
  relaxed or any engine change): the direct ROI also excludes the shadow
  terminator band (lit pixels with an unlit 8-neighbour of the same sphere),
  a shading discontinuity like the silhouette. Evidence: one pixel of the
  metallic-1 row (sphere 31, pixel 644,98) was lit in the reference (0.096)
  and unlit in the engine, an isolated N·L sign flip from normal
  interpolation; thresholds unchanged. After the rule the metallic 0 and 1
  rows pass with max 0.69 % / 0.43 % and the intermediate rows still fail.
- Furnace amendment from the first green run (2026-11-13, ticket 03): the
  per-pixel bound (no pixel above 1.005) applies to a deterministic evaluation,
  now `furnace-resolve` (`--reflections off`: the resolve computes the declared
  environment terms itself; measured max 1.00000). `furnace-base` runs the
  stochastic F13 specular transport, whose per-pixel maxima are estimator
  noise: a pure-metal sphere, which cannot gain energy, reaches 1.013 / 1.017 /
  1.019 with seeds 1/2/3 while its mean is 0.9997, and only 6 of ~32 pixels
  above 1.005 recur across the three seeds. Its gate is the per-sphere mean
  (energy in expectation) at the same bound 1.005 (measured max 1.00001); the
  per-pixel maximum is reported. The bound value is unchanged.
- DI amendment (2026-11-13, ticket 07 harness): the direct gate of the point-light
  grid applies to the DI estimator, captured unfiltered (`--lighting-denoise
  off`). With the custom denoiser the DI image of one delta light deviates from
  the base model by up to 18 % per pixel on the metal row (sharp highlights
  blurred). Unfiltered, ReSTIR and clustered DI equal the base model at p99
  ≤ 0.07 % on every row; the denoiser belongs to ticket 08. RT-visibility
  scenarios use a 2-pixel terminator band. Evidence: unfiltered, 14 of 15 668
  lit-ROI pixels were black in the engine and lit in the reference, all exactly
  2 px from the shadow boundary. The engine's shadow-ray offset on the
  tessellated facets moves the terminator. Thresholds are unchanged.
