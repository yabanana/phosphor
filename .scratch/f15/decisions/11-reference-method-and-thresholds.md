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
