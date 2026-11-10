# DDGI bounds and world-space history, 2026-11-10

This increment is not F13/F14 closure. It keeps all acceptance limits and
the 16x8x16 grid / 64 rays per probe workload.

## Changes retained on the development branch

Probe fitting now uses the world-space positions of indexed geometry instead
of the union of per-mesh spheres. A long, thin wall no longer expands the
vertical fit by its horizontal radius. The existing conservative motion
envelope remains for moving roots. Lighting already rejects the F9-only
deformation diagnostic, whose raster bounds are not published.

DDGI world-space atlases now survive camera cuts and pixel-extent changes.
Explicit resets and changed scene/geometry/material content still reset them.
The atmosphere producer separately attests continuous celestial clock updates.
Only DDGI may keep its atlas across those updates, and only while scene,
geometry, material, RT geometry and sampled-local-light identities remain
unchanged. Local light edits, discontinuities and explicit resets do not gain
this exception. Per-pixel reservoir histories retain their existing epochs.
The sun/environment radiance epoch still changes; no stale epoch is reported
as current. Pure unit checks cover every hard-reset identity and the explicit
reset flag.

The physical irradiance unit control of the independent renderer measured
(23.7736,22.0307,19.6985), against analytic (23.7695,22.0243,19.6888).
Maximum relative difference is 0.0493%. That unoccluded control checks units,
not the complete scene's transport or material filtering.

## Measured quality limits

The fixed frame119 conditional diagnostic uses the converged independent
transport estimate with the same primary diffuse material factors. It does
not certify independent primary material filtering.

Relative RMSE against that diagnostic:
- Before: 28.136.
- Indexed-geometry fit alone: 0.988.
- Fit plus world-space camera/celestial history: 1.251.

The combined change removes most of the prior excess, but the remaining
error is still unacceptable. The fit-only image is strongly underlit, so its
near-unity relative error must not be confused with a correct image.

The camera-cut control uses a frozen sun at the matching endpoint hour12.75.
Sky and irradiance inputs agree; direction differences are about 1.5e-8.
Keeping the world atlas changes its result from the reset one-bounce estimate
toward the accumulated estimate. This is a scoped history-domain check, not
a replacement for the dynamic quality corpus.

Rejected experiments remain only as build artifacts:
- Hard front-hemisphere probe weighting increases relative RMSE to 1.606.
- Shifting both probe latitude and azimuth increases this sample's error to
  1.655; it was not adopted from a single-frame sampling argument.
- Exact receiver-to-probe visibility rays reduce the error to 0.983 but leave
  unsupported/zero regions and add up to eight rays per receiver. Not adopted.
- Existing ReSTIR GI at this camera-cut endpoint yields relative RMSE1.076.
  It is not promoted as a solution.
- A Mitsuba volpath control is consistent in scale but still has 7.15% seed
  disagreement at65,536 spp, above2%. No new reference is admitted.

## Native HDR and ray-edge controls

The same authored 96-frame public-SDK HDR reduction was ported to the public
Metal3 MTLFXTemporalDenoisedScaler API. Its ordinary control passes at
0.390625% error; wide HDR still fails at5.17578125%, exactly as Metal4.
Actual SDK-packed input is checked. Changing command-encoding APIs is therefore
not an HDR fix. No output fitting or threshold adjustment is applied.

The frozen F9 edge witness was also tested without the triangle-only
intersector hint and with PreferFastIntersection build flags. The edge still
misses in both cases, while interior controls hit. Neither flag change is
adopted. F9 remains58/60 from the previous qualified residual correction.

## Validation and performance

All nine CTest groups and13 selected functional controls pass.
Five new dynamic Sponza captures pass GPU/API validation.
The 300-frame time-jump, all-volumes-moving, static-history positive and
history-corruption negative controls pass their functional/numerical checks.
A first attempt to launch the final GPU controls was rejected by the serial
GPU lock while the benchmark still owned it; that run is not counted.
Fresh output directories were used for the complete successful run.

The first timing series overlapped TerrainV1 and is rejected. After that app
exited, a second series failed all three5% drift checks. A third series
completed every256-frame measurement and has valid baseline drifts of
0.0014%,1.34%,0.71%, but the candidate GPU p50 is slower by8.90%,9.07%,11.45%.
Candidate p95 is70.7258,65.5469,74.2828ms in that environment. Zero steady
allocations and AOT availability checks remain enforced.

Even the unchanged, historically frozen binary measured86.93ms p95 in a
separate control, versus its previous quiet result near23ms. The cause of
the session-wide timing shift is not established. This does not turn the
candidate's measured relative slowdown into a pass or a speed improvement.
The16.67ms budget remains failed.

## Evidence

Raw artifacts:
- build/f13-f14-tight-grid: code trials, indexed fit/camera/sun captures,
  unit/functional/reset checks, measured timing and contamination receipts.
- build/f13-f14-tight-grid-idle-perf: complete valid-drift comparison reports.
- build/f13-f14-probe-visibility: rejected exact-visibility implementation.
- build/f13-f14-restir-quality-control: existing mode comparison.
- build/f13-f14-volpath-control: independent transport cross-check.
- build/f13-f14-sdk-hdr-metal3: public Metal3 source, JSONL and stderr.
- build/f13-f14-f9-edge-flags: unchanged-ray native controls.

No merge, push or F15/F16 advancement is justified by these results.
