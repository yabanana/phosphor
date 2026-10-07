# F13/F14 tester corpus — WRITTEN / NOT EXECUTED

New files only in this source package:

- src/testbench/reflection_validation.h/cpp and test_reflection_validation.cpp.
- tools/temporal_light_metrics.py and test_temporal_light_metrics.py.
- tools/f13_f14_check.py.

Root registers ReflectionValidation under --reflection-scene NAME on bench6:
mirror, roughness, ao-cavity, probe-parallax, moving-light, disocclusion.
The fixture has actual meshes, exact texture-free material factors, complete
visible+caster roles, tracked transforms/material/light writes, WORLD metre
units, an emitter behind the camera, roughness0.04/0.1/0.25/0.5/0.9, cavity/
open controls, local color walls and deterministic camera/light scripts.
At fixed60Hz, emissive step t2s corresponds to frame119 and camera cut t4s to239.
The independent mirror geometry and tracked-step/cut unit tests are WRITTEN.

The runner defaults PLAN ONLY. Example commands for THE TESTER, not executed:

    python3 tools/f13_f14_check.py --binary build/release/phosphor --out <new-plan-dir>
    python3 tools/f13_f14_check.py --run --manifest <new-plan-dir>/frozen-manifest.json --binary build/release/phosphor --out <new-plan-dir>

--run is explicit and serial, uses run_checked raw exit/EXIT marker/validation
log evidence and a cooperating GPU lock. It never configures/builds, installs,
renders a surrogate reference or retries a failure. An existing frozen plan
cannot be mutated with --only/--roi-config; make a new plan to change selection.
The actual binary is hashed before the first case and between cases.

All commands use named --capture-linear-signal values. Root's numeric namespace:
HDR0, indirect1, direct2, shadow-mask3, shadow-position4, shadow-normal5,
F13 specular6, AO7. This package never hardcodes6/7 into CLI strings.
Specular/AO captures are raw signal diagnostics; combined HDR clips compare
actual custom/SDK output so the same raw signal cannot masquerade as filtering.

The manifest covers RT/SSR/probes/off reflection, GTAO/RTAO/off AO, raw/custom/
requested MetalFX, exact material/roughness and temporal cases, view counts/
resize/reset, unshadowed probe capture with RT OFF, six-PFM cooked probe use,
atmosphere zenith/horizon/space/time jump, fog, full-rate versus reconstructed
clouds and all volumes moving. Actual schema10 lighting fields and LIGHTING
check lines must agree with requested/effective controls. RT report absence is
the documented not-instantiated schema10 state; a present report must agree.
Missing factory must identify factory/gateway and select custom. Native SDK
selection alone does not certify its output units/lifetime/gateway ownership.

Metrics preserve LINEAR Float32 PFM and exact resolution/frame IDs; no gamma,
resize, temporal alignment fitting or automatic gate tuning. Every ROI scale
and threshold is frozen before execution. F8 numeric caps remain0.06p95/
0.15max/0.02flicker/0.25support-ghost/4frame recovery; overrides may only tighten
them. Linear HDR uses its explicit radiance scale, so the existing F8 SDR suite
still runs unchanged as a separate regression gate.

Independent equations: regional RMS/signed bias; flicker of error residual
after removing actual reference change; projection of residual onto the prior
reference image's change; attraction OUTSIDE the current spatial support
envelope; recovery requires consecutive stable frames, not one lucky sample.
Written closed-form tests use exact step/one-frame lag/corrupt cut/alternating
bias/NaN/mismatched resolution and exact Beer-Lambert homogeneous fog.

Eight-ray/full-rate images are explicitly controls, not independent or
converged references. --references accepts a separate validated/independent
clip mapping with the exact frozen manifest SHA; --numeric-evidence matches
manifest/binary and recomputes fog or compares supplied independent adaptive-
Simpson/Gauss/exact-clock numbers. Nonfinite/missing/stale samples fail.
Missing references remain PENDING. Remaining unimplemented negative/numerical
hooks are listed in every plan/report, never replaced with fake run commands.

Real remaining tester/root hooks:

- F13 foreign-view/history/normal/motion corruption CLI/readbacks.
- F14 stale/omitted LUT and foreign cloud/fog history controls.
- Same-frame LUT readback against independent quadrature.
- Actual homogeneous fog configuration heightFalloff0 and source/sigma/distance
  readback; ordinary --fog on alone is NOT that analytic fixture.
- Solar disk wide-HDR orientation/scalar probe and actual SDK preExposure
  output-unit/constant/impulse/lifetime fixtures after the owned gateway change.

corpus_complete and phase_accepted remain false while those controls are absent.
No roadmap tick or F13/F14 acceptance follows from source or a smoke pass.
Hard boundary: STOP_AFTER_F14. No F15+ runner/implementation was introduced.
