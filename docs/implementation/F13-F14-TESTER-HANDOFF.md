# F13/F14 tester corpus — WRITTEN / NOT EXECUTED

New files only in this source package:

- src/testbench/reflection_validation.h/cpp and test_reflection_validation.cpp.
- tools/temporal_light_metrics.py and test_temporal_light_metrics.py.
- tools/f13_f14_check.py.
- tools/volume_snapshot_check.py and written test_volume_snapshot_check.py.
- Genuine SDK fixture/runner package in F13-SDK-FIXTURE-HANDOFF.md.

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
resize/reset, internal unshadowed probe capture with RT OFF, external six-PFM cooked probe use,
atmosphere zenith/horizon/space/time jump, fog, full-rate versus reconstructed
clouds and all volumes moving. Actual schema10 lighting fields and LIGHTING
check lines must agree with requested/effective controls. RT report absence is
the documented not-instantiated schema10 state; a present report must agree.
Internal capture uses --reflections probes --reflection-capture-probe with NO
cooked input path, and must report actual-static-raster-unshadowed. It does NOT
export PFMs. The separate cooked case requires tester-authored --probe-dir input
with EXACT px/nx/py/ny/pz/nz.pfm names and64x64 faces; no surrogate is generated.
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
clip mapping with the exact frozen manifest SHA. Optional --numeric-evidence
still compares separately supplied independent numbers without relaxing its
existing gates. Nonfinite/missing/stale samples fail. Missing converged quality
clips and actual SDK/lifetime results remain evidence PENDING, rather than
being described as absent source hooks.

Supported frozen negative commands now include --debug-reflection-corrupt
history/motion/normal, --debug-lighting-corrupt pdf/light/overflow in ReSTIR DI,
and --debug-volume-corrupt units/history/light/lut with their actual required
paths. Expected raw renderer exit AND EXIT marker are1, plus positive actual
LIGHTING failure counts/FAIL lines; a GPU crash/validation error cannot pass.
--debug-history-corrupt is the legacy F8 pose overwrite and safe F13 rejection
stimulus (requires --debug-visibility); --debug-motion-scale affects ordinary F8
temporal only. Neither is mislabeled as a guaranteed F13 nonzero-exit control.

All F14 cases now request their actual native --volume-oracle directory. Before
each renderer process, the runner writes provenance.json with frozen source,
binary, metallib and manifest hashes, and per-case timeout. It automatically
reads immutable phosphor.volume-oracle.v1 snapshots; an external numeric-evidence
file is unnecessary for the implemented LUT, solar and homogeneous readbacks.
Full expected/submitted atmosphere96-word and fog112-word u32 parameter blocks,
actual shader generation/source-at-prepare hash, device counters, complete RGB+T
rows and exact expected/produced epochs are required. Positive snapshots must
match every recomputed gate. Negatives require an armed control and a real failed
numeric/epoch/device-counter result; an unexercised request does not pass.

Native sparse gates are separately frozen from VolumeOracleSettings BEFORE any
run: transmittance absolute0.02/relative0.05, multiple/sky0.02/0.25,
fog2e-5/2e-4, solar0.01/2e-4. These exploratory sparse approximations do not
replace retained image, external-number or F8 gates, and do not certify a full
transport solution. The bridge recomputes each component comparison rather
than accepting its passed flag. It independently reconstructs metre fog prefix
distance from the actual camera/grid ABI and checks Beer-Lambert/source integral;
solar uses E/(pi*sin(radius)^2) for toward and zero for away/tangent. Declared
noon/space cases require actual solar output above half range. Snapshots at other
clock times may physically have smaller solar energy without an arbitrary gate.

Real remaining tester evidence:

- Actual runs of the implemented same-frame quadrature, homogeneous and solar
  readback hooks, including armed real negatives and complete provenance.
- Actual SDK unit/scaling/channel/resize/view/cut/reload fixtures after the owned
  gateway change; optional existing process-at-exit leaks diagnostics. Submitted
  retirement counters alone do not certify final destruction.
- Frozen independent converged quality/reference clips and all retained gates.

corpus_complete and phase_accepted remain false while that evidence is pending.
No roadmap tick or F13/F14 acceptance follows from source or a smoke pass.
Hard boundary: STOP_AFTER_F14. No F15+ runner/implementation was introduced.

Final root integration supersedes the earlier missing-source-hook list. All
listed F13 guide/history, F14 LUT/history/fog/solar and native SDK fixture hooks
are now concrete source paths; consult root HANDOFF.md for files/APIs and commit
boundaries. Their actual execution and quality/lifetime evidence remain pending.
New native F14 bridge consumes phosphor.volume-oracle.v1/cases directly. SDK
fixtures have their separate frozen runner. ID8 indirect-diffuse-filtered exports
actual custom-selected GI E as Lo, while raw1 remains unchanged. No threshold was
relaxed and no writer test/build/GPU proof was run.
