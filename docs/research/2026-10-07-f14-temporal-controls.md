# F14 clock discontinuities and eligible-history controls

Source correction and CPU fixture validation only. No renderer, Metal
compilation, C++ build or GPU run was performed for this change. F14 temporal
quality and phase acceptance remain unverified.

## Clock ownership

The audit found an existing real time jump in `Engine::update`: every N
presented frames it adds another half configured day before calling
`AtmospherePasses::prepareLighting`. The backend's `explicitJump` boolean
only marks the discontinuity for history resets. Adding a second offset in
the backend would double the jump.

The existing formula now lives in the portable, tested helper
`atmosphereClockSeconds(sceneSeconds, frame, dayLength, jumpEvery, freeze)`:

`effective = (freeze ? 0 : sceneSeconds) + floor(frame / N) * dayLength / 2`.

With N=0 the offset is zero. With both controls off, normal clock behavior
is preserved. `--atmo-freeze-clock` holds the base time at zero; `--start-hour`
still chooses the celestial hour, and explicit jumps still advance the clock.
The freeze also stops clock-driven cloud advection, consistent with the shared
clock contract. It does not pause application camera, objects or materials.

Every volume diagnostic snapshot now records the exact host double seconds,
delta seconds, u64 clock epoch, reset flag, day fraction and freeze flag.
These accompany the full uploaded GPU parameter words. The CPU snapshot
validator checks the float time fields against the host values when exercising
temporal controls. No GPU ABI layout changes.

## Continuous-light limitation

The production `VolumeSignal` includes the environment revision, whose exact
key includes current sun/moon directions and irradiances. A continuously
moving clock therefore resets fog/cloud history each frame. This remains the
declared conservative policy; a moving-sun run cannot certify history reuse.
Admitting approximate temporal reuse under gradually changing illumination
needs measured error/reprojection work, rather than omitting the light revision.

The prior history negative bypassed this policy by setting HISTORY_VALID
after `HistoryRegistry::begin` had reset it. The control now requires genuinely
eligible history after all ordinary extent, camera, geometry, material,
pipeline, light and clock checks. It never sets HISTORY_VALID on a reset.
When eligibility changes, the graph's diagnostic injection shape is rebuilt.

The original negative path still mutates actual GPU history identity and
intentionally exercises its invalid-consumption detector. It runs only on
eligible history: no synthetic failure bit replaces a real history read. The
runner requires eligible frozen-clock reuse, a genuinely armed control and
positive invalid-history/read counters. A never-armed run cannot pass.

## Matched positive and negative

`tools/f13_f14_check.py` now defines `volume-history-static-positive` and
`negative-f14-history` on the static `ao-cavity` fixture with the celestial
clock explicitly frozen. The prior `disocclusion` fixture moves its object
and camera and is unsuitable for isolating an eligible-history negative.
The homogeneous analytical fog fixture deliberately disables history and is
rejected in combination with this negative control.

The positive must show actual naturally eligible history reuse with no
numerical/history errors. The negative must additionally show actual foreign
history consumption and detection. Counters currently aggregate fog/cloud
reuse: this pair establishes that at least one eligible volume signal was
exercised. Per-signal reconstruction quality, wind advection, per-view motion,
cuts, resize and mixed-media behavior retain their separate acceptance work.

The portable clock test covers unchanged production time, cumulative half-day
jumps, freeze+jump combinations, real clock/sky epochs and HistoryRegistry
reset/reuse transitions. The Python snapshot fixture rejects armed histories
that override a reset, missing eligibility and mismatched host/GPU times.
Five Python snapshot tests and the runner's static/frozen case assertions
were executed successfully. C++ tests are written for the root tester.
