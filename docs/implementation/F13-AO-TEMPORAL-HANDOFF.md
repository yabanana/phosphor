# F13.6 minimum independent physical AO control — NON VERIFICATA

Branch `codex/f13-temporal-independent-review`, writer worktree
`/Users/danielsan/.codex/worktrees/f10-f12-development/phosphor`.
The aggregator checkout was read only. No configure/build, compiler/MSL, CPU/GPU
test, renderer or numerical oracle was executed. Check performed: git diff --check.
SDK/native radiometry/lifecycle and GI defaults are unchanged in this package.
Other F13.6 controls remain explicitly PENDING; no phase acceptance or F15/OPT.

Apply ONLY the assigned incrementals AFTER writer prewarm fixc42b814 (already
integrated by the aggregator), not the full old writer foundation:

1. `d43b6ac`: control plan published BEFORE source implementation.
2. `323bc12`: actual completed custom-AO history checkpoints (agentc67f3c49).
3. `7525d87`: physical fixture, capture9 and immutable per-frame sidecars.
4. `7100de5`: frozen analytic verifier/protocol/authored negatives (agent78ec528).
5. Final registration/identity/handoff commit follows this list.

IMPORTANT MERGE OWNERSHIP: preserve the aggregator's VisibilityRenderer
`composedColor_`/immutable resolve-output alias fix, F14 stamp read-before-write,
exact receiver corrections and all other owned fixes. Preserve its existing
`wide-emission` fixture and BLACK offcamera anchor atz20. This patch adds only
`ao-temporal-wall`; union scenario names during any conflict, never replace the
aggregator's complete reflection_validation.cpp or remove that anchor.

## Minimum independent reference

See docs/research/2026-10-07-f13-independent-temporal-control.md for the source
review and first preregistration. Physical geometry has only two opaque planes:
receiverz0 facing+Z,8m square; both-sided wallx.8 spanningy/z[-4,4]; camera(0,0,3)
fixed. Actual production RTAO radius2 feeds actual production custom DenoisePasses.
At fixture positive-update ordinal32 the wall moves tox8 once. No synthetic
noisy signal is substituted for the renderer. World geometry/camera/ROI precede
candidate pixels. For known pixel-centre receiverx left of the wall, visibility
is1-(acos(a)-a*sqrt(1-a*a))/pi, a=(wall_x-x)/R, or1 whena>=1. This is the exact
cosine-hemisphere projected disk area; not RT8, the GPU shader or denoiser output.

The oracle is ONLY for ROI+halo known receivers, not a fabricated full-frame
reference. Protocol is tools/testdata/f13_ao_temporal_protocol.json:64frames at
128x96, one view, signal9, ROI[48,38,16,20], visibility scale1, step32, native
camera/pixel centres. The three5x5 atrous iterations have14px maximum support;
geometry validates that halo before quality. Weak coverage/reference change
fails; the ROI is never refit. General wall geometry uses absolute perpendicular
distance; coincident points are rejected, selected domain stays strictly left.

## Concrete runtime hooks

`--bench6 --reflection-scene ao-temporal-wall --rt on --ao rtao --ao-radius2`
with fixed visibility/native geometry and exclusive AO selects the physical scene.
`--capture-linear-signal ao-filtered` is ID9 and requires custom denoise. It reads
ReflectionPasses::filteredAO(), the actual selected output, replicating R to RGB.
Raw AO7, raw GI1 and filtered GI8 remain their existing separate signals.

LinearCapture stores camera/geometry/ordinal/provenance at actual prepare and
publishes sidecar `frame-%06llu.json` beside the PFM AFTER real GPU completion.
ReflectionPasses::completedAOCheckpoint(slot) delegates the actual encoded
DenoisePasses checker, with completion and GPU frame/view/pixel guards. Existing
spare words4..7 are real reused count/max length/GPU view/frame. Epoch, revision,
reset and iteration count are immutable tags of the actual encoded check. The
baseline16..31 must demonstrate reuse; step32 must show reset,length1,reused0,
newidentity; held post frames must resume reuse. No denoise math, setting, default,
limit, ray count or biased correction is changed.

## Tester entry point — WRITTEN, NOT EXECUTED

`tools/f13_ao_temporal_check.py` defaults PLAN ONLY. `--run --manifest` consumes
an immutable plan using an existing binary, serial GPU lock and run_checked.
It records/checks source,binary,metallib,manifest hashes before/after execution,
requires actual checker success and64 matched scalar PFM/sidecar records, then
calculates the independent analytic reference only for the known domain.
Actual GPU frame IDs must equal fixture ordinal0..63; rebasing/simulation drift,
foreign view, missing reset, stale history, mismatched source or invalid scalar
channels fail. File hashes are preserved; images are not resized/exposed/aligned.

Existing caps stay.06p95/.15max/.02flicker/.25ghost/4frame recovery. In addition,
event-anchored old-state fraction is computed PER PIXEL then maxed, using the
same.25cap throughout the posthold: this detects a late one-pixel rebound that
RMS-only metrics can miss after the instantaneous reference delta becomes zero.
Recovery must remain passing through the last post frame. Black, lag1, lag5,
foreign/omitted reset, late whole-ROI and one-pixel partial rebounds are authored
independent tool negatives; tests registered in CMake but NOT EXECUTED.

This source control covers custom AO noise/history/geometry-epoch response on
fixed known receivers. Moving-camera reprojection, actual primary disocclusion,
GI transport and roughness-dependent specular reference remain PENDING. F12
Mitsuba diffuse/linearity protocols remain the route for separate GI evidence;
128-frame upstream GI convergence is not a replacement for the F13 history cap.
No source artifact or future positive of this one control closes F13.6/F14.
