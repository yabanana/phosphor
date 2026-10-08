# F13.6 independent temporal control — plan before implementation

SOURCE ONLY / NON VERIFIED. No configure/build, compiler, CPU/GPU test, renderer
or reference/oracle execution. The aggregator checkout remains read-only. SDK
native radiometry/lifecycle and GI defaults are excluded. No phase acceptance.

## Findings

The current moving-light/disocclusion/roughness scenes are useful stress clips,
but generic screen-fixed metrics have no receiver correspondence, event identity
or proof that history was reused. Moving world content continuously resets epochs.
Scale1 plus a tiny change threshold can let all-black output pass a dark signal.
Signed global ghost projection can cancel; recovery can pass vacuously without
an event or with only two lucky frames. External independent=true is not an
image/source identity proof. RT8 remains only a noisy control.

F12 supplies a valid route for isolated diffuse GI on matte, opaque, untextured
role-equivalent scenes with exact Mitsuba snapshots and convergence. Its current
emissive-step physical linearity proof and F12 <=128-frame convergence gate do
not demonstrate the separate F13 <=4-frame history rejection gate. A diffuse
oracle cannot certify the metallic moving-light/roughness full HDR. Preserve
F12 physical20%bias/30%NRMSE gates; do not rename upstream GI convergence as
custom-filter latency or invent a same-implementation reference.

## Minimum selected control

Use the REAL production RTAO producer and REAL custom DenoisePasses on a physical
known receiver, without a synthetic noisy image or offline surrogate filter.
Two opaque planes: receiver z0/normal+Z, size8m; wall x.8, spanning y/z[-4,4],
DOUBLE_SIDED and visible+caster. Camera stays (0,0,3), looking at origin. AO
radius2m; no GI, reflections, lights, SDK, post temporal, camera motion or jitter.
The wall moves once to x8m at positive fixture update ordinal32. Capture64 real
frames:0..31 wall-near,32..63 wall-outside-radius. Fixed128x96 viewport, seed1001.
No existing production default or sample count is changed.

For each known receiver point x, a=(wall_x-x)/R. Cosine hemisphere sampling is
uniform on its projected unit disk. The visible fraction is exactly

    V = 1 - (acos(a) - a*sqrt(1-a*a))/pi, for 0<a<1
    V = 1 for a>=1; selected receivers always have a>0

The selected ROI remains entirely left of the wall, so the physical near case
uses0<a<1 and the post casea>1. The finite wall covers every possible hit within
R. Its normal is orthogonal to the receiver normal. The ray-origin normal offset
does not change x or this intersection/radius condition. Primary receiver points
come independently from the recorded camera/pixel-centre ray and z0 plane, never
from candidate AO. World dimensions/roles/camera are checked against the fixed
recipe; all references are dimensional visibility[0,1], not radiance.

Freeze ROI pixelxywh[48,38,16,20]. It is on the same receiver and at least14px
from other primary geometry at the declared128x96 camera. Three5x5 atrous passes
have a conservative14px support (2*(1+2+4)); do not reuse the generic F8 1px
support as an invented kernel footprint. No ROI/scale is chosen from output.
Reference mean must be>=.5 and mean step change>=.15. Coverage and identity must
pass before quality. Signal is the actual selected custom AO, new scalar capture9;
raw7 remains a diagnostic comparison from the same physical input/seed.

## Frozen gates and witnesses

Keep the existing F8 numeric caps: spatial RMSEp95<=.06, max<=.15, residual
flicker<=.02, support ghost<=.25, recovery<=4frames with RMSE<=.06. Visibility
scale is exactly1. Gate old-frame attraction per pixel before aggregation, so
opposite residual signs cannot cancel; use current and prior independent physical
reference. Recovery must remain passing through the full post hold, not two frames.
No pixel fitting, exposure fitting, adaptive seed/count, motion alignment or retry.

Require immutable sidecar per actual GPU frame: frame/view/extent/signal9,
fixture ordinal, wall_x/R/camera/receiver recipe, source/binary/metallib/manifest
hashes, and completed REAL GPU denoise check. Existing spare check words record
actual reused histories/max length/view/frame. Baseline16..31 must show actual
history reuse, step32 must show history reset/length1, and after+4 the new history
must be valid and reused. Exact contiguous frame IDs and fixture ordinal==GPU
frame prevent treating missed drawable acquisitions as an unchanged event.

Actual black output, retained near-wall output, one-frame delayed physical
reference, five-frame delayed reference and post-recovery rebound are checker
negatives. Wrong GPU view/frame or retained history must also fail metadata
validation. Their failure is required for a meaningful positive, without changing
product/math thresholds. Raw and filtered outcomes remain separate and retained.

## Implementation scope

Only add the physical fixture, scalar selected-AO capture, minimal immutable
sidecars/completed-history witness, and a source-only frozen planner/independent
analytic verifier with authored unit controls. Existing generic temporal tools
remain unchanged; they are diagnostics for their existing clips. No new shader
reference, SDK operation, GI preset, optimization or phase tick.

This verifies a minimum real custom AO/noise/epoch-response path. It does not
certify GI transport, specular GGX, moving-camera reprojection or newly disoccluded
receivers. Those need their distinct F12-equivalent physical references and true
receiver correspondences. Passing this one control cannot close all F13.6/F14.
