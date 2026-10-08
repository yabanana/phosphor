# F10 solar quality: measured correction, 2026-10-07

The frozen solar oracle gates pass after two implementation corrections. This
result covers the analytic opaque-plate corpus only; it does not close F10,
the alpha/DRS/multiview protocol, contact shadows, unavailable Apple9 hardware,
or performance acceptance.

## Reproducible evidence

Root tester ran `tools/f10_quality.py --only 'oracle_*'` sequentially on the
Apple M5 Max, macOS 27.2, under Metal API and shader validation. The exact
commands, raw exits, executable/shader hashes, PFM hashes and checker counts
are in `build/f10-quality-oracle-v2/manifest.json`; metrics are in its
`quality.json`. All four jobs passed their functional checks with raw exit 0,
EXIT 0 and no allowed validation errors. These are real GPU results; the
diagnosis replica described below was CPU-only.

- Source: `11ee3266eb32a2a3db755fb9a707495024e73fe9`, clean tracked source.
- Binary SHA256: `094a6201e85b134d2eb57896642769c774c1c243ffc88b2fb267c5fde6753ae2`.
- Metallib SHA256: `1fb335f8d9a91b2aa589807b8160879ee7b5b315d9e8d5846a3304864981b5fd`.
- Policy SHA256: `e46c895cf7af6921694c06eff1f8c61e8c6a09c751124a18c2d1605f15e13e29`.

The original failed run remains `build/f10-quality-oracle-v1`. The v2 directory
is a new run, not a changed policy: all tolerances, geometry, camera, seeds,
sampling cadence and analytic ROIs are identical. The comparison uses 17,410
receiver pixels, 1,946 umbra pixels and 17 profile rows per caster height.

| Metric | Before CSM | After CSM | Before RT | After RT | Frozen limit |
|---|---:|---:|---:|---:|---:|
| Visibility MAE | .010994 | .001959 | .004876 | .000988 | .05 |
| Mean umbra visibility | .046156 | .000200 | .000020 | .000128 | .02 |
| Width relative error, 8 m | 1.421106 | .035211 | .234698 | .030148 | .30 |
| Width relative error, 32 m | .400607 | .023207 | .378680 | .017628 | .30 |
| Edge error pixels, 8 m | .179123 | .256934 | .083806 | .052453 | 1.5 |
| Edge error pixels, 32 m | 1.000843 | .592515 | .340964 | .371275 | 1.5 |

The v2 RT temporal flicker RMS is .025435 versus .355086 with reset every frame,
below the frozen 0.9 ratio limit. The wrong-origin-bias negative remains
detected: fully lit receiver loss is 1.0, exceeding its fixed 0.5 minimum. Mean
lit loss and signed bias pass as well; neither undefined rows nor unresolved
widths were discarded. The manifest deliberately retains `phase_accepted:false`.

## Causal corrections

**Bernoulli history.** `shadow_temporal` used the current raw 3x3 minimum and
maximum to clip the accumulated visibility. For a truly static penumbra with
visibility p=.1, all nine raw samples are black with probability .9^9≈.387; that
event does not make the expected visibility zero. The clip repeatedly erased
valid history. In the failed GPU evidence, 32 m penumbra width was .621× analytic,
while the same reset-every-frame control was .956×. The fix removes the raw
extrema clip, retaining geometric/identity/revision rejection, the 16-sample
history cap, raw moments and the separate spatial filter. An exact CPU test
enumerates all 512 independent raw neighborhoods and checks preservation of
the expected visibility; the old clipped recurrence is a negative control.

**PCSS blocker volume.** The CSM builder always added the 500 m fallback reach
even when given complete scene caster bounds. For this first cascade it put
the light near plane at 503.872 m and enlarged the blocker search to 2.343 m,
although actual penumbra radii are 37.2 mm and 148.8 mm. Sixteen blocker samples
could miss the plate or mix the unrelated 8 m/32 m plates, producing the overly
wide profile. The builder now uses fallback reach only when no caster bounds
are supplied. Receiver frustum coverage, conservative full caster bounds and
texel/bias guards remain. The minimum half-texel filter radius is 7.76 mm here,
smaller than both physical penumbrae, and was not modified.

The caller audit is part of the fix: `ShadowPasses::fullCasterBounds` evaluates
the uploaded motion table and parent-before-child hierarchy, then bounds every
valid shadow caster independently of camera visibility. A nonempty bounds
span is explicitly documented as complete. An empty scene still takes the
old conservative fallback. CPU regression checks all caster-bound corners in
all four cascade depth ranges, reach independence when bounds are present,
and unchanged XY stabilization. A separate approximate CPU raster/filter
replica supported this diagnosis; it is not substituted for the v2 GPU proof.

Subsequent alpha/overflow, DRS/resize/multiview/history, offscreen-caster and
cache-stress runs must keep their own evidence and acceptance status. No
performance inference is made from validation-enabled oracle run durations.
