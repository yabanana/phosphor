# F10 quality and temporal protocol v1 — NOT EXECUTED ON GPU

Source base: `8e533bfec3254dca2684de539470ca8ab4e0ea55`. The integrating tester
adds the existing LinearCapture routes `shadow=3`, `shadow-position=4`,
`shadow-normal=5`; F13 reserves 6/7. No Engine/CLI or GI/reference code is changed
by this protocol. No renderer/build/GPU run produced these thresholds.
The source and `POLICY` hash in `tools/f10_quality.py` freeze the experiment.
Preserve every failed result. Do not adjust thresholds against candidate images;
a changed policy is a separately identified experiment, never a replacement.

## Observable signals and ownership

- `shadow`: final filtered/contact-disabled visibility. **R is dimensionless
  visibility** in [0,1], not HDR radiance; G/B may carry variance/history metadata.
- `shadow-position`: receiver WORLD point, RGB Float32.
- `shadow-normal`: receiver WORLD geometric normal (half on GPU, Float32 PFM).
  Cleared/invalid normals have xyz=0. Every valid normal must have length within
  0.005 of one. **Validity = norm(xyz)>0**, so a valid point at the world origin
  remains distinguishable from invalid pixels. NaN/nonunit data fail.
- Capture the same frame independently for each signal. These static fixtures
  and fixed-step/native-upscale runs have no camera jitter or moving material.
  A missing/extra capture, changed extent or incomplete frame checker fails.
- PFM remains pre-exposure, pre-upscale and pre-tonemap. No color/display transform
  is part of the numeric comparison. Root owns all renderer executions.

The historical `f10_history_rejection` job used `disocclusion`, an emissive Cornell
fixture with **no directional sun**. Its previous result remains valid functional
execution evidence but cannot certify sun-history acceptance. Every new solar
job requires a non-sentinel `lighting.sun_index` as well as real checker results.

## Two deliberately small fixtures

`shadow-penumbra`: floor y=0, two zero-thickness opaque planes at x=-1.2/+1.2,
y=8/32 metres, z=0, each full size 0.8 x 2 metres in X/Z. Toward-sun is +Y and
angular radius is exactly 0.00465 radians. The camera is (0,3,6), aimed at (0,0,0),
60 degrees vertical, 16:9. Casters are outside the camera; they must remain in
shadow caster lists/TLAS. The |receiver z|<0.2 strip cannot see a plate's Z end:
0.2 + 32*tan(0.00465) <1. At 1280x720 both penumbrae have measurable width;
the previous shadow-bias fixture's 0.02/0.5/2m heights are mostly subpixel at640p.

`alpha-mip-shadow`: a camera-facing receiver at z=0, opaque backing at z=-0.2,
procedural256x256 checker alpha0/255, cutoff0.75. Every mip1 texel averages to
approximately0.5, below cutoff; mip0 has transmitting and surviving regions.
Camera (0,1.5,4) looks at (0,1.5,0). An off-camera caster at (-4,2,1) projects onto
the receiver under toward-sun (-4,0.5,1). Materials/geometries are shared ordinary
engine objects, not a test shader. CPU source tests verify texture and geometry;
actual sampler/footprint coverage is a GPU sensitivity gate below.

## Independent visibility oracle and fixed gates

For a horizontal receiver coordinate x and a plate center c at height h, let
r=h*tan(alpha). The projected-disk CDF is

`F(u)=0 / [0.5+(asin(u)+u*sqrt(1-u*u))/pi] / 1` outside/inside [-1,1].

The analytic visibility is `1-[F((c+0.4-x)/r)-F((c-0.4-x)/r)]`. The plates'
angular footprints do not overlap, so their blocked probabilities add. This
oracle does not call the engine's shadow math. Uniform projected disk versus
uniform-solid-angle cone introduces a bounded error below3.3e-5 at this radius
(from the projection Jacobian), far below the gates. Rays/receiver points use
the declared camera and pixel centers, not a candidate image fit.

CSM: frame127. RT:128 frames, captures every8; average only frame64 onward.
The ROI is the analytic floor strip |z|<0.2, |x|<5.5. Freeze:

| Predicate | Limit |
|---|---:|
| Visibility MAE over ROI | <=0.05 |
| Absolute mean visibility bias | <=0.03 |
| Mean loss in analytically fully-lit pixels | <=0.02 |
| Mean visibility in analytic umbra | <=0.02 |
| Median relative 10–90% edge width error, each plate | <=30% |
| Median 50% edge-location error, each plate | <=1.5 pixels |
| Minimum expected edge width / available rows | >=3 pixels / >=8 rows |

Only crossing localization uses fixed equal-weight isotonic PAVA to avoid
counting stochastic reversals as multiple edge locations. Raw images determine
all error/flicker metrics. No spatial mask is selected from candidate quality.
Missing crossings fail; the profiler never manufactures a zero error.

Bias control reverses W&B using the actual `--debug-lighting-corrupt bias` path.
It must exit0 with LIGHTING PASS but produce mean loss >=0.5 in the oracle's lit
region; the corresponding normal RT job must pass its positive oracle. A generic
nonzero image difference is insufficient. Contact is OFF in the physical oracle.

Temporal control uses identical RT settings with `--history-reset-every 1`.
Settled-tail frame-to-frame RMS in the analytic penumbra must be positive in that
reset control, and temporal RMS must be <=90% of it. This catches a path that
never actually accumulates despite reporting healthy state.

## DRS, forced overflow and exact receiver agreement

There is **no** `--debug-meshlet-overflow` option. The existing mechanism is
`--debug-meshlets 1 --debug-meshlets-corrupt count`: candidate capacity becomes0
on every frame, the indexed fallback renders, and the meshlet checker must FAIL
specifically for overflow. Raw exit and EXIT marker must be1. LIGHTING must still
PASS for every frame with lighting.failures=0. Any other FAIL, GPU/validation
error, timeout or signal rejects the run.

At native scales1/.75/.5 and effectiveApple9 scale.5, compare normal visibility
against forced indexed overflow, same frame7, native upscaler, history reset
on every frame. Capture all three signals. Fixed gates:

- Validity masks identical over **all pixels**, no edge exemptions.
- Valid normals unit within0.005; normalized normal dot >=0.999.
- World point error <=2e-4*max(1,|pointA|,|pointB|), every valid pixel.
- Shadow-mask MAE <=0.001; maximum <=1/16+1/1024 (one configured PCSS tap plus
  FP16 allowance). Geometry gates remain strict even when a mask is all lit.
- Global `--debug-neutral-mip-bias` control at scale.5 must change world position
  by >0.02m or validity on >=0.5% of all pixels. This independently establishes
  that the corpus exercises distinct mip0/mip1 coverage. It is not a substitute
  for the normal-versus-overflow comparison.

The normal-sign CPU negative flips one valid N. Comparison must reject it; taking
abs(dot) would incorrectly hide the forward-prepass orientation defect. The root
fix orients the indexed cross(dx,dy) toward camera exactly like visibility.

## Views, resize, DRS and cuts

Four runs use views1/2/3/4, static penumbra fixture and RT sun,160 frames:
`--post --upscaler native --temporal-script --resolution-script 11`
`--resize-every 40 --history-reset-every 23`.
Capture every17 frames: it is coprime to each view count, covers all four DRS
states and includes the camera cut at frame119. Camera/view offset follows the
existing declared script; actual PFM extents are read instead of guessed from
HiDPI window sizes. Require all views, at least four distinct captured extents,
complete frame checkers and lit/umbra interior error <=0.02 at every captured
frame. The static oracle and reset-control tests above separately test penumbra
quality/noise. This is not a universal moving-scene ghosting or F13 claim.

## Tester execution

Use an already provisioned NumPy environment (e.g. the project's quality venv).
The runner never installs packages. Default writes a fresh manifest only:

```sh
python tools/f10_quality.py --self-test
python tools/f10_quality.py --list
python tools/f10_quality.py --app build/lighting/phosphor --output build/f10-quality-plan
python tools/f10_quality.py --app build/lighting/phosphor --output build/f10-oracle --only 'oracle_*' --run
python tools/f10_quality.py --app build/lighting/phosphor --output build/f10-alpha --only 'alpha_*' --run
python tools/f10_quality.py --app build/lighting/phosphor --output build/f10-lifecycle --only 'lifecycle_*' --run
python tools/f10_quality.py --analyze build/f10-alpha
```

All jobs are sequential and API+shader validation is enabled. `--only` may produce
partial evidence: unavailable comparison partners remain pending, never passed.
Manifest stores source/binary/shader hashes via run_checked, policy hash, raw exit,
report, capture hashes and same-frame tags. No retries or reference replacement.
Numeric passes cover only selected protocol gates. Physical M3, performance,
contact-shadow quality, cache motion quality and F13 remain separate acceptance.
