# Native MetalFX: independent authored-patch temporal check

The 96-frame native SDK capture fails the frozen temporal-quality gate on its
unannounced signal replacement at frame 64. Ordinary motion and the explicit
reset at frame 48 remain within the retained limits. This is evidence about
one controlled input clip, not F13.6 acceptance or a general SDK correctness
claim. No renderer, SDK or GPU process was run during this analysis.

## Reference and protocol frozen before pixels

The source audit established a **7×7 square**, with center
`(32 + frame % 64, 48)`, on a 128×96 image. Patch RGB is (.75,.5,.25), background
RGB is (.125,.125,.125). Normal/roughness change at x64 independently of the
moving patch. The image is authored deterministically, so its exact noiseless
linear RGB is an independent analytic reference; no captured output is used
to derive it.

Motion is (-1,0) input pixels only inside the patch when phase > 0; elsewhere it
is zero. The host resets history at frames 0 and 48. At frame 64 the patch jumps
from x95 to x32, motion becomes zero, and no reset or signal-epoch change is
supplied. The event is therefore evaluated as an **unannounced source-signal
replacement**, not evidence for correct reprojection of a moving object or
a camera-cut contract.

The [protocol](../results/F13-SDK-channels-quality-protocol.json) was frozen at
2026-10-07T18:35:46.214821Z and committed as `06a427a` before reading any SDK
PFM pixels. Its SHA256 is
`037e952f7fd4b7ef6327b42f97c3b9e104211898537acf753d458d0e65e56353`.
Both ROIs were derived from known geometry before analysis:

- Full image: x0/y0, 128×96.
- Swept patch: x28/y44, 72×9, the union of all possible authored centers with
  the 3-pixel half-size and 1-pixel reconstruction support.

No frames, borders, pixels or outliers are dropped. There is no intensity
crop, resized reference, fitted scale, time shift or lag compensation.
Pre-exposure remains 1 and all values stay linear.

The unchanged caps are RMSE p95≤.06, worst-frame RMSE≤.15, mean residual
flicker≤.02, support-aware ghost fraction≤.25, and recovery within 4 frames
below RMSE .06 for two consecutive frames. Support radius is 1. Recovery
events are startup 0, explicit reset 48 and unannounced source replacement 64.

## Measured results

| Metric | Full image | Fixed swept ROI | Retained limit |
|---|---:|---:|---:|
| Linear RMSE p95 | .015894 | **.069077** | .06 |
| Worst-frame RMSE | .026169 | .113953 | .15 |
| Mean residual flicker | .000450 | .007908 | .02 |
| Maximum support ghost fraction | **.333625** | **.333625** | .25 |
| Startup/reset 48 recovery | 0/0 frames | 0/0 frames | 4 |
| Source replacement 64 recovery | 0 frames | **8 frames** | 4 |

The global recovery statistic is diluted by the small signal footprint; the
predefined swept ROI exposes the eight-frame recovery. Its RMSE exceeds .06
only on frames 64–71, returning below the bound at 72 and remaining below it
at 73. Support ghost exceeds .25 only at frame 64. All other measured frames
are below those spatial/support limits. SDK and restored physical PFMs have
maximum absolute difference 0 at unit pre-exposure.

The contrast/centroid diagnostics were declared before pixel access and are
not additional fitted gates. At frame 63, positive-red centroid x94.677 follows
authored x95. At frame 64 it remains x81.874 while the reference is x32; mean
contrast inside the new patch is only (.0467,.0558,.0882) times the expected
RGB contrast. At frame 72, red centroid is x40.017 versus expected 40, while
patch red contrast has recovered to approximately .561. The remaining edge
filtering is part of the measured SDK output, not removed by analysis.

## Independent sensitivity checks

Before opening SDK PFMs, CPU controls established that the exact oracle
passes and three defective clips fail: all-background, one-frame delayed
patch and opposite-direction motion. The global ROI alone accepts the
opposite-direction clip; the fixed geometry ROI rejects it. This demonstrates
why the signal-sized ROI is needed even when the full-image numbers look
small. It does not constitute an actual SDK run with corrupted motion.

`tools/f13_sdk_channels_quality.py` implements the existing temporal metric
definitions with vectorized arithmetic. Its scalar cross-check against
`temporal_light_metrics.analyse_clips` agrees within 1e-12 on the CPU fixture.
The metric source hash matches the pre-frozen protocol. No F8 cap changed.

## Evidence identity and limits

Input directory:
`build/f13-native-units-v1/channels-motion-normal-roughness/actual-sdk` in the
root integration worktree. All 96 metadata records confirm native SDK encoding,
128×96 input/output, unit pre-exposure and successful authored/packed guide
checks. Analysis hashes every input: 96 JSON records and 192 SDK/physical PFMs.
Recorded provenance is:

- Source digest: `276065e588677bc2786d8f26315d59fb1c59a0375ed8cba132402efa70b07903`.
- Binary SHA256: `b340e0111329539153791f661ad80e876063d2f3acffb8a7b53ca65e6cd51618`.
- Capture manifest SHA256: `a676efd901a93627c75794a7cef27c9b5e2078ae489804c4f6386bbccb994e47`.

The [compact result](../results/F13-SDK-channels-quality-2026-10-07.json)
contains per-ROI gates, controls, selected diagnostics and artifact hashes.
Full per-frame metrics and all 288 input hashes are retained in the review
worktree `/Users/danielsan/.codex/worktrees/f9-shaders/phosphor`, under
`build/f13-sdk-channels-quality-v1/analysis/result.json`, SHA256
`e8912e243d26bc35f1468f5822906bbcd6b6cf457000e6d11f209e7a0f346301`.
The frozen protocol and CPU-control evidence remain in the same parent
directory. Raw native captures were not modified.

The failing replacement has no corresponding history invalidation in this
fixture. Production light/material revision handling may reset that history,
so these results alone cannot attribute an integration or SDK bug. A fresh
controlled native run must distinguish continuous motion from a correctly
signalled replacement/reset/reactive event while retaining this failed
evidence. General noisy-lighting denoising, per-view motion, disocclusion,
scene changes and the full F13.6 corpus remain separate acceptance work.
