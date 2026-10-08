# Native SDK authored-patch quality after known-wrap invalidation

The corrected 96-frame channels capture passes both frozen ROIs and every
retained temporal-quality cap. Recovery at the known source replacement on
frame 64 improves from 8 frames to 0. The original failed unannounced-change
capture, protocol and report remain intact. This passes one noiseless
authored-patch fixture; it does not close F13.6 or validate wide-HDR policy.

Root executed the native GPU capture. This follow-up only inspected existing
metadata and performed CPU analysis; it did not launch a renderer or build.

## Unchanged numerical experiment

The analytic reference remains a 7×7 square centered at
`(32 + frame % 64, 48)`, RGB (.75,.5,.25), on RGB (.125,.125,.125) background,
at 128×96 and pre-exposure 1. Full-image and geometry-derived swept ROI
`[28,44,72,9]`, support radius 1, scale 1, all 96 frames and recovery events are
unchanged. The numerical configuration is exactly equal to the original
protocol, SHA256 `3ae1be9dd1fbd2c2a0c39192b48a5b5ff17a1ff120a0086bdc707b5d46bcc322`.

The [corrected-run protocol](../results/F13-SDK-channels-wrap-quality-protocol.json)
was frozen before reading its SDK pixels at 2026-10-07T19:07:01.539299Z,
commit `f44ecf1`, SHA256
`248c05bb2880f7b84537d87546670bf2845fdce08ca7383f5a4dc703a2a87bb1`.
Its input path and declared source invalidation change; numerical gates do not.

| Metric | Original swept ROI | Corrected swept ROI | Cap |
|---|---:|---:|---:|
| RMSE p95 | .069077 | .055912 | .06 |
| Worst-frame RMSE | .113953 | .056517 | .15 |
| Mean residual flicker | .007908 | .007592 | .02 |
| Maximum support ghost fraction | .333625 | .035063 | .25 |
| Source replacement 64 recovery | 8 frames | 0 frames | 4 frames |

Corrected full-image values are RMSE p95 .012904, maximum .013036, mean residual
flicker .000439 and support ghost .035063. Startup and explicit frame 48 reset
also recover at 0 frames. The exact analytic positive still passes; CPU
all-background, one-frame-lag and opposite-direction controls still fail.

## Reset reached the native SDK at the intended boundary

All 96 JSON records show exactly one native encode, successful guide/invariant
checks and the expected reset delivery. Actual reset frames are exactly
**0,48,64**. Input signal epoch stays 1; effective source epochs are 2 on
frames 0–47, 3 on 48–63, and 4 on 64–95. The epoch increment persists after the
wrap, while continuous frames receive no additional reset. No camera cut is
reported. The analysis tool now enforces these fields for protocols declaring
a known source replacement.

The 64 SDK PFM files preceding the replacement, frames 0–63, are byte-for-byte
identical between original and corrected runs. The correction first affects
the intended frame 64. Its corrected output is also byte-for-byte identical
to the corrected frame 0, which has the same authored patch position and a
fresh native history.

At frame 64, the positive-red centroid is x31.952 versus authored x32; the
original capture retained x81.874. Mean RGB patch-contrast ratios are now
(.7175,.7220,.7299), versus (.0467,.0558,.0882) previously. These diagnostics
retain the same fixed support and are not fitted quality gates. Edge
filtering remains measurable. SDK and restored physical images are exactly
equal for all 96 frames at unit pre-exposure.

## Evidence and scope

Actual input directory in the root integration worktree:
`build/f13-channels-wrap-v1/channels-motion-normal-roughness/actual-sdk`.
Recorded source digest is
`05d7007574e4781b8db195d6ec746f77624c3ca4aefd946fa92bd0276b7d38a8`,
binary SHA256
`fa9ba06e57f8e1db32a446a10c2ab9a72579a23ebed3b7210a1b6f14048a6875`,
and capture-manifest SHA256
`388aa016d87d8093f95a7d516b411e2fca9c510f4951d7194c6fd3eb41e9dbe2`.

The [compact result](../results/F13-SDK-channels-wrap-quality-2026-10-07.json)
records both-ROI metrics, actual reset/epoch ranges, controls, diagnostics and
the preserved negative comparison. Full per-frame metrics and hashes for
all 288 input artifacts are in the review worktree
`/Users/danielsan/.codex/worktrees/f9-shaders/phosphor`, file
`build/f13-sdk-channels-wrap-quality-v1/analysis-with-reset-checks/result.json`,
SHA256 `1f6dec227956169c57260a822bee1a81752ea3c95347f4a46364d9ab075c227b`.
The sibling `reset-metadata.json` and `before-after.json` retain the metadata
audit and complete image comparison. Raw captures and earlier failed results
were not changed.

The result supports invalidating known authored source replacements and
retaining history during the preceding continuous motion. It does not imply
universal denoiser quality, correct reprojection through arbitrary camera
changes, noisy lighting convergence, wide-HDR unit correctness or production
policy promotion. Those retain their own independent acceptance work.
