# F11: raw direct-energy validation after history-chain expiry correction

The corrected estimator passes all nine static reuse cases and all 459
preregistered RGB/region comparisons. The preserved earlier run fails 22
comparisons across four cases. The single-point PDF control independently
detects the expected factor-of-two energy change. F11 remains open for
ManyLights, post-denoiser quality, lifecycle and performance work through F13.

The compact machine-readable handoff is
[F11-energy-chain-age-2026-10-07.json](../results/F11-energy-chain-age-2026-10-07.json).
It records raw evidence paths, hashes, source identities, per-case ratios,
validation scope and the exact CPU counterexample. Report preparation only
read existing artifacts: no build, renderer or GPU process was launched, and
no raw evidence was modified.

## Experiment and acceptance rule

Root ran `tools/f11_energy_check.py` serially on Apple M5 Max/macOS 27.2.
Resolution is 128×72, with 32 warmup and 256 measured frames per run
(frames 32–287). Candidates use seeds 1–32 and references use disjoint seeds
1001–1032. Each complete run mean is one statistical observation; correlated
frames and pixels do not increase the sample count.

Three fixtures—eight punctual lights, area lights, and Cornell emissive
triangles—each exercise spatial-only, temporal-only and combined reuse.
References are the brute estimator on the same fixture/source snapshot.
The nine candidate sets share three reference sets and have three additional
stability witnesses: 387 runs per ensemble. Each run reports 288 lighting
checks, zero lighting failures and zero GPU failures. Early/late source
snapshots are part of the runner's static-scene checks.

The signal is raw direct illumination in RGBA32Float, captured as linear
Float32 RGB PFM. Local visibility is disabled with `--shadows off`; GI, post
processing, jitter, DRS and camera/light motion do not enter this experiment.
The status files record Metal API/shader-validation environment variables as
unset. These energy runs therefore provide numerical and runtime-checker
evidence; API/shader validation has its own test lane.

For each of 17 fixed regions and three channels, the gate evaluates the
difference of candidate/reference run means using a Bonferroni-adjusted Welch
Student-t interval, with familywise alpha .01 across 459 comparisons:

- **FAIL:** absolute difference exceeds the interval half-width plus the
  frozen FP32 allowance, `gamma4096 × abs(reference)`, where
  `gamma4096 = 1/4095`.
- **INCONCLUSIVE:** the difference is within that bound, but interval
  half-width exceeds 2% of the reference mean.
- **PASS:** both conditions are satisfied.

The 2% value is a precision requirement on the interval, not an extra 2%
energy-error allowance. Student-t intervals remain a finite-sample diagnostic.
The before/after manifests have identical cases, seeds, frame range, regions,
signal and statistical/numerical limits. Their shared protocol digest is
`09d2f0e572e769c5dab48498378f6368db6d1ebcfef0f61608ec80104defa2b7`;
the JSON lists the exact fields and canonical serialization used for it.

## Results

Ratios below are whole-image sums of mean RGB divided by their corresponding
reference. Acceptance uses every region/channel comparison, rather than this
summary ratio alone.

| Case | Earlier energy ratio | Earlier failed comparisons | Corrected energy ratio | Corrected comparisons |
|---|---:|---:|---:|---:|
| Point8 spatial | 1.000368422 | 0 | 1.000388341 | 51/51 PASS |
| Point8 temporal | 1.004190525 | 9 | 1.001169828 | 51/51 PASS |
| Point8 full | 1.004062400 | 7 | 1.001098600 | 51/51 PASS |
| Areas spatial | .999994910 | 0 | .999996127 | 51/51 PASS |
| Areas temporal | 1.001221479 | 3 | 1.000013604 | 51/51 PASS |
| Areas full | 1.001331726 | 3 | 1.000199990 | 51/51 PASS |
| Emissive spatial | 1.000167252 | 0 | 1.000047760 | 51/51 PASS |
| Emissive temporal | 1.000074963 | 0 | .999876413 | 51/51 PASS |
| Emissive full | 1.000008796 | 0 | 1.000069326 | 51/51 PASS |

The earlier 22 rejections are FAIL, with zero INCONCLUSIVE results. After
correction there are 459 PASS, zero FAIL and zero INCONCLUSIVE. The narrowest
precision margin is Point8 full, region 6, green: interval half-width
`6.751015119e-6`, below its `6.773301384e-6` precision limit. This is a passing
result at the frozen sample count, with little precision margin in that ROI.

The preserved directories are `build/f11-energy-reuse-ensemble` and
`build/f11-energy-reuse-chainage-v2`. They use different integration commits
and each has its own same-source brute reference; other renderer integration
changes are present between them. The isolated causal evidence comes from
the estimator counterexample and correction below, while these GPU ensembles
establish that the corrected integrated renderer satisfies the fixed gate.

## Why the correction is necessary

Commit `970d7ce` makes expiry track the complete incorporated proposal history
independently of the endpoint that happens to win weighted selection. The
previous code copied age only when an endpoint won, allowing a fresh selected
endpoint to rejuvenate a reservoir that still contained old proposal mass.
Discarding that mass based on endpoint age conditioned the estimator on
selection. The correction preserves proposal PDFs, normalization, M/age caps
and receiver compatibility; counted zero/blocked mass advances the chain too.

The independent rational enumeration uses two positive proposal weights
{1,3}, equal probability, maxM 2 and age limit 1. At frame index 4 the old
expectation is `182666318/91265265 = 2.0014878387741493`, against integral 2.
Corrected chain age gives exactly 2 at every enumerated frame, indices 0–6.
See [the estimator review](2026-10-07-restir-history-expiry.md) and
[exact proof artifact](../results/F11-history-expiry-exact.json). This proof
does not depend on shader sampling, floating-point allowances or visibility.

## Single-point sensitivity control

`build/f11-energy-point1-chainage` contains five successful runtime/checker
runs and three exact comparisons, each checking every measured pixel/frame:

| Control | Expected ratio | Measured ratio |
|---|---:|---:|
| Clustered versus brute | 1 | 1.0000000000000000 |
| Fresh reservoir versus brute | 1 | .9999999938557079 |
| Isolated proposal PDF multiplied by 2 | .5 | .49999999692785396 |

The altered PDF shader lives in an isolated copy with its own source/AIR/
metallib hashes. Its ordinary lighting checker still passes, while the
unmodified unity-energy gate rejects the image
(`unity_energy_gate_rejected:true`). The control passes because the expected
half-energy result is observed and the erroneous unity result is detected.
There are zero exact-frame violations against the expected factor and zero
nonzero samples on reference-black pixels. Maximum relative error on nonzero
pixels for fresh/PDF×2 is `1.4907746393e-7`.

## Evidence identity and handoff

| Dataset | Recorded workspace commit | Manifest SHA256 |
|---|---|---|
| Earlier ensemble | `f4629c8d0ea0cd7ae58b3195ae737e3bc22e8e2e` | `539e1365811e3f02340b82cc1cfd0f0f778cc76d28b6ef4cf8b0728d9a649b3b` |
| Corrected ensemble | `68b635c836b8b13bf094a2c23eb69d0f0775ba08` | `13486f098ec09ebbc83abd60eddfe819e4f05051e4a1c128f9eb4e7fb18ceab0` |
| Point1 control | `778560d0f011598fd8c1df04f527f4f3bb0b3817` | `290ea9c4f4b3239a5deeb162d6b87ce668467c01d4681ac094b7bb97d8decc72` |

All recorded tracked patches are empty. The corrected ensemble's recorded
binary hash is `fc67e1574e40d63b2c5e3af82c2c4b356d824902aaacae4bcbe9b0894e0e0ddb`
and metallib hash is
`375bd183d6fd9b68fc18fa952e3042ccb739e33f59e1b829c99859e3dfee6ad5`.
The JSON preserves the other artifact and summary hashes, including the
separately linked PDF×2 metallib. They identify the recorded executions;
subsequent integration builds can replace the current build-directory binary.

This handoff verified all 779 retained mean-PFM hashes, all run/aggregate
statuses and report counters, the summary-to-manifest hashes, and every
comparison's status arithmetic. The original retention policy keeps all mean
PFM/Float64 NPY images and per-frame hashes/ROI statistics, plus designated
endpoint captures; most intermediate PFM frames were consumed and deleted by
the original runner. They have not been reconstructed or replaced.

Remaining F11 acceptance includes the ManyLights workload and costs, direct
illumination with actual local visibility, post-F13 denoiser image/temporal
quality, moving receivers/emitters and disocclusion, lifecycle/resize/view
transitions, and performance on the declared hardware tier. No phase or
roadmap completion checkbox is changed by this report.
