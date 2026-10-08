# F13 wide HDR: one descriptor-auto-exposure experiment

Source-only candidate on `dbcd36da513be5d14a38cd75830909b04f423588`. No renderer, GPU workload, C++ build or SDK compilation was executed by this agent. The shared integration checkout was not edited.

## Frozen corrective hypothesis

Storage pre-exposure q=1/64 maps the physical target [368640,128,64] to the supplied HALF input [5760,2,1]. Apple's `preExposure` contract tells the SDK to undo that same input scaling. This is distinct from scene exposure. A unit manual exposure texture therefore does not normalize this extreme scene. This is a hypothesis about the observed wide-HDR loss, not evidence of a documented hardware or SDK precision limit.

One variable changes: `TemporalDenoisedScalerDescriptor.isAutoExposureEnabled=true`. Apple documents that this lets the SDK calculate exposure per image and causes it to ignore the supplied manual exposure texture. We leave that texture at 1, record its GPU readback as a **provided manual value**, and never claim that it reveals the SDK's internal exposure. The adapter records the actual public descriptor getter and fails closed if it differs from the requested mode.

Official primary documentation, checked 2026-10-07:

- [Descriptor automatic exposure](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerdescriptor/isautoexposureenabled): per-image automatic calculation; manual texture ignored; default false.
- [Pre-exposure](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure): declares the existing color premultiplication for the SDK to undo.
- [Exposure texture](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/exposuretexture): 1x1 R16Float multiplier, ignored by automatic exposure.

The local Xcode `MetalFX.framework/Headers/MTLFXTemporalDenoisedScaler.h` agrees (lines 103–112 and 338–355). No private API, ownership workaround, process isolation, mask bypass or output compensation is introduced.

## Exact invariants and result interpretation

- One case: `wide-hdr-auto-exposure`, 96 actual native frames 0..95, 128x96, same initial prewarm and two 48-frame phases as the existing run.
- Physical target [368640,128,64], q=1/64, packed [5760,2,1], supplied exposure texture 1, strength mask 0; guides and fixture shaders unchanged.
- Existing SDK-output hypothesis, physical restore `SDK / q`, and every threshold unchanged. `GATES`, `constant_oracle` and `impulse_oracle` are AST-identical to baseline.
- PASS still requires native encoding, actual descriptor auto mode, all packed-channel checks, finite output, exact frame coverage and the existing physical target equation. A descriptor boolean or a finite image alone cannot pass.
- Source-experiment flag `--denoised-fixture-auto-exposure` is rejected outside the explicit wide-HDR/pre-exposed fixture. Production defaults remain manual and `OutputScale::Unverified`; no production policy promotion.
- Failure preserves all raw results and the wide-HDR gate. It does not justify choosing a new exposure/output compensation or relaxing thresholds.

## Patch and source verification

`wide-hdr-auto-exposure.patch` (10 source/test files). SHA256: `ad0506771bc1eaa07427559a1d0e042fab02276f5138cddba7cfb8feefd3f45d`.

`git apply --check` passed on the unchanged integration checkout. Eleven Python synthetic tests passed, including rejection of changed target/q/frame count, hidden command-mode mismatch, an unconfigured or mismatched actual descriptor, ignored-texture misreporting, and the observed severe radiometric loss. `python-tests.log` contains the full CPU-only result. The C++ CLI test was written but not compiled or run here.

`plan-only-smoke/sdk-frozen-manifest.json` is a planner-only self-test using a deliberately nonexistent binary path; do not use it for GPU execution. It contains exactly one case and was created without launching any binary.

## Root execution, after apply/build

From `/Users/danielsan/.codex/worktrees/f9-core/phosphor`, use a fresh output directory. First freeze the actual compiled binary and unchanged gates:

```sh
python3 tools/metalfx_denoise_fixture_check.py --binary build/lighting/phosphor --out build/f13-native-wide-auto-v1 --source-root . --gateway native --frames 96 --resolution 128x96 --denoised-fixture-auto-exposure
```

Then consume that manifest without repeating or changing experiment flags:

```sh
python3 tools/metalfx_denoise_fixture_check.py --binary build/lighting/phosphor --out build/f13-native-wide-auto-v1 --source-root . --run --manifest build/f13-native-wide-auto-v1/sdk-frozen-manifest.json
```

The runner serializes through the existing GPU lock and refuses evidence overwrite. It records actual descriptor mode in `prewarm.json`, every frame JSON and `summary.json`, plus the manual-texture semantics. Compare unchanged linear SDK/physical PFM equations against `build/f13-native-units-v1/wide-hdr-scaled/actual-sdk`; do not renormalize either image.
